"""Analisis de senal compartido: envolventes, onsets y el indice de fusion.

Las tres primeras funciones se suben desde scripts/32_listen_rolling_marble
(que ahora las importa de aqui). band_envelopes y fusion_index dan soporte
al modulo de blend profundo (vocoder fisico y metrica del paper).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


def smooth_env(w: np.ndarray, sr: int, win_ms: float = 5.0,
               fs_out: float | None = None) -> np.ndarray:
    """Envolvente |hilbert| suavizada con media movil de win_ms.

    Si fs_out se indica, decima al ritmo aproximado fs_out (hop = sr//fs_out).
    """
    env = np.abs(signal.hilbert(w.astype(np.float64)))
    win = max(8, int(win_ms / 1000.0 * sr))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    if fs_out is not None:
        hop = max(1, int(sr // fs_out))
        env = env[::hop]
    return env


def detect_onsets(w: np.ndarray, sr: int, expected_rate: float) -> np.ndarray:
    """Onsets por pico de envolvente HP>1kHz (indices de sample).

    La distancia minima entre picos se liga a la tasa esperada (0.5/rate)
    para no contar doble los sub-picos de un mismo evento.
    """
    sos = signal.butter(4, 1000, btype="high", fs=sr, output="sos")
    hp = signal.sosfilt(sos, w.astype(np.float64))
    env = np.abs(hp)
    win = max(8, int(0.005 * sr))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    thr = 0.35 * env.max()
    min_dist = max(int(0.008 * sr), int(0.5 / max(expected_rate, 1.0) * sr))
    peaks, _ = signal.find_peaks(env, height=thr, distance=min_dist)
    return peaks


def mod_index(w: np.ndarray, sr: int) -> float:
    """Indice de modulacion de envolvente (std/mean). Ruido continuo da
    valores bajos; texturas de eventos, altos."""
    env = smooth_env(w, sr, 5.0)
    return float(env.std() / (env.mean() + 1e-12))


def band_envelopes(w: np.ndarray, sr: int, n_bands: int = 12,
                   lo: float = 80.0, hi: float = 8000.0,
                   win_ms: float = 8.0, order: int = 4,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Envolventes por banda a FULL RATE (para el vocoder fisico).

    Bordes log-espaciados en [lo, hi]; bandpass butter por banda; envolvente
    rectificada + media movil win_ms. Devuelve (envs (n_bands, n), centros_hz).
    """
    hi = min(hi, sr / 2 - 200)
    edges = np.geomspace(lo, hi, n_bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    n = len(w)
    envs = np.zeros((n_bands, n), dtype=np.float32)
    win = max(8, int(win_ms / 1000.0 * sr))
    kernel = np.ones(win, dtype=np.float64) / win
    x = w.astype(np.float64)
    for k in range(n_bands):
        sos = signal.butter(order, [edges[k], edges[k + 1]], btype="band",
                            fs=sr, output="sos")
        band = signal.sosfiltfilt(sos, x)
        envs[k] = np.convolve(np.abs(band), kernel, mode="same").astype(np.float32)
    return envs, centers


@dataclass
class FusionReport:
    """Resultado del indice de fusion entre dos bandas atribuibles."""
    rho: float          # correlacion maxima (+-20 ms) de envolventes
    onset_f1: float     # F1 de coincidencia de onsets, corregido por azar
    fi: float           # 0.5*max(rho,0) + 0.5*max(f1,0)
    p_value: float      # test de permutacion circular


def _band(w: np.ndarray, sr: int, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    hi = min(hi, sr / 2 - 200)
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, w.astype(np.float64))


def _onset_f1(on_a: np.ndarray, on_b: np.ndarray, tol: int) -> float:
    if len(on_a) == 0 or len(on_b) == 0:
        return 0.0
    matches = 0
    j = 0
    for a in on_a:
        while j < len(on_b) and on_b[j] < a - tol:
            j += 1
        if j < len(on_b) and abs(int(on_b[j]) - int(a)) <= tol:
            matches += 1
    return 2.0 * matches / (len(on_a) + len(on_b))


def _max_corr(ea: np.ndarray, eb: np.ndarray, max_lag: int) -> float:
    ea = ea - ea.mean()
    eb = eb - eb.mean()
    denom = float(np.sqrt((ea ** 2).sum() * (eb ** 2).sum()) + 1e-18)
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.dot(ea[lag:], eb[: len(eb) - lag]))
        else:
            c = float(np.dot(ea[:lag], eb[-lag:]))
        best = max(best, c / denom)
    return best


def fusion_index(w: np.ndarray, sr: int,
                 band_a: tuple[float, float], band_b: tuple[float, float],
                 expected_rate: float, n_perm: int = 200,
                 seed: int = 0) -> FusionReport:
    """Indice de fusion entre las bandas atribuibles a dos cuerpos.

    En una suma ponderada de procesos independientes, las envolventes y los
    onsets de las dos bandas no estan acoplados => FI ~ 0 y p ~ 0.5. En un
    blend fisico real (excitacion compartida / cross-drive), FI alto y
    p << 0.01. FI mide acoplamiento causal de envolventes/onsets; un gating
    trivial tambien puntua, por eso se acompana de escucha y calidad.
    """
    fs_env = 200.0
    wa = _band(w, sr, band_a)
    wb = _band(w, sr, band_b)
    ea = smooth_env(wa, sr, 5.0, fs_out=fs_env)
    eb = smooth_env(wb, sr, 5.0, fs_out=fs_env)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n], eb[:n]
    max_lag = int(0.020 * fs_env)

    on_a = detect_onsets(wa, sr, expected_rate)
    on_b = detect_onsets(wb, sr, expected_rate)
    tol = int(0.010 * sr)

    rho = _max_corr(ea, eb, max_lag)
    f1_raw = _onset_f1(on_a, on_b, tol)

    # Permutaciones circulares: rompen el alineamiento temporal conservando
    # la estadistica de cada banda. La correccion por azar del F1 se aplica
    # IGUAL al valor observado y a cada permutacion, y el p-valor compara
    # FIs corregidos entre si.
    rng = np.random.default_rng(seed)
    total_n = len(w)
    rho_perm = np.zeros(n_perm)
    f1_perm = np.zeros(n_perm)
    for i in range(n_perm):
        shift_env = int(rng.integers(int(0.1 * n), int(0.9 * n)))
        rho_perm[i] = _max_corr(ea, np.roll(eb, shift_env), max_lag)
        shift = int(rng.integers(int(0.1 * total_n), int(0.9 * total_n)))
        on_b_rot = np.sort((on_b + shift) % total_n)
        f1_perm[i] = _onset_f1(on_a, on_b_rot, tol)

    f_chance = float(f1_perm.mean())
    denom = max(1.0 - f_chance, 1e-9)
    f1_corr = (f1_raw - f_chance) / denom
    fi = 0.5 * max(rho, 0.0) + 0.5 * max(f1_corr, 0.0)
    fi_perm = (0.5 * np.maximum(rho_perm, 0.0)
               + 0.5 * np.maximum((f1_perm - f_chance) / denom, 0.0))
    p = float((np.sum(fi_perm >= fi) + 1) / (n_perm + 1))
    return FusionReport(rho=float(rho), onset_f1=float(f1_corr),
                        fi=float(fi), p_value=p)


@dataclass
class UnityReport:
    """Indice de unicidad de stream entre las bandas ACTIVAS de un render."""
    mean_rho: float
    min_rho: float
    usi: float
    n_bands_active: int


def stream_unity_index(w: np.ndarray, sr: int, n_bands: int = 8,
                       lo: float = 150.0, hi: float = 6000.0,
                       fs_env: float = 200.0,
                       active_floor_db: float = -30.0) -> UnityReport:
    """Unicidad de stream: correlaciones de envolvente entre pares de
    bandas ACTIVAS. UNA corriente => todas las bandas respiran juntas
    (mean_rho alto y min_rho alto); DOS corrientes => algun par cruzado
    cae (~0), y min_rho lo castiga.

    Complemento del fusion_index: FI detecta que dos cuerpos se ACOPLAN;
    USI detecta que ya no hay dos cuerpos. ORIENTATIVO y honesto: un burst
    unico o un gating global tambien puntuan alto -- la escucha manda.
    """
    envs, _ = band_envelopes(w, sr, n_bands=n_bands, lo=lo, hi=hi)
    hop = max(1, int(sr // fs_env))
    envs = envs[:, ::hop]
    rms = np.sqrt((envs ** 2).mean(axis=1))
    floor = rms.max() * 10 ** (active_floor_db / 20.0)
    active = np.where(rms >= floor)[0]
    if len(active) < 2:
        return UnityReport(1.0, 1.0, 1.0, int(len(active)))
    max_lag = int(0.010 * fs_env)
    rhos = []
    for ii in range(len(active)):
        for jj in range(ii + 1, len(active)):
            rhos.append(_max_corr(envs[active[ii]].astype(np.float64),
                                  envs[active[jj]].astype(np.float64), max_lag))
    mean_rho = float(np.mean(rhos))
    min_rho = float(np.min(rhos))
    usi = 0.5 * mean_rho + 0.5 * max(min_rho, 0.0)
    return UnityReport(mean_rho, min_rho, usi, int(len(active)))


def crest_factor_db(w: np.ndarray) -> float:
    """Factor de cresta (pico/RMS) en dB.

    Detector AUXILIAR del defecto de "dos capas": una banda donde el padre
    A esta silenciado y el padre B se cuela crudo tiende a un pico aislado
    sin cuerpo (cresta alta, ~49 dB medidos en la config rota del brief,
    frente a ~20 dB cuando el color toma peso de A). NO forma parte de
    ningun indice compuesto de este fichero: es un sintoma, no una medida
    de fusion. Se expone por si sirve como columna de diagnostico aparte.
    """
    x = np.asarray(w, dtype=np.float64)
    peak = float(np.abs(x).max()) if x.size else 0.0
    rms = float(np.sqrt((x ** 2).mean())) if x.size else 0.0
    if peak < 1e-18 or rms < 1e-18:
        return 0.0
    return float(20.0 * np.log10(peak / rms))


@dataclass
class CompositeFusionReport:
    """Metrica compuesta de fusion (FCI) y sus 4 componentes por separado."""
    mci: float   # coherencia de modulacion 2-16 Hz ponderada por energia
    sso: float   # solape espectral de los PADRES post-alineacion
    dop: float   # tasa de onsets huerfanos grave<->agudo
    bri: float   # identidad residual del padre B original
    fci: float   # 0.4*mci + 0.3*sso - 0.2*dop - 0.1*bri


# Registro compartido por MCI y SSO: mismo rango audible que usa
# stream_unity_index por defecto (150-6000 Hz). No esta fijado por el brief
# para SSO; se reutiliza aqui por coherencia interna del fichero y para que
# re-ponderar el FCI no dependa de dos literales distintos.
_FUSION_LO_HZ = 150.0
_FUSION_HI_HZ = 6000.0
_FUSION_FS_ENV = 200.0  # tasa nominal tras decimar (como en stream_unity_index)


def _mci(blend: np.ndarray, sr: int, n_bands: int) -> float:
    """Coherencia de modulacion 2-16 Hz, ponderada por energia de banda.

    band_envelopes da envolventes a full rate; se decima a ~200 Hz igual que
    el USI (hop = sr//200, sin anti-aliasing -- ver nota en el docstring de
    composite_fusion). Cada envolvente decimada se filtra pasa-banda 2-16 Hz
    (la banda de modulacion perceptual) antes de correlacionar todos los
    pares con _max_corr. El peso rms_i*rms_j (energia del envolvente CRUDO,
    antes del filtro 2-16 Hz) reemplaza al min_rho/floor binario del USI:
    una banda casi muda (el padre A a -90 dB del contexto del brief) pesa
    ~1e-4 en vez de contar igual o vetar todo el indice.
    """
    envs, _ = band_envelopes(blend, sr, n_bands=n_bands, lo=_FUSION_LO_HZ,
                             hi=_FUSION_HI_HZ)
    hop = max(1, int(sr // _FUSION_FS_ENV))
    envs = envs[:, ::hop].astype(np.float64)
    # Guarda de longitud: sosfiltfilt necesita mas muestras que el padlen
    # del filtro; con clips muy cortos (<~150 ms decimados) no hay banda de
    # modulacion 2-16 Hz que extraer de forma fiable.
    if envs.shape[1] < 32:
        return 0.0
    rms = np.sqrt((envs ** 2).mean(axis=1))
    sos = signal.butter(4, [2.0, 16.0], btype="band", fs=_FUSION_FS_ENV,
                        output="sos")
    mod = np.array([signal.sosfiltfilt(sos, e) for e in envs])
    max_lag = int(0.010 * _FUSION_FS_ENV)
    num = 0.0
    den = 0.0
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            w = float(rms[i] * rms[j])
            rho = _max_corr(mod[i], mod[j], max_lag)
            num += w * rho
            den += w
    return float(num / den) if den > 1e-18 else 0.0


def _sso(parent_a: np.ndarray, parent_b_aligned: np.ndarray, sr: int,
         n_bands: int) -> float:
    """Solape espectral (Bhattacharyya) de los perfiles de potencia por
    banda de los dos padres, en el mismo registro audible que MCI/USI.

    Perfil de potencia = energia media por banda de band_envelopes,
    normalizada a suma 1. sum(sqrt(p_a*p_b)) esta en [0, 1]: 1 si los
    perfiles son identicos, ~0 si los padres viven en bandas disjuntas.
    Solo depende de los padres, no del blend: mide si la alineacion de
    registro puso a B donde vive A, no si la fusion "sono" bien.
    """
    envs_a, _ = band_envelopes(parent_a, sr, n_bands=n_bands, lo=_FUSION_LO_HZ,
                               hi=_FUSION_HI_HZ)
    envs_b, _ = band_envelopes(parent_b_aligned, sr, n_bands=n_bands,
                               lo=_FUSION_LO_HZ, hi=_FUSION_HI_HZ)
    power_a = (envs_a.astype(np.float64) ** 2).mean(axis=1)
    power_b = (envs_b.astype(np.float64) ** 2).mean(axis=1)
    p_a = power_a / (power_a.sum() + 1e-18)
    p_b = power_b / (power_b.sum() + 1e-18)
    return float(np.sum(np.sqrt(p_a * p_b)))


def _count_orphans(on_a: np.ndarray, on_b: np.ndarray, tol: int) -> int:
    """Cuenta onsets de on_a sin pareja en on_b a distancia <= tol samples.

    Busqueda del vecino mas cercano por indice (ambos arrays vienen
    ordenados de detect_onsets), simetrica y sin el doble conteo posible
    de _onset_f1 (que no se usa aqui a proposito: mide huerfanos, no F1).
    """
    if len(on_a) == 0:
        return 0
    if len(on_b) == 0:
        return len(on_a)
    orphans = 0
    for a in on_a:
        idx = int(np.searchsorted(on_b, a))
        best = float(abs(int(on_b[idx]) - int(a))) if idx < len(on_b) else np.inf
        if idx > 0:
            best = min(best, float(abs(int(on_b[idx - 1]) - int(a))))
        if best > tol:
            orphans += 1
    return orphans


def _dop(blend: np.ndarray, sr: int) -> float:
    """Tasa de onsets huerfanos entre banda grave (<500 Hz) y aguda
    (>1500 Hz) del blend: dos capas con agendas temporales propias generan
    onsets que no coinciden dentro de +-15 ms; una fusion real, si.

    expected_rate se deriva del propio blend en dos pasadas: una deteccion
    permisiva (expected_rate=50, apenas restringe la distancia minima entre
    picos) da una cuenta de eventos que, dividida por la duracion, se usa
    como expected_rate real para detectar onsets en cada banda.
    """
    low = _band(blend, sr, (40.0, 500.0))
    high = _band(blend, sr, (1500.0, 20000.0))
    dur = len(blend) / sr
    if dur <= 0:
        return 0.0
    permissive = detect_onsets(blend, sr, expected_rate=50.0)
    expected_rate = max(len(permissive) / dur, 1.0)
    on_low = detect_onsets(low, sr, expected_rate)
    on_high = detect_onsets(high, sr, expected_rate)
    tol = int(0.015 * sr)
    total = len(on_low) + len(on_high)
    if total == 0:
        return 0.0
    huerfanos = (_count_orphans(on_low, on_high, tol)
                 + _count_orphans(on_high, on_low, tol))
    return float(huerfanos / total)


def _bri(blend: np.ndarray, parent_a: np.ndarray,
         parent_b_original: np.ndarray, sr: int) -> float:
    """Identidad residual del padre B sin alinear: si el blend sigue la
    agenda temporal de B_original MAS que la de A, la identidad de B se
    esta colando como capa separada pese a la alineacion.

    max_lag = 10 ms a 200 Hz, igual que stream_unity_index (no fijado de
    forma explicita por el brief para BRI; se reutiliza la tolerancia ya
    establecida en el fichero para este tipo de correlacion de envolvente
    decimada, en vez de inventar una nueva).
    """
    e_blend = smooth_env(blend, sr, 5.0, fs_out=_FUSION_FS_ENV)
    e_a = smooth_env(parent_a, sr, 5.0, fs_out=_FUSION_FS_ENV)
    e_b = smooth_env(parent_b_original, sr, 5.0, fs_out=_FUSION_FS_ENV)
    n = min(len(e_blend), len(e_a), len(e_b))
    e_blend, e_a, e_b = e_blend[:n], e_a[:n], e_b[:n]
    max_lag = int(0.010 * _FUSION_FS_ENV)
    corr_b = _max_corr(e_blend, e_b, max_lag)
    corr_a = _max_corr(e_blend, e_a, max_lag)
    return float(max(0.0, corr_b - corr_a))


def composite_fusion(blend: np.ndarray, parent_a: np.ndarray,
                     parent_b_aligned: np.ndarray,
                     parent_b_original: np.ndarray, sr: int,
                     n_bands: int = 12) -> CompositeFusionReport:
    """Indice compuesto de fusion (FCI): pre-filtro que ORDENA candidatos
    DENTRO de una misma pareja de padres, antes de la escucha humana.

    NO es un gate y NO compara entre parejas distintas (SSO ya varia solo
    por el solape natural de cada pareja de padres, no por la calidad de
    la fusion). Nace de una leccion medida en V8: stream_unity_index (USI)
    no discrimina -- satura igual con un gating global o un burst
    compartido que con una fusion real, porque correlaciona envolventes de
    banda entera con un floor binario de "banda activa". El FCI ataca eso
    por cuatro vias que el USI no tiene: restringe la correlacion a la
    banda de modulacion 2-16 Hz (el ritmo perceptual, no el transitorio
    de banda ancha que domina la correlacion cruda), pondera por energia
    (rms_i*rms_j) en vez de min_rho/floor binario, y suma dos componentes
    ausentes del USI -- solape espectral de los padres tras la alineacion
    de registro (SSO) e identidad residual del padre B sin alinear (BRI).

    Los 4 componentes se devuelven por separado a proposito: si el FCI
    agregado ordena mal candidatos que el oido distingue, re-ponderar es
    cambiar la formula de fci en una linea, sin re-renderizar audio.

    ORIENTATIVO y con las mismas limitaciones honestas que el USI: MCI
    puede saturar (un gating global tambien sube la correlacion en 2-16 Hz
    de todas las bandas), y la decimacion a ~200 Hz no tiene anti-aliasing
    (comparte esa limitacion con stream_unity_index a proposito, para no
    reinventar el pipeline). SSO toma pocos valores distintos porque solo
    depende de los padres, no del blend concreto. Ninguno de los 4
    sustituye la escucha.

    parent_b_aligned es el B que realmente entro en la fusion (re-renderizado
    en el registro de A si hubo alineacion); parent_b_original es el B sin
    alinear. Cuando no hubo alineacion, ambos son el mismo array: es el caso
    correcto, no uno especial.

    Determinista: mismos argumentos dan siempre el mismo resultado (sin
    aleatoriedad en ningun componente).
    """
    mci = _mci(blend, sr, n_bands)
    sso = _sso(parent_a, parent_b_aligned, sr, n_bands)
    dop = _dop(blend, sr)
    bri = _bri(blend, parent_a, parent_b_original, sr)
    fci = 0.4 * mci + 0.3 * sso - 0.2 * dop - 0.1 * bri
    return CompositeFusionReport(mci=mci, sso=sso, dop=dop, bri=bri, fci=fci)
