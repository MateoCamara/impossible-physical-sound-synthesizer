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
