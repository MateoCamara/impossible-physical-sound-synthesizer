"""Capa de COMPOSICION de la cadena de fusion (v12-F2): orquesta lo que ya
existe en blend.py (primitivas de fusion) y blend_recipes.py (banco de
padres) en vez de anadir mecanismos fisicos nuevos.

Por que existe: la demo actual funde dos padres con UNA llamada a
auditory_chimera_colored y suena a DOS capas superpuestas. Medido en
trueno(env) x vidrio(fina) @ 6 bandas: el padre A esta a -56/-85/-94 dB en
3 de las 6 bandas, y esas bandas emiten el padre B crudo -- la chimera no
tiene nada de A que modular ahi. La pareja que si funciona de oido
(fuego x vidrio) es justo la que tiene solape espectral completo entre
padres. De ahi que esta cadena empiece por ALINEAR LOS REGISTROS de los
padres antes de fundir (Slaney, Covell & Lassiter, ICASSP 1996: fundir
sonidos de registro distinto se percibe como DOS objetos; alinear antes de
fundir colapsa la percepcion en uno), y opcionalmente alinee tambien el
TIEMPO de los eventos de la materia a los picos de A (cue de onset comun,
Bregman): register_hz mide el registro, warp_to_anchors/reschedule_drips_to
alinean el tiempo, y render_fusion orquesta todo el pipeline con un dict
meta que deja constancia de lo que realmente paso (no de lo que se pidio).

RAZON CRITICA (repetida aqui porque es la mitad del valor de este modulo):
la alineacion puede fallar EN SILENCIO. Si render_fusion no midiera el
registro CONSEGUIDO (no el pedido), una busqueda de una hora podria
concluir "alinear no ayuda" cuando en realidad no se alineo nada. Por eso
el meta de render_fusion siempre re-mide sobre el audio re-renderizado, y
por eso warp_to_anchors reporta cuantas anclas encontro y cuantas emparejo.

Este modulo NO modifica blend.py ni blend_recipes.py -- los consume. Reusa
deliberadamente _render_drips (blend_recipes.py), que es privada por
convencion de modulo pero publica de facto entre los ficheros de
impossible_mix.physics: el brief de esta tarea pide explicitamente
reutilizarla para el camino exacto de goteo en vez de reimplementarla.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import signal

from impossible_mix.physics.analysis import band_envelopes, detect_onsets, smooth_env
from impossible_mix.physics.blend import (
    BodySpec,
    EventSchedule,
    align_droplet_radius_to_hz,
    auditory_chimera,
    auditory_chimera_colored,
    physical_vocoder,
)
from impossible_mix.physics.blend_recipes import (
    PARENT_MOVABLE,
    _render_drips,
    chimera_parent_v4,
)
from impossible_mix.physics.texture_stats import (
    TextureStats,
    impose_statistics,
    texture_statistics,
)

# ====================================================================
# register_hz: medida de registro (centroide log-frecuencial por RMS)
# ====================================================================

def register_hz(w: np.ndarray, sr: int, n_bands: int = 24,
                lo: float = 60.0, hi: float = 8000.0) -> float:
    """Centroide log-frecuencial ponderado por energia (RMS) por banda:
    resume EN QUE REGISTRO (agudo/grave) vive el peso perceptual de `w`.

    exp(sum(rms_k * log(fc_k)) / sum(rms_k)), sobre las envolventes por
    banda de analysis.band_envelopes (mismo filterbank butter log-espaciado
    que usa el resto del motor -- vocoder fisico y metrica de fusion). Es
    la medida de registro que define el objetivo de alineacion en
    render_fusion: SIEMPRE se llama sobre audio ya renderizado, nunca sobre
    un valor pedido, precisamente para poder detectar cuando la alineacion
    no ha conseguido lo que se le pidio (ver modulo doc, RAZON CRITICA).

    Caso degenerado (silencio total, rms de todas las bandas ~0): el
    denominador se protege con un epsilon y la funcion devuelve exp(0)=1.0
    Hz en vez de lanzar excepcion o dividir por cero -- un valor claramente
    fuera de cualquier rango fisico util, facil de detectar aguas abajo.
    """
    envs, fcs = band_envelopes(w, sr, n_bands=n_bands, lo=lo, hi=hi)
    rms = np.sqrt((envs.astype(np.float64) ** 2).mean(axis=1))
    total = float(rms.sum()) + 1e-12
    return float(np.exp(np.sum(rms * np.log(fcs)) / total))


# Rango de registro seguro que se le permite pedir a chimera_parent_v4.
# GUARDARRAIL (ver brief tarea 1 y 2): goteo/canica con register_hz muy
# bajo (1-5 Hz probado) CUELGAN el sintetizador de gotas (>15s, sin
# lanzar excepcion) -- el radio de burbuja se dispara sin limite superior
# (_bubble_freq_from_radius = 3.26/max(r_m,1e-4), sin recorte). El grid
# de la tarea 4 va a barrer registros con align=True sobre las 6 parejas
# de PAREJAS_V12, asi que este recorte tiene que aplicarse SIEMPRE antes
# de pasar register_hz a chimera_parent_v4, no solo para goteo/canica: es
# mas barato acotar siempre que decidir caso a caso que padre es el
# peligroso. 40 Hz - 12 kHz cubre con margen holgado el registro de los
# ocho padres del banco v10 (goteo natural ~1480 Hz Minnaert, campana_tela
# ~450-1200 Hz, vidrio ~1800-9800 Hz de modos) sin acercarse al extremo
# bajo peligroso ni al recorte superior de vidrio/campana_tela documentado
# en chimera_parent_v4.
_REGISTER_CLAMP_LO_HZ = 40.0
_REGISTER_CLAMP_HI_HZ = 12_000.0


def _clamp_register_hz(hz: float) -> tuple[float, bool]:
    """Recorta al rango seguro; devuelve (valor recortado, se recorto?)."""
    clamped = float(np.clip(hz, _REGISTER_CLAMP_LO_HZ, _REGISTER_CLAMP_HI_HZ))
    return clamped, (clamped != hz)


# ====================================================================
# warp_to_anchors: alineacion temporal de B a los picos de A
# ====================================================================

# Tasa de eventos "esperada" generica que se le pasa a detect_onsets y a
# _envelope_peaks (para fijar la distancia minima entre picos). No hay un
# valor correcto universal -- los padres van de trueno (cross-driven,
# 1.5-16 Hz) a fuego (crepitar denso) a goteo (10 Hz fijo) -- asi que esto
# es un punto medio deliberado, no medido, documentado aqui para que quien
# ajuste el barrido de la tarea 4 sepa que es un hiperparametro de ESTE
# modulo (queda tambien en meta["onset_expected_rate_hz"] de render_fusion
# para que no quede escondido).
_ONSET_EXPECTED_RATE_HZ = 8.0


def _envelope_peaks(w: np.ndarray, sr: int, expected_rate: float) -> np.ndarray:
    """Picos de la envolvente de BANDA ANCHA (Hilbert suavizado), para las
    anclas de A.

    A diferencia de analysis.detect_onsets (que filtra pasa-alto >1kHz
    ANTES de picar picos): medido en vivo, trueno tiene su banda >1kHz a
    -37 dB relativo al RMS de banda completa, y detect_onsets(trueno,...)
    encuentra 1 (UN) onset en 8s -- esta pickeando ruido de punto flotante,
    no eventos reales (picos de envolvente banda ancha sobre la MISMA senal
    encuentran 26). Como A puede ser justo ese tipo de padre grave sin
    agudos (trueno, fuego, oceano...), usar detect_onsets para las anclas
    de A repetiria en el warp exactamente el fallo silencioso que la
    alineacion de registro ya tuvo que resolver (ver modulo doc, RAZON
    CRITICA): anclas que PARECEN validas pero no representan nada.

    Para B (la materia: goteo/vidrio/canica/campana_tela) SI se usa
    detect_onsets en render_fusion -- todos esos padres tienen transitorios
    de agudos reales (verificado: >1kHz entre -7 y -0.1 dB relativo al RMS
    de banda completa en los seis padres del banco movil), asi que el
    filtro pasa-alto ahi es correcto, no un problema.
    """
    env = smooth_env(w, sr, win_ms=5.0)
    thr = 0.35 * float(env.max())
    min_dist = max(int(0.008 * sr), int(0.5 / max(expected_rate, 1.0) * sr))
    peaks, _ = signal.find_peaks(env, height=thr, distance=min_dist)
    return peaks


def _schedule_from_envelope_peaks(w: np.ndarray, sr: int,
                                  expected_rate: float) -> EventSchedule:
    """Construye un EventSchedule a partir de _envelope_peaks(w) en vez de
    blend.schedule_from_audio (que usa analysis.detect_onsets, pasa-alto
    >1kHz, por dentro).

    RULING sobre el brief (aplicado tras hallazgo medido): el brief define
    el warp como "onsets de B hacia los PICOS DE LA ENVOLVENTE de A", y
    solo menciona schedule_from_audio para el caso especial de goteo como
    atajo de implementacion. Medido en vivo: schedule_from_audio(trueno,
    sr, 8.0) encuentra 1 evento en 8s (el filtro pasa-alto interno lee
    ruido de punto flotante en un padre sin agudos, -37 dB relativo en
    banda >1kHz) donde _envelope_peaks(trueno,...) encuentra 26 -- el
    mismo problema que _envelope_peaks ya existe para evitar en el camino
    generico de warp_to_anchors. La especificacion de la ancla ("picos de
    envolvente") manda sobre el atajo de implementacion nombrado en el
    brief; por eso reschedule_drips_to usa esta funcion (unica via, sin
    fallback condicional a schedule_from_audio: mas simple y uniforme con
    el resto del modulo, que siempre usa _envelope_peaks para las anclas
    de A).

    Misma normalizacion de amplitud y misma logica de dur_hint que
    schedule_from_audio (para que el comportamiento aguas abajo de
    _render_drips -- que lee sched.amps como amplitud por evento -- no
    cambie de significado): amplitud = envolvente suavizada en el pico,
    normalizada al maximo; dur_hint = hueco hasta el siguiente pico (el
    ultimo hueco se repite para el ultimo evento); sin picos, EventSchedule
    vacio.
    """
    peaks = _envelope_peaks(w, sr, expected_rate)
    if len(peaks) == 0:
        return EventSchedule(
            starts=np.zeros(0, dtype=np.int64),
            amps=np.zeros(0, dtype=np.float64),
            vels=np.zeros(0, dtype=np.float64),
            dur_hint=np.zeros(0, dtype=np.float64),
            meta={"source": "envelope_peaks", "expected_rate": expected_rate},
        )
    env = smooth_env(w, sr, win_ms=5.0)
    env_idx = np.clip(peaks, 0, len(env) - 1)
    amps_at_peaks = env[env_idx]
    peak_max = float(amps_at_peaks.max())
    amps_norm = amps_at_peaks / (peak_max + 1e-12)
    if len(peaks) > 1:
        gaps_s = np.diff(peaks).astype(np.float64) / sr
        last_gap = gaps_s[-1] if len(gaps_s) else 1.0 / max(expected_rate, 1e-6)
        dur_hint = np.append(gaps_s, last_gap)
    else:
        dur_hint = np.array([1.0 / max(expected_rate, 1e-6)], dtype=np.float64)
    return EventSchedule(
        starts=peaks.astype(np.int64), amps=amps_norm.astype(np.float64),
        vels=np.ones(len(peaks), dtype=np.float64),
        dur_hint=dur_hint.astype(np.float64),
        meta={"source": "envelope_peaks", "expected_rate": expected_rate},
    )


def _monotone_match(src: np.ndarray, dst: np.ndarray,
                    max_shift: float) -> list[tuple[float, float]]:
    """Empareja cada elemento de `src` (ascendente) con el elemento de
    `dst` (ascendente) mas cercano AUN NO USADO, avanzando un puntero de
    forma monotona (la ancla usada nunca decrece de un onset al siguiente).
    Descarta el par si |src-dst| > max_shift. O(len(src)+len(dst)),
    determinista, sin aleatoriedad.
    """
    matches: list[tuple[float, float]] = []
    j = 0
    n_dst = len(dst)
    for s in src:
        while j + 1 < n_dst and abs(dst[j + 1] - s) <= abs(dst[j] - s):
            j += 1
        if j < n_dst and abs(dst[j] - s) <= max_shift:
            matches.append((float(s), float(dst[j])))
            j += 1
    return matches


def _time_map(n: int, matches: list[tuple[float, float]]) -> np.ndarray:
    """Mapa de tiempo por tramos t_in[t_out] a partir de pares (src, dst)
    ya emparejados y monotonos (ver _monotone_match). Ancla los extremos
    del clip a identidad (0->0, n-1->n-1) para no dejar el mapa
    indefinido fuera del primer/ultimo emparejamiento, y fuerza t_in a NO
    decrecer nunca (`np.maximum.accumulate`) -- eso es "limitar la
    pendiente para no invertir el tiempo": sin esto, dos anclas muy
    cercanas en dst pero muy separadas en src invertirian el sentido de la
    lectura de `w` entre ellas. Factorizado aparte de warp_to_anchors para
    poder testear la garantia de monotonia directamente sobre el mapa
    (sobre audio ya resampleado no se puede recuperar).
    """
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if not matches:
        return np.arange(n, dtype=np.float64)
    dst_pts = np.concatenate(([0.0], [d for _, d in matches], [float(n - 1)]))
    src_pts = np.concatenate(([0.0], [s for s, _ in matches], [float(n - 1)]))
    dst_pts, uniq_idx = np.unique(dst_pts, return_index=True)
    src_pts = src_pts[uniq_idx]
    src_pts = np.maximum.accumulate(src_pts)
    t_out = np.arange(n, dtype=np.float64)
    return np.clip(np.interp(t_out, dst_pts, src_pts), 0.0, float(n - 1))


def warp_to_anchors(w: np.ndarray, sr: int, src_onsets: np.ndarray,
                    dst_anchors: np.ndarray, max_shift_ms: float = 60.0,
                    return_meta: bool = False):
    """Alinea temporalmente los eventos de `w` (tipicamente B) a las
    anclas `dst_anchors` (tipicamente los picos de envolvente de A): cue
    de onset comun de Bregman -- si B "respira" en el mismo instante que
    A, se perciben como un unico stream.

    Emparejamiento greedy MONOTONO (`_monotone_match`): cada onset de B se
    empareja con la ancla mas cercana aun no usada; se descarta el par si
    |delta| > max_shift_ms. Con los pares emparejados se construye un mapa
    de tiempo por tramos (`_time_map`, np.interp sobre una malla monotona
    garantizada -- nunca invierte el sentido de la lectura) y se
    resamplea `w` LINEALMENTE sobre ese mapa. Nada de phase-vocoder: esto
    estira/comprime el tiempo re-muestreando, con el cambio de tono que
    conlleva un resample lineal -- aceptable para el desplazamiento
    pequeno (<=60ms por defecto) que pide max_shift_ms; para eventos que
    haya que reagendar por completo (goteo) usar reschedule_drips_to en
    su lugar, que no tiene este efecto secundario.

    `return_meta=False` (por defecto) devuelve solo el audio warpeado,
    igual que antes. `return_meta=True` devuelve `(audio, diag)` con
    `diag = {"n_anchors", "n_onsets_matched", "n_onsets_total",
    "warp_max_rate_dev"}` -- los mismos diagnosticos que render_fusion
    necesita para su meta. Esta es la UNICA implementacion del warp
    generico del modulo: render_fusion la llama con `return_meta=True` en
    vez de reimplementar `_monotone_match`/`_time_map`/`np.interp` por su
    cuenta, para que `max_shift_ms` tenga una sola fuente de verdad (si
    cambia el default aqui, cambia tambien el de render_fusion, que no le
    pasa un valor propio).
    """
    n = len(w)
    max_shift = max_shift_ms / 1000.0 * sr
    src = np.sort(np.asarray(src_onsets, dtype=np.float64))
    dst = np.sort(np.asarray(dst_anchors, dtype=np.float64))
    matches = _monotone_match(src, dst, max_shift)
    t_in = _time_map(n, matches)
    y = np.interp(t_in, np.arange(n, dtype=np.float64),
                 w.astype(np.float64)).astype(np.float32)
    if not return_meta:
        return y
    warp_max_rate_dev = (float(np.abs(np.diff(t_in) - 1.0).max())
                         if len(t_in) > 1 else 0.0)
    diag = {
        "n_anchors": len(dst),
        "n_onsets_matched": len(matches),
        "n_onsets_total": len(src),
        "warp_max_rate_dev": warp_max_rate_dev,
    }
    return y, diag


def reschedule_drips_to(a: np.ndarray, sr: int, n: int, seed: int, *,
                        expected_rate: float = _ONSET_EXPECTED_RATE_HZ,
                        radius_mm: float = 2.2,
                        surface: str = "water") -> np.ndarray:
    """Caso especial `goteo`: en vez de resamplear linealmente un goteo ya
    renderizado (warp_to_anchors, que emborronaria el click de cada
    impacto), RE-AGENDA gotas nuevas exactamente en los picos de la
    envolvente de `a`.

    _schedule_from_envelope_peaks(a, sr, expected_rate) lee los picos de
    envolvente banda ancha de `a` y sus amplitudes (normalizadas) y
    produce un EventSchedule -- MISMAS anclas que warp_to_anchors usa para
    el resto de padres (ver _envelope_peaks: picos de envolvente, no
    onsets de transitorios agudos); blend_recipes._render_drips sintetiza
    una gota REAL (synth_drip_event) en cada evento de ese schedule. El
    resultado es exacto y determinista: cada gota nace justo cuando `a`
    respira, con su timbre de impacto intacto -- preferible al warp de
    senal cuando aplica (B == "goteo").

    HISTORIAL (por que no es blend.schedule_from_audio, que el brief
    original nombraba para este camino): la primera version de esta
    funcion usaba schedule_from_audio, que filtra pasa-alto >1kHz antes de
    picar picos (analysis.detect_onsets). Medido en trueno(a): 1 evento
    reagendado en 8s (RMS de B cae ~-18 dB) frente a 26 picos de
    envolvente reales sobre la MISMA senal -- trueno esta a -37 dB
    relativo en banda >1kHz, asi que detect_onsets estaba pickeando ruido
    de punto flotante, no eventos. El brief define la ancla del warp como
    "picos de la envolvente de A"; schedule_from_audio era solo un atajo
    de implementacion para este caso que resulto incorrecto con un padre
    grave sin agudos marcados. Corregido tras ruling explicito para usar
    las mismas anclas que el camino generico de warp_to_anchors, sin
    fallback condicional a schedule_from_audio (uniforme y mas simple).

    `radius_mm` debe ser el mismo radio que ya se uso para renderizar el
    goteo (2.2mm por defecto, o align_droplet_radius_to_hz(registro) si
    hubo alineacion de registro) para que el timbre de las gotas
    reagendadas sea consistente con el resto del pipeline.
    """
    sched = _schedule_from_envelope_peaks(a, sr, expected_rate)
    n_evt = len(sched.starts)
    sched.radii_mm = (np.full(n_evt, radius_mm, dtype=np.float64)
                      if n_evt else None)
    return _render_drips(sched, sr, n, seed, surface=surface)


# ====================================================================
# PHYSICAL_BODIES: mapa padre-materia -> fabrica de BodySpec
# ====================================================================
# Factorias en vez de BodySpec ya construidos: vidrio y campana_tela son
# estaticos, pero canica/goteo son "minnaert" y su BodySpec.ref (el radio)
# depende del registro con el que se quiera excitar el cuerpo -- no hay un
# BodySpec unico posible para ellos sin conocer ese registro primero. El
# brief describe esto como "Mapa padre-materia -> BodySpec"; aqui es un
# mapa padre-materia -> (registro_hz -> BodySpec) para poder representar
# los cuatro casos con una unica estructura de datos consultable por
# pertenencia (`nombre in PHYSICAL_BODIES`), que es como la tarea 4 lo usa
# para decidir si method="vocoder" aplica a una pareja.

def _body_vidrio(_register_hz: float | None) -> BodySpec:
    return BodySpec("surface", "glass")


def _body_campana_tela(_register_hz: float | None) -> BodySpec:
    # Aproximacion documentada en el brief: perfil "fabric" (existe en
    # SURFACE_PROFILES) con cola alargada (t60_scale=2.0) para acercarse al
    # timbre de campana de tela. El brief deja anotado un fallback si suena
    # mal (BodySpec("modal", "metal", t60_scale=...)) pero no fija el
    # t60_scale de ese fallback -- decidirlo sin escuchar queda fuera de
    # alcance de esta tarea; se deja constancia aqui para quien escuche.
    return BodySpec("surface", "fabric", t60_scale=2.0)


def _body_minnaert(register_hz_val: float | None) -> BodySpec:
    reg = register_hz_val if register_hz_val is not None else 200.0
    return BodySpec("minnaert", align_droplet_radius_to_hz(reg))


PHYSICAL_BODIES: dict[str, Callable[[float | None], BodySpec]] = {
    "vidrio": _body_vidrio,
    "campana_tela": _body_campana_tela,
    "canica": _body_minnaert,
    "goteo": _body_minnaert,
}


# ====================================================================
# FusionSpec + render_fusion: orquestacion
# ====================================================================

_METHODS = frozenset({"chimera", "chimera_plana", "vocoder", "suma"})


@dataclass
class FusionSpec:
    """Configuracion completa de una fusion. `env_parent` pone la
    envolvente/dinamica (A), `fine_parent` pone la estructura fina/materia
    (B) -- misma convencion que blend_recipes.CHIMERA_PAIRS."""
    env_parent: str
    fine_parent: str
    n_bands: int = 8
    align: bool = False
    warp: bool = False
    color_mix: float | None = None
    a_floor_db: float = -40.0
    stats_finish: bool = False
    method: str = "chimera"
    duration_s: float = 8.0
    seed: int = 42


# Registro del sr asociado a cada dict de cache, indexado por id(cache) --
# deliberadamente FUERA del propio dict de cache (no como una entrada mas
# dentro de el). Guardar el guardarraíl como clave dentro del cache
# mezclaria un int con las entradas (nombre, duration_s, seed, register_hz,
# density_mul) -> ndarray que espera la tarea 4; si esta itera
# cache.values() o usa len(cache) como "numero de padres cacheados",
# tropezaria con esa entrada extra. Aqui vive aparte y el dict de cache que
# ve el llamante queda con SOLO las claves literales que pide el brief.
#
# Riesgo aceptado y documentado: este registro nunca libera entradas
# (id(cache) -> sr se queda para siempre), asi que si un dict de cache se
# recolecta y Python REUTILIZA su id() para un dict nuevo y no relacionado
# que tambien se use como cache con un sr distinto, el guardarraíl podria
# disparar una alarma FALSA (ValueError sobre un uso legitimo). Nunca al
# reves: no puede devolver audio del sr equivocado en silencio, que es el
# fallo real que este guardarraíl existe para evitar. En el uso previsto
# (un cache por pareja, vivo durante todo un barrido) este riesgo es
# teorico.
_CACHE_SR_REGISTRY: dict[int, int] = {}


def _cached_parent(cache: dict | None, name: str, duration_s: float, seed: int,
                   sr: int, register_hz_val: float | None,
                   density_mul: float = 1.0) -> np.ndarray:
    """Renderiza (o recupera de `cache`) un padre via chimera_parent_v4.

    Clave de cache: (nombre, duration_s, seed, register_hz, density_mul) --
    literal segun el brief, SIN sr, y son las UNICAS entradas que este
    modulo escribe en `cache`. Eso es correcto mientras un mismo `cache`
    se use siempre al mismo sr (el uso previsto: un barrido de la tarea 4
    que fija sr=44100 para todas sus ~100 llamadas por pareja); si se
    reutilizase el mismo dict de cache con dos sr distintos devolveria
    audio del sr equivocado sin avisar. Guardarraíl barato: el sr de cada
    `cache` se anota en _CACHE_SR_REGISTRY (fuera del propio dict, ver su
    comentario) la primera vez que se usa, y se lanza un error claro si
    una llamada posterior trae un sr distinto para el mismo `cache`.
    """
    if cache is not None:
        cache_id = id(cache)
        cached_sr = _CACHE_SR_REGISTRY.get(cache_id)
        if cached_sr is None:
            _CACHE_SR_REGISTRY[cache_id] = sr
        elif cached_sr != sr:
            raise ValueError(
                f"cache de render_fusion creado con sr={cached_sr} y "
                f"reutilizado con sr={sr}: la clave de cache no incluye sr "
                "(asi lo pide el brief), asi que mezclar sr distintos en el "
                "mismo dict devolveria audio del sr equivocado. Usa un "
                "cache por sr."
            )
    key = (name, duration_s, seed, register_hz_val, density_mul)
    if cache is not None and key in cache:
        return cache[key]
    w = chimera_parent_v4(name, duration_s=duration_s, seed=seed, sr=sr,
                          register_hz=register_hz_val, density_mul=density_mul)
    if cache is not None:
        cache[key] = w
    return w


def _normalize_peak(w: np.ndarray, peak_max: float = 0.95) -> np.ndarray:
    """Recorta el pico a peak_max SOLO si lo supera (igual que _norm en
    blend_recipes.py): no infla senales que ya estan por debajo del techo."""
    peak = float(np.abs(w).max() + 1e-9)
    if peak > peak_max:
        w = w * (peak_max / peak)
    return w.astype(np.float32)


def _interp_corr_psd(corr_a: np.ndarray, corr_b: np.ndarray,
                     alpha: float) -> np.ndarray:
    """Interpolacion de matrices de correlacion proyectada a PSD (mismo
    procedimiento que texture_stats.interp_stats): autovalores negativos
    de la interpolacion lineal se recortan a 0 antes de re-normalizar la
    diagonal a 1, para que el resultado siga siendo una correlacion valida
    (Cholesky, dentro de impose_statistics, la necesita PSD)."""
    corr = (1 - alpha) * corr_a + alpha * corr_b
    vals, vecs = np.linalg.eigh((corr + corr.T) / 2)
    corr = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d)


def _apply_stats_finish(blend: np.ndarray, a: np.ndarray, sr: int, *,
                        alpha: float = 0.5, n_iter: int = 8,
                        seed: int = 0) -> np.ndarray:
    """Acabado opcional (stats_finish): texture_statistics del blend como
    base; empuja SOLO env_corr y mod_power hacia las del padre A (alpha)
    y re-sintetiza con impose_statistics.

    AVISO (es sintesis, no un filtro de pulido): impose_statistics arranca
    desde ruido (`rng.standard_normal(n)*0.1`, ver texture_stats.py) e
    impone las estadisticas objetivo iterativamente -- NUNCA ve la forma
    de onda del blend, solo sus estadisticas. El brief lo llama "pulido,
    no sintesis", pero la funcion que invoca es literalmente sintesis
    desde ruido; lo que sobrevive del blend original es su PERFIL
    ESTADISTICO (env_mean/env_std/env_skew se mantienen intactos, solo se
    desplazan env_corr y mod_power hacia A), no su forma de onda exacta.
    Documentado tambien en el informe de esta tarea para quien compare
    stats_finish=True contra stats_finish=False en el barrido de la
    tarea 4: no es una comparacion "mismo audio + retoque", es "blend" vs.
    "textura resintetizada con el perfil estadistico del blend".
    """
    n = min(len(blend), len(a))
    stats_blend = texture_statistics(blend[:n], sr)
    stats_a = texture_statistics(a[:n], sr)
    corr = _interp_corr_psd(stats_blend.env_corr, stats_a.env_corr, alpha)
    mod_power = (1 - alpha) * stats_blend.mod_power + alpha * stats_a.mod_power
    finish_stats = TextureStats(
        env_mean=stats_blend.env_mean, env_std=stats_blend.env_std,
        env_skew=stats_blend.env_skew, env_corr=corr, mod_power=mod_power,
        band_edges=stats_blend.band_edges, mod_edges=stats_blend.mod_edges,
    )
    return impose_statistics(finish_stats, sr, n / sr, n_iter=n_iter, seed=seed)


# color_mix y a_floor_db solo los lee auditory_chimera_colored, que solo se
# invoca bajo method="chimera" (paso 3 de render_fusion). Bajo los otros
# tres methods esos dos campos de FusionSpec se ignoran en silencio si no
# se anota en algun sitio -- y como NINGUNO de los dos cambia la firma de
# auditory_chimera/_normalize_peak/physical_vocoder, dos FusionSpec que
# solo difieran en color_mix/a_floor_db con method="chimera_plana" (o
# "suma", o "vocoder") producen audio BIT-IDENTICO. Para que el barrido de
# la tarea 4 no lo confunda con dos condiciones experimentales distintas,
# render_fusion anota en meta["parametros_inertes"] cuales de los campos
# del spec no tuvieron ningun efecto en ESTE render, dado su method.
#
# Dos casos adicionales, mismo tipo de bug, que este mapa estatico NO
# alcanza a cubrir por si solo (se resuelven con un `if` extra al poblar
# meta, ver mas abajo en render_fusion):
# - "suma" tambien deja inerte n_bands: `_normalize_peak(0.6*a+0.6*b)` no
#   lo lee en absoluto -- el barrido de la tarea 4 sondea n_bands en
#   {2..32} y tambien emite una fila-ancla method="suma"; sin esto, esa
#   fila registraria un n_bands que no hizo nada.
# - "chimera" deja inerte a_floor_db CUANDO color_mix es None (el default
#   de FusionSpec): en blend.py, auditory_chimera_colored solo lee
#   a_floor_db dentro de `if color_mix is not None:` -- con color_mix=None
#   se toma la rama color_from (bit-identica a la version sin suelo) y
#   a_floor_db nunca se evalua. El brief de la tarea 4 incluye
#   explicitamente una celda "baseline V11 exacto (coloreada,
#   color_mix=None, ...)" que cae justo en este caso.
_INERT_PARAMS_BY_METHOD: dict[str, tuple[str, ...]] = {
    "chimera": (),
    "chimera_plana": ("color_mix", "a_floor_db"),
    "suma": ("color_mix", "a_floor_db", "n_bands"),
    "vocoder": ("color_mix", "a_floor_db"),
}


def render_fusion(spec: FusionSpec, sr: int = 44_100,
                  cache: dict | None = None) -> tuple[np.ndarray, dict]:
    """Orquesta la cadena completa de fusion segun `spec`. Devuelve
    (audio, meta). Determinista: mismos spec y sr dan salida bit-identica
    (con o sin cache -- el cache solo evita recomputo, nunca cambia el
    resultado).

    Pasos (ver modulo doc para el porque):
    1. Renderiza los dos padres via chimera_parent_v4 (cacheable). Si
       spec.align: mide register_hz del padre ANCLA y re-renderiza el
       padre MOVIL (el que este en PARENT_MOVABLE) en ese registro,
       recortado a [40, 12000] Hz por seguridad (ver _REGISTER_CLAMP_*).
       Si ninguno de los dos padres es movil, align queda registrado como
       no-op en meta (align_noop=True) y no se toca nada.
       Si AMBOS son moviles se mueve fine_parent (la materia) y se usa
       env_parent (la dinamica) como ancla -- coincide con el unico
       ejemplo del brief (goteo alineado al registro de trueno) y con que
       PARENT_MOVABLE es exactamente el conjunto de padres-materia: mover
       la dinamica que define "que fenomeno es este" en vez de la materia
       tendria mas sentido invertido solo si algun dia se necesita, pero
       no hay caso de uso para ello en las parejas curadas actuales.
    2. Warp opcional de B (fine_parent) hacia los picos de A (env_parent).
       Caso especial B=="goteo": reschedule_drips_to (exacto, sin
       resample de senal). Caso general: onsets de B via
       analysis.detect_onsets (B es siempre materia, con agudos reales) y
       anclas de A via _envelope_peaks (banda ancha -- ver su docstring:
       detect_onsets sobre A puede estar leyendo ruido de punto flotante
       si A no tiene energia en agudos, p.ej. trueno).
    3. Fusion segun method (chimera / chimera_plana / vocoder / suma).
    4. Acabado opcional stats_finish (ver _apply_stats_finish).

    NOTA: warp no tiene efecto perceptible en method="vocoder": el
    articulador de physical_vocoder es SIEMPRE `a` (env_parent) sin
    warpear, y `b` (fine_parent, la que si se warpea) solo se usa para
    medir el registro del BodySpec, no como audio. Las celdas
    warp=True x method="vocoder" del barrido de la tarea 4 seran casi
    identicas a warp=False x method="vocoder"; se deja tal cual (cambiar
    el articulador para method="vocoder" no lo pide el brief y mezclaria
    dos decisiones de diseno independientes) pero queda anotado aqui y en
    el informe de la tarea para que no se lea como que el warp fallo.

    NOTA: color_mix y a_floor_db solo los consume auditory_chimera_colored,
    es decir SOLO aplican bajo method="chimera". Bajo "chimera_plana"
    (auditory_chimera no los acepta), "suma" (formula fija 0.6a+0.6b) y
    "vocoder" (physical_vocoder no los acepta) quedan sin efecto: dos
    FusionSpec que solo difieran en esos dos campos con el mismo method no
    "chimera" dan audio bit-identico. Dos matices mas finos, mismo
    problema: `n_bands` tambien queda inerte bajo "suma" (no lo lee la
    formula 0.6a+0.6b); y `a_floor_db` queda inerte incluso bajo
    "chimera" cuando `color_mix is None` (el default de FusionSpec) --
    auditory_chimera_colored solo evalua a_floor_db dentro de la rama
    `color_mix is not None`. meta["parametros_inertes"] (ver
    _INERT_PARAMS_BY_METHOD y el `if` que lo completa mas abajo) deja
    constancia de TODO esto por render, no solo por method, para que el
    barrido de la tarea 4 no lo lea como condiciones experimentales
    distintas cuando no lo son.
    """
    if spec.method not in _METHODS:
        raise ValueError(f"method desconocido: {spec.method!r} "
                         f"(validos: {sorted(_METHODS)})")

    parametros_inertes = list(_INERT_PARAMS_BY_METHOD[spec.method])
    if spec.method == "chimera" and spec.color_mix is None:
        # a_floor_db solo se lee dentro de la rama `color_mix is not None`
        # de auditory_chimera_colored (blend.py) -- con color_mix=None
        # (el default de FusionSpec) esta inerte tambien bajo "chimera".
        # Ver comentario largo junto a _INERT_PARAMS_BY_METHOD.
        parametros_inertes.append("a_floor_db")

    meta: dict = {"n_bands": spec.n_bands,
                 "onset_expected_rate_hz": _ONSET_EXPECTED_RATE_HZ,
                 "parametros_inertes": parametros_inertes}

    seed_env = spec.seed
    seed_fine = spec.seed + 17

    # --- 1. padres + alineacion opcional de registro ---
    a = _cached_parent(cache, spec.env_parent, spec.duration_s, seed_env, sr,
                       None, 1.0)
    b = _cached_parent(cache, spec.fine_parent, spec.duration_s, seed_fine, sr,
                       None, 1.0)

    moved_parent: str | None = None
    align_noop = False
    register_target_hz: float | None = None
    register_achieved_hz: float | None = None
    register_error_hz: float | None = None
    register_before_move_hz: float | None = None
    register_clamped = False

    if spec.align:
        fine_movable = spec.fine_parent in PARENT_MOVABLE
        env_movable = spec.env_parent in PARENT_MOVABLE
        if not fine_movable and not env_movable:
            align_noop = True
        else:
            if fine_movable:
                moved_parent, moved_seed, anchor_audio, audio_antes_de_mover = (
                    spec.fine_parent, seed_fine, a, b)
            else:
                moved_parent, moved_seed, anchor_audio, audio_antes_de_mover = (
                    spec.env_parent, seed_env, b, a)
            # register_before_move_hz: registro del padre MOVIL (moved_parent)
            # medido sobre su render de ANTES de re-renderizarlo con el
            # register_hz objetivo (su render "natural"). audio_antes_de_mover
            # es justo eso -- el audio del padre que SI se va a mover, tal
            # como estaba antes de moverse (nombre elegido para no
            # confundirlo con "el padre que no se mueve", que es
            # anchor_audio). Sin este dato, register_error_hz (medido vs. el
            # OBJETIVO, no vs. donde estaba antes) no distingue "la
            # alineacion funciono pero register_hz tiene un sesgo de
            # medida" de "la alineacion no movio nada" -- ambos casos
            # pueden dar un error absoluto grande. Con register_before_move_hz
            # el criterio correcto es "se acerco al objetivo", no "el error
            # absoluto es pequeno" (ver informe de esta tarea).
            register_before_move_hz = register_hz(audio_antes_de_mover, sr)
            target_raw = register_hz(anchor_audio, sr)
            register_target_hz, register_clamped = _clamp_register_hz(target_raw)
            moved_audio = _cached_parent(cache, moved_parent, spec.duration_s,
                                         moved_seed, sr, register_target_hz, 1.0)
            register_achieved_hz = register_hz(moved_audio, sr)
            register_error_hz = abs(register_achieved_hz - register_target_hz)
            if moved_parent == spec.fine_parent:
                b = moved_audio
            else:
                a = moved_audio

    meta.update(
        moved_parent=moved_parent, align_noop=align_noop,
        register_target_hz=register_target_hz,
        register_achieved_hz=register_achieved_hz,
        register_error_hz=register_error_hz,
        register_before_move_hz=register_before_move_hz,
        register_clamped=register_clamped,
    )

    # Longitudes consistentes ANTES de deteccion de onsets/warp/fusion:
    # distintos generadores fisicos pueden diferir en 1-2 muestras por
    # redondeo de duration_s*sr.
    n_common = min(len(a), len(b))
    a, b = a[:n_common], b[:n_common]

    # --- 2. warp opcional de B hacia los picos de A ---
    if spec.warp:
        if spec.fine_parent == "goteo":
            radius = (align_droplet_radius_to_hz(register_target_hz)
                      if register_target_hz is not None else 2.2)
            b = reschedule_drips_to(a, sr, n_common, seed_fine,
                                    expected_rate=_ONSET_EXPECTED_RATE_HZ,
                                    radius_mm=radius)
            # n_evt: cuantos picos de envolvente de A uso reschedule_drips_to
            # (MISMAS anclas, _envelope_peaks -- ver su docstring y el
            # HISTORIAL en reschedule_drips_to: ya NO es
            # schedule_from_audio/detect_onsets, que pickeaba ruido de
            # punto flotante en padres graves como trueno). Aqui
            # matched==total==n_anchors por construccion (CADA pico de A
            # se convierte en una gota, no hay descarte por max_shift como
            # en el camino generico) -- eso no es una "tasa de exito", es
            # el numero total de eventos reagendados.
            n_evt = len(_envelope_peaks(a, sr, _ONSET_EXPECTED_RATE_HZ))
            meta.update(warp_method="reschedule_drips", n_anchors=n_evt,
                       n_onsets_matched=n_evt, n_onsets_total=n_evt,
                       warp_max_rate_dev=None)
        else:
            # Unica fuente de verdad para el warp generico: warp_to_anchors
            # con return_meta=True (ver su docstring). Antes este bloque
            # reimplementaba _monotone_match/_time_map/np.interp en linea,
            # con max_shift_ms hardcodeado por separado -- eso desacoplaba
            # el comportamiento de render_fusion del default publico de
            # warp_to_anchors. No se le pasa max_shift_ms explicitamente
            # para heredar SIEMPRE el default de la funcion publica.
            src = detect_onsets(b, sr, _ONSET_EXPECTED_RATE_HZ)
            dst = _envelope_peaks(a, sr, _ONSET_EXPECTED_RATE_HZ)
            b, warp_diag = warp_to_anchors(b, sr, src, dst, return_meta=True)
            meta.update(warp_method="signal_warp", **warp_diag)

    # --- 3. fusion segun method ---
    if spec.method == "chimera":
        out = auditory_chimera_colored(a, b, sr, n_bands=spec.n_bands,
                                       color_mix=spec.color_mix,
                                       a_floor_db=spec.a_floor_db)
    elif spec.method == "chimera_plana":
        out = auditory_chimera(a, b, sr, n_bands=spec.n_bands)
    elif spec.method == "suma":
        out = _normalize_peak(0.6 * a + 0.6 * b)
    else:  # "vocoder"
        body_factory = PHYSICAL_BODIES.get(spec.fine_parent)
        if body_factory is None:
            raise ValueError(
                f"method='vocoder' pero fine_parent={spec.fine_parent!r} no "
                f"tiene BodySpec en PHYSICAL_BODIES ({sorted(PHYSICAL_BODIES)}). "
                "El llamante debe filtrar por PHYSICAL_BODIES antes de pedir "
                "method='vocoder' para este padre-materia -- no hay un "
                "fallback razonable que no falle a medias."
            )
        vocoder_register = register_hz(b, sr)
        body = body_factory(vocoder_register)
        meta["vocoder_body_register_hz"] = vocoder_register
        out = physical_vocoder(a, sr, body, n_bands=spec.n_bands)

    meta["method_effective"] = spec.method

    # --- 4. acabado opcional ---
    if spec.stats_finish:
        out = _apply_stats_finish(out, a, sr, alpha=0.5, n_iter=8, seed=spec.seed)

    return out, meta
