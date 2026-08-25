"""Fase 1 del blend profundo v7: separa el "cuando/como ocurre cada evento"
(scheduler) del "como suena" (render) para los motores que lo tenian
inline (granular, liquid, exotic). El patron ya existe en droplet.py
(`_roll_schedule` -> `RollSchedule` consumido por `_render_contact_bus`);
aqui se generaliza a un tipo comun, `EventSchedule`, que las fases
posteriores usaran para compartir un mismo tren de eventos entre varios
cuerpos que suenan.

GATE DURO de esta fase: cada extraccion debe mantener bit-identidad de
las salidas publicas (ver scripts/34_listen_blends.py --check). Por eso
`amps`/`vels`/`dur_hint` se guardan en float64 aunque el docstring del
dataclass hable de "float32": los renders originales operaban con floats
Python (precision doble) en la multiplicacion `amp * evt[...]`, y el
NEP 50 de numpy trata un escalar float64 "fuerte" distinto de un float
Python "debil" al mezclarse con arrays float32 -- para reproducir el
mismo patron bit a bit el consumidor debe volver a un `float()` (Python
puro) antes de multiplicar, igual que hace `_render_contact_bus` con
`sched.amps`. Ver impossible_mix/physics/droplet.py:_render_contact_bus.

Nota sobre `starts` "ordenado": el docstring original de este dataclass
(ver brief) asume starts ordenado. Para granular y splash el ORDEN DE
GENERACION (nearest-anchor+jitter / exponential raw_times) no es
monotono en el tiempo, y los eventos SE SOLAPAN (la duracion del grano
supera el espaciado nominal) -- reordenar por tiempo cambiaria el orden
de acumulacion float32 (`+=`) y romperia el checksum dorado. Por eso
`schedule_from_grains`/`_splash_schedule` devuelven starts en orden de
GENERACION, no de tiempo, y el render los recorre en ese mismo orden
(indice a indice). pour/crackles SI son monotonos por construccion (el
`while t < n` avanza `t` siempre hacia delante) y no tienen este problema.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from impossible_mix.physics.analysis import band_envelopes, detect_onsets, smooth_env


@dataclass
class EventSchedule:
    """Agenda comun de eventos, independiente del motor que la genero."""
    starts: np.ndarray        # int64 samples (orden de GENERACION; ver modulo doc)
    amps: np.ndarray          # float64 (precision completa; ver modulo doc)
    vels: np.ndarray          # velocity factor por evento (1.0 si no aplica)
    dur_hint: np.ndarray      # duracion estimada por evento (s)
    radii_mm: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def schedule_from_roll(sched, p) -> EventSchedule:
    """Adapta un `droplet.RollSchedule` (motor v3/v4 de rolling droplet) al
    `EventSchedule` comun. radii = radius_scales * droplet_radius_mm.

    `p` puede venir sin los campos derivados completados (chirp/decay); se
    completan aqui via `_derive_params` (igual que hacen los synth_* de
    droplet.py) solo para estimar `dur_hint` -- no afecta al render real
    de rolling droplet, que sigue viviendo en droplet.py sin tocar.
    """
    from impossible_mix.physics.droplet import _derive_params

    dp = _derive_params(p)
    n = len(sched.starts)
    dur_s = (float(dp.chirp_duration_ms or 0.0) + float(dp.decay_ms or 0.0)) / 1000.0
    radii = (sched.radius_scales * dp.droplet_radius_mm
             if len(sched.radius_scales) else np.zeros(0))
    return EventSchedule(
        starts=sched.starts.astype(np.int64),
        amps=np.asarray(sched.amps, dtype=np.float64),
        vels=np.asarray(sched.vels, dtype=np.float64),
        dur_hint=np.full(n, dur_s, dtype=np.float64),
        radii_mm=np.asarray(radii, dtype=np.float64),
        meta={"asp_idx": sched.asp_idx, "source": "roll"},
    )


def schedule_from_grains(sched: EventSchedule) -> EventSchedule:
    """`granular._grain_schedule` ya construye el `EventSchedule` comun
    directamente (con un import diferido de este modulo para evitar el
    ciclo blend<->granular vía exotic, que importa granular). Este
    adaptador es la identidad, y existe solo por simetria de nombres con
    `schedule_from_drips`/`schedule_from_splash`/`schedule_from_crackles`.
    """
    return sched


def schedule_from_splash(sched, p) -> EventSchedule:
    """Adapta un `liquid.SplashSchedule` (cascada de burbujas de synth_splash)
    al `EventSchedule` comun. `dur_hint` se estima con la fisica de Minnaert
    de droplet.py (chirp+decay) para el radio de cada burbuja -- no se usa
    en el render real de synth_splash, que sigue viviendo en liquid.py.
    """
    from impossible_mix.physics.droplet import DropletParams, _derive_params

    n = len(sched.offsets)
    dur_hint = np.empty(n, dtype=np.float64)
    for i in range(n):
        dp = _derive_params(DropletParams(droplet_radius_mm=float(sched.radii_mm[i]),
                                           viscosity=p.viscosity))
        dur_hint[i] = (float(dp.chirp_duration_ms or 0.0) + float(dp.decay_ms or 0.0)) / 1000.0

    return EventSchedule(
        starts=sched.offsets.astype(np.int64),
        amps=np.asarray(sched.amps, dtype=np.float64),
        vels=np.asarray(sched.vels, dtype=np.float64),
        dur_hint=dur_hint,
        radii_mm=np.asarray(sched.radii_mm, dtype=np.float64),
        meta={"seeds": sched.seeds, "source": "splash"},
    )


def schedule_from_drips(sched, p) -> EventSchedule:
    """Adapta un `liquid.DripSchedule` (loop de synth_pour) al
    `EventSchedule` comun. `vels` no existe en el pour original (no hay
    velocity_factor por gota); se rellena a 1.0. `dur_hint` estimado igual
    que en `schedule_from_splash`.
    """
    from impossible_mix.physics.droplet import DropletParams, _derive_params

    n = len(sched.starts)
    dur_hint = np.empty(n, dtype=np.float64)
    for i in range(n):
        dp = _derive_params(DropletParams(droplet_radius_mm=float(sched.radii_mm[i]),
                                           viscosity=p.viscosity))
        dur_hint[i] = (float(dp.chirp_duration_ms or 0.0) + float(dp.decay_ms or 0.0)) / 1000.0

    return EventSchedule(
        starts=sched.starts.astype(np.int64),
        amps=np.asarray(sched.amps, dtype=np.float64),
        vels=np.ones(n, dtype=np.float64),
        dur_hint=dur_hint,
        radii_mm=np.asarray(sched.radii_mm, dtype=np.float64),
        meta={"seeds": sched.seeds, "source": "pour"},
    )


def schedule_from_crackles(events, sr: int = 44_100) -> EventSchedule:
    """Adapta una lista de `exotic.CracklePop` (pops de synth_fire) al
    `EventSchedule` comun. `sr` solo se usa para convertir `pop_n`
    (muestras) a `dur_hint` (segundos); por defecto usa la SR estandar del
    proyecto ya que `CracklePop` no la guarda. `vels` no aplica (siempre
    1.0). `meta` conserva fc1/fc2/t60 por si una fase posterior quiere
    reconstruir el timbre exacto de cada pop.
    """
    n = len(events)
    if n == 0:
        return EventSchedule(
            starts=np.zeros(0, dtype=np.int64),
            amps=np.zeros(0, dtype=np.float64),
            vels=np.zeros(0, dtype=np.float64),
            dur_hint=np.zeros(0, dtype=np.float64),
            meta={"source": "crackles"},
        )
    starts = np.asarray([e.idx for e in events], dtype=np.int64)
    amps = np.asarray([e.amp for e in events], dtype=np.float64)
    dur_hint = np.asarray([e.pop_n / sr for e in events], dtype=np.float64)
    return EventSchedule(
        starts=starts,
        amps=amps,
        vels=np.ones(n, dtype=np.float64),
        dur_hint=dur_hint,
        meta={
            "source": "crackles",
            "fc1": np.asarray([e.fc1 for e in events], dtype=np.float64),
            "fc2": np.asarray([e.fc2 for e in events], dtype=np.float64),
            "t60": np.asarray([e.t60 for e in events], dtype=np.float64),
        },
    )


def schedule_from_audio(w: np.ndarray, sr: int, expected_rate: float) -> EventSchedule:
    """Fallback universal: detecta eventos en audio ya renderizado via
    onset-picking (analysis.detect_onsets) y usa la envolvente suavizada
    (analysis.smooth_env) en los picos como amplitud, normalizada a 1.0.

    No forma parte de ningun checksum dorado (es una funcion nueva, no una
    extraccion), asi que no tiene requisito de bit-identidad.
    """
    onsets = detect_onsets(w, sr, expected_rate)
    if len(onsets) == 0:
        return EventSchedule(
            starts=np.zeros(0, dtype=np.int64),
            amps=np.zeros(0, dtype=np.float64),
            vels=np.zeros(0, dtype=np.float64),
            dur_hint=np.zeros(0, dtype=np.float64),
            meta={"source": "audio", "expected_rate": expected_rate},
        )
    env = smooth_env(w, sr, win_ms=5.0)
    env_idx = np.clip(onsets, 0, len(env) - 1)
    amps_at_peaks = env[env_idx]
    peak_max = float(amps_at_peaks.max())
    amps_norm = amps_at_peaks / (peak_max + 1e-12)

    if len(onsets) > 1:
        gaps_s = np.diff(onsets).astype(np.float64) / sr
        last_gap = gaps_s[-1] if len(gaps_s) else 1.0 / max(expected_rate, 1e-6)
        dur_hint = np.append(gaps_s, last_gap)
    else:
        dur_hint = np.array([1.0 / max(expected_rate, 1e-6)], dtype=np.float64)

    return EventSchedule(
        starts=onsets.astype(np.int64),
        amps=amps_norm.astype(np.float64),
        vels=np.ones(len(onsets), dtype=np.float64),
        dur_hint=dur_hint.astype(np.float64),
        meta={"source": "audio", "expected_rate": expected_rate},
    )


# ====================================================================
# Fase 2: mecanismos de fusion -- cross-drive, excitacion compartida y
# vocoder fisico. Estos NO tienen requisito de bit-identidad (no son
# extracciones de codigo existente); consumen `EventSchedule` de la F1
# y los catalogos de material de droplet.py/modal.py sin tocarlos (salvo
# el kwarg `q_traj` quirurgico en `_driven_resonator`, ver droplet.py).
# ====================================================================

def cross_drive(source: np.ndarray, sr: int, *, out_range: tuple[float, float],
                smoothing_hz: float = 8.0, curve: float = 1.0,
                norm: str = "p98") -> np.ndarray:
    """Convierte la envolvente de `source` en una trayectoria de control
    para otro cuerpo (p.ej. la tasa de un tren de eventos o la frecuencia
    de un resonador driven). Esto es lo que permite que dos motores fisicos
    distintos "se escuchen" el uno al otro sin compartir excitacion.

    Pipeline: envolvente hilbert suavizada (`analysis.smooth_env`) -> paso
    bajo butter(2, smoothing_hz) (elimina el rizado rapido, deja solo el
    gesto macro) -> normalizacion robusta a [0,1] por percentiles 2-98
    (`norm="p98"`, evita que un unico pico dispare el rango; `norm="minmax"`
    usa el rango exacto) con clip -> curva de respuesta `**curve` (curve>1
    comprime el centro, <1 lo expande) -> mapeo afin a `out_range`.
    """
    env = smooth_env(source, sr, win_ms=5.0)
    sos = signal.butter(2, smoothing_hz, btype="low", fs=sr, output="sos")
    env_s = signal.sosfiltfilt(sos, env.astype(np.float64))

    if norm == "minmax":
        lo_p, hi_p = float(env_s.min()), float(env_s.max())
    else:
        lo_p, hi_p = (float(v) for v in np.percentile(env_s, [2, 98]))
    span = max(hi_p - lo_p, 1e-12)
    norm01 = np.clip((env_s - lo_p) / span, 0.0, 1.0)
    curved = norm01 ** max(curve, 1e-6)

    lo_out, hi_out = out_range
    traj = lo_out + curved * (hi_out - lo_out)
    return traj.astype(np.float32)


def schedule_from_rate(rate_traj: np.ndarray, sr: int, seed: int, *,
                       amp_traj: np.ndarray | None = None,
                       jitter_ms: float = 3.0,
                       radius_traj: np.ndarray | None = None) -> EventSchedule:
    """Tren de eventos inhomogeneo dirigido por una trayectoria de tasa
    (Hz por sample, p.ej. la salida de `cross_drive`). Mismo truco que el
    acumulador de fase `theta` de `droplet._roll_schedule`: se integra la
    tasa a una fase continua y se dispara un evento en cada cruce de
    entero (proceso de Poisson no homogeneo determinista + jitter).
    """
    rng = np.random.default_rng(seed)
    n = len(rate_traj)
    if n == 0:
        return EventSchedule(
            starts=np.zeros(0, dtype=np.int64), amps=np.zeros(0, dtype=np.float64),
            vels=np.zeros(0, dtype=np.float64), dur_hint=np.zeros(0, dtype=np.float64),
            meta={"source": "rate"},
        )

    phi = np.cumsum(np.asarray(rate_traj, dtype=np.float64)) / sr
    n_events = int(np.floor(phi[-1])) if phi[-1] >= 1.0 else 0
    if n_events == 0:
        return EventSchedule(
            starts=np.zeros(0, dtype=np.int64), amps=np.zeros(0, dtype=np.float64),
            vels=np.zeros(0, dtype=np.float64), dur_hint=np.zeros(0, dtype=np.float64),
            meta={"source": "rate"},
        )

    targets = np.arange(1, n_events + 1, dtype=np.float64)
    idx = np.searchsorted(phi, targets).astype(np.int64)
    idx = np.clip(idx, 0, n - 1)
    jitter = rng.normal(0.0, max(jitter_ms, 0.0) * 1e-3 * sr, size=n_events)
    idx = np.clip(idx + np.round(jitter).astype(np.int64), 0, n - 1)
    order = np.argsort(idx, kind="stable")
    starts = idx[order]

    amps = (np.asarray(amp_traj, dtype=np.float64)[starts]
            if amp_traj is not None else np.ones(n_events, dtype=np.float64))
    vels = np.ones(n_events, dtype=np.float64)
    local_rate = np.clip(np.asarray(rate_traj, dtype=np.float64)[starts], 1e-6, None)
    dur_hint = 1.0 / local_rate
    radii = (np.asarray(radius_traj, dtype=np.float64)[starts]
             if radius_traj is not None else None)

    return EventSchedule(
        starts=starts, amps=amps, vels=vels, dur_hint=dur_hint,
        radii_mm=radii, meta={"source": "rate", "seed": seed},
    )


@dataclass
class BodySpec:
    """Describe UN cuerpo resonante que puede ser excitado por un tren de
    eventos compartido (`shared_excitation`) o por una excitacion moldeada
    a partir de otro audio (`physical_vocoder`).
    """
    kind: str                          # "surface" | "modal" | "minnaert"
    ref: str | object | float          # nombre en SURFACE_PROFILES / modal.PROFILES, o radius_mm para minnaert
    gain: float = 1.0
    q: float = 12.0                    # minnaert
    f_traj: np.ndarray | None = None   # minnaert con radio (frecuencia) variable
    t60_scale: float = 1.0


def excitation_from_schedule(sched: EventSchedule, sr: int, n_total: int, seed: int,
                             click_ms: float = 1.0,
                             color_hz: tuple[float, float] | None = None) -> np.ndarray:
    """Excitacion comun (ruido) para `shared_excitation`: una ráfaga corta
    de ruido por evento de `sched`, con amplitud `amp*vel`. `click_ms` es
    el SUELO de la duracion (un impacto nunca excita menos que eso); el
    `dur_hint` de cada evento la ALARGA cuando el evento es mas "lento"
    (contactos espaciados, p.ej. rodadura a baja velocidad piden una
    excitacion algo mas ancha que un grano denso), con un tope duro de
    10 ms (mas alla de eso ya no excita un impacto, suena a rafaga
    sostenida). Rafagas solapadas SUMAN (`+=`): eso es justo lo que se
    necesita para que dos cuerpos alimentados de la misma excitacion
    conserven la microestructura temporal compartida.
    """
    rng = np.random.default_rng(seed)
    exc = np.zeros(n_total, dtype=np.float32)
    n_evt = len(sched.starts)
    have_dur = len(sched.dur_hint) == n_evt and n_evt > 0
    for i in range(n_evt):
        start = int(sched.starts[i])
        if start < 0 or start >= n_total:
            continue
        dur_ms = click_ms
        if have_dur:
            dh_ms = float(sched.dur_hint[i]) * 1000.0
            if dh_ms > dur_ms:
                dur_ms = dh_ms
        dur_ms = min(dur_ms, 10.0)
        burst_n = max(2, int(dur_ms / 1000.0 * sr))
        end = min(n_total, start + burst_n)
        seg_n = end - start
        if seg_n <= 0:
            continue
        amp = float(sched.amps[i]) if len(sched.amps) == n_evt else 1.0
        vel = float(sched.vels[i]) if len(sched.vels) == n_evt else 1.0
        burst = rng.standard_normal(burst_n).astype(np.float32)
        env = np.exp(-3.0 * np.arange(burst_n) / burst_n).astype(np.float32)
        exc[start:end] += (amp * vel) * (burst * env)[:seg_n]

    if color_hz is not None:
        lo, hi = color_hz
        hi = min(hi, sr / 2 - 200)
        if lo < hi and lo > 0:
            sos = signal.butter(3, [lo, hi], btype="band", fs=sr, output="sos")
            exc = signal.sosfiltfilt(sos, exc).astype(np.float32)
    return exc


def _mode_bank(exc: np.ndarray, freqs, gains, t60_s: float, sr: int, *,
              inharmonicity: float = 0.0, seed: int = 0) -> np.ndarray:
    """Banco de resonadores biquad ringing (patron `[1,0,-1] / [1,-2r*cos(th),r^2]`
    de `droplet._wet_contact_event`, replicado aqui para correrlo UNA sola
    vez sobre la excitacion entera en vez de por-evento: eso da colas
    resonantes reales que se solapan entre eventos consecutivos, en vez
    de colas truncadas evento a evento.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(len(exc), dtype=np.float32)
    r = float(np.exp(-6.91 / max(t60_s * sr, 1e-3)))
    for fc, mg in zip(freqs, gains):
        fc = float(fc)
        if fc <= 0 or fc >= sr / 2 - 100:
            continue
        fc_j = fc * (1.0 + inharmonicity * rng.uniform(-1.0, 1.0)) if inharmonicity else fc
        th = 2 * np.pi * fc_j / sr
        mode = signal.lfilter([1.0, 0.0, -1.0],
                              [1.0, -2 * r * np.cos(th), r * r],
                              exc).astype(np.float32)
        out += float(mg) * mode
    return out


def render_body(exc: np.ndarray, body: BodySpec, sr: int, seed: int) -> np.ndarray:
    """Renderiza UN cuerpo (`BodySpec`) sobre una excitacion ya generada
    (por `excitation_from_schedule` o `physical_vocoder`). NO aplica
    `body.gain` -- lo hace el llamante (`shared_excitation`), porque el
    mismo `BodySpec`/render puede pesar distinto en mezclas distintas.
    """
    if body.kind == "surface":
        from impossible_mix.physics.droplet import SURFACE_PROFILES

        surf = SURFACE_PROFILES[body.ref]
        t60_s = surf.t60_ms / 1000.0 * body.t60_scale
        return _mode_bank(exc, surf.modes_hz, surf.mode_gains, t60_s, sr,
                          inharmonicity=surf.inharmonicity, seed=seed)

    if body.kind == "modal":
        from impossible_mix.physics import modal

        profile = modal.PROFILES[body.ref]
        freqs = modal.modal_frequencies(profile)
        gains = modal._gain_curve(profile.n_modes, profile.spectrum_shape)
        gains = gains / (gains.sum() + 1e-9)
        t60_s = profile.damping_ms / 1000.0 * body.t60_scale
        return _mode_bank(exc, freqs, gains, t60_s, sr, seed=seed)

    if body.kind == "minnaert":
        from impossible_mix.physics.droplet import _bubble_freq_from_radius, _driven_resonator

        if body.f_traj is not None:
            f_traj = np.asarray(body.f_traj, dtype=np.float64)
        else:
            f0 = _bubble_freq_from_radius(float(body.ref))
            f_traj = np.full(len(exc), f0, dtype=np.float64)
        return _driven_resonator(exc.astype(np.float32), f_traj, body.q, sr)

    raise ValueError(f"BodySpec.kind desconocido: {body.kind!r}")


def shared_excitation(sched: EventSchedule, bodies: list[BodySpec], sr: int,
                      n_total: int, seed: int, return_stems: bool = False):
    """Un unico tren de eventos (`sched`) excita varios cuerpos a la vez:
    la firma de la excitacion compartida (microtiming, amplitudes) queda
    correlada en todos los stems, lo que es justo lo que mide `fusion_index`
    en analysis.py. Cada cuerpo usa `seed + indice` para su banco de modos
    (jitter de inarmonicidad propio) mantiendo determinismo total.
    """
    exc = excitation_from_schedule(sched, sr, n_total, seed)
    stems = [render_body(exc, body, sr, seed + i) for i, body in enumerate(bodies)]
    mix = np.zeros(n_total, dtype=np.float32)
    for body, stem in zip(bodies, stems):
        mix += np.float32(body.gain) * stem
    if return_stems:
        return mix, exc, stems
    return mix


def physical_vocoder(articulator: np.ndarray, sr: int, body: BodySpec,
                     n_bands: int = 12, lo: float = 80.0, hi: float = 8000.0,
                     env_smooth_ms: float = 8.0, depth: float = 1.0,
                     seed: int = 0) -> np.ndarray:
    """Vocoder FISICO: `articulator` module la EXCITACION de `body`, no su
    render.

    Por que: las colas resonantes de un cuerpo fisico deben ringear segun
    SU propio t60 despues de cada gesto del articulador -- si en vez de eso
    se modulase el render (`render_body(...) * envolvente(articulator)`),
    cada cola quedaria cortada/reabierta al ritmo del articulador, dando
    el tremolo caracteristico de un vocoder electronico de 1970 en vez de
    un objeto fisico real reaccionando a un estimulo. Modulando la
    excitacion, el resonador de `body` sigue libre para decaer con su
    propia fisica una vez que la excitacion cesa.

    Implementacion: `analysis.band_envelopes(articulator)` da envolventes
    por banda a FULL RATE; se sintetiza ruido blanco (seed) filtrado en
    esas mismas bandas y se pesa cada banda por su envolvente (`depth=1`)
    o por una mezcla con la media de la banda (`depth<1`, difumina la
    articulacion -- en `depth=0` la excitacion pierde toda huella temporal
    del articulador). Esa excitacion moldeada entra a `render_body`.
    """
    n = len(articulator)
    envs, _centers = band_envelopes(articulator, sr, n_bands=n_bands, lo=lo, hi=hi,
                                    win_ms=env_smooth_ms)
    hi_safe = min(hi, sr / 2 - 200)
    edges = np.geomspace(lo, hi_safe, n_bands + 1)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float64)
    exc = np.zeros(n, dtype=np.float64)
    depth = float(np.clip(depth, 0.0, 1.0))
    for k in range(n_bands):
        f_lo, f_hi = float(edges[k]), float(edges[k + 1])
        if f_lo >= f_hi:
            continue
        sos = signal.butter(4, [f_lo, f_hi], btype="band", fs=sr, output="sos")
        band_noise = signal.sosfiltfilt(sos, noise)
        env_k = envs[k].astype(np.float64)
        if depth < 1.0:
            env_k = (1.0 - depth) * float(env_k.mean()) + depth * env_k
        exc += band_noise * env_k

    out = render_body(exc.astype(np.float32), body, sr, seed)
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


# ====================================================================
# (c) Morphing de fisica: interpolar las LEYES, no las salidas
# ====================================================================
@dataclass
class HandoffSpec:
    """Traspaso rodadura -> goteo al final del clip (la canica se derrite).

    A partir de t_norm la excitacion de rodadura decae (rate -> 0) mientras
    un tren creciente de gotas inyecta su excitacion EN EL MISMO resonador
    de agua (mismo estado zi): el cuerpo liquido nunca se corta, solo
    cambia quien lo excita. Cero crossfade de renders.
    """
    t_norm: float = 0.7
    rate_end_hz: float = 18.0
    drip_gain: float = 0.8


def synth_rolling_droplet_morphed(p, morph: dict[str, np.ndarray],
                                  sr: int = 44_100,
                                  surface_pair: tuple[str, str] = ("metal", "water"),
                                  handoff: HandoffSpec | None = None,
                                  return_debug: bool = False):
    """Gota rodante con TRAYECTORIAS de parametros fisicos (morphing).

    morph: claves opcionales {"radius_mm", "viscosity", "roll_velocity_hz",
    "hardness_x"} con arrays a cualquier resolucion (se upsamplean con
    interp lineal). hardness_x en [0,1] cruza equal-power entre los DOS
    bancos de material de surface_pair — ambos oyen la MISMA excitacion:
    no son dos capas, son dos coloraciones del mismo contacto.

    UN solo render continuo: theta = cumsum(rate) mantiene la fase de
    revolucion; el resonador Minnaert recibe f(t)=3.26/r(t) y Q(t) de la
    viscosidad; el handoff inyecta las gotas en su mismo estado.
    """
    from dataclasses import replace as _dc_replace

    from impossible_mix.physics.droplet import (
        SURFACE_PROFILES, _derive_params, _driven_resonator, _get_surface,
        _roll_schedule, _surface_profile_wave, synth_drip_event,
    )

    p = _derive_params(_dc_replace(p))
    n = int(p.duration_s * sr)
    rng = np.random.default_rng(p.seed)

    def _up(key: str, default: float) -> np.ndarray:
        arr = morph.get(key)
        if arr is None:
            return np.full(n, float(default))
        arr = np.asarray(arr, dtype=np.float64)
        return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr)

    r_traj = np.maximum(_up("radius_mm", p.droplet_radius_mm), 0.2)
    visc_traj = np.clip(_up("viscosity", p.viscosity), 0.0, 1.0)
    rate_traj = np.maximum(
        _up("roll_velocity_hz", p.roll_velocity_hz), 0.0) * max(p.contact_density_mul, 0.1)
    x_traj = np.clip(_up("hardness_x", 0.0), 0.0, 1.0)

    surf_a = SURFACE_PROFILES[surface_pair[0]]
    surf_b = SURFACE_PROFILES[surface_pair[1]]

    # Agenda + perfil de vuelta con tasa variable (fase continua via cumsum)
    sched = _roll_schedule(p, sr, n, rng, rate_traj=rate_traj)
    e = _surface_profile_wave(p, sched, sr, n, rng)
    rate_ref = max(float(rate_traj.max()), 1e-6)
    e = (e * np.sqrt(np.clip(rate_traj / rate_ref, 0.0, 1.0))).astype(np.float32)

    # Excitacion: MISMO ruido por dos bandas de material, crossfade equal-power
    noise = rng.standard_normal(n).astype(np.float32)

    def _bp(surf) -> np.ndarray:
        lo, hi = surf.click_color_hz
        hi = min(hi, sr / 2 - 200)
        sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        return signal.sosfiltfilt(sos, noise).astype(np.float32)

    xa = np.cos(0.5 * np.pi * x_traj).astype(np.float32)
    xb = np.sin(0.5 * np.pi * x_traj).astype(np.float32)
    excited = (_bp(surf_a) * xa + _bp(surf_b) * xb) * e

    # Resonador de agua CONTINUO: f de r(t), Q de viscosidad(t)
    sos_lp = signal.butter(2, 25.0, btype="low", fs=sr, output="sos")
    e_s = signal.sosfiltfilt(sos_lp, e.astype(np.float64)).astype(np.float32)
    fm_depth = 0.04 + 0.04 * p.path_roughness
    f_traj = (3.26 / (r_traj * 1e-3)) * (1.0 + fm_depth * (e_s - float(e_s.mean())))
    f_traj = np.clip(f_traj, 40.0, sr / 2 - 500).astype(np.float32)
    q_traj = (14.0 - 10.0 * visc_traj).astype(np.float32)

    water_exc = excited.copy()
    drips_detail = np.zeros(n, dtype=np.float32)
    drip_starts = np.zeros(0, dtype=np.int64)
    if handoff is not None:
        i0 = int(np.clip(handoff.t_norm, 0.0, 0.95) * n)
        drip_rate = np.zeros(n)
        drip_rate[i0:] = np.linspace(0.0, handoff.rate_end_hz, n - i0)
        dsched = schedule_from_rate(drip_rate, sr, p.seed + 7,
                                    radius_traj=r_traj)
        if len(dsched.starts):
            drip_starts = dsched.starts
            water_exc += 1.2 * excitation_from_schedule(
                dsched, sr, n, p.seed + 11, click_ms=1.2, color_hz=(300, 4000))
            for i, s in enumerate(dsched.starts):
                pd = _dc_replace(
                    p, droplet_radius_mm=float(dsched.radii_mm[i]),
                    bubble_freq_start_hz=None, bubble_freq_end_hz=None,
                    chirp_duration_ms=None, decay_ms=None,
                    bounce_amount=0.0)
                evt = synth_drip_event(pd, sr, velocity_factor=1.0,
                                       seed_override=p.seed + 1000 + int(s))
                end = min(n, int(s) + len(evt))
                drips_detail[s:end] += (handoff.drip_gain
                                        * float(dsched.amps[i])
                                        * evt[: end - int(s)])

    water = _driven_resonator(water_exc, f_traj, 12.0, sr, q_traj=q_traj)
    water = water * (0.9 * (1.0 - 0.6 * visc_traj)
                     * p.body_resonance_mix).astype(np.float32)

    # Bancos de material A y B sobre la MISMA excitacion, cruzados x(t)
    def _bank(surf) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        t60_s = surf.t60_ms * 0.5 / 1000.0
        for fc, mg in zip(surf.modes_hz, surf.mode_gains):
            if fc <= 0 or fc >= sr / 2 - 100:
                continue
            mg_eff = mg * min(1.0, (4000.0 / fc) ** 2) if fc > 4000 else mg
            q_m = float(np.clip(0.455 * fc * t60_s, 2.0, 25.0))
            bw = fc / q_m
            f_lo, f_hi = max(50.0, fc - bw / 2), min(sr / 2 - 100, fc + bw / 2)
            if f_lo >= f_hi:
                continue
            try:
                sos_m = signal.butter(2, [f_lo, f_hi], btype="band", fs=sr,
                                      output="sos")
                out += mg_eff * signal.sosfiltfilt(sos_m, excited).astype(np.float32)
            except ValueError:
                continue
        return out

    material = (_bank(surf_a) * xa + _bank(surf_b) * xb) * (
        p.surface_ring_mix * 0.6)

    s = float(np.clip(p.smoothness, 0.0, 1.0))
    out = (0.30 * (1.0 - 0.8 * s)) * excited + water + material + drips_detail

    # Rumor grave gated por e(t) (banda interpolada por el punto medio del morph)
    if p.continuous_layer_mix > 0.01:
        x_mid = float(x_traj.mean())
        cl_lo = surf_a.click_color_hz[0] * (1 - x_mid) + surf_b.click_color_hz[0] * x_mid
        cl_hi = surf_a.click_color_hz[1] * (1 - x_mid) + surf_b.click_color_hz[1] * x_mid
        lo, hi = max(120.0, cl_lo * 0.4), min(cl_hi * 0.5, 3000.0)
        if lo >= hi:
            lo, hi = 120.0, 1200.0
        sos_r = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        rumble_noise = signal.sosfilt(
            sos_r, rng.standard_normal(n).astype(np.float32)).astype(np.float32)
        gate = e / (float(e.max()) + 1e-9)
        out = out + (0.35 * p.continuous_layer_mix
                     * (1.0 - 0.5 * visc_traj).astype(np.float32)) * gate * rumble_noise

    fade_n = min(int(0.02 * sr), n // 8)
    if fade_n > 4:
        fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_n))).astype(np.float32)
        out[:fade_n] *= fade
        out[-fade_n:] *= fade[::-1]
    peak = float(np.max(np.abs(out)) + 1e-9)
    scale = (0.95 / peak) if peak > 0.95 else 1.0
    out = (out * scale).astype(np.float32)
    if return_debug:
        return out, dict(
            water=(water * scale).astype(np.float32),
            material=(material * scale).astype(np.float32),
            drips=(drips_detail * scale).astype(np.float32),
            e=e, f_traj=f_traj, sched=sched, drip_starts=drip_starts,
        )
    return out


# ====================================================================
# v8: trasplante de ley por modo/evento (hibridos de identidad)
# ====================================================================
# Un banco modal donde cada modo, al ser excitado, NO mantiene su
# frecuencia: sigue la ley de OTRO fenomeno (p.ej. el glide ascendente de
# Minnaert del drip) y su amortiguamiento puede seguir la ley de van den
# Doel evaluada en la frecuencia INSTANTANEA (el decay se acelera mientras
# el modo sube) -- imposible en un biquad de Q fijo, trivial en sintesis
# aditiva por evento. Esta es la pieza que convierte "dos cuerpos
# acoplados" en UN objeto con fisica interna contradictoria.

_LN1000 = 6.907755


@dataclass
class ChirpLaw:
    """Ley de trayectoria de frecuencia trasplantada a cada modo."""
    start_ratio: float = 0.45   # f inicial relativa al modo (ley del drip)
    end_ratio: float = 1.0      # 1.0 aterriza EN el modo (conserva material);
                                # 1.6 = ley Minnaert literal (detuning deliberado)
    glide_ms: float = 30.0      # duracion del glide en f_ref
    f_ref_hz: float = 1800.0
    per_mode_scale: str = "sqrt"  # "sqrt": los modos altos chirpean mas rapido
    stagger_ms: float = 3.0     # retardo de onset creciente por modo


@dataclass
class DampingLaw:
    """Ley de amortiguamiento trasplantada.

    "vdd": van den Doel puro d(f) (los modos agudos se evaporan);
    "vdd_shape": conserva el t60 del material en f_ref pero con la FORMA
    f-dependiente de vdd (identidad material + comportamiento agua);
    "material": t60 fijo clasico.
    """
    kind: str = "vdd_shape"
    t60_ref_ms: float | None = None
    f_ref_hz: float = 1800.0
    scale: float = 1.0


def vdd_damping_rate(f_hz: np.ndarray) -> np.ndarray:
    """d(f) = 0.043 f_kHz + 0.0014 f_kHz^1.5 [ms^-1] (van den Doel 2005),
    identica a droplet._bubble_t60_ms pero vectorizada sobre trayectorias."""
    f_khz = np.asarray(f_hz, dtype=np.float64) / 1000.0
    return 0.043 * f_khz + 0.0014 * f_khz ** 1.5


def _damping_fn(damping: DampingLaw):
    if damping.kind == "vdd":
        return lambda f: vdd_damping_rate(f) / max(damping.scale, 1e-6)
    if damping.kind == "vdd_shape":
        t60 = damping.t60_ref_ms or 300.0
        ref = float(vdd_damping_rate(np.array([damping.f_ref_hz]))[0])
        k = (_LN1000 / t60) / max(ref, 1e-9)
        return lambda f: vdd_damping_rate(f) * k / max(damping.scale, 1e-6)
    t60 = damping.t60_ref_ms or 300.0
    return lambda f: np.full_like(np.asarray(f, dtype=np.float64),
                                  (_LN1000 / t60) / max(damping.scale, 1e-6))


def chirping_modal_bank(sched: EventSchedule, modes_hz, mode_gains, sr: int,
                        n_total: int, *, chirp: ChirpLaw, damping: DampingLaw,
                        inharmonicity: float = 0.0, seed: int = 0,
                        attack_ms: float = 0.8, max_event_s: float = 4.0,
                        exc_click: tuple[float, float] | None = None) -> np.ndarray:
    """Banco modal con ley de frecuencia y amortiguamiento TRASPLANTADAS.

    Sintesis aditiva por evento: cada (evento, modo) es una sinusoide con
    f(t) geometrica start->end y envolvente exp(-integral d(f(t)) dt) con
    d evaluada en la frecuencia instantanea.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(n_total, dtype=np.float32)
    d_of = _damping_fn(damping)
    n_att = max(2, int(attack_ms / 1000.0 * sr))

    for i, s in enumerate(sched.starts):
        a_i = float(sched.amps[i]) * float(sched.vels[i])
        for k, (f0, g) in enumerate(zip(modes_hz, mode_gains)):
            if f0 <= 0 or f0 >= sr / 2 - 200:
                continue
            f0k = f0 * (1.0 + inharmonicity * rng.uniform(-1, 1))
            f_start = f0k * chirp.start_ratio
            f_end = min(f0k * chirp.end_ratio, sr / 2 - 200)
            glide_ms_k = chirp.glide_ms
            if chirp.per_mode_scale == "sqrt":
                glide_ms_k = chirp.glide_ms * float(np.sqrt(chirp.f_ref_hz / f0k))
            glide_n = max(8, int(glide_ms_k / 1000.0 * sr))
            d_end = float(d_of(np.array([f_end]))[0])          # ms^-1
            tail_n = int(min(_LN1000 / max(d_end, 1e-6) / 1000.0, max_event_s) * sr)
            s_k = int(s) + int(k * chirp.stagger_ms / 1000.0 * sr)
            n_evt = min(glide_n + tail_n, n_total - s_k)
            if n_evt <= n_att:
                continue
            t_g = np.arange(glide_n) / glide_n
            f_glide = f_start * (f_end / max(f_start, 1e-6)) ** t_g
            if n_evt > glide_n:
                f_t = np.concatenate([f_glide, np.full(n_evt - glide_n, f_end)])
            else:
                f_t = f_glide[:n_evt]
            # Envolvente con amortiguamiento instantaneo integrado (d en ms^-1)
            env = np.exp(-np.cumsum(d_of(f_t)) * (1000.0 / sr) / 1000.0 * 1000.0)
            env = env.astype(np.float32)
            env[:n_att] *= (np.linspace(0, 1, n_att) ** 0.7).astype(np.float32)
            phase = 2 * np.pi * np.cumsum(f_t) / sr + 2 * np.pi * rng.random()
            out[s_k:s_k + n_evt] += (a_i * g * env
                                     * np.sin(phase).astype(np.float32))
    if exc_click is not None:
        out += 0.3 * excitation_from_schedule(sched, sr, n_total, seed + 1,
                                              color_hz=exc_click)
    return out


def f_traj_from_schedule(sched: EventSchedule, sr: int, n_total: int, *,
                         chirp: ChirpLaw, f_floor_hz: float,
                         relax_tau_ms: float = 80.0) -> np.ndarray:
    """Contorno de pitch por muestra desde un schedule conocido (sin
    pitch-tracking): en cada start, glide 0.45->end_ratio de f_M(r_i);
    entre eventos, relajacion exponencial hacia f_floor_hz."""
    f = np.full(n_total, float(f_floor_hz), dtype=np.float64)
    alpha = float(np.exp(-1.0 / (relax_tau_ms / 1000.0 * sr)))
    events = []
    for i, s in enumerate(sched.starts):
        r_mm = float(sched.radii_mm[i]) if sched.radii_mm is not None else 2.0
        f_m = 3.26 / max(r_mm * 1e-3, 1e-4)
        glide_n = max(8, int(chirp.glide_ms / 1000.0 * sr))
        events.append((int(s), f_m, glide_n))
    cur = float(f_floor_hz)
    ev_idx = 0
    active = None  # (start, f_m, glide_n)
    for n in range(n_total):
        if ev_idx < len(events) and n >= events[ev_idx][0]:
            active = events[ev_idx]
            ev_idx += 1
        if active is not None:
            s0, f_m, glide_n = active
            k = n - s0
            if k < glide_n:
                t = k / glide_n
                cur = (f_m * chirp.start_ratio
                       * (chirp.end_ratio / chirp.start_ratio) ** t)
            else:
                active = None
                cur = f_m * chirp.end_ratio
        else:
            cur = f_floor_hz + (cur - f_floor_hz) * alpha
        f[n] = cur
    return np.clip(f, 30.0, sr / 2 - 500).astype(np.float32)


# ====================================================================
# v9-M1: chimeras auditivas (Smith, Delgutte & Oxenham, Nature 2002)
# ====================================================================
def auditory_chimera(a: np.ndarray, b: np.ndarray, sr: int,
                     n_bands: int = 16, lo: float = 80.0,
                     hi: float = 8820.0) -> np.ndarray:
    """Chimera auditiva: envolvente temporal de A x estructura fina de B.

    Receta de Smith, Delgutte & Oxenham (2002, Nature 416:87-90): filterbank
    pasa-banda (1-64 bandas, 80-8820 Hz), por banda se factoriza cada sonido
    en envolvente y estructura fina via Hilbert, y se multiplica la
    envolvente de A por la estructura fina (cos de la fase) de B; la suma es
    UNA senal fusionada. El numero de bandas gobierna que padre domina la
    identidad percibida: pocas bandas -> gana la estructura fina (B);
    muchas (~16+) -> gana la envolvente (A).
    """
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    hi = min(hi, sr / 2 - 200)
    edges = np.geomspace(lo, hi, n_bands + 1)
    out = np.zeros(n, dtype=np.float64)
    for k in range(n_bands):
        f_lo, f_hi = edges[k], edges[k + 1]
        if n_bands == 1:
            f_lo, f_hi = lo, hi
        sos = signal.butter(4, [f_lo, f_hi], btype="band", fs=sr, output="sos")
        band_a = signal.sosfiltfilt(sos, a)
        band_b = signal.sosfiltfilt(sos, b)
        env_a = np.abs(signal.hilbert(band_a))
        fine_b = np.cos(np.angle(signal.hilbert(band_b)))
        out += env_a * fine_b
    peak = float(np.abs(out).max() + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


# ====================================================================
# v9-M2: alineacion de registro antes de fundir (Slaney et al., 1996)
# ====================================================================
def align_droplet_radius_to_hz(target_hz: float) -> float:
    """Radio de gota (mm) cuyo f_M de Minnaert cae EN target_hz.

    Slaney, Covell & Lassiter (ICASSP 1996): el crossfade de dos sonidos con
    pitch distinto se percibe como DOS objetos; alinear las estructuras
    armonicas ANTES de fundir colapsa la percepcion en UNO. En un motor
    parametrico la alineacion es fisica exacta: f_M = 3.26/r => r = 3.26/f.
    """
    return float(3.26 / max(target_hz, 1.0) * 1000.0)


def auditory_chimera_colored(a: np.ndarray, b: np.ndarray, sr: int,
                             n_bands: int = 16, lo: float = 80.0,
                             hi: float = 8820.0,
                             color_from: str = "b",
                             color_mix: float | None = None,
                             a_floor_db: float = -40.0) -> np.ndarray:
    """Chimera con balance espectral heredado de un padre (anti-estridencia).

    La chimera plana da a todas las bandas el peso de la envolvente de A,
    lo que puede sonar estridente cuando la materia (B) tiene agudos
    intensos. Aqui cada banda se pondera ademas por la energia RELATIVA
    natural del padre elegido (color_from="b": la materia impone tambien
    su color espectral de largo plazo; "a": lo impone la dinamica).

    color_mix (0..1, opcional): en vez de heredar el color al 100% de un
    solo padre via color_from, interpola geometricamente (en log-dominio)
    el peso por banda entre el RMS de A (color_mix=0.0) y el de B
    (color_mix=1.0): band_w[k] = exp((1-color_mix)*log(rms_a[k]+eps) +
    color_mix*log(rms_b[k]+eps)), normalizado por el maximo como siempre.
    Por defecto es None, que conserva el comportamiento previo via
    color_from (bit-identico); cuando se da, color_from se ignora. NOTA:
    el color resultante es constante en el tiempo (una unica mezcla para
    todo el clip); variarlo en el tiempo queda fuera de alcance de esta
    version (ver v12). color_mix NO se valida ni se recorta a [0,1]: un
    valor fuera de ese rango extrapola la interpolacion geometrica en vez
    de fallar (queda finito por el termino +eps, pero deja de ser una
    interpolacion real entre A y B).

    a_floor_db (solo aplica si color_mix no es None): suelo de presencia de
    A. Diagnostico medido en trueno(env) x vidrio(fina) @ 6 bandas: A
    (trueno) esta a -56, -85 y -94.6 dB en 3 de las 6 bandas -- casi
    silencio. El bug de fondo esta en `e_norm = env_a / (env_a.mean() +
    eps)`: en una banda donde A no tiene energia, env_a es ruido de punto
    flotante, y dividir por su propia media (tambien ruido) INFLA ese
    ruido a una envolvente de amplitud ~1, asi que la banda emite
    `fine_b` (la estructura de B) a peso casi pleno -- B crudo, sin
    modular por A. Con color_from (color_mix=None) esto sigue sin
    arreglar: band_w ahi depende SOLO del padre de referencia elegido, no
    de si A esta presente, y no se toca para no romper la bit-identidad.
    En la ruta color_mix, la ponderacion geometrica ya atenua algo las
    bandas donde A es debil, pero no basta para silenciar del todo un
    -94 dB; este suelo, en dB relativos al pico de RMS de A entre bandas,
    aplica una rampa continua (no un corte duro) que fuerza a 0 el peso de
    una banda cuando A cae por debajo de threshold = max(rms_a) *
    10**(a_floor_db/20): atten[k] = clip(rms_a[k]/threshold, 0, 1),
    multiplicado sobre band_w tras normalizar. Se deja como parametro (no
    una constante fija) porque el barrido de parametros posterior lo va a
    explorar.

    HONESTIDAD sobre el extremo color_mix=1.0 ("100% color de B" segun la
    formula de arriba): el suelo esta activo SIEMPRE que color_mix no es
    None, tambien en ese extremo, porque atten depende solo de rms_a (no
    de color_mix). Eso rompe la pureza del extremo -- la salida en
    color_mix=1.0 NO es identica a auditory_chimera_colored(...,
    color_from="a") ni a una version sin suelo: en pruebas con trueno(A) x
    vidrio(B), n_bands=16 (el valor por defecto), la RMS de salida cambia
    del orden de 4 dB frente a tener el suelo desactivado (a_floor_db muy
    negativo). Ademas, tras multiplicar por atten, band_w YA NO queda
    normalizado a maximo 1 (en ese mismo caso el maximo cae a ~0.09-0.13):
    la frase "normalizado por el maximo como siempre" de mas arriba solo
    describe el paso anterior al suelo, no el estado final de band_w
    cuando color_mix no es None.
    """
    n = min(len(a), len(b))
    a4 = a[:n].astype(np.float64)
    b4 = b[:n].astype(np.float64)
    hi = min(hi, sr / 2 - 200)
    edges = np.geomspace(lo, hi, n_bands + 1)
    ref = b4 if color_from == "b" else a4
    out = np.zeros(n, dtype=np.float64)
    band_w = []
    rms_a_list = []
    parts = []
    eps = 1e-12
    for k in range(n_bands):
        sos = signal.butter(4, [edges[k], edges[k + 1]], btype="band",
                            fs=sr, output="sos")
        band_a = signal.sosfiltfilt(sos, a4)
        band_b = signal.sosfiltfilt(sos, b4)
        band_ref = band_a if color_from == "a" else band_b
        env_a = np.abs(signal.hilbert(band_a))
        fine_b = np.cos(np.angle(signal.hilbert(band_b)))
        # normalizar la envolvente de A por banda y recolorear con la
        # energia del padre de referencia
        e_norm = env_a / (env_a.mean() + 1e-12)
        parts.append(e_norm * fine_b)
        if color_mix is None:
            band_w.append(float(np.sqrt((band_ref ** 2).mean())))
        else:
            rms_a = float(np.sqrt((band_a ** 2).mean()))
            rms_b = float(np.sqrt((band_b ** 2).mean()))
            rms_a_list.append(rms_a)
            band_w.append(float(np.exp((1.0 - color_mix) * np.log(rms_a + eps)
                                       + color_mix * np.log(rms_b + eps))))
    band_w = np.asarray(band_w)
    band_w = band_w / (band_w.max() + 1e-12)
    if color_mix is not None:
        rms_a_arr = np.asarray(rms_a_list)
        threshold = rms_a_arr.max() * (10.0 ** (a_floor_db / 20.0)) + eps
        atten = np.clip(rms_a_arr / threshold, 0.0, 1.0)
        band_w = band_w * atten
    for k in range(n_bands):
        out += band_w[k] * parts[k]
    peak = float(np.abs(out).max() + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
