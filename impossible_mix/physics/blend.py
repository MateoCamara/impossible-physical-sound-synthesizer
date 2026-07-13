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
