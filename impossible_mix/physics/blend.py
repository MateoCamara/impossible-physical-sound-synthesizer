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

from impossible_mix.physics.analysis import detect_onsets, smooth_env


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
