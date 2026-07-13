"""Los cuatro blends escaparate del mecanismo de fusion fisica (v7).

Cada receta acopla DOS fenomenos al nivel de la fisica (cross-drive,
excitacion compartida, morphing o vocoder), y cada una tiene su ancla de
suma ponderada equivalente (mismos fenomenos, procesos independientes)
para la comparacion A/B y para la metrica de fusion del paper.

Todas devuelven mono float32 con pico <= 0.95, deterministas por seed.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from impossible_mix.physics.blend import (
    BodySpec,
    HandoffSpec,
    cross_drive,
    excitation_from_schedule,
    render_body,
    schedule_from_rate,
    synth_rolling_droplet_morphed,
)
from impossible_mix.physics.analysis import band_envelopes
from impossible_mix.physics.droplet import DropletParams, synth_drip_event
from impossible_mix.physics.droplet_presets import get_preset
from impossible_mix.physics.exotic import _thunder_rumble, _fire_bed, _fire_crackle_events
from impossible_mix.physics.blend import schedule_from_crackles


def _norm(w: np.ndarray, peak: float = 0.95) -> np.ndarray:
    p = float(np.max(np.abs(w)) + 1e-9)
    if p > peak:
        w = w * (peak / p)
    return w.astype(np.float32)


def _render_drips(sched, sr: int, n: int, seed: int, gain: float = 1.0,
                  surface: str = "water") -> np.ndarray:
    """Tren de gotas reales (synth_drip_event) desde un EventSchedule."""
    out = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(sched.starts):
        r = float(sched.radii_mm[i]) if sched.radii_mm is not None else 2.0
        pd = DropletParams(droplet_radius_mm=r, surface_profile=surface,
                           bounce_amount=0.0, seed=seed)
        evt = synth_drip_event(pd, sr, velocity_factor=float(sched.vels[i]),
                               seed_override=seed + 1000 + int(s))
        end = min(n, int(s) + len(evt))
        out[s:end] += gain * float(sched.amps[i]) * evt[: end - int(s)]
    return out


# ====================================================================
# 1. Trueno que gotea (cross-drive + excitacion compartida)
# ====================================================================
def blend_thunder_drips(duration_s: float = 8.0, seed: int = 42,
                        glass_gain: float = 0.35, coupling: float = 1.0,
                        sr: int = 44_100) -> np.ndarray:
    """El rumble del trueno GOBIERNA la lluvia: mas trueno = gotas mas
    densas y gordas; y cada gota pinga ademas el cielo de vidrio
    (excitacion compartida). coupling=0 degrada a tasa constante."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    rumble, rumble_env = _thunder_rumble(rng, sr, n, distance=0.5, intensity=0.9)

    rate_traj = cross_drive(rumble, sr, out_range=(1.5, 16.0), smoothing_hz=6.0)
    radius_traj = cross_drive(rumble, sr, out_range=(1.2, 3.2), smoothing_hz=4.0)
    if coupling < 1.0:
        rate_traj = coupling * rate_traj + (1 - coupling) * float(rate_traj.mean())
        radius_traj = coupling * radius_traj + (1 - coupling) * float(radius_traj.mean())
    amp_traj = 0.4 + 0.6 * cross_drive(rumble, sr, out_range=(0.0, 1.0))

    sched = schedule_from_rate(rate_traj, sr, seed + 3, amp_traj=amp_traj,
                               radius_traj=radius_traj)
    drips = _render_drips(sched, sr, n, seed)
    exc = excitation_from_schedule(sched, sr, n, seed + 5, click_ms=1.0,
                                   color_hz=(800, 6000))
    glass = render_body(exc, BodySpec("surface", "glass"), sr, seed + 9)
    glass = glass / (float(np.abs(glass).max()) + 1e-9)

    return _norm(0.9 * rumble + 1.0 * drips + glass_gain * glass)


def sum_thunder_drips(duration_s: float = 8.0, seed: int = 42,
                      sr: int = 44_100) -> np.ndarray:
    """Ancla: los MISMOS fenomenos como suma ponderada de procesos
    independientes (tasa de gotas constante = media del blend)."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    rumble, _ = _thunder_rumble(rng, sr, n, distance=0.5, intensity=0.9)
    rate = np.full(n, 8.0)
    sched = schedule_from_rate(rate, sr, seed + 101,
                               radius_traj=np.full(n, 2.2))
    drips = _render_drips(sched, sr, n, seed + 200)
    return _norm(0.9 * rumble + 1.0 * drips)


# ====================================================================
# 2. Fuego de cristal liquido (excitacion compartida + cross-drive)
# ====================================================================
def blend_glass_fire(duration_s: float = 8.0, seed: int = 42,
                     water_gain: float = 0.8, glass_gain: float = 0.6,
                     sr: int = 44_100) -> np.ndarray:
    """Cada crepitar del fuego pinga SIMULTANEAMENTE un cuerpo de agua
    (Minnaert) y un cuerpo de vidrio: un solo evento, dos materiales.
    La respiracion del bed espesa el agua (viscosidad via Q del resonador)."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    bed, _ = _fire_bed(rng, sr, n, intensity=0.7)
    n_pops = int(40 * 0.5 * duration_s)
    pops = _fire_crackle_events(rng, sr, n, n_pops)
    sched = schedule_from_crackles(pops, sr)

    exc = excitation_from_schedule(sched, sr, n, seed + 5, click_ms=1.0,
                                   color_hz=(800, 6000))
    visc_traj = cross_drive(bed, sr, out_range=(0.05, 0.7), smoothing_hz=3.0)
    f_M = 3.26 / (2.0e-3)
    from impossible_mix.physics.droplet import _driven_resonator
    water = _driven_resonator(exc, np.full(n, f_M, dtype=np.float32), 12.0, sr,
                              q_traj=(14.0 - 10.0 * visc_traj).astype(np.float32))
    water = water / (float(np.abs(water).max()) + 1e-9)
    water = water * (1.0 - 0.6 * visc_traj).astype(np.float32)
    glass = render_body(exc, BodySpec("surface", "glass"), sr, seed + 9)
    glass = glass / (float(np.abs(glass).max()) + 1e-9)

    return _norm(0.3 * bed + water_gain * water + glass_gain * glass)


def sum_glass_fire(duration_s: float = 8.0, seed: int = 42,
                   sr: int = 44_100) -> np.ndarray:
    """Ancla: fuego + agua + vidrio con TRES procesos independientes
    (schedules de crackles distintos para cada cuerpo)."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    bed, _ = _fire_bed(rng, sr, n, intensity=0.7)
    n_pops = int(40 * 0.5 * duration_s)
    s1 = schedule_from_crackles(_fire_crackle_events(
        np.random.default_rng(seed + 300), sr, n, n_pops), sr)
    s2 = schedule_from_crackles(_fire_crackle_events(
        np.random.default_rng(seed + 400), sr, n, n_pops), sr)
    from impossible_mix.physics.droplet import _driven_resonator
    e1 = excitation_from_schedule(s1, sr, n, seed + 6, color_hz=(800, 6000))
    e2 = excitation_from_schedule(s2, sr, n, seed + 7, color_hz=(800, 6000))
    water = _driven_resonator(e1, np.full(n, 3.26 / 2.0e-3, dtype=np.float32),
                              12.0, sr)
    water = water / (float(np.abs(water).max()) + 1e-9)
    glass = render_body(e2, BodySpec("surface", "glass"), sr, seed + 9)
    glass = glass / (float(np.abs(glass).max()) + 1e-9)
    return _norm(0.3 * bed + 0.8 * water + 0.6 * glass)


# ====================================================================
# 3. Canica que se derrite (morphing de fisica)
# ====================================================================
_MELT_MORPH = {
    "hardness_x": np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
    "radius_mm": np.array([2.2, 2.2, 3.2, 3.5, 3.5]),
    "viscosity": np.array([0.0, 0.1, 0.35, 0.15, 0.05]),
    "roll_velocity_hz": np.array([14.0, 14.0, 10.0, 6.0, 0.0]),
}


def blend_melting_marble(duration_s: float = 10.0, seed: int = 42,
                         with_handoff: bool = True,
                         sr: int = 44_100) -> np.ndarray:
    """Canica metalica que se derrite en gota liquida y acaba goteando.
    UN render continuo: las LEYES se interpolan (radio->Minnaert, material
    A->B sobre la misma excitacion, viscosidad->Q), sin crossfade."""
    p = get_preset("water", duration_s=duration_s, seed=seed)
    handoff = HandoffSpec(t_norm=0.7, rate_end_hz=18.0) if with_handoff else None
    return synth_rolling_droplet_morphed(p, _MELT_MORPH, sr, ("metal", "water"),
                                         handoff)


def sum_melting_marble(duration_s: float = 10.0, seed: int = 42,
                       sr: int = 44_100) -> np.ndarray:
    """Ancla: metal roll y water roll INDEPENDIENTES con crossfade de
    salidas (el morphing de superficie de siempre)."""
    from impossible_mix.physics.droplet import synth_rolling_droplet
    from dataclasses import replace
    p = get_preset("water", duration_s=duration_s, seed=seed)
    w_metal = synth_rolling_droplet(replace(p, surface_profile="metal"), sr)
    w_water = synth_rolling_droplet(replace(p, surface_profile="water",
                                            seed=seed + 1), sr)
    n = min(len(w_metal), len(w_water))
    x = np.linspace(0, 1, n, dtype=np.float32)
    return _norm(w_metal[:n] * np.cos(0.5 * np.pi * x)
                 + w_water[:n] * np.sin(0.5 * np.pi * x))


# ====================================================================
# 4. El trueno habla agua (vocoder fisico)
# ====================================================================
def blend_thunder_speaks_water(duration_s: float = 8.0, seed: int = 42,
                               depth: float = 0.9, dry_drips: float = 0.0,
                               sr: int = 44_100) -> np.ndarray:
    """El cuerpo del trueno ARTICULADO por las envolventes por banda de un
    tren de gotas: el rumble conserva su identidad grave pero habla con la
    chispa del goteo. Modula la excitacion, no el render: las colas
    reverberantes del trueno decaen segun SU fisica tras cada gesto."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)

    # Articulador: goteo denso
    sched = schedule_from_rate(np.full(n, 10.0), sr, seed + 3,
                               radius_traj=np.full(n, 2.2))
    articulator = _render_drips(sched, sr, n, seed)

    # Excitacion vocoded (12 bandas del articulador sobre ruido)
    envs, _ = band_envelopes(articulator, sr, n_bands=12, lo=80.0, hi=6000.0)
    if depth < 1.0:
        means = envs.mean(axis=1, keepdims=True)
        envs = (1 - depth) * means + depth * envs
    exc = np.zeros(n, dtype=np.float32)
    edges = np.geomspace(80.0, 6000.0, 13)
    noise = np.random.default_rng(seed + 11).standard_normal(n).astype(np.float32)
    for k in range(12):
        sos = signal.butter(4, [edges[k], min(edges[k + 1], sr / 2 - 200)],
                            btype="band", fs=sr, output="sos")
        exc += signal.sosfiltfilt(sos, noise).astype(np.float32) * envs[k]

    # Cuerpo de trueno: coloracion grave + decay + reverb (cadena de synth_thunder)
    sos_body = signal.butter(4, [30, 600], btype="band", fs=sr, output="sos")
    body = signal.sosfiltfilt(sos_body, exc.astype(np.float64)).astype(np.float32)
    t = np.arange(n) / sr
    body *= np.exp(-t / (0.55 * duration_s)).astype(np.float32)
    ir_n = int(0.8 * sr)
    ir = (np.random.default_rng(seed + 13).standard_normal(ir_n)
          * np.exp(-np.linspace(0, 7, ir_n))).astype(np.float32)
    body = signal.fftconvolve(body, ir)[:n].astype(np.float32)
    body = body / (float(np.abs(body).max()) + 1e-9)

    out = body
    if dry_drips > 0:
        out = out + dry_drips * articulator
    return _norm(out)


def sum_thunder_speaks_water(duration_s: float = 8.0, seed: int = 42,
                             sr: int = 44_100) -> np.ndarray:
    """Ancla: trueno real + el mismo tren de gotas, sumados sin acople."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    rumble, _ = _thunder_rumble(rng, sr, n, distance=0.5, intensity=0.9)
    sched = schedule_from_rate(np.full(n, 10.0), sr, seed + 3,
                               radius_traj=np.full(n, 2.2))
    drips = _render_drips(sched, sr, n, seed + 500)
    return _norm(0.9 * rumble + 0.5 * drips)
