"""Sintesis de friccion mejorada: scrape, drag, raspado.

Modelo extendido respecto a la version simple:
  - Velocity envelope con stick-slip (saltos discretos tipicos del scrape
    real, no solo modulacion suave)
  - Cuerpo material-aware: el banco de resonadores depende del SurfaceProfile
    (reusa el catalogo de droplet.py)
  - Multi-mode body excited por el ruido modulado por velocity
  - Texture micro-impacts cuando la rugosidad es alta (no solo amplitud
    spikes, sino impactos reales de grano)

Referencias:
  - Avanzini, Crosato (2006) 'Friction models for sound synthesis'
  - Serafin (2004) 'The sound of friction: real-time models'
  - Stick-slip: Coulomb friction discontinua entre fases adherida y deslizante.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from impossible_mix.physics.droplet import (
    SURFACE_PROFILES,
    SurfaceProfile,
    surface_from_hardness,
)


@dataclass
class FrictionParams:
    """Parametros fisicos del scrape/drag."""
    surface_profile: str | None = None        # 'wood', 'ceramic', 'metal'... None => derivar de hardness
    surface_hardness: float = 0.6             # usado si surface_profile=None
    roughness: float = 0.5                    # 0=pulido, 1=lija gruesa
    velocity_mean: float = 0.7                # 0..1 velocidad media
    velocity_jitter: float = 0.4              # 0=movimiento suave, 1=irregular
    stick_slip: float = 0.3                   # 0=no stick-slip; 1=fuertes saltos discretos
    stick_slip_rate_hz: float = 30.0          # eventos slip por segundo (medio)
    pressure: float = 0.6                     # 0=apenas roza, 1=fuerte
    duration_s: float = 5.0
    seed: int = 0
    # Legacy compat fields
    body_resonance_hz: float | None = None    # override del modal principal
    body_q: float | None = None


def _resolve_surface(p: FrictionParams) -> SurfaceProfile:
    if p.surface_profile and p.surface_profile in SURFACE_PROFILES:
        return SURFACE_PROFILES[p.surface_profile]
    return surface_from_hardness(p.surface_hardness)


def _velocity_envelope_with_stickslip(p: FrictionParams, sr: int) -> np.ndarray:
    """Envolvente de velocidad con stick-slip dynamics.

    Continuous part: ruido lento bandpassed (movimiento humano).
    Stick-slip part: serie de eventos discretos donde el surface se 'rompe'
    de la fase adherida — cada uno con attack rapido + decay corto.
    """
    n = int(p.duration_s * sr)
    rng = np.random.default_rng(p.seed)
    # Componente continua
    raw = rng.standard_normal(n).astype(np.float32)
    sos = signal.butter(2, max(0.5, 3 + 12 * p.velocity_jitter), btype="low",
                        fs=sr, output="sos")
    cont = signal.sosfiltfilt(sos, raw).astype(np.float32)
    cont = (cont - cont.mean()) / (cont.std() + 1e-9)
    env = np.clip(p.velocity_mean + 0.4 * p.velocity_jitter * cont, 0.0, 1.5)

    # Stick-slip: serie de saltos
    if p.stick_slip > 0.05:
        # Tasa total de slips proporcional a stick_slip y velocity
        slip_rate = p.stick_slip_rate_hz * (0.3 + 1.5 * p.stick_slip) * (0.5 + p.velocity_mean)
        n_slips = int(slip_rate * p.duration_s)
        for _ in range(n_slips):
            idx = int(rng.uniform(0, n - 100))
            attack_n = max(2, int(0.0008 * sr))
            decay_n = max(10, int(rng.uniform(0.003, 0.015) * sr))
            # Bump corto
            bump = np.concatenate([
                np.linspace(0, 1, attack_n),
                np.exp(-np.linspace(0, 4, decay_n))
            ]).astype(np.float32)
            amp = rng.uniform(0.4, 1.0) * p.stick_slip
            end = min(n, idx + len(bump))
            env[idx:end] = np.maximum(env[idx:end], bump[: end - idx] * amp + p.velocity_mean * 0.5)
    return env.astype(np.float32)


def synth_scrape(p: FrictionParams, sr: int = 44_100) -> np.ndarray:
    """Scrape/drag mejorado. Pipeline:
      1) Ruido bandpass cuya banda depende del material
      2) Modulacion por velocity envelope con stick-slip
      3) Excitacion del banco modal del surface (multi-mode)
      4) Micro-impactos discretos cuando rugosidad alta
    """
    n = int(p.duration_s * sr)
    rng = np.random.default_rng(p.seed + 1)
    surface = _resolve_surface(p)

    # 1) Excitacion: ruido bandpass del material
    noise = rng.standard_normal(n).astype(np.float32)
    cl_lo, cl_hi = surface.click_color_hz
    cl_hi_safe = min(cl_hi, sr / 2 - 200)
    sos = signal.butter(4, [cl_lo, cl_hi_safe], btype="band", fs=sr, output="sos")
    bright_noise = signal.sosfiltfilt(sos, noise).astype(np.float32)

    # 2) Modular por velocity envelope con stick-slip
    env = _velocity_envelope_with_stickslip(p, sr)
    excited = bright_noise * env

    # 3) Body multi-mode response del surface
    body = np.zeros(n, dtype=np.float32)
    t60_s = surface.t60_ms / 1000.0 * 0.7  # body un poco mas corto que un golpe completo
    for fc, mg in zip(surface.modes_hz, surface.mode_gains):
        if fc <= 0 or fc >= sr / 2 - 100:
            continue
        fc_j = fc * (1 + surface.inharmonicity * rng.uniform(-1, 1))
        # Override si user paso body_resonance_hz
        if p.body_resonance_hz is not None:
            fc_j = p.body_resonance_hz * (1 + 0.6 * mg)
        q = float(p.body_q) if p.body_q is not None else (3 + 8 * (1 - surface.inharmonicity))
        bw = fc_j / max(q, 0.5)
        try:
            sos_band = signal.butter(2,
                                      [max(50, fc_j - bw / 2), min(sr / 2 - 100, fc_j + bw / 2)],
                                      btype="band", fs=sr, output="sos")
            body += mg * signal.sosfiltfilt(sos_band, excited).astype(np.float32)
        except ValueError:
            continue

    out = 0.55 * excited + 0.6 * body

    # 4) Micro-impactos cuando rugosidad alta (granos de la lija)
    if p.roughness > 0.1:
        grain_rate = 30 + 250 * p.roughness
        n_grains = int(p.duration_s * grain_rate)
        for _ in range(n_grains):
            idx = int(rng.uniform(0, n - 10))
            # Cada micro-impacto: pulso muy corto + breve resonancia del surface
            grain_n = 6
            spike = rng.standard_normal(grain_n).astype(np.float32) * rng.uniform(0.2, 0.8) * p.roughness
            # Atenuado por velocity local
            local_vel = env[idx]
            out[idx:idx + grain_n] += spike * local_vel

    out = out * (0.3 + 0.7 * p.pressure)
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
