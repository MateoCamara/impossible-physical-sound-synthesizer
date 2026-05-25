"""Sintesis fisica de eventos liquidos no rodantes: splash, pour, drip aislado.

Cubre los eventos liquidos que NO son rolling droplet:
  - splash: impacto liquido contra superficie (multiples burbujas a la vez)
  - pour: stream continuo + multiples chirps superpuestos
  - drip: una sola gota (importable desde droplet.py si se prefiere)

Reusa drip_event de droplet.py como primitiva.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from impossible_mix.physics.droplet import DropletParams, synth_drip_event


@dataclass
class SplashParams:
    """Parametros del splash (impacto liquido contra superficie)."""
    intensity: float = 0.7              # 0..1 (mas = mas burbujas + mas amplitud)
    bubble_size_mean_mm: float = 3.0    # gotas mas grandes = chirps mas graves
    bubble_size_var: float = 0.6        # variabilidad de tamanos
    n_bubbles: int = 30                 # cuantas burbujas en el splash
    spread_ms: float = 80.0             # dispersion temporal de las burbujas
    viscosity: float = 0.1
    duration_s: float = 5.0
    seed: int = 0


def synth_splash(p: SplashParams, sr: int = 44_100) -> np.ndarray:
    """Splash: estallido inicial + N burbujas dispersas en spread_ms."""
    n = int(p.duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(p.seed)

    # Pre-impacto: ruido coloreado corto (entrada del objeto en el agua)
    pre_n = int(0.015 * sr)
    pre = rng.standard_normal(pre_n).astype(np.float32) * p.intensity
    sos = signal.butter(4, [300, 4000], btype="band", fs=sr, output="sos")
    pre = signal.sosfiltfilt(sos, pre).astype(np.float32)
    out[: pre_n] += pre

    # Lanzar n_bubbles drips dispersos
    spread_samples = int(p.spread_ms / 1000.0 * sr)
    for i in range(int(p.n_bubbles * p.intensity)):
        radius = p.bubble_size_mean_mm * (1 + p.bubble_size_var * rng.uniform(-0.7, 1.5))
        radius = max(0.5, radius)
        d = DropletParams(droplet_radius_mm=radius, viscosity=p.viscosity,
                          surface_hardness=0.2, roll_velocity_hz=1, path_roughness=0,
                          duration_s=0.5, seed=p.seed + i)
        evt = synth_drip_event(d, sr)
        amp = rng.uniform(0.3, 1.0) * p.intensity
        offset = int(rng.uniform(0, spread_samples))
        end = min(n, offset + len(evt))
        out[offset:end] += amp * evt[: end - offset]

    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


@dataclass
class PourParams:
    """Pour: stream liquido continuo."""
    flow_rate: float = 0.6              # 0..1 (mas = mas denso de gotas)
    bubble_size_mean_mm: float = 1.5
    viscosity: float = 0.1
    duration_s: float = 5.0
    seed: int = 0


def synth_pour(p: PourParams, sr: int = 44_100) -> np.ndarray:
    """Pour: como rolling_droplet con muchisimas mas gotas casi continuas
    + bed de ruido blanco bandpass (turbulencia)."""
    n = int(p.duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(p.seed)

    # Densidad de drips
    drip_rate = 30 + 200 * p.flow_rate
    period_samples = sr / drip_rate
    t = 0.0
    while t < n:
        radius = p.bubble_size_mean_mm * (1 + 0.4 * rng.uniform(-0.5, 0.8))
        radius = max(0.3, radius)
        d = DropletParams(droplet_radius_mm=radius, viscosity=p.viscosity,
                          surface_hardness=0.1, roll_velocity_hz=1, path_roughness=0,
                          duration_s=0.2, seed=p.seed + int(t))
        evt = synth_drip_event(d, sr)
        amp = rng.uniform(0.4, 0.9)
        start = int(t)
        end = min(n, start + len(evt))
        out[start:end] += amp * evt[: end - start]
        t += period_samples * (1 + 0.6 * rng.uniform(-0.7, 0.7))

    # Bed de turbulencia
    bed = rng.standard_normal(n).astype(np.float32) * 0.05 * p.flow_rate
    sos = signal.butter(4, [400, 3500], btype="band", fs=sr, output="sos")
    bed = signal.sosfiltfilt(sos, bed).astype(np.float32)
    out = out + bed

    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
