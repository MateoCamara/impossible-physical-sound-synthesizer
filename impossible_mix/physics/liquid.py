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
    """Splash mejorado: estallido inicial + cascada de burbujas a distintas
    profundidades temporales.

    Mejoras respecto a la version simple:
      - Pre-impacto con whoosh (ruido coloreado con sweep descendente)
      - Cascada de burbujas con DECAY DE TASA: muchas al principio,
        progresivamente menos hacia el final (decay natural del splash)
      - Burbujas mas pequenas al inicio (microbubbles) y mas grandes despues
        (la energia se reparte de fino a grueso por dinamica del jet)
      - Subsurface bed: ruido bajo bandpass simulando agitacion subacuatica
    """
    n = int(p.duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(p.seed)

    # 1) Pre-impacto (whoosh) — ruido coloreado con sweep descendente
    pre_n = int(0.025 * sr)
    pre = rng.standard_normal(pre_n).astype(np.float32) * p.intensity
    # Sweep cutoff de 6kHz a 1kHz a lo largo del whoosh (simula entrada del objeto)
    sweep_envelope = np.linspace(6000, 1500, pre_n)
    pre_filtered = np.zeros_like(pre)
    # Aplicar filtros en bloques pequenos (aprox del sweep)
    block_n = max(20, pre_n // 10)
    for i in range(0, pre_n, block_n):
        block = pre[i:i + block_n]
        cutoff = float(sweep_envelope[min(i, pre_n - 1)])
        cutoff_safe = min(cutoff, sr / 2 - 300)
        if cutoff_safe < 200:
            cutoff_safe = 200
        sos = signal.butter(2, [200, cutoff_safe], btype="band", fs=sr, output="sos")
        try:
            pre_filtered[i:i + block_n] = signal.sosfiltfilt(sos, block).astype(np.float32)
        except ValueError:
            pre_filtered[i:i + block_n] = block
    out[: pre_n] += pre_filtered

    # 2) Cascada de burbujas con tasa decreciente
    n_bubbles_total = int(p.n_bubbles * p.intensity)
    spread_samples = int(p.spread_ms / 1000.0 * sr)
    # Distribucion temporal: mas densa al inicio, decae exponencialmente
    # Generamos tiempos sampleados de distribucion exponencial
    raw_times = rng.exponential(scale=spread_samples * 0.3, size=n_bubbles_total)
    raw_times = np.clip(raw_times, 0, spread_samples).astype(int)

    for i, offset in enumerate(raw_times):
        # Burbujas iniciales pequenas (microbubbles del impacto), despues mas grandes
        progress = i / max(n_bubbles_total, 1)
        # Radio crece con i: microbubbles -> bubbles principales
        radius_growth = 0.4 + 1.6 * progress
        radius = p.bubble_size_mean_mm * radius_growth * (1 + p.bubble_size_var * rng.uniform(-0.5, 1.2))
        radius = max(0.3, radius)
        # Velocidad: la primera fase tiene impactos mas duros
        velocity = float(np.clip(rng.normal(1.0 + 0.3 * (1 - progress), 0.2), 0.4, 1.5))
        d = DropletParams(droplet_radius_mm=radius, viscosity=p.viscosity,
                          surface_hardness=0.15, roll_velocity_hz=1, path_roughness=0,
                          duration_s=0.6, seed=p.seed + i * 3)
        evt = synth_drip_event(d, sr, velocity_factor=velocity)
        amp = rng.uniform(0.25, 1.0) * p.intensity * (0.5 + 0.5 * (1 - progress * 0.5))
        end = min(n, offset + len(evt))
        out[offset:end] += amp * evt[: end - offset]

    # 3) Subsurface bed: agitacion subacuatica de baja frecuencia post-impacto
    bed_start = pre_n
    bed_n = n - bed_start
    if bed_n > 100:
        bed = rng.standard_normal(bed_n).astype(np.float32) * 0.1 * p.intensity
        sos = signal.butter(2, [80, 600], btype="band", fs=sr, output="sos")
        bed = signal.sosfiltfilt(sos, bed).astype(np.float32)
        # Envolvente exponencial decreciente
        bed_env = np.exp(-np.linspace(0, 4, bed_n)).astype(np.float32)
        out[bed_start:] += bed * bed_env * 0.3

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
