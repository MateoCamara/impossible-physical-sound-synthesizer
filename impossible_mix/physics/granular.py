"""Sintesis granular: nubes estocasticas de micro-impactos.

Cubre: grava rodando, pasos en grava, arena cayendo, gravel scrape.
Cada grano es un mini-impacto modal corto; densidad y dispersion temporal
dan el caracter macro.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from impossible_mix.physics.modal import (
    MaterialModalProfile,
    PROFILES,
    synth_modal_impact,
)


@dataclass
class GranularParams:
    """Parametros fisicos del flow granular."""
    grain_material: str = "rock"        # perfil modal para cada grano
    density_hz: float = 80.0            # granos por segundo
    density_jitter: float = 0.7         # 0=regular, 1=fuertemente aleatorio
    grain_size_mm: float = 8.0          # tamano medio del grano (afecta freq base)
    size_variance: float = 0.4          # variabilidad de tamano
    energy: float = 0.6                 # intensidad media (0..1)
    spatial_spread: float = 0.7         # 0=fuente puntual, 1=disperso
    duration_s: float = 5.0
    seed: int = 0


def synth_granular_flow(p: GranularParams, sr: int = 44_100) -> np.ndarray:
    """Genera nube granular: grava rodando, pasos, arena cayendo.

    Cada grano es un impacto modal corto desplazado en el tiempo segun
    densidad/jitter. Tamano del grano modula la frecuencia del modal.
    """
    n = int(p.duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(p.seed)

    base_profile = PROFILES.get(p.grain_material, PROFILES["rock"])
    period_samples = sr / max(p.density_hz, 0.5)

    t = 0.0
    grain_count = 0
    while t < n:
        # Tamano del grano -> frecuencia inversa (grano grande = mas grave)
        size = p.grain_size_mm * (1 + p.size_variance * rng.uniform(-0.7, 1.2))
        size = max(0.5, size)
        f_factor = 8.0 / size  # 8mm -> factor 1.0; 4mm -> 2.0; 16mm -> 0.5
        profile = MaterialModalProfile(
            name=f"{p.grain_material}_grain_{grain_count}",
            n_modes=max(2, base_profile.n_modes - 2),
            fundamental_hz=base_profile.fundamental_hz * f_factor,
            spacing=base_profile.spacing,
            damping_ms=max(5.0, base_profile.damping_ms * 0.1),  # granos decaen rapidisimo
            spectrum_shape=base_profile.spectrum_shape,
            inharmonicity=base_profile.inharmonicity,
            seed=base_profile.seed + grain_count,
        )
        # Duracion del evento grano: muy corto
        grain_dur = 0.04 + 0.04 * (size / 8.0)
        evt = synth_modal_impact(profile, sr,
                                 duration_s=grain_dur,
                                 impact_time_s=0.001,
                                 impact_strength=p.energy * rng.uniform(0.4, 1.2),
                                 sharpness=1.5 + 0.5 * rng.random())
        start = int(t)
        end = min(n, start + len(evt))
        if start < n:
            out[start:end] += evt[: end - start]

        # Avanzar tiempo segun densidad + jitter
        offset = period_samples * (1 + p.density_jitter * rng.uniform(-0.9, 0.9))
        t += max(period_samples * 0.1, offset)
        grain_count += 1

    # Background bed sutil (suelo del flujo)
    if p.spatial_spread > 0.2:
        bed = rng.standard_normal(n).astype(np.float32) * 0.02 * p.spatial_spread
        sos = signal.butter(4, [200, 4000], btype="band", fs=sr, output="sos")
        bed = signal.sosfiltfilt(sos, bed).astype(np.float32)
        # Modular por la envolvente RMS del flujo
        win = max(1, int(0.04 * sr))
        rms = np.sqrt(np.convolve(out * out, np.ones(win) / win, mode="same"))
        rms = rms / (rms.max() + 1e-9)
        out = out + bed * rms * 0.5

    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
