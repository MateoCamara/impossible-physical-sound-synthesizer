"""Sintesis granular mejorada: nubes estocasticas de micro-impactos donde
cada grano tiene su propia (a) energia/velocidad de contacto, (b) tamano
fisico, (c) perfil mineral, y (d) acoplamiento con la superficie sobre
la que cae.

Cubre: grava rodando, pasos en grava, arena cayendo, gravel scrape,
escombros, granito, etc.

Mejoras respecto a la version anterior:
  - GrainProfile catalog: pebble, fine_gravel, coarse_gravel, sand,
    crushed_glass, broken_ceramic, basalt — cada uno con su firma modal
  - Velocity per grain: contactos mas duros suenan mas brillantes y secos
  - Surface coupling: la superficie donde cae el grano aporta una cola
    modal corta (reutiliza SURFACE_PROFILES de droplet)
  - Variabilidad: cada grano tiene jitter de freq, damping, y duracion
  - Cluster bursts: opcionalmente agrupar granos en ráfagas (mas natural
    que un Poisson uniforme — los granos reales caen en grupos)
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
class GrainProfile:
    """Firma fisica de un tipo de grano individual."""
    name: str
    base_freq_hz: float       # frecuencia modal base
    spread_octaves: float     # cuanto varia la freq por grano (en octavas)
    damping_ms: float         # t60 medio
    inharmonicity: float      # jitter relativo
    n_modes: int = 3          # modos por grano
    spectral_tilt_db_oct: float = -3.0  # tilt del banco modal


GRAIN_PROFILES: dict[str, GrainProfile] = {
    "pebble":         GrainProfile("pebble",         base_freq_hz=900,  spread_octaves=0.7, damping_ms=22, inharmonicity=0.4, n_modes=3),
    "fine_gravel":    GrainProfile("fine_gravel",    base_freq_hz=1600, spread_octaves=0.9, damping_ms=12, inharmonicity=0.5, n_modes=3),
    "coarse_gravel":  GrainProfile("coarse_gravel",  base_freq_hz=550,  spread_octaves=0.6, damping_ms=35, inharmonicity=0.45, n_modes=4),
    "sand":           GrainProfile("sand",           base_freq_hz=3500, spread_octaves=1.2, damping_ms=4,  inharmonicity=0.7, n_modes=2),
    "crushed_glass":  GrainProfile("crushed_glass",  base_freq_hz=2800, spread_octaves=0.8, damping_ms=120, inharmonicity=0.1, n_modes=4),
    "broken_ceramic": GrainProfile("broken_ceramic", base_freq_hz=2000, spread_octaves=0.5, damping_ms=80, inharmonicity=0.15, n_modes=4),
    "basalt":         GrainProfile("basalt",         base_freq_hz=420,  spread_octaves=0.5, damping_ms=30, inharmonicity=0.55, n_modes=3),
    "ice":            GrainProfile("ice",            base_freq_hz=2400, spread_octaves=0.9, damping_ms=180, inharmonicity=0.08, n_modes=3),
    # --- Nuevos perfiles granulares ---
    "ice_shards":     GrainProfile("ice_shards",     base_freq_hz=3200, spread_octaves=1.0, damping_ms=80,  inharmonicity=0.15, n_modes=3),
    "snow_crunch":    GrainProfile("snow_crunch",    base_freq_hz=2200, spread_octaves=1.5, damping_ms=3,   inharmonicity=0.85, n_modes=2),
    "ash":            GrainProfile("ash",            base_freq_hz=4500, spread_octaves=1.3, damping_ms=2,   inharmonicity=0.90, n_modes=2),
}


@dataclass
class GranularParams:
    """Parametros del flujo granular."""
    grain_profile: str = "pebble"               # nombre del perfil de grano
    surface_profile: str = "stone"              # superficie sobre la que caen los granos
    density_hz: float = 80.0                    # granos por segundo (medio)
    density_jitter: float = 0.7                 # 0=regular, 1=fuertemente aleatorio
    size_variance: float = 0.4                  # variabilidad de tamano (=freq)
    energy_mean: float = 0.6                    # intensidad media (0..1)
    energy_jitter: float = 0.5                  # variabilidad por grano
    cluster_factor: float = 0.4                 # 0=poisson; 1=fuertemente agrupado en ráfagas
    surface_coupling: float = 0.35              # 0=sin cola material; 1=fuerte
    duration_s: float = 5.0
    seed: int = 0
    # Compatibilidad con API anterior
    grain_material: str | None = None           # alias de grain_profile (legacy)
    grain_size_mm: float | None = None          # ahora derivado del profile
    energy: float | None = None                 # alias legacy de energy_mean
    spatial_spread: float = 0.7                 # bed-noise modulation


# Mapping legacy: composer y otros codigos llamaban con grain_material=rock/metal/wood
LEGACY_MATERIAL_TO_PROFILE = {
    "rock": "pebble",
    "stone": "pebble",
    "metal": "broken_ceramic",   # no perfecto pero suena metal-ish
    "wood": "coarse_gravel",
    "glass": "crushed_glass",
    "earth": "sand",
    "gravel": "fine_gravel",
}


def _resolve_profile(p: GranularParams) -> GrainProfile:
    name = p.grain_profile
    # Legacy alias: si vino grain_material y el grain_profile sigue default, mapear
    if p.grain_material and p.grain_profile == "pebble":
        name = LEGACY_MATERIAL_TO_PROFILE.get(p.grain_material, "pebble")
    if name not in GRAIN_PROFILES:
        name = "pebble"
    return GRAIN_PROFILES[name]


def _synth_single_grain(profile: GrainProfile, sr: int, velocity: float,
                         seed: int) -> np.ndarray:
    """Genera UN grano individual: banco de resonadores cortos con jitter."""
    rng = np.random.default_rng(seed)
    # Velocidad afecta: amplitud, brillo (freq sube), duracion (mas corto)
    vel = max(0.2, velocity)
    dur_ms = profile.damping_ms * (0.7 + 0.6 / vel)
    n = max(8, int(dur_ms / 1000.0 * sr))
    out = np.zeros(n, dtype=np.float32)
    # Excitacion: pulso corto + ruido attack
    exc_n = max(2, int(0.0008 * sr / vel))
    exc = rng.standard_normal(exc_n).astype(np.float32) * vel
    exc_full = np.zeros(n, dtype=np.float32)
    exc_full[:exc_n] = exc
    # Variabilidad de freq por grano: dispersion octavas
    freq_jitter = profile.spread_octaves * rng.uniform(-1, 1)
    base = profile.base_freq_hz * (2 ** freq_jitter) * (0.85 + 0.3 * vel)
    # Banco modal con tilt
    gains = np.array([10 ** (profile.spectral_tilt_db_oct * k / 20) for k in range(profile.n_modes)])
    gains /= gains.sum()
    t60_s = dur_ms / 1000.0
    for k, g in enumerate(gains):
        fc = base * (1 + 0.7 * k) * (1 + profile.inharmonicity * rng.uniform(-1, 1))
        if fc <= 50 or fc >= sr / 2 - 100:
            continue
        r = float(np.exp(-6.91 / max(t60_s * sr, 1e-3)))
        theta = 2 * np.pi * fc / sr
        a = np.array([1.0, -2 * r * np.cos(theta), r * r])
        b = np.array([1.0, 0.0, -1.0])
        out += g * signal.lfilter(b, a, exc_full).astype(np.float32)
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out * (0.7 / peak)).astype(np.float32) if peak > 0 else out


def synth_granular_flow(p: GranularParams, sr: int = 44_100) -> np.ndarray:
    """Flujo granular completo con cluster bursts, velocity per grain y
    surface coupling."""
    n_total = int(p.duration_s * sr)
    out = np.zeros(n_total, dtype=np.float32)
    rng = np.random.default_rng(p.seed)
    profile = _resolve_profile(p)
    energy_mean = float(p.energy if p.energy is not None else p.energy_mean)
    period_samples = sr / max(p.density_hz, 0.5)

    # Cluster bursts: dividimos en "anclas" cada ~0.5 s; cuando cluster>0 los
    # granos se concentran cerca de las anclas.
    n_anchors = max(2, int(p.duration_s * 2))
    anchors = np.linspace(0, n_total, n_anchors)
    anchor_widths = (period_samples * 3) * (1 - p.cluster_factor)  # ancho de cluster

    grain_count = 0
    t = 0.0
    while t < n_total:
        # Posicion: si cluster_factor alto, se acerca a la ancla mas cercana
        if p.cluster_factor > 0.05:
            nearest = anchors[np.argmin(np.abs(anchors - t))]
            jitter = rng.normal(0, anchor_widths) if anchor_widths > 0 else 0
            sample_pos = int(np.clip(nearest + jitter, 0, n_total - 1))
        else:
            sample_pos = int(t)

        # Velocity per grain: gausiana centrada en 1.0 con std proporcional a energy_jitter
        velocity = float(np.clip(rng.normal(1.0, 0.3 + 0.4 * p.energy_jitter), 0.3, 2.0))
        # Energia por grano
        amp = velocity * energy_mean * rng.uniform(0.5, 1.3)
        # Generar grano
        evt = _synth_single_grain(profile, sr, velocity, seed=p.seed + grain_count * 7 + 11)
        # Mezclar
        end = min(n_total, sample_pos + len(evt))
        out[sample_pos:end] += amp * evt[: end - sample_pos]
        # Avanzar tiempo
        t += period_samples * (1 + p.density_jitter * rng.uniform(-0.85, 0.85))
        grain_count += 1

    # Surface coupling: cuando los granos caen sobre una superficie, hay una
    # cola modal de la superficie excitada por toda la nube
    if p.surface_coupling > 0.05:
        surface = SURFACE_PROFILES.get(p.surface_profile, SURFACE_PROFILES["stone"])
        # IR del surface: impulso pasa por todos los modos
        ir_n = min(int(surface.t60_ms / 1000.0 * sr * 2), int(0.5 * sr))
        if ir_n > 100:
            ir = np.zeros(ir_n, dtype=np.float32)
            impulse = np.zeros(ir_n, dtype=np.float32)
            impulse[:8] = rng.standard_normal(8).astype(np.float32) * 0.3
            t60_s = surface.t60_ms / 1000.0
            for fc, mg in zip(surface.modes_hz, surface.mode_gains):
                if fc <= 0 or fc >= sr / 2 - 100:
                    continue
                fc_j = fc * (1 + surface.inharmonicity * rng.uniform(-1, 1))
                r = float(np.exp(-6.91 / max(t60_s * sr, 1e-3)))
                theta = 2 * np.pi * fc_j / sr
                a = np.array([1.0, -2 * r * np.cos(theta), r * r])
                b = np.array([1.0, 0.0, -1.0])
                ir += mg * signal.lfilter(b, a, impulse).astype(np.float32)
            # Convolucion ligera entre la nube y la IR del surface
            tail = signal.fftconvolve(out, ir, mode="full")[:n_total].astype(np.float32)
            out = out + tail * 0.3 * p.surface_coupling

    # Background bed sutil
    if p.spatial_spread > 0.2:
        bed = rng.standard_normal(n_total).astype(np.float32) * 0.02 * p.spatial_spread
        sos = signal.butter(4, [200, 4000], btype="band", fs=sr, output="sos")
        bed = signal.sosfiltfilt(sos, bed).astype(np.float32)
        win = max(1, int(0.04 * sr))
        rms = np.sqrt(np.convolve(out * out, np.ones(win) / win, mode="same"))
        rms_n = rms / (rms.max() + 1e-9)
        out = out + bed * rms_n * 0.5

    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
