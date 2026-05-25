"""Fisicas exoticas: rain, fire, thunder, glass_break, ocean_wave.

Cada una compuesta a partir de las primitivas existentes (modal, drip,
granular, friction) + envolventes especificas. Util para demos y para
mostrar alcance del marco mas alla de las 3 combinaciones canonicas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from impossible_mix.physics.droplet import DropletParams, synth_drip_event
from impossible_mix.physics.granular import GranularParams, synth_granular_flow
from impossible_mix.physics.modal import (
    MaterialModalProfile,
    PROFILES as MODAL_PROFILES,
    modal_frequencies,
    synth_modal_impact,
)


def synth_rain(duration_s: float = 5.0, intensity: float = 0.6,
               drop_size_mm: float = 1.2, sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Nube densa de drips minusculos = lluvia."""
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    rate = 40 + 200 * intensity
    period = sr / rate
    t = 0
    while t < n:
        radius = drop_size_mm * (1 + 0.5 * rng.uniform(-0.5, 0.8))
        radius = max(0.3, radius)
        d = DropletParams(droplet_radius_mm=radius, viscosity=0.0,
                          surface_hardness=0.0, roll_velocity_hz=1, path_roughness=0,
                          duration_s=0.12, seed=seed + int(t))
        evt = synth_drip_event(d, sr)
        amp = rng.uniform(0.2, 0.7) * (0.4 + intensity)
        start = int(t)
        end = min(n, start + len(evt))
        out[start:end] += amp * evt[: end - start]
        t += period * (1 + 0.7 * rng.uniform(-0.8, 0.8))
    # Background hiss bandpass alto
    noise = rng.standard_normal(n).astype(np.float32) * 0.05 * intensity
    sos = signal.butter(2, [500, 5000], btype="band", fs=sr, output="sos")
    bed = signal.sosfiltfilt(sos, noise).astype(np.float32)
    out = out + bed
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_fire(duration_s: float = 5.0, intensity: float = 0.7,
               crackle_density: float = 0.5, sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Fuego: bed caotico bandpass + crackles aleatorios (pops cortos)."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    # Bed: ruido marrón modulado
    bed = rng.standard_normal(n).astype(np.float32) * 0.15
    sos = signal.butter(4, [80, 1500], btype="band", fs=sr, output="sos")
    bed = signal.sosfiltfilt(sos, bed).astype(np.float32)
    # Slow amplitude modulation (oscilacion de llama)
    am_freq = 1.5 + rng.random() * 2
    am = 0.6 + 0.4 * np.sin(2 * np.pi * am_freq * np.arange(n) / sr)
    bed = bed * am.astype(np.float32) * intensity
    # Crackles: pops aleatorios cortos
    out = bed.copy()
    pop_rate = 4 + 25 * crackle_density
    n_pops = int(duration_s * pop_rate)
    for _ in range(n_pops):
        idx = int(rng.uniform(0, n - 100))
        pop_len = rng.integers(50, 200)
        pop = rng.standard_normal(pop_len).astype(np.float32) * 0.6
        envelope = np.exp(-np.linspace(0, 5, pop_len))
        pop *= envelope
        out[idx:idx + pop_len] += pop * rng.uniform(0.3, 1.0) * intensity
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_thunder(duration_s: float = 6.0, distance: float = 0.5,
                  intensity: float = 0.9, sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Trueno: sweep modal grave + ruido coloreado decreciente.
    distance 0=cerca (golpe seco), 1=lejos (rumble difuso).
    """
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float32)
    # 1) Crack inicial (si cerca) - impulso filtrado alto
    if distance < 0.6:
        crack_n = int(0.05 * sr)
        crack = rng.standard_normal(crack_n).astype(np.float32) * (1 - distance)
        sos = signal.butter(4, [400, 5000], btype="band", fs=sr, output="sos")
        crack = signal.sosfiltfilt(sos, crack).astype(np.float32)
        out[: crack_n] += crack * 0.8
    # 2) Rumble grave largo (decay exponencial 3-5s)
    rumble = rng.standard_normal(n).astype(np.float32)
    # Lowpass progresivamente mas oscuro con la distancia
    f_high = 600 - 400 * distance
    sos = signal.butter(4, [30, max(80, f_high)], btype="band", fs=sr, output="sos")
    rumble = signal.sosfiltfilt(sos, rumble).astype(np.float32)
    decay = np.exp(-np.linspace(0, 4 - 2 * distance, n)).astype(np.float32)
    rumble = rumble * decay * (0.4 + 0.6 * (1 - distance * 0.5)) * intensity
    out = out + rumble
    # 3) Reverberacion via convolucion con impulso exponencial corto
    ir_n = int(0.4 * sr)
    ir = (np.exp(-np.linspace(0, 6, ir_n)) * rng.standard_normal(ir_n)).astype(np.float32) * 0.3
    out_rev = np.convolve(out, ir, mode="full")[:n].astype(np.float32)
    out = 0.7 * out + 0.4 * out_rev
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_glass_break(duration_s: float = 4.0, n_shards: int = 30,
                      sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Cristal rompiendose: pico modal glass + cascada granular de shards."""
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    # 1) Crack inicial: ruido bandpass agudo corto
    crack_n = int(0.025 * sr)
    crack = rng.standard_normal(crack_n).astype(np.float32)
    sos = signal.butter(4, [2000, 8000], btype="band", fs=sr, output="sos")
    crack = signal.sosfiltfilt(sos, crack).astype(np.float32) * 0.9
    out[: crack_n] += crack
    # 2) Cascada de shards: N impactos modales glass cortos dispersos
    glass = MODAL_PROFILES["glass"]
    spread_samples = int(0.6 * sr)  # dispersion en 600 ms
    for k in range(n_shards):
        offset = int(rng.uniform(0.01, spread_samples / sr) * sr)
        sub_profile = MaterialModalProfile(
            name=f"shard_{k}",
            n_modes=4,
            fundamental_hz=glass.fundamental_hz * (0.5 + 1.5 * rng.random()),
            spacing=glass.spacing,
            damping_ms=glass.damping_ms * (0.3 + 0.7 * rng.random()),
            spectrum_shape="flat",
            inharmonicity=glass.inharmonicity,
            seed=seed + k,
        )
        evt = synth_modal_impact(sub_profile, sr,
                                 duration_s=duration_s - offset / sr,
                                 impact_time_s=0.001,
                                 impact_strength=rng.uniform(0.3, 1.0),
                                 sharpness=2.0)
        amp = rng.uniform(0.2, 0.7) / (1 + k * 0.05)
        end = min(n, offset + len(evt))
        out[offset:end] += amp * evt[: end - offset]
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_ocean_wave(duration_s: float = 6.0, breaking_intensity: float = 0.7,
                     sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Ola del mar rompiendo: build-up (noise crescendo) + crash (granular bubbles) + receding (LPF decay)."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float32)

    # Fase 1: build-up (0..40%) — ruido bandpass crescendo
    p1_end = int(0.4 * n)
    bed = rng.standard_normal(p1_end).astype(np.float32) * 0.2
    sos = signal.butter(2, [200, 1800], btype="band", fs=sr, output="sos")
    bed = signal.sosfiltfilt(sos, bed).astype(np.float32)
    crescendo = np.linspace(0.1, 1.0, p1_end).astype(np.float32) ** 1.5
    out[: p1_end] += bed * crescendo

    # Fase 2: crash (40..60%) — burbujas densas y bandpass amplio
    p2_start, p2_end = p1_end, int(0.6 * n)
    p2_n = p2_end - p2_start
    crash = rng.standard_normal(p2_n).astype(np.float32) * 0.6 * breaking_intensity
    sos = signal.butter(2, [100, 5000], btype="band", fs=sr, output="sos")
    crash = signal.sosfiltfilt(sos, crash).astype(np.float32)
    out[p2_start:p2_end] += crash
    # Burbujas dentro del crash
    for _ in range(int(40 * breaking_intensity)):
        offset = p2_start + int(rng.uniform(0, p2_n - 100))
        d = DropletParams(droplet_radius_mm=0.8 + 1.5 * rng.random(),
                          viscosity=0.05, surface_hardness=0.1,
                          roll_velocity_hz=1, path_roughness=0,
                          duration_s=0.2, seed=seed + offset)
        evt = synth_drip_event(d, sr)
        end = min(n, offset + len(evt))
        out[offset:end] += 0.3 * evt[: end - offset]

    # Fase 3: receding (60..100%) — decay lento con LPF
    p3_start = p2_end
    p3_n = n - p3_start
    recede = rng.standard_normal(p3_n).astype(np.float32) * 0.3
    sos = signal.butter(2, [80, 1200], btype="band", fs=sr, output="sos")
    recede = signal.sosfiltfilt(sos, recede).astype(np.float32)
    decay = np.exp(-np.linspace(0, 3, p3_n)).astype(np.float32)
    out[p3_start:] += recede * decay
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out
