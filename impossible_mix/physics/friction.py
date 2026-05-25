"""Sintesis fisica de friccion: scrape, drag, raspado.

Modelo: noise excitation modulada por velocidad relativa, filtrada por
banco resonante del material (que actua como "cuerpo" de la superficie).
La velocidad fluctua (no es constante) para simular irregularidad humana
y rugosidad de la superficie.

Referencias:
  - Avanzini, Crosato (2006) 'Friction models for sound synthesis'
  - Serafin (2004) 'The sound of friction: real-time models'
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class FrictionParams:
    """Parametros fisicos del scrape/drag."""
    surface_hardness: float = 0.6       # 0=tela, 1=metal/roca
    roughness: float = 0.5              # 0=superficie pulida, 1=lija gruesa
    velocity_mean: float = 0.7          # 0..1 velocidad media (modula bright + density)
    velocity_jitter: float = 0.4        # 0=movimiento suave, 1=irregular
    pressure: float = 0.6               # 0=apenas roza, 1=fuerte
    body_resonance_hz: float = 1200.0   # resonancia principal del material
    body_q: float = 4.0                 # Q del cuerpo (mas alto = mas tonal)
    duration_s: float = 5.0
    seed: int = 0


def _velocity_envelope(p: FrictionParams, sr: int) -> np.ndarray:
    """Velocidad relativa fluctuante (envolvente 0..1) para modular el ruido."""
    n = int(p.duration_s * sr)
    rng = np.random.default_rng(p.seed)
    # Ruido lento (1-8 Hz) modulado y centrado en velocity_mean
    raw = rng.standard_normal(n).astype(np.float32)
    # Filtro paso bajo aleatorio para suavizar -> envolvente "manual"
    sos = signal.butter(2, max(0.5, 3 + 12 * p.velocity_jitter), btype="low", fs=sr, output="sos")
    env = signal.sosfiltfilt(sos, raw).astype(np.float32)
    env = (env - env.mean()) / (env.std() + 1e-9)
    env = np.clip(p.velocity_mean + 0.4 * p.velocity_jitter * env, 0.0, 1.2)
    # Anadir micro-pulsos cuando roughness alta (saltos de grano)
    if p.roughness > 0.1:
        spike_rate_hz = 8 + 80 * p.roughness
        n_spikes = int(p.duration_s * spike_rate_hz)
        idx = rng.integers(0, n, n_spikes)
        env[idx] = np.clip(env[idx] * (1 + 1.5 * p.roughness), 0, 1.5)
    return env.astype(np.float32)


def synth_scrape(p: FrictionParams, sr: int = 44_100) -> np.ndarray:
    """Genera scrape/drag completo.

    Pipeline:
      1) Ruido coloreado (bandpass que depende de hardness)
      2) Modulado por velocity envelope
      3) Pasado por resonadores del cuerpo del material
      4) Amplitud final por pressure
    """
    n = int(p.duration_s * sr)
    rng = np.random.default_rng(p.seed + 1)

    # 1) Excitacion: ruido bandpass
    noise = rng.standard_normal(n).astype(np.float32)
    # Hardness alta -> bandpass mas alto y estrecho
    f_low = 300 + 1500 * p.surface_hardness
    f_high = 2000 + 6000 * p.surface_hardness
    sos = signal.butter(4, [f_low, min(f_high, sr / 2 - 100)], btype="band",
                        fs=sr, output="sos")
    bright_noise = signal.sosfiltfilt(sos, noise).astype(np.float32)

    # Capa de "grain" cuando roughness alta: tren de pulsos cortos aleatorios
    if p.roughness > 0.05:
        grain_rate = 20 + 200 * p.roughness
        n_grains = int(p.duration_s * grain_rate)
        grain_idx = rng.integers(0, n - 5, n_grains)
        grain_amp = rng.uniform(0.3, 1.0, n_grains).astype(np.float32) * (0.4 + 0.6 * p.roughness)
        for idx, amp in zip(grain_idx, grain_amp):
            bright_noise[idx:idx + 5] += amp * rng.standard_normal(5).astype(np.float32)

    # 2) Modular por velocidad
    env = _velocity_envelope(p, sr)
    excited = bright_noise * env

    # 3) Banco de resonadores: cuerpo del material
    # Resonancia principal + 2-3 armónicos
    body = np.zeros(n, dtype=np.float32)
    n_body_modes = 3 if p.body_q > 2 else 1
    for k in range(n_body_modes):
        fc = p.body_resonance_hz * (1.0 + 0.6 * k * (1 + 0.3 * rng.uniform()))
        if fc >= sr / 2:
            continue
        Q = p.body_q * (0.8 + 0.4 * rng.uniform())
        bw = fc / max(Q, 0.5)
        sos = signal.butter(2, [max(50, fc - bw / 2), min(sr / 2 - 100, fc + bw / 2)],
                            btype="band", fs=sr, output="sos")
        gain = 0.7 / (1 + k * 0.5)
        body += gain * signal.sosfiltfilt(sos, excited).astype(np.float32)

    out = 0.6 * excited + 0.5 * body
    out = out * (0.3 + 0.7 * p.pressure)
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
