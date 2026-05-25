"""Procesado espacial: paneo estereo, distancia (air absorption + delay).

Hace que un mono se vuelva estereo con direccionalidad y profundidad. Los
modelos son aproximaciones razonables (equal-power paneo, LPF de absorcion
del aire, atenuacion 1/r, delay por velocidad del sonido).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

C_SOUND_MS = 343.0  # velocidad del sonido m/s


def equal_power_pan(mono: np.ndarray, pan: float) -> np.ndarray:
    """pan in [-1, 1]; devuelve stereo (T, 2). -1 = totalmente L, +1 = R."""
    pan = float(np.clip(pan, -1, 1))
    # Ley equal-power: L = cos((pan+1)*pi/4), R = sin((pan+1)*pi/4)
    angle = (pan + 1) * np.pi / 4
    l_gain = float(np.cos(angle))
    r_gain = float(np.sin(angle))
    out = np.empty((len(mono), 2), dtype=np.float32)
    out[:, 0] = mono * l_gain
    out[:, 1] = mono * r_gain
    return out


def air_absorption(mono: np.ndarray, sr: int, distance_m: float) -> np.ndarray:
    """Approxima absorcion del aire en altas frecuencias.
    Lowpass cuyo cutoff cae con la distancia. 1m: ~ open; 50m: ~ 2 kHz.
    """
    if distance_m < 0.5:
        return mono
    # cutoff_hz aprox: 20kHz a <1m, 2kHz a 50m
    cutoff_hz = max(800.0, 20000.0 * np.exp(-distance_m / 25.0))
    if cutoff_hz > sr / 2 - 100:
        return mono
    sos = signal.butter(2, cutoff_hz, btype="low", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, mono).astype(np.float32)


def distance_attenuation(mono: np.ndarray, distance_m: float,
                         reference_distance_m: float = 1.0) -> np.ndarray:
    """Atenuacion 1/r normalizada a la distancia de referencia."""
    distance_m = max(0.1, distance_m)
    factor = reference_distance_m / distance_m
    return mono * float(factor)


def propagation_delay(mono: np.ndarray, sr: int, distance_m: float) -> np.ndarray:
    """Anade delay de propagacion al inicio (silencio de t = distance / c)."""
    delay_samples = int(distance_m / C_SOUND_MS * sr)
    if delay_samples <= 0:
        return mono
    return np.concatenate([np.zeros(delay_samples, dtype=np.float32), mono.astype(np.float32)])


def place_source(mono: np.ndarray, sr: int, distance_m: float = 1.0,
                 pan: float = 0.0, with_delay: bool = False) -> np.ndarray:
    """Pipeline completo: aplica absorcion + atenuacion + (opcional) delay + paneo.
    Devuelve estereo (T, 2)."""
    y = mono.astype(np.float32)
    y = air_absorption(y, sr, distance_m)
    y = distance_attenuation(y, distance_m)
    if with_delay:
        y = propagation_delay(y, sr, distance_m)
    return equal_power_pan(y, pan)


def stereo_sum(stereo_list: list[np.ndarray]) -> np.ndarray:
    """Suma varias fuentes estereo (alinea longitudes con padding)."""
    if not stereo_list:
        return np.zeros((0, 2), dtype=np.float32)
    max_len = max(s.shape[0] for s in stereo_list)
    out = np.zeros((max_len, 2), dtype=np.float32)
    for s in stereo_list:
        out[: s.shape[0]] += s
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
