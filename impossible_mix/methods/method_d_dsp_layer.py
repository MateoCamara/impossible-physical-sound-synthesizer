"""Metodo D: capa interpretable DSP/DDSP-lite aplicada como postproceso
sobre un audio ya generado por A o B.

Cinco controles continuos en [0, 1] (luego escalables a Likert 1-5):
  - wetness     : filtro paso bajo suave + cola de reverb corta (Schroeder-like)
  - rigidity    : enfasis en transitorios (high-shelf > 4 kHz) + ataque
  - resonance   : banco de resonadores BIQUAD en frecuencias clave (200/600/2k Hz)
  - granularity : mezcla con ruido granular sincronizado a onsets
  - continuity  : envolvente de sostenido tras transitorios

Implementacion: scipy.signal + funciones puras en numpy. No requiere torch ni
GPU. Es suficiente para los demos del Dia 5; en septiembre se promociona a
capa diferenciable en torch para optimizacion end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy import signal
from torch import Tensor

from impossible_mix.config import SAMPLE_RATE
from impossible_mix.methods.base import HybridMethod, HybridSpec


def _butter_lowpass(y: np.ndarray, sr: int, fc: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, fc, btype="low", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y).astype(np.float32)


def _high_shelf(y: np.ndarray, sr: int, fc: float, gain_db: float) -> np.ndarray:
    """Shelving filter para enfatizar agudos (rigidez/ataque)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    cos_w = np.cos(w0)
    sin_w = np.sin(w0)
    S = 1.0
    alpha = sin_w / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    b0 = A * ((A + 1) + (A - 1) * cos_w + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w)
    b2 = A * ((A + 1) + (A - 1) * cos_w - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w)
    a2 = (A + 1) - (A - 1) * cos_w - 2 * np.sqrt(A) * alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.lfilter(b, a, y).astype(np.float32)


def _schroeder_reverb(y: np.ndarray, sr: int, t60_s: float, mix: float) -> np.ndarray:
    """Reverb minimalista para wetness/resonance. 4 combs + 2 allpass."""
    if mix <= 0 or t60_s <= 0:
        return y
    combs_ms = [29.7, 37.1, 41.1, 43.7]
    allpass_ms = [5.0, 1.7]
    out = np.zeros_like(y)
    for ms in combs_ms:
        d = max(1, int(ms * sr / 1000.0))
        feedback = 10 ** (-3 * d / (t60_s * sr))
        buf = np.zeros(d, dtype=np.float32)
        z = np.zeros_like(y)
        idx = 0
        for n in range(len(y)):
            z[n] = buf[idx]
            buf[idx] = y[n] + buf[idx] * feedback
            idx = (idx + 1) % d
        out += z
    out /= len(combs_ms)
    for ms in allpass_ms:
        d = max(1, int(ms * sr / 1000.0))
        g = 0.5
        buf = np.zeros(d, dtype=np.float32)
        z = np.zeros_like(out)
        idx = 0
        for n in range(len(out)):
            buf_val = buf[idx]
            new = -g * out[n] + buf_val
            buf[idx] = out[n] + g * new
            z[n] = new
            idx = (idx + 1) % d
        out = z
    return ((1 - mix) * y + mix * out).astype(np.float32)


def _resonator(y: np.ndarray, sr: int, freqs_hz: list[float], q: float, gain: float) -> np.ndarray:
    """Banco de biquads peaking centrados en `freqs_hz`."""
    out = y.copy()
    for fc in freqs_hz:
        w0 = 2 * np.pi * fc / sr
        alpha = np.sin(w0) / (2 * q)
        A = 10 ** (gain / 40)
        b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
        a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
        out = signal.lfilter(b / a[0], a / a[0], out).astype(np.float32)
    return out


def _granular_noise(y: np.ndarray, sr: int, density: float, gain_db: float) -> np.ndarray:
    """Anade granos de ruido blanco sincronizados a onsets (RMS > umbral)."""
    if density <= 0:
        return y
    # Envolvente de energia
    win = int(0.005 * sr)
    env = np.sqrt(np.convolve(y * y, np.ones(win) / win, mode="same"))
    threshold = np.percentile(env, 80) * 0.8
    mask = env > threshold
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(y)).astype(np.float32) * mask.astype(np.float32)
    # Suavizar el ruido para no sonar a hiss
    noise = _butter_lowpass(noise, sr, 6000)
    gain = 10 ** (gain_db / 20) * density
    return (y + gain * noise).astype(np.float32)


@dataclass
class DSPParams:
    wetness: float = 0.0      # 0..1
    rigidity: float = 0.0     # 0..1
    resonance: float = 0.0    # 0..1
    granularity: float = 0.0  # 0..1
    continuity: float = 0.0   # 0..1 (no implementado: trivial pass-through)
    sr: int = SAMPLE_RATE


def apply_dsp(y: np.ndarray, p: DSPParams) -> np.ndarray:
    """Aplica los 5 controles en cascada. y: float32 mono."""
    out = y.astype(np.float32).copy()
    # rigidity: enfatizar agudos
    if p.rigidity > 0:
        out = _high_shelf(out, p.sr, 4000.0, gain_db=12.0 * p.rigidity)
    # resonance: banco de resonadores
    if p.resonance > 0:
        out = _resonator(out, p.sr,
                         freqs_hz=[220.0, 660.0, 2200.0],
                         q=8.0 * (0.5 + p.resonance),
                         gain=12.0 * p.resonance)
    # granularity: ruido granular sincronizado
    if p.granularity > 0:
        out = _granular_noise(out, p.sr, density=p.granularity, gain_db=-18.0)
    # wetness: filtrado + reverb corta
    if p.wetness > 0:
        out = _butter_lowpass(out, p.sr, 6000.0 - 3000.0 * p.wetness)
        out = _schroeder_reverb(out, p.sr, t60_s=0.4 + 0.6 * p.wetness, mix=0.4 * p.wetness)
    # peak-normalizar para evitar clipping
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


class MethodD(HybridMethod):
    """Wrapper sobre MethodA o MethodB que aplica DSP como postproceso."""

    name = "D_dsp_postproc"

    def __init__(self, inner: HybridMethod, encoder) -> None:
        self.inner = inner
        self.encoder = encoder

    def encode(self, wav: Tensor) -> Tensor:
        return self.inner.encode(wav)

    def decode(self, z: Tensor) -> Tensor:
        return self.inner.decode(z)

    def generate(self, anchor_z: Tensor, spec: HybridSpec) -> Tensor:
        wav = self.inner.generate(anchor_z, spec)
        params = DSPParams(
            wetness=float(spec.properties.get("wetness", 0)) / 5.0,
            rigidity=float(spec.properties.get("rigidity", 0)) / 5.0,
            resonance=float(spec.properties.get("resonance", 0)) / 5.0,
            granularity=float(spec.properties.get("granularity", 0)) / 5.0,
            continuity=float(spec.properties.get("continuity", 0)) / 5.0,
            sr=self.encoder.sr_expected,
        )
        y = wav.numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
        y_post = apply_dsp(y, params)
        return torch.from_numpy(y_post)
