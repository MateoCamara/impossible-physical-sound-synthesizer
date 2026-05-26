"""Friction/scrape, differentiable (PyTorch).

Modelo simplificado pero diferenciable de un scrape/drag:

    1. Excitacion: ruido blanco (pre-fijado con seed) modulado por una
       envolvente de velocidad lenta.
    2. Filtro paso-banda parametrizable (cutoff_lo, cutoff_hi).
    3. Banco de resonadores del cuerpo del material (suma de damped
       sinusoides, igual que modal.py).

Parametros aprendibles (cinco escalares):
    surface_hardness  — afecta cutoff del bandpass del exciter
    velocity_mean     — amplitud media de la envolvente
    body_freq_hz      — frecuencia principal del cuerpo del material
    body_t60_s        — decay del cuerpo
    gain              — gain global

El "stick-slip" (saltos discretos del scrape real) se omite en la
version diferenciable: introduce discontinuidades y no es necesario
para recuperar la firma espectral.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


_LN1000 = 6.907755


@dataclass
class FrictionParamsT:
    """Parametros diferenciables del scrape/drag.

    surface_hardness: 0..1, modula la banda del exciter
                      (banda baja ~ fabric, alta ~ metal)
    velocity_mean:    0..1, amplitud media
    body_freq_hz:     frecuencia principal del cuerpo
    body_t60_s:       decay del cuerpo (s)
    gain:             gain global
    """
    surface_hardness: torch.Tensor    # ()
    velocity_mean: torch.Tensor       # ()
    body_freq_hz: torch.Tensor        # ()
    body_t60_s: torch.Tensor          # ()
    gain: torch.Tensor                # ()

    seed: int = 0

    @classmethod
    def physical_init(cls, surface_hardness: float = 0.5,
                       velocity_mean: float = 0.6,
                       body_freq_hz: float = 1200.0,
                       body_t60_s: float = 0.05,
                       gain: float = 0.5,
                       device: str | torch.device = "cpu",
                       requires_grad: bool = True,
                       seed: int = 0) -> "FrictionParamsT":
        def t(v):
            x = torch.tensor(float(v), device=device, dtype=torch.float32)
            return x.requires_grad_(requires_grad) if requires_grad else x
        return cls(
            surface_hardness=t(surface_hardness),
            velocity_mean=t(velocity_mean),
            body_freq_hz=t(body_freq_hz),
            body_t60_s=t(body_t60_s),
            gain=t(gain),
            seed=seed,
        )

    def trainable(self) -> list[torch.Tensor]:
        return [self.surface_hardness, self.velocity_mean,
                self.body_freq_hz, self.body_t60_s, self.gain]

    def clamp_(self):
        with torch.no_grad():
            self.surface_hardness.clamp_(0.0, 1.0)
            self.velocity_mean.clamp_(0.05, 1.5)
            self.body_freq_hz.clamp_(50.0, 12000.0)
            self.body_t60_s.clamp_(1e-3, 2.0)
            self.gain.clamp_(0.01, 2.0)


def _sinc_bandpass_kernel(cutoff_lo_hz: torch.Tensor, cutoff_hi_hz: torch.Tensor,
                           sr: int, n_taps: int = 127) -> torch.Tensor:
    """FIR bandpass kernel diferenciable via formula sinc.

    n_taps debe ser impar. cutoff_lo < cutoff_hi.
    """
    device = cutoff_lo_hz.device
    dtype = cutoff_lo_hz.dtype
    half = (n_taps - 1) // 2
    n = torch.arange(-half, half + 1, device=device, dtype=dtype)
    # Sinc(2 f n / sr) -> lowpass; lowpass(hi) - lowpass(lo) = bandpass
    # Add epsilon to avoid 0/0 at n=0
    pi = torch.pi
    lp_hi = torch.where(
        n == 0,
        2 * cutoff_hi_hz / sr,
        torch.sin(2 * pi * cutoff_hi_hz * n / sr) / (pi * n + 1e-9),
    )
    lp_lo = torch.where(
        n == 0,
        2 * cutoff_lo_hz / sr,
        torch.sin(2 * pi * cutoff_lo_hz * n / sr) / (pi * n + 1e-9),
    )
    bp = lp_hi - lp_lo
    # Hamming window para reducir ripple
    w = 0.54 - 0.46 * torch.cos(2 * pi * torch.arange(n_taps, device=device, dtype=dtype) / (n_taps - 1))
    return (bp * w).to(dtype)


def _make_noise_template(n: int, seed: int, device: str | torch.device,
                          dtype: torch.dtype) -> torch.Tensor:
    """Ruido blanco pre-fijado por seed. NO diferenciable (constante)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


def _make_velocity_envelope(n: int, seed: int, sr: int,
                             device: str | torch.device,
                             dtype: torch.dtype) -> torch.Tensor:
    """Envolvente de velocidad LFO + ruido lento, fija por seed."""
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    raw = torch.randn(n, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    # Filtrar a paso bajo (~6 Hz) para suavizar
    # Aproximacion: usar un kernel exponencial corto
    win_n = max(2, int(0.08 * sr))   # ~80 ms ventana
    win = torch.exp(-torch.linspace(0, 4, win_n, device=device, dtype=dtype))
    win = win / win.sum()
    # Envelope no necesita gradiente (es constante), pero usamos FFT por
    # consistencia y velocidad.
    n_full = n + win_n - 1
    RAW = torch.fft.rfft(raw, n=n_full)
    WIN = torch.fft.rfft(win, n=n_full)
    smooth_full = torch.fft.irfft(RAW * WIN, n=n_full)
    pad = win_n - 1
    smooth = smooth_full[pad // 2 : pad // 2 + n]
    # Normalizar a [0, 1]
    smooth = smooth - smooth.min()
    smooth = smooth / (smooth.max() + 1e-9)
    return smooth


def synth_scrape_diff(p: FrictionParamsT, sr: int, n_samples: int,
                       n_fir_taps: int = 127) -> torch.Tensor:
    """Genera un scrape diferenciable.

    Pipeline:
      1) ruido blanco pre-fijado (no diferenciable; constante por seed)
      2) bandpass FIR diferenciable; cutoffs derivados de surface_hardness
      3) modulado por envolvente de velocidad (pre-fijada por seed)
      4) sumado con la respuesta del cuerpo modal (1 modo: body_freq_hz,
         body_t60_s)
      5) gain global
    """
    device = p.body_freq_hz.device
    dtype = p.body_freq_hz.dtype

    # 1) Ruido
    noise = _make_noise_template(n_samples, p.seed, device, dtype)

    # 2) Cutoffs diferenciables del bandpass
    # surface_hardness 0 -> banda baja (300-1500 Hz), 1 -> banda alta (2k-8k Hz)
    cl = 300.0 + 1700.0 * p.surface_hardness
    ch = 2000.0 + 6000.0 * p.surface_hardness
    cl_safe = torch.clamp(cl, min=50.0)
    # torch.clamp no admite Tensor en min y float en max simultaneamente;
    # combinamos con torch.maximum/minimum.
    ch_min_floor = cl_safe + 100.0
    ch_max_floor = torch.tensor(sr / 2 - 200, device=ch.device, dtype=ch.dtype)
    ch_safe = torch.maximum(ch, ch_min_floor)
    ch_safe = torch.minimum(ch_safe, ch_max_floor)
    bp_kernel = _sinc_bandpass_kernel(cl_safe, ch_safe, sr, n_taps=n_fir_taps)
    # FFT convolution (mucho mas rapido bajo autograd que F.conv1d)
    n_full = n_samples + n_fir_taps - 1
    NOISE = torch.fft.rfft(noise, n=n_full)
    KER = torch.fft.rfft(bp_kernel, n=n_full)
    bp_full = torch.fft.irfft(NOISE * KER, n=n_full)
    # center-trim para mantener fase
    pad = n_fir_taps - 1
    bp_noise = bp_full[pad // 2 : pad // 2 + n_samples]

    # 3) Envolvente de velocidad pre-fija + velocity_mean
    vel_env = _make_velocity_envelope(n_samples, p.seed, sr, device, dtype)
    vel_env = vel_env * p.velocity_mean
    excited = bp_noise * vel_env

    # 4) Body modal: 1 modo (suma damped sin a freq body_freq_hz)
    t = torch.arange(n_samples, device=device, dtype=dtype) / sr
    body_tau = torch.clamp(p.body_t60_s, min=1e-3) / _LN1000
    # Convolucion exc * exp-sinusoide: equivalente a filtrar exc por un
    # resonador. Implementacion economica: multiplicar el ruido excitado
    # por sin + decay como en modal, escalando por la energia del exciter.
    body_osc = torch.sin(2 * torch.pi * p.body_freq_hz * t)
    body_decay = torch.exp(-t / body_tau)
    # Mezclar exciter con la respuesta del cuerpo. Para que el cuerpo
    # responda de forma realista, sumamos una version "filtrada" del ruido
    # multiplicada por el oscilador del cuerpo.
    body_signal = excited * body_osc * 0.5 + bp_noise * body_decay * body_osc * 0.3

    out = (excited + body_signal) * p.gain
    return out
