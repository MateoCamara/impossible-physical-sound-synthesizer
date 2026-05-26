"""Drip event differentiable (PyTorch).

Forma diferenciable del modelo Minnaert + envelope + surface modal tail:

    f(t) = f_start * (f_end / f_start) ** (t / chirp_dur)    para t < chirp_dur
    chirp(t) = sin(2*pi * cumsum(f(t)) / sr)
    env(t)   = (cosine attack) * exp(-decay * (t / chirp_dur))
    surface_tail(t) = sum_k g_k * exp(-t / tau_k) * sin(2*pi*f_k*t)

Todos los parametros expuestos son tensores diferenciables. El "click"
inicial (ruido) es no-diferenciable y se incluye como constante (seed-
fijado) si se quiere realismo; aqui lo dejamos opcional.

Parametros fisicos clave para el fitting:
    radius_mm -> determina f_start y f_end (Minnaert)
    viscosity -> alarga chirp_dur y acorta decay
    surface_modes_hz, surface_t60s_s, surface_gains -> body del material
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


_LN1000 = 6.907755


@dataclass
class DripParamsT:
    """Parametros diferenciables del drip event.

    Todos son tensors 0-D (escalares) o 1-D (vectores) y pueden tener
    requires_grad=True para optimizacion por gradiente.
    """
    radius_mm: torch.Tensor          # ()  — droplet radius in millimeters
    viscosity: torch.Tensor          # ()  — 0..1
    surface_modes_hz: torch.Tensor   # (M,) — modal freqs of the surface
    surface_t60s_s: torch.Tensor     # (M,) — t60 of each surface mode (s)
    surface_gains: torch.Tensor      # (M,) — gains (will be softmax'd)
    chirp_amp: torch.Tensor          # ()  — gain of the bubble chirp
    decay_scale: torch.Tensor        # ()  — multiplies the envelope decay
    capillary_ringing: torch.Tensor  # ()  — 0..1 weight of capillary layer

    @classmethod
    def physical_init(cls, radius_mm: float = 2.0, viscosity: float = 0.0,
                       device: str | torch.device = "cpu",
                       requires_grad: bool = True) -> "DripParamsT":
        """Crea parametros sensatos para un drip de 2 mm sobre ceramic."""
        def t(v, shape=()):
            x = torch.tensor(v, device=device, dtype=torch.float32)
            return x.requires_grad_(requires_grad) if requires_grad else x
        # Surface profile aproximado de "ceramic"
        return cls(
            radius_mm=t(radius_mm),
            viscosity=t(viscosity),
            surface_modes_hz=t([1100.0, 2400.0, 4800.0, 7200.0]),
            surface_t60s_s=t([0.35, 0.18, 0.10, 0.06]),
            surface_gains=t([0.35, 0.30, 0.20, 0.15]),
            chirp_amp=t(0.7),
            decay_scale=t(1.0),
            capillary_ringing=t(0.5),
        )

    def trainable(self) -> list[torch.Tensor]:
        ts = [self.radius_mm, self.viscosity, self.surface_modes_hz,
              self.surface_t60s_s, self.surface_gains, self.chirp_amp,
              self.decay_scale, self.capillary_ringing]
        return [t for t in ts if t.requires_grad]

    def clamp_(self):
        """Mantiene los parametros en rangos fisicamente razonables (in-place)."""
        with torch.no_grad():
            self.radius_mm.clamp_(0.3, 8.0)
            self.viscosity.clamp_(0.0, 1.0)
            self.surface_modes_hz.clamp_(20.0, 16000.0)
            self.surface_t60s_s.clamp_(1e-3, 5.0)
            self.chirp_amp.clamp_(0.05, 1.5)
            self.decay_scale.clamp_(0.2, 5.0)
            self.capillary_ringing.clamp_(0.0, 1.0)


def _minnaert_freq(radius_mm: torch.Tensor) -> torch.Tensor:
    """f_M = 3.26 / r (m) en Hz."""
    return 3.26 / (radius_mm * 1e-3)


def synth_drip_event_diff(p: DripParamsT, sr: int, n_samples: int) -> torch.Tensor:
    """Genera un drip event diferenciable.

    Returns: tensor 1-D float32 de longitud n_samples.
    """
    device = p.radius_mm.device
    dtype = p.radius_mm.dtype
    t = torch.arange(n_samples, device=device, dtype=dtype) / sr  # (N,)

    f_minnaert = _minnaert_freq(p.radius_mm)
    f_start = f_minnaert * 0.45
    f_end = f_minnaert * 1.6
    # Duracion del chirp (s): crece con radius y viscosity
    chirp_dur_s = (15.0 + 8.0 * p.radius_mm) * (1.0 + 1.5 * p.viscosity) / 1000.0
    # Decay envelope (s)
    decay_dur_s = (50.0 + 30.0 * p.radius_mm) * (1.0 - 0.6 * p.viscosity) / 1000.0
    decay_dur_s = torch.clamp(decay_dur_s, min=1e-3)

    # ---- 1. Bubble chirp ----
    # Smooth time-normalization: t_norm = t / chirp_dur_s saturado a 1
    t_norm = torch.clamp(t / torch.clamp(chirp_dur_s, min=1e-4), max=1.0)
    # f(t) = f_start * (f_end/f_start) ** t_norm  (ramp exponencial)
    ratio = torch.clamp(f_end / torch.clamp(f_start, min=1.0), min=1.001)
    f_t = f_start * (ratio ** t_norm)
    # Fase acumulada (integral de la frecuencia)
    phase = 2 * torch.pi * torch.cumsum(f_t, dim=0) / sr
    chirp = torch.sin(phase)
    # Envelope: ataque suave (raised cosine corto) + decay exponencial
    attack_n = max(2, int(0.001 * sr))
    attack_env = torch.zeros(n_samples, device=device, dtype=dtype)
    attack_env[:attack_n] = 0.5 * (1 - torch.cos(torch.pi * torch.arange(attack_n,
                                                                          device=device,
                                                                          dtype=dtype) / attack_n))
    attack_env[attack_n:] = 1.0
    # Decay despues del attack
    decay_rate = p.decay_scale * _LN1000 / decay_dur_s  # exp(-rate * t) ~= -60 dB at decay_dur
    decay_env = torch.exp(-decay_rate * t)
    env = attack_env * decay_env
    chirp_signal = p.chirp_amp * chirp * env

    # ---- 2. Surface modal tail (banco modal del material) ----
    M = p.surface_modes_hz.shape[0]
    # Cada modo: g_k * sin(2*pi*f_k*t) * exp(-t / tau_k)
    tau = p.surface_t60s_s / _LN1000
    decay_surf = torch.exp(-t.unsqueeze(0) / tau.unsqueeze(1))            # (M, N)
    osc_surf = torch.sin(2 * torch.pi * p.surface_modes_hz.unsqueeze(1) * t.unsqueeze(0))
    gains_n = torch.softmax(p.surface_gains, dim=0)
    surface_tail = (gains_n.unsqueeze(1) * decay_surf * osc_surf).sum(dim=0)
    # Excitacion modal: modulada por una envolvente muy corta al inicio
    surf_exc_env = torch.exp(-25.0 * t)  # ~40 ms efectivos
    surface_tail = surface_tail * surf_exc_env * 0.5

    # ---- 3. Capillary ringing (oscilacion alta, corta) ----
    f_cap = f_end * 2.2  # ~2x Minnaert
    cap_decay = torch.exp(-200.0 * t)   # ~5 ms efectivos
    cap_signal = (p.capillary_ringing * 0.25 *
                   torch.sin(2 * torch.pi * f_cap * t) * cap_decay)

    out = chirp_signal + surface_tail + cap_signal
    return out
