"""Modal impact, differentiable (PyTorch).

Implementacion como suma de sinusoides amortiguadas en forma cerrada:

    y(t) = sum_k g_k * exp(-t / tau_k) * sin(2*pi*f_k*t)    para t >= 0

donde tau_k = t60_k / 6.91 y t60_k es el tiempo de decaimiento a -60 dB
del modo k. Cada (f_k, t60_k, g_k) es un tensor con requires_grad=True
si queremos optimizar via gradiente.

Para incluir un excitador de duracion no-cero, convolucionamos la salida
con un exciter corto (un tensor de forma fija o aprendida) usando
`F.conv1d`, que es nativamente diferenciable.

Ventaja sobre IIR scipy.signal: completamente vectorizable, sin recursion,
gradientes limpios.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class ModalParamsT:
    """Parametros modales diferenciables.

    - freqs_hz: (K,) tensor — frecuencias de cada modo (Hz)
    - t60s_s:   (K,) tensor — tiempo de decaimiento a -60 dB de cada modo (s)
    - gains:    (K,) tensor — gain de cada modo (sera renormalizado por sum)
    - exciter:  (E,) tensor opcional — forma del exciter en samples; si None,
                se usa un impulso de Dirac (1 sample).
    """
    freqs_hz: torch.Tensor
    t60s_s: torch.Tensor
    gains: torch.Tensor
    exciter: torch.Tensor | None = None

    @classmethod
    def from_lists(cls, freqs_hz: list[float], t60s_s: list[float],
                   gains: list[float], requires_grad: bool = True,
                   device: str | torch.device = "cpu") -> "ModalParamsT":
        return cls(
            freqs_hz=torch.tensor(freqs_hz, device=device, requires_grad=requires_grad),
            t60s_s=torch.tensor(t60s_s, device=device, requires_grad=requires_grad),
            gains=torch.tensor(gains, device=device, requires_grad=requires_grad),
        )

    def trainable(self) -> list[torch.Tensor]:
        ts = [self.freqs_hz, self.t60s_s, self.gains]
        if self.exciter is not None and self.exciter.requires_grad:
            ts.append(self.exciter)
        return [t for t in ts if t.requires_grad]


def synth_modal_impact_diff(p: ModalParamsT, sr: int, n_samples: int,
                             impact_time_s: float = 0.0,
                             clamp_min_t60_s: float = 1e-3) -> torch.Tensor:
    """Genera un golpe modal diferenciable.

    p.freqs_hz, p.t60s_s, p.gains deben ser tensors 1-D de la misma longitud K.

    Returns: tensor 1-D float32 de longitud n_samples.
    """
    device = p.freqs_hz.device
    dtype = p.freqs_hz.dtype
    K = p.freqs_hz.shape[0]
    assert p.t60s_s.shape == (K,) and p.gains.shape == (K,)

    t = torch.arange(n_samples, device=device, dtype=dtype) / sr  # (N,)

    # Tau y dampings; clamp para evitar log(0) o decays infinitos
    t60_safe = torch.clamp(p.t60s_s, min=clamp_min_t60_s)
    tau = t60_safe / 6.907755  # ln(1000)
    # decay shape: (K, N)
    decay = torch.exp(-t.unsqueeze(0) / tau.unsqueeze(1))
    # oscilador shape: (K, N)
    osc = torch.sin(2 * torch.pi * p.freqs_hz.unsqueeze(1) * t.unsqueeze(0))
    # peso por modo, normalizado a sum=1 para estabilidad de optimizacion
    gains_n = torch.softmax(p.gains, dim=0)
    y = (gains_n.unsqueeze(1) * decay * osc).sum(dim=0)  # (N,)

    # Convolucionar con exciter si esta definido
    if p.exciter is not None and p.exciter.numel() > 1:
        # F.conv1d espera (batch, ch, length) tanto en input como en weight.
        exc = p.exciter.flip(0).unsqueeze(0).unsqueeze(0)  # (1,1,E)
        y_in = y.unsqueeze(0).unsqueeze(0)                  # (1,1,N)
        # padding mode "same" manual: pad a la izquierda exc.numel()-1
        pad = p.exciter.numel() - 1
        y_padded = F.pad(y_in, (pad, 0))
        y = F.conv1d(y_padded, exc).squeeze(0).squeeze(0)
        # Truncar a n_samples
        y = y[: n_samples]

    # Aplicar offset de tiempo de impacto: padear al inicio con ceros
    if impact_time_s > 0:
        start = int(impact_time_s * sr)
        if start > 0:
            y = torch.cat([torch.zeros(start, device=device, dtype=dtype),
                            y[: n_samples - start]])

    return y
