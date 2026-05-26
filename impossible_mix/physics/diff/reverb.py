"""Convolution reverb, differentiable (PyTorch).

Esto es el caso "trivial" del DDSP: la convolucion ya es nativamente
diferenciable via F.conv1d, asi que solo necesitamos:

  1. Una IR aprendible (tensor de N samples con requires_grad=True).
  2. La funcion synth_reverb_diff(dry, ir_params) = dry conv ir, con
     wet/dry mix opcional.
  3. Un init razonable (ruido decreciente exponencial con t60 dado).

Caso de uso clave para el paper: dado un dry conocido y un wet target,
recuperar la IR del espacio (deconvolution by gradient descent). Equivale
a "esta gota cae en un espacio reverberante, ¿cual es la firma de ese
espacio?".

Regularizacion implicita: el init es decay exponencial, y la loss STFT
+ la magnitud baja del ruido inicial actuan como prior de "IR fisica".
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


_LN1000 = 6.907755


@dataclass
class IRParamsT:
    """Impulse response aprendible (PyTorch).

    ir_samples: tensor 1-D float con la IR. Si requires_grad=True, los
                samples son entrenables.
    sr:         sample rate de la IR.
    """
    ir_samples: torch.Tensor
    sr: int = 44_100

    @classmethod
    def from_exp_decay(cls, n_samples: int, sr: int = 44_100,
                        t60_s: float = 1.0, seed: int = 0,
                        device: str | torch.device = "cpu",
                        requires_grad: bool = True) -> "IRParamsT":
        """Init con ruido blanco × decay exponencial de t60 dado."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(n_samples, generator=g, dtype=torch.float32).to(device)
        t = torch.arange(n_samples, device=device, dtype=torch.float32) / sr
        decay_factor = _LN1000 / max(t60_s, 1e-3)
        env = torch.exp(-decay_factor * t)
        ir = noise * env * 0.1
        if requires_grad:
            ir = ir.requires_grad_(True)
        return cls(ir_samples=ir, sr=sr)

    def trainable(self) -> list[torch.Tensor]:
        return [self.ir_samples] if self.ir_samples.requires_grad else []

    def clamp_(self, max_abs: float = 1.0):
        with torch.no_grad():
            self.ir_samples.clamp_(-max_abs, max_abs)


def synth_reverb_diff(dry: torch.Tensor, ir_params: IRParamsT,
                       mix: float = 1.0) -> torch.Tensor:
    """Aplica reverb por convolucion diferenciable.

    dry: tensor 1-D mono.
    mix: 0.0 = solo dry; 1.0 = solo wet; intermedios = blend.

    Returns: tensor de longitud len(dry) (la cola se trunca a la longitud
    del input para evitar shifts).
    """
    n = dry.shape[-1]
    ir = ir_params.ir_samples
    n_ir = ir.shape[-1]
    # FFT convolution: O(N log N) vs O(N*K) de F.conv1d.
    # Para IRs largas (>1000 samples) es ordenes de magnitud mas rapido
    # bajo autograd. Una IR de 35k samples × dry de 26k: ~50 ms vs ~5 s.
    n_full = n + n_ir - 1
    DRY = torch.fft.rfft(dry, n=n_full)
    IR = torch.fft.rfft(ir, n=n_full)
    wet_full = torch.fft.irfft(DRY * IR, n=n_full)
    wet = wet_full[:n]
    out = (1 - mix) * dry + mix * wet
    return out
