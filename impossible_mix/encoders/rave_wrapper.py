"""Unica puerta de acceso al encoder. Cargamos EnCodec 24kHz preentrenado
como encoder primario (decision dia 1, ver data/ENCODER_DECISION.md).

El espacio latente es el de `encoder.encoder(x)` (continuo, dim=128 por
defecto, longitud temporal T' = T/320 a 24 kHz). El metodo A trabaja
sobre la media temporal; el metodo C/D pueden usar la secuencia completa.

Para decoder, EnCodec espera la secuencia temporal completa. Cuando A
manipula la media (vector), reconstruimos la secuencia repitiendo el vector
T' veces y aplicando el decoder; esto pierde dinamica temporal pero
preserva el contenido timbrico. Si la calidad no convence, se prueba RAVE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from impossible_mix.config import (
    ENCODEC_CHECKPOINT_PATH,
    PREFERRED_ENCODER,
    RAVE_CHECKPOINT_PATH,
    SAMPLE_RATE,
)


class EncoderWrapper:
    """Wrapper unificado. Decide en __init__ que backend usar."""

    def __init__(self, kind: str | None = None, device: str = "cpu") -> None:
        self.kind = (kind or PREFERRED_ENCODER).lower()
        self.device = device
        self.sr_expected: int = SAMPLE_RATE
        self.latent_dim: int = -1
        self.t_per_second: float = 0.0
        self._model = None
        self._load()

    def _load(self) -> None:
        if self.kind == "rave" and RAVE_CHECKPOINT_PATH and Path(RAVE_CHECKPOINT_PATH).exists():
            self._load_rave()
        elif self.kind == "encodec":
            self._load_encodec()
        else:
            raise RuntimeError(
                f"No se pudo resolver encoder ({self.kind}). "
                "Define RAVE_CHECKPOINT_PATH en .env o instala el extra [encodec]."
            )

    def _load_rave(self) -> None:
        raise NotImplementedError(
            "Carga de RAVE pendiente: requiere checkpoint del otro ordenador."
        )

    def _load_encodec(self) -> None:
        from encodec import EncodecModel
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        model.eval().to(self.device)
        self._model = model
        self.sr_expected = 24_000
        self.latent_dim = model.encoder.dimension  # 128
        # EnCodec downsampling factor a 24 kHz es 320 -> 75 frames por segundo
        self.t_per_second = self.sr_expected / 320.0

    @torch.no_grad()
    def encode_sequence(self, wav: Tensor, sr_in: int = SAMPLE_RATE) -> Tensor:
        """wav (..., T_in) -> z (dim, T'). Mono asumido (o reduce a mono)."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)               # (1, C, T)
        if wav.shape[1] > 1:
            wav = wav.mean(1, keepdim=True)
        if sr_in != self.sr_expected:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr_in, self.sr_expected)
        wav = wav.to(self.device)
        z = self._model.encoder(wav).squeeze(0)  # (dim, T')
        return z

    def encode_mean(self, wav: Tensor, sr_in: int = SAMPLE_RATE) -> Tensor:
        return self.encode_sequence(wav, sr_in).mean(dim=-1)

    @torch.no_grad()
    def decode_sequence(self, z: Tensor) -> Tensor:
        """z (dim, T') -> wav (T_out,). Devuelve mono float32 en CPU."""
        if z.dim() == 2:
            z = z.unsqueeze(0)  # (1, dim, T')
        z = z.to(self.device)
        wav = self._model.decoder(z).squeeze(0).squeeze(0).cpu()
        return wav

    def decode_from_mean(self, z_mean: Tensor, n_frames: int) -> Tensor:
        """Repite el vector medio T' veces y decodifica. Pierde dinamica
        temporal: usar solo cuando el metodo es agnostico al tiempo (A simple).
        Para A++ y C, decoder se invoca sobre la secuencia ya editada.
        """
        seq = z_mean.unsqueeze(-1).expand(-1, n_frames)  # (dim, T')
        return self.decode_sequence(seq)

    def expected_frames(self, duration_s: float) -> int:
        return int(round(duration_s * self.t_per_second))


_global: EncoderWrapper | None = None


def get_encoder(kind: str | None = None, device: str = "cpu") -> EncoderWrapper:
    """Singleton para evitar cargar el modelo dos veces."""
    global _global
    if _global is None or (kind and _global.kind != kind.lower()):
        _global = EncoderWrapper(kind, device)
    return _global


# Solo expone EnCodec; el nombre 'rave_wrapper' es legacy del plan inicial.
__all__ = ["EncoderWrapper", "get_encoder"]
