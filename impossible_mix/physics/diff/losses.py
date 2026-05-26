"""Spectral losses for parameter fitting.

Multi-resolution STFT (Yamamoto et al. 2020) — el estandar de facto en
DDSP. Computa STFTs a distintas resoluciones y combina L1 sobre la
log-magnitud + L2 (Frobenius) sobre la magnitud lineal:

    L = sum_i [ ||log|S_i(pred)| - log|S_i(target)|||_1
              + ||(|S_i(pred)| - |S_i(target)|)||_F ]

Mas robusta a desalineamientos temporales que un MSE en tiempo, y mas
fiel perceptualmente que solo log-mag.
"""
from __future__ import annotations

import torch


_DEFAULT_FFT_SIZES = (512, 1024, 2048)
_DEFAULT_HOPS = (128, 256, 512)
_DEFAULT_WINS = (480, 960, 1920)


def _stft_mag(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int,
              window: torch.Tensor) -> torch.Tensor:
    """STFT magnitud, sin gradient-killing edge cases."""
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window,
                      center=True, return_complex=True)
    return spec.abs()


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: tuple[int, ...] = _DEFAULT_FFT_SIZES,
    hop_lengths: tuple[int, ...] = _DEFAULT_HOPS,
    win_lengths: tuple[int, ...] = _DEFAULT_WINS,
    log_eps: float = 1e-7,
    lin_weight: float = 1.0,
    log_weight: float = 1.0,
) -> torch.Tensor:
    """Sum of L1 log-magnitude + L2 spectral-convergence across resolutions.

    pred, target: 1-D tensors of equal length. Returns a scalar loss.
    """
    assert pred.shape == target.shape, f"shape mismatch {pred.shape} vs {target.shape}"
    total = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for n_fft, hop, win in zip(fft_sizes, hop_lengths, win_lengths):
        win_tensor = torch.hann_window(win, device=pred.device, dtype=pred.dtype)
        S_pred = _stft_mag(pred, n_fft, hop, win, win_tensor)
        S_tgt = _stft_mag(target, n_fft, hop, win, win_tensor)
        # Log-magnitude L1
        log_loss = (torch.log(S_pred + log_eps) - torch.log(S_tgt + log_eps)).abs().mean()
        # Spectral convergence: ||S_pred - S_tgt||_F / ||S_tgt||_F
        sc = ((S_pred - S_tgt).pow(2).sum().sqrt() /
              (S_tgt.pow(2).sum().sqrt() + log_eps))
        total = total + log_weight * log_loss + lin_weight * sc
    return total
