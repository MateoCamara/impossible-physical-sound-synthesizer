"""Inverse fitting loop: given a target waveform, optimize the physical
parameters of a drip event (or modal impact) to reproduce it via
gradient descent on a spectral loss.

This is the "DDSP-style" use case: no neural network, just the
parametric engine made differentiable + a standard optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from impossible_mix.physics.diff.droplet import DripParamsT, synth_drip_event_diff
from impossible_mix.physics.diff.losses import multi_resolution_stft_loss


@dataclass
class FitResult:
    params: DripParamsT
    final_pred: torch.Tensor
    loss_history: list[float]
    final_loss: float


def fit_drip_event(
    target_wav: torch.Tensor,
    sr: int,
    initial: DripParamsT | None = None,
    n_iters: int = 300,
    lr: float = 1e-2,
    log_every: int = 20,
    verbose: bool = True,
    freeze_surface: bool = False,
) -> FitResult:
    """Ajusta DripParamsT por gradiente para reproducir target_wav.

    target_wav: tensor 1-D float32 con el audio objetivo (mono).
    initial:    DripParamsT con requires_grad=True. Si None, se crea uno
                 inicializado a un drip "canonico" (radius=2 mm, viscosity=0,
                 ceramic surface).
    n_iters, lr: hiperparametros del Adam. Se aplica un schedule cosine
                 decay automatico de lr a lr/10 sobre n_iters.
    freeze_surface: si True, surface_modes/t60s/gains no se entrenan (solo
                 radius, viscosity, chirp_amp, decay_scale, capillary). Reduce
                 dimensiones y mejora convergencia cuando la superficie ya
                 esta calibrada.

    Returns: FitResult con parametros ajustados, audio final, y log de loss.
    """
    if initial is None:
        initial = DripParamsT.physical_init(radius_mm=2.0, viscosity=0.0,
                                              device=target_wav.device,
                                              requires_grad=True)
    p = initial
    n_samples = target_wav.shape[0]

    if freeze_surface:
        # Desactivar gradiente sobre los parametros de superficie
        with torch.no_grad():
            for name in ("surface_modes_hz", "surface_t60s_s", "surface_gains"):
                t = getattr(p, name)
                t.requires_grad_(False)

    trainable = p.trainable()
    optim = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_iters,
                                                            eta_min=lr / 10)
    history: list[float] = []

    for it in range(n_iters):
        optim.zero_grad()
        pred = synth_drip_event_diff(p, sr, n_samples)
        loss = multi_resolution_stft_loss(pred, target_wav)
        loss.backward()
        optim.step()
        scheduler.step()
        p.clamp_()
        history.append(float(loss.detach()))
        if verbose and (it % log_every == 0 or it == n_iters - 1):
            r = float(p.radius_mm.detach())
            v = float(p.viscosity.detach())
            print(f"  iter {it:4d}  loss={history[-1]:.4f}  "
                   f"radius={r:.2f}mm  visc={v:.3f}  lr={optim.param_groups[0]['lr']:.4f}")

    with torch.no_grad():
        final_pred = synth_drip_event_diff(p, sr, n_samples)
        final_loss = float(multi_resolution_stft_loss(final_pred, target_wav))
    return FitResult(params=p, final_pred=final_pred,
                      loss_history=history, final_loss=final_loss)


def fit_drip_event_multistart(
    target_wav: torch.Tensor,
    sr: int,
    starts: list[tuple[float, float]] | None = None,
    **kwargs,
) -> FitResult:
    """Multi-start: prueba varias inicializaciones de (radius, viscosity)
    y devuelve el mejor resultado por loss final. Mas robusto que un solo
    init en espacios no-convexos como el chirp Minnaert.
    """
    if starts is None:
        starts = [(0.8, 0.0), (2.0, 0.0), (3.5, 0.3), (5.0, 0.6)]
    best: FitResult | None = None
    for r0, v0 in starts:
        init = DripParamsT.physical_init(radius_mm=r0, viscosity=v0,
                                           device=target_wav.device,
                                           requires_grad=True)
        if kwargs.get("verbose", True):
            print(f"\n>>> Start: radius={r0:.1f} mm  viscosity={v0:.2f}")
        res = fit_drip_event(target_wav, sr, initial=init, **kwargs)
        if best is None or res.final_loss < best.final_loss:
            best = res
    return best  # type: ignore[return-value]
