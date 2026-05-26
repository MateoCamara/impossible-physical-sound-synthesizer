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
from impossible_mix.physics.diff.modal import ModalParamsT, synth_modal_impact_diff


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


@dataclass
class ModalFitResult:
    params: ModalParamsT
    final_pred: torch.Tensor
    loss_history: list[float]
    final_loss: float


def _init_modal_params(n_modes: int, sr: int,
                        device: str | torch.device = "cpu",
                        f_low: float = 200.0, f_high: float = 6000.0,
                        t60_init_s: float = 0.5) -> ModalParamsT:
    """Inicializa modal con freqs log-espaciados, t60s iguales, gains uniformes."""
    freqs = torch.logspace(
        torch.log10(torch.tensor(f_low)).item(),
        torch.log10(torch.tensor(min(f_high, sr / 2 - 200))).item(),
        n_modes, device=device, dtype=torch.float32,
    ).requires_grad_(True)
    t60s = torch.full((n_modes,), t60_init_s, device=device,
                      dtype=torch.float32).requires_grad_(True)
    gains = torch.zeros(n_modes, device=device, dtype=torch.float32).requires_grad_(True)
    return ModalParamsT(freqs_hz=freqs, t60s_s=t60s, gains=gains)


def init_modal_from_target(target_wav: torch.Tensor, sr: int,
                            n_modes: int = 6, n_fft: int = 8192,
                            f_low: float = 80.0, f_high: float = 12000.0,
                            t60_init_s: float = 0.5,
                            min_peak_separation_hz: float = 80.0) -> ModalParamsT:
    """Inicializa freqs en los K picos espectrales mas prominentes del target.

    Esto convierte el problema "search blind" en "refinar local" que es lo
    que hace que el modal fitting converja en pocas iteraciones. Tras la
    inicializacion, freqs/t60s/gains siguen siendo entrenables.
    """
    device = target_wav.device
    # Magnitude spectrum de un FFT grande (resolucion fina)
    spec = torch.fft.rfft(target_wav, n=n_fft)
    mag = spec.abs()
    freqs_axis = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(device)
    # Mascara: solo bins en [f_low, f_high]
    band = (freqs_axis >= f_low) & (freqs_axis <= min(f_high, sr / 2 - 200))
    mag_masked = mag.clone()
    mag_masked[~band] = 0
    # Encontrar K picos locales separados por al menos min_peak_separation_hz
    bin_width = freqs_axis[1] - freqs_axis[0]
    min_sep_bins = max(1, int(min_peak_separation_hz / bin_width.item()))
    selected: list[int] = []
    for _ in range(n_modes):
        if mag_masked.max() <= 0:
            break
        idx = int(mag_masked.argmax())
        selected.append(idx)
        # Suprimir vecindad
        lo = max(0, idx - min_sep_bins)
        hi = min(mag_masked.numel(), idx + min_sep_bins + 1)
        mag_masked[lo:hi] = 0
    if len(selected) < n_modes:
        # Rellenar con un log-spacing si no hay suficientes picos detectados
        missing = n_modes - len(selected)
        extra = torch.logspace(
            torch.log10(torch.tensor(f_low * 2)).item(),
            torch.log10(torch.tensor(min(f_high, sr / 2 - 200))).item(),
            missing, device=device, dtype=torch.float32,
        )
        peak_freqs = freqs_axis[selected].tolist() + extra.tolist()
    else:
        peak_freqs = freqs_axis[selected].tolist()
    freqs = torch.tensor(peak_freqs, device=device,
                          dtype=torch.float32).requires_grad_(True)
    t60s = torch.full((n_modes,), t60_init_s, device=device,
                      dtype=torch.float32).requires_grad_(True)
    # Inicializar gains proporcionales al log de las magnitudes detectadas
    if len(selected) > 0:
        peak_mags = mag[torch.tensor(selected, device=device)]
        # Normalizar y pasar a log para que softmax produzca pesos sensatos
        init_gains = torch.log(peak_mags + 1e-6).detach().clone()
        if len(init_gains) < n_modes:
            init_gains = torch.cat([
                init_gains,
                torch.full((n_modes - len(init_gains),), init_gains.mean().item(),
                            device=device, dtype=torch.float32),
            ])
        gains = init_gains.requires_grad_(True)
    else:
        gains = torch.zeros(n_modes, device=device,
                             dtype=torch.float32).requires_grad_(True)
    return ModalParamsT(freqs_hz=freqs, t60s_s=t60s, gains=gains)


def _clamp_modal_(p: ModalParamsT, sr: int) -> None:
    """Mantiene los parametros en rangos fisicos (in-place)."""
    with torch.no_grad():
        p.freqs_hz.clamp_(20.0, sr / 2 - 200)
        p.t60s_s.clamp_(1e-3, 5.0)
        # gains se softmax-an dentro de synth_modal_impact_diff, sin clamp


def fit_modal_impact(
    target_wav: torch.Tensor,
    sr: int,
    n_modes: int = 6,
    initial: ModalParamsT | None = None,
    n_iters: int = 250,
    lr: float = 3e-2,
    log_every: int = 30,
    verbose: bool = True,
    init_from_target: bool = True,
) -> ModalFitResult:
    """Ajusta ModalParamsT (freqs, t60s, gains) por gradiente para reproducir
    target_wav.

    n_modes: K de modos a usar.
    initial:  ModalParamsT preconfigurado con requires_grad=True. Si None y
              init_from_target=True (default), las freqs se inicializan en
              los K picos espectrales mas prominentes del target — esto es
              critico para converger en pocas iteraciones.
              Si init_from_target=False, se usa log-spacing uniforme.

    Returns: ModalFitResult con parametros ajustados, audio final, y log.
    """
    if initial is None:
        if init_from_target:
            initial = init_modal_from_target(target_wav, sr, n_modes=n_modes)
        else:
            initial = _init_modal_params(n_modes, sr, device=target_wav.device)
    p = initial
    n_samples = target_wav.shape[0]

    trainable = p.trainable()
    optim = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_iters,
                                                            eta_min=lr / 10)
    history: list[float] = []

    for it in range(n_iters):
        optim.zero_grad()
        pred = synth_modal_impact_diff(p, sr, n_samples)
        loss = multi_resolution_stft_loss(pred, target_wav)
        loss.backward()
        optim.step()
        scheduler.step()
        _clamp_modal_(p, sr)
        history.append(float(loss.detach()))
        if verbose and (it % log_every == 0 or it == n_iters - 1):
            top3_freqs = sorted(p.freqs_hz.detach().tolist())[:3]
            top3_str = ", ".join(f"{f:.0f}" for f in top3_freqs)
            print(f"  iter {it:4d}  loss={history[-1]:.4f}  "
                   f"freqs[:3]=[{top3_str}] Hz")

    with torch.no_grad():
        final_pred = synth_modal_impact_diff(p, sr, n_samples)
        final_loss = float(multi_resolution_stft_loss(final_pred, target_wav))
    return ModalFitResult(params=p, final_pred=final_pred,
                           loss_history=history, final_loss=final_loss)


def fit_modal_impact_multistart(
    target_wav: torch.Tensor,
    sr: int,
    n_modes_candidates: tuple[int, ...] = (4, 6, 8),
    **kwargs,
) -> ModalFitResult:
    """Multi-start sobre el numero de modos K: prueba varios K y devuelve
    el mejor por loss final. Mas lento pero util cuando no sabes a priori
    cuantos modos tiene el target.
    """
    best: ModalFitResult | None = None
    for K in n_modes_candidates:
        if kwargs.get("verbose", True):
            print(f"\n>>> Start: n_modes={K}")
        res = fit_modal_impact(target_wav, sr, n_modes=K, **kwargs)
        if best is None or res.final_loss < best.final_loss:
            best = res
    return best  # type: ignore[return-value]


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
