"""Inverse fitting demo for rolling_droplet: target → recovered params.

Generates a synthetic rolling droplet with known ground-truth physical
parameters, then fits the differentiable model to recover them via
gradient descent on a multi-resolution STFT loss.

Reports:
  - Loss curve (initial → final)
  - Recovered vs ground-truth for radius, viscosity, contact angle,
    surface tension
  - Saves before/after waveforms + spectrograms as PNG

Run:
  python scripts/31_inverse_rolling_droplet_fitting.py
  python scripts/31_inverse_rolling_droplet_fitting.py --target path/to/real_droplet.wav

This is the DDSP-style use case: no neural network, just the
parametric physics engine made differentiable + a standard optimizer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.physics.diff import (
    RollingDropletParamsT,
    synth_rolling_droplet_diff,
    fit_rolling_droplet,
)
from impossible_mix.utils import load_wav_mono, save_fit_report, save_wav, seed_everything


def load_wav(path: str, target_sr: int = 44_100) -> torch.Tensor:
    """Load a mono wav as a float32 torch tensor at target_sr."""
    return torch.from_numpy(load_wav_mono(Path(path), target_sr))


def make_synthetic_target(
    sr: int = 44_100,
    duration_s: float = 1.5,
    radius_mm: float = 3.5,
    viscosity: float = 0.05,
    contact_angle_deg: float = 150.0,
    surface_tension_n_m: float = 0.072,
    seed: int = 42,
) -> tuple[torch.Tensor, dict]:
    """Generate a target with known parameters for ground-truth recovery."""
    p = RollingDropletParamsT.physical_init(
        radius_mm=radius_mm, viscosity=viscosity,
        roll_velocity_hz=14.0, path_roughness=0.35,
        contact_angle_deg=contact_angle_deg,
        surface_tension_n_m=surface_tension_n_m,
        sr=sr, duration_s=duration_s, seed=seed,
        requires_grad=False,
    )
    with torch.no_grad():
        target = synth_rolling_droplet_diff(p)
    gt = {
        "radius_mm": radius_mm, "viscosity": viscosity,
        "contact_angle_deg": contact_angle_deg,
        "surface_tension_n_m": surface_tension_n_m,
    }
    return target.detach(), gt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=str, default=None,
                        help="WAV file to fit. If absent, generates a synthetic target.")
    parser.add_argument("--out-dir", type=str, default="results/diff_rolling/")
    parser.add_argument("--n-iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the fitting model's random structure.")
    parser.add_argument("--freeze-surface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multistart", type=int, default=1,
                        help="Number of random inits; pick best loss.")
    args = parser.parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sr = 44_100
    rng = np.random.default_rng(args.seed)

    # 1) Build target
    if args.target:
        print(f"Loading target wav: {args.target}")
        target = load_wav(args.target, target_sr=sr)
        gt = None
    else:
        print("Generating synthetic target (radius=3.5mm, viscosity=0.05, θ=150°)")
        target, gt = make_synthetic_target(sr=sr, duration_s=1.5)
    n = target.shape[0]
    print(f"Target: {n} samples ({n/sr:.2f}s), peak={target.abs().max():.3f}")
    save_wav(str(out_dir / "target.wav"), target.numpy(), sr=sr)

    # 2) Fit
    best_result = None
    for k in range(args.multistart):
        print(f"\n=== Multistart {k+1}/{args.multistart} ===")
        # Vary initial radius/contact_angle for different starts
        r_init = 2.5 if k == 0 else (1.5 + 4.0 * rng.random())
        theta_init = 110.0 if k == 0 else (40 + 120 * rng.random())
        initial = RollingDropletParamsT.physical_init(
            radius_mm=r_init, viscosity=0.0,
            contact_angle_deg=theta_init, surface_tension_n_m=0.072,
            sr=sr, duration_s=n / sr, seed=args.seed + k,
            requires_grad=True,
        )
        print(f"  init: r={r_init:.2f}mm  θ={theta_init:.0f}°")
        result = fit_rolling_droplet(
            target_wav=target, sr=sr,
            initial=initial, seed=args.seed + k,
            n_iters=args.n_iters, lr=args.lr,
            log_every=max(10, args.n_iters // 10),
            verbose=True,
            freeze_surface=args.freeze_surface,
        )
        if best_result is None or result.final_loss < best_result.final_loss:
            best_result = result
            print(f"  ✓ new best: loss={result.final_loss:.4f}")

    result = best_result

    # 3) Report
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Loss: {result.loss_history[0]:.4f} → {result.final_loss:.4f}")
    print()
    recovered = {
        "radius_mm": float(result.params.radius_mm.detach()),
        "viscosity": float(result.params.viscosity.detach()),
        "contact_angle_deg": float(result.params.contact_angle_deg.detach()),
        "surface_tension_n_m": float(result.params.surface_tension_n_m.detach()),
    }
    if gt is None:
        print("Recovered parameters (no ground truth available):")
        for k, v in recovered.items():
            print(f"  {k:25s}  {v:.4f}")
    else:
        print(f"  {'param':22s}  {'GT':>10s}   {'recovered':>10s}   {'err':>10s}")
        for k in recovered:
            gt_v, rec_v = gt[k], recovered[k]
            err = abs(rec_v - gt_v)
            err_pct = err / max(abs(gt_v), 1e-6) * 100
            print(f"  {k:22s}  {gt_v:10.4f}   {rec_v:10.4f}   {err:.4f} ({err_pct:.1f}%)")

    # 4) Save audio outputs
    save_wav(str(out_dir / "recovered.wav"),
              result.final_pred.detach().numpy(), sr=sr)
    print(f"\nWAVs saved to {out_dir}/  (target.wav, recovered.wav)")

    report_path = save_fit_report(
        out_dir, engine="rolling_droplet", recovered=recovered, gt=gt,
        loss_history=result.loss_history,
        extra={"multistart": args.multistart},
    )
    print(f"Fit report saved to {report_path}")

    # 5) Loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(result.loss_history)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Multi-res STFT loss")
        axes[0].set_title(f"Inverse fitting loss curve "
                           f"({result.loss_history[0]:.3f} → {result.final_loss:.3f})")
        axes[0].grid(True)

        # Spectrograms before / after
        from scipy import signal as sg
        f_tgt, t_tgt, S_tgt = sg.spectrogram(target.numpy(), fs=sr, nperseg=1024)
        f_rec, t_rec, S_rec = sg.spectrogram(result.final_pred.detach().numpy(),
                                              fs=sr, nperseg=1024)
        axes[1].imshow(10 * np.log10(S_tgt + 1e-9), aspect="auto",
                        extent=[0, t_tgt[-1], 0, f_tgt[-1]],
                        origin="lower", cmap="magma")
        axes[1].set_title("Target spectrogram")
        axes[1].set_ylim(0, 5000)
        axes[1].set_ylabel("Hz")
        plt.tight_layout()
        plt.savefig(out_dir / "fit_diagnostics.png", dpi=100)
        print(f"Diagnostic plot saved to {out_dir}/fit_diagnostics.png")
    except ImportError:
        print("(matplotlib not installed — skipping diagnostic plot)")


if __name__ == "__main__":
    main()
