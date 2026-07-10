"""Demo: inverse problem para drip events via DDSP diferenciable.

Dado un audio target (real o sintetizado), encuentra los parametros
fisicos (radius_mm, viscosity, etc.) que mas se aproximan a el por
descenso de gradiente sobre una loss multi-resolution STFT.

Modos:
    --demo                          Genera un target sintetico y verifica
                                     recuperacion exacta de parametros.
    --target /path/to/audio.wav     Carga un wav real y reporta los
                                     parametros recuperados.

Otros flags:
    --multistart       Prueba 4 inicializaciones distintas (mas robusto
                       pero ~4x mas lento).
    --n-iters N        Iteraciones de Adam (default 150).
    --lr X             Learning rate inicial (default 0.05, cosine decay).
    --out path.wav     Guarda audio recuperado.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import SAMPLE_RATE
from impossible_mix.physics.diff import (
    DripParamsT,
    fit_drip_event,
    fit_drip_event_multistart,
    synth_drip_event_diff,
)
from impossible_mix.utils import load_wav_mono, save_fit_report, save_wav, seed_everything


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Genera un target sintetico (radius=3.5mm visc=0.4) y "
                           "verifica recuperacion.")
    mode.add_argument("--target", type=Path,
                      help="Audio WAV objetivo (real o sintetico). Mono recomendado.")
    ap.add_argument("--multistart", action="store_true",
                    help="Multiples inicializaciones (4 starts, ~4x mas lento).")
    ap.add_argument("--n-iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--init-radius", type=float, default=2.5)
    ap.add_argument("--init-viscosity", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=None,
                    help="Guarda audio recuperado (wav 44.1 kHz).")
    ap.add_argument("--out-dir", type=Path, default=Path("results/diff_fits/drip"),
                    help="Directorio donde guardar params.json con el reporte del fit.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Semilla global (default: impossible_mix.config.SEED).")
    args = ap.parse_args()
    seed_everything(args.seed)

    # 1. Obtener target wav
    if args.demo:
        print("=== DEMO MODE ===")
        print("Generating synthetic target: radius=3.5 mm, viscosity=0.4")
        n_samples = SAMPLE_RATE // 2
        target_p = DripParamsT.physical_init(radius_mm=3.5, viscosity=0.4,
                                              requires_grad=False)
        with torch.no_grad():
            target = synth_drip_event_diff(target_p, SAMPLE_RATE, n_samples)
        gt = {"radius_mm": 3.5, "viscosity": 0.4}
    else:
        if not args.target.exists():
            print(f"ERROR: archivo no encontrado: {args.target}")
            return 1
        print(f"=== Loading target: {args.target} ===")
        n_samples = SAMPLE_RATE // 2
        wav = load_wav_mono(args.target, SAMPLE_RATE, max_seconds=n_samples / SAMPLE_RATE)
        target = torch.from_numpy(wav)
        gt = None

    # 2. Fitting
    init = DripParamsT.physical_init(radius_mm=args.init_radius,
                                       viscosity=args.init_viscosity,
                                       requires_grad=True)
    print(f"\nFitting with init radius={args.init_radius:.2f} mm, "
          f"viscosity={args.init_viscosity:.2f}, n_iters={args.n_iters}, "
          f"freeze_surface=True...")
    t0 = time.time()
    if args.multistart:
        print("Multi-start ON (4 inits)")
        result = fit_drip_event_multistart(target, SAMPLE_RATE,
                                             n_iters=args.n_iters, lr=args.lr,
                                             log_every=max(1, args.n_iters // 5),
                                             freeze_surface=True)
    else:
        result = fit_drip_event(target, SAMPLE_RATE, initial=init,
                                  n_iters=args.n_iters, lr=args.lr,
                                  log_every=max(1, args.n_iters // 5),
                                  freeze_surface=True)
    elapsed = time.time() - t0

    # 3. Reporte
    rp = result.params
    print("\n=== RESULT ===")
    rec = {
        "radius_mm": float(rp.radius_mm.detach()),
        "viscosity": float(rp.viscosity.detach()),
        "chirp_amp": float(rp.chirp_amp.detach()),
        "decay_scale": float(rp.decay_scale.detach()),
        "capillary": float(rp.capillary_ringing.detach()),
    }
    for k, v in rec.items():
        print(f"  RECOVERED {k:20s} = {v:.4f}")
    if gt is not None:
        print("\n=== GROUND TRUTH ===")
        for k, v in gt.items():
            err = abs(rec[k] - v)
            pct = 100 * err / max(abs(v), 1e-9)
            print(f"  TARGET    {k:20s} = {v:.4f}   |   error {err:.4f} ({pct:.1f}%)")
    print(f"\nFinal loss: {result.final_loss:.4f}  (started at {result.loss_history[0]:.4f})")
    print(f"Elapsed: {elapsed:.1f}s for {args.n_iters} iters")

    # 4. Guardar audio si se pide
    if args.out:
        save_wav(args.out, result.final_pred.detach().numpy(), SAMPLE_RATE)
        print(f"\nRecovered audio saved to {args.out}")

    report_path = save_fit_report(
        args.out_dir, engine="drip", recovered=rec, gt=gt,
        loss_history=result.loss_history, final_loss=result.final_loss,
        extra={"elapsed_s": elapsed},
    )
    print(f"Fit report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
