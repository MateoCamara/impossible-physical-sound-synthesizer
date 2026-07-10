"""Demo: recuperar la IR de un espacio (DDSP-style).

Dado un audio dry (conocido) y un wet target (mismo dry pasado por
algun espacio), aprende la IR del espacio por gradiente sobre una
multi-resolution STFT loss.

La convolucion se implementa via FFT (O(N log N)), 70x mas rapida que
F.conv1d bajo autograd para IRs largas.

Modos:
    --demo                       Genera dry+wet sinteticos y recupera la IR.
    --dry path.wav --wet path.wav  Modo real con tus audios.

Flags:
    --ir-seconds X             Longitud de la IR a aprender (default 0.8s).
    --n-iters N                Iteraciones (default 150).
    --lr X                     LR (default 3e-3).
    --init-t60 X               t60 inicial de la IR (default 0.4 s).
    --sparsity X               Peso L1 sobre la IR (default 1e-4).
    --out-ir path              Guarda IR recuperada.
    --out-wet path             Guarda audio reconstruido.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import SAMPLE_RATE
from impossible_mix.physics.diff import (
    DripParamsT,
    IRParamsT,
    fit_reverb_ir,
    synth_drip_event_diff,
    synth_reverb_diff,
)
from impossible_mix.utils import load_wav_mono, save_fit_report, save_wav, seed_everything


def _estimate_t60(ir: np.ndarray, sr: int) -> float | None:
    """Estima t60 (s) por T30 extrapolado: pendiente de la curva de
    decaimiento de Schroeder (integracion inversa) entre -5 dB y -35 dB,
    extrapolada x2. Devuelve None si la IR no decae lo suficiente."""
    energy = np.cumsum(ir[::-1].astype(np.float64) ** 2)[::-1]
    energy = energy / (energy[0] + 1e-20)
    edc_db = 10 * np.log10(energy + 1e-20)
    idx_5 = np.argmax(edc_db <= -5)
    idx_35 = np.argmax(edc_db <= -35)
    if idx_35 <= idx_5 or edc_db[idx_35] > -35:
        return None
    t30 = (idx_35 - idx_5) / sr
    return 2.0 * t30


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Genera dry (drip 2 mm) + wet (exp-decay IR t60=0.8 s).")
    mode.add_argument("--dry", type=Path)
    ap.add_argument("--wet", type=Path)
    ap.add_argument("--ir-seconds", type=float, default=0.8)
    ap.add_argument("--n-iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--init-t60", type=float, default=0.4)
    ap.add_argument("--sparsity", type=float, default=1e-4)
    ap.add_argument("--out-ir", type=Path, default=None)
    ap.add_argument("--out-wet", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/diff_fits/reverb"),
                    help="Directorio donde guardar params.json (metricas + losses) "
                         "y la IR recuperada.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Semilla global (default: impossible_mix.config.SEED).")
    args = ap.parse_args()
    seed_everything(args.seed)

    sr = SAMPLE_RATE
    ir_len = int(args.ir_seconds * sr)

    if args.demo:
        print("=== DEMO MODE ===")
        dp = DripParamsT.physical_init(radius_mm=2.0, requires_grad=False)
        with torch.no_grad():
            dry = synth_drip_event_diff(dp, sr, int(0.6 * sr))
        ir_target = IRParamsT.from_exp_decay(ir_len, sr=sr, t60_s=0.8,
                                              seed=42, requires_grad=False)
        with torch.no_grad():
            wet_target = synth_reverb_diff(dry, ir_target, mix=1.0)
        print(f"  Dry: drip event len={len(dry)/sr:.2f}s")
        print(f"  Wet: dry * exp-decay IR (t60=0.80 s)")
        gt = {"t60_s": 0.8}
    else:
        if args.dry is None or args.wet is None:
            print("ERROR: --dry y --wet son obligatorios en modo real")
            return 1
        print(f"=== Loading dry={args.dry}, wet={args.wet} ===")
        dry_np = load_wav_mono(args.dry, sr)
        wet_np = load_wav_mono(args.wet, sr)
        n = min(len(dry_np), len(wet_np))
        dry_np, wet_np = dry_np[:n], wet_np[:n]
        dry = torch.from_numpy(dry_np)
        wet_target = torch.from_numpy(wet_np)
        gt = None

    print(f"\nFitting IR (length={ir_len/sr:.2f}s, {ir_len} samples) "
           f"with n_iters={args.n_iters}, lr={args.lr}...")
    t0 = time.time()
    res = fit_reverb_ir(
        dry, wet_target, sr,
        ir_length_samples=ir_len,
        n_iters=args.n_iters, lr=args.lr,
        log_every=max(1, args.n_iters // 5),
        sparsity_weight=args.sparsity,
        init_t60_s=args.init_t60,
    )
    elapsed = time.time() - t0
    ir_np = res.ir_params.ir_samples.detach().numpy().astype(np.float32)
    ir_rms = float(np.sqrt(np.mean(ir_np ** 2)))
    ir_peak = float(np.max(np.abs(ir_np)))
    t60_est = _estimate_t60(ir_np, sr)
    print(f"\nFinal loss: {res.final_loss:.4f}  (start {res.loss_history[0]:.4f})")
    print(f"IR rms = {ir_rms:.5f}")
    print(f"IR peak = {ir_peak:.5f}")
    print(f"IR t60 (estimated) = {t60_est if t60_est is not None else 'N/A'}")
    print(f"Elapsed: {elapsed:.1f}s for {args.n_iters} iters")

    if args.out_ir:
        save_wav(args.out_ir, ir_np, sr)
        print(f"Recovered IR saved to {args.out_ir}")
    if args.out_wet:
        save_wav(args.out_wet, res.final_pred.detach().numpy(), sr)
        print(f"Reconstructed wet saved to {args.out_wet}")

    # Guardar siempre la IR recuperada en out-dir (ademas de --out-ir si se pide)
    save_wav(args.out_dir / "recovered_ir.wav", ir_np, sr)

    rec = {"t60_s": t60_est, "ir_rms": ir_rms, "ir_peak": ir_peak}
    report_path = save_fit_report(
        args.out_dir, engine="reverb", recovered=rec, gt=gt,
        loss_history=res.loss_history, final_loss=res.final_loss,
        extra={"elapsed_s": elapsed, "ir_seconds": args.ir_seconds},
    )
    print(f"Fit report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
