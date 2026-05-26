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
import soundfile as sf
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


def _load_wav(path: Path, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, target_sr)
            wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    return wav.astype(np.float32)


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
    args = ap.parse_args()

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
    else:
        if args.dry is None or args.wet is None:
            print("ERROR: --dry y --wet son obligatorios en modo real")
            return 1
        print(f"=== Loading dry={args.dry}, wet={args.wet} ===")
        dry_np = _load_wav(args.dry, sr)
        wet_np = _load_wav(args.wet, sr)
        n = min(len(dry_np), len(wet_np))
        dry_np, wet_np = dry_np[:n], wet_np[:n]
        dry = torch.from_numpy(dry_np)
        wet_target = torch.from_numpy(wet_np)

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
    print(f"\nFinal loss: {res.final_loss:.4f}  (start {res.loss_history[0]:.4f})")
    print(f"IR rms = {float(res.ir_params.ir_samples.detach().pow(2).mean().sqrt()):.5f}")
    print(f"IR peak = {float(res.ir_params.ir_samples.detach().abs().max()):.5f}")
    print(f"Elapsed: {elapsed:.1f}s for {args.n_iters} iters")

    if args.out_ir:
        sf.write(str(args.out_ir),
                  res.ir_params.ir_samples.detach().numpy().astype(np.float32),
                  sr)
        print(f"Recovered IR saved to {args.out_ir}")
    if args.out_wet:
        sf.write(str(args.out_wet),
                  res.final_pred.detach().numpy().astype(np.float32),
                  sr)
        print(f"Reconstructed wet saved to {args.out_wet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
