"""Demo: recuperar parametros de friction/scrape via DDSP.

Cinco escalares aprendibles: surface_hardness, velocity_mean,
body_freq_hz, body_t60_s, gain. El ruido y la envolvente de velocidad
son pre-fijados por seed (constantes del problema).

Modos:
    --demo                 Genera target sintetico (hardness=0.8, body=2000Hz).
    --target audio.wav     Audio real de un scrape/drag.

Flags:
    --seed N
    --n-iters N
    --lr X
    --out path.wav         Guarda audio reconstruido.
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
    FrictionParamsT,
    fit_friction,
    synth_scrape_diff,
)


def _load_wav(path: Path, target_sr: int, max_seconds: float) -> np.ndarray:
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
    n_max = int(max_seconds * target_sr)
    if len(wav) > n_max:
        wav = wav[:n_max]
    elif len(wav) < n_max:
        wav = np.pad(wav, (0, n_max - len(wav)))
    return wav.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Target sintetico: hardness=0.8, body=2000Hz, t60=0.04s.")
    mode.add_argument("--target", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--n-iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sr = SAMPLE_RATE
    n = int(args.duration * sr)

    if args.demo:
        print("=== DEMO MODE ===")
        print("Generating synthetic scrape target:")
        print("  hardness=0.8 velocity_mean=0.7 body_freq=2000Hz body_t60=0.04s gain=0.5")
        tgt = FrictionParamsT.physical_init(
            surface_hardness=0.8, velocity_mean=0.7, body_freq_hz=2000.0,
            body_t60_s=0.04, gain=0.5, requires_grad=False, seed=args.seed,
        )
        with torch.no_grad():
            target = synth_scrape_diff(tgt, sr, n)
        gt = {"surface_hardness": 0.8, "velocity_mean": 0.7,
              "body_freq_hz": 2000.0, "body_t60_s": 0.04, "gain": 0.5}
    else:
        if not args.target.exists():
            print(f"ERROR: archivo no encontrado: {args.target}")
            return 1
        print(f"=== Loading target: {args.target} ===")
        wav = _load_wav(args.target, sr, args.duration)
        target = torch.from_numpy(wav)
        gt = None

    print(f"\nFitting (n_iters={args.n_iters}, seed={args.seed}, lr={args.lr})...")
    t0 = time.time()
    res = fit_friction(target, sr, n_iters=args.n_iters, lr=args.lr,
                        log_every=max(1, args.n_iters // 5), seed=args.seed)
    elapsed = time.time() - t0

    p = res.params
    rec = {
        "surface_hardness": float(p.surface_hardness.detach()),
        "velocity_mean": float(p.velocity_mean.detach()),
        "body_freq_hz": float(p.body_freq_hz.detach()),
        "body_t60_s": float(p.body_t60_s.detach()),
        "gain": float(p.gain.detach()),
    }
    print("\n=== RECOVERED ===")
    for k, v in rec.items():
        print(f"  {k:18s} = {v:.4f}")
    if gt is not None:
        print("\n=== GROUND TRUTH ===")
        for k, v in gt.items():
            err = abs(rec[k] - v)
            pct = 100 * err / max(abs(v), 1e-9)
            print(f"  TARGET {k:18s} = {v:.4f}   |   err {err:.4f} ({pct:.1f}%)")
        print("\nNote: body_t60_s tends to collapse to its lower bound — the")
        print("STFT loss is weakly sensitive to very short modal decays.")
    print(f"\nFinal loss: {res.final_loss:.4f}  (started at {res.loss_history[0]:.4f})")
    print(f"Elapsed: {elapsed:.1f}s for {args.n_iters} iters")

    if args.out:
        sf.write(str(args.out),
                  res.final_pred.detach().numpy().astype(np.float32),
                  sr)
        print(f"Reconstructed audio saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
