"""Demo: inverse problem para flujos granulares via DDSP diferenciable.

Recupera el perfil acustico de un grain cloud (base_freq, spread,
damping, density, gain) desde un audio target.

Limitacion conocida (documentada): el granular es una textura espectral
difusa, no peaks discretos. (base_freq_hz, spread_octaves) son
matematicamente redundantes — la loss STFT baja sin recuperar los
parametros exactos. Para recuperar parametros exactos, pasar
--init-base-freq con un guess razonable (e.g. centroide observado).

Modos:
    --demo                 Genera un target sintetico (pebble-ish) y verifica.
    --target audio.wav     Carga un wav real (textura granular) y reporta.

Flags utiles:
    --seed N               Seed del schedule (fijo durante el fit).
    --n-grains N           Numero de granos (default 200).
    --n-iters N            Iteraciones de Adam (default 180).
    --lr X                 LR inicial (default 0.08, cosine decay).
    --init-base-freq HZ    Override del init de base_freq_hz.
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
    GranularFlowParamsT,
    fit_granular_flow,
    synth_granular_flow_diff,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Genera target sintetico (pebble: base=550Hz, "
                           "spread=0.6oct, damping=35ms).")
    mode.add_argument("--target", type=Path,
                      help="Audio WAV objetivo (textura granular mono).")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-grains", type=int, default=80)
    ap.add_argument("--n-iters", type=int, default=180)
    ap.add_argument("--lr", type=float, default=8e-2)
    ap.add_argument("--init-base-freq", type=float, default=None,
                    help="Override base_freq_hz inicial (Hz). Si no, spectral centroid.")
    ap.add_argument("--duration", type=float, default=2.0,
                    help="Duracion del audio si --demo, o longitud usada si --target.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Guarda audio reconstruido (wav 44.1 kHz).")
    args = ap.parse_args()

    sr = SAMPLE_RATE
    n_samples = int(args.duration * sr)

    # 1. Target
    if args.demo:
        print("=== DEMO MODE ===")
        print("Generating synthetic target (pebble-ish):")
        print("  base_freq_hz=550, spread_octaves=0.6, damping_ms=35")
        tgt = GranularFlowParamsT.physical_init(
            base_freq_hz=550.0, spread_octaves=0.6, damping_ms=35.0,
            log_density_amp=0.5, gain=0.5, requires_grad=False,
            seed=args.seed, n_grains=args.n_grains, grain_dur_ms=200.0,
            duration_s=args.duration,
        )
        with torch.no_grad():
            target = synth_granular_flow_diff(tgt, sr, n_samples)
        gt = {"base_freq_hz": 550.0, "spread_octaves": 0.6, "damping_ms": 35.0}
    else:
        if not args.target.exists():
            print(f"ERROR: archivo no encontrado: {args.target}")
            return 1
        print(f"=== Loading target: {args.target} ===")
        wav, sr_in = sf.read(str(args.target), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr_in != sr:
            print(f"  Resampling {sr_in} -> {sr} Hz")
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr_in, target_sr=sr)
            except ImportError:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr_in, sr)
                wav = resample_poly(wav, sr // g, sr_in // g).astype(np.float32)
        if len(wav) > n_samples:
            wav = wav[:n_samples]
        else:
            wav = np.pad(wav, (0, n_samples - len(wav)))
        target = torch.from_numpy(wav.astype(np.float32))
        gt = None

    # 2. Fitting
    print(f"\nFitting granular profile: n_grains={args.n_grains}, "
          f"n_iters={args.n_iters}, init_base_freq={args.init_base_freq}")
    t0 = time.time()
    result = fit_granular_flow(
        target, sr, duration_s=args.duration,
        n_iters=args.n_iters, lr=args.lr,
        log_every=max(1, args.n_iters // 6),
        seed=args.seed, n_grains=args.n_grains,
        init_base_freq_hz=args.init_base_freq,
    )
    elapsed = time.time() - t0

    # 3. Reporte
    p = result.params
    rec = {
        "base_freq_hz": float(p.base_freq_hz.detach()),
        "spread_octaves": float(p.spread_octaves.detach()),
        "damping_ms": float(p.damping_ms.detach()),
        "log_density_amp": float(p.log_density_amp.detach()),
        "gain": float(p.gain.detach()),
    }
    print("\n=== RECOVERED PROFILE ===")
    for k, v in rec.items():
        print(f"  {k:18s} = {v:.4f}")
    if gt is not None:
        print("\n=== GROUND TRUTH ===")
        for k, v in gt.items():
            err = abs(rec[k] - v)
            pct = 100 * err / max(abs(v), 1e-9)
            print(f"  TARGET {k:18s} = {v:.4f}   |   err {err:.4f} ({pct:.1f}%)")
        print("\nNote: base_freq_hz / spread_octaves are mathematically")
        print("redundant in this model. Recovery accuracy depends on init.")

    print(f"\nFinal loss: {result.final_loss:.4f}  (started at {result.loss_history[0]:.4f})")
    print(f"Elapsed: {elapsed:.1f}s for {args.n_iters} iters")

    if args.out:
        sf.write(str(args.out),
                  result.final_pred.detach().numpy().astype(np.float32),
                  sr)
        print(f"\nReconstructed audio saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
