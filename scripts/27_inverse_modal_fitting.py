"""Demo: inverse problem para impactos modales via DDSP diferenciable.

Dado un audio de impacto (real o sintetizado) descompone el sonido en
K modos resonantes: encuentra (frecuencia, t60, ganancia) de cada modo
por gradiente sobre una multi-resolution STFT loss.

Estrategia clave: las frecuencias se inicializan en los K picos
espectrales mas prominentes del target (init_from_target=True). Sin
este init informado, el espacio de busqueda tiene demasiados optimos
locales para Adam.

Modos:
    --demo                          Genera un target sintetico de 4 modos
                                     y verifica recuperacion.
    --target /path/to/audio.wav     Carga un wav real y reporta los modos
                                     recuperados.

Flags utiles:
    --n-modes K            Numero de modos a buscar (default 6).
    --multistart           Prueba K=4, 6, 8 y devuelve el mejor.
    --n-iters N            Iteraciones de Adam (default 200).
    --lr X                 Learning rate inicial (default 0.03).
    --no-peak-init         Desactiva la inicializacion por picos (peor pero
                            mas honesto si quieres comparar metodologias).
    --out path.wav         Guarda audio reconstruido.
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
    ModalParamsT,
    fit_modal_impact,
    fit_modal_impact_multistart,
    synth_modal_impact_diff,
)
from impossible_mix.utils import load_wav_mono, save_fit_report, save_wav, seed_everything


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Target sintetico: 4 modos en [440, 880, 1320, 1760] Hz "
                           "con t60s [1.2, 0.6, 0.3, 0.15] s.")
    mode.add_argument("--target", type=Path,
                      help="Audio WAV objetivo (impacto/golpe mono recomendado).")
    ap.add_argument("--n-modes", type=int, default=6)
    ap.add_argument("--multistart", action="store_true",
                    help="Prueba K=4, 6, 8 y devuelve el mejor (~3x mas lento).")
    ap.add_argument("--n-iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--no-peak-init", action="store_true",
                    help="Desactiva init por picos espectrales del target.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Guarda audio reconstruido (wav 44.1 kHz).")
    ap.add_argument("--out-dir", type=Path, default=Path("results/diff_fits/modal"),
                    help="Directorio donde guardar params.json con el reporte del fit.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Semilla global (default: impossible_mix.config.SEED).")
    args = ap.parse_args()
    seed_everything(args.seed)

    # 1. Target
    if args.demo:
        print("=== DEMO MODE ===")
        print("Generating synthetic target with 4 modes:")
        print("  freqs = [440, 880, 1320, 1760] Hz")
        print("  t60s  = [1.2, 0.6, 0.3, 0.15] s")
        n_samples = SAMPLE_RATE // 2
        target_p = ModalParamsT.from_lists(
            freqs_hz=[440.0, 880.0, 1320.0, 1760.0],
            t60s_s=[1.2, 0.6, 0.3, 0.15],
            gains=[1.0, 0.5, 0.3, 0.15],
            requires_grad=False,
        )
        with torch.no_grad():
            target = synth_modal_impact_diff(target_p, SAMPLE_RATE, n_samples)
        gt = {"freqs_hz": [440.0, 880.0, 1320.0, 1760.0],
              "t60s_s": [1.2, 0.6, 0.3, 0.15]}
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
    print(f"\nFitting modal impact with K={args.n_modes}, n_iters={args.n_iters}, "
          f"init_from_target={not args.no_peak_init}...")
    t0 = time.time()
    if args.multistart:
        print("Multi-start ON over n_modes_candidates=(4, 6, 8)")
        result = fit_modal_impact_multistart(
            target, SAMPLE_RATE,
            n_modes_candidates=(4, 6, 8),
            n_iters=args.n_iters, lr=args.lr,
            log_every=max(1, args.n_iters // 5),
            init_from_target=not args.no_peak_init,
        )
    else:
        result = fit_modal_impact(
            target, SAMPLE_RATE, n_modes=args.n_modes,
            n_iters=args.n_iters, lr=args.lr,
            log_every=max(1, args.n_iters // 5),
            init_from_target=not args.no_peak_init,
        )
    elapsed = time.time() - t0

    # 3. Reporte
    p = result.params
    K = p.freqs_hz.numel()
    order = sorted(range(K), key=lambda i: p.freqs_hz[i].item())
    freqs = [round(p.freqs_hz.detach()[i].item(), 1) for i in order]
    t60s = [round(p.t60s_s.detach()[i].item(), 3) for i in order]
    gains_sm = torch.softmax(p.gains.detach(), dim=0)
    gains = [round(gains_sm[i].item(), 3) for i in order]

    print("\n=== RECOVERED MODES (sorted by frequency) ===")
    print(f"{'#':>3} {'freq_hz':>10} {'t60_s':>8} {'gain':>8}")
    for i, (f, t, g) in enumerate(zip(freqs, t60s, gains)):
        print(f"{i:>3} {f:>10} {t:>8} {g:>8}")

    mode_matches = None
    if gt is not None:
        print("\n=== GROUND TRUTH ===")
        mode_matches = []
        for i, (f_t, t_t) in enumerate(zip(gt["freqs_hz"], gt["t60s_s"])):
            # Match con el modo recuperado mas cercano en freq
            closest = min(range(K), key=lambda j: abs(freqs[j] - f_t))
            ef = abs(freqs[closest] - f_t)
            et = abs(t60s[closest] - t_t)
            print(f"  TARGET mode {i}  freq={f_t:.1f} Hz t60={t_t:.3f} s   |   "
                   f"err freq={ef:.2f} Hz ({100*ef/f_t:.2f}%)  "
                   f"err t60={et:.4f} s ({100*et/t_t:.1f}%)")
            mode_matches.append({
                "target_mode": i, "freq_hz": f_t, "t60_s": t_t,
                "matched_index": closest, "err_freq_hz": ef,
                "err_freq_pct": 100 * ef / f_t, "err_t60_s": et,
                "err_t60_pct": 100 * et / t_t,
            })

    print(f"\nFinal loss: {result.final_loss:.4f}  "
          f"(started at {result.loss_history[0]:.4f})")
    print(f"Elapsed: {elapsed:.1f}s for {args.n_iters} iters")

    # 4. Guardar audio si se pide
    if args.out:
        save_wav(args.out, result.final_pred.detach().numpy(), SAMPLE_RATE)
        print(f"\nReconstructed audio saved to {args.out}")

    rec = {"freqs_hz": freqs, "t60s_s": t60s, "gains": gains}
    report_path = save_fit_report(
        args.out_dir, engine="modal", recovered=rec, gt=gt,
        loss_history=result.loss_history,
        extra={"elapsed_s": elapsed, "n_modes": K, "mode_matches": mode_matches},
    )
    print(f"Fit report saved to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
