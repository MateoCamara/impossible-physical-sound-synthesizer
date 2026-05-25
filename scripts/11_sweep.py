"""CLI para barrer una propiedad sobre un ancla.

Ejemplos:
    # Mas liquido sobre el rolling de metal: -0.2 a +1.5 en 8 pasos
    python scripts/11_sweep.py --anchor fs365161 --more liquid --from -0.2 --to 1.5 --steps 8

    # Mas humedo (modificador latente) sobre un impacto en roca
    python scripts/11_sweep.py --anchor 9c95dfbea5d2 --more wetness --from 0 --to 1.5 --steps 6

    # Combinar dos knobs y barrer uno de ellos
    python scripts/11_sweep.py --anchor fs365161 --more liquid --base "wetness=0.5" --from 0 --to 1 --steps 5

    # Ancla por ruta a wav externo
    python scripts/11_sweep.py --anchor /path/to/mi_audio.wav --more rock --from 0 --to 1 --steps 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.controller import PropertyController


def parse_base(s: str) -> dict[str, float]:
    """'wetness=0.5,liquid=0.3' -> {'wetness': 0.5, 'liquid': 0.3}"""
    out: dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v.strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", help="clip_id, ruta a wav o 'tensor'")
    ap.add_argument("--more", help="propiedad a barrer (liquid, rock, wetness...)")
    ap.add_argument("--from", dest="from_", type=float, default=0.0)
    ap.add_argument("--to", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--base", default="", help="knobs adicionales fijos: 'wetness=0.5,liquid=0.3'")
    ap.add_argument("--out", type=Path, default=Path("outputs/sweeps"))
    ap.add_argument("--list-knobs", action="store_true", help="listar knobs disponibles y salir")
    args = ap.parse_args()

    ctrl = PropertyController()
    if args.list_knobs:
        for cat, props in ctrl.list_knobs().items():
            print(f"{cat:12s}: {props}")
        return 0

    if not args.anchor or not args.more:
        ap.error("--anchor y --more son obligatorios salvo con --list-knobs")
    print(f"Anchor: {args.anchor}")
    ctrl.set_anchor(args.anchor)

    base = parse_base(args.base)
    if base:
        print(f"Base knobs (fijos): {base}")

    amounts = list(np.linspace(args.from_, args.to, args.steps))
    sweep_id = f"{ctrl._anchor_id}__{args.more}__{int(time.time())}"
    out_dir = args.out / sweep_id
    print(f"Sweep '{args.more}' = {[round(a,2) for a in amounts]} -> {out_dir}")

    # Aplicar base + sweep
    if base:
        # Cada step: combinar base + sweep
        from impossible_mix.metrics.quality import quality_verdict
        from impossible_mix.config import SAMPLE_RATE
        import soundfile as sf
        for amount in amounts:
            knobs = {**base, args.more: amount}
            z, wav, info = ctrl.apply(knobs)
            wav_np = wav.numpy().astype("float32")
            q = quality_verdict(wav_np, SAMPLE_RATE)
            out_dir.mkdir(parents=True, exist_ok=True)
            base_tag = "_".join(f"{k}={v:.2f}" for k, v in base.items())
            fname = f"{ctrl._anchor_id}__{base_tag}__{args.more}{amount:+.2f}.wav"
            sf.write(str(out_dir / fname), wav_np, SAMPLE_RATE)
            print(f"  {args.more}={amount:+.2f}  rms={q.rms_db:6.1f} dyn={q.dynamic_range_db:5.1f}  {q.verdict.upper()}")
    else:
        ctrl.sweep(args.more, amounts, out_dir=out_dir, verbose=True)

    print(f"\nWavs en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
