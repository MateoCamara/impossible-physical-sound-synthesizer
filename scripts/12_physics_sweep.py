"""CLI para sweeps fisicos sobre el composer parametrico.

Ejemplos:
    # Knobs y escenas disponibles
    python scripts/12_physics_sweep.py --list-scenes
    python scripts/12_physics_sweep.py --list-knobs

    # Mas humedo sobre gravel rodando
    python scripts/12_physics_sweep.py \\
        --material gravel --interaction roll \\
        --more wetness --from 0 --to 1 --steps 6

    # Combo imposible: rolling droplet + variar viscosidad fisica via wetness
    python scripts/12_physics_sweep.py \\
        --material liquid --interaction roll \\
        --more wetness --from 0 --to 1 --steps 6

    # Combo IMPOSIBLE con overlay (gravel scrape + liquid pour)
    python scripts/12_physics_sweep.py \\
        --material gravel --interaction scrape \\
        --overlay-material liquid --overlay-interaction pour --overlay-weight 0.4 \\
        --more granularity --from 0 --to 1 --steps 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.physics_controller import KNOBS, PhysicsController


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default=None)
    ap.add_argument("--interaction", default=None)
    ap.add_argument("--overlay-material", default=None)
    ap.add_argument("--overlay-interaction", default=None)
    ap.add_argument("--overlay-weight", type=float, default=0.5)
    ap.add_argument("--more", default=None, help=f"knob: {KNOBS}")
    ap.add_argument("--from", dest="from_", type=float, default=0.0)
    ap.add_argument("--to", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=Path("outputs/physics_sweeps"))
    ap.add_argument("--list-knobs", action="store_true")
    ap.add_argument("--list-scenes", action="store_true")
    args = ap.parse_args()

    if args.list_knobs:
        print(f"knobs: {KNOBS}")
        return 0
    if args.list_scenes:
        for m, ints in PhysicsController.list_scenes().items():
            print(f"{m:8s} -> {ints}")
        return 0

    if not args.material or not args.interaction or not args.more:
        ap.error("--material, --interaction y --more son obligatorios salvo con --list-*")

    ctrl = PhysicsController(seed=args.seed, duration_s=args.duration)
    ctrl.set_scene(
        material=args.material, interaction=args.interaction,
        overlay_material=args.overlay_material,
        overlay_interaction=args.overlay_interaction,
        overlay_weight=args.overlay_weight,
    )
    amounts = list(np.linspace(args.from_, args.to, args.steps))
    sweep_id = f"{ctrl._scene_name()}__{args.more}__{int(time.time())}"
    out_dir = args.out / sweep_id
    print(f"Scene: {ctrl._scene_name()}")
    print(f"Sweep {args.more} = {[round(a,2) for a in amounts]} -> {out_dir}")
    ctrl.sweep(args.more, amounts, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
