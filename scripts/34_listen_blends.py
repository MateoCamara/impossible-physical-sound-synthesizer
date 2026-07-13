"""Blends profundos v7: checksums dorados, matriz de escucha y smokes.

Fase 0: `--record` graba los sha256 de renders canonicos de las funciones
que la v7 refactoriza (red de seguridad de bit-identidad). `--check`
verifica los checksums (y, cuando existan, los smokes de los blends).

Uso:
    python scripts/34_listen_blends.py --record   # SOLO en Fase 0, una vez
    python scripts/34_listen_blends.py --check    # gate de cada fase
    python scripts/34_listen_blends.py            # (F4) matriz de escucha
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impossible_mix.physics.composer import compose_impossible  # noqa: E402
from impossible_mix.physics.droplet import synth_rolling_droplet  # noqa: E402
from impossible_mix.physics.droplet_presets import get_preset  # noqa: E402
from impossible_mix.physics.exotic import (  # noqa: E402
    synth_fire, synth_glass_thunder, synth_thunder,
)
from impossible_mix.physics.granular import GranularParams, synth_granular_flow  # noqa: E402
from impossible_mix.physics.liquid import (  # noqa: E402
    PourParams, SplashParams, synth_pour, synth_splash,
)

SR = 44_100
GOLDEN_PATH = Path(__file__).with_name("golden_checksums_v7.json")


def _canonical_renders() -> dict[str, np.ndarray]:
    """Renders canonicos y deterministas de las funciones protegidas."""
    return {
        "granular_flow": synth_granular_flow(
            GranularParams(duration_s=2.0, seed=42), SR),
        "splash": synth_splash(SplashParams(duration_s=2.0, seed=42), SR),
        "pour": synth_pour(PourParams(duration_s=2.0, seed=42), SR),
        "fire": synth_fire(duration_s=2.0, seed=42, sr=SR),
        "thunder": synth_thunder(duration_s=2.0, seed=42, sr=SR),
        "glass_thunder": synth_glass_thunder(duration_s=2.0, seed=42, sr=SR),
        "rolling_droplet": synth_rolling_droplet(
            get_preset("water", duration_s=2.0, seed=42), SR),
        "compose_impossible_sum": compose_impossible(
            "rock", "impact", overlay_material="liquid",
            overlay_interaction="splash", overlay_weight=0.5,
            duration_s=2.0, seed=42, sr=SR),
    }


def _sha(w: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(w, dtype=np.float32).tobytes()).hexdigest()


def record_golden() -> int:
    sums = {name: _sha(w) for name, w in _canonical_renders().items()}
    GOLDEN_PATH.write_text(json.dumps(sums, indent=1), encoding="utf-8")
    print(f"{len(sums)} checksums dorados grabados en {GOLDEN_PATH.name}:")
    for k, v in sums.items():
        print(f"  {k}: {v[:16]}…")
    return 0


def check_golden() -> list[str]:
    if not GOLDEN_PATH.exists():
        return [f"no existe {GOLDEN_PATH.name}: corre --record en Fase 0"]
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    failures = []
    for name, w in _canonical_renders().items():
        got = _sha(w)
        if name not in golden:
            failures.append(f"{name}: sin checksum dorado")
        elif got != golden[name]:
            failures.append(f"{name}: render cambio ({got[:12]}… != {golden[name][:12]}…)")
    return failures


def run_checks() -> int:
    failures = check_golden()
    # Los smokes de blends (FI, determinismo por escaparate) se anaden en F4.
    if failures:
        print("CHECKS FALLIDOS:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"Checksums dorados OK ({GOLDEN_PATH.name}).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true",
                    help="graba los checksums dorados (solo Fase 0)")
    ap.add_argument("--check", action="store_true", help="verifica gates")
    args = ap.parse_args()
    if args.record:
        return record_golden()
    if args.check:
        return run_checks()
    print("La matriz de escucha de blends llega en la Fase 4; usa --check/--record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
