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


# ------------------------------------------------------------------
# F4: escaparates, indice de fusion y matriz de escucha
# ------------------------------------------------------------------
# (receta, fn_blend, fn_suma, banda_A, banda_B, expected_rate)
def _fi_cases():
    from impossible_mix.physics import blend_recipes as br
    return [
        ("trueno_gotea", lambda: br.blend_thunder_drips(8.0, 42),
         lambda: br.sum_thunder_drips(8.0, 42), (30, 300), (900, 4000), 8),
        ("fuego_cristal", lambda: br.blend_glass_fire(8.0, 42),
         lambda: br.sum_glass_fire(8.0, 42), (1350, 1750), (3800, 4700), 14),
        ("canica_derrite", lambda: br.blend_melting_marble(10.0, 42),
         lambda: br.sum_melting_marble(10.0, 42), (2000, 6000), (600, 1600), 28),
        ("trueno_habla", lambda: br.blend_thunder_speaks_water(8.0, 42, dry_drips=0.15),
         lambda: br.sum_thunder_speaks_water(8.0, 42), (30, 300), (900, 3000), 10),
    ]


def run_checks() -> int:
    failures = check_golden()

    from impossible_mix.physics.analysis import fusion_index
    for name, blend_fn, sum_fn, ba, bb, rate in _fi_cases():
        w = blend_fn()
        if not np.isfinite(w).all():
            failures.append(f"{name}: NaN/Inf")
        if np.abs(w).max() > 0.9501:
            failures.append(f"{name}: pico {np.abs(w).max():.3f} > 0.95")
        if not np.array_equal(w, blend_fn()):
            failures.append(f"{name}: no determinista")
        fb = fusion_index(w, SR, ba, bb, rate, n_perm=100, seed=1)
        fs = fusion_index(sum_fn(), SR, ba, bb, rate, n_perm=100, seed=1)
        ratio = fb.fi / max(fs.fi, 0.05)
        if not (ratio > 3.0 and fb.p_value < 0.011):
            failures.append(
                f"{name}: FI no discrimina (blend {fb.fi:.3f} vs suma {fs.fi:.3f}, "
                f"ratio {ratio:.1f}, p {fb.p_value:.3f})")
        else:
            print(f"  FI {name}: blend {fb.fi:.3f} vs suma {fs.fi:.3f} "
                  f"(x{ratio:.1f}, p={fb.p_value:.3f})")

    if failures:
        print("CHECKS FALLIDOS:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"Checks OK: checksums dorados + 4 escaparates (FI, determinismo).")
    return 0


LEEME = """# Escucha v7: blends PROFUNDOS vs sumas ponderadas

Pregunta central: **¿se oye UN evento fisico en dos materiales, o dos
pistas superpuestas como en los 9x?**

- **00-01 trueno_gotea**: el trueno GOBIERNA la lluvia (mas trueno = gotas
  mas densas y gordas) y cada gota pinga el cielo de vidrio. 01 sin vidrio
  y acople a medias.
- **10-11 fuego_cristal**: cada crepitar pinga A LA VEZ agua y vidrio (un
  evento, dos cuerpos); la respiracion del fuego espesa el agua. 11 solo
  vidrio (sin agua).
- **20-21 canica_derrite**: canica metalica que se derrite en gota y acaba
  goteando — las LEYES se interpolan en un render continuo. 21 sin el
  goteo final.
- **30-31 trueno_habla**: el trueno articulado por el goteo (vocoder
  fisico). 31 con las gotas secas audibles al 15%.
- **90-93 ANCLAS**: los mismos fenomenos como suma ponderada de procesos
  independientes — "esto era lo de antes". Compara 00vs90, 10vs91,
  20vs92, 30vs93.

Di cuales blends te vuelan la cabeza y cuales no, y que les falta.
"""


def render_matrix(out_dir: Path) -> int:
    from impossible_mix.physics import blend_recipes as br
    from impossible_mix.utils import save_wav
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [
        ("00_trueno_gotea__a", lambda: br.blend_thunder_drips(8.0, 42)),
        ("01_trueno_gotea__b_suave", lambda: br.blend_thunder_drips(8.0, 42, glass_gain=0.0, coupling=0.5)),
        ("10_fuego_cristal__a", lambda: br.blend_glass_fire(8.0, 42)),
        ("11_fuego_cristal__b_solo_vidrio", lambda: br.blend_glass_fire(8.0, 42, water_gain=0.0, glass_gain=0.9)),
        ("20_canica_derrite__a", lambda: br.blend_melting_marble(10.0, 42)),
        ("21_canica_derrite__b_sin_goteo", lambda: br.blend_melting_marble(10.0, 42, with_handoff=False)),
        ("30_trueno_habla__a", lambda: br.blend_thunder_speaks_water(8.0, 42)),
        ("31_trueno_habla__b_gotas_secas", lambda: br.blend_thunder_speaks_water(8.0, 42, dry_drips=0.15)),
        ("90_ancla_suma__trueno_gotas", lambda: br.sum_thunder_drips(8.0, 42)),
        ("91_ancla_suma__fuego_glass", lambda: br.sum_glass_fire(8.0, 42)),
        ("92_ancla_suma__metal_mas_agua", lambda: br.sum_melting_marble(10.0, 42)),
        ("93_ancla_suma__trueno_mas_gotas", lambda: br.sum_thunder_speaks_water(8.0, 42)),
    ]
    for name, fn in items:
        w = fn()
        save_wav(out_dir / f"{name}.wav", w, SR)
        print(f"  {name}.wav  peak={np.abs(w).max():.2f}")
    (out_dir / "LEEME.md").write_text(LEEME, encoding="utf-8")
    print(f"\n{len(items)} clips en {out_dir}/ (+ LEEME.md)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true",
                    help="graba los checksums dorados (solo Fase 0)")
    ap.add_argument("--check", action="store_true", help="verifica gates")
    ap.add_argument("--out-dir", type=Path, default=Path("escucha_AB/blends_v7"))
    args = ap.parse_args()
    if args.record:
        return record_golden()
    if args.check:
        return run_checks()
    return render_matrix(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
