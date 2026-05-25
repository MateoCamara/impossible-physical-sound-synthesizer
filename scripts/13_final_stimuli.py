"""Genera la bateria final de 24 estimulos para el test perceptual del Dia 8.

Estructura:
  3 combos imposibles x 8 variaciones cada uno =
      A) rolling droplet      (liquid x roll)        sweep [drop_size, wetness, continuity, granularity]
      B) liquid rock impact   (rock x impact + liquid splash overlay)  sweep [wetness, resonance, overlay]
      C) wet gravel scrape    (gravel x scrape + liquid pour overlay)  sweep [granularity, wetness, overlay]

Cada combo tiene un 'punto canonico' (mid) y 7 vecinos en knobs principales.
Esto da material para:
  - Test perceptual A/B/X
  - Espectrogramas comparados
  - Sweep figures
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import SAMPLE_RATE
from impossible_mix.metrics.quality import quality_verdict
from impossible_mix.physics_controller import PhysicsController


OUT_DIR = Path("perceptual_test/stimuli")
MANIFEST = Path("perceptual_test/stimuli_manifest.csv")


def gen_rolling_droplet(ctrl: PhysicsController) -> list[dict]:
    """A: rolling droplet, 8 variantes."""
    ctrl.set_scene(material="liquid", interaction="roll")
    rows = []
    # punto canonico
    variants = [
        ("canonical",   dict(wetness=0.7, continuity=0.5, granularity=0.3)),
        ("very_wet",    dict(wetness=1.0, continuity=0.5, granularity=0.3)),
        ("dry_drip",    dict(wetness=0.2, continuity=0.2, granularity=0.5)),
        ("slow_roll",   dict(wetness=0.7, continuity=0.15, granularity=0.3)),
        ("fast_roll",   dict(wetness=0.7, continuity=0.85, granularity=0.3)),
        ("grainy_path", dict(wetness=0.7, continuity=0.5, granularity=0.9)),
        ("smooth_path", dict(wetness=0.7, continuity=0.5, granularity=0.0)),
        ("big_droplet", dict(wetness=0.7, continuity=0.5, granularity=0.3, rigidity=0.0)),
    ]
    for name, mods in variants:
        wav = ctrl.render(**mods)
        q = quality_verdict(wav, SAMPLE_RATE)
        path = OUT_DIR / f"A_rolling_droplet__{name}.wav"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), wav, SAMPLE_RATE)
        rows.append(dict(combo="rolling_droplet", variant=name, path=str(path),
                         wetness=mods.get("wetness", ""), continuity=mods.get("continuity", ""),
                         granularity=mods.get("granularity", ""), rigidity=mods.get("rigidity", ""),
                         rms_db=round(q.rms_db, 2), dyn_db=round(q.dynamic_range_db, 2),
                         verdict=q.verdict))
        print(f"  A {name:15s}  rms={q.rms_db:6.1f} dyn={q.dynamic_range_db:5.1f}  {q.verdict}")
    return rows


def gen_liquid_rock_impact(ctrl: PhysicsController) -> list[dict]:
    """B: rock impact + liquid splash overlay, 8 variantes."""
    rows = []
    variants = [
        ("canonical",     0.55, dict(wetness=0.7, resonance=0.4)),
        ("dry_impact",    0.05, dict(wetness=0.1, resonance=0.5)),
        ("full_splash",   0.9,  dict(wetness=0.95, resonance=0.3)),
        ("hard_rock",     0.45, dict(wetness=0.6, resonance=0.7, rigidity=0.9)),
        ("soft_resonant", 0.5,  dict(wetness=0.6, resonance=0.95, rigidity=0.2)),
        ("dampened",      0.55, dict(wetness=0.7, resonance=0.1)),
        ("watery_thunk",  0.75, dict(wetness=0.9, resonance=0.3, rigidity=0.4)),
        ("dry_resonant",  0.2,  dict(wetness=0.2, resonance=0.85)),
    ]
    for name, ow, mods in variants:
        ctrl.set_scene(material="rock", interaction="impact",
                       overlay_material="liquid", overlay_interaction="splash",
                       overlay_weight=ow)
        wav = ctrl.render(**mods)
        q = quality_verdict(wav, SAMPLE_RATE)
        path = OUT_DIR / f"B_liquid_rock_impact__{name}.wav"
        sf.write(str(path), wav, SAMPLE_RATE)
        rows.append(dict(combo="liquid_rock_impact", variant=name, path=str(path),
                         overlay_weight=ow,
                         wetness=mods.get("wetness", ""), resonance=mods.get("resonance", ""),
                         rigidity=mods.get("rigidity", ""),
                         rms_db=round(q.rms_db, 2), dyn_db=round(q.dynamic_range_db, 2),
                         verdict=q.verdict))
        print(f"  B {name:15s} ow={ow:.2f}  rms={q.rms_db:6.1f} dyn={q.dynamic_range_db:5.1f}  {q.verdict}")
    return rows


def gen_wet_gravel_scrape(ctrl: PhysicsController) -> list[dict]:
    """C: gravel scrape + liquid pour overlay, 8 variantes."""
    rows = []
    variants = [
        ("canonical",      0.35, dict(wetness=0.6, granularity=0.7)),
        ("dry_gravel",     0.05, dict(wetness=0.1, granularity=0.7)),
        ("muddy",          0.7,  dict(wetness=0.95, granularity=0.4)),
        ("fine_grain",     0.35, dict(wetness=0.6, granularity=0.2)),
        ("coarse_grain",   0.35, dict(wetness=0.6, granularity=1.0)),
        ("slow_scrape",    0.3,  dict(wetness=0.5, granularity=0.7, continuity=0.2)),
        ("fast_scrape",    0.35, dict(wetness=0.5, granularity=0.7, continuity=0.9)),
        ("flooded",        0.85, dict(wetness=0.95, granularity=0.6)),
    ]
    for name, ow, mods in variants:
        ctrl.set_scene(material="gravel", interaction="scrape",
                       overlay_material="liquid", overlay_interaction="pour",
                       overlay_weight=ow)
        wav = ctrl.render(**mods)
        q = quality_verdict(wav, SAMPLE_RATE)
        path = OUT_DIR / f"C_wet_gravel_scrape__{name}.wav"
        sf.write(str(path), wav, SAMPLE_RATE)
        rows.append(dict(combo="wet_gravel_scrape", variant=name, path=str(path),
                         overlay_weight=ow,
                         wetness=mods.get("wetness", ""), granularity=mods.get("granularity", ""),
                         continuity=mods.get("continuity", ""),
                         rms_db=round(q.rms_db, 2), dyn_db=round(q.dynamic_range_db, 2),
                         verdict=q.verdict))
        print(f"  C {name:15s} ow={ow:.2f}  rms={q.rms_db:6.1f} dyn={q.dynamic_range_db:5.1f}  {q.verdict}")
    return rows


def main() -> int:
    ctrl = PhysicsController(seed=42, duration_s=5.0)

    # Limpiar directorio anterior
    for f in OUT_DIR.glob("*.wav"):
        f.unlink()

    print("=== A: ROLLING DROPLET (la estrella) ===")
    rows_a = gen_rolling_droplet(ctrl)
    print("\n=== B: LIQUID ROCK IMPACT ===")
    rows_b = gen_liquid_rock_impact(ctrl)
    print("\n=== C: WET GRAVEL SCRAPE ===")
    rows_c = gen_wet_gravel_scrape(ctrl)

    all_rows = rows_a + rows_b + rows_c
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    # union de keys para tener todas las cols
    all_keys = []
    for r in all_rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote {len(all_rows)} estimulos a {OUT_DIR}")
    print(f"Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
