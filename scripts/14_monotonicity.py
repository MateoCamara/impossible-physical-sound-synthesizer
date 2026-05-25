"""Demuestra numericamente la monotonia de los sweeps fisicos.

Para cada escena imposible (rolling_droplet, liquid_rock_impact, wet_gravel_scrape)
y cada knob fisico (wetness, granularity, rigidity, resonance, continuity),
genera un sweep de 7 puntos uniformes en [0, 1] y calcula:

  - RMS dBFS
  - Dynamic range dB
  - Spectral centroid Hz
  - Spectral flatness
  - Log-spec distance al punto medio (amount=0.5)

Despues calcula la correlacion de Spearman entre el valor del knob y cada
metrica. Si |rho| > 0.7 para al menos una metrica, el knob es 'monotonic'
en sentido practico: produce cambio acustico ordenado y por tanto los
sweeps soportan un "mas X" perceptualmente coherente.

Output:
  results/monotonicity/<scene>__<knob>__metrics.csv  (7 filas, todas metricas)
  results/monotonicity/summary.csv                    (rho Spearman por (scene, knob, metric))
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import SAMPLE_RATE
from impossible_mix.metrics.quality import (
    crest_factor_db,
    dynamic_range_db,
    log_spec_distance,
    peak_dbfs,
    rms_dbfs,
    spectral_centroid_hz,
    spectral_flatness,
)
from impossible_mix.physics_controller import KNOBS, PhysicsController


SCENES = [
    {"name": "A_rolling_droplet",
     "scene": dict(material="liquid", interaction="roll")},
    {"name": "B_liquid_rock_impact",
     "scene": dict(material="rock", interaction="impact",
                   overlay_material="liquid", overlay_interaction="splash",
                   overlay_weight=0.55)},
    {"name": "C_wet_gravel_scrape",
     "scene": dict(material="gravel", interaction="scrape",
                   overlay_material="liquid", overlay_interaction="pour",
                   overlay_weight=0.35)},
]

N_STEPS = 7  # puntos por sweep
OUT_DIR = Path("results/monotonicity")


def measure(wav: np.ndarray, sr: int, ref_wav: np.ndarray | None = None) -> dict[str, float]:
    """Mide todas las metricas relevantes para un wav."""
    m = dict(
        rms_db=rms_dbfs(wav),
        peak_db=peak_dbfs(wav),
        crest_db=crest_factor_db(wav),
        dyn_db=dynamic_range_db(wav, sr=sr),
        centroid_hz=spectral_centroid_hz(wav, sr),
        flatness=spectral_flatness(wav, sr),
    )
    if ref_wav is not None:
        m["log_spec_dist_ref"] = log_spec_distance(ref_wav, wav, sr)
    return m


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    values = np.linspace(0.0, 1.0, N_STEPS).tolist()

    for scene_def in SCENES:
        sname = scene_def["name"]
        print(f"\n=== {sname} ===")
        ctrl = PhysicsController(seed=42, duration_s=5.0)
        ctrl.set_scene(**scene_def["scene"])
        ref_wav = ctrl.render(wetness=0.5, granularity=0.5)  # baseline neutral
        for knob in KNOBS:
            metrics_per_step = []
            for v in values:
                # Render con solo este knob movido, resto en 0.5
                fixed = {k: 0.5 for k in KNOBS}
                fixed[knob] = v
                wav = ctrl.render(**fixed)
                m = measure(wav, SAMPLE_RATE, ref_wav=ref_wav)
                m["knob_value"] = v
                metrics_per_step.append(m)

            # Escribir CSV por (scene, knob)
            csv_path = OUT_DIR / f"{sname}__{knob}__metrics.csv"
            keys = ["knob_value"] + [k for k in metrics_per_step[0] if k != "knob_value"]
            with csv_path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(metrics_per_step)

            # Calcular Spearman para cada metrica
            knob_vals = np.array([m["knob_value"] for m in metrics_per_step])
            line_parts = [f"  {knob:12s}:"]
            for metric in ["rms_db", "dyn_db", "centroid_hz", "flatness", "log_spec_dist_ref"]:
                vals = np.array([m[metric] for m in metrics_per_step])
                if np.std(vals) < 1e-6:
                    rho, pval = 0.0, 1.0
                else:
                    rho, pval = spearmanr(knob_vals, vals)
                summary_rows.append(dict(
                    scene=sname, knob=knob, metric=metric,
                    spearman_rho=round(float(rho), 3),
                    spearman_p=round(float(pval), 4),
                    is_monotonic=abs(float(rho)) > 0.7,
                ))
                marker = "**" if abs(rho) > 0.7 else "  "
                line_parts.append(f" {marker}{metric}={rho:+.2f}{marker}")
            print(" ".join(line_parts))

    # Summary
    summary_path = OUT_DIR / "summary.csv"
    with summary_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Resumen agregado
    print(f"\n=== Resumen agregado ===")
    by_knob = {}
    for r in summary_rows:
        key = (r["scene"], r["knob"])
        if r["is_monotonic"]:
            by_knob.setdefault(key, []).append(r["metric"])
    print(f"Knobs con al menos una metrica monotonica (|rho|>0.7):")
    for (scene, knob), metrics in sorted(by_knob.items()):
        print(f"  {scene} x {knob:12s} : {metrics}")

    n_monotonic = sum(1 for r in summary_rows if r["is_monotonic"])
    total_knobs = len(SCENES) * len(KNOBS)
    knobs_with_signal = len(by_knob)
    print(f"\n  Total mediciones monotonicas: {n_monotonic} de {len(summary_rows)}")
    print(f"  Knobs con senal monotonica: {knobs_with_signal} de {total_knobs}")
    print(f"\nCSVs en {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
