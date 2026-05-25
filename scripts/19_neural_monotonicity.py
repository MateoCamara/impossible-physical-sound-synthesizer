"""Mide la monotonia de los sweeps NEURALES generados anteriormente con el
controlador del metodo A (EnCodec direcciones latentes) y compara con la
monotonia del marco fisico.

Sweeps disponibles en outputs/sweeps/ ya tienen 6 puntos cada uno.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.metrics.quality import (
    dynamic_range_db,
    rms_dbfs,
    spectral_centroid_hz,
    spectral_flatness,
)


SWEEPS_DIR = Path("outputs/sweeps")
OUT_DIR = Path("results/monotonicity")


def parse_amount(filename: str) -> float | None:
    # Patrones: 'xxx__knob+1.50.wav' o 'xxx__knob+0.00.wav'
    m = re.search(r"__\w+([+\-]\d+\.\d+)\.wav$", filename)
    if m:
        return float(m.group(1))
    return None


def analyze_sweep_dir(sweep_dir: Path) -> dict:
    wavs = sorted(sweep_dir.glob("*.wav"))
    if not wavs:
        return None
    points = []
    for w in wavs:
        amt = parse_amount(w.name)
        if amt is None:
            continue
        y, sr = sf.read(str(w), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        points.append(dict(
            amount=amt,
            rms_db=rms_dbfs(y),
            dyn_db=dynamic_range_db(y, sr=sr),
            centroid_hz=spectral_centroid_hz(y, sr),
            flatness=spectral_flatness(y, sr),
        ))
    if len(points) < 3:
        return None
    df = pd.DataFrame(sorted(points, key=lambda x: x["amount"]))
    amounts = df["amount"].values
    rhos = {}
    for col in ["rms_db", "dyn_db", "centroid_hz", "flatness"]:
        if df[col].std() < 1e-6:
            rhos[col] = 0.0
        else:
            rho, _ = spearmanr(amounts, df[col].values)
            rhos[col] = round(float(rho), 3)
    return dict(
        sweep=sweep_dir.name,
        n_points=len(points),
        rhos=rhos,
        max_abs_rho=round(max(abs(v) for v in rhos.values()), 3),
        is_monotonic=any(abs(v) > 0.7 for v in rhos.values()),
    )


def main() -> int:
    sweeps = [d for d in SWEEPS_DIR.iterdir() if d.is_dir()]
    print(f"Encontrados {len(sweeps)} sweeps neurales en {SWEEPS_DIR}")
    rows = []
    for d in sweeps:
        r = analyze_sweep_dir(d)
        if r is None:
            continue
        rows.append(r)
        marker = "**" if r["is_monotonic"] else "  "
        print(f"  {marker}{r['sweep']:55s}  rhos={r['rhos']}  max|rho|={r['max_abs_rho']}")

    n_mono = sum(r["is_monotonic"] for r in rows)
    print(f"\nNEURAL sweeps monotonicos: {n_mono}/{len(rows)}")

    # Cargar tambien la monotonia del marco fisico
    physics_summary = pd.read_csv(OUT_DIR / "summary.csv")
    physics_pairs = (physics_summary.assign(abs_rho=physics_summary.spearman_rho.abs())
                     .groupby(["scene","knob"])["abs_rho"].max().reset_index())
    n_physics_mono = int((physics_pairs.abs_rho > 0.7).sum())
    n_physics_total = len(physics_pairs)
    print(f"PHYSICS pairs monotonicos: {n_physics_mono}/{n_physics_total}")

    # Tabla comparativa
    cmp = pd.DataFrame([
        dict(method="PHYSICS", n_pairs=n_physics_total, n_monotonic=n_physics_mono,
             pct_monotonic=round(100 * n_physics_mono / n_physics_total, 1)),
        dict(method="NEURAL (latent)", n_pairs=len(rows), n_monotonic=n_mono,
             pct_monotonic=round(100 * n_mono / max(len(rows),1), 1)),
    ])
    cmp.to_csv(OUT_DIR / "neural_vs_physics_monotonicity.csv", index=False)
    print("\n=== Comparativa monotonia ===")
    print(cmp.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
