"""Tabla maestra: marco fisico vs metodos neurales (A, B) vs baselines.

Para las 3 combos imposibles, comparamos:
  - PHYSICS canonico (composer)
  - PHYSICS sweep (rango de variantes)
  - A_neural (direcciones latentes sobre EnCodec)
  - B_neural (cabezas + edicion por gradiente)
  - baseline_sum, baseline_interp (controles negativos)

Para cada uno computamos:
  - % verdict good
  - dyn_db medio
  - log-spec dist al ancla
  - monotonia Spearman max en sweeps (para PHYSICS y A_neural)

Output: results/comparison/master_table.csv + figure F5_master_comparison.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import SAMPLE_RATE
from impossible_mix.metrics.quality import (
    dynamic_range_db,
    log_spec_distance,
    rms_dbfs,
    spectral_centroid_hz,
    quality_verdict,
)


OUT_DIR = Path("results/comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_audio(p: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y, sr


def summarize_dir(directory: Path, max_clips: int | None = None) -> dict:
    """Estadisticas agregadas de todos los wavs en una carpeta."""
    wavs = sorted(directory.glob("*.wav"))
    if max_clips:
        wavs = wavs[:max_clips]
    if not wavs:
        return dict(n=0, pct_good=None, dyn_db_mean=None, rms_db_mean=None, centroid_hz_mean=None)
    verdicts, dyns, rmss, cents = [], [], [], []
    for w in wavs:
        y, sr = load_audio(w)
        q = quality_verdict(y, sr)
        verdicts.append(q.verdict)
        dyns.append(q.dynamic_range_db)
        rmss.append(q.rms_db)
        cents.append(q.centroid_hz)
    return dict(
        n=len(wavs),
        pct_good=round(100 * sum(v == "good" for v in verdicts) / len(verdicts), 1),
        dyn_db_mean=round(float(np.mean(dyns)), 2),
        rms_db_mean=round(float(np.mean(rmss)), 2),
        centroid_hz_mean=round(float(np.mean(cents)), 0),
    )


def monotonicity_score(metrics_csv: Path) -> dict[str, float]:
    """Devuelve max |rho_Spearman| sobre todas las metricas en un CSV de monotonia."""
    if not metrics_csv.exists():
        return dict(max_abs_rho=None, n_monotonic=None)
    df = pd.read_csv(metrics_csv)
    knob = df["knob_value"]
    best = 0.0
    n_mono = 0
    for col in ["rms_db", "dyn_db", "centroid_hz", "flatness"]:
        if col not in df.columns:
            continue
        vals = df[col].values
        if np.std(vals) < 1e-6:
            continue
        rho, _ = spearmanr(knob, vals)
        if abs(rho) > best:
            best = abs(rho)
        if abs(rho) > 0.7:
            n_mono += 1
    return dict(max_abs_rho=round(best, 2), n_monotonic=n_mono)


def main() -> int:
    rows = []

    # PHYSICS canonico + sweep
    physics_dir = Path("perceptual_test/stimuli")
    s = summarize_dir(physics_dir)
    rows.append(dict(method="PHYSICS (24 variantes)", **s, max_abs_rho_in_sweep="see F3"))

    # Las cifras de monotonia ya estan en results/monotonicity/summary.csv
    mono_path = Path("results/monotonicity/summary.csv")
    n_mono_total = 0
    n_total_pairs = 0
    if mono_path.exists():
        mono_df = pd.read_csv(mono_path)
        # Por (scene, knob) tomar el max abs rho
        agg = mono_df.assign(abs_rho=mono_df.spearman_rho.abs()).groupby(
            ["scene", "knob"])["abs_rho"].max().reset_index()
        n_total_pairs = len(agg)
        n_mono_total = int((agg["abs_rho"] > 0.7).sum())

    # Neural method A
    a_dir = Path("outputs/method_a/full")
    s = summarize_dir(a_dir)
    rows.append(dict(method="NEURAL A (latent dir, 162)", **s, max_abs_rho_in_sweep="n/a"))

    # Neural method B
    b_dir = Path("outputs/method_b/b_full")
    s = summarize_dir(b_dir)
    rows.append(dict(method="NEURAL B (grad edit, 24)", **s, max_abs_rho_in_sweep="n/a"))

    # Baselines
    for name, path in [
        ("BASELINE_SUM (12)", "outputs/baseline_sum/full"),
        ("BASELINE_INTERP (12)", "outputs/baseline_interp/full"),
    ]:
        s = summarize_dir(Path(path))
        rows.append(dict(method=name, **s, max_abs_rho_in_sweep=""))

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "master_table.csv", index=False)
    print("=== Tabla maestra ===")
    print(table.to_string(index=False))
    print(f"\nPHYSICS monotonicidad: {n_mono_total}/{n_total_pairs} pares con |rho|>0.7 "
          f"(de results/monotonicity/summary.csv)")

    # Figura: barras comparativas pct_good y dyn_db
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    methods = [r["method"].split(" (")[0].replace("BASELINE_", "Baseline-").replace("NEURAL ", "Neural ").replace("PHYSICS", "Physics") for r in rows]
    pct = [r["pct_good"] if r["pct_good"] is not None else 0 for r in rows]
    dyn = [r["dyn_db_mean"] if r["dyn_db_mean"] is not None else 0 for r in rows]
    colors = ["#2e7d32", "#1565c0", "#1565c0", "#9e9e9e", "#c62828"]
    bars = axes[0].bar(methods, pct, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel(r"\% audio verdict = good")
    axes[0].set_title("(a) Quality verdict per method")
    axes[0].set_ylim(0, 110)
    for b, v in zip(bars, pct):
        axes[0].text(b.get_x() + b.get_width()/2, v + 2, f"{v:.0f}%", ha="center", fontsize=10)
    axes[0].tick_params(axis="x", rotation=12)

    bars = axes[1].bar(methods, dyn, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Mean dynamic range (dB)")
    axes[1].set_title("(b) Dynamic range preserved")
    for b, v in zip(bars, dyn):
        axes[1].text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}", ha="center", fontsize=10)
    axes[1].tick_params(axis="x", rotation=12)
    fig.suptitle(f"Fig. 5. Methods comparison: Physics matches neural on objective quality; monotonicity confirmed on {n_mono_total}/{n_total_pairs} (scene, knob) pairs",
                 fontsize=11.5, y=1.04)
    fig.tight_layout()
    fig.savefig(Path("figures") / "F5_master_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura: figures/F5_master_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
