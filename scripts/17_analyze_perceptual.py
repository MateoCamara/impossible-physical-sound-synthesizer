"""Analisis estadistico de las respuestas del test perceptual.

Cada CSV que descarga el formulario tiene formato:
  listener,experience,stim_file,combo,variant,material_pred,interaction_pred,
  impossibility_likert,coherence_likert

Este script:
  - Carga todos los CSVs de perceptual_test/responses/
  - Calcula estadisticas descriptivas por (combo, variant) y por (combo)
  - Test no parametrico (Friedman) para detectar diferencias entre variantes
  - Tasa de identificacion de material/interaccion vs ground truth
  - Correlacion Spearman entre 'impossibility' percibida y wetness/overlay
  - Output: results/perceptual/{summary.csv, friedman.csv, plot.png}

Uso:
    python scripts/17_analyze_perceptual.py
    python scripts/17_analyze_perceptual.py --responses /custom/path/responses/
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESPONSES_DIR = Path("perceptual_test/responses")
OUT_DIR = Path("results/perceptual")


# Ground truth para cada estimulo (lo que esperamos que el oyente reporte
# si "funciona perceptualmente"). Para imposibles esperamos al menos uno
# de los dos componentes (material primario u overlay) detectado.
GROUND_TRUTH_MATERIAL = {
    "rolling_droplet":     ["liquid"],                     # solo liquid
    "liquid_rock_impact":  ["rock", "liquid"],             # cualquiera valido
    "wet_gravel_scrape":   ["gravel", "liquid"],
}
GROUND_TRUTH_INTERACTION = {
    "rolling_droplet":     ["roll", "drip"],
    "liquid_rock_impact":  ["impact", "splash"],
    "wet_gravel_scrape":   ["scrape", "pour"],
}


def load_all_responses(d: Path) -> pd.DataFrame:
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
    frames = []
    csv_files = sorted(d.glob("*.csv"))
    if not csv_files:
        print(f"!! No hay CSVs en {d}. Esperando respuestas del formulario.")
        return pd.DataFrame()
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["src_file"] = f.name
            frames.append(df)
        except Exception as e:
            print(f"!! Error leyendo {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Estadisticas descriptivas por (combo, variant)."""
    g = df.groupby(["combo", "variant"]).agg(
        n=("listener", "count"),
        n_listeners=("listener", "nunique"),
        impossibility_mean=("impossibility_likert", "mean"),
        impossibility_std=("impossibility_likert", "std"),
        coherence_mean=("coherence_likert", "mean"),
        coherence_std=("coherence_likert", "std"),
    ).round(3).reset_index()
    return g


def identification_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de aciertos material e interaccion contra ground truth (any-of)."""
    df = df.copy()
    df["material_ok"] = df.apply(
        lambda r: r["material_pred"] in GROUND_TRUTH_MATERIAL.get(r["combo"], []),
        axis=1,
    )
    df["interaction_ok"] = df.apply(
        lambda r: r["interaction_pred"] in GROUND_TRUTH_INTERACTION.get(r["combo"], []),
        axis=1,
    )
    g = df.groupby(["combo"]).agg(
        n=("listener", "count"),
        n_listeners=("listener", "nunique"),
        material_acc=("material_ok", "mean"),
        interaction_acc=("interaction_ok", "mean"),
    ).round(3).reset_index()
    return g


def friedman_per_combo(df: pd.DataFrame) -> list[dict]:
    """Friedman: en cada combo, las 8 variantes producen distinto Likert?"""
    rows = []
    for combo in df["combo"].unique():
        sub = df[df["combo"] == combo]
        # Wide format: filas = oyentes, cols = variantes
        for metric in ["impossibility_likert", "coherence_likert"]:
            wide = sub.pivot_table(index="listener", columns="variant", values=metric, aggfunc="mean")
            wide = wide.dropna(how="any")
            if wide.shape[0] < 3 or wide.shape[1] < 3:
                rows.append(dict(combo=combo, metric=metric, n_listeners=wide.shape[0],
                                 stat=None, pval=None, note="too few listeners or variants"))
                continue
            try:
                stat, pval = friedmanchisquare(*[wide[c].values for c in wide.columns])
                rows.append(dict(combo=combo, metric=metric, n_listeners=wide.shape[0],
                                 n_variants=wide.shape[1],
                                 stat=round(float(stat), 3),
                                 pval=round(float(pval), 4),
                                 sig=("**" if pval < 0.05 else "")))
            except Exception as e:
                rows.append(dict(combo=combo, metric=metric, n_listeners=wide.shape[0],
                                 stat=None, pval=None, note=str(e)))
    return rows


def plot_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    """Boxplot Likert por variante, agrupado por combo."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey="row")
    combos = ["rolling_droplet", "liquid_rock_impact", "wet_gravel_scrape"]
    metrics = ["impossibility_likert", "coherence_likert"]
    titles = {"impossibility_likert": "Impossibility",
              "coherence_likert": "Coherence"}
    for r, metric in enumerate(metrics):
        for c, combo in enumerate(combos):
            ax = axes[r, c]
            sub = df[df["combo"] == combo]
            variants = sorted(sub["variant"].unique())
            data = [sub[sub["variant"] == v][metric].values for v in variants]
            if not any(len(d) > 0 for d in data):
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue
            ax.boxplot(data, labels=variants, showmeans=True)
            ax.set_title(f"{combo.replace('_',' ')} -- {titles[metric]}")
            ax.set_ylabel("Likert 1-7")
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylim(0.5, 7.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, default=RESPONSES_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_all_responses(args.responses)
    if df.empty:
        print("Sin datos. El analisis correra cuando haya CSVs en", args.responses)
        return 0
    print(f"Total respuestas: {len(df)} | oyentes unicos: {df['listener'].nunique()}")

    summary = basic_summary(df)
    summary.to_csv(args.out / "summary.csv", index=False)
    print("\n=== Resumen por (combo, variant) ===")
    print(summary.to_string(index=False))

    ident = identification_rates(df)
    ident.to_csv(args.out / "identification.csv", index=False)
    print("\n=== Tasas de identificacion (material e interaccion vs ground truth) ===")
    print(ident.to_string(index=False))

    fr = friedman_per_combo(df)
    pd.DataFrame(fr).to_csv(args.out / "friedman.csv", index=False)
    print("\n=== Friedman por combo ===")
    for r in fr:
        print(f"  {r}")

    plot_boxplot(df, args.out / "boxplot.png")
    print(f"\nFiguras y CSVs en {args.out}")

    # Resumen de una linea para el abstract
    n_listeners = df["listener"].nunique()
    overall_imp = df["impossibility_likert"].mean()
    overall_coh = df["coherence_likert"].mean()
    print(f"\n>>> Para abstract: n={n_listeners} oyentes, "
          f"impossibility={overall_imp:.2f}/7, coherence={overall_coh:.2f}/7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
