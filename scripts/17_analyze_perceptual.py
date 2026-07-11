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
  - Output: results/perceptual/{summary.csv, identification.csv, friedman.csv,
    knob_correlations.csv, boxplot.png}

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
MANIFEST_PATH = Path("perceptual_test/stimuli_manifest.csv")

# Columnas de "knobs" fisicos del manifest (perceptual_test/stimuli_manifest.csv)
# usadas para la correlacion Spearman con impossibility_likert.
KNOB_COLUMNS = ["wetness", "continuity", "granularity", "rigidity",
                "overlay_weight", "resonance"]


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

# Alias de nombres de combo que aparecen en distintos scripts del pipeline.
# scripts/06_generate_method_a.py (y 07/08/09, mas antiguos) usan "rolling_drop",
# mientras que scripts/13_final_stimuli.py, 16_build_perceptual_form.py y el
# manifest commiteado (perceptual_test/stimuli_manifest.csv) usan
# "rolling_droplet". Si el CSV de respuestas llega con la variante vieja,
# sin normalizar el `.get(combo, [])` de ground truth devuelve `[]` en
# silencio y la tasa de identificacion queda en 0 sin ningun error visible.
COMBO_ALIASES = {
    "rolling_drop": "rolling_droplet",
}


def normalize_combo(s):
    """Normaliza un valor de combo: strip + lower + resolucion de alias.

    Deja pasar NaN/None tal cual (no hay nada que normalizar).
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return s
    key = str(s).strip().lower()
    return COMBO_ALIASES.get(key, key)


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
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "combo" in out.columns:
        raw = out["combo"]
        normalized = raw.apply(normalize_combo)
        aliased = sorted(set(raw[normalized != raw].astype(str)))
        if aliased:
            print(f">> Normalizando alias de combo detectados en las respuestas: {aliased} "
                  f"-> {[normalize_combo(a) for a in aliased]}")
        out["combo"] = normalized
    return out


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
    # Defensa extra: normalizar de nuevo por si esta funcion se llama sobre un
    # df que no paso por load_all_responses (p.ej. tests).
    df["combo"] = df["combo"].apply(normalize_combo)

    unknown = sorted(
        set(df["combo"]) - set(GROUND_TRUTH_MATERIAL) | set(df["combo"]) - set(GROUND_TRUTH_INTERACTION)
    )
    if unknown:
        print(f"!! AVISO: combo(s) sin ground truth definido tras normalizar: {unknown}. "
              f"material_ok/interaction_ok seran False para estas filas -- "
              f"revisa COMBO_ALIASES o GROUND_TRUTH_MATERIAL/GROUND_TRUTH_INTERACTION.")

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


def knob_correlations(df: pd.DataFrame, manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Correlacion Spearman entre 'impossibility' percibida y los knobs fisicos
    del manifest (wetness/continuity/granularity/rigidity/overlay_weight/
    resonance), calculada por separado dentro de cada combo.

    Defensivo: si falta el manifest, las columnas combo/variant, o no hay
    columnas de knobs reconocibles, emite un warning y devuelve un
    DataFrame vacio sin lanzar excepcion.

    CAVEAT estadistico: la correlacion agrupa (pool) las respuestas de todos
    los oyentes y variantes dentro de cada combo, ignorando la dependencia
    por medidas repetidas del mismo oyente. Los p-valores son por tanto
    anticonservadores: usarlos como descriptivos. Para inferencia formal,
    agregar por variante (mediana entre oyentes) o calcular rho por oyente
    y contrastar la distribucion. Se reporta n_listeners por fila para
    hacer visible el grado de pooling.
    """
    if not manifest_path.exists():
        print(f"!! AVISO: no existe el manifest {manifest_path}; se omite knob_correlations.")
        return pd.DataFrame()
    try:
        manifest = pd.read_csv(manifest_path)
    except Exception as e:
        print(f"!! AVISO: error leyendo manifest {manifest_path}: {e}; se omite knob_correlations.")
        return pd.DataFrame()

    if "combo" not in manifest.columns or "variant" not in manifest.columns:
        print(f"!! AVISO: manifest {manifest_path} sin columnas combo/variant; se omite knob_correlations.")
        return pd.DataFrame()
    if "combo" not in df.columns or "variant" not in df.columns or "impossibility_likert" not in df.columns:
        print("!! AVISO: respuestas sin columnas combo/variant/impossibility_likert; se omite knob_correlations.")
        return pd.DataFrame()

    knob_cols = [c for c in KNOB_COLUMNS if c in manifest.columns]
    if not knob_cols:
        print(f"!! AVISO: manifest {manifest_path} sin ninguna columna de knob conocida "
              f"({KNOB_COLUMNS}); se omite knob_correlations.")
        return pd.DataFrame()

    manifest = manifest.copy()
    manifest["combo"] = manifest["combo"].apply(normalize_combo)
    for c in knob_cols:
        manifest[c] = pd.to_numeric(manifest[c], errors="coerce")

    resp = df.copy()
    resp["combo"] = resp["combo"].apply(normalize_combo)

    merged = resp.merge(
        manifest[["combo", "variant"] + knob_cols],
        on=["combo", "variant"],
        how="left",
    )

    listener_col = next((c for c in ("listener_id", "participant", "listener") if c in merged.columns), None)
    rows = []
    for combo, sub in merged.groupby("combo"):
        for knob in knob_cols:
            cols = ["impossibility_likert", knob] + ([listener_col] if listener_col else [])
            valid = sub[cols].dropna(subset=["impossibility_likert", knob])
            if valid[knob].nunique() < 3 or len(valid) < 3:
                continue
            try:
                rho, pval = spearmanr(valid["impossibility_likert"], valid[knob])
            except Exception as e:
                print(f"!! AVISO: spearmanr fallo para combo={combo} knob={knob}: {e}")
                continue
            if rho != rho:  # NaN check
                continue
            rows.append(dict(
                combo=combo, knob=knob, n=len(valid),
                n_listeners=(int(valid[listener_col].nunique()) if listener_col else -1),
                rho=round(float(rho), 3),
                pval=round(float(pval), 4),
                sig=("**" if pval < 0.05 else ""),
            ))

    if not rows:
        print("!! AVISO: no hubo suficientes datos (>=3 valores distintos de knob) "
              "para calcular ninguna correlacion Spearman.")
    return pd.DataFrame(rows)


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
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                     help="CSV con los knobs fisicos por (combo, variant), "
                          "para la correlacion Spearman.")
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

    corr = knob_correlations(df, manifest_path=args.manifest)
    print("\n=== Correlacion Spearman: impossibility_likert vs knobs del manifest ===")
    if corr.empty:
        print("  (sin correlaciones -- ver avisos arriba)")
    else:
        corr.to_csv(args.out / "knob_correlations.csv", index=False)
        print(corr.to_string(index=False))

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
