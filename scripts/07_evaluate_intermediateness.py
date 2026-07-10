"""Evalua intermediateness y distancia a centroides de TODOS los hibridos
generados (segun el manifest) y produce una tabla CSV con metricas por clip.

Es la metrica objetiva mas barata (no requiere ViSQOL/FAD/CLAP) y la primera
que se reporta en el abstract del Dia 9.

Definicion:
  Sea z el embedding del hibrido, mu_src y mu_tgt los centroides de las
  clases ancla y objetivo (en la dimension material o interaction), todos
  estandarizados por z-score dimension a dimension con las estadisticas
  (media, desviacion) del corpus completo (`bundle.Z_mean`) ANTES de medir
  distancias. Sin esto, las dimensiones del espacio EnCodec con mayor
  varianza dominan la norma euclidea y las ratios dejan de ser comparables
  entre ejes.

  intermediateness_material = ||z - mu_src_mat|| / (||z - mu_src_mat|| + ||z - mu_tgt_mat||)

  - 0.5 = perfectamente entre ambas clases
  - <0.5 = sesgado al ancla (peor para mezcla imposible)
  - >0.5 = sesgado al objetivo
  - 0 o 1 = colapsado a una clase pura

  intermediateness_interaction solo esta definida cuando la interaccion
  fuente y la interaccion destino del combo difieren. En los tres combos
  actuales (COMBO_DEFS) src_interaction == tgt_interaction siempre (los
  centroides de interaccion fuente y destino son el MISMO centroide), asi
  que `inter_int` se reporta como NaN y `d_src_int`/`d_tgt_int` colapsan en
  una unica columna `d_int` (distancia al centroide de esa interaccion
  compartida). Si en el futuro se anade un combo con interacciones
  distintas, `inter_int` volvera a calcularse normalmente para ese combo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import RESULTS_DIR
from impossible_mix.data.dataset import load_corpus
from impossible_mix.encoders.rave_wrapper import get_encoder
from impossible_mix.methods.method_a_directions import build_direction_bank


# Mapeo de combo -> (src_material, src_interaction, tgt_material, tgt_interaction)
COMBO_DEFS = {
    "rolling_drop":       ("metal", "roll",   "liquid", "roll"),
    "liquid_rock_impact": ("rock",  "impact", "liquid", "impact"),
    "wet_gravel_scrape":  ("rock",  "scrape", "gravel", "scrape"),
}


def encode_wav(encoder, path: Path) -> torch.Tensor:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return encoder.encode_mean(torch.from_numpy(y), sr_in=sr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True,
                    help="results/method_a/<run>_manifest.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    bundle = load_corpus()
    encoder = get_encoder("encodec")
    bank = build_direction_bank(bundle)

    # Estandarizacion z-score por dimension con estadisticas del corpus
    # completo. Sin esto, las dimensiones de mayor varianza del espacio
    # EnCodec dominan la norma euclidea y las ratios de intermediateness
    # dejan de ser comparables entre ejes (ver docstring del modulo).
    Z = bundle.Z_mean
    mu_c = Z.mean(dim=0)
    sigma_c = Z.std(dim=0).clamp_min(1e-6)

    def standardize(x: torch.Tensor) -> torch.Tensor:
        return (x - mu_c) / sigma_c

    std_mat_centroids = {k: standardize(v) for k, v in bank.material_centroids.items()}
    std_int_centroids = {k: standardize(v) for k, v in bank.interaction_centroids.items()}

    manifest = pd.read_csv(args.manifest)
    rows = []
    for _, r in manifest.iterrows():
        combo = r["combo"]
        path = Path(r["path"])
        if not path.exists():
            continue
        src_mat, src_int, tgt_mat, tgt_int = COMBO_DEFS[combo]
        mu_smat = std_mat_centroids[src_mat]
        mu_tmat = std_mat_centroids[tgt_mat]
        mu_sint = std_int_centroids[src_int]
        mu_tint = std_int_centroids[tgt_int]

        z = standardize(encode_wav(encoder, path))
        d_smat = (z - mu_smat).norm().item()
        d_tmat = (z - mu_tmat).norm().item()
        inter_mat = d_smat / (d_smat + d_tmat + 1e-9)

        row = dict(
            method=r["method"], combo=combo, anchor=r["anchor"],
            alpha=r["alpha"], beta=r["beta"], gamma=r["gamma"],
            d_src_mat=d_smat, d_tgt_mat=d_tmat,
            inter_mat=inter_mat,
            path=str(path),
        )
        if src_int == tgt_int:
            # Mismo centroide de interaccion en ambos lados: la ratio no
            # mide nada (siempre 0.5 por construccion). Colapsamos las dos
            # distancias en una sola columna y marcamos inter_int como NaN.
            row["d_int"] = (z - mu_sint).norm().item()
            row["inter_int"] = float("nan")
        else:
            d_sint = (z - mu_sint).norm().item()
            d_tint = (z - mu_tint).norm().item()
            row["d_src_int"] = d_sint
            row["d_tgt_int"] = d_tint
            row["inter_int"] = d_sint / (d_sint + d_tint + 1e-9)
        rows.append(row)

    out_path = args.out or args.manifest.with_name(args.manifest.stem + "_metrics.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")

    # Resumen por (method, combo)
    print("\nResumen por (method, combo):")
    agg_kwargs = dict(
        n=("inter_mat", "count"),
        inter_mat_mean=("inter_mat", "mean"),
        inter_mat_std=("inter_mat", "std"),
        d_tgt_mat_mean=("d_tgt_mat", "mean"),
    )
    if df["inter_int"].notna().any():
        agg_kwargs["inter_int_mean"] = ("inter_int", "mean")
    summary = df.groupby(["method", "combo"]).agg(**agg_kwargs).round(3)
    print(summary.to_string())
    if df["inter_int"].isna().all():
        print("\n(inter_int es NaN para todos los combos: src_interaction == "
              "tgt_interaction en COMBO_DEFS, la metrica no esta definida; "
              "ver d_int por columna colapsada)")

    print("\nMejores 10 hibridos por intermediateness material (cerca de 0.5):")
    df["score"] = (df["inter_mat"] - 0.5).abs()
    cols = ["method", "combo", "alpha", "beta", "gamma", "inter_mat", "d_tgt_mat"]
    print(df.sort_values("score").head(10)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
