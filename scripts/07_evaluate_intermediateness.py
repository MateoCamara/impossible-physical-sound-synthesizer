"""Evalua intermediateness y distancia a centroides de TODOS los hibridos
generados (segun el manifest) y produce una tabla CSV con metricas por clip.

Es la metrica objetiva mas barata (no requiere ViSQOL/FAD/CLAP) y la primera
que se reporta en el abstract del Dia 9.

Definicion:
  Sea z el embedding del hibrido, mu_src y mu_tgt los centroides de las
  clases ancla y objetivo (en la dimension material o interaction).

  intermediateness_material = ||z - mu_src_mat|| / (||z - mu_src_mat|| + ||z - mu_tgt_mat||)

  - 0.5 = perfectamente entre ambas clases
  - <0.5 = sesgado al ancla (peor para mezcla imposible)
  - >0.5 = sesgado al objetivo
  - 0 o 1 = colapsado a una clase pura
"""
from __future__ import annotations

import argparse
import csv
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

    manifest = pd.read_csv(args.manifest)
    rows = []
    for _, r in manifest.iterrows():
        combo = r["combo"]
        path = Path(r["path"])
        if not path.exists():
            continue
        src_mat, src_int, tgt_mat, tgt_int = COMBO_DEFS[combo]
        mu_smat = bank.material_centroids[src_mat]
        mu_tmat = bank.material_centroids[tgt_mat]
        mu_sint = bank.interaction_centroids[src_int]
        mu_tint = bank.interaction_centroids[tgt_int]

        z = encode_wav(encoder, path)
        d_smat = (z - mu_smat).norm().item()
        d_tmat = (z - mu_tmat).norm().item()
        d_sint = (z - mu_sint).norm().item()
        d_tint = (z - mu_tint).norm().item()
        inter_mat = d_smat / (d_smat + d_tmat + 1e-9)
        inter_int = d_sint / (d_sint + d_tint + 1e-9) if src_int != tgt_int else 0.5
        rows.append(dict(
            method=r["method"], combo=combo, anchor=r["anchor"],
            alpha=r["alpha"], beta=r["beta"], gamma=r["gamma"],
            d_src_mat=d_smat, d_tgt_mat=d_tmat,
            d_src_int=d_sint, d_tgt_int=d_tint,
            inter_mat=inter_mat, inter_int=inter_int,
            path=str(path),
        ))

    out_path = args.out or args.manifest.with_name(args.manifest.stem + "_metrics.csv")
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")

    # Resumen por (method, combo)
    df = pd.DataFrame(rows)
    print("\nResumen por (method, combo):")
    summary = df.groupby(["method", "combo"]).agg(
        n=("inter_mat", "count"),
        inter_mat_mean=("inter_mat", "mean"),
        inter_mat_std=("inter_mat", "std"),
        d_tgt_mat_mean=("d_tgt_mat", "mean"),
    ).round(3)
    print(summary.to_string())

    print("\nMejores 10 hibridos por intermediateness material (cerca de 0.5):")
    df["score"] = (df["inter_mat"] - 0.5).abs()
    cols = ["method", "combo", "alpha", "beta", "gamma", "inter_mat", "d_tgt_mat"]
    print(df.sort_values("score").head(10)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
