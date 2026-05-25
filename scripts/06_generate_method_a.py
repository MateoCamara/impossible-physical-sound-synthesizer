"""Genera el primer lote de hibridos con Metodo A (direcciones latentes).

Para cada combinacion objetivo (rolling drop / liquid rock impact / wet
gravel scrape), selecciona un ancla por clase y aplica direcciones de
material e interaccion con un grid de pesos. Tambien produce baselines.

Salida:
  outputs/method_a/<run>/<combo>_a{alpha}_b{beta}_<anchor>.wav
  outputs/baseline_sum/<run>/...
  results/method_a/generation_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import OUTPUTS_DIR, RESULTS_DIR, SAMPLE_RATE
from impossible_mix.data.dataset import load_corpus
from impossible_mix.data.labels import MODIFIERS
from impossible_mix.encoders.rave_wrapper import get_encoder
from impossible_mix.methods.centroids import build_centroids_by_label, centroid
from impossible_mix.methods.method_a_directions import (
    DirectionBank,
    MethodA,
    baseline_additive_sum,
    baseline_linear_interp,
    build_direction_bank,
)
from impossible_mix.methods.base import HybridSpec


# Combinaciones objetivo del plan (3 anclas obligatorias del abstract).
# `src_*` es el bucket del clip ancla; `tgt_*` lo que queremos sintetizar.
TARGETS: list[dict[str, Any]] = [
    {
        "name": "rolling_drop",
        "anchor_filter": ("metal", "roll"),   # mejor estructura temporal disponible
        "tgt_material": "liquid",
        "tgt_interaction": "roll",            # mantener interaccion roll
        "mods": {"wetness": 5, "granularity": 4, "continuity": 5},
    },
    {
        "name": "liquid_rock_impact",
        "anchor_filter": ("rock", "impact"),
        "tgt_material": "liquid",
        "tgt_interaction": "impact",
        "mods": {"wetness": 5, "rigidity": 4},
    },
    {
        "name": "wet_gravel_scrape",
        "anchor_filter": ("rock", "scrape"),   # gravel x scrape no esta en corpus -> usar rock como ancla
        "tgt_material": "gravel",
        "tgt_interaction": "scrape",
        "mods": {"wetness": 5, "granularity": 5},
    },
]

ALPHAS = [0.3, 0.6, 0.9]
BETAS = [0.3, 0.6, 0.9]
GAMMAS = [0.0, 0.5, 1.0]


def pick_anchor(bundle, material: str, interaction: str, idx_offset: int = 0):
    """Selecciona el clip mas cercano al centroide del bucket como ancla canonica.
    idx_offset permite variar el ancla para diversidad.
    """
    Z = bundle.Z_mean
    mask = torch.tensor([
        m == material and i == interaction
        for m, i in zip(bundle.materials, bundle.interactions)
    ])
    if mask.sum() == 0:
        raise ValueError(f"No hay clips en bucket {material} x {interaction}")
    cent = centroid(Z, mask)
    dists = (Z - cent.unsqueeze(0)).norm(dim=-1)
    dists = dists.masked_fill(~mask, float("inf"))
    order = torch.argsort(dists)
    return int(order[idx_offset].item())


def save_wav_24k(wav: torch.Tensor, path: Path, sr_in: int = 24_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_44 = torchaudio.functional.resample(
        wav.unsqueeze(0), sr_in, SAMPLE_RATE
    ).squeeze(0).numpy()
    sf.write(str(path), wav_44, SAMPLE_RATE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=f"run_{int(time.time())}")
    ap.add_argument("--quick", action="store_true",
                    help="solo 1 ancla por combo y un punto del grid (sanity)")
    args = ap.parse_args()

    print("Cargando corpus + encoder...")
    bundle = load_corpus()
    encoder = get_encoder("encodec")
    bank = build_direction_bank(bundle)
    method = MethodA(encoder, bundle, bank=bank)
    print(f"  N={bundle.n}  dim={bundle.Z_mean.shape[1]}")
    print(f"  bank materials={list(bank.material_centroids)}")
    print(f"  bank interactions={list(bank.interaction_centroids)}")
    print(f"  bank modifiers={list(bank.modifier_high)}")

    out_dir = OUTPUTS_DIR / "method_a" / args.run_name
    sum_dir = OUTPUTS_DIR / "baseline_sum" / args.run_name
    interp_dir = OUTPUTS_DIR / "baseline_interp" / args.run_name
    manifest_path = RESULTS_DIR / "method_a" / f"{args.run_name}_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    n_frames = encoder.expected_frames(5.0)

    for target in TARGETS:
        src_mat, src_int = target["anchor_filter"]
        anchors = [pick_anchor(bundle, src_mat, src_int, k) for k in range(1 if args.quick else 2)]
        for ai, anchor_idx in enumerate(anchors):
            z_anchor_seq = bundle.Z_seq[anchor_idx]   # (dim, T') - PRESERVA dinamica
            anchor_id = bundle.clip_ids[anchor_idx]
            print(f"\n>>> {target['name']}  ancla={anchor_id}  ({src_mat} x {src_int})")

            # Baselines (sobre la secuencia tambien)
            tgt_cent_mat = bank.material_centroids.get(target["tgt_material"])
            tgt_cent_int = bank.interaction_centroids.get(target["tgt_interaction"])
            if tgt_cent_mat is not None and tgt_cent_int is not None:
                # 1) Suma simple: z[:,t] = z_anchor[:,t] + mu_tgt_mat + mu_tgt_int
                z_sum_seq = z_anchor_seq + tgt_cent_mat.unsqueeze(-1) + tgt_cent_int.unsqueeze(-1)
                wav_sum = encoder.decode_sequence(z_sum_seq)
                p = sum_dir / f"{target['name']}_anchor{ai}.wav"
                save_wav_24k(wav_sum, p, sr_in=encoder.sr_expected)
                rows.append(dict(method="baseline_sum", combo=target["name"], anchor=anchor_id,
                                 alpha="", beta="", gamma="", path=str(p)))

                # 2) Interpolacion convexa hacia el centroide combinado (aplicada frame a frame)
                tgt_mean_vec = (tgt_cent_mat + tgt_cent_int) / 2  # (dim,)
                z_interp_seq = 0.5 * z_anchor_seq + 0.5 * tgt_mean_vec.unsqueeze(-1)
                wav_interp = encoder.decode_sequence(z_interp_seq)
                p = interp_dir / f"{target['name']}_anchor{ai}.wav"
                save_wav_24k(wav_interp, p, sr_in=encoder.sr_expected)
                rows.append(dict(method="baseline_interp", combo=target["name"], anchor=anchor_id,
                                 alpha="", beta="", gamma="", path=str(p)))

            # Metodo A: grid (o un solo punto si --quick)
            alpha_set = [0.6] if args.quick else ALPHAS
            beta_set = [0.6] if args.quick else BETAS
            gamma_set = [0.5] if args.quick else GAMMAS
            for alpha in alpha_set:
                for beta in beta_set:
                    for gamma in gamma_set:
                        spec = HybridSpec(
                            target_material=target["tgt_material"],
                            target_interaction=target["tgt_interaction"],
                            properties={
                                "src_material": src_mat,
                                "src_interaction": src_int,
                                **target["mods"],
                            },
                            weights={
                                "material": alpha,
                                "interaction": beta,
                                **{m: gamma for m in MODIFIERS if m in target["mods"]},
                            },
                        )
                        wav = method.generate(z_anchor_seq, spec)
                        tag = f"a{alpha}_b{beta}_g{gamma}_anchor{ai}"
                        p = out_dir / f"{target['name']}_{tag}.wav"
                        save_wav_24k(wav, p, sr_in=encoder.sr_expected)
                        rows.append(dict(method="A", combo=target["name"], anchor=anchor_id,
                                         alpha=alpha, beta=beta, gamma=gamma, path=str(p)))

    with manifest_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "combo", "anchor", "alpha", "beta", "gamma", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nGenerados {len(rows)} clips. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
