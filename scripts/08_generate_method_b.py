"""Genera hibridos con Metodo B (cabezas + edicion por gradiente) sobre las
mismas combinaciones objetivo y anclas que el Metodo A, para permitir
comparacion 1-vs-1.

Cabezas cargadas desde results/method_b/heads_fused.pt (entrenadas con
step->impact fusion).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import OUTPUTS_DIR, RESULTS_DIR, SAMPLE_RATE
from impossible_mix.data.dataset import load_corpus
from impossible_mix.encoders.rave_wrapper import get_encoder
from impossible_mix.methods.centroids import centroid
from impossible_mix.methods.method_a_directions import build_direction_bank
from impossible_mix.methods.method_b_heads import (
    INTERACTION_LIST,
    MATERIAL_LIST,
    MethodB,
    build_heads,
)
from impossible_mix.methods.base import HybridSpec


TARGETS = [
    {"name": "rolling_drop",       "anchor_filter": ("metal", "roll"),
     "tgt_material": "liquid", "tgt_interaction": "roll",   # roll se mapea a roll
     "mods": {"wetness": 5, "granularity": 4, "continuity": 5}},
    {"name": "liquid_rock_impact", "anchor_filter": ("rock", "impact"),
     "tgt_material": "liquid", "tgt_interaction": "impact",
     "mods": {"wetness": 5, "rigidity": 4}},
    {"name": "wet_gravel_scrape",  "anchor_filter": ("rock", "scrape"),
     "tgt_material": "gravel", "tgt_interaction": "scrape",
     "mods": {"wetness": 5, "granularity": 5}},
]

# Pesos para edicion por gradiente (sweep limitado)
N_ITERS_SET = [50, 150]
LAMBDA_PRIOR_SET = [0.05, 0.2]


def pick_anchor(bundle, material, interaction, idx_offset=0):
    Z = bundle.Z_mean
    mask = torch.tensor([
        m == material and i == interaction
        for m, i in zip(bundle.materials, bundle.interactions)
    ])
    cent = centroid(Z, mask)
    dists = (Z - cent.unsqueeze(0)).norm(dim=-1).masked_fill(~mask, float("inf"))
    return int(torch.argsort(dists)[idx_offset].item())


def save_wav(wav, path, sr_in):
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_44 = torchaudio.functional.resample(wav.unsqueeze(0), sr_in, SAMPLE_RATE).squeeze(0).numpy()
    sf.write(str(path), wav_44, SAMPLE_RATE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=f"b_{int(time.time())}")
    ap.add_argument("--heads-ckpt", type=Path, default=RESULTS_DIR / "method_b" / "heads_fused.pt")
    args = ap.parse_args()

    print("Cargando corpus, encoder y cabezas...")
    bundle = load_corpus()
    encoder = get_encoder("encodec")
    # Aplicar fusion step->impact al bundle para consistencia con cabezas
    bundle.interactions = ["impact" if x == "step" else x for x in bundle.interactions]

    ck = torch.load(args.heads_ckpt, map_location="cpu", weights_only=False)
    heads = build_heads(ck["latent_dim"], hidden=ck["hidden"])
    heads.load_state_dict(ck["state_dict"])
    for h in (heads.head_material, heads.head_interaction, heads.head_modifiers):
        h.eval()
    method = MethodB(encoder, heads)
    print(f"  N={bundle.n}, latent_dim={ck['latent_dim']}, fused step->impact: {ck.get('step_fused_with_impact', False)}")

    out_dir = OUTPUTS_DIR / "method_b" / args.run_name
    manifest_path = RESULTS_DIR / "method_b" / f"{args.run_name}_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for target in TARGETS:
        src_mat, src_int_orig = target["anchor_filter"]
        # Cuando target_interaction era step, aqui ya es impact
        tgt_int = "impact" if target["tgt_interaction"] == "step" else target["tgt_interaction"]
        # idem para anchor_filter
        src_int = "impact" if src_int_orig == "step" else src_int_orig
        anchors = [pick_anchor(bundle, src_mat, src_int, k) for k in range(2)]
        for ai, anchor_idx in enumerate(anchors):
            z_anchor = bundle.Z_seq[anchor_idx]  # (dim, T') preserva dinamica
            anchor_id = bundle.clip_ids[anchor_idx]
            print(f"\n>>> {target['name']}  ancla={anchor_id}  ({src_mat} x {src_int})")
            for n_iters in N_ITERS_SET:
                for lambda_prior in LAMBDA_PRIOR_SET:
                    spec = HybridSpec(
                        target_material=target["tgt_material"],
                        target_interaction=tgt_int,
                        properties=target["mods"],
                        weights={"n_iters": n_iters, "lr": 5e-3, "lambda_prior": lambda_prior},
                    )
                    t0 = time.time()
                    wav = method.generate(z_anchor, spec)
                    dt = time.time() - t0
                    tag = f"iters{n_iters}_prior{lambda_prior}_anchor{ai}"
                    p = out_dir / f"{target['name']}_{tag}.wav"
                    save_wav(wav, p, sr_in=encoder.sr_expected)
                    print(f"    iters={n_iters} prior={lambda_prior} took {dt:.1f}s")
                    rows.append(dict(method="B", combo=target["name"], anchor=anchor_id,
                                     n_iters=n_iters, lambda_prior=lambda_prior, path=str(p)))

    with manifest_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nGenerados {len(rows)} hibridos B. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
