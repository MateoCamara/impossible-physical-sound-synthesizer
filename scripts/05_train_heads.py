"""Entrena las cabezas del Metodo B sobre embeddings cacheados.

Split estratificado por (material x interaction), 80/20. Sin data augmentation
ni shuffling sofisticado: las cabezas son MLPs pequenas, basta con Adam y
algunas decenas de epochs.

Salida:
    results/method_b/heads_<timestamp>.pt
    results/method_b/training_metrics.csv

Sanity para abstract: F1 macro material >= 0.7, R^2 modifiers >= 0.5.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import RESULTS_DIR, SEED
from impossible_mix.data.dataset import EmbeddingClassificationDataset, load_corpus
from impossible_mix.methods.method_b_heads import (
    INTERACTION_LIST,
    MATERIAL_LIST,
    build_heads,
    train_step,
)


def stratified_indices(materials, interactions, test_size=0.2, seed=SEED):
    strata = [f"{m}|{i}" for m, i in zip(materials, interactions)]
    idx = np.arange(len(strata))
    # Algunas combinaciones tienen <5 muestras (gravel x roll, rock x roll);
    # sklearn requiere >= 2. Filtramos buckets sin gemelos.
    counts = {}
    for s in strata:
        counts[s] = counts.get(s, 0) + 1
    valid_mask = np.array([counts[s] >= 2 for s in strata])
    idx_valid = idx[valid_mask]
    strata_valid = [strata[i] for i in idx_valid]
    train, test = train_test_split(idx_valid, test_size=test_size, stratify=strata_valid, random_state=seed)
    return list(train), list(test)


def evaluate(heads, ds, indices):
    heads.head_material.eval(); heads.head_interaction.eval(); heads.head_modifiers.eval()
    y_m_pred, y_m_true = [], []
    y_i_pred, y_i_true = [], []
    mods_pred, mods_true = [], []
    with torch.no_grad():
        for i in indices:
            sample = ds[i]
            z = sample["z"].unsqueeze(0)
            y_m_pred.append(heads.head_material(z).argmax(-1).item())
            y_m_true.append(sample["material_y"].item())
            y_i_pred.append(heads.head_interaction(z).argmax(-1).item())
            y_i_true.append(sample["interaction_y"].item())
            mods_pred.append(heads.head_modifiers(z).squeeze(0).numpy())
            mods_true.append(sample["modifiers_y"].numpy())
    return dict(
        f1_material=f1_score(y_m_true, y_m_pred, average="macro", zero_division=0),
        f1_interaction=f1_score(y_i_true, y_i_pred, average="macro", zero_division=0),
        r2_modifiers=r2_score(np.array(mods_true), np.array(mods_pred)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True,
                    help="Pesos inversamente proporcionales al support (default ON).")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "method_b")
    args = ap.parse_args()

    print("Cargando corpus + embeddings...")
    bundle = load_corpus()
    print(f"  N={bundle.n} clips, dim={bundle.Z_mean.shape[1]}")
    ds = EmbeddingClassificationDataset(bundle)

    train_idx, test_idx = stratified_indices(bundle.materials, bundle.interactions)
    print(f"  train={len(train_idx)}, test={len(test_idx)}")
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True)

    torch.manual_seed(SEED)
    heads = build_heads(bundle.Z_mean.shape[1], hidden=args.hidden)
    optim = torch.optim.Adam(list(heads.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # Pesos por clase inversos al support de train (sin tocar test)
    w_mat = w_int = None
    if args.class_weights:
        train_y_mat = [bundle.materials[i] for i in train_idx]
        train_y_int = [bundle.interactions[i] for i in train_idx]
        cm = Counter(train_y_mat); ci = Counter(train_y_int)
        # peso = N_total / (N_clases * support_c); clases inexistentes -> 1.0
        N_t = len(train_y_mat)
        w_mat = torch.tensor([
            N_t / (len(MATERIAL_LIST) * max(cm.get(m, 0), 1)) for m in MATERIAL_LIST
        ], dtype=torch.float32)
        w_int = torch.tensor([
            N_t / (len(INTERACTION_LIST) * max(ci.get(x, 0), 1)) for x in INTERACTION_LIST
        ], dtype=torch.float32)
        print(f"  class weights material: {dict(zip(MATERIAL_LIST, w_mat.tolist()))}")
        print(f"  class weights interaction: {dict(zip(INTERACTION_LIST, w_int.tolist()))}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "training_metrics.csv"
    with metrics_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "loss_total", "loss_material", "loss_interaction", "loss_modifiers",
                    "f1_material", "f1_interaction", "r2_modifiers"])

        t0 = time.time()
        for ep in range(args.epochs):
            heads.head_material.train(); heads.head_interaction.train(); heads.head_modifiers.train()
            totals = {"loss_total": 0.0, "loss_material": 0.0, "loss_interaction": 0.0, "loss_modifiers": 0.0}
            n_batches = 0
            for batch in train_loader:
                losses = train_step(heads, batch["z"], batch["material_y"],
                                    batch["interaction_y"], batch["modifiers_y"],
                                    weight_material=w_mat, weight_interaction=w_int)
                optim.zero_grad()
                losses["loss_total"].backward()
                optim.step()
                for k in totals: totals[k] += losses[k].item()
                n_batches += 1
            for k in totals: totals[k] /= max(n_batches, 1)
            ev = evaluate(heads, ds, test_idx)
            w.writerow([ep, totals["loss_total"], totals["loss_material"], totals["loss_interaction"],
                        totals["loss_modifiers"], ev["f1_material"], ev["f1_interaction"], ev["r2_modifiers"]])
            if ep % 10 == 0 or ep == args.epochs - 1:
                print(f"  ep {ep:3d}  L={totals['loss_total']:.3f}  "
                      f"F1_mat={ev['f1_material']:.3f}  F1_int={ev['f1_interaction']:.3f}  "
                      f"R2_mods={ev['r2_modifiers']:.3f}")

    ckpt_path = args.out_dir / "heads.pt"
    torch.save({
        "state_dict": heads.state_dict(),
        "latent_dim": bundle.Z_mean.shape[1],
        "hidden": args.hidden,
        "materials": MATERIAL_LIST,
        "interactions": INTERACTION_LIST,
        "test_idx": test_idx,
    }, ckpt_path)
    print(f"\nGuardadas cabezas en {ckpt_path}")
    print(f"Tiempo total: {time.time()-t0:.1f}s")

    final = evaluate(heads, ds, test_idx)
    print(f"\nMetricas finales test:")
    for k, v in final.items():
        gate = " (OK, >= umbral)" if (
            (k == "f1_material" and v >= 0.7) or
            (k == "f1_interaction" and v >= 0.7) or
            (k == "r2_modifiers" and v >= 0.5)
        ) else ""
        print(f"  {k}: {v:.3f}{gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
