"""Aplica la capa DSP del Metodo D sobre los hibridos A mas prometedores
(top-N por intermediateness) y produce demos comparativos: A puro vs A+D.

El paper de septiembre promociona D a capa diferenciable end-to-end; aqui
en el sprint del abstract solo necesitamos demos audibles.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import OUTPUTS_DIR, RESULTS_DIR, SAMPLE_RATE
from impossible_mix.methods.method_d_dsp_layer import DSPParams, apply_dsp


# Configuraciones de DSP por combinacion objetivo
DSP_PRESETS = {
    "rolling_drop":       DSPParams(wetness=0.9, granularity=0.7, continuity=0.8, resonance=0.2, sr=SAMPLE_RATE),
    "liquid_rock_impact": DSPParams(wetness=0.85, rigidity=0.4, resonance=0.3, sr=SAMPLE_RATE),
    "wet_gravel_scrape":  DSPParams(wetness=0.75, granularity=0.9, rigidity=0.3, sr=SAMPLE_RATE),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path,
                    default=RESULTS_DIR / "method_a" / "full_manifest_metrics.csv")
    ap.add_argument("--top-per-combo", type=int, default=5)
    ap.add_argument("--run-name", default="d_full")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    df = df[df.method == "A"].copy()
    df["score"] = (df["inter_mat"] - 0.5).abs() + df["d_tgt_mat"] / 50.0  # penaliza tambien d_tgt
    out_dir = OUTPUTS_DIR / "method_d" / args.run_name
    manifest_path = RESULTS_DIR / "method_d" / f"{args.run_name}_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for combo, preset in DSP_PRESETS.items():
        sub = df[df.combo == combo].sort_values("score").head(args.top_per_combo)
        print(f"\n>>> {combo}  top-{args.top_per_combo} from A  preset={preset}")
        for _, r in sub.iterrows():
            in_path = Path(r["path"])
            y, sr_in = sf.read(str(in_path), dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            if sr_in != SAMPLE_RATE:
                # ya esta a 44.1 desde el script 06 (resampleado al guardar)
                pass
            y_post = apply_dsp(y, preset)
            anchor_suffix = Path(r["path"]).stem.split("_anchor")[-1] if "_anchor" in r["path"] else "0"
            tag = f"a{r['alpha']}_b{r['beta']}_g{r['gamma']}_anc{anchor_suffix}"
            out_path = out_dir / f"{combo}_{tag}_DSP.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), y_post, SAMPLE_RATE)
            rows.append(dict(method="D", combo=combo, alpha=r["alpha"], beta=r["beta"],
                             gamma=r["gamma"], source_a=str(in_path), out=str(out_path),
                             wet=preset.wetness, gran=preset.granularity,
                             rig=preset.rigidity, res=preset.resonance, cont=preset.continuity))
            print(f"  -> {out_path.name}")

    with manifest_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nGenerados {len(rows)} hibridos D (A+DSP). Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
