"""Fusiona data/labels.csv (ULFC) + data/raw/freesound/labels_freesound.csv
en un unico data/labels.csv con todos los clips disponibles.

Idempotente: deduplica por clip_id; si una fila ya existe la preserva.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import LABELS_CSV


def main() -> int:
    fs_labels = Path("data/raw/freesound/labels_freesound.csv")
    if not fs_labels.exists():
        print(f"!! No existe {fs_labels}. Ejecuta antes scripts/01_freesound_download.py")
        return 1

    main_df = pd.read_csv(LABELS_CSV)
    fs_df = pd.read_csv(fs_labels)
    print(f"ULFC labels: {len(main_df)} | Freesound labels: {len(fs_df)}")

    merged = pd.concat([main_df, fs_df], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset="clip_id", keep="first")
    after = len(merged)
    print(f"Tras dedup: {after} (eliminados {before - after} duplicados)")

    # Backup del labels.csv anterior antes de sobreescribir
    backup = LABELS_CSV.with_suffix(".csv.bak")
    if not backup.exists():
        main_df.to_csv(backup, index=False)
        print(f"Backup ULFC-only en {backup}")
    merged.to_csv(LABELS_CSV, index=False)

    print("\nDistribucion por source:")
    print(merged["source"].value_counts())
    print("\nDistribucion por (material, interaction):")
    counts = merged.groupby(["material", "interaction"]).size().reset_index(name="n")
    print(counts.sort_values("n", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
