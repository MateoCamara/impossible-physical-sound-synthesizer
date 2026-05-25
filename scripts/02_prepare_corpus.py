"""Convierte todos los clips referenciados en data/labels.csv (col `notes`
contiene `path=...` con la ruta original) a wav 44.1 kHz mono 5s con peak
normalizado a -1 dBFS y los escribe en data/processed/<clip_id>.wav.

Acepta tanto ULFC (chunks/) como Freesound (mp3 previews). Filtra clips
demasiado silenciosos (RMS < -50 dBFS) y los marca con `notes_extra=skipped`.

Uso:
    python scripts/02_prepare_corpus.py [--limit N]

Tras esto, data/processed/ deberia tener una entrada por cada fila de
labels.csv. Si labels.csv tiene 720 + 200 freesound = 920 entradas,
esperamos ~920 wavs procesados (excluyendo silencios).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import (
    CLIP_SAMPLES,
    LABELS_CSV,
    PROCESSED_DIR,
    REPO_ROOT,
    SAMPLE_RATE,
)
from impossible_mix.data.curation import (
    fit_duration,
    is_too_quiet,
    load_mono,
    peak_normalize,
    save_wav,
)

# `path=` es siempre el ultimo campo en `notes`; capturamos hasta el final
# para permitir espacios en el nombre (ULFC tiene "WALK M SLOW", etc.).
PATH_RE = re.compile(r"path=(.+)$")


def resolve_source_path(notes: str, source: str) -> Path | None:
    """Extrae la ruta original desde la columna `notes` y la resuelve."""
    if not notes:
        return None
    m = PATH_RE.search(notes)
    if not m:
        return None
    rel = m.group(1).strip()
    if source == "ulfc":
        return REPO_ROOT / "ULTIMATE-FOOTSTEP-COLLECTION" / rel
    if source == "freesound":
        # En labels_freesound.csv el path viene relativo a `data/`
        # (p.ej. "raw/freesound/liquid/drip/12345.mp3").
        if rel.startswith("data/"):
            return REPO_ROOT / rel
        return REPO_ROOT / "data" / rel
    return REPO_ROOT / rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Limitar para pruebas (0 = todo).")
    ap.add_argument("--labels", type=Path, default=LABELS_CSV)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    if args.limit:
        df = df.head(args.limit)
    print(f"Procesando {len(df)} clips -> {args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "skipped_quiet": 0, "missing_path": 0, "missing_file": 0, "error": 0}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src = resolve_source_path(str(row.get("notes", "")), str(row.get("source", "")))
        if src is None:
            stats["missing_path"] += 1
            continue
        if not src.exists():
            stats["missing_file"] += 1
            continue
        try:
            y = load_mono(src, SAMPLE_RATE)
            if is_too_quiet(y):
                stats["skipped_quiet"] += 1
                continue
            y = fit_duration(y, CLIP_SAMPLES)
            y = peak_normalize(y)
            out = args.out_dir / f"{row['clip_id']}.wav"
            save_wav(out, y, SAMPLE_RATE)
            stats["ok"] += 1
        except Exception as e:
            stats["error"] += 1
            print(f"  ! {row['clip_id']}: {e}")

    print("\nResumen:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0 if stats["ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
