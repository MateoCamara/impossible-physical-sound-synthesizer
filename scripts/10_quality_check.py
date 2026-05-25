"""Auto-evaluacion de calidad de TODOS los wavs en outputs/.
Filtra 'basura' antes de escucha humana y produce un CSV ordenado.

Uso:
    python scripts/10_quality_check.py [--root outputs] [--out results/quality_report.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.metrics.quality import quality_verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, default=Path("results/quality_report.csv"))
    args = ap.parse_args()

    wavs = sorted(args.root.rglob("*.wav"))
    print(f"Analizando {len(wavs)} wavs bajo {args.root}")

    rows = []
    for wav in tqdm(wavs):
        try:
            y, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            r = quality_verdict(y, sr)
            # Parse method/combo/run del path
            parts = wav.relative_to(args.root).parts
            method = parts[0] if len(parts) > 0 else ""
            run = parts[1] if len(parts) > 1 else ""
            rows.append(dict(
                method=method, run=run, name=wav.name,
                verdict=r.verdict, reasons="; ".join(r.reasons),
                rms_db=round(r.rms_db, 2), peak_db=round(r.peak_db, 2),
                crest_db=round(r.crest_db, 2),
                flatness=round(r.flatness, 3), centroid_hz=round(r.centroid_hz, 0),
                dynamic_range_db=round(r.dynamic_range_db, 2),
                path=str(wav),
            ))
        except Exception as e:
            print(f"  ! {wav.name}: {e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # Resumen por verdict
    from collections import Counter
    verdicts = Counter(r["verdict"] for r in rows)
    print("\nDistribucion de verdicts:")
    for v, n in verdicts.most_common():
        print(f"  {v:12s}: {n}")

    # Por method
    print("\nVerdicts por method:")
    by_method = {}
    for r in rows:
        m = r["method"]
        by_method.setdefault(m, Counter())[r["verdict"]] += 1
    for m, c in by_method.items():
        total = sum(c.values())
        pct_good = 100 * c.get("good", 0) / total
        print(f"  {m:18s} total={total:3d}  good={c.get('good',0):3d} ({pct_good:.0f}%)  susp={c.get('suspicious',0):3d}  garbage={c.get('garbage',0):3d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
