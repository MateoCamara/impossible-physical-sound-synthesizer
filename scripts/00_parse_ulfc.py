"""Parsea el corpus ULTIMATE-FOOTSTEP-COLLECTION/chunks y genera un primer
etiquetado automatico en data/labels.csv basado en el nombre de fichero.

Patron observado:
    <SHOE>_<MATERIAL>_<TAKE>_<ACTION...>_<MIC>-chunk<N>.wav
    SHOE  in {BOOT, DRESS, FLAT, HEEL, SNEAK}
    MIC   in {416, KMR81}
    ACTION puede llevar tokens espaciados: "WALK XX SLOW", "STAIR FAST"

Mapea codigos ULFC -> taxonomia cerrada (impossible_mix.data.labels)
con valores por defecto de los 5 modificadores que se afinan dia 2/4.

Uso:
    python scripts/00_parse_ulfc.py \\
        --chunks-dir ULTIMATE-FOOTSTEP-COLLECTION/chunks \\
        --out data/labels.csv \\
        --sample-per-bucket 60          # opcional: subset balanceado
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# Permite ejecutar el script sin pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.data.labels import (
    LABELS_COLUMNS,
    ULFC_ACTION_MAP,
    ULFC_MATERIAL_MAP,
    Interaction,
    Material,
)

# stem ejemplo: "BOOT_ASPH_01_WALK XX SLOW_416-chunk0"
FILENAME_RE = re.compile(
    r"^(?P<shoe>[A-Z]+)_(?P<material>[A-Z0-9]+)_(?P<take>[A-Z0-9\-]+)_(?P<rest>.+)-chunk(?P<chunk>\d+)$"
)
MIC_TOKENS = {"416", "KMR81"}


def parse_action_tokens(rest: str) -> tuple[str, str]:
    """rest = 'WALK XX SLOW_416' -> (action_code, mic). Robust to spaces."""
    if "_" in rest:
        action_part, mic = rest.rsplit("_", 1)
    else:
        action_part, mic = rest, ""
    # Primer token alfanumerico sirve como action_code base.
    action_token = action_part.strip().split()[0]
    return action_token, mic


def default_modifiers(material: Material, interaction: Interaction) -> dict[str, int]:
    """Valores Likert 1-5 razonables como inicializacion del weak label."""
    wetness = 1  # corpus completo es seco
    rigidity = {
        Material.ROCK: 5, Material.METAL: 5, Material.EARTH: 2,
        Material.WOOD: 4, Material.FABRIC: 1, Material.GRAVEL: 3,
        Material.LIQUID: 1, Material.OTHER: 3,
    }.get(material, 3)
    resonance = {
        Material.METAL: 4, Material.ROCK: 3, Material.WOOD: 3,
        Material.EARTH: 1, Material.FABRIC: 1, Material.GRAVEL: 2,
        Material.LIQUID: 2, Material.OTHER: 2,
    }.get(material, 2)
    granularity = {
        Material.GRAVEL: 5, Material.EARTH: 3, Material.FABRIC: 2,
        Material.WOOD: 2, Material.ROCK: 2, Material.METAL: 2,
        Material.LIQUID: 4, Material.OTHER: 3,
    }.get(material, 3)
    continuity = {
        Interaction.STEP: 3, Interaction.SCRAPE: 4, Interaction.DRAG: 5,
        Interaction.ROLL: 5, Interaction.POUR: 5, Interaction.SPLASH: 2,
        Interaction.DRIP: 1, Interaction.IMPACT: 1,
    }.get(interaction, 3)
    return dict(wetness=wetness, rigidity=rigidity, resonance=resonance,
                granularity=granularity, continuity=continuity)


def clip_id_from(path: Path) -> str:
    return hashlib.sha1(path.name.encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--sample-per-bucket", type=int, default=0,
                    help="Si >0, muestreo balanceado por (material x interaction).")
    ap.add_argument("--seed", type=int, default=20260520)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    for wav in sorted(args.chunks_dir.glob("*.wav")):
        m = FILENAME_RE.match(wav.stem)
        if not m:
            skipped.append(wav.name)
            continue
        material_code = m.group("material")
        action_token, _ = parse_action_tokens(m.group("rest"))
        material = ULFC_MATERIAL_MAP.get(material_code)
        interaction = ULFC_ACTION_MAP.get(action_token)
        if material is None or interaction is None:
            skipped.append(wav.name)
            continue
        mods = default_modifiers(material, interaction)
        row = {
            "clip_id": clip_id_from(wav),
            "source": "ulfc",
            "parent_id": "",
            "material": material.value,
            "interaction": interaction.value,
            **mods,
            "notes": f"path={wav.relative_to(args.chunks_dir.parent)}",
        }
        rows.append(row)
        buckets[(material.value, interaction.value)].append(row)

    if args.sample_per_bucket > 0:
        sampled: list[dict[str, object]] = []
        for key, items in buckets.items():
            rng.shuffle(items)
            sampled.extend(items[: args.sample_per_bucket])
        rows = sampled

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LABELS_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Skipped {len(skipped)} files (no match for regex or unmapped codes).")
    if skipped[:5]:
        print("First skipped:", skipped[:5])
    print("\nDistribution by (material, interaction):")
    counts = defaultdict(int)
    for r in rows:
        counts[(r["material"], r["interaction"])] += 1
    for (mat, intx), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {mat:8s} x {intx:8s} : {n:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
