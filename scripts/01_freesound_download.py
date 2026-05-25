"""Descarga previews HQ MP3 de Freesound para rellenar los gaps documentados
en data/FREESOUND_GAPS.md.

Usa la API v2 con autenticacion por token (gratuita). Solo descarga previews
HQ (~128 kbps mp3). El paso 02_prepare_corpus.py los convierte a wav 44.1 kHz.

Para descargar wavs originales hace falta OAuth2 (no implementado aqui);
los previews HQ son suficientes para investigacion de mezcla latente.

Salida:
    data/raw/freesound/<material>/<interaction>/<sound_id>.mp3
    data/raw/freesound/manifest.json           (metadata: id, name, license, query)
    data/raw/freesound/labels_freesound.csv    (lineas para appendear a labels.csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.data.labels import LABELS_COLUMNS, Interaction, Material

load_dotenv()

import os  # noqa: E402

API = "https://freesound.org/apiv2"
KEY = os.getenv("FREESOUND_API_KEY", "")


# Cada bucket (material, interaction) lleva varias queries cortas. El script
# las ejecuta una a una y deduplica por sound_id antes de cortar a `max`.
# Freesound Solr trata mal el OR entre frases largas; mejor varias queries.
QUERIES: list[dict[str, Any]] = [
    # --- LIQUID ---
    {"material": Material.LIQUID, "interaction": Interaction.DRIP,
     "qs": ["water drop", "dripping", "drip", "faucet drip"], "max": 20},
    {"material": Material.LIQUID, "interaction": Interaction.SPLASH,
     "qs": ["water splash", "splash", "liquid splash"], "max": 20},
    {"material": Material.LIQUID, "interaction": Interaction.POUR,
     "qs": ["pouring water", "water pour", "pour"], "max": 15},
    {"material": Material.LIQUID, "interaction": Interaction.IMPACT,
     "qs": ["water hit", "water impact", "liquid impact"], "max": 15},
    # --- GRAVEL ---
    {"material": Material.GRAVEL, "interaction": Interaction.STEP,
     "qs": ["gravel walk", "gravel footstep", "pebbles step"], "max": 20},
    {"material": Material.GRAVEL, "interaction": Interaction.ROLL,
     "qs": ["pebbles falling", "gravel rolling", "stones rolling", "rocks tumbling"], "max": 15},
    {"material": Material.GRAVEL, "interaction": Interaction.POUR,
     "qs": ["pouring sand", "pouring gravel", "pebbles pouring"], "max": 10},
    # --- WOOD ---
    {"material": Material.WOOD, "interaction": Interaction.IMPACT,
     "qs": ["wood impact", "wood knock", "wooden hit", "wood hit"], "max": 20},
    {"material": Material.WOOD, "interaction": Interaction.STEP,
     "qs": ["wooden footstep", "wood walking", "wood floor footstep"], "max": 15},
    {"material": Material.WOOD, "interaction": Interaction.DRAG,
     "qs": ["wood drag", "dragging wood", "wood slide"], "max": 10},
    # --- FABRIC ---
    {"material": Material.FABRIC, "interaction": Interaction.DRAG,
     "qs": ["cloth drag", "fabric movement", "silk slide", "cloth slide"], "max": 15},
    {"material": Material.FABRIC, "interaction": Interaction.SCRAPE,
     "qs": ["cloth rustle", "fabric rub", "fabric rustle"], "max": 15},
    # --- ROLL puro (anclajes de estructura temporal) ---
    {"material": Material.METAL, "interaction": Interaction.ROLL,
     "qs": ["ball rolling", "marble rolling", "metal ball roll"], "max": 15},
    {"material": Material.ROCK, "interaction": Interaction.ROLL,
     "qs": ["stone rolling", "rock rolling", "boulder rolling"], "max": 10},
]


# Modificadores por defecto (Likert 1-5) por (material, interaction).
# Mismas heuristicas que en scripts/00_parse_ulfc.py para coherencia.
def default_modifiers(material: Material, interaction: Interaction) -> dict[str, int]:
    wet = 5 if material is Material.LIQUID else 1
    rigidity = {
        Material.ROCK: 5, Material.METAL: 5, Material.WOOD: 4,
        Material.GRAVEL: 3, Material.EARTH: 2, Material.FABRIC: 1,
        Material.LIQUID: 1, Material.OTHER: 3,
    }[material]
    resonance = {
        Material.METAL: 4, Material.ROCK: 3, Material.WOOD: 3,
        Material.LIQUID: 2, Material.GRAVEL: 2, Material.EARTH: 1,
        Material.FABRIC: 1, Material.OTHER: 2,
    }[material]
    granularity = {
        Material.GRAVEL: 5, Material.LIQUID: 4, Material.EARTH: 3,
        Material.FABRIC: 2, Material.WOOD: 2, Material.ROCK: 2,
        Material.METAL: 2, Material.OTHER: 3,
    }[material]
    continuity = {
        Interaction.DRIP: 1, Interaction.IMPACT: 1, Interaction.STEP: 3,
        Interaction.SCRAPE: 4, Interaction.SPLASH: 2, Interaction.ROLL: 5,
        Interaction.POUR: 5, Interaction.DRAG: 5,
    }[interaction]
    return dict(wetness=wet, rigidity=rigidity, resonance=resonance,
                granularity=granularity, continuity=continuity)


FIELDS = "id,name,license,duration,samplerate,channels,previews,tags,username"
# duration 1.5..10 (tras peak-norm cortaremos a 5s); license CC0/CC-BY/CC-Sampling+;
# evitamos noncommercial-only para tener libertad en test perceptual publico.
LICENSE_FILTER = (
    'license:"Creative Commons 0" OR '
    'license:"Attribution" OR '
    'license:"Attribution 4.0"'
)


def search_one(q: str, max_results: int, page_size: int = 30) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while len(items) < max_results:
        params = {
            "query": q,
            "filter": f"duration:[0.5 TO 10] samplerate:[44100 TO 192000] ({LICENSE_FILTER})",
            "fields": FIELDS,
            "page_size": min(page_size, max_results - len(items)),
            "page": page,
            "token": KEY,
        }
        r = requests.get(f"{API}/search/text/", params=params, timeout=30)
        if r.status_code != 200:
            print(f"    ! HTTP {r.status_code}: {r.text[:200]}")
            return items
        body = r.json()
        batch = body.get("results", [])
        if not batch:
            break
        items.extend(batch)
        if not body.get("next"):
            break
        page += 1
        time.sleep(0.3)
    return items[:max_results]


def search_bucket(queries: list[str], max_total: int) -> list[dict[str, Any]]:
    """Une resultados de varias queries y deduplica por sound_id."""
    seen: dict[int, dict[str, Any]] = {}
    per_query = max(5, max_total // len(queries) + 5)  # margen para dedup
    for q in queries:
        for item in search_one(q, per_query):
            seen.setdefault(item["id"], item)
    return list(seen.values())[:max_total]


def download_preview(item: dict[str, Any], out_path: Path) -> bool:
    url = item.get("previews", {}).get("preview-hq-mp3")
    if not url:
        return False
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True
    r = requests.get(url, timeout=60, stream=True)
    if r.status_code != 200:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        for chunk in r.iter_content(8192):
            fh.write(chunk)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw/freesound"))
    ap.add_argument("--max-multiplier", type=float, default=1.0,
                    help="Multiplica los maximos por query (1.0 = como QUERIES).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Solo lista resultados, no descarga.")
    args = ap.parse_args()

    if not KEY or len(KEY) < 20:
        print("ERROR: FREESOUND_API_KEY no esta en .env (o es invalida).")
        return 1

    manifest: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for q in QUERIES:
        material: Material = q["material"]
        interaction: Interaction = q["interaction"]
        max_n = int(q["max"] * args.max_multiplier)
        print(f"\n=== {material.value} x {interaction.value} :: {q['qs']} (max {max_n})")
        results = search_bucket(q["qs"], max_n)
        print(f"  -> {len(results)} candidates")
        if args.dry_run:
            for r in results[:3]:
                print(f"    {r['id']:>10}  {r['license'][:40]:40s}  {r['duration']:.2f}s  {r['name'][:60]}")
            continue
        bucket = args.out_dir / material.value / interaction.value
        for item in tqdm(results, desc=f"{material.value}/{interaction.value}", leave=False):
            sound_id = item["id"]
            out_path = bucket / f"{sound_id}.mp3"
            ok = download_preview(item, out_path)
            if not ok:
                continue
            mods = default_modifiers(material, interaction)
            clip_id = f"fs{sound_id}"
            manifest.append({
                "clip_id": clip_id,
                "freesound_id": sound_id,
                "name": item["name"],
                "license": item["license"],
                "username": item["username"],
                "duration": item["duration"],
                "samplerate": item["samplerate"],
                "channels": item["channels"],
                "tags": item.get("tags", []),
                "queries": q["qs"],
                "material": material.value,
                "interaction": interaction.value,
                "local_path": str(out_path.relative_to(args.out_dir.parent.parent)),
            })
            csv_rows.append({
                "clip_id": clip_id,
                "source": "freesound",
                "parent_id": "",
                "material": material.value,
                "interaction": interaction.value,
                **mods,
                "notes": f"fs={sound_id} license={item['license'][:30]} path={out_path.relative_to(args.out_dir.parent.parent)}",
            })
            time.sleep(0.2)

    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (args.out_dir / "labels_freesound.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LABELS_COLUMNS))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDescargados {len(manifest)} previews a {args.out_dir}")
    print(f"Manifest: {args.out_dir/'manifest.json'}")
    print(f"Labels nuevas: {args.out_dir/'labels_freesound.csv'} ({len(csv_rows)} filas)")
    print("\nPara fusionar con labels.csv principal:")
    print(f"  tail -n +2 {args.out_dir/'labels_freesound.csv'} >> data/labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
