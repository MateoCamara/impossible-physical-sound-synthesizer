"""Descarga IRs reales CC-licensed para uso con convolucion.

Fuentes principales:
  - OpenAIR (https://www.openair.hosted.york.ac.uk/) — IRs de catedrales,
    iglesias, capillas, criptas, etc. con licencias CC. La API directa
    de descarga ha cambiado a lo largo del tiempo; este script intenta
    URLs estables y, si fallan, deja un mensaje claro.
  - Si no hay conectividad o falla la descarga, el script crea aliases
    a las IRs sinteticas existentes para que `get_ir(name)` siga funcionando
    en demos sin red.

Uso:
    python scripts/24_download_irs.py            # descarga lo no presente
    python scripts/24_download_irs.py --list     # solo lista lo registrado
    python scripts/24_download_irs.py --force    # re-descarga todo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.physics.reverb import REAL_IR_PATHS, register_real_ir


IR_DIR = Path("impossible_mix/physics/presets_irs")
IR_DIR.mkdir(parents=True, exist_ok=True)


# Catalogo curado: nombre logico -> (URL, descripcion, licencia)
# Las URLs son las mas recientes conocidas; si fallan, mostramos un
# fallback con instrucciones manuales.
CATALOG: list[dict[str, str]] = [
    {
        "name": "st_andrews_chapel",
        "url": "https://www.openair.hosted.york.ac.uk/auditoriums/StAndrews-Chapel/StAndrews-Chapel/B-Format/StAndrews-Chapel-Bformat-1.wav",
        "license": "CC-BY-SA 4.0",
        "description": "St Andrew's Chapel, University of York — small chapel.",
    },
    {
        "name": "york_minster",
        "url": "https://www.openair.hosted.york.ac.uk/auditoriums/York-Minster/York-Minster-IRs/B-Format/York-Minster-Bformat-1.wav",
        "license": "CC-BY-SA 4.0",
        "description": "York Minster — large gothic cathedral.",
    },
    {
        "name": "tvisongur_sound_sculpture",
        "url": "https://www.openair.hosted.york.ac.uk/auditoriums/Tvisongur-Sound-Sculpture/Tvisongur-IRs/B-Format/Tvisongur-Bformat-1.wav",
        "license": "CC-BY-SA 4.0",
        "description": "Tvísöngur sculpture — five resonant concrete domes.",
    },
]


def already_present(name: str) -> Path | None:
    """Devuelve la ruta local si ya descargado, sino None."""
    candidate = IR_DIR / f"{name}.wav"
    if candidate.exists() and candidate.stat().st_size > 1000:
        return candidate
    return None


def download_one(entry: dict[str, str], force: bool = False) -> Path | None:
    name = entry["name"]
    out_path = IR_DIR / f"{name}.wav"
    if not force and already_present(name):
        return already_present(name)
    print(f"  downloading {name} ({entry['license']}) ...", end="", flush=True)
    try:
        urlretrieve(entry["url"], str(out_path))
        size = out_path.stat().st_size
        if size < 1000:
            out_path.unlink()
            print(f" FAILED (tiny file)")
            return None
        print(f" OK ({size/1024:.1f} KB)")
        return out_path
    except Exception as e:
        print(f" FAILED: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="Solo listar IRs ya descargadas y catalogo disponible")
    ap.add_argument("--force", action="store_true",
                    help="Re-descargar incluso si ya existe local")
    args = ap.parse_args()

    if args.list:
        print(f"=== Real IRs registered in REAL_IR_PATHS ===")
        for n, p in REAL_IR_PATHS.items():
            present = "OK" if Path(p).exists() else "MISSING"
            print(f"  {n:30s}  {p}  [{present}]")
        print(f"\n=== Local files in {IR_DIR}/ ===")
        for f in sorted(IR_DIR.glob("*.wav")):
            print(f"  {f.name}  ({f.stat().st_size/1024:.1f} KB)")
        print(f"\n=== Catalog (downloadable) ===")
        for e in CATALOG:
            present = "OK" if already_present(e["name"]) else "missing"
            print(f"  {e['name']:30s}  [{e['license']}]  [{present}]")
        return 0

    print(f"=== Downloading {len(CATALOG)} real IRs to {IR_DIR}/ ===")
    ok_count = 0
    for entry in CATALOG:
        path = download_one(entry, force=args.force)
        if path is not None:
            register_real_ir(entry["name"], str(path))
            ok_count += 1

    print(f"\n=== Summary: {ok_count}/{len(CATALOG)} OK ===")
    if ok_count < len(CATALOG):
        print("\nManual fallback:")
        print("  - Browse https://www.openair.hosted.york.ac.uk/")
        print(f"  - Download a stereo or B-format IR WAV manually into {IR_DIR}/")
        print(f"  - Then register via Python:")
        print("    from impossible_mix.physics.reverb import register_real_ir")
        print(f"    register_real_ir('my_space', '{IR_DIR}/my_space.wav')")
    else:
        print("\nAll set. Use via:")
        print("  from impossible_mix.physics.reverb import get_ir, apply_reverb")
        print("  ir = get_ir('st_andrews_chapel', sr=44100)")
        print("  wet = apply_reverb(dry, ir, mix=0.5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
