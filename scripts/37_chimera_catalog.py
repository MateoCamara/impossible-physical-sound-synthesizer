"""Gran catalogo de chimeras (v10b): la gota rodante en ambos papeles +
barrido amplio de parejas + variantes anti-estridencia coloreadas.

Uso:
    python scripts/37_chimera_catalog.py            # renderiza el catalogo
    python scripts/37_chimera_catalog.py --check    # smoke minimo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impossible_mix.physics.blend import (  # noqa: E402
    auditory_chimera, auditory_chimera_colored,
)
from impossible_mix.physics.blend_recipes import (  # noqa: E402
    CHIMERA_PARENTS_V2, chimera_bands_heuristic, chimera_parent_v2,
)

SR = 44_100
DUR = 8.0
SEED = 42
OUT = Path("escucha_AB/chimeras_catalogo")


def _pairs():
    """(grupo, nombre, env_parent, fine_parent, n_bands, colored)."""
    pairs = []
    # Grupo A: la GOTA RODANTE en ambos papeles (peticion del usuario)
    others = [p for p in CHIMERA_PARENTS_V2 if p != "canica"]
    for o in others:
        pairs.append(("A_canica_dinamica", f"canica_hecha_de_{o}",
                      "canica", o, chimera_bands_heuristic("canica"), True))
    for o in others:
        pairs.append(("B_canica_materia", f"{o}_hecho_de_canica",
                      o, "canica", chimera_bands_heuristic(o), True))
    # Grupo C: barrido amplio de otras parejas prometedoras
    broad = [
        ("trueno", "goteo"), ("trueno", "vidrio"), ("trueno", "campana_tela"),
        ("fuego", "vidrio"), ("fuego", "goteo"), ("fuego", "campana_tela"),
        ("lluvia", "vidrio"), ("lluvia", "campana_tela"), ("lluvia", "grava"),
        ("oceano", "vidrio"), ("oceano", "grava"), ("oceano", "goteo"),
        ("grava", "goteo"), ("grava", "campana_tela"), ("grava", "vidrio"),
        ("vertido", "vidrio"), ("vertido", "campana_tela"),
        ("goteo", "vidrio"), ("campana_tela", "goteo"), ("vidrio", "fuego"),
    ]
    for a, b in broad:
        pairs.append(("C_barrido", f"{a}_x_{b}", a, b,
                      chimera_bands_heuristic(a), True))
    # Grupo D: A/B de estridencia sobre las dos validadas
    pairs.append(("D_color", "fuego_vidrio__plana", "fuego", "vidrio", 16, False))
    pairs.append(("D_color", "fuego_vidrio__coloreada", "fuego", "vidrio", 16, True))
    pairs.append(("D_color", "trueno_agua__plana", "trueno", "goteo", 4, False))
    pairs.append(("D_color", "trueno_agua__coloreada", "trueno", "goteo", 4, True))
    return pairs


def render(out_dir: Path) -> int:
    from impossible_mix.utils import save_wav
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, int], np.ndarray] = {}

    def parent(name, seed):
        key = (name, seed)
        if key not in cache:
            cache[key] = chimera_parent_v2(name, DUR, seed, SR)
        return cache[key]

    idx = 0
    last_group = None
    for group, name, a, b, nb, colored in _pairs():
        if group != last_group:
            print(f"--- {group}")
            last_group = group
        wa = parent(a, SEED)
        wb = parent(b, SEED + 17)
        fn = auditory_chimera_colored if colored else auditory_chimera
        w = (fn(wa, wb, SR, n_bands=nb, color_from="b") if colored
             else fn(wa, wb, SR, n_bands=nb))
        peak = float(np.abs(w).max() + 1e-9)
        w = (w * (0.9 / peak)).astype(np.float32)
        fname = f"{group}__{idx:02d}_{name}.wav"
        save_wav(out_dir / fname, w, SR)
        print(f"  {fname} ({nb}b{'|color' if colored else ''})")
        idx += 1
    (out_dir / "LEEME.md").write_text(
        "# Catalogo de chimeras v10b\n\n"
        "- **A_canica_dinamica**: la gota rodante pone la DINAMICA "
        "(su ritmo de rodadura), la materia es cada otro fenomeno.\n"
        "- **B_canica_materia**: la gota rodante es la MATERIA (su textura "
        "acuosa) y cada otro fenomeno pone la dinamica.\n"
        "- **C_barrido**: 20 parejas nuevas en sus puntos heuristicos.\n"
        "- **D_color**: A/B anti-estridencia — 'coloreada' hereda el balance "
        "espectral del padre-materia (deberia sonar menos estridente que "
        "'plana' manteniendo la identidad).\n\n"
        "Marca tus favoritas (3-5 para el paper) y di si la version "
        "coloreada arregla la estridencia.\n", encoding="utf-8")
    print(f"\n{idx} clips en {out_dir}/")
    return 0


def check() -> int:
    a = chimera_parent_v2("canica", 3.0, SEED, SR)
    b = chimera_parent_v2("fuego", 3.0, SEED + 17, SR)
    w1 = auditory_chimera_colored(a, b, SR, 12)
    ok = (np.isfinite(w1).all() and np.abs(w1).max() <= 0.9501
          and np.array_equal(w1, auditory_chimera_colored(a, b, SR, 12)))
    print("smoke:", "OK" if ok else "FALLO")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    return check() if args.check else render(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
