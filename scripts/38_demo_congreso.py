"""Paquete de demo para el congreso: catalogo masivo de chimeras + pagina.

Genera demo_congreso/ con:
  audio/estrellas/       las parejas validadas, 8 s, 2 variantes de bandas
  audio/barridos/        el "dial de identidad": n_bands 1->32 en un clip
  audio/antes_despues/   6 s de suma ponderada + 6 s de chimera (el pitch)
  audio/catalogo/        producto curado GOOD_ENV x GOOD_FINE, 6 s
  index.html             pagina estatica autonoma (file:// o http.server)

Uso:
    python scripts/38_demo_congreso.py --check
    python scripts/38_demo_congreso.py            # render completo (~10 min)
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impossible_mix.physics.blend import auditory_chimera_colored  # noqa: E402
from impossible_mix.physics.blend_recipes import (  # noqa: E402
    GOOD_ENV, GOOD_FINE, chimera_bands_heuristic, chimera_parent_v3,
)

SR = 44_100
SEED = 42
OUT = Path("demo_congreso")

STARS = [
    # (nombre, env, fine, bandas_a, bandas_b)
    ("trueno_hecho_de_agua", "trueno", "goteo", 4, 8),
    ("fuego_hecho_de_vidrio", "fuego", "vidrio", 16, 8),
    ("canica_hecha_de_fuego", "canica", "fuego", 12, 16),
    ("trueno_hecho_de_canica", "trueno", "canica", 6, 12),
]
SWEEP_BANDS = (1, 2, 4, 8, 16, 32)


class _ParentCache:
    def __init__(self, duration_s: float):
        self.dur = duration_s
        self._c: dict[str, np.ndarray] = {}

    def get(self, name: str, role: str) -> np.ndarray:
        seed = SEED if role == "env" else SEED + 17
        key = f"{name}|{role}"
        if key not in self._c:
            self._c[key] = chimera_parent_v3(name, self.dur, seed, SR)
        return self._c[key]


def _norm9(w: np.ndarray) -> np.ndarray:
    peak = float(np.abs(w).max() + 1e-9)
    return (w * (0.9 / peak)).astype(np.float32)


def _chimera(cache: _ParentCache, env: str, fine: str, nb: int) -> np.ndarray:
    return _norm9(auditory_chimera_colored(cache.get(env, "env"),
                                           cache.get(fine, "fine"),
                                           SR, n_bands=nb))


def _sweep_clip(cache: _ParentCache, env: str, fine: str,
                seg_s: float = 3.0, xfade_s: float = 0.15) -> np.ndarray:
    """n_bands recorre SWEEP_BANDS en segmentos con crossfade corto."""
    seg_n = int(seg_s * SR)
    xf_n = int(xfade_s * SR)
    segs = []
    for nb in SWEEP_BANDS:
        w = _chimera(cache, env, fine, nb)
        segs.append(w[:seg_n])
    total = seg_n * len(segs) - xf_n * (len(segs) - 1)
    out = np.zeros(total, dtype=np.float32)
    pos = 0
    fade_in = np.linspace(0, 1, xf_n, dtype=np.float32)
    for i, s in enumerate(segs):
        s = s.copy()
        if i > 0:
            s[:xf_n] *= fade_in
        if i < len(segs) - 1:
            s[-xf_n:] *= fade_in[::-1]
        out[pos:pos + seg_n] += s
        pos += seg_n - xf_n
    return _norm9(out)


def _before_after(cache: _ParentCache, env: str, fine: str, nb: int,
                  half_s: float = 6.0, gap_s: float = 0.3) -> np.ndarray:
    n = int(half_s * SR)
    a = cache.get(env, "env")[:n]
    b = cache.get(fine, "fine")[:n]
    suma = _norm9(0.6 * a + 0.6 * b)
    chim = _chimera(cache, env, fine, nb)[:n]
    return np.concatenate([suma, np.zeros(int(gap_s * SR), dtype=np.float32), chim])


def build_items(duration_s: float = 6.0):
    """(categoria, fichero_relativo, titulo, descripcion, render_fn)."""
    cache8 = _ParentCache(8.0)
    cache6 = _ParentCache(duration_s)
    items = []
    for name, env, fine, nb_a, nb_b in STARS:
        for nb in (nb_a, nb_b):
            items.append((
                "estrellas", f"audio/estrellas/{name}__{nb}bandas.wav",
                f"{name.replace('_', ' ')} ({nb} bandas)",
                f"Dinámica: {env}. Materia: {fine}. {nb} bandas.",
                lambda e=env, f=fine, n=nb: _chimera(cache8, e, f, n)))
    for name, env, fine, *_ in STARS:
        items.append((
            "barridos", f"audio/barridos/{name}__dial.wav",
            f"Dial de identidad: {name.replace('_', ' ')}",
            f"n_bands recorre {SWEEP_BANDS}: la identidad transita de la "
            f"materia ({fine}) a la dinámica ({env}) dentro de UNA señal.",
            lambda e=env, f=fine: _sweep_clip(cache8, e, f)))
    for name, env, fine, nb_a, _ in STARS:
        items.append((
            "antes_despues", f"audio/antes_despues/{name}__suma_vs_chimera.wav",
            f"Antes/después: {name.replace('_', ' ')}",
            "Primeros 6 s: suma ponderada (dos capas). Últimos 6 s: chimera "
            "(una identidad). Mismos dos padres.",
            lambda e=env, f=fine, n=nb_a: _before_after(cache8, e, f, n)))
    star_set = {(e, f) for _, e, f, *_ in STARS}
    for env in GOOD_ENV:
        for fine in GOOD_FINE:
            if env == fine or (env, fine) in star_set:
                continue
            nb = chimera_bands_heuristic(env)
            items.append((
                f"catalogo/{env}", f"audio/catalogo/{env}_hecho_de_{fine}.wav",
                f"{env} hecho de {fine}",
                f"{nb} bandas (heurística).",
                lambda e=env, f=fine, n=nb: _chimera(cache6, e, f, n)))
    return items


def _write_html(items, out_dir: Path) -> None:
    groups: dict[str, list] = {}
    for cat, rel, title, desc, _ in items:
        groups.setdefault(cat.split("/")[0] if "/" not in cat else cat, []).append(
            (rel, title, desc))
    order = (["estrellas", "barridos", "antes_despues"]
             + sorted(g for g in groups if g.startswith("catalogo/")))
    titles = {"estrellas": "⭐ Estrellas", "barridos": "🎛️ El dial de identidad",
              "antes_despues": "↔️ Antes (suma) / después (chimera)"}
    parts = ["""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>¿Cómo suena una gota que rueda? — Demo de chimeras</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#101418;color:#e8e8e8}
h1{font-size:1.5rem} h2{margin-top:2.2rem;border-bottom:1px solid #333;padding-bottom:.3rem}
.clip{margin:.9rem 0;padding:.7rem;background:#1a2027;border-radius:8px}
.clip b{display:block;margin-bottom:.15rem} .clip small{color:#9ab}
audio{width:100%;margin-top:.4rem}
p.lead{color:#9ab}
</style></head><body>
<h1>¿Cómo suena una gota que rueda? — Chimeras de sonidos imposibles</h1>
<p class="lead">Fusión de identidad por chimera auditiva (Smith, Delgutte &amp;
Oxenham, <i>Nature</i> 2002): un fenómeno pone la <b>dinámica</b> (envolvente),
otro pone la <b>materia</b> (estructura fina) — una sola señal por construcción.
Motor de síntesis físico paramétrico; todo determinista y reproducible.</p>
"""]
    for g in order:
        if g not in groups:
            continue
        parts.append(f"<h2>{html.escape(titles.get(g, g.replace('catalogo/', 'Catálogo — dinámica: ')))}</h2>")
        for rel, title, desc in groups[g]:
            parts.append(
                f'<div class="clip"><b>{html.escape(title)}</b>'
                f'<small>{html.escape(desc)}</small>'
                f'<audio controls preload="none" src="{html.escape(rel)}"></audio></div>')
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def render(out_dir: Path) -> int:
    from impossible_mix.utils import save_wav
    items = build_items()
    for cat, rel, title, _, fn in items:
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        save_wav(path, fn(), SR)
        print(f"  {rel}")
    _write_html(items, out_dir)
    print(f"\n{len(items)} clips + index.html en {out_dir}/")
    return 0


def check() -> int:
    cache = _ParentCache(3.0)
    failures = []
    for env, fine, nb in (("trueno", "goteo", 4), ("fuego", "vidrio", 16),
                          ("canica", "fuego", 12)):
        w = _chimera(cache, env, fine, nb)
        if not (np.isfinite(w).all() and np.abs(w).max() <= 0.9501
                and np.array_equal(w, _chimera(cache, env, fine, nb))):
            failures.append(f"{env}x{fine}")
    items = build_items()
    n_cat = sum(1 for c, *_ in items if c.startswith("catalogo/"))
    print(f"items: {len(items)} (catalogo {n_cat}, estrellas "
          f"{sum(1 for c, *_ in items if c == 'estrellas')}, "
          f"barridos {sum(1 for c, *_ in items if c == 'barridos')}, "
          f"antes_despues {sum(1 for c, *_ in items if c == 'antes_despues')})")
    if failures:
        print("FALLOS:", failures)
        return 1
    print("smoke OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    return check() if args.check else render(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
