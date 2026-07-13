"""Hibridos de identidad v8: smokes y matriz de escucha.

Uso:
    python scripts/35_listen_identity_hybrids.py --check   # gates duros
    python scripts/35_listen_identity_hybrids.py           # matriz de escucha

Gates duros: checksums dorados (via script 34), no-NaN/pico/determinismo
por hibrido, y glide de Minnaert MEDIBLE en H1/H2 (el pitch sube dentro
del evento). El indice de unicidad de stream (USI) se imprime como
columna INFORMATIVA: en pruebas no discrimina de forma fiable una-voz vs
dos-voces (los eventos banda-ancha correlacionan casi todos los pares de
bandas tambien en la suma), asi que NO es gate — la escucha manda.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impossible_mix.physics import blend_recipes as br  # noqa: E402
from impossible_mix.physics import identity_hybrids as ih  # noqa: E402
from impossible_mix.physics.analysis import stream_unity_index  # noqa: E402

SR = 44_100
OUT_DIR = Path("escucha_AB/hibridos_v8")


def _measure_glide(w: np.ndarray, sr: int, t0_s: float, dur_s: float,
                   band_hz: tuple[float, float],
                   nperseg: int = 256) -> tuple[float, float]:
    """Frecuencia de pico (STFT) al principio y al final de la ventana.

    nperseg fija la resolucion (sr/nperseg Hz por bin): para glides graves
    (H1, 46->163 Hz) hace falta ventana larga (4096 -> ~11 Hz/bin)."""
    from scipy import signal as sg
    seg = w[int(t0_s * sr): int((t0_s + dur_s) * sr)]
    f, t, Z = sg.stft(seg, sr, nperseg=nperseg, noverlap=int(nperseg * 0.875))
    m = (f > band_hz[0]) & (f < band_hz[1])
    peaks = f[m][np.abs(Z[m]).argmax(axis=0)]
    q = max(1, len(peaks) // 4)
    return float(np.median(peaks[:q])), float(np.median(peaks[-q:]))


HYBRIDS = [
    ("hybrid_thunder_is_droplet", lambda: ih.hybrid_thunder_is_droplet(seed=42)),
    ("hybrid_bubbling_glass_impact", lambda: ih.hybrid_bubbling_glass_impact(seed=42)),
    ("hybrid_bubbling_glass_roll", lambda: ih.hybrid_bubbling_glass_roll(seed=42)),
    ("hybrid_rain_of_thunders", lambda: ih.hybrid_rain_of_thunders(seed=42)),
    ("hybrid_identity_vocoder", lambda: ih.hybrid_identity_vocoder(seed=42)),
]


def run_checks() -> int:
    failures = []
    # 1. Checksums dorados via script 34 (protege todo lo publico)
    r = subprocess.run([sys.executable, str(Path(__file__).with_name("34_listen_blends.py")),
                        "--check"], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    if r.returncode != 0:
        failures.append(f"script 34 --check fallo:\n{r.stdout[-500:]}")

    # 2. Por hibrido: NaN/pico/determinismo
    renders = {}
    for name, fn in HYBRIDS:
        w = fn()
        renders[name] = w
        if not np.isfinite(w).all():
            failures.append(f"{name}: NaN/Inf")
        if np.abs(w).max() > 0.9501:
            failures.append(f"{name}: pico {np.abs(w).max():.3f}")
        if not np.array_equal(w, fn()):
            failures.append(f"{name}: no determinista")

    # 3. Glide de Minnaert medible
    f_ini, f_fin = _measure_glide(renders["hybrid_thunder_is_droplet"], SR,
                                  0.0, 0.35, (35, 400), nperseg=4096)
    if not f_fin > 1.25 * f_ini:
        failures.append(f"H1 sin glide grave ({f_ini:.0f}->{f_fin:.0f} Hz)")
    else:
        print(f"  H1 glide: {f_ini:.0f} -> {f_fin:.0f} Hz")
    f_ini2, f_fin2 = _measure_glide(renders["hybrid_bubbling_glass_impact"], SR,
                                    0.35, 0.035, (700, 2200))
    if not f_fin2 > 1.3 * f_ini2:
        failures.append(f"H2 sin glide modal ({f_ini2:.0f}->{f_fin2:.0f} Hz)")
    else:
        print(f"  H2 glide: {f_ini2:.0f} -> {f_fin2:.0f} Hz")

    if failures:
        print("CHECKS FALLIDOS:")
        for f in failures:
            print("  -", f)
        return 1
    print("Checks OK: checksums + 5 hibridos (NaN/pico/determinismo) + glides.")
    return 0


LEEME = """# Escucha v8: HIBRIDOS DE IDENTIDAD

Pregunta central: **¿oyes UN objeto imposible — una sola voz cuya fisica
no cuadra — o sigues oyendo dos cosas correlacionadas?**

- **00/01 el trueno ES una gota gigante**: cero capas de trueno real; es
  una burbuja de Minnaert de 32 mm (102 Hz) con crack capilar gigante y
  wobble por los modos de Rayleigh reales del radio. 01 es la variante
  "limpia" (diagnostico: ¿suena a laser?).
- **10/11 cristal que burbujea**: cada modo del vidrio hace el glide de
  una burbuja y su decay se acelera al subir (ley van den Doel). 10
  impactos, 11 rodadura.
- **20/21 lluvia de truenos diminutos**: cada gota es un trueno
  miniaturizado (crack + rumble descendente). 21 mas escasa (grano legible).
- **30/31 vocoder de identidad**: tu favorito v7 cerrando las tres brechas
  (mismo registro, pitch compartido, estructura fina del goteo). 31 con
  mas cuerpo.
- **80/81 anclas v7** (trueno_habla, trueno_gotea): ¿el hibrido los mata?
- **90 ancla suma**: la referencia "dos corrientes".

Di cuales suenan por fin a UNA mezcla imposible y cuales no, y que falla.
"""


def render_matrix(out_dir: Path) -> int:
    from impossible_mix.utils import save_wav
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [
        ("00_trueno_es_gota__a", lambda: ih.hybrid_thunder_is_droplet(seed=42)),
        ("01_trueno_es_gota__b_limpio", lambda: ih.hybrid_thunder_is_droplet(
            seed=42, dirt=0.0, n_subbubbles=1)),
        ("10_cristal_burbujea__a_impacto", lambda: ih.hybrid_bubbling_glass_impact(seed=42)),
        ("11_cristal_burbujea__b_rodadura", lambda: ih.hybrid_bubbling_glass_roll(seed=42)),
        ("20_lluvia_truenos__a", lambda: ih.hybrid_rain_of_thunders(seed=42)),
        ("21_lluvia_truenos__b_escasa", lambda: ih.hybrid_rain_of_thunders(
            seed=42, density_hz=6.0)),
        ("30_vocoder_identidad__a", lambda: ih.hybrid_identity_vocoder(seed=42)),
        ("31_vocoder_identidad__b_cuerpo", lambda: ih.hybrid_identity_vocoder(
            seed=42, fine=0.5, body_q=12.0)),
        ("80_ancla_v7__trueno_habla", lambda: br.blend_thunder_speaks_water(8.0, 42)),
        ("81_ancla_v7__trueno_gotea", lambda: br.blend_thunder_drips(8.0, 42)),
        ("90_ancla_suma__trueno_gotas", lambda: br.sum_thunder_drips(8.0, 42)),
    ]
    print(f"{'clip':38s} {'pico':>5s} {'USI(informativo)':>17s}")
    for name, fn in items:
        w = fn()
        save_wav(out_dir / f"{name}.wav", w, SR)
        r = stream_unity_index(w, SR)
        print(f"{name:38s} {np.abs(w).max():5.2f} {r.usi:17.3f}")
    (out_dir / "LEEME.md").write_text(LEEME, encoding="utf-8")
    print(f"\n{len(items)} clips en {out_dir}/ (+ LEEME.md)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    if args.check:
        return run_checks()
    return render_matrix(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
