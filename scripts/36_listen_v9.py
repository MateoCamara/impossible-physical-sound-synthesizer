"""Fusion v9 fundamentada en literatura: matriz de escucha + smokes.

Tres metodos con evidencia perceptual publicada:
  M1 chimeras auditivas (Smith, Delgutte & Oxenham, Nature 2002)
  M2 alineacion de registro antes de fundir (Slaney et al., ICASSP 1996)
  M3 morphing de estadisticas de textura (McDermott & Simoncelli, Neuron 2011)

Uso:
    python scripts/36_listen_v9.py --check   # gates
    python scripts/36_listen_v9.py           # matriz (los morphs tardan ~1 min c/u)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impossible_mix.physics import blend_recipes as br  # noqa: E402
from impossible_mix.physics.analysis import smooth_env  # noqa: E402
from impossible_mix.physics.blend import auditory_chimera, schedule_from_rate  # noqa: E402
from impossible_mix.physics.exotic import _thunder_rumble, synth_fire, synth_rain  # noqa: E402
from impossible_mix.physics.texture_stats import (  # noqa: E402
    impose_statistics, stats_distance, texture_statistics, texture_morph,
)

SR = 44_100
OUT_DIR = Path("escucha_AB/fusion_v9")


def _parents_thunder_drips(duration_s: float = 6.0, seed: int = 42):
    n = int(duration_s * SR)
    rng = np.random.default_rng(seed)
    rumble, _ = _thunder_rumble(rng, SR, n, 0.5, 0.9)
    sched = schedule_from_rate(np.full(n, 10.0), SR, seed + 3,
                               radius_traj=np.full(n, 2.2))
    drips = br._render_drips(sched, SR, n, seed + 3)
    return rumble, drips


def run_checks() -> int:
    failures = []
    r = subprocess.run([sys.executable, str(Path(__file__).with_name("34_listen_blends.py")),
                        "--check"], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    if r.returncode != 0:
        failures.append("script 34 --check fallo")

    # M1: chimera — envolvente de A, estructura fina de B, determinista
    rumble, drips = _parents_thunder_drips()
    w = auditory_chimera(rumble, drips, SR, n_bands=16)
    if not np.isfinite(w).all() or np.abs(w).max() > 0.9501:
        failures.append("chimera: NaN o pico")
    if not np.array_equal(w, auditory_chimera(rumble, drips, SR, n_bands=16)):
        failures.append("chimera: no determinista")
    ea = smooth_env(rumble, SR, 20, fs_out=200)
    eo = smooth_env(w, SR, 20, fs_out=200)
    m = min(len(ea), len(eo))
    c_env = float(np.corrcoef(ea[:m], eo[:m])[0, 1])
    if c_env < 0.8:
        failures.append(f"chimera: envolvente no sigue a A (corr {c_env:.2f})")
    else:
        print(f"  M1 chimera: corr(env, A) = {c_env:.3f}")

    # M3: sanidad de la reimplementacion (reimponer lluvia ~ lluvia)
    rain = synth_rain(duration_s=4.0, intensity=0.7, sr=SR, seed=42)
    fire = synth_fire(duration_s=4.0, intensity=0.7, sr=SR, seed=42)
    s_rain = texture_statistics(rain, SR)
    s_fire = texture_statistics(fire, SR)
    re_rain, hist = impose_statistics(s_rain, SR, 4.0, n_iter=12, seed=7,
                                      return_history=True)
    if not hist[-1] < hist[0]:
        failures.append(f"texturas: no converge ({hist[0]:.2f}->{hist[-1]:.2f})")
    s_re = texture_statistics(re_rain, SR)
    d_r, d_f = stats_distance(s_re, s_rain), stats_distance(s_re, s_fire)
    if not d_r < d_f:
        failures.append(f"texturas: sanidad falla (d_rain {d_r:.2f} >= d_fire {d_f:.2f})")
    else:
        print(f"  M3 sanidad: d(re,rain)={d_r:.2f} << d(re,fire)={d_f:.2f}; "
              f"converge {hist[0]:.2f}->{hist[-1]:.2f}")

    # M2: alineadas renderizan y son deterministas
    for fn in (br.blend_thunder_drips_aligned, br.blend_glass_fire_aligned):
        w = fn(4.0, 42)
        if not (np.isfinite(w).all() and np.abs(w).max() <= 0.9501
                and np.array_equal(w, fn(4.0, 42))):
            failures.append(f"{fn.__name__}: NaN/pico/determinismo")

    if failures:
        print("CHECKS FALLIDOS:")
        for f in failures:
            print("  -", f)
        return 1
    print("Checks OK: chimera + sanidad de texturas + alineadas + checksums.")
    return 0


LEEME = """# Escucha v9: fusion FUNDAMENTADA EN LA LITERATURA

Tres metodos con evidencia perceptual publicada de producir UNA identidad:

- **00-03 CHIMERAS** (Smith, Delgutte & Oxenham, Nature 2002): envolvente
  del trueno x estructura fina del goteo, UNA senal por construccion. El
  numero de bandas es el dial de identidad: en 00 (1 banda) domina el
  agua; en 03 (32 bandas) domina el trueno. ¿Se OYE la identidad
  transitar dentro de una sola voz?
- **10-11 chimeras fuego/vidrio** (10: env fuego x fina vidrio; 11 inversa).
- **20-21 ALINEADAS** (Slaney et al., ICASSP 1996: alinear registros antes
  de fundir colapsa dos objetos en uno). 20 = trueno_gotea con gotas
  GIGANTES cuyo pitch vive en la banda del rumble (compara con 80, la v7
  desalineada). 21 = fuego_cristal con el agua retunada al modo del vidrio.
- **30-33 MORPH DE ESTADISTICAS** (McDermott & Simoncelli, Neuron 2011):
  30/31/32 = lluvia<->fuego con alpha 0.25/0.5/0.75 — UNA textura
  estadisticamente intermedia, no dos superpuestas. 33 = control de
  sanidad (lluvia re-sintetizada desde sus estadisticas: debe sonar a
  lluvia).
- **80/90 anclas**: la v7 desalineada y la suma.

¿Cual de los TRES METODOS produce por fin una mezcla imposible de verdad?
"""


def render_matrix(out_dir: Path) -> int:
    from impossible_mix.utils import save_wav
    out_dir.mkdir(parents=True, exist_ok=True)
    rumble, drips = _parents_thunder_drips(8.0, 42)
    fire8 = synth_fire(duration_s=8.0, intensity=0.7, sr=SR, seed=42)
    from impossible_mix.physics.exotic import synth_glass_break
    glass8 = synth_glass_break(duration_s=8.0, n_shards=60, sr=SR, seed=42)
    rain5 = synth_rain(duration_s=5.0, intensity=0.7, sr=SR, seed=42)
    fire5 = synth_fire(duration_s=5.0, intensity=0.7, sr=SR, seed=42)

    items = [
        ("00_chimera_trueno_gotas__1banda", lambda: auditory_chimera(rumble, drips, SR, 1)),
        ("01_chimera_trueno_gotas__4bandas", lambda: auditory_chimera(rumble, drips, SR, 4)),
        ("02_chimera_trueno_gotas__16bandas", lambda: auditory_chimera(rumble, drips, SR, 16)),
        ("03_chimera_trueno_gotas__32bandas", lambda: auditory_chimera(rumble, drips, SR, 32)),
        ("10_chimera_fuego_x_vidrio", lambda: auditory_chimera(fire8, glass8, SR, 16)),
        ("11_chimera_vidrio_x_fuego", lambda: auditory_chimera(glass8, fire8, SR, 16)),
        ("20_alineado_trueno_gotea", lambda: br.blend_thunder_drips_aligned(8.0, 42)),
        ("21_alineado_fuego_cristal", lambda: br.blend_glass_fire_aligned(8.0, 42)),
        ("30_morph_lluvia_fuego__a025", lambda: texture_morph(rain5, fire5, 0.25, SR, n_iter=16, seed=7)),
        ("31_morph_lluvia_fuego__a050", lambda: texture_morph(rain5, fire5, 0.50, SR, n_iter=16, seed=7)),
        ("32_morph_lluvia_fuego__a075", lambda: texture_morph(rain5, fire5, 0.75, SR, n_iter=16, seed=7)),
        ("33_control__lluvia_reimpuesta", lambda: impose_statistics(
            texture_statistics(rain5, SR), SR, 5.0, n_iter=16, seed=7)),
        ("80_ancla_v7__trueno_gotea_desalineado", lambda: br.blend_thunder_drips(8.0, 42)),
        ("90_ancla_suma__trueno_gotas", lambda: br.sum_thunder_drips(8.0, 42)),
    ]
    for name, fn in items:
        w = fn()
        # Igualar pico a 0.9 para que la comparacion de oido no este sesgada
        # por sonoridad (las chimeras salen naturalmente bajas).
        peak = float(np.abs(w).max() + 1e-9)
        w = (w * (0.9 / peak)).astype(np.float32)
        save_wav(out_dir / f"{name}.wav", w, SR)
        print(f"  {name}.wav  peak={np.abs(w).max():.2f}")
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
