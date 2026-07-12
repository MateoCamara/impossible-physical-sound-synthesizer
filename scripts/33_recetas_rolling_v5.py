"""Recetas de exploracion v5 de la gota rodante: suavidad y nucleo tonal.

Feedback v4: "frotar ropa mojada, aspero, no verdaderamente continuo; lava
es la menos aspera". Cada receta ataca una causa: smoothness (saca la AM de
la zona de aspereza 15-60 Hz), noise_darkness (banda grave, no tela),
tonal_mix (zumbido de canica) y sing_mix (canto de copa).

Uso:
    python scripts/33_recetas_rolling_v5.py            # renderiza las recetas
    python scripts/33_recetas_rolling_v5.py --check    # smoke minimo
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impossible_mix.physics.droplet import (  # noqa: E402
    DropletParams,
    synth_rolling_droplet,
)
from impossible_mix.physics.droplet_presets import get_preset  # noqa: E402
from impossible_mix.utils import save_wav  # noqa: E402

SR = 44_100
DUR_S = 6.0
SEED = 42
OUT_DIR = Path("escucha_AB/canica_v5")


def _water(**ov) -> DropletParams:
    p = get_preset("water", duration_s=DUR_S, seed=SEED)
    return replace(p, **ov) if ov else p


RECIPES: list[tuple[str, DropletParams]] = [
    ("00_ref__v4_actual", _water()),
    ("01_ref__lava_favorita", get_preset("lava", duration_s=DUR_S, seed=SEED)),
    ("10_suave__s05", _water(smoothness=0.5)),
    ("11_suave__s08", _water(smoothness=0.8)),
    ("20_oscuro__d07", _water(noise_darkness=0.7)),
    ("21_oscuro_suave__s06_d07", _water(smoothness=0.6, noise_darkness=0.7)),
    ("30_tonal__zumbido", _water(tonal_mix=0.8, smoothness=0.5,
                                 noise_darkness=0.5, accent_gain=0.2)),
    ("31_tonal_puro__ruido_min", _water(tonal_mix=1.0, smoothness=0.8,
                                        noise_darkness=0.9,
                                        continuous_core_mix=0.5)),
    ("40_canto__copa", _water(sing_mix=0.6, smoothness=0.6, noise_darkness=0.6)),
    ("41_canto_zumbido", _water(sing_mix=0.4, tonal_mix=0.6, smoothness=0.7,
                                noise_darkness=0.7)),
    ("50_lava_agua", _water(smoothness=0.7, noise_darkness=0.8, tonal_mix=0.6,
                            rev_wobble_depth=0.3)),
    ("51_lava_agua_canto", _water(smoothness=0.7, noise_darkness=0.8,
                                  tonal_mix=0.6, rev_wobble_depth=0.3,
                                  sing_mix=0.4)),
]


LEEME = """# Escucha v5: recetas de suavidad y nucleo tonal

Pregunta central: **¿cual se acerca mas a una CANICA CONTINUA rodando
mojada?** Ordena tus 2-3 favoritas y di que le sobra/falta a cada una.

Anclas: 00 = la v4 que criticaste ("frotar ropa mojada, aspero");
01 = la lava que te gusto (la menos aspera).

- **10-11 (suave)**: la modulacion sale de la zona de aspereza del oido.
- **20-21 (oscuro)**: el ruido pasa de banda brillante (tela) a cuerpo grave.
- **30-31 (zumbido)**: aparece el "hum" tonal continuo de canica; 31 es casi
  sin ruido.
- **40-41 (canto de copa)**: la canica "canta" en el modo grave del material.
- **50-51 (lava-agua)**: la receta de suavidad de la lava aplicada a los
  parametros del agua (misma velocidad y tono que 00).
"""


def _roughness_index(w: np.ndarray, sr: int) -> float:
    """Energia de la envolvente en la banda de aspereza 15-60 Hz (relativa)."""
    from scipy import signal as sg
    env = np.abs(sg.hilbert(w.astype(np.float64)))
    env = env - env.mean()
    sos = sg.butter(2, [15.0, 60.0], btype="band", fs=sr, output="sos")
    rough = sg.sosfiltfilt(sos, env)
    return float(np.sqrt(np.mean(rough ** 2)) / (np.sqrt(np.mean(env ** 2)) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="smoke minimo")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.check:
        p = RECIPES[6][1]  # 30_tonal__zumbido: ejercita tonal+smooth+dark
        w = synth_rolling_droplet(p, SR)
        ok = (np.isfinite(w).all() and np.abs(w).max() <= 0.9501
              and np.array_equal(w, synth_rolling_droplet(p, SR)))
        # defaults v4 intactos: receta 00 debe ser identica al motor sin overrides
        w00 = synth_rolling_droplet(_water(), SR)
        w00b = synth_rolling_droplet(get_preset("water", duration_s=DUR_S, seed=SEED), SR)
        ok = ok and np.array_equal(w00, w00b)
        print("smoke:", "OK" if ok else "FALLO")
        return 0 if ok else 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'receta':32s} {'pico':>5s} {'aspereza(15-60Hz)':>18s}")
    for name, params in RECIPES:
        w = synth_rolling_droplet(params, SR)
        save_wav(args.out_dir / f"{name}.wav", w, SR)
        print(f"{name:32s} {np.abs(w).max():5.2f} {_roughness_index(w, SR):18.3f}")
    (args.out_dir / "LEEME.md").write_text(LEEME, encoding="utf-8")
    print(f"\n{len(RECIPES)} recetas en {args.out_dir}/ (+ LEEME.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
