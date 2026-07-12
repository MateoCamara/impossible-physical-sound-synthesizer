"""Matriz de escucha del motor v3 "canica mojada" + smokes numericos.

Renderiza variantes de la gota rodante a escucha_AB/canica_mojada/ (sin
trackear) con un LEEME.md generado, para validacion de oido del usuario
antes de portar el motor a DDSP/web y regenerar el material del paper.

Uso:
    python scripts/32_listen_rolling_marble.py            # renderiza la matriz
    python scripts/32_listen_rolling_marble.py --check    # solo smokes (exit!=0 si fallan)
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
    synth_rolling_droplet_legacy,
)
from impossible_mix.physics.droplet_presets import get_preset  # noqa: E402
from impossible_mix.utils import save_wav  # noqa: E402

SR = 44_100
DUR_S = 6.0
SEED = 42
OUT_DIR = Path("escucha_AB/canica_mojada")


def _canonical(**overrides) -> DropletParams:
    p = get_preset("water", duration_s=DUR_S, seed=SEED)
    return replace(p, **overrides) if overrides else p


def build_matrix() -> list[tuple[str, DropletParams, bool]]:
    """(nombre, params, usar_legacy)."""
    m: list[tuple[str, DropletParams, bool]] = []
    m.append(("00_ab__legacy_water", _canonical(), True))
    m.append(("01_ab__nueva_water", _canonical(), False))
    for name, v in (("10_vel__lenta_6hz", 6.0), ("11_vel__media_14hz", 14.0),
                    ("12_vel__rapida_26hz", 26.0)):
        m.append((name, _canonical(roll_velocity_hz=v), False))
    m.append(("20_agua__mojada_visc00", _canonical(viscosity=0.0), False))
    m.append(("21_agua__seca_visc07", _canonical(viscosity=0.7), False))
    for name, preset in (("30_mat__water", "water"), ("31_mat__mercury", "mercury"),
                         ("32_mat__lava", "lava")):
        m.append((name, get_preset(preset, duration_s=DUR_S, seed=SEED), False))
    for name, w in (("40_wobble__off_000", 0.0), ("41_wobble__def_022", 0.22),
                    ("42_wobble__alto_045", 0.45)):
        m.append((name, _canonical(rev_wobble_depth=w), False))
    for name, k in (("50_asper__3", 3), ("51_asper__6", 6), ("52_asper__10", 10)):
        m.append((name, _canonical(asperities_per_rev=k), False))
    for name, d in (("60_dens__x1", 1.0), ("61_dens__x2", 2.0), ("62_dens__x3", 3.0)):
        m.append((name, _canonical(contact_density_mul=d), False))
    for name, r in (("70_rumble__off", 0.0), ("71_rumble__def", 0.75),
                    ("72_rumble__x2", 1.5)):
        m.append((name, _canonical(continuous_layer_mix=r), False))
    for name, dr in (("80_drift__0", 0.0), ("81_drift__005", 0.05),
                     ("82_drift__02", 0.2)):
        m.append((name, _canonical(pattern_drift=dr), False))
    return m


LEEME = """# Escucha: motor v3 "canica mojada" (rama rolling-v3)

Objetivo: gota que RUEDA como canica mojada — ritmo de micro-contactos
cuasi-periodicos donde cada contacto es un blip acuoso (chirp Minnaert),
sin la "aspiradora" (ruido continuo) del motor anterior.

## Que escuchar

- **00 vs 01 (el A/B clave)**: ¿desaparece la aspiradora? ¿se oye agua en
  cada contacto y rodadura en el ritmo?
- **10-12 (velocidad)**: ¿mas velocidad se lee como rodar mas rapido, no
  como "mas gotas cayendo"?
- **20-21 (agua)**: mojada = blips vivos con pops; seca/viscosa = mate.
- **30-32 (materiales)**: ¿mercury metalica y nerviosa, lava lenta y grave,
  siguen diferenciadas?
- **40-42 (wobble)**: ¿la canica "respira" (acelera/frena por vuelta)?
  ¿0.45 exagera?
- **50-52 (asperezas/vuelta)**: ¿cambia el "dibujo" ritmico de la rodadura?
- **60-62 (densidad)**: ¿x1 suena a rebote/metralleta? ¿x3 se emborrona?
- **70-72 (rumor gated)**: ¿sin rumor queda hueco o bastan las colas?
- **80-82 (drift)**: ¿el patron que precesa lentamente se percibe mas
  organico que el drift 0?

Di el NOMBRE DE FICHERO de lo que suene mal y que le sobra/falta.
"""


# ------------------------------------------------------------------
# Smokes
# ------------------------------------------------------------------
def _detect_onsets(w: np.ndarray, sr: int, expected_rate: float) -> np.ndarray:
    """Onsets por pico de envolvente HP>1kHz (indices de sample).

    La distancia minima entre picos se liga a la tasa esperada (0.5/rate)
    para no contar doble el click y el pico del chirp de un mismo contacto.
    """
    from scipy import signal as sg
    sos = sg.butter(4, 1000, btype="high", fs=sr, output="sos")
    hp = sg.sosfilt(sos, w.astype(np.float64))
    env = np.abs(hp)
    win = max(8, int(0.005 * sr))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    thr = 0.35 * env.max()
    min_dist = max(int(0.008 * sr), int(0.5 / max(expected_rate, 1.0) * sr))
    peaks, _ = sg.find_peaks(env, height=thr, distance=min_dist)
    return peaks


def _mod_index(w: np.ndarray, sr: int) -> float:
    """Indice de modulacion de envolvente (std/mean). Una 'aspiradora'
    (ruido continuo) da valores bajos; una rodadura de eventos, altos."""
    from scipy import signal as sg
    env = np.abs(sg.hilbert(w.astype(np.float64)))
    win = max(8, int(0.005 * sr))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    return float(env.std() / (env.mean() + 1e-12))


def run_checks() -> int:
    failures = []
    p = _canonical()
    w = synth_rolling_droplet(p, SR)

    # 1. No-NaN/Inf y pico
    if not np.isfinite(w).all():
        failures.append("NaN/Inf en la canonica")
    if np.abs(w).max() > 0.9501:
        failures.append(f"pico {np.abs(w).max():.3f} > 0.95")

    # 2. Densidad de eventos: scheduler exacto + deteccion en audio ±30%
    from impossible_mix.physics.droplet import _derive_params, _roll_schedule
    p_sched = _derive_params(_canonical())
    sched = _roll_schedule(p_sched, SR, int(DUR_S * SR), np.random.default_rng(p_sched.seed))
    target_rate = p.roll_velocity_hz * p.contact_density_mul
    sched_rate = len(sched.starts) / DUR_S
    if abs(sched_rate - target_rate) > 0.1 * target_rate:
        failures.append(f"scheduler {sched_rate:.1f}/s != objetivo {target_rate:.1f}/s")
    onsets = _detect_onsets(w, SR, target_rate)
    measured = len(onsets) / DUR_S
    if not (0.7 * target_rate <= measured <= 1.3 * target_rate):
        failures.append(
            f"densidad audible {measured:.1f}/s fuera de ±30% del objetivo {target_rate:.1f}/s")

    # 3. Anti-aspiradora: indice de modulacion de envolvente muy superior
    # al motor legacy (y alto en absoluto). Con colas que solapan por diseno
    # el criterio de 'huecos silenciosos' no aplica; la modulacion si.
    mi_v3 = _mod_index(w, SR)
    w_legacy = synth_rolling_droplet_legacy(_canonical(), SR)
    mi_legacy = _mod_index(w_legacy, SR)
    if not (mi_v3 > 0.35 and mi_v3 > 1.3 * mi_legacy):
        failures.append(
            f"modulacion v3 {mi_v3:.2f} insuficiente (legacy {mi_legacy:.2f}; "
            f"se exige >0.35 y >1.3x legacy)")

    # 4. Periodicidad de vuelta: autocorrelacion de las amplitudes del
    # scheduler con pico en lag K (el patron de asperezas se repite por
    # revolucion). Se mide en el schedule: el audio crudo es demasiado
    # ruidoso para este test y la densidad audible ya se valida en (2).
    if len(sched.amps) > 30:
        amps = sched.amps - sched.amps.mean()
        K = 4 + int(round(4 * p.path_roughness))
        ac = np.correlate(amps, amps, mode="full")[len(amps) - 1:]
        ac = ac / (ac[0] + 1e-12)
        if ac[K] < 0.25:
            failures.append(
                f"sin periodicidad de vuelta en el schedule (ac[K={K}]={ac[K]:.2f})")
    # 5. Determinismo
    w2 = synth_rolling_droplet(_canonical(), SR)
    if not np.array_equal(w, w2):
        failures.append("no determinista con el mismo seed")

    if failures:
        print("SMOKES FALLIDOS:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"Smokes OK: scheduler {sched_rate:.1f}/s, audible {measured:.1f}/s "
          f"(objetivo {target_rate:.1f}), modulacion v3 {mi_v3:.2f} vs legacy "
          f"{mi_legacy:.2f}, determinista.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="solo smokes")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.check:
        return run_checks()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, params, legacy in build_matrix():
        fn = synth_rolling_droplet_legacy if legacy else synth_rolling_droplet
        w = fn(params, SR)
        save_wav(args.out_dir / f"{name}.wav", w, SR)
        print(f"  {name}.wav  peak={np.abs(w).max():.2f}")
    (args.out_dir / "LEEME.md").write_text(LEEME, encoding="utf-8")
    print(f"\n{len(build_matrix())} variantes en {args.out_dir}/ (+ LEEME.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
