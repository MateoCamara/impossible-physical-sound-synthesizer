"""Matriz de escucha del motor v4 "canica continua mojada" + smokes.

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
    synth_rolling_droplet_v3,
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


def build_matrix() -> list[tuple[str, DropletParams, str]]:
    """(nombre, params, motor) con motor en {"legacy", "v3", "v4"}."""
    m: list[tuple[str, DropletParams, str]] = []
    m.append(("00_ab__legacy_aspiradora", _canonical(), "legacy"))
    m.append(("01_ab__v3_goteo", _canonical(), "v3"))
    m.append(("02_ab__v4_canica", _canonical(), "v4"))
    for name, v in (("10_vel__lenta_6hz", 6.0), ("11_vel__media_14hz", 14.0),
                    ("12_vel__rapida_26hz", 26.0)):
        m.append((name, _canonical(roll_velocity_hz=v), "v4"))
    m.append(("20_agua__mojada_visc00", _canonical(viscosity=0.0), "v4"))
    m.append(("21_agua__seca_visc07", _canonical(viscosity=0.7), "v4"))
    for name, preset in (("30_mat__water", "water"), ("31_mat__mercury", "mercury"),
                         ("32_mat__lava", "lava")):
        m.append((name, get_preset(preset, duration_s=DUR_S, seed=SEED), "v4"))
    for name, w in (("40_wobble__off_000", 0.0), ("41_wobble__def_022", 0.22),
                    ("42_wobble__alto_045", 0.45)):
        m.append((name, _canonical(rev_wobble_depth=w), "v4"))
    for name, k in (("50_asper__3", 3), ("51_asper__6", 6), ("52_asper__10", 10)):
        m.append((name, _canonical(asperities_per_rev=k), "v4"))
    for name, f in (("60_fusion__manual_00", 0.0), ("61_fusion__auto", None),
                    ("62_fusion__manual_10", 1.0)):
        m.append((name, _canonical(fusion=f), "v4"))
    for name, a in (("70_acc__off_00", 0.0), ("71_acc__def_035", 0.35),
                    ("72_acc__x2_07", 0.7)):
        m.append((name, _canonical(accent_gain=a), "v4"))
    for name, c in (("80_core__off_00", 0.0), ("81_core__def_08", 0.8),
                    ("82_core__alto_12", 1.2)):
        m.append((name, _canonical(continuous_core_mix=c), "v4"))
    for name, fl in (("90_floor__005", 0.05), ("91_floor__def_018", 0.18),
                     ("92_floor__035", 0.35)):
        m.append((name, _canonical(profile_floor=fl), "v4"))
    return m


LEEME = """# Escucha: motor v4 "canica continua mojada" (rama rolling-v3)

Pregunta central: **¿02 suena a canica CONTINUA rodando — ni al goteo
"ti ti ti" de 01, ni a la aspiradora de 00?**

El motor v4 es un contacto continuo (la canica nunca deja el suelo): ruido
de contacto modulado por el perfil de rugosidad de cada vuelta, con el tono
acuoso Minnaert excitado en continuo y los blips discretos solo como
acentos en los baches grandes.

## Que escuchar

- **00 / 01 / 02 (el A/B/C clave)**: aspiradora vs goteo vs canica.
- **10-12 (velocidad)**: ¿se lee como rodar mas rapido?
- **20-21 (agua)**: mojada = viva con pops; viscosa = mate.
- **30-32 (materiales)**: ¿mercury nerviosa, lava lenta y grave?
- **40-42 (wobble)**: ¿la canica respira por vuelta? ¿0.45 exagera?
- **50-52 (asperezas/vuelta)**: ¿cambia el dibujo ritmico?
- **60-62 (fusion)**: 0 = contactos casi separados, 1 = todo fundido.
  ¿Donde esta el punto bueno? (61 es el auto por velocidad.)
- **70-72 (acentos)**: ¿sin acentos (70) pierde el detalle? ¿0.7 vuelve
  al ti-ti-ti?
- **80-82 (nucleo)**: 80 es ~la v3 pura; ¿82 satura de textura?
- **90-92 (suelo del perfil)**: 92 es el borde de la aspiradora — ¿se nota?

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


def _smooth_env(w: np.ndarray, sr: int) -> np.ndarray:
    """Envolvente |hilbert| suavizada 5 ms, decimada a ~400 Hz."""
    from scipy import signal as sg
    env = np.abs(sg.hilbert(w.astype(np.float64)))
    win = max(8, int(0.005 * sr))
    env = np.convolve(env, np.ones(win) / win, mode="same")
    hop = max(1, sr // 400)
    return env[::hop]


def _rev_periodicity_peak(w: np.ndarray, sr: int, t_rev: float) -> float:
    """Pico de autocorrelacion de la envolvente en lags [0.7, 1.3]*T_rev."""
    env = _smooth_env(w, sr)
    fs_env = 400.0
    x = env - env.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = max(2, int(0.7 * t_rev * fs_env))
    hi = min(len(ac) - 1, int(1.3 * t_rev * fs_env))
    if hi <= lo:
        return 0.0
    return float(ac[lo:hi].max())


def run_checks() -> int:
    failures = []
    p = _canonical()
    w = synth_rolling_droplet(p, SR)

    # 1. No-NaN/Inf, pico y determinismo
    if not np.isfinite(w).all():
        failures.append("NaN/Inf en la canonica")
    if np.abs(w).max() > 0.9501:
        failures.append(f"pico {np.abs(w).max():.3f} > 0.95")
    if not np.array_equal(w, synth_rolling_droplet(_canonical(), SR)):
        failures.append("no determinista con el mismo seed")

    # 2. Scheduler exacto y densidad de ACENTOS
    from impossible_mix.physics.droplet import (
        _ROLL_ACCENT_THR, _derive_params, _roll_schedule)
    p_sched = _derive_params(_canonical())
    sched = _roll_schedule(p_sched, SR, int(DUR_S * SR), np.random.default_rng(p_sched.seed))
    rate = p.roll_velocity_hz * p.contact_density_mul
    K = 4 + int(round(4 * p.path_roughness))
    t_rev = K / rate
    sched_rate = len(sched.starts) / DUR_S
    if abs(sched_rate - rate) > 0.1 * rate:
        failures.append(f"scheduler {sched_rate:.1f}/s != objetivo {rate:.1f}/s")
    mask = sched.pattern_amp[sched.asp_idx] > _ROLL_ACCENT_THR
    mask |= sched.asp_idx == int(np.argmax(sched.pattern_amp))
    accent_rate = float(mask.sum()) / DUR_S
    if not (0.15 * rate <= accent_rate <= 0.7 * rate):
        failures.append(
            f"tasa de acentos {accent_rate:.1f}/s fuera de [0.15, 0.7]x{rate:.0f}")
    onsets = _detect_onsets(w, SR, accent_rate)
    measured = len(onsets) / DUR_S
    if not (0.5 * accent_rate <= measured <= 1.6 * accent_rate):
        failures.append(
            f"acentos audibles {measured:.1f}/s fuera de [0.5,1.6]x{accent_rate:.1f} "
            f"(enterrados en el nucleo o duplicados)")

    # 3. Anti-aspiradora v4 (tres condiciones obligatorias)
    w_legacy = synth_rolling_droplet_legacy(_canonical(), SR)
    mi_v4 = _mod_index(w, SR)
    mi_legacy = _mod_index(w_legacy, SR)
    if not (mi_v4 > 0.20 and mi_v4 > 1.15 * mi_legacy):
        failures.append(
            f"modulacion v4 {mi_v4:.2f} insuficiente (legacy {mi_legacy:.2f}; "
            f"se exige >0.20 y >1.15x)")
    pk_v4 = _rev_periodicity_peak(w, SR, t_rev)
    pk_legacy = _rev_periodicity_peak(w_legacy, SR, t_rev)
    if not (pk_v4 >= 0.12 and pk_v4 >= 2.0 * pk_legacy):
        failures.append(
            f"sin periodicidad de vuelta audible (pico env-AC {pk_v4:.2f}, "
            f"legacy {pk_legacy:.2f}; se exige >=0.12 y >=2x legacy)")
    env = _smooth_env(w, SR)
    valley_ratio = float(np.percentile(env, 10) / (np.percentile(env, 50) + 1e-12))
    if valley_ratio > 0.55:
        failures.append(
            f"valles poco profundos p10/p50={valley_ratio:.2f} > 0.55 (zona aspiradora)")

    # 4. Periodicidad del patron en el schedule (igual que v3)
    if len(sched.amps) > 30:
        amps = sched.amps - sched.amps.mean()
        ac = np.correlate(amps, amps, mode="full")[len(amps) - 1:]
        ac = ac / (ac[0] + 1e-12)
        if ac[K] < 0.25:
            failures.append(
                f"sin periodicidad de vuelta en el schedule (ac[K={K}]={ac[K]:.2f})")

    # 5. Monotonia de los knobs v4 (renders cortos)
    mi_f0 = _mod_index(synth_rolling_droplet(_canonical(fusion=0.0), SR), SR)
    mi_f1 = _mod_index(synth_rolling_droplet(_canonical(fusion=1.0), SR), SR)
    if not mi_f0 > mi_f1:
        failures.append(f"fusion no monotona (mi(f=0)={mi_f0:.2f} <= mi(f=1)={mi_f1:.2f})")
    # accent_gain: mas acentos => mas modulacion (el conteo de onsets no
    # discrimina porque el propio nucleo es "bacheado" al mismo ritmo).
    mi_acc_hi = _mod_index(synth_rolling_droplet(_canonical(accent_gain=0.7), SR), SR)
    mi_acc_off = _mod_index(synth_rolling_droplet(_canonical(accent_gain=0.0), SR), SR)
    if not mi_acc_hi > mi_acc_off:
        failures.append(
            f"accent_gain no monotono (mod 0.7={mi_acc_hi:.2f} <= 0.0={mi_acc_off:.2f})")
    env_c0 = _smooth_env(synth_rolling_droplet(_canonical(continuous_core_mix=0.0), SR), SR)
    vr_c0 = float(np.percentile(env_c0, 10) / (np.percentile(env_c0, 50) + 1e-12))
    if not vr_c0 < valley_ratio:
        failures.append(
            f"core no monotono en valles (p10/p50 core=0 {vr_c0:.2f} >= core=0.8 {valley_ratio:.2f})")

    if failures:
        print("SMOKES FALLIDOS:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"Smokes OK: scheduler {sched_rate:.1f}/s, acentos {accent_rate:.1f}/s "
          f"(audibles {measured:.1f}/s), mod {mi_v4:.2f} (legacy {mi_legacy:.2f}), "
          f"periodicidad vuelta {pk_v4:.2f} (legacy {pk_legacy:.2f}), "
          f"valles p10/p50 {valley_ratio:.2f}, determinista.")
    return 0


_ENGINES = {
    "legacy": synth_rolling_droplet_legacy,
    "v3": synth_rolling_droplet_v3,
    "v4": synth_rolling_droplet,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="solo smokes")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.check:
        return run_checks()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, params, engine in build_matrix():
        w = _ENGINES[engine](params, SR)
        save_wav(args.out_dir / f"{name}.wav", w, SR)
        print(f"  {name}.wav  peak={np.abs(w).max():.2f}")
    (args.out_dir / "LEEME.md").write_text(LEEME, encoding="utf-8")
    print(f"\n{len(build_matrix())} variantes en {args.out_dir}/ (+ LEEME.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
