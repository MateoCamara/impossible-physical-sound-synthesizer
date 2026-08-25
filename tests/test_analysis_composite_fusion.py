"""Tests de la metrica compuesta de fusion (composite_fusion, analysis.py).

No hay tests/ previo en el repo ni pytest instalado en .venv (esta en
pyproject.toml como dependencia opcional 'dev', pero no instalada aqui).
Se escribe en estilo pytest (funciones test_*, asserts planos) para que
corra con pytest si algun dia se instala, y ademas trae un runner __main__
que ejecuta los mismos test_* a mano e imprime PASS/FAIL, igual que los
--check de scripts/34 y scripts/35 (asserts + print + exit code).

Uso:
    PYTHONPATH=. .venv/bin/python tests/test_analysis_composite_fusion.py
    PYTHONPATH=. .venv/bin/pytest tests/test_analysis_composite_fusion.py  (si hay pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impossible_mix.physics.analysis import (  # noqa: E402
    _band, _bri, _dop, _mci, _sso, composite_fusion,
)

SR = 44_100


def _t(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(dur_s * sr)) / sr


def _norm(w: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.abs(w).max()
    return w * (peak / m) if m > 1e-12 else w


def _pulse_train(times: np.ndarray, sr: int, n: int, freq: float,
                 dur_ms: float = 15.0) -> np.ndarray:
    """Tren de golpes tonales cortos (envolvente Hann) en 'times' (s)."""
    w = np.zeros(n)
    dur_s = dur_ms / 1000.0
    for tt in times:
        i0 = int(tt * sr)
        i1 = min(n, i0 + int(dur_s * sr))
        if i1 <= i0:
            continue
        seg_t = np.arange(i1 - i0) / sr
        env = np.hanning(2 * len(seg_t))[: len(seg_t)]
        w[i0:i1] += np.sin(2 * np.pi * freq * seg_t) * env
    return w


# --- Test 1 (brief): MCI alto con bandas en fase, claramente menor con ---
# --- modulaciones independientes.                                      ---

def test_mci_high_when_bands_in_phase_low_when_independent() -> None:
    dur = 2.0
    n = int(dur * SR)
    t = _t(dur)
    rng = np.random.default_rng(0)

    # Caso "en fase": ruido de banda ancha con UNA sola modulacion de
    # amplitud a 6 Hz compartida por todo el espectro -> todas las bandas
    # de band_envelopes deberian respirar juntas.
    carrier = rng.standard_normal(n)
    mod_common = 0.5 * (1.0 + np.sin(2 * np.pi * 6.0 * t))
    blend_inphase = _norm(carrier * mod_common)

    # Caso "independiente": dos regiones espectrales, cada una con su
    # propia modulacion (distinta frecuencia y fase) sin relacion entre si.
    carrier_lo = _band(rng.standard_normal(n), SR, (150.0, 800.0))
    carrier_hi = _band(rng.standard_normal(n), SR, (2000.0, 6000.0))
    mod_lo = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    mod_hi = 0.5 * (1.0 + np.sin(2 * np.pi * 13.0 * t + 1.7))
    blend_indep = _norm(carrier_lo * mod_lo + carrier_hi * mod_hi)

    mci_inphase = _mci(blend_inphase, SR, 12)
    mci_indep = _mci(blend_indep, SR, 12)

    assert mci_inphase > 0.8, f"MCI en fase deberia ser alto, dio {mci_inphase}"
    assert mci_indep < mci_inphase - 0.3, (
        f"MCI independiente ({mci_indep}) deberia ser claramente menor "
        f"que en fase ({mci_inphase})")


# --- Test 2 (brief): SSO bajo con padres en bandas disjuntas, alto con ---
# --- padres en el mismo rango.                                        ---

def test_sso_low_when_disjoint_high_when_same_range() -> None:
    dur = 2.0
    n = int(dur * SR)
    rng = np.random.default_rng(1)

    parent_a = _band(rng.standard_normal(n), SR, (200.0, 400.0))
    parent_b_disjoint = _band(rng.standard_normal(n), SR, (4000.0, 5500.0))
    parent_b_same = _band(rng.standard_normal(n), SR, (180.0, 420.0))

    sso_disjoint = _sso(parent_a, parent_b_disjoint, SR, 12)
    sso_same = _sso(parent_a, parent_b_same, SR, 12)

    assert sso_disjoint < 0.1, f"SSO disjunto deberia ser bajo, dio {sso_disjoint}"
    assert sso_same > 0.8, f"SSO mismo registro deberia ser alto, dio {sso_same}"


# --- Test extra (no pedido explicitamente, pero el informe exige medir  ---
# --- los 4 componentes): DOP alto con onsets huerfanos, bajo si         ---
# --- coinciden.                                                        ---

def test_dop_low_when_onsets_coincide_high_when_orphaned() -> None:
    dur = 2.0
    n = int(dur * SR)
    times = np.arange(0.1, 1.9, 0.2)

    low = _pulse_train(times, SR, n, 300.0)
    high_sync = _pulse_train(times, SR, n, 3000.0)
    high_orphan = _pulse_train(times + 0.09, SR, n, 3000.0)  # 90 ms >> tol 15 ms

    dop_sync = _dop(_norm(low + high_sync), SR)
    dop_orphan = _dop(_norm(low + high_orphan), SR)

    assert dop_sync < 0.1, f"DOP sincrono deberia ser bajo, dio {dop_sync}"
    assert dop_orphan > 0.8, f"DOP con onsets huerfanos deberia ser alto, dio {dop_orphan}"


# --- Test extra: BRI ~0 cuando el blend sigue a A, alto cuando sigue a ---
# --- B_original sin alinear.                                          ---

def test_bri_zero_when_blend_follows_a_positive_when_follows_b() -> None:
    dur = 2.0
    n = int(dur * SR)
    t = _t(dur)
    rng = np.random.default_rng(3)

    env_a = 0.5 * (1.0 + np.sin(2 * np.pi * 0.5 * t - np.pi / 2))
    parent_a = rng.standard_normal(n) * env_a

    env_b = np.zeros(n)
    for c in (0.3, 1.0, 1.6):
        w = int(0.05 * SR)
        i0 = int(c * SR)
        i1 = min(n, i0 + w)
        env_b[i0:i1] += np.hanning(w)[: i1 - i0]
    parent_b = rng.standard_normal(n) * env_b

    bri_isA = _bri(parent_a.copy(), parent_a, parent_b, SR)
    bri_isB = _bri(parent_b.copy(), parent_a, parent_b, SR)

    assert bri_isA == 0.0, f"BRI cuando el blend ES A deberia ser 0, dio {bri_isA}"
    assert bri_isB > 0.5, f"BRI cuando el blend ES B deberia ser alto, dio {bri_isB}"


# --- Test 3 (brief): composite_fusion es determinista. ---

def test_composite_fusion_is_deterministic() -> None:
    dur = 1.5
    n = int(dur * SR)
    t = _t(dur)
    rng = np.random.default_rng(7)

    parent_a = _norm(_band(rng.standard_normal(n), SR, (150.0, 1200.0)))
    parent_b_original = _norm(_band(rng.standard_normal(n), SR, (2000.0, 5000.0)))
    parent_b_aligned = _norm(_band(rng.standard_normal(n), SR, (150.0, 1200.0)))
    mod = 0.5 * (1.0 + np.sin(2 * np.pi * 5.0 * t))
    blend = _norm(0.6 * parent_a * mod + 0.4 * parent_b_aligned * mod)

    # Copias frescas por cada llamada: si composite_fusion mutase algun
    # array de entrada, la segunda llamada veria datos distintos.
    args1 = (blend.copy(), parent_a.copy(), parent_b_aligned.copy(),
             parent_b_original.copy(), SR)
    args2 = (blend.copy(), parent_a.copy(), parent_b_aligned.copy(),
             parent_b_original.copy(), SR)

    r1 = composite_fusion(*args1)
    r2 = composite_fusion(*args2)

    assert r1 == r2, f"composite_fusion no determinista: {r1} != {r2}"

    # Y que no mute las entradas (unica forma realista de perder
    # determinismo en llamadas encadenadas sobre los mismos padres).
    assert np.array_equal(args1[0], blend)
    assert np.array_equal(args1[1], parent_a)
    assert np.array_equal(args1[2], parent_b_aligned)
    assert np.array_equal(args1[3], parent_b_original)

    print(f"  {r1}")


ALL_TESTS = [
    test_mci_high_when_bands_in_phase_low_when_independent,
    test_sso_low_when_disjoint_high_when_same_range,
    test_dop_low_when_onsets_coincide_high_when_orphaned,
    test_bri_zero_when_blend_follows_a_positive_when_follows_b,
    test_composite_fusion_is_deterministic,
]


def main() -> int:
    failures = []
    for fn in ALL_TESTS:
        try:
            fn()
        except AssertionError as e:
            failures.append(f"{fn.__name__}: {e}")
            print(f"FAIL {fn.__name__}: {e}")
        else:
            print(f"PASS {fn.__name__}")
    if failures:
        print(f"\n{len(failures)}/{len(ALL_TESTS)} tests fallidos.")
        return 1
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} tests OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
