"""Sintesis modal: banco de resonadores excitados por impulsos.

Un sonido de impacto modal se modela como sum_i a_i * exp(-d_i * t) * sin(2*pi*f_i * t)
donde (f_i, d_i, a_i) son frecuencia/damping/amplitud del modo i. Implementamos
con biquads IIR resonantes que actuan como osciladores amortiguados al recibir
un impulso de Dirac.

Perfiles materiales: cada material define un patron tipico de (f_modos,
damping, espectro de gains). Sweepear "mas X" cambia estos parametros.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal


@dataclass
class MaterialModalProfile:
    """Perfil modal de un material.

    n_modes: cuantos resonadores.
    fundamental_hz: frecuencia base.
    spacing: factor multiplicativo entre modos. >1 modos arrastrados separados,
             <2 mas tonal. 'random' inarmonico (caso roca/grava).
    damping_ms: t60 medio en ms. 1 = muerto; 100+ = resonante.
    spectrum_shape: 'flat', 'tilted_low', 'tilted_high'  - como decae amplitud con f.
    inharmonicity: 0..1 - cuanta desviacion aleatoria de las freq teoricas.
    """
    name: str
    n_modes: int
    fundamental_hz: float
    spacing: float           # 1.5..2.0 tonal; >2 inarmonico
    damping_ms: float        # t60 aproximado por modo
    spectrum_shape: str = "tilted_low"
    inharmonicity: float = 0.0
    seed: int = 0


PROFILES: dict[str, MaterialModalProfile] = {
    "metal":  MaterialModalProfile("metal",  n_modes=8, fundamental_hz=900,
                                   spacing=1.85, damping_ms=800,
                                   spectrum_shape="tilted_low", inharmonicity=0.05),
    "rock":   MaterialModalProfile("rock",   n_modes=6, fundamental_hz=350,
                                   spacing=2.3, damping_ms=80,
                                   spectrum_shape="tilted_low", inharmonicity=0.6),
    "wood":   MaterialModalProfile("wood",   n_modes=5, fundamental_hz=280,
                                   spacing=1.7, damping_ms=200,
                                   spectrum_shape="tilted_low", inharmonicity=0.2),
    "glass":  MaterialModalProfile("glass",  n_modes=10, fundamental_hz=1800,
                                   spacing=1.95, damping_ms=1200,
                                   spectrum_shape="flat", inharmonicity=0.02),
    "earth":  MaterialModalProfile("earth",  n_modes=4, fundamental_hz=120,
                                   spacing=2.0, damping_ms=30,
                                   spectrum_shape="tilted_low", inharmonicity=0.8),
    "fabric": MaterialModalProfile("fabric", n_modes=3, fundamental_hz=200,
                                   spacing=2.5, damping_ms=15,
                                   spectrum_shape="tilted_low", inharmonicity=0.5),
}


def _gain_curve(n: int, shape: str) -> np.ndarray:
    """Distribucion de amplitud entre modos."""
    if shape == "flat":
        return np.ones(n)
    if shape == "tilted_low":
        return 1.0 / (1 + np.arange(n) * 0.4)
    if shape == "tilted_high":
        return 1.0 / (1 + (n - 1 - np.arange(n)) * 0.4)
    return np.ones(n)


def modal_frequencies(profile: MaterialModalProfile) -> np.ndarray:
    """Devuelve frecuencias modales (Hz) segun el perfil."""
    rng = np.random.default_rng(profile.seed)
    n = profile.n_modes
    # Base: f_0 * spacing^k para k=0..n-1
    base = profile.fundamental_hz * (profile.spacing ** np.arange(n))
    # Inarmonicidad: jitter relativo
    jitter = rng.uniform(-1, 1, n) * profile.inharmonicity
    return base * (1 + 0.3 * jitter)


def _modal_resonator(sr: int, freq_hz: float, t60_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Biquad resonador 2-polos clavado en freq_hz con tiempo de decaimiento t60_s."""
    # Polos cerca del ciruculo unidad para alta Q
    if t60_s <= 0:
        t60_s = 0.001
    r = float(np.exp(-6.91 / (t60_s * sr + 1e-6)))   # 6.91 = ln(1000)
    theta = 2 * np.pi * freq_hz / sr
    a1 = -2 * r * np.cos(theta)
    a2 = r * r
    b = np.array([1.0, 0.0, -1.0])  # diferencia, agarra ringing
    a = np.array([1.0, a1, a2])
    return b, a


def synth_modal_impact(
    profile: MaterialModalProfile,
    sr: int,
    duration_s: float = 5.0,
    impact_time_s: float = 0.05,
    impact_strength: float = 1.0,
    sharpness: float = 1.0,
) -> np.ndarray:
    """Genera un golpe modal: un impulso filtrado por banco de resonadores.

    sharpness >1 acentua transitorio (impulso mas corto/duro).
    """
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    # Excitacion: pulso corto + ruido coloreado al ataque
    exc_n = max(2, int(0.0005 * sr / max(sharpness, 0.1)))
    exc = np.zeros(n, dtype=np.float32)
    start = max(0, int(impact_time_s * sr))
    rng = np.random.default_rng(profile.seed)
    # Mezcla: 70% impulso unitario + 30% ruido blanco corto
    end = min(n, start + exc_n)
    exc[start:end] = impact_strength * (0.7 + 0.3 * rng.standard_normal(end - start))

    freqs = modal_frequencies(profile)
    gains = _gain_curve(profile.n_modes, profile.spectrum_shape)
    gains = gains / (gains.sum() + 1e-9)
    for fh, g in zip(freqs, gains):
        if fh <= 0 or fh >= sr / 2:
            continue
        # Damping aleatorio por modo (mas damping en modos altos)
        rel = fh / profile.fundamental_hz
        t60 = profile.damping_ms / 1000.0 / max(rel ** 0.3, 1.0)
        b, a = _modal_resonator(sr, fh, t60)
        y = signal.lfilter(b, a, exc)
        out += g * y.astype(np.float32)

    # Normalizacion suave
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


def synth_modal_roll(
    profile: MaterialModalProfile,
    sr: int,
    duration_s: float = 5.0,
    rate_hz: float = 12.0,
    jitter: float = 0.4,
    strength: float = 0.6,
) -> np.ndarray:
    """Tren cuasiperiodico de impactos modales: simula rodadura.

    rate_hz: impactos por segundo (bola rodando).
    jitter: 0 = regular metronomo; 1 = totalmente aleatorio.
    """
    rng = np.random.default_rng(profile.seed + 7)
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    period = sr / rate_hz
    t = 0.0
    while t < n:
        t_perturb = period * (1 + jitter * rng.uniform(-0.6, 0.6))
        impact_time_s = t / sr
        stroke_strength = strength * rng.uniform(0.5, 1.2)
        sharpness = 0.5 + 1.5 * rng.random()
        hit = synth_modal_impact(profile, sr, duration_s=duration_s,
                                 impact_time_s=impact_time_s,
                                 impact_strength=stroke_strength,
                                 sharpness=sharpness)
        out += hit
        t += t_perturb
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


# ---------- Interpolacion entre perfiles (knob "mas X") ----------

def blend_profiles(
    p_src: MaterialModalProfile,
    p_tgt: MaterialModalProfile,
    amount: float,
    name: str | None = None,
) -> MaterialModalProfile:
    """Interpola los parametros numericos. amount=0 -> src, amount=1 -> tgt,
    amount=2 -> extrapola mas alla del target."""
    a = float(amount)
    return MaterialModalProfile(
        name=name or f"{p_src.name}_to_{p_tgt.name}_{amount:.2f}",
        n_modes=int(round(p_src.n_modes + a * (p_tgt.n_modes - p_src.n_modes))),
        fundamental_hz=p_src.fundamental_hz + a * (p_tgt.fundamental_hz - p_src.fundamental_hz),
        spacing=p_src.spacing + a * (p_tgt.spacing - p_src.spacing),
        damping_ms=p_src.damping_ms + a * (p_tgt.damping_ms - p_src.damping_ms),
        spectrum_shape=p_tgt.spectrum_shape if a > 0.5 else p_src.spectrum_shape,
        inharmonicity=max(0.0, p_src.inharmonicity + a * (p_tgt.inharmonicity - p_src.inharmonicity)),
        seed=p_src.seed,
    )
