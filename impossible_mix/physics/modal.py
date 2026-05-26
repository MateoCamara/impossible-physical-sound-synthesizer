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
    # --- Nuevos materiales (catalogo ampliado) ---
    "rubber": MaterialModalProfile("rubber", n_modes=4, fundamental_hz=160,
                                    spacing=2.2, damping_ms=30,
                                    spectrum_shape="tilted_low", inharmonicity=0.4),
    "bone":   MaterialModalProfile("bone",   n_modes=5, fundamental_hz=520,
                                    spacing=1.9, damping_ms=120,
                                    spectrum_shape="tilted_low", inharmonicity=0.25),
    "ice":    MaterialModalProfile("ice",    n_modes=7, fundamental_hz=1500,
                                    spacing=1.92, damping_ms=900,
                                    spectrum_shape="flat", inharmonicity=0.10),
    "chitin": MaterialModalProfile("chitin", n_modes=6, fundamental_hz=950,
                                    spacing=2.1, damping_ms=180,
                                    spectrum_shape="tilted_low", inharmonicity=0.30),
    # Impossible material: very inharmonic + bright tilt
    "plasma": MaterialModalProfile("plasma", n_modes=12, fundamental_hz=400,
                                    spacing=2.1, damping_ms=400,
                                    spectrum_shape="tilted_high", inharmonicity=0.95),
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


@dataclass
class EnvelopeParams:
    """Envolvente ADSR explicita para el excitador.

    Si se pasa a synth_modal_impact, sobrescribe la forma hardcoded del
    parametro `excitation_shape` y permite control fino del ataque y
    sostenido. Util para campanas con golpe largo, scrapes con ataque
    progresivo, etc.

    attack_ms: tiempo en que la amplitud sube de 0 a 1 (rampa coseno).
    hold_ms:   tiempo durante el cual el exciter mantiene amplitud sustain.
    release_ms: tiempo en que la amplitud cae de sustain a 0 (exponencial).
    sustain_db: nivel del hold en dBFS (negativo). 0 = mismo nivel que el
                peak post-attack; -6 = mitad de amplitud.
    """
    attack_ms: float = 1.0
    hold_ms: float = 0.0
    release_ms: float = 5.0
    sustain_db: float = -6.0


def _adsr_envelope(n: int, sr: int, p: EnvelopeParams) -> np.ndarray:
    """Construye envolvente ADSR de longitud n."""
    a_n = max(1, int(p.attack_ms / 1000.0 * sr))
    h_n = max(0, int(p.hold_ms / 1000.0 * sr))
    r_n = max(1, int(p.release_ms / 1000.0 * sr))
    sustain_gain = float(10 ** (p.sustain_db / 20))
    env = np.zeros(n, dtype=np.float32)
    # Attack: raised cosine de 0 a 1
    attack_seg = 0.5 * (1 - np.cos(np.pi * np.arange(a_n) / a_n)).astype(np.float32)
    env[: min(n, a_n)] = attack_seg[: min(n, a_n)]
    # Hold a nivel sustain
    if a_n < n and h_n > 0:
        end_h = min(n, a_n + h_n)
        env[a_n:end_h] = sustain_gain
    # Release exponencial de sustain a 0
    rel_start = min(n, a_n + h_n)
    if rel_start < n:
        rel_n = min(r_n, n - rel_start)
        env[rel_start:rel_start + rel_n] = sustain_gain * np.exp(-np.linspace(0, 5, rel_n))
    return env


# Excitation shapes: el "exciter" se forma con esta funcion en lugar del
# pulso corto + ruido. Cambia drasticamente el ataque sin tocar el banco modal.
def _make_exciter(n: int, sr: int, shape: str, strength: float, sharpness: float,
                   rng: np.random.Generator,
                   envelope_params: EnvelopeParams | None = None) -> np.ndarray:
    """Genera el excitador de longitud n con la forma especificada.

    Si envelope_params no es None, ignora el `shape` hardcoded y usa una
    envolvente ADSR explicita aplicada a ruido blanco.
    """
    # Camino ADSR explicito (sobrescribe shape)
    if envelope_params is not None:
        total_ms = envelope_params.attack_ms + envelope_params.hold_ms + envelope_params.release_ms
        m_n = max(8, int(total_ms / 1000.0 * sr))
        env = _adsr_envelope(m_n, sr, envelope_params)
        noise = rng.standard_normal(m_n).astype(np.float32) * 0.7 + 0.3
        exc_block = (env * noise * strength).astype(np.float32)
        exc = np.zeros(n, dtype=np.float32)
        end = min(n, m_n)
        exc[:end] = exc_block[:end]
        return exc
    if shape == "felt":
        # Mallet acolchado: ataque suave (raised cosine), banda media
        m_n = max(8, int(0.005 / max(sharpness, 0.1) * sr))
        env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(m_n) / m_n))
        signal_raw = rng.standard_normal(m_n).astype(np.float32) * 0.4 + 0.6
        exc_block = (env * signal_raw * strength).astype(np.float32)
    elif shape == "wood":
        # Mallet duro: ataque corto, banda amplia
        m_n = max(2, int(0.0008 / max(sharpness, 0.1) * sr))
        signal_raw = rng.standard_normal(m_n).astype(np.float32) * 0.3 + 0.7
        env = np.exp(-np.linspace(0, 4, m_n))
        exc_block = (env * signal_raw * strength).astype(np.float32)
    elif shape == "steel":
        # Steel pick: instantaneo y brillante (peak con cola muy corta)
        m_n = max(2, int(0.0003 / max(sharpness, 0.1) * sr))
        signal_raw = rng.standard_normal(m_n).astype(np.float32) * 0.2 + 0.8
        env = np.exp(-np.linspace(0, 6, m_n))
        exc_block = (env * signal_raw * strength).astype(np.float32)
    elif shape == "brush":
        # Brush/scrape: excitacion sostenida con noise modulado
        m_n = max(50, int(0.04 * sr))
        signal_raw = rng.standard_normal(m_n).astype(np.float32) * 0.6
        # Modulacion AM lenta para simular pasada de cerdas
        am = 0.5 + 0.5 * np.sin(2 * np.pi * np.linspace(0, 8, m_n))
        exc_block = (signal_raw * am * strength * 0.6).astype(np.float32)
    elif shape == "impulse":
        # Impulso unitario puro (Dirac)
        m_n = 2
        exc_block = np.array([strength, 0], dtype=np.float32)
    else:
        # Default: similar al original (pulso + noise)
        m_n = max(2, int(0.0005 * sr / max(sharpness, 0.1)))
        exc_block = (strength * (0.7 + 0.3 * rng.standard_normal(m_n))).astype(np.float32)
    exc = np.zeros(n, dtype=np.float32)
    end = min(n, m_n)
    exc[:end] = exc_block[:end]
    return exc


EXCITATION_SHAPES = ("default", "felt", "wood", "steel", "brush", "impulse")


def synth_modal_impact(
    profile: MaterialModalProfile,
    sr: int,
    duration_s: float = 5.0,
    impact_time_s: float = 0.05,
    impact_strength: float = 1.0,
    sharpness: float = 1.0,
    coupling: float = 0.0,
    excitation_shape: str = "default",
    velocity: float = 1.0,
    damping_anisotropy: float = 0.5,
    t60_per_mode: dict[int, float] | None = None,
    envelope_params: "EnvelopeParams | None" = None,
) -> np.ndarray:
    """Genera un golpe modal mejorado.

    sharpness > 1 acentua transitorio (impulso mas corto/duro).
    coupling 0..1: cross-modulation entre modos consecutivos.
    excitation_shape: 'default'|'felt'|'wood'|'steel'|'brush'|'impulse' —
        controla la forma del exciter sin tocar las resonancias.
    velocity: 0.5..2.0 — afecta amplitud del exciter, brillo y un toque
        de pitch shift (real: el material vibra mas tenso al ser golpeado fuerte).
    damping_anisotropy: 0..1 — cuanto mas, los modos altos decaen mas rapido
        que la fundamental (real en metales y placas). 0 = todos los modos
        decaen igual.
    t60_per_mode: dict opcional {mode_index: t60_seconds} para override
        individual del decay de cada modo. Modos no listados usan el calculo
        anisotropico estandar. Permite efectos como "solo la fundamental
        suena, las parciales mueren al instante".
    envelope_params: EnvelopeParams opcional con ataque/hold/release/sustain
        en ms. Si None, usa el shape hardcoded ('felt', 'wood', etc).
    """
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(profile.seed)

    # Construir excitador con la forma pedida (o con envelope ADSR custom)
    eff_strength = impact_strength * float(np.clip(velocity, 0.3, 2.0))
    exc_local = _make_exciter(n, sr, excitation_shape, eff_strength, sharpness, rng,
                              envelope_params=envelope_params)
    start = max(0, int(impact_time_s * sr))
    exc = np.zeros(n, dtype=np.float32)
    end_idx = min(n, start + len(exc_local))
    exc[start:end_idx] = exc_local[: end_idx - start]

    # Pitch-velocity coupling: las freqs suben ligeramente con velocity
    velocity_pitch_factor = 1.0 + 0.04 * (velocity - 1.0)  # ~4% por unidad de velocity
    freqs = modal_frequencies(profile) * velocity_pitch_factor
    gains = _gain_curve(profile.n_modes, profile.spectrum_shape)
    gains = gains / (gains.sum() + 1e-9)

    # Renderizar cada modo por separado (luego aplicar coupling si pedido)
    mode_responses: list[np.ndarray] = []
    for fh, g in zip(freqs, gains):
        if fh <= 0 or fh >= sr / 2:
            mode_responses.append(np.zeros(n, dtype=np.float32))
            continue
        rel = fh / profile.fundamental_hz
        # Damping: o bien override per-mode, o anisotropico estandar
        mode_idx = mode_responses.__len__()  # indice del modo actual
        if t60_per_mode is not None and mode_idx in t60_per_mode:
            t60 = float(t60_per_mode[mode_idx])
        else:
            # Damping anisotropico: modos altos decaen mas rapido.
            # exponente base 0.3 -> con anisotropy=1 sube a 0.9 (efecto fuerte)
            aniso_exp = 0.3 + 0.6 * damping_anisotropy
            t60 = profile.damping_ms / 1000.0 / max(rel ** aniso_exp, 1.0)
        b, a = _modal_resonator(sr, fh, t60)
        y = signal.lfilter(b, a, exc).astype(np.float32)
        mode_responses.append(g * y)

    if coupling > 0.01 and len(mode_responses) > 1:
        # Cross-modulation entre modos consecutivos: ring modulation suave
        # m_k_coupled = m_k * (1 + coupling * m_{k+1})
        coupled = []
        for k, m in enumerate(mode_responses):
            if k < len(mode_responses) - 1:
                # Normalizar modulator a -1..1 antes de modular
                modulator = mode_responses[k + 1]
                peak_m = float(np.max(np.abs(modulator)) + 1e-9)
                modulator_n = modulator / peak_m if peak_m > 0 else modulator
                coupled.append(m * (1 + coupling * modulator_n))
            else:
                coupled.append(m)
        for c in coupled:
            out += c
    else:
        for m in mode_responses:
            out += m

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
