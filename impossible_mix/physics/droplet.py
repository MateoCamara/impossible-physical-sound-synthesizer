"""Sintesis fisica de 'rolling droplet': una gota liquida rodando sobre
una superficie. Modelo paramétrico mejorado con (1) ataque natural en
multiples bandas, (2) bubble pop al final, (3) bouncing/elastic rebound,
(4) acoplamiento real con superficie material-aware, (5) brillo y
sharpness escalado con velocidad de contacto, (6) variabilidad inter-evento.

Cada contacto droplet-superficie genera un 'drip event' compuesto de:
  - Impacto (varios componentes en cascada: click corto agudo + ringing
    capilar de la lamina liquida + chirp de bubble formation)
  - Pequeno bounce/rebound (impacto secundario debil)
  - Cola modal de la superficie con perfil propio del material
  - Decay envelope multibanda

Referencias:
  - Van den Doel, K. (2005) 'Physically-based models for liquid sounds'
  - Drumm, I. (2010) 'Synthesis of bubble sounds'
  - Minnaert formula: f_M = 3.26 / r (r en metros)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


# ====================================================================
# Surface profiles: cada superficie tiene una firma modal distinta
# ====================================================================
@dataclass
class SurfaceProfile:
    """Caracterizacion modal de la superficie sobre la que cae/rueda la gota."""
    name: str
    modes_hz: tuple[float, ...]         # frecuencias modales (banco de resonadores)
    mode_gains: tuple[float, ...]       # gain relativo de cada modo (suma debe ~1)
    t60_ms: float                       # decay del modal en ms
    inharmonicity: float = 0.05         # jitter relativo de freqs
    click_color_hz: tuple[float, float] = (3000, 9000)  # bandpass del click inicial


# Catalogo de superficies. Mas se pueden anadir facilmente.
SURFACE_PROFILES: dict[str, SurfaceProfile] = {
    "fabric":   SurfaceProfile("fabric",   modes_hz=(180, 320),
                               mode_gains=(0.7, 0.3), t60_ms=8,
                               click_color_hz=(500, 2500)),
    "wood":     SurfaceProfile("wood",     modes_hz=(280, 720, 1450, 2400),
                               mode_gains=(0.4, 0.3, 0.2, 0.1), t60_ms=80,
                               click_color_hz=(800, 5000)),
    "ceramic":  SurfaceProfile("ceramic",  modes_hz=(1100, 2400, 4800, 7200),
                               mode_gains=(0.35, 0.3, 0.2, 0.15), t60_ms=350,
                               click_color_hz=(2000, 9000), inharmonicity=0.03),
    "glass":    SurfaceProfile("glass",    modes_hz=(1800, 4200, 7100, 9800),
                               mode_gains=(0.3, 0.3, 0.25, 0.15), t60_ms=600,
                               click_color_hz=(3000, 10000), inharmonicity=0.02),
    "metal":    SurfaceProfile("metal",    modes_hz=(900, 2200, 5100, 8800),
                               mode_gains=(0.35, 0.3, 0.2, 0.15), t60_ms=900,
                               click_color_hz=(3500, 11000), inharmonicity=0.06),
    "stone":    SurfaceProfile("stone",    modes_hz=(380, 880, 1800),
                               mode_gains=(0.5, 0.3, 0.2), t60_ms=60,
                               click_color_hz=(1000, 5000), inharmonicity=0.5),
    "water":    SurfaceProfile("water",    modes_hz=(420, 900),
                               mode_gains=(0.7, 0.3), t60_ms=25,
                               click_color_hz=(400, 2500), inharmonicity=0.2),
    # --- Nuevos perfiles (catalogo ampliado) ---
    "rubber":   SurfaceProfile("rubber",   modes_hz=(110, 250),
                               mode_gains=(0.7, 0.3), t60_ms=12,
                               click_color_hz=(200, 1500), inharmonicity=0.15),
    "leather":  SurfaceProfile("leather",  modes_hz=(240, 480, 900),
                               mode_gains=(0.5, 0.3, 0.2), t60_ms=25,
                               click_color_hz=(400, 2200), inharmonicity=0.20),
    "mud":      SurfaceProfile("mud",      modes_hz=(150, 320),
                               mode_gains=(0.6, 0.4), t60_ms=20,
                               click_color_hz=(200, 1500), inharmonicity=0.55),
    "ice":      SurfaceProfile("ice",      modes_hz=(2000, 4400, 7800),
                               mode_gains=(0.4, 0.35, 0.25), t60_ms=400,
                               click_color_hz=(3000, 10000), inharmonicity=0.04),
    "plastic":  SurfaceProfile("plastic",  modes_hz=(520, 1100, 2400),
                               mode_gains=(0.45, 0.35, 0.20), t60_ms=60,
                               click_color_hz=(1500, 7000), inharmonicity=0.10),
    "cork":     SurfaceProfile("cork",     modes_hz=(380, 780),
                               mode_gains=(0.6, 0.4), t60_ms=35,
                               click_color_hz=(800, 3500), inharmonicity=0.45),
}


# Per-material voicing presets: multiplicadores aplicados encima de los mix
# levels del usuario para que cada superficie tenga firma sonica clara.
# Estos son hyper-parametros perceptuales tuneados de oido.
MATERIAL_VOICING: dict[str, dict[str, float]] = {
    "fabric":  {"body_mul": 0.30, "cavity_mul": 0.50, "ring_mul": 0.40, "discrete_mul": 1.0},
    "wood":    {"body_mul": 0.60, "cavity_mul": 0.80, "ring_mul": 0.80, "discrete_mul": 1.2},
    "ceramic": {"body_mul": 1.00, "cavity_mul": 0.70, "ring_mul": 1.10, "discrete_mul": 1.0},
    "glass":   {"body_mul": 1.20, "cavity_mul": 0.50, "ring_mul": 1.30, "discrete_mul": 1.0},
    "metal":   {"body_mul": 1.10, "cavity_mul": 0.40, "ring_mul": 1.50, "discrete_mul": 0.9},
    "stone":   {"body_mul": 0.40, "cavity_mul": 1.20, "ring_mul": 0.70, "discrete_mul": 1.1},
    "water":   {"body_mul": 0.50, "cavity_mul": 0.80, "ring_mul": 0.50, "discrete_mul": 0.8},
    "rubber":  {"body_mul": 0.25, "cavity_mul": 0.40, "ring_mul": 0.30, "discrete_mul": 0.7},
    "leather": {"body_mul": 0.45, "cavity_mul": 0.70, "ring_mul": 0.50, "discrete_mul": 0.9},
    "mud":     {"body_mul": 0.20, "cavity_mul": 0.60, "ring_mul": 0.20, "discrete_mul": 0.5},
    "ice":     {"body_mul": 1.00, "cavity_mul": 0.60, "ring_mul": 1.40, "discrete_mul": 1.0},
    "plastic": {"body_mul": 0.70, "cavity_mul": 0.60, "ring_mul": 0.90, "discrete_mul": 1.0},
    "cork":    {"body_mul": 0.35, "cavity_mul": 0.60, "ring_mul": 0.40, "discrete_mul": 0.8},
}
_DEFAULT_VOICING = {"body_mul": 1.0, "cavity_mul": 1.0, "ring_mul": 1.0, "discrete_mul": 1.0}


def surface_from_hardness(hardness: float) -> SurfaceProfile:
    """Mapea hardness 0..1 a un perfil de superficie razonable, ordenado
    de mas blando (rubber) a mas duro (metal). Posiciones intermedias se
    interpolan al perfil mas cercano.
    """
    h = float(np.clip(hardness, 0, 1))
    order = ["rubber", "fabric", "cork", "leather", "mud", "wood",
             "ceramic", "plastic", "stone", "ice", "glass", "metal"]
    idx = min(int(h * (len(order) - 1) + 0.5), len(order) - 1)
    return SURFACE_PROFILES[order[idx]]


# ====================================================================
# Droplet params
# ====================================================================
@dataclass
class DropletParams:
    """Parametros fisicos de la rolling droplet."""
    droplet_radius_mm: float = 2.0      # tamano de la gota (mm). 1-5 mm tipico.
    viscosity: float = 0.0              # 0=agua pura, 1=miel
    surface_hardness: float = 0.5       # 0=tela, 1=metal — selecciona surface profile
    surface_profile: str | None = None  # override directo del perfil de superficie
    roll_velocity_hz: float = 14.0      # impactos por segundo
    path_roughness: float = 0.35        # 0=metronomo perfecto; 1=totalmente azar
    duration_s: float = 5.0
    seed: int = 0

    # NUEVOS parametros fisicos (4-5):
    bounce_amount: float = 0.35         # cantidad de rebote post-impacto (0=sin, 1=fuerte)
    bounce_chain_length: int = 1        # numero de rebotes encadenados (1=solo el primero)
    bounce_decay: float = 0.55          # factor de decay por rebote (energia*0.55 cada vez)
    velocity_to_brightness: float = 1.0 # cuanto sube el brillo con velocidad (multiplicador)
    inter_event_variability: float = 0.6  # 0=todos iguales; 1=muy variables
    capillary_ringing: float = 0.5      # contribucion del ringing capilar 0..1
    drying_factor: float = 0.0          # 0=sin drying, 1=la escena se seca al final del clip
    continuous_layer_mix: float = 0.75  # nivel de la capa de ruido coloreado por surface (antes 0.4)
    body_resonance_strength: float = 0.6  # intensidad del eco modal del surface en la capa continua
    # Capas continuas "rodillo de agua" (rework perceptual):
    body_resonance_mix: float = 0.7     # Minnaert sostenido (impedance-coupled)
    cavity_mix: float = 0.4             # resonancia Helmholtz, atada al modo mas bajo
    slosh_mix: float = 0.1              # heuristico legacy (bajado, ahora rayleigh_mix es protagonista)
    shimmer_depth: float = 0.2          # AM lenta tipo capillary ripple
    surface_ring_mix: float = 0.7       # voz del material (modos continuos sostenidos)
    discrete_mix: float = 0.5           # ticks discretos con click_color material-flavored
    # Nueva fisica:
    microbubble_mix: float = 0.5        # cloud de N microburbujas (multi-Minnaert)
    rayleigh_mix: float = 0.35          # modos exactos de oscilacion de forma (ω² = n(n-1)(n+2)σ/ρr³)
    stickslip_mix: float = 0.4          # micro-impactos por asperezas en rolling
    contact_angle_deg: float = 110      # wetting: 30=hidrofilico, 110=neutral, 170=lotus
    surface_tension_n_m: float = 0.072  # N/m (agua=0.072, aceites~0.03)

    # Parametros derivables (auto-completados):
    bubble_freq_start_hz: float | None = None
    bubble_freq_end_hz: float | None = None
    chirp_duration_ms: float | None = None
    decay_ms: float | None = None


def _bubble_freq_from_radius(radius_mm: float) -> float:
    """Minnaert frequency (Hz) para una burbuja de aire en agua."""
    r_m = radius_mm * 1e-3
    return 3.26 / max(r_m, 1e-4)


def _bubble_t60_ms(radius_mm: float) -> float:
    """Tiempo de decaimiento (-60 dB, ms) de una burbuja de Minnaert.

    Damping fisicamente derivado del radio segun van den Doel (2005): la
    burbuja radia como una sinusoide amortiguada ``sin(2*pi*f*t)*exp(-d*t)``
    cuyo coeficiente de amortiguamiento crece con la frecuencia,

        d = 0.043 f + 0.0014 f^{3/2}     (f en kHz, d en ms^-1),

    de modo que las burbujas pequenas (agudas) decaen mas rapido que las
    grandes. Sustituye al antiguo decay heuristico, acoplando explicitamente
    el amortiguamiento al radio en vez de fijarlo de forma independiente.
    """
    f_khz = _bubble_freq_from_radius(radius_mm) / 1000.0
    d = 0.043 * f_khz + 0.0014 * f_khz ** 1.5  # ms^-1
    return 6.907755 / max(d, 1e-6)             # ln(1000) / d


def _derive_params(p: DropletParams) -> DropletParams:
    f_minnaert = _bubble_freq_from_radius(p.droplet_radius_mm)
    if p.bubble_freq_end_hz is None:
        p.bubble_freq_end_hz = f_minnaert * 1.6
    if p.bubble_freq_start_hz is None:
        p.bubble_freq_start_hz = f_minnaert * 0.45
    if p.chirp_duration_ms is None:
        p.chirp_duration_ms = (15 + 8 * p.droplet_radius_mm) * (1 + 1.5 * p.viscosity)
    if p.decay_ms is None:
        # Decay acoplado al radio (van den Doel) y acortado por la viscosidad.
        p.decay_ms = _bubble_t60_ms(p.droplet_radius_mm) * (1 - 0.6 * p.viscosity)
    return p


def _get_surface(p: DropletParams) -> SurfaceProfile:
    if p.surface_profile is not None and p.surface_profile in SURFACE_PROFILES:
        return SURFACE_PROFILES[p.surface_profile]
    return surface_from_hardness(p.surface_hardness)


# ====================================================================
# Drip event: ahora con velocity, variability y bounce
# ====================================================================
def synth_drip_event(p: DropletParams, sr: int,
                      velocity_factor: float = 1.0,
                      seed_override: int | None = None) -> np.ndarray:
    """Sintetiza UN solo evento drip.

    velocity_factor 1.0 = velocidad nominal. >1.0 = contacto mas duro
    (mas brillante, attack mas afilado). <1.0 = contacto suave.

    seed_override permite hacer cada evento del rolling distinto.
    """
    p = _derive_params(p)
    surf = _get_surface(p)
    seed = seed_override if seed_override is not None else p.seed
    rng = np.random.default_rng(seed)

    # ---- 1. Bubble chirp principal (Minnaert resonance) ----
    chirp_dur_ms = p.chirp_duration_ms
    # Velocidad influye en duracion: contactos rapidos => chirps mas cortos
    chirp_dur_ms = chirp_dur_ms * (0.7 + 0.6 / max(velocity_factor, 0.3))
    chirp_n = max(8, int(chirp_dur_ms / 1000.0 * sr))

    decay_ms_v = p.decay_ms * (1.0 / max(velocity_factor ** 0.3, 0.5))
    decay_n = int(decay_ms_v / 1000.0 * sr)
    total_n = chirp_n + decay_n + int(0.08 * sr)  # extra cola por surface modal

    t_chirp = np.linspace(0, chirp_dur_ms / 1000.0, chirp_n, endpoint=False)
    f_start = float(p.bubble_freq_start_hz)
    f_end = float(p.bubble_freq_end_hz)
    # Variabilidad: cada evento un poco distinto en freq objetivo
    var = p.inter_event_variability
    f_end *= (1.0 + var * rng.uniform(-0.15, 0.15))
    if f_end > f_start:
        f_t = f_start * (f_end / f_start) ** (t_chirp / (chirp_dur_ms / 1000.0))
    else:
        f_t = np.linspace(f_start, f_end, chirp_n)
    phase = 2 * np.pi * np.cumsum(f_t) / sr
    chirp = np.sin(phase).astype(np.float32)

    # Envolvente: ataque rapido + decay exponencial
    env_attack_n = max(2, int(0.0008 * sr))
    env = np.ones(chirp_n, dtype=np.float32)
    env[:env_attack_n] = np.linspace(0, 1, env_attack_n) ** 0.7  # attack curvo
    decay_factor = np.exp(-np.linspace(0, 4 * (1 - p.viscosity * 0.5), chirp_n - env_attack_n))
    env[env_attack_n:] = decay_factor
    chirp = chirp * env

    out = np.zeros(total_n, dtype=np.float32)
    out[: chirp_n] = chirp * 0.7

    # ---- 2. Click inicial (bandpass color del material, prolongado) ----
    # Ampliado a 3 ms con decay exponencial para que el "color" del
    # material (click_color_hz) se perciba claramente.
    click_n = max(4, int(0.003 * sr))
    click = rng.standard_normal(click_n).astype(np.float32)
    cl_lo, cl_hi = surf.click_color_hz
    cl_hi_safe = min(cl_hi, sr / 2 - 200)
    if cl_lo < cl_hi_safe:
        sos = signal.butter(3, [cl_lo, cl_hi_safe], btype="band", fs=sr, output="sos")
        click = signal.sosfiltfilt(sos, click).astype(np.float32)
    # Envolvente exponencial: el click decae rapidamente
    click_env = np.exp(-3 * np.arange(click_n) / click_n).astype(np.float32)
    click *= click_env
    click *= 0.5 * velocity_factor * p.velocity_to_brightness
    out[: click_n] += click

    # ---- 3. Capillary ringing (oscilacion corta de la lamina liquida) ----
    if p.capillary_ringing > 0.05:
        # Frecuencia capilar ~ 2-4x Minnaert, muy corta
        f_cap = float(p.bubble_freq_end_hz) * 2.2
        cap_n = max(8, int(0.012 * sr))
        t_cap = np.arange(cap_n) / sr
        cap_env = np.exp(-np.linspace(0, 6, cap_n))
        cap = np.sin(2 * np.pi * f_cap * t_cap) * cap_env
        cap *= 0.25 * p.capillary_ringing * velocity_factor
        end = min(total_n, click_n + cap_n)
        out[click_n:end] += cap[: end - click_n].astype(np.float32)

    # ---- 4. Surface modal tail (banco de resonadores del material) ----
    # Excitar con un impulso corto en t=0 cada modo del surface
    impulse_n = max(2, int(0.001 * sr))
    impulse = np.zeros(total_n, dtype=np.float32)
    impulse[:impulse_n] = rng.standard_normal(impulse_n).astype(np.float32) * 0.3
    surf_response = np.zeros(total_n, dtype=np.float32)
    t60_s = surf.t60_ms / 1000.0
    # Hardness escala el gain del surface tail
    surf_overall_gain = 0.18 + 0.35 * p.surface_hardness * velocity_factor
    for fc, mg in zip(surf.modes_hz, surf.mode_gains):
        if fc <= 0 or fc >= sr / 2 - 100:
            continue
        # Inharmonicidad: jitter por modo
        fc_j = fc * (1 + surf.inharmonicity * rng.uniform(-1, 1))
        r = float(np.exp(-6.91 / max(t60_s * sr, 1e-3)))
        theta = 2 * np.pi * fc_j / sr
        a_coef = np.array([1.0, -2 * r * np.cos(theta), r * r])
        b_coef = np.array([1.0, 0.0, -1.0])
        mode_resp = signal.lfilter(b_coef, a_coef, impulse).astype(np.float32)
        surf_response += mg * mode_resp
    out += surf_overall_gain * surf_response

    # ---- 5. Bouncing CHAIN: rebotes encadenados con decay geometrico ----
    if p.bounce_amount > 0.05 and p.bounce_chain_length >= 1:
        # Tiempo del primer rebote depende del radio y viscosidad (gota grande tarda mas)
        bounce_delay_ms_base = (20 + 12 * p.droplet_radius_mm) * (1 + 0.5 * p.viscosity)
        cumulative_offset_n = 0
        current_amp = p.bounce_amount * 0.4
        for k in range(p.bounce_chain_length):
            # Cada rebote sucesivo es mas corto en tiempo (decay) y mas debil
            shrink = (p.bounce_decay ** 0.5) ** k  # delay shrinks slower than amp
            delay_n = int(bounce_delay_ms_base / 1000.0 * sr * shrink)
            cumulative_offset_n += delay_n
            if cumulative_offset_n >= total_n - 50:
                break
            # Mini chirp con decay creciente
            mini_n = max(20, chirp_n // (2 + k))
            mini_t = np.linspace(0, chirp_dur_ms / (2 + k) / 1000.0, mini_n, endpoint=False)
            if mini_t[-1] > 0:
                mini_f = f_start * (f_end / f_start) ** (mini_t / mini_t[-1])
            else:
                mini_f = np.full(mini_n, f_start)
            mini_phase = 2 * np.pi * np.cumsum(mini_f) / sr
            mini_chirp = np.sin(mini_phase).astype(np.float32)
            mini_env = np.exp(-np.linspace(0, 5 + k, mini_n))
            mini = (mini_chirp * mini_env * current_amp).astype(np.float32)
            end = min(total_n, cumulative_offset_n + mini_n)
            out[cumulative_offset_n:end] += mini[: end - cumulative_offset_n]
            current_amp *= p.bounce_decay

    # ---- 6. Bubble pop final (opcional: pop seco al colapsar la burbuja) ----
    # Solo si baja viscosidad: con viscosidad alta no hay pop
    if p.viscosity < 0.4 and p.capillary_ringing > 0.1:
        pop_idx = chirp_n + int(decay_n * 0.4)
        if pop_idx < total_n - 50:
            pop = rng.standard_normal(40).astype(np.float32) * 0.15 * (1 - p.viscosity) * velocity_factor
            sos = signal.butter(2, [800, min(4000, sr / 2 - 200)], btype="band", fs=sr, output="sos")
            pop = signal.sosfiltfilt(sos, pop).astype(np.float32)
            # decay rapidisimo
            pop *= np.exp(-np.linspace(0, 8, len(pop)))
            end = min(total_n, pop_idx + len(pop))
            out[pop_idx:end] += pop[: end - pop_idx]

    # Normalizar el evento
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0:
        out = out * (0.7 / peak)
    return out.astype(np.float32)


# ====================================================================
# Capa continua de rumor (anti 'tacatacataca')
# ====================================================================
def _continuous_roll_layer(
    p: DropletParams,
    surface: SurfaceProfile,
    sr: int,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Capa continua de rumor de rodadura.

    Convierte la percepción de 'tacatacataca' (eventos drip discretos
    aislados) a 'rrrrrr con ticks' (rodadura fluida con golpes superpuestos).
    Replica el pipeline de friction.synth_scrape:
      1) Envolvente de velocidad lenta (LFO modulado por path_roughness +
         densidad de contacto derivada de roll_velocity_hz).
      2) Ruido bandpass coloreado por surface.click_color_hz.
      3) Banco modal del surface excitado por el ruido modulado, para que
         las dos capas suenen del mismo material.

    El llamador típicamente modula la salida por el RMS local del tren
    de drips para que el rumor crezca durante clusters de contactos.
    """
    if n_total < 100:
        return np.zeros(n_total, dtype=np.float32)

    # 1) Envolvente: nivel base ∝ roll_velocity_hz + LFO ~5 Hz
    base_level = 0.4 + 0.6 * min(p.roll_velocity_hz / 25.0, 1.0)
    lfo_raw = rng.standard_normal(n_total).astype(np.float32)
    lfo_cutoff = max(0.5, 2.0 + 6.0 * p.path_roughness)
    sos_lpf = signal.butter(2, lfo_cutoff, btype="low", fs=sr, output="sos")
    lfo = signal.sosfiltfilt(sos_lpf, lfo_raw).astype(np.float32)
    lfo = (lfo - lfo.mean()) / (lfo.std() + 1e-9)
    env = np.clip(base_level + 0.3 * p.path_roughness * lfo, 0.0, 1.3)

    # 2) Ruido bandpass coloreado por el surface
    noise = rng.standard_normal(n_total).astype(np.float32)
    cl_lo, cl_hi = surface.click_color_hz
    cl_hi_safe = min(cl_hi, sr / 2 - 200)
    sos_band = signal.butter(4, [cl_lo, cl_hi_safe], btype="band",
                              fs=sr, output="sos")
    bright_noise = signal.sosfiltfilt(sos_band, noise).astype(np.float32)
    excited = bright_noise * env

    # 3) Banco modal del surface excitado por el ruido modulado
    body = np.zeros(n_total, dtype=np.float32)
    for fc, mg in zip(surface.modes_hz, surface.mode_gains):
        if fc <= 0 or fc >= sr / 2 - 100:
            continue
        fc_j = fc * (1 + surface.inharmonicity * rng.uniform(-1, 1))
        q = 4.0 + 6.0 * (1 - surface.inharmonicity)
        bw = fc_j / max(q, 0.5)
        f_lo = max(50, fc_j - bw / 2)
        f_hi = min(sr / 2 - 100, fc_j + bw / 2)
        if f_lo >= f_hi:
            continue
        try:
            sos_mode = signal.butter(2, [f_lo, f_hi], btype="band",
                                      fs=sr, output="sos")
            body += mg * signal.sosfiltfilt(sos_mode, excited).astype(np.float32)
        except ValueError:
            continue

    # Viscosidad atenúa (una gota de miel rodando es casi muda)
    viscosity_atten = 1.0 - 0.5 * p.viscosity
    layer = (0.45 * excited + p.body_resonance_strength * body) * viscosity_atten
    return layer.astype(np.float32)


# ====================================================================
# Capas continuas: el caracter "rodillo de agua"
# ====================================================================
# Una gota rodando NO es una serie de impactos: es un contacto continuo
# de una masa liquida deformable contra la superficie. Cuatro capas:
#   A) Body resonance: Minnaert sostenido durante toda la rodadura,
#      AM modulada por velocidad, FM wobble por path_roughness.
#   B) Cavity:         resonancia Helmholtz del bolsillo de aire atrapado
#                      entre la gota y la superficie (200-600 Hz).
#   C) Sloshing:       subarmonico de deformacion (0.4-0.6 x f_M).
#   D) Shimmer:        AM lenta aplicada a A+B+C, calidad "viva" de agua.

def _slow_lfo(n: int, sr: int, cutoff_hz: float,
              rng: np.random.Generator) -> np.ndarray:
    """LFO lento normalizado: ruido pasado por LPF a cutoff_hz, media 0, std 1."""
    raw = rng.standard_normal(n).astype(np.float32)
    sos = signal.butter(2, max(0.5, cutoff_hz), btype="low", fs=sr, output="sos")
    lfo = signal.sosfiltfilt(sos, raw).astype(np.float32)
    return ((lfo - lfo.mean()) / (lfo.std() + 1e-9)).astype(np.float32)


def _body_resonance_layer(p: DropletParams, surface: SurfaceProfile,
                          sr: int, n: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Layer A: Minnaert tone sostenido. La AMPLITUD se acopla a la
    impedancia acustica de la superficie via t60_ms: superficies duras
    (largo t60) reflejan energia Minnaert hacia la gota -> body fuerte;
    blandas (t60 corto) absorben -> body debil."""
    fM = _bubble_freq_from_radius(p.droplet_radius_mm)
    am_hz = 2 + 6 * min(p.roll_velocity_hz / 25, 1)
    am = _slow_lfo(n, sr, am_hz, rng)
    fm = _slow_lfo(n, sr, am_hz * 0.6, rng)
    fm_depth = 0.03 + 0.05 * p.path_roughness
    # Impedancia (t60) + Wetting (contact angle)
    t60_factor = min(1.0, surface.t60_ms / 600)
    impedance_gain = 0.3 + 0.7 * t60_factor
    contact_area = _contact_area_factor(p.contact_angle_deg)
    wetting_gain = 0.4 + 0.6 * (1 - contact_area)
    visc_atten = (1.0 - 0.6 * p.viscosity) * impedance_gain * wetting_gain
    f_inst = fM * (1 + fm_depth * fm)
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    am_env = np.clip(0.75 + 0.25 * am, 0, None)
    return (0.7 * visc_atten * am_env * np.sin(phase)).astype(np.float32)


def _cavity_resonance_layer(p: DropletParams, surface: SurfaceProfile,
                             sr: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Layer B: resonancia Helmholtz ligada al modo mas bajo de la superficie,
    con frecuencia y amplitud moduladas por wetting (contact angle)."""
    min_mode = min(surface.modes_hz) if surface.modes_hz else 1000
    contact_area = _contact_area_factor(p.contact_angle_deg)
    cavity_vol_factor = 0.3 + 0.7 * (1 - contact_area)
    base_f_cavity = max(120.0, min(1200.0, min_mode * 0.6))
    # Helmholtz f propto 1/sqrt(V): mayor volumen -> menor frecuencia
    f_cavity = base_f_cavity / np.sqrt(cavity_vol_factor)
    am_hz = 2 + 6 * min(p.roll_velocity_hz / 25, 1)
    am = _slow_lfo(n, sr, am_hz, rng)
    visc_atten = 1.0 - 0.6 * p.viscosity
    cavity_amp_gain = 0.5 + 0.5 * (1 - contact_area)
    phase = 2 * np.pi * f_cavity * np.arange(n) / sr + np.pi / 2
    am_env = np.clip(0.75 + 0.25 * am, 0, None)
    return (0.5 * visc_atten * cavity_amp_gain * am_env * np.sin(phase)).astype(np.float32)


def _surface_ringing_layer(p: DropletParams, surface: SurfaceProfile,
                            sr: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Layer E (NEW): continuous surface ringing — la "voz del material".
    Por cada modo de la superficie, una sinusoide sostenida en fc[k] con
    amplitud proporcional a gains[k] x sustain (derivado de t60_ms).
    Excitada continuamente por el mismo LFO que body/cavity.
    Esto es lo que hace que el metal "cante" y la madera suene mate."""
    am_hz = 2 + 6 * min(p.roll_velocity_hz / 25, 1)
    am = _slow_lfo(n, sr, am_hz, rng)
    fm_depth = 0.005 + 0.015 * p.path_roughness
    visc_atten = 1.0 - 0.6 * p.viscosity
    t60_factor = min(1.0, surface.t60_ms / 600)
    sustain_gain = 0.2 + 0.8 * t60_factor
    base_amp = 0.55 * visc_atten * sustain_gain
    out = np.zeros(n, dtype=np.float32)
    n_modes = len(surface.modes_hz)
    for m, (fc, g) in enumerate(zip(surface.modes_hz, surface.mode_gains)):
        if fc <= 0 or fc >= sr / 2 - 100:
            continue
        # Per-mode slow FM (cada modo con su propio jitter)
        mode_fm = _slow_lfo(n, sr, am_hz * (0.7 + 0.4 * m / max(n_modes, 1)), rng)
        f_inst = fc * (1 + fm_depth * mode_fm)
        phase_offset = 2 * np.pi * rng.random()
        phase = 2 * np.pi * np.cumsum(f_inst) / sr + phase_offset
        am_env = np.clip(0.7 + 0.3 * am, 0, None)
        out += (base_amp * g * am_env * np.sin(phase)).astype(np.float32)
    return out


def _sloshing_layer(p: DropletParams, sr: int, n: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Layer C: subarmonico de deformacion (0.4-0.6 x f_M) con tremolo."""
    fM = _bubble_freq_from_radius(p.droplet_radius_mm)
    f_slosh = fM * (0.4 + 0.2 * rng.random())
    trem_hz = 3 + 3 * rng.random()
    base_amp = 0.3 * (0.3 + 0.7 * p.path_roughness)
    t = np.arange(n) / sr
    trem = 0.5 + 0.5 * np.sin(2 * np.pi * trem_hz * t)
    phase = 2 * np.pi * f_slosh * t
    return (base_amp * trem * np.sin(phase)).astype(np.float32)


def _apply_shimmer(data: np.ndarray, sr: int, depth: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Layer D: AM lenta aleatoria (5-15 Hz) sobre la mezcla continua."""
    if depth < 0.01:
        return data
    n = len(data)
    sh_hz = 5 + 10 * rng.random()
    lfo = _slow_lfo(n, sr, sh_hz, rng)
    mult = np.clip(1 + depth * lfo, 0, None)
    return (data * mult).astype(np.float32)


# ====================================================================
# Wetting / contact angle physics
# ====================================================================
def _contact_area_factor(angle_deg: float) -> float:
    """Contact area A propto (1 + cos(theta))/2. A=1 spread, A=0 lotus."""
    return (1 + np.cos(np.deg2rad(angle_deg))) / 2


def _micro_bubble_cloud_layer(p: DropletParams, sr: int, n: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Layer F (NEW): N=8-20 microburbujas con f_M = 3.26/r_i.
    Distribucion log-normal de radios centrada en r_padre/3. Mas
    microburbujas con contact angle alto (lotus entrega mas air entrainment)."""
    contact = _contact_area_factor(p.contact_angle_deg)
    base_n = 6 + int(14 * contact * (1 - p.viscosity))
    n_bubbles = max(2, base_n)
    am_hz = 2 + 6 * min(p.roll_velocity_hz / 25, 1)
    am = _slow_lfo(n, sr, am_hz, rng)
    visc_atten = 1.0 - 0.6 * p.viscosity
    base_amp = 0.5 * visc_atten / np.sqrt(n_bubbles)
    out = np.zeros(n, dtype=np.float32)
    for _ in range(n_bubbles):
        r_bubble = (p.droplet_radius_mm / 3) * np.exp(0.6 * rng.standard_normal())
        r_safe = float(np.clip(r_bubble, 0.05, p.droplet_radius_mm * 0.8))
        f_bubble = 3.26 / (r_safe * 1e-3)
        if f_bubble >= sr / 2 - 200:
            continue
        decay_n = max(200, int(0.05 * sr * (r_safe / p.droplet_radius_mm) ** 2))
        onset = int(rng.uniform(0, n * 0.3))
        phase_off = 2 * np.pi * rng.random()
        amp_weight = (r_safe / (p.droplet_radius_mm / 3)) ** 1.5
        b_amp = base_amp * amp_weight * rng.uniform(0.5, 1.0)
        bubble_fm = _slow_lfo(n, sr, am_hz * rng.uniform(0.5, 1.5), rng)
        idxs = np.arange(onset, n)
        t = (idxs - onset).astype(np.float32)
        decay = np.exp(-t / decay_n)
        re_trigger = np.clip(0.7 + 0.3 * am[onset:], 0, None)
        f_inst = f_bubble * (1 + 0.02 * bubble_fm[onset:])
        phase = phase_off + 2 * np.pi * np.cumsum(f_inst) / sr
        out[onset:] += (b_amp * decay * re_trigger * np.sin(phase)).astype(np.float32)
    return out


def _rayleigh_modes_layer(p: DropletParams, sr: int, n: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Layer C (NEW): modos exactos de oscilacion de forma de una esfera liquida libre.

        omega_n^2 = n(n-1)(n+2) * sigma / (rho * r^3)

    Para gota de radio 2 mm en agua (sigma=0.072 N/m, rho=1000 kg/m^3):
        f_2 ~ 43 Hz, f_3 ~ 83 Hz, f_4 ~ 128 Hz, f_5 ~ 179 Hz, f_6 ~ 234 Hz.

    Referencia: Rayleigh (1879) 'On the capillary phenomena of jets'."""
    r = max(0.1, p.droplet_radius_mm) * 1e-3
    rho = 1000.0
    sigma = p.surface_tension_n_m
    out = np.zeros(n, dtype=np.float32)
    base_amp = 0.4 * (0.4 + 0.6 * p.path_roughness)
    t_idx = np.arange(n)
    for mode in range(2, 7):
        omega2 = mode * (mode - 1) * (mode + 2) * sigma / (rho * r ** 3)
        if omega2 <= 0:
            continue
        f_mode = np.sqrt(omega2) / (2 * np.pi)
        if f_mode >= sr / 2 - 50:
            continue
        mode_amp = base_amp / mode
        trigger_hz = 2 + 8 * p.path_roughness
        triggers = _slow_lfo(n, sr, trigger_hz, rng)
        phase_off = 2 * np.pi * rng.random()
        phase = phase_off + 2 * np.pi * f_mode * t_idx / sr
        trig = 0.5 + 0.5 * triggers
        out += (mode_amp * trig * np.sin(phase)).astype(np.float32)
    return out


def _rolling_stickslip_layer(p: DropletParams, surface: SurfaceProfile,
                              sr: int, n: int,
                              rng: np.random.Generator) -> np.ndarray:
    """Layer G (NEW): micro-impactos por asperezas en rolling.
    Las asperezas de la superficie son brevemente capturadas por fuerzas
    capilares y liberadas; cada release es un click filtrado por click_color.

    Referencia: Persson (2001) 'Theory of rubber friction and contact mechanics'."""
    rate_hz = (5 + 60 * p.path_roughness) * (0.5 + p.roll_velocity_hz / 20)
    n_events = int(rate_hz * n / sr)
    if n_events < 1:
        return np.zeros(n, dtype=np.float32)
    contact = _contact_area_factor(p.contact_angle_deg)
    stick_prob = 0.3 + 0.6 * contact
    cl_lo, cl_hi = surface.click_color_hz
    cl_hi_safe = min(cl_hi, sr / 2 - 200)
    if cl_lo >= cl_hi_safe:
        return np.zeros(n, dtype=np.float32)
    sos = signal.butter(2, [cl_lo, cl_hi_safe], btype="band", fs=sr, output="sos")
    visc_atten = 1.0 - 0.7 * p.viscosity
    out = np.zeros(n, dtype=np.float32)
    for _ in range(n_events):
        if rng.random() > stick_prob:
            continue
        start = int(rng.uniform(0, n - 100))
        burst_len = max(8, int(rng.uniform(0.0003, 0.0011) * sr))
        amp = rng.uniform(0.15, 0.55) * p.path_roughness * visc_atten
        noise = rng.standard_normal(burst_len).astype(np.float32)
        filtered = signal.sosfilt(sos, noise).astype(np.float32)
        env = np.exp(-4 * np.arange(burst_len) / burst_len).astype(np.float32)
        end = min(n, start + burst_len)
        out[start:end] += amp * env[: end - start] * filtered[: end - start]
    return out


# ====================================================================
# Rolling droplet: continuous-contact layers + discrete ticks texture
# ====================================================================
def synth_rolling_droplet(p: DropletParams, sr: int = 44_100) -> np.ndarray:
    """Rolling droplet con jerarquia perceptual NUEVA:
    los layers continuos (body + cavity + sloshing + shimmer) son los
    protagonistas (60-80% energia); los drip ticks discretos son textura
    sutil (20-40%). Esto da la percepcion de masa liquida rodando,
    no de gotas cayendo rapido.
    """
    p = _derive_params(p)
    n_total = int(p.duration_s * sr)
    out = np.zeros(n_total, dtype=np.float32)
    rng = np.random.default_rng(p.seed)
    period_samples = sr / max(p.roll_velocity_hz, 0.1)
    surface = _get_surface(p)
    voicing = MATERIAL_VOICING.get(surface.name, _DEFAULT_VOICING)

    # === 1) Capas continuas (PROTAGONISTAS) =========================
    if p.body_resonance_mix > 0.01:
        body = _body_resonance_layer(p, surface, sr, n_total, rng)
        out += body * p.body_resonance_mix * voicing["body_mul"]
    if p.cavity_mix > 0.01:
        cavity = _cavity_resonance_layer(p, surface, sr, n_total, rng)
        out += cavity * p.cavity_mix * voicing["cavity_mul"]
    # Layer C (NEW): modos exactos de Rayleigh (sustituye al sloshing heuristico)
    if p.rayleigh_mix > 0.01:
        rayleigh = _rayleigh_modes_layer(p, sr, n_total, rng)
        out += rayleigh * p.rayleigh_mix
    # Legacy slosh (heuristico, default bajo)
    if p.slosh_mix > 0.01:
        slosh = _sloshing_layer(p, sr, n_total, rng)
        out += slosh * p.slosh_mix
    # Layer F (NEW): cloud de microburbujas (multi-Minnaert)
    if p.microbubble_mix > 0.01:
        micro = _micro_bubble_cloud_layer(p, sr, n_total, rng)
        out += micro * p.microbubble_mix
    # Layer E: surface ringing (la voz del material)
    if p.surface_ring_mix > 0.01:
        ring = _surface_ringing_layer(p, surface, sr, n_total, rng)
        out += ring * p.surface_ring_mix * voicing["ring_mul"]
    # Layer G (NEW): rolling stick-slip (micro-impactos por asperezas)
    if p.stickslip_mix > 0.01:
        stick = _rolling_stickslip_layer(p, surface, sr, n_total, rng)
        out += stick * p.stickslip_mix
    # Shimmer aplicado ANTES de añadir noise/ticks (solo afecta al body liquido)
    if p.shimmer_depth > 0.01:
        out = _apply_shimmer(out, sr, p.shimmer_depth, rng)

    # === 2) Capa de ruido coloreado por surface (sin RMS gating) ====
    if p.continuous_layer_mix > 0.01:
        continuous = _continuous_roll_layer(p, surface, sr, n_total, rng)
        velocity_floor = 0.4 + 0.6 * min(p.roll_velocity_hz / 25, 1)
        out = out + continuous * velocity_floor * p.continuous_layer_mix

    # === 3) Drip ticks discretos (TEXTURA con click_color material) ==
    if p.discrete_mix > 0.01:
        # Wetting modula bounce y capillary: lotus rebota mas, hidrofilico
        # menos; mas contacto = mas deformacion capilar.
        contact_area = _contact_area_factor(p.contact_angle_deg)
        wetting_bounce = 0.5 + 1.5 * (1 - contact_area)
        capillary_factor = 0.5 + 0.7 * contact_area
        # Copia temporal del params con bounce/capillary ajustados
        from dataclasses import replace
        p_ticks = replace(
            p,
            bounce_amount=p.bounce_amount * wetting_bounce,
            capillary_ringing=p.capillary_ringing * capillary_factor,
        )
        ticks = np.zeros(n_total, dtype=np.float32)
        n_variants = 1 + int(round(p.inter_event_variability * 5))
        variants = [
            synth_drip_event(p_ticks, sr, velocity_factor=1.0, seed_override=p.seed + 100 + k)
            for k in range(n_variants)
        ]
        t_sample = 0.0
        while t_sample < n_total:
            offset = period_samples * (1 + p.path_roughness * rng.uniform(-0.7, 0.7))
            start = int(t_sample)
            if start >= n_total:
                break
            velocity_factor = float(np.clip(
                rng.normal(loc=1.0, scale=0.25 + 0.4 * p.path_roughness), 0.4, 1.8
            ))
            amp = velocity_factor * rng.uniform(0.55, 1.0)
            if p.inter_event_variability > 0.1 and abs(velocity_factor - 1.0) > 0.25:
                evt = synth_drip_event(p_ticks, sr, velocity_factor=velocity_factor,
                                        seed_override=p.seed + start)
            else:
                evt = variants[rng.integers(0, n_variants)]
            end = min(n_total, start + len(evt))
            ticks[start:end] += amp * evt[: end - start]
            t_sample += offset
        out = out + ticks * p.discrete_mix * voicing["discrete_mul"]

    # === 4) Drying tail =============================================
    if p.drying_factor > 0.05:
        half = n_total // 2
        dry_env = np.ones(n_total, dtype=np.float32)
        ramp = np.linspace(0, 1, n_total - half).astype(np.float32)
        dry_env[half:] = 1.0 - p.drying_factor * (1 - np.exp(-3 * ramp))
        out = out * dry_env

    # === 5) Fade-in/out (avoids filter edge transients) ============
    fade_n = min(int(0.02 * sr), n_total // 8)
    if fade_n > 4:
        fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_n))).astype(np.float32)
        out[:fade_n] *= fade
        out[-fade_n:] *= fade[::-1]

    # === 6) Peak-normalise ==========================================
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
