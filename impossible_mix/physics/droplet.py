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
    body_resonance_mix: float = 0.7     # Minnaert sostenido durante el contacto (protagonista)
    cavity_mix: float = 0.4             # resonancia Helmholtz cavidad gota-superficie (~200-600 Hz)
    slosh_mix: float = 0.3              # wobble subarmonico de deformacion
    shimmer_depth: float = 0.2          # AM lenta tipo capillary ripple
    discrete_mix: float = 0.3           # ticks discretos (textura, no protagonista) — antes implicit 1.0

    # Parametros derivables (auto-completados):
    bubble_freq_start_hz: float | None = None
    bubble_freq_end_hz: float | None = None
    chirp_duration_ms: float | None = None
    decay_ms: float | None = None


def _bubble_freq_from_radius(radius_mm: float) -> float:
    """Minnaert frequency (Hz) para una burbuja de aire en agua."""
    r_m = radius_mm * 1e-3
    return 3.26 / max(r_m, 1e-4)


def _derive_params(p: DropletParams) -> DropletParams:
    f_minnaert = _bubble_freq_from_radius(p.droplet_radius_mm)
    if p.bubble_freq_end_hz is None:
        p.bubble_freq_end_hz = f_minnaert * 1.6
    if p.bubble_freq_start_hz is None:
        p.bubble_freq_start_hz = f_minnaert * 0.45
    if p.chirp_duration_ms is None:
        p.chirp_duration_ms = (15 + 8 * p.droplet_radius_mm) * (1 + 1.5 * p.viscosity)
    if p.decay_ms is None:
        p.decay_ms = (50 + 30 * p.droplet_radius_mm) * (1 - 0.6 * p.viscosity)
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

    # ---- 2. Click inicial (bandpass color del material) ----
    click_n = max(2, int(0.0015 * sr))
    click = rng.standard_normal(click_n).astype(np.float32)
    cl_lo, cl_hi = surf.click_color_hz
    cl_hi_safe = min(cl_hi, sr / 2 - 200)
    if cl_lo < cl_hi_safe:
        sos = signal.butter(3, [cl_lo, cl_hi_safe], btype="band", fs=sr, output="sos")
        click = signal.sosfiltfilt(sos, click).astype(np.float32)
    # velocity afecta amplitud del click
    click *= 0.6 * velocity_factor * p.velocity_to_brightness
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


def _body_resonance_layer(p: DropletParams, sr: int, n: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Layer A: Minnaert tone sostenido con AM (velocidad) + FM wobble (rugosidad).
    AM mas sutil (0.75±0.25) para que sea sostenido, no pulsado."""
    fM = _bubble_freq_from_radius(p.droplet_radius_mm)
    am_hz = 2 + 6 * min(p.roll_velocity_hz / 25, 1)
    am = _slow_lfo(n, sr, am_hz, rng)
    fm = _slow_lfo(n, sr, am_hz * 0.6, rng)
    fm_depth = 0.03 + 0.05 * p.path_roughness
    visc_atten = 1.0 - 0.6 * p.viscosity
    f_inst = fM * (1 + fm_depth * fm)
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    am_env = np.clip(0.75 + 0.25 * am, 0, None)  # mas sostenido
    return (0.7 * visc_atten * am_env * np.sin(phase)).astype(np.float32)


def _cavity_resonance_layer(p: DropletParams, surface: SurfaceProfile,
                             sr: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Layer B: resonancia Helmholtz del bolsillo de aire (200-600 Hz)."""
    min_mode = min(surface.modes_hz) if surface.modes_hz else 1000
    hardness = min(1.0, min_mode / 1500)
    f_cavity = 200 + 400 * hardness
    am_hz = 2 + 6 * min(p.roll_velocity_hz / 25, 1)
    am = _slow_lfo(n, sr, am_hz, rng)
    visc_atten = 1.0 - 0.6 * p.viscosity
    phase = 2 * np.pi * f_cavity * np.arange(n) / sr + np.pi / 2
    am_env = np.clip(0.75 + 0.25 * am, 0, None)
    return (0.5 * visc_atten * am_env * np.sin(phase)).astype(np.float32)


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

    # === 1) Capas continuas (PROTAGONISTAS) =========================
    if p.body_resonance_mix > 0.01:
        body = _body_resonance_layer(p, sr, n_total, rng)
        out += body * p.body_resonance_mix
    if p.cavity_mix > 0.01:
        cavity = _cavity_resonance_layer(p, surface, sr, n_total, rng)
        out += cavity * p.cavity_mix
    if p.slosh_mix > 0.01:
        slosh = _sloshing_layer(p, sr, n_total, rng)
        out += slosh * p.slosh_mix
    # Shimmer aplicado ANTES de añadir noise/ticks (solo afecta al body liquido)
    if p.shimmer_depth > 0.01:
        out = _apply_shimmer(out, sr, p.shimmer_depth, rng)

    # === 2) Capa de ruido coloreado por surface (sin RMS gating) ====
    if p.continuous_layer_mix > 0.01:
        continuous = _continuous_roll_layer(p, surface, sr, n_total, rng)
        velocity_floor = 0.4 + 0.6 * min(p.roll_velocity_hz / 25, 1)
        out = out + continuous * velocity_floor * p.continuous_layer_mix

    # === 3) Drip ticks discretos (TEXTURA, atenuados) ===============
    if p.discrete_mix > 0.01:
        ticks = np.zeros(n_total, dtype=np.float32)
        n_variants = 1 + int(round(p.inter_event_variability * 5))
        variants = [
            synth_drip_event(p, sr, velocity_factor=1.0, seed_override=p.seed + 100 + k)
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
                evt = synth_drip_event(p, sr, velocity_factor=velocity_factor,
                                        seed_override=p.seed + start)
            else:
                evt = variants[rng.integers(0, n_variants)]
            end = min(n_total, start + len(evt))
            ticks[start:end] += amp * evt[: end - start]
            t_sample += offset
        out = out + ticks * p.discrete_mix

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
