"""Sintesis fisica de 'rolling droplet': una gota liquida rodando sobre
una superficie. Modelo paramétrico con parametros de significado fisico.

Cada contacto droplet-superficie genera un 'drip event' compuesto de:
  - Impulso corto (excitacion)
  - Chirp ascendente (formacion de burbuja, Helmholtz resonance dinamico)
  - Decay envelope exponencial
  - Opcional: pequena cola modal de la superficie

Una gota rodando es un tren cuasi-periodico de drip events con variabilidad
en intervalo (path roughness) y energia (rolling dynamics).

Referencias:
  - Van den Doel, K. (2005) 'Physically-based models for liquid sounds'
  - Drumm, I. (2010) 'Synthesis of bubble sounds'
  - El chirp de la burbuja modela Minnaert resonance + radius growth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class DropletParams:
    """Parametros fisicos de la rolling droplet."""
    droplet_radius_mm: float = 2.0      # tamano de la gota (mm). 1-5 mm tipico.
    viscosity: float = 0.0              # 0=agua pura, 1=miel. Damping del chirp.
    surface_hardness: float = 0.5       # 0=tela, 1=metal. Brillo modal de superficie.
    roll_velocity_hz: float = 14.0      # impactos por segundo
    path_roughness: float = 0.35        # 0=metronomo perfecto; 1=totalmente azar
    duration_s: float = 5.0
    seed: int = 0

    # Parametros derivables (se calculan en __post_init__ si no se dan)
    bubble_freq_start_hz: float | None = None
    bubble_freq_end_hz: float | None = None
    chirp_duration_ms: float | None = None
    decay_ms: float | None = None
    surface_modal_freq_hz: float | None = None


def _bubble_freq_from_radius(radius_mm: float) -> float:
    """Minnaert frequency (Hz) para una burbuja de aire en agua.
    f_M = 3.26 / r  con r en metros => para r=2mm -> ~1630 Hz.
    Aproximacion: para gota cayendo en superficie, el chirp termina cerca
    de 2*Minnaert y empieza ~0.5*Minnaert.
    """
    r_m = radius_mm * 1e-3
    return 3.26 / max(r_m, 1e-4)


def _derive_params(p: DropletParams) -> DropletParams:
    """Llena campos derivables a partir de los fisicos primarios."""
    f_minnaert = _bubble_freq_from_radius(p.droplet_radius_mm)
    if p.bubble_freq_end_hz is None:
        p.bubble_freq_end_hz = f_minnaert * 1.6
    if p.bubble_freq_start_hz is None:
        p.bubble_freq_start_hz = f_minnaert * 0.45
    if p.chirp_duration_ms is None:
        # Gota grande -> chirp mas lento. Viscosidad -> chirp aun mas lento.
        p.chirp_duration_ms = (15 + 8 * p.droplet_radius_mm) * (1 + 1.5 * p.viscosity)
    if p.decay_ms is None:
        # Decay mas largo si gota mas grande, mas corto si superficie blanda
        p.decay_ms = (50 + 30 * p.droplet_radius_mm) * (1 - 0.6 * p.viscosity)
    if p.surface_modal_freq_hz is None:
        # Mas duro -> modal mas alto
        p.surface_modal_freq_hz = 200 + 1500 * p.surface_hardness
    return p


def synth_drip_event(p: DropletParams, sr: int) -> np.ndarray:
    """Sintetiza UN solo evento drip aislado.

    Composicion:
      a) impulso corto + ruido ataque (1ms)
      b) chirp ascendente con Q alta entre bubble_freq_start y _end
      c) envelope exponencial con decay_ms
      d) opcional cola modal de superficie
    """
    p = _derive_params(p)
    chirp_n = max(8, int(p.chirp_duration_ms / 1000.0 * sr))
    total_n = max(chirp_n, int((p.chirp_duration_ms + p.decay_ms) / 1000.0 * sr))
    rng = np.random.default_rng(p.seed)

    t_chirp = np.linspace(0, p.chirp_duration_ms / 1000.0, chirp_n, endpoint=False)
    # Frequency sweep exponencial
    f_start = float(p.bubble_freq_start_hz)
    f_end = float(p.bubble_freq_end_hz)
    if f_end > f_start:
        f_t = f_start * (f_end / f_start) ** (t_chirp / (p.chirp_duration_ms / 1000.0))
    else:
        f_t = np.linspace(f_start, f_end, chirp_n)
    phase = 2 * np.pi * np.cumsum(f_t) / sr
    chirp = np.sin(phase).astype(np.float32)

    # Envolvente del chirp: ataque rapido + decay
    env_attack = max(1, int(0.001 * sr))
    env = np.ones(chirp_n, dtype=np.float32)
    env[:env_attack] = np.linspace(0, 1, env_attack)
    decay_factor = np.exp(-np.linspace(0, 4 * (1 - p.viscosity * 0.5), chirp_n - env_attack))
    env[env_attack:] = decay_factor
    chirp = chirp * env

    # Cola: continua decaying tras el chirp
    tail_n = total_n - chirp_n
    if tail_n > 0:
        tail = chirp[-1] * np.exp(-np.linspace(0, 6, tail_n)) * 0.4
        chirp = np.concatenate([chirp, tail.astype(np.float32)])

    # Impulso de ataque (ruido corto)
    attack_n = max(2, int(0.0008 * sr))
    attack = rng.standard_normal(attack_n).astype(np.float32) * 0.5
    chirp[:attack_n] += attack

    # Cola modal de la superficie (sub-mix)
    if p.surface_hardness > 0.05:
        surf = np.zeros_like(chirp)
        # Resonador biquad agudo
        fc = float(p.surface_modal_freq_hz)
        t60_s = 0.04 * (1 - 0.5 * p.viscosity) * (0.3 + 0.7 * p.surface_hardness)
        r = float(np.exp(-6.91 / max(t60_s * sr, 1e-3)))
        theta = 2 * np.pi * fc / sr
        a = np.array([1.0, -2 * r * np.cos(theta), r * r])
        b = np.array([1.0, 0, -1.0])
        impulse = np.zeros_like(surf)
        impulse[:attack_n] = rng.standard_normal(attack_n).astype(np.float32) * 0.3
        surf = signal.lfilter(b, a, impulse).astype(np.float32)
        chirp = chirp + 0.5 * p.surface_hardness * surf

    # Normalizar el evento al peak
    peak = float(np.max(np.abs(chirp)) + 1e-9)
    if peak > 0:
        chirp = chirp * (0.7 / peak)
    return chirp.astype(np.float32)


def synth_rolling_droplet(p: DropletParams, sr: int = 44_100) -> np.ndarray:
    """Sintesis completa: tren cuasiperiodico de drip events.

    p define la fisica (radio, viscosidad, dureza superficie, velocidad,
    rugosidad). Devuelve mono float32 [-1, 1].
    """
    p = _derive_params(p)
    n_total = int(p.duration_s * sr)
    out = np.zeros(n_total, dtype=np.float32)
    rng = np.random.default_rng(p.seed)
    period_samples = sr / max(p.roll_velocity_hz, 0.1)

    # Crear UNA plantilla de drip y reutilizarla con jitter de amplitud/timing
    # para coste razonable. Si la fisica cambia entre golpes, se podria regenerar.
    base_evt = synth_drip_event(p, sr)
    evt_n = len(base_evt)

    t_sample = 0.0
    while t_sample < n_total:
        # Jitter de timing (path roughness)
        offset = period_samples * (1 + p.path_roughness * rng.uniform(-0.7, 0.7))
        start = int(t_sample)
        if start >= n_total:
            break
        # Jitter de amplitud (energia de la rodadura)
        amp = rng.uniform(0.55, 1.0) * (1 + 0.2 * p.path_roughness)
        end = min(n_total, start + evt_n)
        out[start:end] += amp * base_evt[: end - start]
        t_sample += offset

    # Background bed: un poco de ruido coloreado bajo nivel (humedad ambiente)
    if p.viscosity < 0.5:
        bed = rng.standard_normal(n_total).astype(np.float32) * 0.015
        sos = signal.butter(4, [200, 1200], btype="band", fs=sr, output="sos")
        bed = signal.sosfiltfilt(sos, bed).astype(np.float32)
        # Modular por la envolvente RMS del tren para que ladre con la rodadura
        win = max(1, int(0.04 * sr))
        rms = np.sqrt(np.convolve(out * out, np.ones(win) / win, mode="same"))
        rms_norm = rms / (rms.max() + 1e-9)
        out = out + bed * rms_norm * 0.4

    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
