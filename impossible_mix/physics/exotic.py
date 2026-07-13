"""Fisicas exoticas: rain, fire, thunder, glass_break, ocean_wave.

Cada una compuesta a partir de las primitivas existentes (modal, drip,
granular, friction) + envolventes especificas. Util para demos y para
mostrar alcance del marco mas alla de las 3 combinaciones canonicas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from impossible_mix.physics.droplet import (
    DropletParams,
    SURFACE_PROFILES,
    synth_drip_event,
)
from impossible_mix.physics.granular import GranularParams, synth_granular_flow
from impossible_mix.physics.modal import (
    MaterialModalProfile,
    PROFILES as MODAL_PROFILES,
    blend_profiles,
    modal_frequencies,
    synth_modal_impact,
)


def synth_rain(duration_s: float = 5.0, intensity: float = 0.6,
               drop_size_mm: float = 1.2, wind_strength: float = 0.0,
               gust_rate_hz: float = 0.3, sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Lluvia con viento + rafagas opcionales.

    wind_strength 0..1: modula la densidad de drips con un LFO lento.
    gust_rate_hz: frecuencia media de rafagas (eventos donde density sube).
    """
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    base_rate = 40 + 200 * intensity

    # Construir envolvente de density modulada por viento y rafagas
    # (samples-rate baja por eficiencia)
    sub_n = max(64, int(duration_s * 20))  # 20 Hz de resolucion
    t_sub = np.linspace(0, duration_s, sub_n)
    # LFO lento (~ wind_strength * 0.5 Hz)
    wind_lfo = 1 + wind_strength * 0.6 * np.sin(2 * np.pi * 0.4 * t_sub +
                                                  2 * np.pi * rng.random())
    # Rafagas: eventos gaussianos centrados en momentos aleatorios
    if wind_strength > 0.05:
        n_gusts = max(1, int(gust_rate_hz * duration_s))
        for _ in range(n_gusts):
            center = rng.uniform(0.3, duration_s - 0.3)
            width = rng.uniform(0.4, 1.5)
            amp = wind_strength * rng.uniform(0.5, 1.2)
            wind_lfo += amp * np.exp(-((t_sub - center) / width) ** 2)
    density_env = np.clip(wind_lfo, 0.3, 3.0)

    # Generar drips siguiendo density variable
    t = 0.0
    while t < n:
        # Density local
        sub_idx = int((t / n) * sub_n)
        density = float(density_env[min(sub_idx, sub_n - 1)])
        period = sr / (base_rate * density)
        radius = drop_size_mm * (1 + 0.5 * rng.uniform(-0.5, 0.8))
        radius = max(0.3, radius)
        # Variabilidad de velocity per drop (lluvia con viento tiene drops mas duros)
        vel = float(np.clip(rng.normal(1.0 + 0.3 * wind_strength, 0.25), 0.5, 1.8))
        d = DropletParams(droplet_radius_mm=radius, viscosity=0.0,
                          surface_hardness=0.0, roll_velocity_hz=1, path_roughness=0,
                          duration_s=0.12, seed=seed + int(t))
        evt = synth_drip_event(d, sr, velocity_factor=vel)
        amp = rng.uniform(0.2, 0.7) * (0.4 + intensity) * vel
        start = int(t)
        end = min(n, start + len(evt))
        out[start:end] += amp * evt[: end - start]
        t += period * (1 + 0.7 * rng.uniform(-0.8, 0.8))

    # Background hiss bandpass alto, modulado tambien por wind
    noise = rng.standard_normal(n).astype(np.float32) * 0.05 * intensity
    sos = signal.butter(2, [500, 5000], btype="band", fs=sr, output="sos")
    bed = signal.sosfiltfilt(sos, noise).astype(np.float32)
    if wind_strength > 0.05:
        # Resamplear density_env a sr para modular bed
        bed_mod = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, sub_n), density_env)
        bed = bed * bed_mod.astype(np.float32) * 0.7
        # Add wind whoosh (low band noise modulado por density_env)
        whoosh = rng.standard_normal(n).astype(np.float32) * 0.08 * wind_strength
        sos_w = signal.butter(2, [100, 1200], btype="band", fs=sr, output="sos")
        whoosh = signal.sosfiltfilt(sos_w, whoosh).astype(np.float32)
        whoosh = whoosh * bed_mod.astype(np.float32) * wind_strength * 0.6
        bed = bed + whoosh
    out = out + bed
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_fire(duration_s: float = 5.0, intensity: float = 0.7,
               crackle_density: float = 0.5, sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Fuego mejorado: bed caotico bandpass + crackles que son
    micro-impactos MODALES de madera (no solo ruido), simulando
    fibras de madera explotando.

    Mejoras:
    - Crackles ahora son mini modal events con perfil 'wood' o
      crushed_glass cuando la madera revienta.
    - El bed se modula por la envolvente de los pops (la combustion
      respira con los crackles).
    - Doble AM con frecuencias dispares simula respiracion irregular.
    """
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    # Bed: ruido marrón modulado por dos AM
    bed = rng.standard_normal(n).astype(np.float32) * 0.15
    sos = signal.butter(4, [80, 1500], btype="band", fs=sr, output="sos")
    bed = signal.sosfiltfilt(sos, bed).astype(np.float32)
    t_n = np.arange(n) / sr
    am_freq1 = 1.5 + rng.random() * 2
    am_freq2 = 0.4 + 0.6 * rng.random()
    am = (0.5 + 0.3 * np.sin(2 * np.pi * am_freq1 * t_n)
              + 0.2 * np.sin(2 * np.pi * am_freq2 * t_n + 1.0))
    bed = bed * am.astype(np.float32) * intensity

    # Crackles: micro-impactos modales de madera o cristal cuando seco
    out = bed.copy()
    pop_rate = 4 + 25 * crackle_density
    n_pops = int(duration_s * pop_rate)
    # Mini perfil modal para los pops (mas rapido que synth_modal_impact)
    for _ in range(n_pops):
        idx = int(rng.uniform(0, n - 200))
        # Cada pop: pulso corto + dos resonancias (madera con micro-fractura)
        pop_n = int(rng.uniform(0.005, 0.025) * sr)
        impulse = np.zeros(pop_n, dtype=np.float32)
        impulse[:max(2, pop_n // 20)] = rng.standard_normal(max(2, pop_n // 20)).astype(np.float32) * 0.7
        # Dos modos: alto (fractura aguda) y medio (cuerpo de la fibra)
        fc1 = rng.uniform(2500, 5500)
        fc2 = rng.uniform(700, 1800)
        t60 = rng.uniform(0.003, 0.012)
        r1 = float(np.exp(-6.91 / max(t60 * sr, 1e-3)))
        r2 = float(np.exp(-6.91 / max(t60 * 1.4 * sr, 1e-3)))
        a1 = np.array([1.0, -2 * r1 * np.cos(2 * np.pi * fc1 / sr), r1 * r1])
        a2 = np.array([1.0, -2 * r2 * np.cos(2 * np.pi * fc2 / sr), r2 * r2])
        b = np.array([1.0, 0.0, -1.0])
        y1 = signal.lfilter(b, a1, impulse).astype(np.float32)
        y2 = signal.lfilter(b, a2, impulse).astype(np.float32)
        pop = 0.6 * y1 + 0.4 * y2
        peak_p = float(np.max(np.abs(pop)) + 1e-9)
        if peak_p > 0:
            pop = pop * (rng.uniform(0.3, 1.0) / peak_p) * intensity
        end = min(n, idx + len(pop))
        out[idx:end] += pop[: end - idx]

    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_thunder(duration_s: float = 6.0, distance: float = 0.5,
                  intensity: float = 0.9, sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Trueno: sweep modal grave + ruido coloreado decreciente.
    distance 0=cerca (golpe seco), 1=lejos (rumble difuso).
    """
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float32)
    # 1) Crack inicial (si cerca) - impulso filtrado alto
    if distance < 0.6:
        crack_n = int(0.05 * sr)
        crack = rng.standard_normal(crack_n).astype(np.float32) * (1 - distance)
        sos = signal.butter(4, [400, 5000], btype="band", fs=sr, output="sos")
        crack = signal.sosfiltfilt(sos, crack).astype(np.float32)
        out[: crack_n] += crack * 0.8
    # 2) Rumble grave largo (decay exponencial 3-5s)
    rumble = rng.standard_normal(n).astype(np.float32)
    # Lowpass progresivamente mas oscuro con la distancia
    f_high = 600 - 400 * distance
    sos = signal.butter(4, [30, max(80, f_high)], btype="band", fs=sr, output="sos")
    rumble = signal.sosfiltfilt(sos, rumble).astype(np.float32)
    decay = np.exp(-np.linspace(0, 4 - 2 * distance, n)).astype(np.float32)
    rumble = rumble * decay * (0.4 + 0.6 * (1 - distance * 0.5)) * intensity
    out = out + rumble
    # 3) Reverberacion via convolucion con impulso exponencial corto
    ir_n = int(0.4 * sr)
    ir = (np.exp(-np.linspace(0, 6, ir_n)) * rng.standard_normal(ir_n)).astype(np.float32) * 0.3
    out_rev = np.convolve(out, ir, mode="full")[:n].astype(np.float32)
    out = 0.7 * out + 0.4 * out_rev
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_glass_break(duration_s: float = 4.0, n_shards: int = 30,
                      sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Cristal rompiendose: pico modal glass + cascada granular de shards."""
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    # 1) Crack inicial: ruido bandpass agudo corto
    crack_n = int(0.025 * sr)
    crack = rng.standard_normal(crack_n).astype(np.float32)
    sos = signal.butter(4, [2000, 8000], btype="band", fs=sr, output="sos")
    crack = signal.sosfiltfilt(sos, crack).astype(np.float32) * 0.9
    out[: crack_n] += crack
    # 2) Cascada de shards: N impactos modales glass cortos dispersos
    glass = MODAL_PROFILES["glass"]
    spread_samples = int(0.6 * sr)  # dispersion en 600 ms
    for k in range(n_shards):
        offset = int(rng.uniform(0.01, spread_samples / sr) * sr)
        sub_profile = MaterialModalProfile(
            name=f"shard_{k}",
            n_modes=4,
            fundamental_hz=glass.fundamental_hz * (0.5 + 1.5 * rng.random()),
            spacing=glass.spacing,
            damping_ms=glass.damping_ms * (0.3 + 0.7 * rng.random()),
            spectrum_shape="flat",
            inharmonicity=glass.inharmonicity,
            seed=seed + k,
        )
        evt = synth_modal_impact(sub_profile, sr,
                                 duration_s=duration_s - offset / sr,
                                 impact_time_s=0.001,
                                 impact_strength=rng.uniform(0.3, 1.0),
                                 sharpness=2.0)
        amp = rng.uniform(0.2, 0.7) / (1 + k * 0.05)
        end = min(n, offset + len(evt))
        out[offset:end] += amp * evt[: end - offset]
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_ocean_wave(duration_s: float = 6.0, breaking_intensity: float = 0.7,
                     sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Ola del mar rompiendo: build-up (noise crescendo) + crash (granular bubbles) + receding (LPF decay)."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float32)

    # Fase 1: build-up (0..40%) — ruido bandpass crescendo
    p1_end = int(0.4 * n)
    bed = rng.standard_normal(p1_end).astype(np.float32) * 0.2
    sos = signal.butter(2, [200, 1800], btype="band", fs=sr, output="sos")
    bed = signal.sosfiltfilt(sos, bed).astype(np.float32)
    crescendo = np.linspace(0.1, 1.0, p1_end).astype(np.float32) ** 1.5
    out[: p1_end] += bed * crescendo

    # Fase 2: crash (40..60%) — burbujas densas y bandpass amplio
    p2_start, p2_end = p1_end, int(0.6 * n)
    p2_n = p2_end - p2_start
    crash = rng.standard_normal(p2_n).astype(np.float32) * 0.6 * breaking_intensity
    sos = signal.butter(2, [100, 5000], btype="band", fs=sr, output="sos")
    crash = signal.sosfiltfilt(sos, crash).astype(np.float32)
    out[p2_start:p2_end] += crash
    # Burbujas dentro del crash
    for _ in range(int(40 * breaking_intensity)):
        offset = p2_start + int(rng.uniform(0, p2_n - 100))
        d = DropletParams(droplet_radius_mm=0.8 + 1.5 * rng.random(),
                          viscosity=0.05, surface_hardness=0.1,
                          roll_velocity_hz=1, path_roughness=0,
                          duration_s=0.2, seed=seed + offset)
        evt = synth_drip_event(d, sr)
        end = min(n, offset + len(evt))
        out[offset:end] += 0.3 * evt[: end - offset]

    # Fase 3: receding (60..100%) — decay lento con LPF
    p3_start = p2_end
    p3_n = n - p3_start
    recede = rng.standard_normal(p3_n).astype(np.float32) * 0.3
    sos = signal.butter(2, [80, 1200], btype="band", fs=sr, output="sos")
    recede = signal.sosfiltfilt(sos, recede).astype(np.float32)
    decay = np.exp(-np.linspace(0, 3, p3_n)).astype(np.float32)
    out[p3_start:] += recede * decay
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


# ====================================================================
# v6: cuatro fenomenos "imposibles" nuevos, recombinando primitivas
# ====================================================================
def synth_burning_water(duration_s: float = 5.0, intensity: float = 0.7,
                        drip_rate_hz: float = 8.0, sr: int = 44_100,
                        seed: int = 0) -> np.ndarray:
    """Agua ardiendo: reusa la ESTRUCTURA del bed de synth_fire (ruido
    marron bandpass [80,1500] con doble AM), pero las chispas ya no son
    micro-impactos modales de madera sino gotas Minnaert reales
    (synth_drip_event) — fuego cuyas chispas son gotas liquidas rodando.

    Un sizzle fino [2000,6000] Hz esta GATED por la envolvente del bed
    (multiplicado por ella, nunca libre) para que nunca suene como un
    hiss independiente.
    """
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)

    # Bed: ruido marron modulado por dos AM (estructura de synth_fire)
    bed = rng.standard_normal(n).astype(np.float32) * 0.15
    sos = signal.butter(4, [80, 1500], btype="band", fs=sr, output="sos")
    bed = signal.sosfiltfilt(sos, bed).astype(np.float32)
    t_n = np.arange(n) / sr
    am_freq1 = 1.5 + rng.random() * 2
    am_freq2 = 0.4 + 0.6 * rng.random()
    am = (0.5 + 0.3 * np.sin(2 * np.pi * am_freq1 * t_n)
              + 0.2 * np.sin(2 * np.pi * am_freq2 * t_n + 1.0)).astype(np.float32)
    bed = bed * am * intensity

    out = bed.copy()

    # Chispas = gotas Minnaert (radio 0.8-2.0 mm) a drip_rate_hz eventos/s
    # con jitter, en vez de crackles modales de madera.
    period_base = sr / max(drip_rate_hz, 0.1)
    t = 0.0
    k = 0
    while t < n:
        radius = float(rng.uniform(0.8, 2.0))
        vel = float(np.clip(rng.normal(1.0, 0.25), 0.5, 1.8))
        d = DropletParams(droplet_radius_mm=radius, viscosity=0.0,
                          surface_hardness=0.0, roll_velocity_hz=1, path_roughness=0,
                          duration_s=0.12, seed=seed + k)
        evt = synth_drip_event(d, sr, velocity_factor=vel, seed_override=seed + 5000 + k)
        # Nota: synth_drip_event ya normaliza el evento a pico ~0.7, asi que
        # este rango de amp esta calibrado para dar al burning_water un pico
        # global comparable al resto de synth_* de este modulo (fire ~0.73,
        # rain ~0.86, thunder/glass_break ~0.95), no solo peak<=0.95.
        amp = rng.uniform(0.6, 1.5) * intensity
        start = int(t)
        end = min(n, start + len(evt))
        out[start:end] += amp * evt[: end - start]
        t += period_base * (1 + 0.6 * rng.uniform(-0.8, 0.8))
        k += 1

    # Sizzle fino GATED por la envolvente del bed (nunca libre)
    sizzle_noise = rng.standard_normal(n).astype(np.float32)
    sos_s = signal.butter(3, [2000, min(6000, sr / 2 - 200)], btype="band", fs=sr, output="sos")
    sizzle = signal.sosfiltfilt(sos_s, sizzle_noise).astype(np.float32)
    bed_env = np.abs(bed) / (float(np.max(np.abs(bed))) + 1e-9)
    out = out + sizzle * bed_env * 0.1 * intensity

    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_glass_thunder(duration_s: float = 6.0, distance: float = 0.4,
                        ring_gain: float = 0.6, sr: int = 44_100,
                        seed: int = 0) -> np.ndarray:
    """Trueno de cristal: estructura de synth_thunder (crack + rumble grave
    con decay exponencial) pero el rumble EXCITA el banco modal de vidrio
    -- patron driven-bandpass de friction.synth_scrape (bandpass butter de
    2o orden por modo sobre la senal excitadora) con los modos de
    droplet.SURFACE_PROFILES['glass'] (1800/4200/7100/9800 Hz), atenuando
    los modos >5 kHz x0.5. En el instante del crack dispara 3-6 anadidos
    de cristal (synth_modal_impact con sub-perfiles glass aleatorizados,
    igual patron que synth_glass_break).
    """
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float32)
    glass_surf = SURFACE_PROFILES["glass"]
    near = distance < 0.6

    # 1) Crack inicial (si cerca) - impulso filtrado alto
    crack_n = int(0.05 * sr)
    if near:
        crack = rng.standard_normal(crack_n).astype(np.float32) * (1 - distance)
        sos = signal.butter(4, [400, 5000], btype="band", fs=sr, output="sos")
        crack = signal.sosfiltfilt(sos, crack).astype(np.float32)
        out[: crack_n] += crack * 0.8

    # 2) Rumble grave largo (decay exponencial), como en synth_thunder
    rumble = rng.standard_normal(n).astype(np.float32)
    f_high = 600 - 400 * distance
    sos = signal.butter(4, [30, max(80, f_high)], btype="band", fs=sr, output="sos")
    rumble = signal.sosfiltfilt(sos, rumble).astype(np.float32)
    decay = np.exp(-np.linspace(0, 4 - 2 * distance, n)).astype(np.float32)
    rumble = rumble * decay * (0.4 + 0.6 * (1 - distance * 0.5))
    out = out + rumble

    # 2b) El rumble EXCITA el banco modal de vidrio (driven-bandpass, patron
    # de friction.synth_scrape: excited = bright_noise * env, luego bandpass
    # por modo). El rumble en si esta confinado a graves (<~500 Hz) por su
    # propio filtrado, asi que NO tiene energia en 1800-9800 Hz para que el
    # bandpass por modo extraiga -- igual que en synth_scrape, lo que
    # "excita" el banco modal es un ruido ANCHO modulado por la envolvente
    # lenta del rumble (su contorno de energia = el "empuje" del trueno).
    rumble_env = np.abs(rumble)
    sos_env = signal.butter(2, 12, btype="low", fs=sr, output="sos")
    rumble_env = signal.sosfiltfilt(sos_env, rumble_env).astype(np.float32)
    broadband = rng.standard_normal(n).astype(np.float32)
    sos_bb = signal.butter(2, [500, min(11000, sr / 2 - 200)], btype="band", fs=sr, output="sos")
    broadband = signal.sosfiltfilt(sos_bb, broadband).astype(np.float32)
    excited = broadband * rumble_env

    modal_bed = np.zeros(n, dtype=np.float32)
    for fc, mg in zip(glass_surf.modes_hz, glass_surf.mode_gains):
        if fc <= 0 or fc >= sr / 2 - 100:
            continue
        fc_j = fc * (1 + glass_surf.inharmonicity * rng.uniform(-1, 1))
        q = 3.0 + 8.0 * (1 - glass_surf.inharmonicity)
        bw = fc_j / max(q, 0.5)
        f_lo = max(50, fc_j - bw / 2)
        f_hi = min(sr / 2 - 100, fc_j + bw / 2)
        if f_lo >= f_hi:
            continue
        sos_mode = signal.butter(2, [f_lo, f_hi], btype="band", fs=sr, output="sos")
        mode_resp = signal.sosfiltfilt(sos_mode, excited).astype(np.float32)
        atten = 0.5 if fc > 5000 else 1.0
        modal_bed += mg * mode_resp * atten
    out = out + modal_bed * ring_gain

    # 3) En el instante del crack: 3-6 anadidos de cristal, mismo patron que
    # synth_glass_break (sub-perfiles glass aleatorizados).
    if near:
        glass_modal = MODAL_PROFILES["glass"]
        n_shards = int(rng.integers(3, 7))
        for k in range(n_shards):
            offset = int(rng.uniform(0, 0.08) * sr)
            sub_profile = MaterialModalProfile(
                name=f"gt_shard_{k}",
                n_modes=4,
                fundamental_hz=glass_modal.fundamental_hz * (0.5 + 1.5 * rng.random()),
                spacing=glass_modal.spacing,
                damping_ms=glass_modal.damping_ms * (0.3 + 0.7 * rng.random()),
                spectrum_shape="flat",
                inharmonicity=glass_modal.inharmonicity,
                seed=seed + 500 + k,
            )
            evt = synth_modal_impact(sub_profile, sr,
                                     duration_s=duration_s - offset / sr,
                                     impact_time_s=0.001,
                                     impact_strength=rng.uniform(0.4, 1.0) * (1 - 0.5 * distance),
                                     sharpness=2.0)
            amp = rng.uniform(0.2, 0.6) * ring_gain
            end = min(n, offset + len(evt))
            out[offset:end] += amp * evt[: end - offset]

    # 4) Reverberacion via convolucion con impulso exponencial corto
    ir_n = int(0.4 * sr)
    ir = (np.exp(-np.linspace(0, 6, ir_n)) * rng.standard_normal(ir_n)).astype(np.float32) * 0.3
    out_rev = np.convolve(out, ir, mode="full")[:n].astype(np.float32)
    out = 0.7 * out + 0.4 * out_rev

    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_mercury_rain(duration_s: float = 5.0, intensity: float = 0.6,
                       drop_radius_mm: float = 0.9, sr: int = 44_100,
                       seed: int = 0) -> np.ndarray:
    """Lluvia de mercurio: el scheduler de gotas de synth_rain (densidad
    modulada por LFO de viento + rafagas) pero cada gota es un
    synth_drip_event sobre superficie 'metal' con capillary_ringing alto y
    viscosity baja -- llovizna que tintinea metalica. El hiss ancho de
    rain se reduce a x0.15: las colas modales metalicas SON la textura.
    """
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    base_rate = 40 + 200 * intensity
    wind_strength = 0.3  # fijo (no expuesto en la firma)

    # Densidad modulada por LFO de viento + rafagas (mismo patron de synth_rain)
    sub_n = max(64, int(duration_s * 20))
    t_sub = np.linspace(0, duration_s, sub_n)
    wind_lfo = 1 + wind_strength * 0.6 * np.sin(2 * np.pi * 0.4 * t_sub +
                                                  2 * np.pi * rng.random())
    n_gusts = max(1, int(0.3 * duration_s))
    for _ in range(n_gusts):
        center = rng.uniform(0.3, duration_s - 0.3)
        width = rng.uniform(0.4, 1.5)
        amp = wind_strength * rng.uniform(0.5, 1.2)
        wind_lfo += amp * np.exp(-((t_sub - center) / width) ** 2)
    density_env = np.clip(wind_lfo, 0.3, 3.0)

    t = 0.0
    k = 0
    while t < n:
        sub_idx = int((t / n) * sub_n)
        density = float(density_env[min(sub_idx, sub_n - 1)])
        period = sr / (base_rate * density)
        radius = float(drop_radius_mm * rng.uniform(0.7, 1.4))
        vel = float(np.clip(rng.normal(1.0 + 0.3 * wind_strength, 0.25), 0.5, 1.8))
        d = DropletParams(droplet_radius_mm=radius, surface_profile="metal",
                          capillary_ringing=0.9, viscosity=0.1,
                          roll_velocity_hz=1, path_roughness=0,
                          duration_s=0.15, seed=seed + k)
        evt = synth_drip_event(d, sr, velocity_factor=vel, seed_override=seed + 6000 + k)
        amp = rng.uniform(0.2, 0.7) * (0.4 + intensity) * vel
        start = int(t)
        end = min(n, start + len(evt))
        out[start:end] += amp * evt[: end - start]
        t += period * (1 + 0.7 * rng.uniform(-0.8, 0.8))
        k += 1

    # Hiss ancho reducido a x0.15: la textura la dan las colas modales metalicas
    noise = rng.standard_normal(n).astype(np.float32) * 0.05 * intensity * 0.15
    sos = signal.butter(2, [500, 5000], btype="band", fs=sr, output="sos")
    bed = signal.sosfiltfilt(sos, noise).astype(np.float32)
    out = out + bed

    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def synth_fabric_bell(duration_s: float = 5.0, size: float = 0.5,
                      softness: float = 0.7, sr: int = 44_100,
                      seed: int = 0) -> np.ndarray:
    """Campana de tela: material IMPOSIBLE. synth_modal_impact con un perfil
    construido con blend_profiles(PROFILES['metal'], PROFILES['fabric'],
    amount=softness) pero CORRIGIENDO a mano los campos que deben quedarse
    de metal: fundamental_hz (escala con size), spacing y n_modes -- solo
    damping_ms y spectrum_shape vienen de la interpolacion hacia fabric.
    Resultado: estructura de campana metalica con timbre mullido.

    Renderiza 2-3 golpes lentos (t~0.3s, ~2.2s, ~4s) con
    excitation_shape='felt' (mallet acolchado).
    """
    rng = np.random.default_rng(seed)
    metal = MODAL_PROFILES["metal"]
    fabric = MODAL_PROFILES["fabric"]
    blended = blend_profiles(metal, fabric, amount=softness, name="fabric_bell_blend")
    profile = MaterialModalProfile(
        name="fabric_bell",
        n_modes=metal.n_modes,
        fundamental_hz=metal.fundamental_hz * (1.4 - 0.8 * size),
        spacing=metal.spacing,
        damping_ms=blended.damping_ms,
        spectrum_shape=blended.spectrum_shape,
        inharmonicity=metal.inharmonicity,
        seed=seed,
    )

    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    hit_times = [t0 for t0 in (0.3, 2.2, 4.0) if t0 < duration_s - 0.05]
    for k, t0 in enumerate(hit_times):
        strength = float(rng.uniform(0.5, 0.8))
        evt = synth_modal_impact(profile, sr, duration_s=duration_s,
                                 impact_time_s=t0,
                                 impact_strength=strength,
                                 sharpness=1.0,
                                 excitation_shape="felt",
                                 seed_override=seed + 300 + k)
        out += evt

    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out
