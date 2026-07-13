"""Hibridos de identidad (v8): UN solo objeto sonoro con fisica contradictoria.

A diferencia de los blends v7 (dos cuerpos acoplados causalmente, que el
analisis de escena auditiva sigue separando en dos corrientes), aqui cada
hibrido es UNA voz: mismo registro, misma estructura fina, mismo contorno
-- y la imposibilidad vive DENTRO del objeto (leyes trasplantadas).

Cuatro hibridos:
  H1 el trueno ES una gota gigante (transposicion de escala, cero trueno real)
  H2 cristal que burbujea (trasplante de ley por modo)
  H3 lluvia de truenos diminutos (transposicion inversa)
  H4 vocoder de identidad unica (registro+pitch+estructura fina compartidos)
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from impossible_mix.physics.analysis import band_envelopes, smooth_env
from impossible_mix.physics.blend import (
    ChirpLaw,
    DampingLaw,
    EventSchedule,
    chirping_modal_bank,
    excitation_from_schedule,
    f_traj_from_schedule,
    schedule_from_rate,
    schedule_from_roll,
)
from impossible_mix.physics.droplet import (
    SURFACE_PROFILES,
    _driven_resonator,
    _roll_schedule,
    _derive_params,
)
from impossible_mix.physics.droplet_presets import get_preset


def _norm(w: np.ndarray, peak: float = 0.95) -> np.ndarray:
    p = float(np.max(np.abs(w)) + 1e-9)
    if p > peak:
        w = w * (peak / p)
    return w.astype(np.float32)


# ====================================================================
# H1 — El trueno ES una gota gigante (transposicion de escala)
# ====================================================================
def hybrid_thunder_is_droplet(duration_s: float = 8.0, seed: int = 42,
                              radius_mm: float = 32.0, n_subbubbles: int = 5,
                              dirt: float = 0.6, sr: int = 44_100) -> np.ndarray:
    """Un trueno sintetizado INTEGRAMENTE con fisica de gota: burbuja de
    Minnaert de ~32 mm (f0 ~ 102 Hz, el registro del trueno), crack = snap
    capilar gigante, wobble = modos de Rayleigh reales del radio gigante
    (sub-audio). Cero capas de trueno real."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    out = np.zeros(n, dtype=np.float32)
    f0 = 3.26 / (radius_mm * 1e-3)          # ~102 Hz para 32 mm

    # --- Crack: snap capilar gigante (click+capilar del drip, x13 escala) ---
    crack_n = int(0.040 * sr)
    crack = rng.standard_normal(crack_n).astype(np.float32)
    sos_c = signal.butter(3, [250, 2600], btype="band", fs=sr, output="sos")
    crack = signal.sosfilt(sos_c, crack).astype(np.float32)
    crack *= np.exp(-3 * np.arange(crack_n) / crack_n).astype(np.float32)
    out[:crack_n] += 0.9 * crack
    f_cap = 2.2 * 1.6 * f0                   # ~360 Hz
    cap_n = int(0.150 * sr)
    t_cap = np.arange(cap_n) / sr
    out[crack_n:crack_n + cap_n] += (
        0.35 * np.sin(2 * np.pi * f_cap * t_cap)
        * np.exp(-np.linspace(0, 6, cap_n))).astype(np.float32)[:max(0, min(cap_n, n - crack_n))]

    # --- Cuerpo: coro de sub-burbujas gigantes, resonadores DRIVEN ---
    # Excitacion: crack + turbulencia grave gateada por el decay global
    t = np.arange(n) / sr
    decay_env = np.exp(-t / (0.45 * duration_s)).astype(np.float32)
    sos_t = signal.butter(2, 400.0, btype="low", fs=sr, output="sos")
    turb = signal.sosfilt(sos_t, rng.standard_normal(n).astype(np.float32)).astype(np.float32)
    exc = np.zeros(n, dtype=np.float32)
    exc[:crack_n] += crack
    exc += (0.25 + 0.75 * dirt) * turb * decay_env

    # Wobble por modos de Rayleigh del radio gigante (fisica de la misma gota)
    r_m = radius_mm * 1e-3
    rho, sigma = 1000.0, 0.072
    wob = np.zeros(n)
    for mode in range(2, 6):
        f_ray = float(np.sqrt(mode * (mode - 1) * (mode + 2) * sigma / (rho * r_m ** 3))
                      / (2 * np.pi))                      # 0.7 - 2.3 Hz
        wob += (1.0 / mode) * np.sin(2 * np.pi * f_ray * t + 2 * np.pi * rng.random())
    wob = wob / (np.abs(wob).max() + 1e-9)

    n_sb = max(1, int(n_subbubbles))
    chirp_ms = 15 + 8 * radius_mm            # ley chirp_duration_ms del drip (~271 ms)
    glide_n = int(chirp_ms / 1000.0 * sr)
    for j in range(n_sb):
        r_j = radius_mm * float(np.exp(0.25 * rng.standard_normal())) if n_sb > 1 else radius_mm
        f_j = 3.26 / (max(r_j, 5.0) * 1e-3)
        onset = int(rng.uniform(0, 0.080) * sr) if n_sb > 1 else 0
        t_g = np.arange(glide_n) / glide_n
        f_glide = 0.45 * f_j * (1.6 / 0.45) ** t_g
        f_hold = 1.6 * f_j * (1.0 + (0.03 + 0.02 * dirt) * wob[glide_n + onset:n])
        f_traj = np.concatenate([np.full(onset, 0.45 * f_j), f_glide,
                                 f_hold])[:n].astype(np.float32)
        # Cuantizar a 0.1 Hz: inaudible y hace efectivo el cache de
        # coeficientes del resonador durante el hold con FM lenta.
        f_traj = np.round(f_traj, 1)
        voice = _driven_resonator(exc, f_traj, 10.0 + 6.0 * rng.random(), sr, blk=128)
        out += (0.9 / np.sqrt(n_sb)) * voice / (float(np.abs(voice).max()) + 1e-9)

    out *= (0.4 + 0.6 * decay_env)
    # Flutter y saturacion suave (suciedad fisica, no cosmetica)
    if dirt > 0.01:
        sos_f = signal.butter(2, 8.0, btype="low", fs=sr, output="sos")
        fl = signal.sosfiltfilt(sos_f, rng.standard_normal(n)).astype(np.float32)
        fl = (fl - fl.mean()) / (fl.std() + 1e-9)
        out *= np.clip(1.0 + 0.15 * dirt * fl, 0.3, None).astype(np.float32)
        out = (np.tanh(1.5 * out) / 1.5).astype(np.float32)

    # Cola convolucionada (el cielo)
    ir_n = int(0.8 * sr)
    ir = (np.random.default_rng(seed + 13).standard_normal(ir_n)
          * np.exp(-np.linspace(0, 7, ir_n))).astype(np.float32)
    wet = signal.fftconvolve(out, ir)[:n].astype(np.float32)
    out = out + 0.5 * wet / (float(np.abs(wet).max()) + 1e-9) * float(np.abs(out).max())
    return _norm(out)


# ====================================================================
# H2 — Cristal que burbujea (trasplante de ley por modo)
# ====================================================================
def hybrid_bubbling_glass_impact(duration_s: float = 4.0, seed: int = 42,
                                 n_hits: int = 3, pure_vdd: bool = False,
                                 sr: int = 44_100) -> np.ndarray:
    """Impactos de vidrio donde cada modo hace el glide ascendente de una
    burbuja y decae con la forma de van den Doel: el material es vidrio,
    su comportamiento temporal es agua."""
    n = int(duration_s * sr)
    surf = SURFACE_PROFILES["glass"]
    starts = np.linspace(0.35, duration_s - 1.2, max(1, n_hits))
    sched = EventSchedule(
        starts=(starts * sr).astype(np.int64),
        amps=np.array([1.0, 0.7, 0.85][:n_hits] + [0.8] * max(0, n_hits - 3)),
        vels=np.ones(n_hits), dur_hint=np.full(n_hits, 0.5))
    damping = (DampingLaw("vdd") if pure_vdd
               else DampingLaw("vdd_shape", t60_ref_ms=600.0, f_ref_hz=1800.0))
    w = chirping_modal_bank(
        sched, surf.modes_hz, surf.mode_gains, sr, n,
        chirp=ChirpLaw(0.45, 1.0, 30.0, 1800.0, "sqrt", 3.0),
        damping=damping, inharmonicity=surf.inharmonicity, seed=seed,
        exc_click=(3000, 10000))
    return _norm(w)


def hybrid_bubbling_glass_roll(duration_s: float = 6.0, seed: int = 42,
                               sr: int = 44_100) -> np.ndarray:
    """Rodadura sobre vidrio que burbujea: los contactos del motor v5
    excitan el banco de vidrio con ley de burbuja trasplantada."""
    n = int(duration_s * sr)
    surf = SURFACE_PROFILES["glass"]
    p = _derive_params(get_preset("water", duration_s=duration_s, seed=seed))
    rng = np.random.default_rng(seed)
    roll = _roll_schedule(p, sr, n, rng)
    sched = schedule_from_roll(roll, p)
    w = chirping_modal_bank(
        sched, surf.modes_hz, surf.mode_gains, sr, n,
        chirp=ChirpLaw(0.45, 1.0, 22.0, 1800.0, "sqrt", 2.0),
        damping=DampingLaw("vdd_shape", t60_ref_ms=350.0, f_ref_hz=1800.0),
        inharmonicity=surf.inharmonicity, seed=seed)
    # rumor debil en el MISMO registro que los modos (una corriente)
    sos = signal.butter(4, [3000, 9000], btype="band", fs=sr, output="sos")
    from impossible_mix.physics.droplet import _surface_profile_wave
    e = _surface_profile_wave(p, roll, sr, n, rng)
    rum = signal.sosfilt(sos, rng.standard_normal(n).astype(np.float32)).astype(np.float32)
    w = w + 0.1 * rum * (e / (float(e.max()) + 1e-9))
    return _norm(w)


# ====================================================================
# H3 — Lluvia de truenos diminutos (transposicion inversa)
# ====================================================================
def hybrid_rain_of_thunders(duration_s: float = 8.0, seed: int = 42,
                            density_hz: float = 12.0, wind: float = 0.4,
                            sr: int = 44_100) -> np.ndarray:
    """Lluvia cuyo grano es un trueno miniaturizado: crack diminuto +
    micro-rumble en dos sub-bandas con decaimientos distintos (la banda
    alta muere antes: la firma espectral DESCENDENTE del trueno, el
    contraste exacto con el chirp ascendente de una gota)."""
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr

    # Scheduler de lluvia: densidad modulada por viento + rafagas
    lfo = np.sin(2 * np.pi * 0.3 * t + 2 * np.pi * rng.random())
    rate_traj = density_hz * (1.0 + wind * 0.6 * lfo)
    for _ in range(3):
        c = rng.uniform(0.15, 0.85) * n
        rate_traj += density_hz * wind * 1.5 * np.exp(-0.5 * ((np.arange(n) - c)
                                                              / (0.06 * n)) ** 2)
    sched = schedule_from_rate(np.maximum(rate_traj, 0.5), sr, seed + 3)

    out = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(sched.starts):
        k = float(rng.uniform(25, 40))
        dur_i = min(3000.0 / k / 1000.0, 0.080)
        g_n = max(64, int(dur_i * sr))
        if int(s) + g_n >= n:
            continue
        a = float(sched.amps[i]) * rng.uniform(0.5, 1.0)
        grain = np.zeros(g_n, dtype=np.float32)
        # crack diminuto
        c_n = max(8, int(0.0025 * sr))
        ck = np.random.default_rng(seed + 50 + i).standard_normal(c_n).astype(np.float32)
        hi_c = min(4000.0 * k, sr / 2 - 200)
        sos_ck = signal.butter(2, [min(400.0 * k, hi_c - 300), hi_c],
                               btype="band", fs=sr, output="sos")
        grain[:c_n] += 0.8 * signal.sosfilt(sos_ck, ck).astype(np.float32)
        # micro-rumble en dos sub-bandas, la alta muere antes
        nz = np.random.default_rng(seed + 90 + i).standard_normal(g_n).astype(np.float32)
        tau = dur_i / 4.0
        tg = np.arange(g_n) / sr
        f_lo, f_hi = 30.0 * k, min(300.0 * k, sr / 2 - 200)
        f_mid = np.sqrt(f_lo * f_hi)
        sos_l = signal.butter(2, [f_lo, f_mid], btype="band", fs=sr, output="sos")
        sos_h = signal.butter(2, [f_mid, f_hi], btype="band", fs=sr, output="sos")
        grain += (signal.sosfilt(sos_l, nz).astype(np.float32)
                  * np.exp(-tg / tau).astype(np.float32))
        grain += (signal.sosfilt(sos_h, nz).astype(np.float32)
                  * np.exp(-tg / (tau / 2.5)).astype(np.float32))
        peak_g = float(np.abs(grain).max() + 1e-9)
        out[int(s):int(s) + g_n] += a * grain / peak_g

    # bed en el MISMO registro que los granos, gateado por la densidad
    sos_b = signal.butter(4, [900, 9000], btype="band", fs=sr, output="sos")
    bed = signal.sosfilt(sos_b, rng.standard_normal(n).astype(np.float32)).astype(np.float32)
    out = out + 0.03 * bed * (rate_traj / rate_traj.max()).astype(np.float32)
    # micro-reverb ("el cielo en miniatura")
    ir_n = int(0.060 * sr)
    ir = (np.random.default_rng(seed + 13).standard_normal(ir_n)
          * np.exp(-np.linspace(0, 6, ir_n))).astype(np.float32)
    wet = signal.fftconvolve(out, ir)[:n].astype(np.float32)
    out = out + 0.35 * wet / (float(np.abs(wet).max()) + 1e-9) * float(np.abs(out).max())
    return _norm(out)


# ====================================================================
# H4 — Vocoder de identidad unica
# ====================================================================
def hybrid_identity_vocoder(duration_s: float = 8.0, seed: int = 42,
                            rate_hz: float = 9.0, radius_mm: float = 2.0,
                            fine: float = 1.0, body_q: float = 22.0,
                            glass_gain: float = 0.5,
                            sr: int = 44_100) -> np.ndarray:
    """El vocoder v7 cerrando las tres brechas de stream: (a) cuerpo y
    articulador en el MISMO registro (400-4500 Hz); (b) el cuerpo sigue el
    CONTORNO DE PITCH del goteo (chirpea con cada gota); (c) la excitacion
    usa la estructura fina del PROPIO articulador. Sin articulador seco."""
    from impossible_mix.physics.blend_recipes import _render_drips
    n = int(duration_s * sr)
    sched = schedule_from_rate(np.full(n, rate_hz), sr, seed + 3,
                               radius_traj=np.full(n, radius_mm))
    articulator = _render_drips(sched, sr, n, seed)

    # (c) estructura fina compartida: articulador HP 300, aplanado por su env lenta
    sos_hp = signal.butter(4, 300.0, btype="high", fs=sr, output="sos")
    art_hp = signal.sosfiltfilt(sos_hp, articulator.astype(np.float64)).astype(np.float32)
    env_slow = smooth_env(art_hp, sr, win_ms=40.0).astype(np.float32)
    art_flat = art_hp / (env_slow + 0.05 * float(env_slow.max()) + 1e-9)
    art_flat = art_flat / (float(np.abs(art_flat).max()) + 1e-9)

    exc = fine * art_flat
    if fine < 1.0:
        envs, _ = band_envelopes(articulator, sr, n_bands=10, lo=400.0, hi=4500.0)
        noise = np.random.default_rng(seed + 11).standard_normal(n).astype(np.float32)
        edges = np.geomspace(400.0, 4500.0, 11)
        voc = np.zeros(n, dtype=np.float32)
        for k in range(10):
            sos_k = signal.butter(4, [edges[k], edges[k + 1]], btype="band",
                                  fs=sr, output="sos")
            voc += signal.sosfiltfilt(sos_k, noise).astype(np.float32) * envs[k]
        voc = voc / (float(np.abs(voc).max()) + 1e-9)
        exc = exc + (1.0 - fine) * voc

    # (b) contorno de pitch: el cuerpo chirpea con cada gota
    f_M = 3.26 / (radius_mm * 1e-3)
    f_traj = f_traj_from_schedule(sched, sr, n,
                                  chirp=ChirpLaw(0.45, 1.6, 30.0),
                                  f_floor_hz=f_M)
    voice = _driven_resonator(exc, f_traj, body_q, sr, blk=64)
    voice = voice / (float(np.abs(voice).max()) + 1e-9)

    # (a) color de material EN BANDA sobre la misma excitacion
    out = voice.copy()
    if glass_gain > 0.01:
        for fc, g in ((1800.0, 0.6), (4200.0, 0.4)):
            r = float(np.exp(-6.91 / max(0.35 * sr, 1e-3)))
            th = 2 * np.pi * fc / sr
            mode = signal.lfilter([1.0, 0.0, -1.0],
                                  [1.0, -2 * r * np.cos(th), r * r],
                                  exc).astype(np.float32)
            out += glass_gain * g * mode / (float(np.abs(mode).max()) + 1e-9)
    return _norm(out)
