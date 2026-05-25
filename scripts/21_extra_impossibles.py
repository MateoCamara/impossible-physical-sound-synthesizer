"""Genera combinaciones imposibles adicionales para demostrar el alcance
del marco fisico. Estos no entran en el abstract piloto pero seran utiles
para la version completa de septiembre y para la sesion presencial.

Cubre:
  - lava_footstep: paso sobre lava (gravel-like + liquid splash + modal calido)
  - boiling_water: agua hirviendo continua (granular liquid + pour bed)
  - wooden_wind: viento de madera (friction modal madera + scrape sostenido)
  - glass_avalanche: cristales cayendo (modal glass + granular)
  - mercury_drip: gota de mercurio (drip con surface_hardness alto y viscosity media)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import SAMPLE_RATE
from impossible_mix.metrics.quality import quality_verdict
from impossible_mix.physics.composer import (
    CompositionSpec,
    compose,
    compose_impossible,
)
from impossible_mix.physics.droplet import DropletParams, synth_rolling_droplet
from impossible_mix.physics.granular import GranularParams, synth_granular_flow
from impossible_mix.physics.modal import (
    MaterialModalProfile,
    synth_modal_impact,
    synth_modal_roll,
)


OUT = Path("outputs/extra_impossibles")
OUT.mkdir(parents=True, exist_ok=True)


def lava_footstep(duration_s=5.0, seed=42) -> np.ndarray:
    """Paso sobre lava: gravel base (gravel densidad media) + splash overlay
    (bubble grandes muy mojadas) + low modal warm bed."""
    sr = SAMPLE_RATE
    # capa 1: gravel (consistencia espesa)
    gran = GranularParams(grain_material="rock", density_hz=60, density_jitter=0.8,
                          grain_size_mm=10, size_variance=0.5, energy=0.5,
                          duration_s=duration_s, seed=seed)
    layer_gravel = synth_granular_flow(gran, sr)
    # capa 2: splash con bubbles grandes y muy wet
    splash_spec = CompositionSpec(material="liquid", interaction="splash",
                                   wetness=0.9, duration_s=duration_s, seed=seed + 1)
    layer_splash = compose(splash_spec, sr) * 0.5
    # capa 3: bed warm modal (basaltic resonator low)
    prof = MaterialModalProfile("lava_bed", n_modes=4, fundamental_hz=80,
                                spacing=2.4, damping_ms=300, inharmonicity=0.7)
    layer_bed = synth_modal_roll(prof, sr, duration_s=duration_s,
                                  rate_hz=4, jitter=0.7, strength=0.35) * 0.4
    n = min(len(layer_gravel), len(layer_splash), len(layer_bed))
    out = layer_gravel[:n] + layer_splash[:n] + layer_bed[:n]
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def boiling_water(duration_s=5.0, seed=42) -> np.ndarray:
    """Agua hirviendo: tren denso de mini-drips + pour bed."""
    sr = SAMPLE_RATE
    # Pour denso como base
    spec_pour = CompositionSpec(material="liquid", interaction="pour",
                                 wetness=0.7, continuity=0.85,
                                 duration_s=duration_s, seed=seed)
    base = compose(spec_pour, sr)
    # Burbujas mini periodicas
    rng = np.random.default_rng(seed)
    bubbles = np.zeros(int(duration_s * sr), dtype=np.float32)
    bubble_rate = 25  # ~25 burbujas por segundo
    period = sr / bubble_rate
    t = 0
    while t < len(bubbles):
        radius = 0.6 + 0.8 * rng.random()
        d = DropletParams(droplet_radius_mm=radius, viscosity=0.0,
                          surface_hardness=0.0, roll_velocity_hz=1, path_roughness=0,
                          duration_s=0.15, seed=seed + int(t))
        from impossible_mix.physics.droplet import synth_drip_event
        evt = synth_drip_event(d, sr)
        start = int(t)
        end = min(len(bubbles), start + len(evt))
        amp = 0.5 + 0.3 * rng.random()
        bubbles[start:end] += amp * evt[: end - start]
        t += period * (1 + 0.5 * rng.uniform(-0.6, 0.6))
    out = base + 0.7 * bubbles
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def wooden_wind(duration_s=5.0, seed=42) -> np.ndarray:
    """Viento de madera: scrape sostenido + modal madera resonante."""
    sr = SAMPLE_RATE
    spec = CompositionSpec(material="wood", interaction="scrape",
                            rigidity=0.5, resonance=0.7, continuity=0.95,
                            granularity=0.2, duration_s=duration_s, seed=seed)
    base = compose(spec, sr)
    # Overlay: ruido coloreado lento (viento)
    rng = np.random.default_rng(seed)
    wind = rng.standard_normal(len(base)).astype(np.float32) * 0.15
    from scipy import signal
    sos = signal.butter(2, [120, 2000], btype="band", fs=sr, output="sos")
    wind = signal.sosfiltfilt(sos, wind).astype(np.float32)
    out = base + wind
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def glass_avalanche(duration_s=5.0, seed=42) -> np.ndarray:
    """Cristales cayendo: granular con perfil modal glass + ramp de densidad."""
    sr = SAMPLE_RATE
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    # Ramp de densidad: empieza poco, crece, se desvanece
    for chunk_i, density in enumerate([5, 15, 40, 80, 60, 25, 10]):
        chunk_n = n // 7
        gran = GranularParams(grain_material="glass" if hasattr(__import__("impossible_mix.physics.modal", fromlist=["PROFILES"]).PROFILES, "get") else "metal",
                              density_hz=density, density_jitter=0.5,
                              grain_size_mm=4, size_variance=0.6, energy=0.5,
                              duration_s=duration_s / 7, seed=seed + chunk_i)
        layer = synth_granular_flow(gran, sr)
        start = chunk_i * chunk_n
        end = min(n, start + len(layer))
        out[start:end] += layer[: end - start]
    peak = float(np.max(np.abs(out)) + 1e-9)
    return (out / peak * 0.95).astype(np.float32) if peak > 0.95 else out


def mercury_drip(duration_s=5.0, seed=42) -> np.ndarray:
    """Gota de mercurio: drip con surface_hardness alto, viscosity moderada."""
    sr = SAMPLE_RATE
    d = DropletParams(droplet_radius_mm=1.0, viscosity=0.3,
                      surface_hardness=0.95, roll_velocity_hz=8,
                      path_roughness=0.1, duration_s=duration_s, seed=seed)
    return synth_rolling_droplet(d, sr)


GENERATORS = {
    "lava_footstep":   lava_footstep,
    "boiling_water":   boiling_water,
    "wooden_wind":     wooden_wind,
    "glass_avalanche": glass_avalanche,
    "mercury_drip":    mercury_drip,
}


def main() -> int:
    print(f"Generating {len(GENERATORS)} extra impossible sounds -> {OUT}")
    for name, gen in GENERATORS.items():
        wav = gen()
        sf.write(str(OUT / f"{name}.wav"), wav, SAMPLE_RATE)
        q = quality_verdict(wav, SAMPLE_RATE)
        print(f"  {name:18s}  rms={q.rms_db:6.1f} dyn={q.dynamic_range_db:5.1f}  {q.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
