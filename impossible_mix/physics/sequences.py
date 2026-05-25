"""DSL minimal para componer SECUENCIAS de eventos fisicos.

Un sonido cinematico raramente es un evento de 5 segundos uniforme. Es una
historia: la gota cae, rueda, rebota, salpica al final. Aqui ofrecemos
una API declarativa para encadenar eventos con sus tiempos relativos.

API:
    seq = Sequence(duration_s=8.0)
    seq.add_at(0.5, drip(radius_mm=1.5))
    seq.add_at(0.8, rolling_droplet(duration_s=4.0, ...))
    seq.add_at(5.0, splash(intensity=0.8))
    wav = seq.render()

Cada `Event` es un dict con (start_s, wav, gain). El render simplemente
acumula y normaliza.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from impossible_mix.config import SAMPLE_RATE
from impossible_mix.physics.composer import CompositionSpec, compose, compose_impossible
from impossible_mix.physics.droplet import DropletParams, synth_drip_event, synth_rolling_droplet
from impossible_mix.physics.liquid import PourParams, SplashParams, synth_pour, synth_splash
from impossible_mix.physics.modal import PROFILES, synth_modal_impact


@dataclass
class Event:
    start_s: float
    wav: np.ndarray
    gain: float = 1.0


@dataclass
class Sequence:
    duration_s: float = 8.0
    sr: int = SAMPLE_RATE
    events: list[Event] = field(default_factory=list)

    def add_at(self, t_s: float, wav: np.ndarray, gain: float = 1.0) -> "Sequence":
        self.events.append(Event(start_s=t_s, wav=wav.astype(np.float32), gain=gain))
        return self

    def render(self, headroom_db: float = -3.0) -> np.ndarray:
        n_total = int(self.duration_s * self.sr)
        out = np.zeros(n_total, dtype=np.float32)
        for e in self.events:
            start = int(e.start_s * self.sr)
            if start >= n_total:
                continue
            end = min(n_total, start + len(e.wav))
            out[start:end] += e.gain * e.wav[: end - start]
        # Headroom
        target = 10 ** (headroom_db / 20)
        peak = float(np.max(np.abs(out)) + 1e-9)
        if peak > target:
            out = out * (target / peak)
        return out.astype(np.float32)


# ====================================================================
# Helpers semanticos: producen wavs para uso directo en Sequence.add_at
# ====================================================================

def evt_drip(radius_mm: float = 2.0, viscosity: float = 0.0,
             surface_hardness: float = 0.5, seed: int = 0,
             sr: int = SAMPLE_RATE) -> np.ndarray:
    p = DropletParams(droplet_radius_mm=radius_mm, viscosity=viscosity,
                       surface_hardness=surface_hardness, roll_velocity_hz=1,
                       path_roughness=0, duration_s=0.4, seed=seed)
    return synth_drip_event(p, sr)


def evt_rolling(duration_s: float = 3.0, radius_mm: float = 2.0,
                viscosity: float = 0.0, surface_hardness: float = 0.5,
                roll_velocity_hz: float = 14.0, path_roughness: float = 0.35,
                seed: int = 0, sr: int = SAMPLE_RATE) -> np.ndarray:
    p = DropletParams(droplet_radius_mm=radius_mm, viscosity=viscosity,
                       surface_hardness=surface_hardness,
                       roll_velocity_hz=roll_velocity_hz,
                       path_roughness=path_roughness,
                       duration_s=duration_s, seed=seed)
    return synth_rolling_droplet(p, sr)


def evt_splash(intensity: float = 0.7, n_bubbles: int = 30, seed: int = 0,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    p = SplashParams(intensity=intensity, n_bubbles=n_bubbles,
                      duration_s=1.0, seed=seed)
    return synth_splash(p, sr)


def evt_impact(material: str = "rock", rigidity: float = 0.5,
               resonance: float = 0.4, seed: int = 0,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    spec = CompositionSpec(material=material, interaction="impact",
                            rigidity=rigidity, resonance=resonance,
                            duration_s=1.5, seed=seed)
    return compose(spec, sr)


def evt_pour(flow_rate: float = 0.6, viscosity: float = 0.1,
             duration_s: float = 2.0, seed: int = 0,
             sr: int = SAMPLE_RATE) -> np.ndarray:
    p = PourParams(flow_rate=flow_rate, viscosity=viscosity,
                    duration_s=duration_s, seed=seed)
    return synth_pour(p, sr)


# ====================================================================
# Recetas cinematicas pre-armadas
# ====================================================================

def droplet_story(duration_s: float = 8.0, seed: int = 42,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """Historia clasica: 1 gota cae, rueda 3s, choca con superficie y splash."""
    seq = Sequence(duration_s=duration_s, sr=sr)
    seq.add_at(0.2, evt_drip(radius_mm=2.5, seed=seed), gain=0.9)
    seq.add_at(1.0, evt_rolling(duration_s=4.0, radius_mm=2.5,
                                 surface_hardness=0.6, roll_velocity_hz=14,
                                 path_roughness=0.35, seed=seed), gain=0.9)
    seq.add_at(5.2, evt_impact(material="rock", rigidity=0.5, seed=seed), gain=0.6)
    seq.add_at(5.5, evt_splash(intensity=0.7, n_bubbles=40, seed=seed), gain=0.8)
    return seq.render()


def mercury_drama(duration_s: float = 8.0, seed: int = 42,
                  sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mercurio cae sobre metal, rueda erratico, choca con multiples impactos."""
    from impossible_mix.physics.droplet_presets import get_preset
    seq = Sequence(duration_s=duration_s, sr=sr)
    p = get_preset("mercury", duration_s=4.0, seed=seed)
    p_drip = DropletParams(droplet_radius_mm=p.droplet_radius_mm,
                            viscosity=p.viscosity,
                            surface_hardness=p.surface_hardness,
                            roll_velocity_hz=1, path_roughness=0,
                            duration_s=0.4, seed=seed)
    seq.add_at(0.1, synth_drip_event(p_drip, sr), gain=1.0)
    seq.add_at(0.6, synth_rolling_droplet(p, sr), gain=1.0)
    seq.add_at(4.5, evt_impact(material="metal", rigidity=0.9, resonance=0.7, seed=seed), gain=0.7)
    seq.add_at(4.8, evt_impact(material="metal", rigidity=0.9, resonance=0.6, seed=seed + 1), gain=0.5)
    return seq.render()


def lava_step_into_water(duration_s: float = 8.0, seed: int = 42,
                          sr: int = SAMPLE_RATE) -> np.ndarray:
    """Paso pesado sobre lava, salpicadura espesa, gotas viscosas cayendo."""
    seq = Sequence(duration_s=duration_s, sr=sr)
    # Paso pesado (impact rocoso bajo con overlay splash)
    base = compose_impossible(base_material="rock", base_interaction="impact",
                               overlay_material="liquid", overlay_interaction="splash",
                               overlay_weight=0.7,
                               modifiers=dict(wetness=0.95, resonance=0.3, rigidity=0.2),
                               duration_s=2.5, seed=seed)
    seq.add_at(0.5, base, gain=1.0)
    # Gotas viscosas cayendo después
    for i, t in enumerate([3.5, 4.6, 5.5, 6.7]):
        d = DropletParams(droplet_radius_mm=3.5, viscosity=0.85,
                          surface_hardness=0.2, roll_velocity_hz=1,
                          path_roughness=0, duration_s=0.6, seed=seed + i)
        seq.add_at(t, synth_drip_event(d, sr), gain=0.7 - i * 0.1)
    return seq.render()
