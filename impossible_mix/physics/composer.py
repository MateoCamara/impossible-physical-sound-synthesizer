"""Composer fisico generico: dado (material, interaction, modifiers, duration),
selecciona los generadores apropiados y combina capas.

Soporta combinaciones IMPOSIBLES (gota rodante, roca liquida, gravel mojado)
porque los generadores son ortogonales: cada uno aporta una capa y la
combinacion fisica resulta del overlay ponderado.

API publica:
    out = compose(material="liquid", interaction="roll", modifiers={...})
    out = compose_impossible(scene={...})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from impossible_mix.physics.droplet import (
    DropletParams,
    synth_drip_event,
    synth_rolling_droplet,
)
from impossible_mix.physics.friction import FrictionParams, synth_scrape
from impossible_mix.physics.granular import GranularParams, synth_granular_flow
from impossible_mix.physics.liquid import (
    PourParams,
    SplashParams,
    synth_pour,
    synth_splash,
)
from impossible_mix.physics.modal import (
    PROFILES as MODAL_PROFILES,
    MaterialModalProfile,
    synth_modal_impact,
    synth_modal_roll,
)


@dataclass
class CompositionSpec:
    """Especificacion de alto nivel de la escena a sintetizar.

    Cada campo es un knob fisico con significado directo. Knobs vacios usan
    defaults razonables.
    """
    material: str = "rock"              # principal
    interaction: str = "impact"
    duration_s: float = 5.0
    seed: int = 0

    # Modificadores fisicos comunes (escala 0..1; >1 extrapola)
    wetness: float = 0.0                # 0=seco, 1=muy mojado
    granularity: float = 0.0            # 0=continuo, 1=muy granular
    rigidity: float = 0.5               # 0=blando, 1=rigido
    resonance: float = 0.5              # 0=mate, 1=muy resonante
    continuity: float = 0.5             # 0=impulsivo, 1=continuo

    # Mezcla entre capas (cuando coexisten)
    layer_weights: dict[str, float] = field(default_factory=dict)


def _modal_profile_for(material: str, rigidity: float, resonance: float) -> MaterialModalProfile:
    """Retorna o construye un perfil modal segun material + modificadores."""
    base = MODAL_PROFILES.get(material, MODAL_PROFILES["rock"])
    # Ajustar damping segun resonance
    return MaterialModalProfile(
        name=f"{material}_r{rigidity:.1f}_q{resonance:.1f}",
        n_modes=base.n_modes,
        fundamental_hz=base.fundamental_hz * (0.5 + rigidity * 1.5),
        spacing=base.spacing,
        damping_ms=base.damping_ms * (0.3 + resonance * 1.7),
        spectrum_shape=base.spectrum_shape,
        inharmonicity=base.inharmonicity,
        seed=base.seed,
    )


def _solid_impact(spec: CompositionSpec, sr: int) -> np.ndarray:
    prof = _modal_profile_for(spec.material, spec.rigidity, spec.resonance)
    n = int(spec.duration_s * sr)
    return synth_modal_impact(prof, sr, duration_s=spec.duration_s,
                              impact_time_s=0.05, impact_strength=0.9,
                              sharpness=1.0 + 1.5 * spec.rigidity)


def _solid_roll(spec: CompositionSpec, sr: int) -> np.ndarray:
    prof = _modal_profile_for(spec.material, spec.rigidity, spec.resonance)
    return synth_modal_roll(prof, sr, duration_s=spec.duration_s,
                            rate_hz=10 + 30 * (1 - spec.continuity * 0.5),
                            jitter=0.3 + 0.5 * (1 - spec.continuity),
                            strength=0.6)


def _scrape(spec: CompositionSpec, sr: int) -> np.ndarray:
    prof = _modal_profile_for(spec.material, spec.rigidity, spec.resonance)
    p = FrictionParams(
        surface_hardness=spec.rigidity,
        roughness=spec.granularity,
        velocity_mean=0.5 + 0.4 * spec.continuity,
        velocity_jitter=0.3 + 0.4 * (1 - spec.continuity),
        pressure=0.6,
        body_resonance_hz=prof.fundamental_hz,
        body_q=2 + 6 * spec.resonance,
        duration_s=spec.duration_s,
        seed=spec.seed,
    )
    return synth_scrape(p, sr)


def _drip(spec: CompositionSpec, sr: int) -> np.ndarray:
    """Una sola gota (un evento drip dentro de duration_s)."""
    n = int(spec.duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    d = DropletParams(droplet_radius_mm=2.0 * (1 + spec.rigidity * 0.5),
                      viscosity=0.5 * (1 - spec.wetness * 0.5),
                      surface_hardness=spec.rigidity,
                      roll_velocity_hz=1, path_roughness=0,
                      duration_s=0.5, seed=spec.seed)
    evt = synth_drip_event(d, sr)
    start = int(0.2 * sr)
    out[start:start + len(evt)] = evt[: n - start]
    return out


def _splash(spec: CompositionSpec, sr: int) -> np.ndarray:
    p = SplashParams(
        intensity=0.6 + 0.4 * spec.wetness,
        bubble_size_mean_mm=2.0 + 3.0 * spec.rigidity,
        n_bubbles=int(20 + 40 * spec.wetness),
        viscosity=0.3 * (1 - spec.wetness * 0.5),
        spread_ms=100 - 50 * spec.continuity,
        duration_s=spec.duration_s,
        seed=spec.seed,
    )
    return synth_splash(p, sr)


def _pour(spec: CompositionSpec, sr: int) -> np.ndarray:
    p = PourParams(
        flow_rate=0.4 + 0.6 * spec.continuity,
        bubble_size_mean_mm=1.5,
        viscosity=0.3 * (1 - spec.wetness * 0.5),
        duration_s=spec.duration_s,
        seed=spec.seed,
    )
    return synth_pour(p, sr)


def _rolling_droplet(spec: CompositionSpec, sr: int) -> np.ndarray:
    """Rolling droplet con knobs derivables de la especificacion comun."""
    d = DropletParams(
        droplet_radius_mm=1.5 + 2.5 * (1 - spec.rigidity),  # mas rigido = gota mas pequena
        viscosity=max(0.0, 0.6 - 0.5 * spec.wetness),       # mas mojado = menos viscoso
        surface_hardness=spec.rigidity,
        roll_velocity_hz=8 + 20 * spec.continuity,
        path_roughness=0.2 + 0.5 * spec.granularity,
        duration_s=spec.duration_s,
        seed=spec.seed,
    )
    return synth_rolling_droplet(d, sr)


def _granular_step(spec: CompositionSpec, sr: int) -> np.ndarray:
    p = GranularParams(
        grain_material=spec.material if spec.material in ("rock", "metal", "wood") else "rock",
        density_hz=30 + 200 * spec.continuity,
        density_jitter=0.4 + 0.5 * (1 - spec.continuity),
        grain_size_mm=4 + 8 * (1 - spec.granularity),
        size_variance=0.3 + 0.4 * spec.granularity,
        energy=0.5,
        spatial_spread=0.5 + 0.3 * spec.granularity,
        duration_s=spec.duration_s,
        seed=spec.seed,
    )
    return synth_granular_flow(p, sr)


# Tabla (material, interaction) -> generador principal
GENERATORS: dict[tuple[str, str], Callable[[CompositionSpec, int], np.ndarray]] = {
    # solidos
    ("rock", "impact"):   _solid_impact,
    ("metal", "impact"):  _solid_impact,
    ("wood", "impact"):   _solid_impact,
    ("rock", "roll"):     _solid_roll,
    ("metal", "roll"):    _solid_roll,
    ("rock", "scrape"):   _scrape,
    ("metal", "scrape"):  _scrape,
    ("wood", "scrape"):   _scrape,
    ("fabric", "drag"):   _scrape,        # drag tratado como scrape mas suave
    ("rock", "drag"):     _scrape,
    # liquidos
    ("liquid", "drip"):   _drip,
    ("liquid", "splash"): _splash,
    ("liquid", "pour"):   _pour,
    ("liquid", "roll"):   _rolling_droplet,        # GOTA RODANTE - imposible estrella
    ("liquid", "impact"): _splash,
    # granular
    ("gravel", "step"):   _granular_step,
    ("gravel", "roll"):   _granular_step,
    ("gravel", "scrape"): _granular_step,
    ("gravel", "pour"):   _granular_step,
    ("earth", "step"):    _granular_step,
}


def compose(spec: CompositionSpec, sr: int = 44_100) -> np.ndarray:
    """Compone una escena a partir de la especificacion."""
    key = (spec.material, spec.interaction)
    if key not in GENERATORS:
        # Fallback razonable: si no hay generador para esa combo, usar el
        # generador del material con la interaccion "principal" segun
        # afinidad fisica
        fallback_int = {
            "liquid": "splash", "gravel": "step", "fabric": "drag",
            "rock": "impact", "metal": "impact", "wood": "impact",
        }.get(spec.material, "impact")
        spec = CompositionSpec(**{**spec.__dict__, "interaction": fallback_int})
        key = (spec.material, spec.interaction)
        if key not in GENERATORS:
            raise ValueError(f"No hay generador para {spec.material} x {spec.interaction}")
    gen = GENERATORS[key]
    return gen(spec, sr)


def compose_impossible(
    base_material: str,
    base_interaction: str,
    overlay_material: str | None = None,
    overlay_interaction: str | None = None,
    overlay_weight: float = 0.5,
    modifiers: dict[str, float] | None = None,
    duration_s: float = 5.0,
    seed: int = 0,
    sr: int = 44_100,
) -> np.ndarray:
    """Compone un sonido IMPOSIBLE: capa base + overlay con peso.

    Ejemplos:
        # Gota rodante = liquid x roll directo (un solo generador)
        compose_impossible("liquid", "roll", modifiers={"continuity": 0.7})

        # Roca liquida = impacto modal de roca + capa de splash
        compose_impossible("rock", "impact", "liquid", "splash", overlay_weight=0.6,
                           modifiers={"wetness": 0.8, "resonance": 0.4})

        # Grava mojada raspando = granular gravel scrape + capa wet
        compose_impossible("gravel", "scrape", "liquid", "pour", overlay_weight=0.3,
                           modifiers={"wetness": 0.6, "granularity": 0.9})
    """
    mods = modifiers or {}
    base_spec = CompositionSpec(
        material=base_material, interaction=base_interaction,
        duration_s=duration_s, seed=seed, **mods,
    )
    base = compose(base_spec, sr)
    if overlay_material is None:
        return base
    overlay_spec = CompositionSpec(
        material=overlay_material, interaction=overlay_interaction or base_interaction,
        duration_s=duration_s, seed=seed + 1, **mods,
    )
    overlay = compose(overlay_spec, sr)
    n = min(len(base), len(overlay))
    out = base[:n] + overlay_weight * overlay[:n]
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
