"""Pack de gotas tematicas: cada una con parametros fisicos calibrados a
su material. Algunas son fisicamente posibles (water, oil), otras son
imposibles (mercury rolling, lava drop) — el mismo motor las cubre todas.
"""
from __future__ import annotations

from impossible_mix.physics.droplet import DropletParams


# Cada preset es un (DropletParams base, descripcion).
# Ajustar parametros segun fisica del fluido / material:
#   - water: Minnaert ~ 1600 Hz para 2 mm, viscosidad baja, surface neutral
#   - oil: viscosidad alta (chirp lento, decay corto), bubble freq similar a agua
#   - milk: viscosidad moderada, dispersa, surface acolchada
#   - mercury: densidad altisima -> chirp mas grave (radius efectivo grande),
#              tension superficial altisima -> rebote (path roughness alto),
#              superficie metalica
#   - lava: viscosidad muy alta, modal grave de burbujas grandes y lentas,
#           surface ardiente
PRESETS: dict[str, dict] = {
    "water": dict(
        droplet_radius_mm=2.0, viscosity=0.0, surface_hardness=0.5,
        roll_velocity_hz=14, path_roughness=0.35,
        desc="Standard water droplet rolling on neutral surface",
    ),
    "oil": dict(
        droplet_radius_mm=2.5, viscosity=0.7, surface_hardness=0.4,
        roll_velocity_hz=10, path_roughness=0.25,
        desc="Olive-oil-like rolling: slower, more muted, less brilliant",
    ),
    "milk": dict(
        droplet_radius_mm=2.2, viscosity=0.4, surface_hardness=0.3,
        roll_velocity_hz=12, path_roughness=0.40,
        desc="Milk: moderately viscous, soft impacts",
    ),
    "mercury": dict(
        droplet_radius_mm=1.0, viscosity=0.2, surface_hardness=0.95,
        roll_velocity_hz=20, path_roughness=0.65,
        desc="Mercury (impossible): metallic surface contact, small dense droplet, high path jitter",
    ),
    "lava": dict(
        droplet_radius_mm=4.0, viscosity=0.9, surface_hardness=0.2,
        roll_velocity_hz=4, path_roughness=0.15,
        desc="Lava droplet (impossible): very viscous, slow, deep bubble resonance",
    ),
    "honey": dict(
        droplet_radius_mm=3.0, viscosity=0.95, surface_hardness=0.3,
        roll_velocity_hz=2, path_roughness=0.1,
        desc="Honey: extreme viscosity, almost no rolling",
    ),
    "water_fast": dict(
        droplet_radius_mm=2.0, viscosity=0.0, surface_hardness=0.6,
        roll_velocity_hz=28, path_roughness=0.6,
        desc="Water droplet rolling fast over rough surface",
    ),
    "tiny_drops_rain": dict(
        droplet_radius_mm=0.7, viscosity=0.0, surface_hardness=0.5,
        roll_velocity_hz=35, path_roughness=0.8,
        desc="Many tiny droplets approaching rain texture",
    ),
}


def get_preset(name: str, duration_s: float = 5.0, seed: int = 42) -> DropletParams:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Available: {list(PRESETS)}")
    p = {k: v for k, v in PRESETS[name].items() if k != "desc"}
    return DropletParams(**p, duration_s=duration_s, seed=seed)
