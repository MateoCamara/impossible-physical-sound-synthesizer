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
    pan: float = 0.0          # -1 = L, 0 = center, +1 = R
    distance_m: float = 0.0   # 0 = sin procesado espacial; >0 aplica air_absorption + atten


@dataclass
class Sequence:
    duration_s: float = 8.0
    sr: int = SAMPLE_RATE
    events: list[Event] = field(default_factory=list)

    def add_at(self, t_s: float, wav: np.ndarray, gain: float = 1.0,
               pan: float = 0.0, distance_m: float = 0.0) -> "Sequence":
        self.events.append(Event(start_s=t_s, wav=wav.astype(np.float32),
                                  gain=gain, pan=pan, distance_m=distance_m))
        return self

    def render(self, headroom_db: float = -3.0, stereo: bool | None = None) -> np.ndarray:
        """Render. Si stereo=True o si algun evento tiene pan!=0/distance>0,
        devuelve (T, 2). Si no, mono (T,) como antes (backwards compat).
        """
        if stereo is None:
            stereo = any(abs(e.pan) > 1e-6 or e.distance_m > 0.0 for e in self.events)
        n_total = int(self.duration_s * self.sr)
        if not stereo:
            out = np.zeros(n_total, dtype=np.float32)
            for e in self.events:
                start = int(e.start_s * self.sr)
                if start >= n_total:
                    continue
                end = min(n_total, start + len(e.wav))
                out[start:end] += e.gain * e.wav[: end - start]
            target = 10 ** (headroom_db / 20)
            peak = float(np.max(np.abs(out)) + 1e-9)
            if peak > target:
                out = out * (target / peak)
            return out.astype(np.float32)

        # Stereo render con paneo + distancia per-event
        from impossible_mix.physics.spatial import place_source, equal_power_pan
        out = np.zeros((n_total, 2), dtype=np.float32)
        for e in self.events:
            start = int(e.start_s * self.sr)
            if start >= n_total:
                continue
            # Si distance > 0, aplicamos pipeline espacial completo (sale (T,2))
            if e.distance_m > 0:
                stereo_evt = place_source(e.wav, self.sr, distance_m=e.distance_m, pan=e.pan)
            else:
                stereo_evt = equal_power_pan(e.wav, e.pan)
            end = min(n_total, start + len(stereo_evt))
            out[start:end] += e.gain * stereo_evt[: end - start]
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


# ====================================================================
# DSL textual: parsear una receta en string a Sequence
# ====================================================================

import re

_DSL_LINE_RE = re.compile(
    r"^\s*(?P<func>\w+)\s*\((?P<args>.*?)\)\s*@\s*(?P<t>[\d.]+)\s*s?"
    r"(?:\s+(?P<extras>.+))?\s*$"
)
_KV_RE = re.compile(r"(\w+)\s*=\s*([^,\s]+)")


def _parse_args(args_str: str) -> dict[str, float | str]:
    """Parsea 'radius_mm=2, viscosity=0.3' -> {'radius_mm': 2.0, 'viscosity': 0.3}.

    Acepta float o string sin comillas (e.g. material=rock).
    """
    out: dict[str, float | str] = {}
    for m in _KV_RE.finditer(args_str):
        key = m.group(1)
        val = m.group(2)
        try:
            out[key] = float(val)
        except ValueError:
            out[key] = val
    return out


def _parse_extras(extras_str: str | None) -> dict[str, float]:
    """Parsea 'gain=0.9 pan=-0.5 distance_m=1.5' -> dict."""
    if not extras_str:
        return {}
    out: dict[str, float] = {}
    for m in _KV_RE.finditer(extras_str):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


# Despachadores por nombre de funcion DSL
_DSL_DISPATCH: dict[str, "callable"] = {}


def _dsl_register(name: str):
    """Decorador para registrar un dispatcher DSL."""
    def deco(fn):
        _DSL_DISPATCH[name] = fn
        return fn
    return deco


@_dsl_register("drip")
def _dsl_drip(args: dict, sr: int) -> np.ndarray:
    return evt_drip(
        radius_mm=float(args.get("radius_mm", 2.0)),
        viscosity=float(args.get("viscosity", 0.0)),
        surface_hardness=float(args.get("surface_hardness", 0.5)),
        seed=int(args.get("seed", 0)),
        sr=sr,
    )


@_dsl_register("roll")
def _dsl_roll(args: dict, sr: int) -> np.ndarray:
    return evt_rolling(
        duration_s=float(args.get("duration_s", 3.0)),
        radius_mm=float(args.get("radius_mm", 2.0)),
        viscosity=float(args.get("viscosity", 0.0)),
        surface_hardness=float(args.get("surface_hardness", 0.5)),
        roll_velocity_hz=float(args.get("roll_velocity_hz", 14.0)),
        path_roughness=float(args.get("path_roughness", 0.35)),
        seed=int(args.get("seed", 0)),
        sr=sr,
    )


@_dsl_register("splash")
def _dsl_splash(args: dict, sr: int) -> np.ndarray:
    return evt_splash(
        intensity=float(args.get("intensity", 0.7)),
        n_bubbles=int(args.get("n_bubbles", 30)),
        seed=int(args.get("seed", 0)),
        sr=sr,
    )


@_dsl_register("impact")
def _dsl_impact(args: dict, sr: int) -> np.ndarray:
    mat = args.get("material", "rock")
    if not isinstance(mat, str):
        mat = "rock"
    return evt_impact(
        material=mat,
        rigidity=float(args.get("rigidity", 0.5)),
        resonance=float(args.get("resonance", 0.4)),
        seed=int(args.get("seed", 0)),
        sr=sr,
    )


@_dsl_register("pour")
def _dsl_pour(args: dict, sr: int) -> np.ndarray:
    return evt_pour(
        flow_rate=float(args.get("flow_rate", 0.6)),
        viscosity=float(args.get("viscosity", 0.1)),
        duration_s=float(args.get("duration_s", 2.0)),
        seed=int(args.get("seed", 0)),
        sr=sr,
    )


def parse_dsl(dsl_string: str, total_duration_s: float | None = None,
              sr: int = SAMPLE_RATE) -> Sequence:
    """Parsea una receta de texto y construye una Sequence lista para render.

    Sintaxis de cada linea (una linea = un evento):

        <func>(arg1=val1, arg2=val2, ...) @ <time>s [extra=val ...]

    Donde:
      - func ∈ {drip, roll, splash, impact, pour}
      - args son keyword (float o string sin comillas)
      - time es float en segundos
      - extras (opcional): gain, pan, distance_m

    Lineas vacias y las que empiezan con # se ignoran. Si total_duration_s
    es None, se infiere como el tiempo del ultimo evento + 2 s de margen.

    Ejemplo:
        drip(radius_mm=2) @ 0.3s gain=0.9 pan=-0.5
        roll(duration_s=3, surface_hardness=0.6) @ 1.0s
        splash(intensity=0.7) @ 4.5s pan=0.4 distance_m=2.0
        impact(material=rock, rigidity=0.8) @ 3.0s
    """
    events_raw: list[tuple[str, dict, float, dict]] = []
    for line_no, raw_line in enumerate(dsl_string.strip().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _DSL_LINE_RE.match(line)
        if not m:
            raise ValueError(f"DSL parse error (line {line_no}): {line!r}")
        func = m.group("func")
        if func not in _DSL_DISPATCH:
            raise ValueError(f"DSL unknown function (line {line_no}): {func!r}. "
                             f"Available: {sorted(_DSL_DISPATCH)}")
        args = _parse_args(m.group("args") or "")
        t = float(m.group("t"))
        extras = _parse_extras(m.group("extras"))
        events_raw.append((func, args, t, extras))

    if not events_raw:
        raise ValueError("DSL has no events to render")

    if total_duration_s is None:
        last_t = max(e[2] for e in events_raw)
        total_duration_s = last_t + 2.0

    seq = Sequence(duration_s=total_duration_s, sr=sr)
    for func, args, t, extras in events_raw:
        wav = _DSL_DISPATCH[func](args, sr)
        seq.add_at(
            t, wav,
            gain=float(extras.get("gain", 1.0)),
            pan=float(extras.get("pan", 0.0)),
            distance_m=float(extras.get("distance_m", 0.0)),
        )
    return seq


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
