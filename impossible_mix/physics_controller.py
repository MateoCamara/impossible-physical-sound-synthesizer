"""Controller fisico: knobs con significado fisico directo sobre el composer.

Sustituye al PropertyController basado en EnCodec/direcciones latentes.
Aqui 'mas X' significa modificar un parametro fisico real (radius del
drop, density del grano, hardness del scrape, etc.) y eso GARANTIZA
cambio perceptual coherente.

Uso tipico:
    from impossible_mix.physics_controller import PhysicsController

    ctrl = PhysicsController()
    ctrl.set_scene(material="liquid", interaction="roll")
    wav = ctrl.render(wetness=0.8, granularity=0.4)
    steps = ctrl.sweep("wetness", [0, 0.25, 0.5, 0.75, 1.0])
    steps = ctrl.sweep("granularity", [0, 0.5, 1.0], out_dir="outputs/sweeps")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from impossible_mix.config import SAMPLE_RATE
from impossible_mix.metrics.quality import QualityReport, quality_verdict
from impossible_mix.physics.composer import (
    CompositionSpec,
    compose,
    compose_impossible,
)


@dataclass
class RenderStep:
    knob_name: str
    knob_value: float
    spec: CompositionSpec
    wav: np.ndarray
    quality: QualityReport
    path: Path | None = None


# Knobs exclusivos del controller; estan en correspondencia 1-1 con
# campos de CompositionSpec (mas wetness/granularity como modifiers fisicos
# que el composer ya consume para parametrizar las capas).
KNOBS = ("wetness", "granularity", "rigidity", "resonance", "continuity")


@dataclass
class PhysicsController:
    sr: int = SAMPLE_RATE
    base_material: str = "rock"
    base_interaction: str = "impact"
    overlay_material: str | None = None
    overlay_interaction: str | None = None
    overlay_weight: float = 0.5
    base_modifiers: dict[str, float] = field(default_factory=dict)
    seed: int = 0
    duration_s: float = 5.0

    # ---------- Setters fluent ----------

    def set_scene(
        self,
        material: str,
        interaction: str,
        overlay_material: str | None = None,
        overlay_interaction: str | None = None,
        overlay_weight: float = 0.5,
        **modifiers: float,
    ) -> "PhysicsController":
        self.base_material = material
        self.base_interaction = interaction
        self.overlay_material = overlay_material
        self.overlay_interaction = overlay_interaction
        self.overlay_weight = overlay_weight
        self.base_modifiers = {k: float(v) for k, v in modifiers.items() if k in KNOBS}
        return self

    def set_seed(self, seed: int) -> "PhysicsController":
        self.seed = seed
        return self

    # ---------- Render y sweep ----------

    def render(self, **knob_overrides: float) -> np.ndarray:
        """Genera el audio con la escena actual + knob overrides."""
        mods = {**self.base_modifiers}
        for k, v in knob_overrides.items():
            if k in KNOBS:
                mods[k] = float(v)
        if self.overlay_material is None:
            spec = CompositionSpec(
                material=self.base_material,
                interaction=self.base_interaction,
                duration_s=self.duration_s,
                seed=self.seed,
                **mods,
            )
            return compose(spec, self.sr)
        return compose_impossible(
            base_material=self.base_material,
            base_interaction=self.base_interaction,
            overlay_material=self.overlay_material,
            overlay_interaction=self.overlay_interaction,
            overlay_weight=self.overlay_weight,
            modifiers=mods,
            duration_s=self.duration_s,
            seed=self.seed,
            sr=self.sr,
        )

    def sweep(
        self,
        knob: str,
        values: Iterable[float],
        out_dir: Path | None = None,
        scene_tag: str | None = None,
        verbose: bool = True,
    ) -> list[RenderStep]:
        """Barre un knob por una lista de valores y opcionalmente guarda wavs."""
        if knob not in KNOBS:
            raise ValueError(f"Knob desconocido: {knob}. Disponibles: {KNOBS}")
        steps: list[RenderStep] = []
        scene_tag = scene_tag or self._scene_name()
        for v in values:
            wav = self.render(**{knob: float(v)})
            q = quality_verdict(wav, self.sr)
            step = RenderStep(
                knob_name=knob, knob_value=float(v),
                spec=CompositionSpec(material=self.base_material, interaction=self.base_interaction),
                wav=wav, quality=q,
            )
            if out_dir is not None:
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{scene_tag}__{knob}={v:+.2f}.wav"
                p = out_dir / fname
                sf.write(str(p), wav, self.sr)
                step.path = p
            steps.append(step)
            if verbose:
                print(f"  {scene_tag} {knob}={v:+.2f}  rms={q.rms_db:6.1f} dyn={q.dynamic_range_db:5.1f}  {q.verdict}")
        return steps

    def grid(
        self,
        knobs: dict[str, Iterable[float]],
        out_dir: Path | None = None,
        scene_tag: str | None = None,
    ) -> list[RenderStep]:
        """Barre varias propiedades simultaneas (producto cartesiano).
        Mas costoso, util para figuras heatmap del paper.
        """
        from itertools import product
        names = list(knobs.keys())
        value_lists = [list(knobs[n]) for n in names]
        scene_tag = scene_tag or self._scene_name()
        steps = []
        for combo in product(*value_lists):
            kv = dict(zip(names, combo))
            wav = self.render(**kv)
            q = quality_verdict(wav, self.sr)
            tag = "_".join(f"{k}={v:+.2f}" for k, v in kv.items())
            step = RenderStep(
                knob_name=",".join(names), knob_value=0.0,
                spec=CompositionSpec(material=self.base_material, interaction=self.base_interaction),
                wav=wav, quality=q,
            )
            if out_dir is not None:
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / f"{scene_tag}__{tag}.wav"
                sf.write(str(p), wav, self.sr)
                step.path = p
            steps.append(step)
        return steps

    # ---------- Helpers ----------

    def _scene_name(self) -> str:
        if self.overlay_material:
            return f"{self.base_material}_{self.base_interaction}_x_{self.overlay_material}_{self.overlay_interaction}"
        return f"{self.base_material}_{self.base_interaction}"

    @staticmethod
    def list_scenes() -> dict[str, list[str]]:
        """Lista de escenas disponibles (combinaciones generador-soportadas)."""
        from impossible_mix.physics.composer import GENERATORS
        out: dict[str, list[str]] = {}
        for (m, i) in GENERATORS:
            out.setdefault(m, []).append(i)
        return out
