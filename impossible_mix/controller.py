"""Controlador interactivo de propiedades: knobs continuos sobre el latente.

API publica:
    ctrl = PropertyController()
    ctrl.list_knobs()                       # ver propiedades disponibles
    ctrl.set_anchor("fs365161")             # o pasar Tensor directamente
    z, wav = ctrl.apply({"liquid": 0.6, "wetness": 0.4})
    sweep = ctrl.sweep("liquid", [0, 0.3, 0.6, 0.9, 1.2])

Internamente reutiliza:
  - encoder (EnCodec)
  - DirectionBank del Metodo A
  - compose_hybrid_seq con direcciones combinadas
  - Postproceso D opcional para modificadores que no esten bien
    capturados por las direcciones latentes (wetness, granularity)

El usuario describe lo que quiere en LENGUAJE DE PROPIEDADES, no de
parametros tecnicos. El controlador traduce.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor

from impossible_mix.config import SAMPLE_RATE
from impossible_mix.data.dataset import CorpusBundle, load_corpus
from impossible_mix.encoders.rave_wrapper import EncoderWrapper, get_encoder
from impossible_mix.metrics.quality import quality_verdict
from impossible_mix.methods.method_a_directions import (
    DirectionBank,
    build_direction_bank,
)
from impossible_mix.methods.method_d_dsp_layer import DSPParams, apply_dsp


# Modificadores que pueden aplicarse via DSP postproceso (no via direccion latente)
# como complemento opcional. Las direcciones latentes manejan la semantica,
# DSP maneja la textura espectral fina.
DSP_MODIFIER_KEYS = {"wetness", "rigidity", "resonance", "granularity", "continuity"}


@dataclass
class SweepStep:
    """Un punto en un sweep de propiedad."""
    amount: float
    z_seq: Tensor
    wav: Tensor          # mono, sr = encoder.sr_expected
    verdict: str
    rms_db: float
    dyn_db: float


@dataclass
class PropertyController:
    """Capa de alto nivel sobre direcciones latentes + DSP.

    Construye una sola vez los centroides/direcciones y las cachea.
    """
    encoder: EncoderWrapper = field(default=None)
    bundle: CorpusBundle | None = None
    bank: DirectionBank | None = None
    _anchor_seq: Tensor | None = None
    _anchor_id: str | None = None

    def __post_init__(self) -> None:
        if self.encoder is None:
            self.encoder = get_encoder("encodec")
        if self.bundle is None:
            self.bundle = load_corpus()
            # Fusionar step->impact para coherencia con cabezas y direcciones
            self.bundle.interactions = [
                "impact" if x == "step" else x for x in self.bundle.interactions
            ]
        if self.bank is None:
            self.bank = build_direction_bank(self.bundle)

    # ---------- Anclas ----------

    def set_anchor(self, anchor: str | Path | Tensor) -> "PropertyController":
        """Acepta clip_id del corpus, ruta a wav, o tensor (dim, T')."""
        if isinstance(anchor, Tensor):
            self._anchor_seq = anchor
            self._anchor_id = "tensor"
            return self
        if isinstance(anchor, (str, Path)) and Path(str(anchor)).exists():
            y, sr = sf.read(str(anchor), dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            self._anchor_seq = self.encoder.encode_sequence(torch.from_numpy(y), sr_in=sr)
            self._anchor_id = Path(str(anchor)).stem
            return self
        # Asumir clip_id del corpus
        if anchor in self.bundle.clip_ids:
            idx = self.bundle.clip_ids.index(anchor)
            self._anchor_seq = self.bundle.Z_seq[idx]
            self._anchor_id = str(anchor)
            return self
        raise ValueError(f"Ancla no encontrada: {anchor}")

    # ---------- Inventario de knobs ----------

    def list_knobs(self) -> dict[str, list[str]]:
        """Devuelve dict {categoria: [propiedades]}."""
        return {
            "material":   list(self.bank.material_centroids.keys()),
            "interaction": list(self.bank.interaction_centroids.keys()),
            "modifier":   list(self.bank.modifier_high.keys()),  # via direccion latente
            "dsp":        sorted(DSP_MODIFIER_KEYS),               # via DSP postproceso
        }

    # ---------- Aplicar knobs ----------

    # Magnitud objetivo de la direccion, expresada como fraccion de la norma
    # POR FRAME del latente del ancla. Con 0.1, amount=1 mueve el latente
    # un ~10% del ancla por frame, lo que ya es audible (vs 0.2% con la
    # version no normalizada).
    direction_strength: float = 0.15

    def _resolve_direction(self, prop: str, src: str | None = None) -> Tensor | None:
        """Devuelve direccion latente (dim,) NORMALIZADA a magnitud audible.

        Sin normalizar, las diferencias entre centroides son ~2-5 sobre
        una norma de anchor ~975 (= 0.2%, inaudible). Renormalizamos cada
        direccion a `direction_strength * |anchor_per_frame|` para que
        `amount=1` produzca un cambio perceptual obvio.
        """
        raw: Tensor | None = None
        if prop in self.bank.material_centroids:
            if src and src in self.bank.material_centroids:
                raw = self.bank.direction_material(src, prop)
            else:
                global_mean = torch.stack(list(self.bank.material_centroids.values())).mean(0)
                raw = self.bank.material_centroids[prop] - global_mean
        elif prop in self.bank.interaction_centroids:
            if src and src in self.bank.interaction_centroids:
                raw = self.bank.direction_interaction(src, prop)
            else:
                global_mean = torch.stack(list(self.bank.interaction_centroids.values())).mean(0)
                raw = self.bank.interaction_centroids[prop] - global_mean
        elif prop in self.bank.modifier_high:
            raw = self.bank.direction_modifier(prop)

        if raw is None:
            return None
        # Normalizar a magnitud relativa al ancla por frame
        if self._anchor_seq is None:
            # Sin ancla aun: norma global del bundle
            return raw / (raw.norm() + 1e-9) * self.direction_strength * 50.0  # fallback
        anchor_norm_per_frame = self._anchor_seq.norm() / (self._anchor_seq.shape[-1] ** 0.5)
        target_mag = self.direction_strength * anchor_norm_per_frame
        return raw / (raw.norm() + 1e-9) * target_mag

    @torch.no_grad()
    def apply(
        self,
        knobs: dict[str, float],
        src_material: str | None = None,
        src_interaction: str | None = None,
        dsp_postproc: dict[str, float] | None = None,
    ) -> tuple[Tensor, Tensor, dict]:
        """Aplica varios knobs al anchor actual.

        knobs: {propiedad: amount} donde propiedad puede ser:
            - material: "liquid", "rock", "gravel", ...
            - interaction: "drip", "roll", "scrape", ...
            - modifier latente: "wetness", "granularity", ...
        dsp_postproc: {wetness, granularity, ...} -> aplicado en 44.1 kHz tras decode
                       Si esta vacio se infiere de knobs si hay coincidencia DSP.

        Returns: (z_seq_modificado, wav_44.1kHz, info_dict)
        """
        assert self._anchor_seq is not None, "Llama a set_anchor primero"
        z = self._anchor_seq.clone()
        info: dict = {"applied_knobs": {}, "skipped": []}
        for prop, amount in knobs.items():
            d = self._resolve_direction(prop, src=src_material if prop in self.bank.material_centroids else (src_interaction if prop in self.bank.interaction_centroids else None))
            if d is None:
                info["skipped"].append(prop)
                continue
            z = z + float(amount) * d.unsqueeze(-1)
            info["applied_knobs"][prop] = amount

        # Decode 24 kHz -> resample a 44.1
        wav24 = self.encoder.decode_sequence(z)
        wav = torchaudio.functional.resample(wav24.unsqueeze(0), self.encoder.sr_expected, SAMPLE_RATE).squeeze(0)

        # DSP postproceso opcional (modificadores fisicos)
        if dsp_postproc is None:
            dsp_postproc = {k: float(v) for k, v in knobs.items() if k in DSP_MODIFIER_KEYS}
        if dsp_postproc:
            params = DSPParams(
                wetness=float(dsp_postproc.get("wetness", 0)) / 5.0 if dsp_postproc.get("wetness", 0) > 1 else float(dsp_postproc.get("wetness", 0)),
                rigidity=float(dsp_postproc.get("rigidity", 0)),
                resonance=float(dsp_postproc.get("resonance", 0)),
                granularity=float(dsp_postproc.get("granularity", 0)),
                continuity=float(dsp_postproc.get("continuity", 0)),
                sr=SAMPLE_RATE,
            )
            wav_np = wav.numpy()
            wav_np = apply_dsp(wav_np, params)
            wav = torch.from_numpy(wav_np)
            info["dsp_applied"] = dict(wetness=params.wetness, rigidity=params.rigidity,
                                      resonance=params.resonance, granularity=params.granularity)

        return z, wav, info

    # ---------- Sweep ----------

    def sweep(
        self,
        property_name: str,
        amounts: Iterable[float],
        out_dir: Path | None = None,
        dsp_postproc: dict[str, float] | None = None,
        verbose: bool = True,
    ) -> list[SweepStep]:
        """Genera audios variando una sola propiedad por `amounts`.

        Si out_dir se proporciona, guarda los wavs como
            {anchor}_{property}+{amount:.2f}.wav
        """
        assert self._anchor_seq is not None, "Llama a set_anchor primero"
        steps: list[SweepStep] = []
        for amount in amounts:
            z, wav, info = self.apply({property_name: amount}, dsp_postproc=dsp_postproc)
            wav_np = wav.numpy().astype(np.float32)
            q = quality_verdict(wav_np, SAMPLE_RATE)
            step = SweepStep(amount=amount, z_seq=z, wav=wav, verdict=q.verdict,
                             rms_db=q.rms_db, dyn_db=q.dynamic_range_db)
            steps.append(step)
            if verbose:
                msg_dsp = " +DSP" if info.get("dsp_applied") else ""
                print(f"  {property_name}{msg_dsp} = {amount:+.2f}  rms={q.rms_db:6.1f} dyn={q.dynamic_range_db:5.1f}  {q.verdict.upper()}  {q.reasons}")
            if out_dir is not None:
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{self._anchor_id}__{property_name}{amount:+.2f}.wav"
                sf.write(str(out_dir / fname), wav_np, SAMPLE_RATE)
        return steps
