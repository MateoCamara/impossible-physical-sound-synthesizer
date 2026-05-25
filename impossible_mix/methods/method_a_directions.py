"""Metodo A: direcciones latentes material-propiedad.

Sin reentrenar el generador. Sobre embeddings de un encoder congelado,
calcula centroides por (material, interaccion, modificador) y construye
direcciones como diferencias entre centroides:

    d_material      = mu(target_material)      - mu(anchor_material)
    d_interaction   = mu(target_interaction)   - mu(anchor_interaction)
    d_wetness_high  = mu(wetness>=4)           - mu(wetness<=2)
    ...

Genera el latente hibrido:

    z_hibrido = z_anchor
                + alpha * d_material
                + beta  * d_interaction
                + gamma_w * d_wetness_high
                + gamma_g * d_granularity_high
                + ...

El usuario decide el ancla (un clip concreto) o usa el centroide de la
clase ancla como ancla canonica.

Baselines incluidos en el modulo:
  - linear_interp:    z = (1-t) * z_a + t * z_b   (interpolacion convexa)
  - additive_sum:     z = z_a + z_b               (suma asimetrica clasica)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor

from impossible_mix.data.dataset import CorpusBundle
from impossible_mix.data.labels import MODIFIERS
from impossible_mix.methods.base import HybridMethod, HybridSpec
from impossible_mix.methods.centroids import (
    build_centroids_by_label,
    centroid,
)


@dataclass
class DirectionBank:
    """Diccionarios precomputados de centroides y direcciones binarias
    para cada modificador continuo (high vs low respecto a Likert 1-5).
    """
    material_centroids: dict[str, Tensor] = field(default_factory=dict)
    interaction_centroids: dict[str, Tensor] = field(default_factory=dict)
    modifier_high: dict[str, Tensor] = field(default_factory=dict)
    modifier_low: dict[str, Tensor] = field(default_factory=dict)

    def direction_material(self, src: str, tgt: str) -> Tensor:
        return self.material_centroids[tgt] - self.material_centroids[src]

    def direction_interaction(self, src: str, tgt: str) -> Tensor:
        return self.interaction_centroids[tgt] - self.interaction_centroids[src]

    def direction_modifier(self, mod: str) -> Tensor:
        return self.modifier_high[mod] - self.modifier_low[mod]


def build_direction_bank(bundle: CorpusBundle, mod_threshold: float = 4.0) -> DirectionBank:
    """Calcula todos los centroides y direcciones binarias por modificador."""
    mat_cents = build_centroids_by_label(bundle.Z_mean, bundle.materials)
    int_cents = build_centroids_by_label(bundle.Z_mean, bundle.interactions)
    mods_high: dict[str, Tensor] = {}
    mods_low: dict[str, Tensor] = {}
    for i, mod in enumerate(MODIFIERS):
        vals = bundle.modifiers[:, i]
        high_mask = vals >= mod_threshold
        low_mask = vals <= (6 - mod_threshold)  # complementario en Likert 1-5
        if high_mask.sum() == 0 or low_mask.sum() == 0:
            # No hay ejemplos suficientes para ese modificador; lo omitimos.
            continue
        mods_high[mod] = centroid(bundle.Z_mean, high_mask)
        mods_low[mod] = centroid(bundle.Z_mean, low_mask)
    return DirectionBank(
        material_centroids=mat_cents,
        interaction_centroids=int_cents,
        modifier_high=mods_high,
        modifier_low=mods_low,
    )


def compose_hybrid_mean(
    z_anchor_mean: Tensor,
    bank: DirectionBank,
    src_material: str,
    src_interaction: str,
    tgt_material: str,
    tgt_interaction: str,
    modifier_weights: dict[str, float] | None = None,
    alpha: float = 0.7,
    beta: float = 0.7,
) -> Tensor:
    """Aplica direcciones sobre la MEDIA temporal (dim,). Pierde dinamica."""
    z = z_anchor_mean.clone()
    if src_material != tgt_material:
        z = z + alpha * bank.direction_material(src_material, tgt_material)
    if src_interaction != tgt_interaction:
        z = z + beta * bank.direction_interaction(src_interaction, tgt_interaction)
    if modifier_weights:
        for mod, w in modifier_weights.items():
            if w == 0 or mod not in bank.modifier_high:
                continue
            z = z + w * bank.direction_modifier(mod)
    return z


def compose_hybrid_seq(
    z_anchor_seq: Tensor,
    bank: DirectionBank,
    src_material: str,
    src_interaction: str,
    tgt_material: str,
    tgt_interaction: str,
    modifier_weights: dict[str, float] | None = None,
    alpha: float = 0.7,
    beta: float = 0.7,
) -> Tensor:
    """Aplica direcciones a cada FRAME del latente (dim, T').
    Esto preserva la dinamica temporal del ancla y le suma el offset
    direccional uniformemente. Es el modo correcto para EnCodec/RAVE.

    z_anchor_seq: (dim, T')
    return: (dim, T')
    """
    z = z_anchor_seq.clone()
    if src_material != tgt_material:
        d = bank.direction_material(src_material, tgt_material).unsqueeze(-1)
        z = z + alpha * d
    if src_interaction != tgt_interaction:
        d = bank.direction_interaction(src_interaction, tgt_interaction).unsqueeze(-1)
        z = z + beta * d
    if modifier_weights:
        for mod, w in modifier_weights.items():
            if w == 0 or mod not in bank.modifier_high:
                continue
            d = bank.direction_modifier(mod).unsqueeze(-1)
            z = z + w * d
    return z


# --- Baselines obligatorios para comparar ---------------------------------

def baseline_linear_interp(z_a: Tensor, z_b: Tensor, t: float = 0.5) -> Tensor:
    return (1 - t) * z_a + t * z_b


def baseline_additive_sum(zs: Iterable[Tensor]) -> Tensor:
    out = None
    for z in zs:
        out = z if out is None else out + z
    return out


# --- Wrapper que respeta la interfaz HybridMethod -------------------------

class MethodA(HybridMethod):
    """Implementacion de la interfaz comun para integracion en el evaluador.

    A diferencia del primer prototipo, ahora `generate` recibe el latente
    completo del ancla (dim, T'), no la media, para preservar dinamica.
    Si se pasa un tensor 1D, se hace tiling pero se avisa en stderr.
    """

    name = "A_directions"

    def __init__(self, encoder, bundle: CorpusBundle, bank: DirectionBank | None = None) -> None:
        self.encoder = encoder
        self.bundle = bundle
        self.bank = bank or build_direction_bank(bundle)

    def encode(self, wav: Tensor) -> Tensor:
        return self.encoder.encode_sequence(wav)

    def decode(self, z: Tensor, n_frames: int | None = None) -> Tensor:
        if z.dim() == 1:
            n = n_frames or self.encoder.expected_frames(5.0)
            return self.encoder.decode_from_mean(z, n)
        return self.encoder.decode_sequence(z)

    def generate(self, anchor_z: Tensor, spec: HybridSpec) -> Tensor:
        """anchor_z se acepta como (dim,) o (dim, T'). Lo segundo preserva dinamica."""
        weights = spec.weights or {}
        alpha = float(weights.get("material", 0.7))
        beta = float(weights.get("interaction", 0.7))
        gamma_mods = {
            mod: float(weights.get(mod, spec.properties.get(mod, 0.0)))
            for mod in MODIFIERS
        }
        src_material = str(spec.properties.get("src_material", spec.target_material))
        src_interaction = str(spec.properties.get("src_interaction", spec.target_interaction))

        if anchor_z.dim() == 1:
            # Compatibilidad legacy: aplicar a media y tile -> baja calidad
            z_h = compose_hybrid_mean(
                anchor_z, self.bank, src_material, src_interaction,
                spec.target_material, spec.target_interaction,
                modifier_weights=gamma_mods, alpha=alpha, beta=beta,
            )
            return self.decode(z_h, n_frames=self.encoder.expected_frames(5.0))

        # Caso correcto: anchor_z es (dim, T')
        z_h = compose_hybrid_seq(
            anchor_z, self.bank, src_material, src_interaction,
            spec.target_material, spec.target_interaction,
            modifier_weights=gamma_mods, alpha=alpha, beta=beta,
        )
        return self.encoder.decode_sequence(z_h)
