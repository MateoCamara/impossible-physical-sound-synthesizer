"""Metodo B: cabezas de atributos sobre el latente congelado + edicion
por descenso de gradiente.

Arquitectura:
  - Encoder congelado (EnCodec). No se reentrena.
  - 3 cabezas pequenas (MLP 2 capas):
      head_material   : (dim,) -> logits sobre Material (8 clases)
      head_interaction: (dim,) -> logits sobre Interaction (8 clases)
      head_modifiers  : (dim,) -> 5 floats (regresion 1-5 sobre MODIFIERS)
  - Entrenamiento: solo las cabezas. Encoder NO en el grafo.

Inferencia (edicion):
  Dado z_0 = encoder(x_anchor), optimizar z hacia clase/propiedades objetivo
  con loss combinada:
      L = -log p(y_mat=tgt_material | z)
          -log p(y_int=tgt_interaction | z)
          + ||mods(z) - mods_target||^2
          + lambda_prior * ||z - z_0||^2

  La ultima penalizacion previene deriva fuera del manifold del encoder.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from impossible_mix.data.labels import MODIFIERS, Interaction, Material
from impossible_mix.methods.base import HybridMethod, HybridSpec


MATERIAL_LIST = [m.value for m in Material]
INTERACTION_LIST = [x.value for x in Interaction]
N_MATERIAL = len(MATERIAL_LIST)
N_INTERACTION = len(INTERACTION_LIST)
N_MODIFIERS = len(MODIFIERS)


class MLPHead(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class AttributeHeads:
    """Conjunto de cabezas + optimizador asociado para B."""
    head_material: MLPHead
    head_interaction: MLPHead
    head_modifiers: MLPHead

    def parameters(self):
        for h in (self.head_material, self.head_interaction, self.head_modifiers):
            yield from h.parameters()

    def state_dict(self):
        return {
            "material": self.head_material.state_dict(),
            "interaction": self.head_interaction.state_dict(),
            "modifiers": self.head_modifiers.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.head_material.load_state_dict(sd["material"])
        self.head_interaction.load_state_dict(sd["interaction"])
        self.head_modifiers.load_state_dict(sd["modifiers"])


def build_heads(latent_dim: int, hidden: int = 128) -> AttributeHeads:
    return AttributeHeads(
        head_material=MLPHead(latent_dim, N_MATERIAL, hidden=hidden),
        head_interaction=MLPHead(latent_dim, N_INTERACTION, hidden=hidden),
        head_modifiers=MLPHead(latent_dim, N_MODIFIERS, hidden=hidden),
    )


def train_step(
    heads: AttributeHeads,
    z_batch: Tensor,
    y_material: Tensor,
    y_interaction: Tensor,
    y_modifiers: Tensor,
    weight_material: Tensor | None = None,
    weight_interaction: Tensor | None = None,
) -> dict[str, Tensor]:
    """Devuelve un dict de losses; la suma es la loss total a backprop-ear.

    weight_* (opcional): pesos por clase para CE (inversamente proporcional
    al support para compensar desbalance del corpus).
    """
    logits_m = heads.head_material(z_batch)
    logits_i = heads.head_interaction(z_batch)
    mods_pred = heads.head_modifiers(z_batch)
    loss_m = F.cross_entropy(logits_m, y_material, weight=weight_material)
    loss_i = F.cross_entropy(logits_i, y_interaction, weight=weight_interaction)
    loss_mods = F.mse_loss(mods_pred, y_modifiers)
    return dict(
        loss_material=loss_m,
        loss_interaction=loss_i,
        loss_modifiers=loss_mods,
        loss_total=loss_m + loss_i + 0.5 * loss_mods,
    )


def edit_latent_by_gradient(
    z_init: Tensor,
    heads: AttributeHeads,
    target_material_idx: int | None = None,
    target_interaction_idx: int | None = None,
    target_modifiers: Tensor | None = None,
    n_iters: int = 150,
    lr: float = 5e-3,
    lambda_prior: float = 0.05,
    lambda_mat: float = 1.0,
    lambda_int: float = 1.0,
    lambda_mods: float = 0.5,
) -> Tensor:
    """Optimiza z para acercarlo a la combinacion objetivo manteniendo cercania
    al embedding original (prior anti-deriva).

    Acepta z_init como (dim,) [legacy: edita la media] o (dim, T') [recomendado:
    edita la secuencia y preserva dinamica temporal]. Las cabezas siempre se
    evaluan sobre la MEDIA del z corriente, asi que en secuencia el gradiente
    fluye por la media de todos los frames a la vez.
    """
    z0 = z_init.detach().clone()
    z = z0.clone().requires_grad_(True)
    optim = torch.optim.Adam([z], lr=lr)
    is_seq = z.dim() == 2
    for _ in range(n_iters):
        optim.zero_grad()
        loss = lambda_prior * (z - z0).pow(2).mean()
        # Vector para las cabezas: media temporal si es secuencia
        z_for_heads = z.mean(dim=-1) if is_seq else z
        if target_material_idx is not None:
            logits_m = heads.head_material(z_for_heads.unsqueeze(0))
            loss = loss + lambda_mat * F.cross_entropy(
                logits_m, torch.tensor([target_material_idx])
            )
        if target_interaction_idx is not None:
            logits_i = heads.head_interaction(z_for_heads.unsqueeze(0))
            loss = loss + lambda_int * F.cross_entropy(
                logits_i, torch.tensor([target_interaction_idx])
            )
        if target_modifiers is not None:
            mods_pred = heads.head_modifiers(z_for_heads.unsqueeze(0)).squeeze(0)
            loss = loss + lambda_mods * (mods_pred - target_modifiers).pow(2).mean()
        loss.backward()
        optim.step()
    return z.detach()


class MethodB(HybridMethod):
    """Edicion latente guiada por cabezas de atributos. Acepta latente
    como (dim,) o (dim, T'); este ultimo preserva dinamica temporal."""

    name = "B_heads"

    def __init__(self, encoder, heads: AttributeHeads) -> None:
        self.encoder = encoder
        self.heads = heads
        self._material_to_idx = {m: i for i, m in enumerate(MATERIAL_LIST)}
        self._interaction_to_idx = {x: i for i, x in enumerate(INTERACTION_LIST)}

    def encode(self, wav: Tensor) -> Tensor:
        return self.encoder.encode_sequence(wav)

    def decode(self, z: Tensor) -> Tensor:
        if z.dim() == 1:
            n = self.encoder.expected_frames(5.0)
            return self.encoder.decode_from_mean(z, n)
        return self.encoder.decode_sequence(z)

    def generate(self, anchor_z: Tensor, spec: HybridSpec) -> Tensor:
        tgt_mat = self._material_to_idx[spec.target_material]
        tgt_int = self._interaction_to_idx[spec.target_interaction]
        tgt_mods = None
        if spec.properties:
            tgt_mods = torch.tensor(
                [float(spec.properties.get(m, 3.0)) for m in MODIFIERS],
                dtype=torch.float32,
            )
        z_edited = edit_latent_by_gradient(
            anchor_z, self.heads,
            target_material_idx=tgt_mat,
            target_interaction_idx=tgt_int,
            target_modifiers=tgt_mods,
            n_iters=int(spec.weights.get("n_iters", 150)),
            lr=float(spec.weights.get("lr", 5e-3)),
            lambda_prior=float(spec.weights.get("lambda_prior", 0.05)),
        )
        return self.decode(z_edited)
