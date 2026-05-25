"""Utilidad compartida por metodos A y por la evaluacion: calculo de
centroides en el espacio latente y direcciones como diferencias.

Funciones:
  centroid(Z, mask)               -> (dim,)
  build_centroids_by_label(Z, lbl)-> dict[label, (dim,)]
  direction(z_from, z_to)         -> z_to - z_from  (sin normalizar)
  unit_direction(z_from, z_to)    -> direccion normalizada L2
  pairwise_distances(Z, query)    -> (N,) distancias a una sola query
"""
from __future__ import annotations

import torch
from torch import Tensor


def centroid(Z: Tensor, mask: Tensor) -> Tensor:
    """Z: (N, dim). mask: (N,) bool. Devuelve (dim,)."""
    if mask.sum() == 0:
        raise ValueError("mask con cero elementos: no se puede calcular centroide.")
    return Z[mask].mean(dim=0)


def build_centroids_by_label(
    Z: Tensor, labels: list[str]
) -> dict[str, Tensor]:
    """Devuelve {label: centroide} para todas las labels presentes."""
    out: dict[str, Tensor] = {}
    unique = sorted(set(labels))
    labels_t = labels  # mantener orden
    for lbl in unique:
        idx = torch.tensor([i for i, x in enumerate(labels_t) if x == lbl])
        if idx.numel() == 0:
            continue
        out[lbl] = Z.index_select(0, idx).mean(dim=0)
    return out


def direction(z_from: Tensor, z_to: Tensor) -> Tensor:
    return z_to - z_from


def unit_direction(z_from: Tensor, z_to: Tensor) -> Tensor:
    d = z_to - z_from
    n = d.norm() + 1e-12
    return d / n


def pairwise_distances(Z: Tensor, query: Tensor) -> Tensor:
    """Z: (N, dim), query: (dim,). Devuelve distancias euclideas (N,)."""
    return (Z - query.unsqueeze(0)).norm(dim=-1)


def cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0) if a.dim() == 1 else a,
        b.unsqueeze(0) if b.dim() == 1 else b,
    )
