"""Interfaz comun para las cuatro propuestas de mezcla imposible.

Si TODOS los metodos respetan esta firma, el evaluador
(scripts/04_compute_metrics.py) no cambia entre metodos.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class HybridSpec:
    """Especificacion de un sonido imposible objetivo.

    `weights` controla la importancia relativa de cada componente
    al combinarlas en el latente (alpha, beta, gamma del plan).
    """
    target_material: str
    target_interaction: str
    properties: dict[str, float] = field(default_factory=dict)  # wetness, rigidity, ...
    weights: dict[str, float] = field(default_factory=lambda: {
        "material": 0.5,
        "interaction": 0.5,
        "modifier": 0.5,
    })


class HybridMethod(ABC):
    """Interfaz de las cuatro propuestas. Todas devuelven wav 1-D float."""

    name: str

    @abstractmethod
    def encode(self, wav: "Tensor") -> "Tensor":
        """wav (samples,) -> z (latent_dim,) en el espacio del encoder."""

    @abstractmethod
    def decode(self, z: "Tensor") -> "Tensor":
        """z (latent_dim,) -> wav (samples,)."""

    @abstractmethod
    def generate(self, anchor_z: "Tensor", spec: HybridSpec) -> "Tensor":
        """Produce un wav hibrido dado un latente ancla y una HybridSpec."""
