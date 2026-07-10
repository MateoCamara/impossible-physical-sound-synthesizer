"""Seeding global reproducible para los scripts de fitting inverso."""
from __future__ import annotations

import random

import numpy as np
import torch

from impossible_mix.config import SEED


def seed_everything(seed: int | None = None) -> int:
    """Siembra random, numpy y torch con `seed` (o `impossible_mix.config.SEED`).

    No llama a `torch.use_deterministic_algorithms` (demasiado invasivo
    para estos scripts: penaliza rendimiento y algunos kernels de FFT
    usados en las losses STFT no tienen variante determinista).
    """
    if seed is None:
        seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
