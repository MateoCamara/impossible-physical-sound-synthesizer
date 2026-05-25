"""Dataset PyTorch para cargar embeddings cacheados + metadata de etiquetado.

No carga audio: el corpus de audio ya esta en data/processed/<clip_id>.wav,
y los embeddings precalculados estan en data/embeddings/encodec.pt como
{means: dict[clip_id, Tensor(dim,)], seqs: dict[clip_id, Tensor(dim, T')]}.

Lo que devuelve este Dataset es el par (embedding, etiquetas) listo para
entrenar las cabezas del metodo B y para construir centroides del metodo A.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from impossible_mix.config import EMBEDDINGS_DIR, LABELS_CSV
from impossible_mix.data.labels import MODIFIERS, Interaction, Material


@dataclass
class CorpusBundle:
    """Tensores agregados para todos los clips del corpus."""
    clip_ids: list[str]
    Z_mean: torch.Tensor          # (N, dim) embeddings medios
    Z_seq: list[torch.Tensor]     # lista de (dim, T') de cada clip
    materials: list[str]
    interactions: list[str]
    modifiers: torch.Tensor       # (N, 5)
    sources: list[str]

    @property
    def n(self) -> int:
        return len(self.clip_ids)


def load_corpus(
    labels_csv: Path = LABELS_CSV,
    embeddings_pt: Path = EMBEDDINGS_DIR / "encodec.pt",
) -> CorpusBundle:
    df = pd.read_csv(labels_csv)
    payload = torch.load(embeddings_pt, map_location="cpu", weights_only=False)
    means: dict[str, torch.Tensor] = payload["means"]
    seqs: dict[str, torch.Tensor] = payload["seqs"]

    keep = df["clip_id"].astype(str).isin(means.keys())
    df = df[keep].reset_index(drop=True)
    clip_ids = df["clip_id"].astype(str).tolist()

    Z_mean = torch.stack([means[c] for c in clip_ids], dim=0)
    Z_seq = [seqs[c] for c in clip_ids]  # lista de (dim, T') con T' = 375 en EnCodec 24kHz/5s
    mod_tensor = torch.tensor(df[list(MODIFIERS)].values, dtype=torch.float32)

    return CorpusBundle(
        clip_ids=clip_ids,
        Z_mean=Z_mean,
        Z_seq=Z_seq,
        materials=df["material"].tolist(),
        interactions=df["interaction"].tolist(),
        modifiers=mod_tensor,
        sources=df["source"].tolist(),
    )


def seq_lookup_by_id(bundle: CorpusBundle) -> dict[str, torch.Tensor]:
    """Dict {clip_id: (dim, T')}."""
    return {cid: seq for cid, seq in zip(bundle.clip_ids, bundle.Z_seq)}


class EmbeddingClassificationDataset(Dataset):
    """Para entrenar las cabezas de B sobre Z_mean."""

    def __init__(self, bundle: CorpusBundle) -> None:
        self.bundle = bundle
        self.material_to_idx = {m.value: i for i, m in enumerate(Material)}
        self.interaction_to_idx = {x.value: i for i, x in enumerate(Interaction)}

    def __len__(self) -> int:
        return self.bundle.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "z": self.bundle.Z_mean[idx],
            "material_y": torch.tensor(self.material_to_idx[self.bundle.materials[idx]]),
            "interaction_y": torch.tensor(self.interaction_to_idx[self.bundle.interactions[idx]]),
            "modifiers_y": self.bundle.modifiers[idx],
        }
