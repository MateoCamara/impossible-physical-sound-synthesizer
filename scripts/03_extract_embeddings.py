"""Extrae embeddings con EnCodec para cada clip procesado en data/processed/
y los guarda como un diccionario {clip_id: tensor} en data/embeddings/encodec.pt.

Tambien guarda el latente medio temporal Y la secuencia completa (por frame)
porque metodos C y D pueden necesitar la dimension temporal.

EnCodec corre en CPU sin problemas (~5-10 ms/clip de 5s en una CPU moderna).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import EMBEDDINGS_DIR, LABELS_CSV, PROCESSED_DIR, SAMPLE_RATE


def load_encodec_24k():
    from encodec import EncodecModel
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=LABELS_CSV)
    ap.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--out", type=Path, default=EMBEDDINGS_DIR / "encodec.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    if args.limit:
        df = df.head(args.limit)
    print(f"Cargando EnCodec 24kHz...")
    model = load_encodec_24k().to(args.device)
    encodec_sr = 24_000
    # Resampleador SAMPLE_RATE -> encodec_sr una vez.
    resampler = torchaudio.transforms.Resample(SAMPLE_RATE, encodec_sr)

    means: dict[str, torch.Tensor] = {}
    seqs: dict[str, torch.Tensor] = {}
    missing = 0
    t0 = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        clip_id = str(row["clip_id"])
        wav_path = args.processed / f"{clip_id}.wav"
        if not wav_path.exists():
            missing += 1
            continue
        y_np, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if y_np.ndim == 2:
            y_np = y_np.mean(axis=1)
        y = torch.from_numpy(y_np).unsqueeze(0)  # (1, T)
        if sr != SAMPLE_RATE:
            y = torchaudio.functional.resample(y, sr, SAMPLE_RATE)
        y24 = resampler(y).unsqueeze(0).to(args.device)  # (1, 1, T)
        with torch.no_grad():
            z = model.encoder(y24)  # (1, dim, T')
        z = z.squeeze(0).cpu()  # (dim, T')
        means[clip_id] = z.mean(dim=-1).float()  # (dim,)
        seqs[clip_id] = z.float()                # (dim, T')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "encodec_24khz_6kbps",
        "latent_dim": next(iter(means.values())).numel() if means else 0,
        "means": means,
        "seqs": seqs,
        "sample_rate_in": SAMPLE_RATE,
        "encodec_sr": encodec_sr,
    }
    torch.save(payload, args.out)
    print(f"\nGuardados {len(means)} embeddings en {args.out}")
    print(f"  dim={payload['latent_dim']}  faltaban={missing}")
    print(f"  tiempo total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
