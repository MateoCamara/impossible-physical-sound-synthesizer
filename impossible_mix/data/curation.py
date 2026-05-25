"""Curacion y normalizacion de clips: resampling, recorte/padding a duracion
fija, normalizacion por peak, conversion a mono.

Politica: nunca tocamos los originales; las versiones procesadas viven en
data/processed/<clip_id>.wav, todas a 44.1 kHz mono, exactamente 5 s, peak=-1 dBFS.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_mono(path: Path, target_sr: int) -> np.ndarray:
    """Carga audio y devuelve mono float32 a target_sr."""
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y.astype(np.float32)


def fit_duration(y: np.ndarray, target_samples: int) -> np.ndarray:
    """Recorta al centro o hace padding hasta target_samples."""
    n = len(y)
    if n == target_samples:
        return y
    if n > target_samples:
        # Centrar el contenido mas energico.
        env = np.abs(y)
        # Maximo de envolvente como ancla; ventana centrada en el.
        peak = int(np.argmax(env))
        half = target_samples // 2
        start = max(0, peak - half)
        end = start + target_samples
        if end > n:
            end = n
            start = end - target_samples
        return y[start:end]
    # padding centrado
    pad = target_samples - n
    left = pad // 2
    right = pad - left
    return np.pad(y, (left, right), mode="constant")


def peak_normalize(y: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(y)) + 1e-12)
    target_amp = 10 ** (target_dbfs / 20)
    return (y / peak * target_amp).astype(np.float32)


def save_wav(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")


def is_too_quiet(y: np.ndarray, min_rms_dbfs: float = -50.0) -> bool:
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    rms_dbfs = 20 * np.log10(rms + 1e-12)
    return rms_dbfs < min_rms_dbfs
