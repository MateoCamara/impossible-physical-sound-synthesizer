"""Reverberacion via IRs sintetizadas y convolucion.

Generamos IRs parametricas a partir de un modelo simple: rafaga de
reflexiones tempranas + cola exponencial decreciente con coloracion
(LPF tilt). Diferentes presets simulan espacios distintos.

API:
    ir = generate_ir(preset="small_room", sr=44100)
    wet = apply_reverb(dry, ir, mix=0.4)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class IRPreset:
    name: str
    t60_s: float                # tiempo de decaimiento global (s)
    pre_delay_ms: float = 10.0
    early_reflections_ms: tuple[float, ...] = (15, 25, 38, 55, 75)
    early_gain: float = 0.6
    color_lpf_hz: float = 6000  # cuanto mas bajo, mas "oscuro"


PRESETS: dict[str, IRPreset] = {
    "dry":          IRPreset("dry",          t60_s=0.05, pre_delay_ms=0,  early_reflections_ms=(), early_gain=0, color_lpf_hz=20000),
    "small_room":   IRPreset("small_room",   t60_s=0.4,  pre_delay_ms=5,  early_reflections_ms=(8,14,22,33,48), early_gain=0.6, color_lpf_hz=8000),
    "medium_hall":  IRPreset("medium_hall",  t60_s=1.3,  pre_delay_ms=20, early_reflections_ms=(28,45,68,95,130), early_gain=0.5, color_lpf_hz=5000),
    "cathedral":    IRPreset("cathedral",    t60_s=4.5,  pre_delay_ms=40, early_reflections_ms=(55,90,140,220,310), early_gain=0.35, color_lpf_hz=3000),
    "cave":         IRPreset("cave",         t60_s=2.8,  pre_delay_ms=25, early_reflections_ms=(40,80,135,200,290), early_gain=0.45, color_lpf_hz=2500),
    "exterior":     IRPreset("exterior",     t60_s=0.2,  pre_delay_ms=2,  early_reflections_ms=(120,250), early_gain=0.15, color_lpf_hz=6000),
}


# Registro de IRs reales (WAV) descargados. Se llena dinamicamente por
# `register_real_ir` o por el script de descarga (scripts/24_download_irs.py).
REAL_IR_PATHS: dict[str, str] = {}


def register_real_ir(name: str, wav_path: str) -> None:
    """Registra una IR real para uso via `get_ir(name)`. wav_path puede ser
    relativo al repo o absoluto."""
    REAL_IR_PATHS[name] = wav_path


def generate_ir(preset: str = "medium_hall", sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Genera una impulse response sintetica para el preset dado."""
    if preset not in PRESETS:
        raise ValueError(f"Preset desconocido: {preset}. Disponibles: {list(PRESETS)}")
    p = PRESETS[preset]
    rng = np.random.default_rng(seed)
    total_n = int((p.pre_delay_ms / 1000 + p.t60_s * 1.5) * sr)
    ir = np.zeros(total_n, dtype=np.float32)
    pre_delay_n = int(p.pre_delay_ms / 1000 * sr)

    # Direct (no incluido, asumimos dry sin direct)
    # Early reflections discretas
    for ms in p.early_reflections_ms:
        idx = pre_delay_n + int(ms / 1000 * sr)
        if 0 <= idx < total_n:
            ir[idx] += p.early_gain * (0.5 + 0.5 * rng.random()) * (1 if rng.random() > 0.3 else -1)

    # Late tail: ruido aleatorio multiplicado por envelope exponencial
    tail_start = pre_delay_n + int(max(p.early_reflections_ms or (10,)) / 1000 * sr)
    tail_n = total_n - tail_start
    if tail_n > 0:
        noise = rng.standard_normal(tail_n).astype(np.float32)
        decay_factor = -6.91 / (p.t60_s * sr + 1e-6)
        envelope = np.exp(decay_factor * np.arange(tail_n)).astype(np.float32)
        ir[tail_start:] += noise * envelope * 0.6

    # Coloracion (LPF para "warmth")
    if p.color_lpf_hz < sr / 2 - 100:
        sos = signal.butter(2, p.color_lpf_hz, btype="low", fs=sr, output="sos")
        ir = signal.sosfiltfilt(sos, ir).astype(np.float32)

    # Normalizar a energia unitaria aprox
    norm = float(np.sqrt(np.sum(ir ** 2)) + 1e-9)
    ir = ir / norm * 0.3
    return ir


def load_ir_from_wav(path: str, sr_target: int = 44_100,
                      normalize: bool = True) -> np.ndarray:
    """Carga una IR desde un WAV, opcionalmente resampleando a sr_target.
    Devuelve mono float32. La normalizacion la deja con peak~0.3 para
    evitar clipping al convolucionar con audio peak-normalized.
    """
    import soundfile as sf
    wav, sr_src = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1).astype(np.float32)
    if sr_src != sr_target:
        try:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr_src, target_sr=sr_target).astype(np.float32)
        except ImportError:
            # Fallback: scipy.signal.resample_poly
            from math import gcd
            g = gcd(sr_src, sr_target)
            up = sr_target // g
            down = sr_src // g
            wav = signal.resample_poly(wav, up, down).astype(np.float32)
    if normalize:
        peak = float(np.max(np.abs(wav)) + 1e-9)
        wav = (wav / peak * 0.3).astype(np.float32)
    return wav


def get_ir(preset: str, sr: int = 44_100, seed: int = 0) -> np.ndarray:
    """Despacha: si `preset` esta en PRESETS, genera IR sintetica.
    Si esta en REAL_IR_PATHS, carga WAV real. Si no, ValueError.
    """
    if preset in PRESETS:
        return generate_ir(preset, sr=sr, seed=seed)
    if preset in REAL_IR_PATHS:
        return load_ir_from_wav(REAL_IR_PATHS[preset], sr_target=sr)
    raise ValueError(f"Preset desconocido: {preset}. "
                     f"Sinteticos: {list(PRESETS)}. Reales: {list(REAL_IR_PATHS)}")


def apply_reverb(dry: np.ndarray, ir: np.ndarray, mix: float = 0.4) -> np.ndarray:
    """Convoluciona dry con ir y mezcla con dry segun mix in [0,1]."""
    if dry.ndim == 2:
        # Estereo: convolucion por canal
        out_l = signal.fftconvolve(dry[:, 0], ir, mode="full")
        out_r = signal.fftconvolve(dry[:, 1], ir, mode="full")
        wet = np.stack([out_l, out_r], axis=-1).astype(np.float32)
        # Asegurar misma longitud que dry
        wet = wet[: len(dry)]
        out = (1 - mix) * dry + mix * wet
    else:
        wet = signal.fftconvolve(dry, ir, mode="full")[: len(dry)].astype(np.float32)
        out = (1 - mix) * dry + mix * wet
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)
