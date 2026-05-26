"""Time warping del audio: freezer granular + time-stretch sin cambiar pitch.

Dos utilidades:

(1) **Granular freezer**: dado un audio existente (e.g. un drip event de
    400 ms), genera un loop suave de N segundos repitiendo una ventana
    "congelada" del audio con cross-fade Hann entre repeticiones. Util para
    sostener un evento corto indefinidamente (resonancias eternas, drones
    de gota suspendida, etc.).

(2) **Time stretch**: estira o comprime el audio sin cambiar el pitch.
    Backend principal: `librosa.effects.time_stretch` (alta calidad,
    phase vocoder). Fallback: phase vocoder casero con STFT/iSTFT
    si librosa no esta disponible.

Ambas reciben numpy arrays float32 mono y devuelven lo mismo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class FreezerParams:
    """Parametros del granular freezer.

    freeze_start_s: instante del audio donde se toma la ventana congelada.
    grain_ms: duracion de cada grano (ventana copiada del audio fuente).
    output_duration_s: longitud total del audio congelado resultante.
    overlap: 0..1 cuanto solapan los grains (0 = sin solape; 0.5 = mitad).
    loop_smoothing: 'hann' para cross-fade Hann; 'linear' para fade lineal.
    seed: para jitter (variabilidad sutil en posicion de origen).
    jitter: 0..1 cuanto puede variar aleatoriamente el inicio de cada grano.
    """
    freeze_start_s: float = 0.5
    grain_ms: float = 80.0
    output_duration_s: float = 5.0
    overlap: float = 0.5
    loop_smoothing: str = "hann"
    seed: int = 0
    jitter: float = 0.1


@dataclass
class TimeStretchParams:
    """Parametros para time-stretch sin cambiar pitch.

    rate: 1.0 = sin cambio. 0.5 = mitad de velocidad (audio dura el doble).
          2.0 = doble velocidad (audio dura la mitad).
    method: 'librosa' usa librosa.effects.time_stretch (recomendado).
            'phase_vocoder' usa implementacion casera basica con STFT.
    n_fft: tamano FFT para el phase vocoder casero.
    """
    rate: float = 1.0
    method: str = "librosa"
    n_fft: int = 2048


def synth_granular_freezer(audio: np.ndarray, p: FreezerParams,
                            sr: int = 44_100) -> np.ndarray:
    """Toma un audio y genera un loop suave de longitud `output_duration_s`
    repitiendo grains de longitud `grain_ms` desde `freeze_start_s` con
    cross-fade.
    """
    if audio.ndim != 1:
        audio = audio.mean(axis=-1).astype(np.float32) if audio.ndim == 2 else audio.flatten()
    rng = np.random.default_rng(p.seed)
    grain_n = max(64, int(p.grain_ms / 1000.0 * sr))
    out_n = max(grain_n, int(p.output_duration_s * sr))
    out = np.zeros(out_n, dtype=np.float32)
    weight = np.zeros(out_n, dtype=np.float32)

    # Ventana de cross-fade
    if p.loop_smoothing == "linear":
        window = np.concatenate([
            np.linspace(0, 1, grain_n // 2, dtype=np.float32),
            np.linspace(1, 0, grain_n - grain_n // 2, dtype=np.float32),
        ])
    else:  # hann default
        window = np.hanning(grain_n).astype(np.float32)

    # Posicion base en el audio fuente
    src_pos_base = int(p.freeze_start_s * sr)
    if src_pos_base + grain_n > len(audio):
        src_pos_base = max(0, len(audio) - grain_n)

    hop_n = max(1, int(grain_n * (1.0 - p.overlap)))
    out_pos = 0
    while out_pos < out_n:
        # Jitter en posicion de origen (variabilidad sutil)
        jit_samples = int(p.jitter * grain_n * rng.uniform(-0.5, 0.5))
        src_pos = max(0, min(len(audio) - grain_n, src_pos_base + jit_samples))
        grain = audio[src_pos:src_pos + grain_n].astype(np.float32) * window
        end = min(out_n, out_pos + grain_n)
        out[out_pos:end] += grain[: end - out_pos]
        weight[out_pos:end] += window[: end - out_pos]
        out_pos += hop_n

    # Normalizar por peso para OLA limpio
    weight = np.where(weight < 1e-6, 1.0, weight)
    out = out / weight
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


def _phase_vocoder_stretch(audio: np.ndarray, rate: float, n_fft: int = 2048,
                            sr: int = 44_100) -> np.ndarray:
    """Phase vocoder casero: STFT, manipula phases, iSTFT.
    Funciona razonablemente sin necesitar librosa pero con calidad inferior.
    """
    hop = n_fft // 4
    f, t, Z = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                          padded=True, boundary="zeros")
    n_frames = Z.shape[1]
    new_n_frames = max(1, int(round(n_frames / rate)))
    # Indices fraccionales en el spectro original
    indices = np.linspace(0, n_frames - 1, new_n_frames)
    mag = np.abs(Z)
    phase = np.angle(Z)
    new_mag = np.zeros((Z.shape[0], new_n_frames), dtype=np.float32)
    new_phase = np.zeros_like(new_mag)
    cumulative_phase = phase[:, 0]
    for i, idx in enumerate(indices):
        i0 = int(np.floor(idx))
        i1 = min(n_frames - 1, i0 + 1)
        frac = idx - i0
        new_mag[:, i] = (1 - frac) * mag[:, i0] + frac * mag[:, i1]
        if i > 0:
            # Diferencia de phase esperada por hop original
            dphi = phase[:, i1] - phase[:, i0]
            # Wrap a [-pi, pi]
            dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
            cumulative_phase = cumulative_phase + dphi
        new_phase[:, i] = cumulative_phase
    new_Z = new_mag * np.exp(1j * new_phase)
    _, out = signal.istft(new_Z, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


def apply_time_stretch(audio: np.ndarray, p: TimeStretchParams,
                        sr: int = 44_100) -> np.ndarray:
    """Estira/comprime audio sin cambiar pitch.

    rate=1.0 -> identico. rate>1 -> audio mas rapido (mas corto).
    rate<1 -> audio mas lento (mas largo).
    """
    if audio.ndim != 1:
        audio = audio.mean(axis=-1).astype(np.float32) if audio.ndim == 2 else audio.flatten()
    rate = float(p.rate)
    if abs(rate - 1.0) < 1e-3:
        return audio.astype(np.float32)
    if p.method == "librosa":
        try:
            import librosa
            stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
            return stretched.astype(np.float32)
        except ImportError:
            pass
    # Fallback casero
    return _phase_vocoder_stretch(audio, rate, n_fft=p.n_fft, sr=sr)
