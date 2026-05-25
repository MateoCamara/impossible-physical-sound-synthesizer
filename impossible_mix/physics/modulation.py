"""Modulacion temporal de knobs: un knob puede ser una CURVA en el tiempo,
no solo un valor estatico. Esto convierte sonidos planos de 5s en eventos
evolutivos: una gota que se acelera, una rodadura que se humedece, etc.

API:
    from impossible_mix.physics.modulation import KnobCurve

    # Rampa lineal de 0 a 1 a lo largo del clip
    KnobCurve.ramp(0.0, 1.0)
    # LFO oscilante 0..1 a 0.5 Hz
    KnobCurve.lfo(0.5, lo=0.0, hi=1.0)
    # Curva exponencial decreciente
    KnobCurve.exp_decay(1.0, tau_s=2.0, floor=0.1)
    # Curva custom
    KnobCurve.from_points([(0, 0.0), (0.3, 1.0), (1.0, 0.2)])

El composer evolutivo `compose_evolving` toma un PhysicsController con
una escena fija y rinde el clip generando audios en ventanas cortas
(hop=0.5 s) con knobs interpolados desde las curvas. Cross-fade entre
ventanas para evitar clicks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class KnobCurve:
    """Curva valor(t) para t in [0, 1] (normalizado)."""
    fn: Callable[[float], float]

    def at(self, t_norm: float) -> float:
        return float(self.fn(min(1.0, max(0.0, t_norm))))

    @staticmethod
    def constant(v: float) -> "KnobCurve":
        return KnobCurve(lambda t: v)

    @staticmethod
    def ramp(start: float, end: float) -> "KnobCurve":
        return KnobCurve(lambda t: start + (end - start) * t)

    @staticmethod
    def exp_ramp(start: float, end: float, gamma: float = 2.0) -> "KnobCurve":
        return KnobCurve(lambda t: start + (end - start) * (t ** gamma))

    @staticmethod
    def lfo(freq_cycles_per_clip: float, lo: float = 0.0, hi: float = 1.0,
            phase: float = 0.0) -> "KnobCurve":
        center = (lo + hi) / 2
        amp = (hi - lo) / 2
        return KnobCurve(lambda t: center + amp * np.sin(2 * np.pi * freq_cycles_per_clip * t + phase))

    @staticmethod
    def exp_decay(peak: float, tau_norm: float = 0.3, floor: float = 0.0) -> "KnobCurve":
        """Decae exponencialmente desde `peak` hacia `floor` con escala tau (normalizada)."""
        return KnobCurve(lambda t: floor + (peak - floor) * float(np.exp(-t / max(tau_norm, 1e-3))))

    @staticmethod
    def from_points(points: list[tuple[float, float]]) -> "KnobCurve":
        """Interpolacion lineal entre puntos (t_norm, valor)."""
        ts = np.array([p[0] for p in points])
        vs = np.array([p[1] for p in points])
        return KnobCurve(lambda t: float(np.interp(t, ts, vs)))


def compose_evolving(
    controller,
    knob_curves: dict[str, KnobCurve],
    duration_s: float = 8.0,
    window_s: float = 0.6,
    hop_s: float = 0.3,
    sr: int = 44_100,
) -> np.ndarray:
    """Sintetiza un audio evolutivo donde cada knob sigue su curva.

    Estrategia: renderizamos ventanas overlap-add. Cada ventana usa los
    valores de knobs interpolados al tiempo central de la ventana.
    Cross-fade Hann entre ventanas para evitar clicks.
    """
    n_total = int(duration_s * sr)
    win_n = int(window_s * sr)
    hop_n = int(hop_s * sr)
    out = np.zeros(n_total + win_n, dtype=np.float32)
    weight_sum = np.zeros_like(out)
    hann = np.hanning(win_n).astype(np.float32)

    # backup de duracion original del controller
    prev_dur = controller.duration_s
    controller.duration_s = window_s

    start = 0
    while start < n_total:
        t_center = (start + win_n / 2) / sr
        t_norm = t_center / duration_s
        knob_values = {k: c.at(t_norm) for k, c in knob_curves.items()}
        wav_chunk = controller.render(**knob_values)
        wav_chunk = wav_chunk[:win_n]  # asegurar tamano exacto
        if len(wav_chunk) < win_n:
            wav_chunk = np.pad(wav_chunk, (0, win_n - len(wav_chunk)))
        out[start:start + win_n] += wav_chunk * hann
        weight_sum[start:start + win_n] += hann
        start += hop_n

    # Evitar division por cero al normalizar
    weight_sum = np.where(weight_sum < 1e-6, 1.0, weight_sum)
    out = out[:n_total] / weight_sum[:n_total]
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.95:
        out = out * (0.95 / peak)

    # Restaurar
    controller.duration_s = prev_dur
    return out.astype(np.float32)
