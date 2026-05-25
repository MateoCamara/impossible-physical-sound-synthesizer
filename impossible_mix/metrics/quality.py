"""Metricas objetivas baratas para detectar 'basura ruidosa' sin escuchar.

Idea: filtrar antes de invertir tiempo en escucha humana. Un audio decente
debe pasar varios sanity simultaneamente:
  - RMS dBFS > -35 dB         : no aplastado a silencio
  - peak dBFS > -10 dB        : tiene transitorios audibles
  - crest_factor > 4 dB       : no es ruido constante
  - spectral_flatness < 0.5   : no es ruido blanco/rosa puro
  - spectral_centroid < 8 kHz : no es solo hiss en agudos
  - dynamic_range > 6 dB      : tiene variacion temporal

Ademas comparativos contra el ANCLA original:
  - log_spec_distance(anchor, hybrid)  : ~0 = identico ancla, >0.5 = se aleja
  - rms_ratio(anchor, hybrid)          : energia conservada? deberia ~1.0

Una funcion `quality_verdict` resume todo en {good, suspicious, garbage}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


def _safe_db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def rms_dbfs(y: np.ndarray) -> float:
    return _safe_db(float(np.sqrt(np.mean(y ** 2) + 1e-12)))


def peak_dbfs(y: np.ndarray) -> float:
    return _safe_db(float(np.max(np.abs(y)) + 1e-12))


def crest_factor_db(y: np.ndarray) -> float:
    """peak / rms en dB. Ruido tonal: 3-6 dB. Transitorio impulsivo: 15-25 dB."""
    return peak_dbfs(y) - rms_dbfs(y)


def spectral_flatness(y: np.ndarray, sr: int) -> float:
    """Wiener entropy: 1 = ruido blanco, 0 = tono puro. Media sobre frames."""
    f, t, Z = stft(y, fs=sr, nperseg=2048, noverlap=1024)
    mag = np.abs(Z) + 1e-12
    geo = np.exp(np.mean(np.log(mag), axis=0))
    arith = np.mean(mag, axis=0) + 1e-12
    return float(np.mean(geo / arith))


def spectral_centroid_hz(y: np.ndarray, sr: int) -> float:
    """Centroide medio en Hz."""
    f, t, Z = stft(y, fs=sr, nperseg=2048, noverlap=1024)
    mag = np.abs(Z) + 1e-12
    cent_per_frame = (f[:, None] * mag).sum(0) / mag.sum(0)
    return float(np.mean(cent_per_frame))


def dynamic_range_db(y: np.ndarray, win_ms: float = 30.0, sr: int = 44100) -> float:
    """Diferencia entre RMS de la ventana mas energica y la mas silenciosa
    (en dB). Sonido constante: ~0. Sonido con transitorios: >10 dB."""
    win = max(1, int(win_ms * sr / 1000.0))
    if len(y) < 2 * win:
        return 0.0
    n_wins = len(y) // win
    rmss = []
    for k in range(n_wins):
        w = y[k * win : (k + 1) * win]
        rmss.append(float(np.sqrt(np.mean(w ** 2) + 1e-12)))
    rmss = np.array(rmss)
    return _safe_db(rmss.max()) - _safe_db(rmss.min() + 1e-12)


def log_spec_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Distancia log-espectral media entre dos audios de igual longitud."""
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    fa, ta, Za = stft(a, fs=sr, nperseg=1024, noverlap=512)
    fb, tb, Zb = stft(b, fs=sr, nperseg=1024, noverlap=512)
    log_a = np.log(np.abs(Za) + 1e-9)
    log_b = np.log(np.abs(Zb) + 1e-9)
    return float(np.sqrt(np.mean((log_a - log_b) ** 2)))


@dataclass
class QualityReport:
    rms_db: float
    peak_db: float
    crest_db: float
    flatness: float
    centroid_hz: float
    dynamic_range_db: float
    verdict: str  # "good" | "suspicious" | "garbage"
    reasons: list[str]


def quality_verdict(y: np.ndarray, sr: int) -> QualityReport:
    rms = rms_dbfs(y)
    peak = peak_dbfs(y)
    crest = crest_factor_db(y)
    flat = spectral_flatness(y, sr)
    cent = spectral_centroid_hz(y, sr)
    dyn = dynamic_range_db(y, sr=sr)

    # Un evento corto pero claro (drip, impact aislado) tiene RMS bajo
    # (mucha duracion es silencio) pero peak alto y crest >12 dB. Lo
    # consideramos OK si peak>-15 dBFS aunque el RMS sea bajo.
    rms_is_ok = rms > -45 or (peak > -15 and crest > 12)
    peak_is_ok = peak > -20

    reasons: list[str] = []
    if not rms_is_ok:
        reasons.append(f"silent (rms={rms:.1f} dBFS, peak={peak:.1f})")
    if not peak_is_ok:
        reasons.append(f"no transients (peak={peak:.1f} dBFS)")
    if flat > 0.5:
        reasons.append(f"noise-like (flatness={flat:.2f})")
    if dyn < 1.5:
        reasons.append(f"flat dynamics (range={dyn:.1f} dB)")
    # "Hiss only" solo se penaliza si NO hay transitorios fuertes (crest bajo).
    # Un impacto corto en silencio sube el centroide pero NO es hiss, sino
    # un evento valido (e.g. drip, splash, modal short hit).
    if cent > 9000 and crest < 10:
        reasons.append(f"hiss only (centroid={cent:.0f} Hz, crest={crest:.1f})")

    if len(reasons) >= 2:
        verdict = "garbage"
    elif len(reasons) == 1:
        verdict = "suspicious"
    else:
        verdict = "good"

    return QualityReport(
        rms_db=rms, peak_db=peak, crest_db=crest, flatness=flat,
        centroid_hz=cent, dynamic_range_db=dyn,
        verdict=verdict, reasons=reasons,
    )
