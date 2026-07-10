"""IO y reporting compartidos para los scripts de fitting inverso (26-31).

Centraliza: carga de WAV a mono/target_sr (con resample de calidad via
librosa y fallback a scipy.signal.resample_poly), escritura de WAV, y
persistencia de los resultados de un fit (params recuperados vs ground
truth, errores, historia de loss) a un `params.json` por corrida — hoy
esos numeros solo existen en stdout y se citan en el paper sin respaldo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def load_wav_mono(path: Path, target_sr: int, *, max_seconds: float | None = None) -> np.ndarray:
    """Carga un WAV, lo mezcla a mono y lo resamplea a `target_sr`.

    Usa librosa (buena calidad) y cae a `scipy.signal.resample_poly` si
    librosa no esta disponible. Si `max_seconds` se da, recorta o
    rellena con ceros al final para forzar esa duracion exacta.
    """
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, target_sr)
            wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    wav = wav.astype(np.float32)
    if max_seconds is not None:
        n_max = int(max_seconds * target_sr)
        if len(wav) > n_max:
            wav = wav[:n_max]
        elif len(wav) < n_max:
            wav = np.pad(wav, (0, n_max - len(wav)))
    return wav


def save_wav(path: Path, audio: np.ndarray, sr: int = 44_100) -> None:
    """Guarda `audio` (mono, float) como WAV, con clip suave a [-1, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), audio, sr)


def _to_jsonable(value: Any) -> Any:
    """Convierte escalares/listas numpy/torch a tipos nativos serializables."""
    if hasattr(value, "detach"):
        value = value.detach().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _compute_errors(gt: dict, recovered: dict) -> dict:
    """Error absoluto y % por clave comun de gt/recovered (mismo calculo
    que los scripts ya imprimen en stdout). Solo compara valores
    escalares; las claves con listas se omiten (el matching por
    frecuencia, si aplica, lo hace el caller)."""
    errors = {}
    for k in recovered:
        if k not in gt:
            continue
        gt_v, rec_v = gt[k], recovered[k]
        if isinstance(gt_v, (list, tuple)) or isinstance(rec_v, (list, tuple)):
            continue
        if gt_v is None or rec_v is None:
            continue
        err = abs(rec_v - gt_v)
        pct = 100 * err / max(abs(gt_v), 1e-9)
        errors[k] = {"abs": err, "pct": pct}
    return errors


def save_fit_report(
    out_dir: Path,
    engine: str,
    recovered: dict,
    gt: dict | None,
    loss_history: list[float],
    extra: dict | None = None,
    final_loss: float | None = None,
) -> Path:
    """Escribe `out_dir/params.json` con el resumen completo de un fit.

    Incluye: engine, argv, gt, recovered, errors (por clave comun de
    gt/recovered), initial_loss, final_loss, n_iters, loss_history
    completo, y cualquier `extra` que el caller quiera adjuntar.

    `final_loss`, si se pasa, es el valor que el script ya imprime como
    "Final loss" (recomputado por el fitter tras el ultimo `clamp_()`
    bajo `torch.no_grad()` — no siempre coincide con `loss_history[-1]`,
    que es la loss del ultimo paso de entrenamiento). Si no se pasa, se
    usa `loss_history[-1]` como aproximacion.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recovered_j = {k: _to_jsonable(v) for k, v in recovered.items()}
    gt_j = {k: _to_jsonable(v) for k, v in gt.items()} if gt is not None else None
    errors = _compute_errors(gt_j, recovered_j) if gt_j is not None else None
    loss_history_j = [float(x) for x in loss_history]
    if final_loss is None:
        final_loss = loss_history_j[-1] if loss_history_j else None
    else:
        final_loss = float(final_loss)

    report = {
        "engine": engine,
        "argv": sys.argv,
        "gt": gt_j,
        "recovered": recovered_j,
        "errors": errors,
        "initial_loss": loss_history_j[0] if loss_history_j else None,
        "final_loss": final_loss,
        "n_iters": len(loss_history_j),
        "loss_history": loss_history_j,
    }
    if extra:
        report["extra"] = {k: _to_jsonable(v) for k, v in extra.items()}

    out_path = out_dir / "params.json"
    out_path.write_text(json.dumps(report, indent=2))
    return out_path
