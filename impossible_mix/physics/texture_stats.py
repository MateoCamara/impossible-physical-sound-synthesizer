"""Morphing de texturas por estadisticas del sistema auditivo (v9-M3).

Subconjunto "lite" del modelo de McDermott & Simoncelli (2011, Neuron
71:926-940): la identidad de una textura sonora vive en estadisticas
promediadas en el tiempo -- momentos de la envolvente por banda coclear,
correlaciones de envolvente entre bandas y espectro de modulacion --, no
en la forma de onda. Imponer un set de estadisticas sobre ruido sintetiza
un ejemplar nuevo de la MISMA textura; interpolar dos sets y sintetizar
produce UNA textura estadisticamente intermedia (no dos superpuestas).

Omisiones respecto al modelo completo (documentadas): kurtosis, filtros
cocleares gammatone (usamos butter log-espaciados), correlaciones C2 de
fase de modulacion, y compresion coclear. Suficiente para texturas densas
tipo lluvia/fuego; el gate de sanidad (reimponer una textura sobre si
misma) valida la reimplementacion antes de usar el morphing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class TextureStats:
    env_mean: np.ndarray        # (B,)
    env_std: np.ndarray         # (B,)
    env_skew: np.ndarray        # (B,)
    env_corr: np.ndarray        # (B, B)
    mod_power: np.ndarray       # (B, M) potencia de modulacion por octavas
    band_edges: np.ndarray      # (B+1,)
    mod_edges: np.ndarray       # (M+1,) Hz


_ENV_FS = 400.0


def _filterbank(w: np.ndarray, sr: int, edges: np.ndarray) -> list[np.ndarray]:
    x = w.astype(np.float64)
    bands = []
    for k in range(len(edges) - 1):
        sos = signal.butter(4, [edges[k], min(edges[k + 1], sr / 2 - 100)],
                            btype="band", fs=sr, output="sos")
        bands.append(signal.sosfiltfilt(sos, x))
    return bands


def _envelope(band: np.ndarray, sr: int) -> np.ndarray:
    env = np.abs(signal.hilbert(band))
    sos = signal.butter(2, 128.0, btype="low", fs=sr, output="sos")
    return np.maximum(signal.sosfiltfilt(sos, env), 0.0)


def texture_statistics(w: np.ndarray, sr: int, n_bands: int = 18,
                       lo: float = 80.0, hi: float = 8000.0) -> TextureStats:
    edges = np.geomspace(lo, min(hi, sr / 2 - 200), n_bands + 1)
    bands = _filterbank(w, sr, edges)
    hop = max(1, int(sr / _ENV_FS))
    envs = np.stack([_envelope(b, sr)[::hop] for b in bands])
    m = envs.mean(axis=1)
    s = envs.std(axis=1)
    z = (envs - m[:, None]) / (s[:, None] + 1e-12)
    skew = (z ** 3).mean(axis=1)
    corr = np.corrcoef(envs)
    # Espectro de modulacion: potencia de la envolvente en octavas 0.5-64 Hz
    mod_edges = np.array([0.5, 1, 2, 4, 8, 16, 32, 64])
    n_env = envs.shape[1]
    freqs = np.fft.rfftfreq(n_env, 1.0 / _ENV_FS)
    mod_power = np.zeros((n_bands, len(mod_edges) - 1))
    for k in range(n_bands):
        spec = np.abs(np.fft.rfft(envs[k] - envs[k].mean())) ** 2
        tot = spec.sum() + 1e-18
        for j in range(len(mod_edges) - 1):
            sel = (freqs >= mod_edges[j]) & (freqs < mod_edges[j + 1])
            mod_power[k, j] = spec[sel].sum() / tot
    return TextureStats(m, s, skew, corr, mod_power, edges, mod_edges)


def interp_stats(a: TextureStats, b: TextureStats, alpha: float) -> TextureStats:
    """Interpolacion en el espacio de estadisticas (lineal; la matriz de
    correlaciones se proyecta a PSD por si la mezcla pierde definicion)."""
    corr = (1 - alpha) * a.env_corr + alpha * b.env_corr
    vals, vecs = np.linalg.eigh((corr + corr.T) / 2)
    corr = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    return TextureStats(
        env_mean=(1 - alpha) * a.env_mean + alpha * b.env_mean,
        env_std=(1 - alpha) * a.env_std + alpha * b.env_std,
        env_skew=(1 - alpha) * a.env_skew + alpha * b.env_skew,
        env_corr=corr,
        mod_power=(1 - alpha) * a.mod_power + alpha * b.mod_power,
        band_edges=a.band_edges, mod_edges=a.mod_edges)


def _impose_moments(env: np.ndarray, mean: float, std: float,
                    skew: float) -> np.ndarray:
    """Momento-matching aproximado: transformacion de potencia monotona
    para el sesgo + afin para media/std, con positividad."""
    e = env - env.min()
    e = e / (e.max() + 1e-12)
    cur_skew = float(((e - e.mean()) ** 3).mean() / (e.std() ** 3 + 1e-12))
    # exponente que empuja el sesgo hacia el objetivo (heuristica estable)
    gamma = float(np.clip(1.0 + 0.35 * (skew - cur_skew), 0.4, 3.0))
    e = e ** gamma
    e = (e - e.mean()) / (e.std() + 1e-12)
    return np.maximum(e * std + mean, 0.0)


def _impose_mod_spectrum(env: np.ndarray, target_row: np.ndarray,
                         mod_edges: np.ndarray) -> np.ndarray:
    """Conforma la potencia de modulacion por octavas via FFT de la envolvente."""
    mu = env.mean()
    x = env - mu
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / _ENV_FS)
    power = np.abs(spec) ** 2
    tot = power.sum() + 1e-18
    gains = np.ones_like(power)
    for j in range(len(mod_edges) - 1):
        sel = (freqs >= mod_edges[j]) & (freqs < mod_edges[j + 1])
        cur = power[sel].sum() / tot
        if cur > 1e-12:
            gains[sel] = np.sqrt(max(target_row[j], 1e-6) / cur)
    x = np.fft.irfft(spec * gains, n=len(x))
    return np.maximum(x + mu, 0.0)


def impose_statistics(stats: TextureStats, sr: int, duration_s: float,
                      n_iter: int = 24, seed: int = 0,
                      return_history: bool = False):
    """Sintetiza una textura desde ruido imponiendo las estadisticas
    iterativamente (moment matching + espectro de modulacion +
    correlaciones entre bandas via Cholesky, reconstruyendo cada banda como
    envolvente nueva x estructura fina actual)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    w = rng.standard_normal(n).astype(np.float64) * 0.1
    edges = stats.band_edges
    n_bands = len(edges) - 1
    hop = max(1, int(sr / _ENV_FS))
    L = np.linalg.cholesky(stats.env_corr + 1e-6 * np.eye(n_bands))
    history = []
    for it in range(n_iter):
        bands = _filterbank(w, sr, edges)
        envs_full = [np.maximum(_envelope(b, sr), 1e-9) for b in bands]
        envs = np.stack([e[::hop] for e in envs_full])
        # 1) espectro de modulacion + momentos por banda
        for k in range(n_bands):
            e = _impose_mod_spectrum(envs[k], stats.mod_power[k], stats.mod_edges)
            envs[k] = _impose_moments(e, stats.env_mean[k], stats.env_std[k],
                                      stats.env_skew[k])
        # 2) correlaciones entre bandas: blanquear y recolorear (Cholesky)
        z = (envs - envs.mean(axis=1, keepdims=True))
        cov = z @ z.T / z.shape[1]
        Lc = np.linalg.cholesky(cov + 1e-9 * np.eye(n_bands))
        z_w = np.linalg.solve(Lc, z)
        z_c = L @ z_w
        envs = np.maximum(z_c * stats.env_std[:, None] / (z_c.std(axis=1, keepdims=True) + 1e-12)
                          + stats.env_mean[:, None], 0.0)
        # 3) reconstruir: envolvente nueva x estructura fina actual.
        # La estructura fina es cos(fase de Hilbert) — acotada en [-1,1]
        # (dividir banda/envolvente explota cuando la envolvente suavizada
        # pasa cerca de cero; misma factorizacion que auditory_chimera).
        w_new = np.zeros(n)
        t_env = np.arange(envs.shape[1]) * hop
        for k in range(n_bands):
            env_up = np.interp(np.arange(n), t_env, envs[k])
            fine = np.cos(np.angle(signal.hilbert(bands[k])))
            w_new += env_up * fine
        w = w_new
        if return_history:
            cur = texture_statistics(w, sr, n_bands, edges[0], edges[-1])
            history.append(stats_distance(cur, stats))
    peak = float(np.abs(w).max() + 1e-9)
    w = (w * (0.95 / peak if peak > 0.95 else 1.0)).astype(np.float32)
    return (w, history) if return_history else w


def stats_distance(a: TextureStats, b: TextureStats) -> float:
    """Distancia normalizada entre sets de estadisticas (para gates)."""
    d = 0.0
    d += float(np.abs(np.log((a.env_mean + 1e-9) / (b.env_mean + 1e-9))).mean())
    d += float(np.abs(np.log((a.env_std + 1e-9) / (b.env_std + 1e-9))).mean())
    d += float(np.abs(a.env_skew - b.env_skew).mean()) * 0.2
    d += float(np.abs(a.env_corr - b.env_corr).mean()) * 2.0
    d += float(np.abs(a.mod_power - b.mod_power).mean()) * 5.0
    return d


def texture_morph(w_a: np.ndarray, w_b: np.ndarray, alpha: float, sr: int,
                  duration_s: float | None = None, n_iter: int = 24,
                  seed: int = 0) -> np.ndarray:
    """UNA textura estadisticamente intermedia entre A y B (no superpuestas)."""
    if duration_s is None:
        duration_s = min(len(w_a), len(w_b)) / sr
    sa = texture_statistics(w_a, sr)
    sb = texture_statistics(w_b, sr)
    return impose_statistics(interp_stats(sa, sb, alpha), sr, duration_s,
                             n_iter=n_iter, seed=seed)
