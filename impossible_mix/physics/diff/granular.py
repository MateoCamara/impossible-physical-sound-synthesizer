"""Granular flow, differentiable (PyTorch).

Reto: el granular real (impossible_mix.physics.granular) es estocastico
- cada grano tiene su propio timing, velocity y jitter de pitch. Esa
estocasticidad no es diferenciable.

Solucion: **schedule pre-generado con seed fijo + perfil de grano
aprendible**.
  - El "schedule" (cuando dispara cada grano, con que jitter relativo
    en pitch y velocity) se sortea una sola vez con un seed y queda
    como constante del problema.
  - Lo entrenable es la descripcion espectro-temporal del *grano tipo*:
    base_freq_hz, spread_octaves, damping_ms, gain, y una "log_density"
    que escala la amplitud global (mas granos contribuyen, mas energia).

Esto permite recuperar el perfil mineral aproximado de una nube
granular real desde audio: ¿es arena, grava fina, grava gruesa, ice
shards? La densidad temporal exacta no se recupera, pero la *firma
acustica* (banda, decay, dispersion) si.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


_LN1000 = 6.907755


@dataclass
class GranularFlowParamsT:
    """Parametros diferenciables del flujo granular.

    Aprendibles (cinco escalares):
      base_freq_hz, spread_octaves, damping_ms, log_density_amp, gain.

    Constantes (no diferenciables):
      duration_s, sr, seed, n_grains, grain_dur_ms (longitud de cada
      grano en samples, fijo).
    """
    base_freq_hz: torch.Tensor          # () Hz
    spread_octaves: torch.Tensor        # () >=0; cuanto mas, mas variabilidad de pitch
    damping_ms: torch.Tensor            # () t60 de cada grano (ms)
    log_density_amp: torch.Tensor       # () log-amplitud del flujo (controla "cuantos granos suenan")
    gain: torch.Tensor                  # () gain global

    # Estado del schedule fijo
    seed: int = 0
    n_grains: int = 200
    grain_dur_ms: float = 40.0
    duration_s: float = 5.0

    @classmethod
    def physical_init(cls, base_freq_hz: float = 2000.0, spread_octaves: float = 0.8,
                       damping_ms: float = 20.0, log_density_amp: float = 0.0,
                       gain: float = 0.5,
                       device: str | torch.device = "cpu",
                       requires_grad: bool = True,
                       **kwargs) -> "GranularFlowParamsT":
        def t(v):
            x = torch.tensor(float(v), device=device, dtype=torch.float32)
            return x.requires_grad_(requires_grad) if requires_grad else x
        return cls(
            base_freq_hz=t(base_freq_hz),
            spread_octaves=t(spread_octaves),
            damping_ms=t(damping_ms),
            log_density_amp=t(log_density_amp),
            gain=t(gain),
            **kwargs,
        )

    def trainable(self) -> list[torch.Tensor]:
        return [self.base_freq_hz, self.spread_octaves, self.damping_ms,
                self.log_density_amp, self.gain]

    def clamp_(self):
        with torch.no_grad():
            self.base_freq_hz.clamp_(80.0, 16000.0)
            self.spread_octaves.clamp_(0.0, 3.0)
            self.damping_ms.clamp_(0.5, 500.0)
            self.log_density_amp.clamp_(-5.0, 5.0)
            self.gain.clamp_(0.01, 2.0)


@dataclass
class _GrainSchedule:
    """Plan precomputado: timing, jitter relativo y velocity de cada grano.
    No diferenciable (todo en numpy/torch sin grad)."""
    grain_starts_samples: torch.Tensor    # (N,) long
    jitter_octaves: torch.Tensor          # (N,) float in [-1, 1]
    velocity_factors: torch.Tensor        # (N,) float in [0.5, 1.0]


def _make_schedule(p: GranularFlowParamsT, sr: int,
                    device: str | torch.device = "cpu") -> _GrainSchedule:
    rng = np.random.default_rng(p.seed)
    n_samples = int(p.duration_s * sr)
    # Tiempos uniformes (con jitter) en [0, duration]
    times = np.sort(rng.uniform(0, p.duration_s, size=p.n_grains))
    starts = (times * sr).astype(np.int64)
    starts = np.clip(starts, 0, max(0, n_samples - 4))
    jitter = rng.uniform(-1, 1, size=p.n_grains).astype(np.float32)
    velo = rng.uniform(0.5, 1.0, size=p.n_grains).astype(np.float32)
    return _GrainSchedule(
        grain_starts_samples=torch.from_numpy(starts).to(device),
        jitter_octaves=torch.from_numpy(jitter).to(device),
        velocity_factors=torch.from_numpy(velo).to(device),
    )


def synth_granular_flow_diff(p: GranularFlowParamsT, sr: int,
                              n_samples: int | None = None,
                              schedule: _GrainSchedule | None = None) -> torch.Tensor:
    """Sintetiza un flujo granular diferenciable.

    Cada grano k es un mini-impacto modal corto:
        g_k(t) = velocity_k * sin(2*pi*f_k*t) * exp(-t/tau)
    con f_k = base_freq_hz * 2 ** (jitter_k * spread_octaves) y
        tau  = damping_ms / 6.91 / 1000.

    La amplitud global se modula por sigmoid(log_density_amp) * gain.
    """
    device = p.base_freq_hz.device
    dtype = p.base_freq_hz.dtype
    if n_samples is None:
        n_samples = int(p.duration_s * sr)
    if schedule is None:
        schedule = _make_schedule(p, sr, device=device)

    # grain_dur efectivo: si grain_dur_ms es pequeno (acoplado al damping),
    # solo cubrimos la parte audible del grano. Forzamos un minimo de 5x el
    # damping para que cada grano decay completo, y un maximo de grain_dur_ms.
    damping_for_dur = float(torch.clamp(p.damping_ms.detach(), min=0.5).item())
    eff_grain_dur_ms = min(p.grain_dur_ms, max(5 * damping_for_dur, 8.0))
    grain_n = max(8, int(eff_grain_dur_ms / 1000.0 * sr))
    t_grain = torch.arange(grain_n, device=device, dtype=dtype) / sr  # (G,)

    # Para cada grano k, freq y decay
    freqs = p.base_freq_hz * (2.0 ** (schedule.jitter_octaves * p.spread_octaves))  # (N,)
    tau = torch.clamp(p.damping_ms, min=0.5) / 1000.0 / _LN1000  # () s
    # Decay shape: comun a todos los granos (mismo tau)
    decay = torch.exp(-t_grain / tau)  # (G,)
    # Sinusoides por grano: (N, G) = freqs(N,1) * t_grain(1,G)
    phase = 2 * torch.pi * freqs.unsqueeze(1) * t_grain.unsqueeze(0)
    osc = torch.sin(phase)  # (N, G)
    # Amplitud por grano
    amp = (p.gain * torch.sigmoid(p.log_density_amp) *
            schedule.velocity_factors)  # (N,)
    # Granos completos: amp(N,1) * decay(G) * osc(N,G) -> (N, G)
    grains = amp.unsqueeze(1) * decay.unsqueeze(0) * osc

    # Scatter-add vectorizado: construir todos los indices destino (N, G) y
    # usar index_add_. Mucho mas rapido que un loop Python bajo autograd.
    starts = schedule.grain_starts_samples  # (N,) long
    N = grains.shape[0]
    G = grains.shape[1]
    # idxs[k, j] = start_k + j; clamp a [0, n_samples - 1] para evitar OOB.
    arange_g = torch.arange(G, device=device, dtype=torch.long)        # (G,)
    idxs = starts.unsqueeze(1) + arange_g.unsqueeze(0)                 # (N, G)
    # Mascara: posiciones validas dentro de [0, n_samples)
    valid = idxs < n_samples                                            # (N, G) bool
    idxs_clamped = idxs.clamp(min=0, max=n_samples - 1)
    grains_masked = grains * valid                                      # (N, G), zero fuera
    out = torch.zeros(n_samples, device=device, dtype=dtype)
    out.index_add_(0, idxs_clamped.reshape(-1), grains_masked.reshape(-1))
    return out
