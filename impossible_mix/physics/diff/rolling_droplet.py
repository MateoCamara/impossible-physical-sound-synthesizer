"""Rolling droplet differentiable (PyTorch).

DDSP-style differentiable port of the full rolling-droplet model. All
physical parameters are torch.Tensors and gradients flow end-to-end,
enabling inverse fitting from real recordings.

Layers mirror impossible_mix.physics.droplet.synth_rolling_droplet:

  A) Body resonance:    sustained Minnaert tone, AM/FM modulated.
                        Amplitude couples to (1) surface acoustic
                        impedance via t60_s, (2) wetting via (1 - cos θ).
  B) Cavity:            Helmholtz tone tied to surface_modes_hz[0] × 0.6,
                        volume scales with (1 - cos θ).
  C) Rayleigh modes:    exact shape oscillations of a free liquid sphere
                        ω_n² = n(n-1)(n+2) σ / (ρ r³)  for n = 2..6.
  E) Surface ringing:   continuous bank of surface modes with t60 sustain.
  F) Microbubble cloud: N=8 entrained microbubbles, radii fixed by seed
                        at construction (log-normal r/3), amplitudes
                        differentiable.
  D) Shimmer:           slow random AM applied to the continuous mix.
                        LFO trace fixed by seed; depth differentiable.
  G) Stick-slip:        asperity events at fixed timestamps (seed),
                        amplitudes differentiable.
  H) Discrete ticks:    NOT PORTED. The numpy engine's discrete-tick layer
                        (synth_drip_event) was never wired into this diff
                        forward pass; `discrete_mix` is kept only as a
                        non-trainable placeholder (see physical_init).

Random structure (microbubble radii, stick-slip schedule, shimmer LFO)
is sampled ONCE at construction with a fixed seed and stored as
non-trainable buffers; only the *amplitudes / mix levels* are
differentiable. This mirrors how `granular.py` handles its grain
schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


_LN1000 = 6.907755
_RHO_WATER = 1000.0  # kg/m³


def _bubble_efold_tau_ms(radius_mm: float) -> float:
    """E-folding time constant (ms) of a Minnaert bubble of given radius.

    van den Doel (2005) damping law ``d = 0.043 f + 0.0014 f^{3/2}`` (f in
    kHz, d in ms^-1); the radiated sinusoid decays as ``exp(-d t) = exp(-t/tau)``
    with ``tau = 1/d``. Couples each microbubble's decay to its radius via
    validated physics instead of an ad-hoc radius power law.
    """
    f_khz = 3.26 / max(radius_mm, 1e-4)          # Minnaert: f[kHz] = 3.26 / r[mm]
    d = 0.043 * f_khz + 0.0014 * f_khz ** 1.5    # ms^-1
    return 1.0 / max(d, 1e-6)


@dataclass
class RollingDropletParamsT:
    """Differentiable parameters of the rolling droplet.

    Tensor fields can be 0-D (scalar) or 1-D (vectors) and may have
    ``requires_grad=True`` for gradient-based optimisation.

    Non-tensor fields (random buffers) are sampled once and held fixed.
    """
    # ---- Physical scalar params (all 0-D tensors) -----------------
    radius_mm: torch.Tensor              # droplet radius (mm)
    viscosity: torch.Tensor              # 0..1
    roll_velocity_hz: torch.Tensor       # rolling speed
    path_roughness: torch.Tensor         # 0..1
    contact_angle_deg: torch.Tensor      # wetting (20..175)
    surface_tension_n_m: torch.Tensor    # N/m (0.02..0.10)
    # ---- Surface modal profile (1-D tensors of length M) ----------
    surface_modes_hz: torch.Tensor       # (M,)
    surface_t60s_s: torch.Tensor         # (M,)
    surface_gains: torch.Tensor          # (M,)  softmax'd before use
    # ---- Layer mix levels (differentiable 0-D scalars) ------------
    body_resonance_mix: torch.Tensor
    cavity_mix: torch.Tensor
    rayleigh_mix: torch.Tensor
    surface_ring_mix: torch.Tensor
    microbubble_mix: torch.Tensor
    stickslip_mix: torch.Tensor
    shimmer_depth: torch.Tensor
    discrete_mix: torch.Tensor
    # ---- Fixed random structure (non-trainable buffers) -----------
    # Microbubble radii (mm), shape (Nmicro,)
    microbubble_radii_mm: torch.Tensor
    # Microbubble onset samples (int-valued floats), shape (Nmicro,)
    microbubble_onsets: torch.Tensor
    # Microbubble phase offsets, shape (Nmicro,)
    microbubble_phases: torch.Tensor
    # Slow LFO trace for body/cavity AM, shape (N,)
    am_lfo: torch.Tensor
    # FM trace for body resonance, shape (N,)
    body_fm: torch.Tensor
    # Rayleigh trigger LFOs, one per mode, shape (5, N)
    rayleigh_triggers: torch.Tensor
    # Surface ringing per-mode FM LFOs, shape (M, N)
    ring_fms: torch.Tensor
    # Stick-slip event positions (int) and amplitudes (float)
    stick_positions: torch.Tensor  # (Nstick,)
    stick_burst_lens: torch.Tensor  # (Nstick,)
    stick_amplitudes: torch.Tensor  # (Nstick,) — differentiable
    stick_noise: torch.Tensor       # (Nstick, max_burst_len) bandpass-filtered noise (FIXED)
    # Shimmer LFO trace, shape (N,)
    shimmer_lfo: torch.Tensor
    # Sample rate + n_samples (for shape consistency)
    _sr: int
    _n_samples: int

    @classmethod
    def physical_init(
        cls,
        radius_mm: float = 2.5,
        viscosity: float = 0.1,
        roll_velocity_hz: float = 14.0,
        path_roughness: float = 0.35,
        contact_angle_deg: float = 110.0,
        surface_tension_n_m: float = 0.072,
        surface_modes_hz: list[float] | None = None,
        surface_t60s_s: list[float] | None = None,
        surface_gains: list[float] | None = None,
        sr: int = 44_100,
        duration_s: float = 2.0,
        n_microbubbles: int = 8,
        n_stickslip: int = 60,
        seed: int = 0,
        device: str | torch.device = "cpu",
        requires_grad: bool = True,
    ) -> "RollingDropletParamsT":
        """Create physically-sensible defaults (ceramic surface).

        Random structure (microbubble radii, stick-slip schedule, LFOs)
        is sampled with the given seed and stored as fixed buffers.
        Only physical/mix parameters are made differentiable.
        """
        if surface_modes_hz is None:
            surface_modes_hz = [1100.0, 2400.0, 4800.0, 7200.0]
        if surface_t60s_s is None:
            surface_t60s_s = [0.35, 0.18, 0.10, 0.06]
        if surface_gains is None:
            surface_gains = [0.35, 0.30, 0.20, 0.15]

        def t(v, grad=True):
            x = torch.tensor(v, device=device, dtype=torch.float32)
            return x.requires_grad_(grad and requires_grad) if (grad and requires_grad) else x

        def buf(v):  # non-trainable buffer
            x = torch.tensor(v, device=device, dtype=torch.float32)
            return x

        n_samples = int(duration_s * sr)
        rng = np.random.default_rng(seed)
        # Sample microbubble radii (log-normal centred at r/3)
        ln_radii = np.log(radius_mm / 3.0) + 0.6 * rng.standard_normal(n_microbubbles)
        radii_mm = np.exp(ln_radii)
        radii_mm = np.clip(radii_mm, 0.05, radius_mm * 0.8)
        # Onsets uniformly in [0, 0.3 * N)
        onsets = rng.uniform(0, n_samples * 0.3, size=n_microbubbles).astype(np.float32)
        # Random phase offsets
        phases = rng.uniform(0, 2 * np.pi, size=n_microbubbles).astype(np.float32)

        # Slow AM LFO (~2-8 Hz from velocity)
        am_hz = 2.0 + 6.0 * min(roll_velocity_hz / 25.0, 1.0)
        am_lfo = _slow_lfo_np(n_samples, sr, am_hz, rng)
        body_fm = _slow_lfo_np(n_samples, sr, am_hz * 0.6, rng)
        shimmer_hz = 5.0 + 10.0 * float(rng.random())
        shimmer_lfo = _slow_lfo_np(n_samples, sr, shimmer_hz, rng)

        # Rayleigh trigger LFOs (one per mode 2..6)
        trigger_hz = 2.0 + 8.0 * path_roughness
        rayleigh_triggers = np.stack([
            _slow_lfo_np(n_samples, sr, trigger_hz, rng) for _ in range(5)
        ])
        # Per-mode FM for surface ringing
        n_smodes = len(surface_modes_hz)
        ring_fms = np.stack([
            _slow_lfo_np(n_samples, sr, am_hz * (0.7 + 0.4 * m / max(n_smodes, 1)), rng)
            for m in range(n_smodes)
        ])

        # Stick-slip schedule: positions + burst lengths + bandpass noise.
        # Bandpass for the noise is approximated by filtering with a 2nd order
        # IIR bandpass at the surface's click_color centre (we use mode[0]
        # as a proxy — proper click_color is recoverable from the gains).
        max_burst_len = max(8, int(0.0012 * sr))
        stick_positions = rng.uniform(0, n_samples - max_burst_len, size=n_stickslip).astype(np.float32)
        stick_burst_lens = rng.uniform(0.0003, 0.0011, size=n_stickslip)
        stick_burst_lens = np.maximum(8, (stick_burst_lens * sr).astype(np.int32)).astype(np.float32)
        # Per-event noise burst — filtered through a fixed bandpass at the
        # geometric mean of [200, 8000] Hz. Material-specific colour comes
        # later by re-scaling amplitudes; the spectral shape of the noise
        # itself is held generic to keep dimensions consistent.
        click_center = np.sqrt(200.0 * 8000.0)
        click_bw = 8000.0 - 200.0
        omega = 2 * np.pi * click_center / sr
        alpha = np.sin(omega) * (click_bw / click_center) / 2
        cosw = np.cos(omega)
        a0 = 1 + alpha
        nb0 = alpha / a0
        nb2 = -alpha / a0
        na1 = -2 * cosw / a0
        na2 = (1 - alpha) / a0
        stick_noise = np.zeros((n_stickslip, max_burst_len), dtype=np.float32)
        for k in range(n_stickslip):
            z1 = z2 = 0.0
            for i in range(max_burst_len):
                x = float(rng.standard_normal())
                y = nb0 * x + z1
                z1 = -na1 * y + z2
                z2 = nb2 * x - na2 * y
                stick_noise[k, i] = y

        return cls(
            radius_mm=t(radius_mm),
            viscosity=t(viscosity),
            roll_velocity_hz=t(roll_velocity_hz, grad=False),  # treated as condition
            path_roughness=t(path_roughness),
            contact_angle_deg=t(contact_angle_deg),
            surface_tension_n_m=t(surface_tension_n_m),
            surface_modes_hz=t(surface_modes_hz),
            surface_t60s_s=t(surface_t60s_s),
            surface_gains=t(surface_gains),
            body_resonance_mix=t(0.7),
            cavity_mix=t(0.4),
            rayleigh_mix=t(0.35),
            surface_ring_mix=t(0.7),
            microbubble_mix=t(0.5),
            stickslip_mix=t(0.4),
            shimmer_depth=t(0.2),
            # Layer H (discrete ticks) is not ported to the diff engine (see
            # module docstring); kept as a non-trainable placeholder so the
            # dataclass shape matches the numpy model, but never optimised.
            discrete_mix=buf(0.5),
            # Fixed buffers
            microbubble_radii_mm=buf(radii_mm),
            microbubble_onsets=buf(onsets),
            microbubble_phases=buf(phases),
            am_lfo=buf(am_lfo),
            body_fm=buf(body_fm),
            rayleigh_triggers=buf(rayleigh_triggers),
            ring_fms=buf(ring_fms),
            stick_positions=buf(stick_positions),
            stick_burst_lens=buf(stick_burst_lens),
            stick_amplitudes=torch.full(
                (n_stickslip,), 0.5, device=device, dtype=torch.float32,
            ).requires_grad_(requires_grad),
            stick_noise=buf(stick_noise),
            shimmer_lfo=buf(shimmer_lfo),
            _sr=sr,
            _n_samples=n_samples,
        )

    def trainable(self) -> list[torch.Tensor]:
        ts = [
            self.radius_mm, self.viscosity, self.path_roughness,
            self.contact_angle_deg, self.surface_tension_n_m,
            self.surface_modes_hz, self.surface_t60s_s, self.surface_gains,
            self.body_resonance_mix, self.cavity_mix, self.rayleigh_mix,
            self.surface_ring_mix, self.microbubble_mix, self.stickslip_mix,
            self.shimmer_depth, self.stick_amplitudes,
        ]
        return [t for t in ts if t.requires_grad]

    def clamp_(self):
        with torch.no_grad():
            self.radius_mm.clamp_(0.3, 8.0)
            self.viscosity.clamp_(0.0, 1.0)
            self.path_roughness.clamp_(0.0, 1.0)
            self.contact_angle_deg.clamp_(20.0, 175.0)
            self.surface_tension_n_m.clamp_(0.02, 0.10)
            self.surface_modes_hz.clamp_(50.0, 16000.0)
            self.surface_t60s_s.clamp_(1e-3, 5.0)
            for mix in (self.body_resonance_mix, self.cavity_mix,
                        self.rayleigh_mix, self.surface_ring_mix,
                        self.microbubble_mix, self.stickslip_mix):
                mix.clamp_(0.0, 1.5)
            self.shimmer_depth.clamp_(0.0, 0.5)
            self.stick_amplitudes.clamp_(0.0, 2.0)


# ====================================================================
# Helper: slow LFO sampled with numpy at construction time
# ====================================================================
def _slow_lfo_np(n: int, sr: int, cutoff_hz: float,
                  rng: np.random.Generator) -> np.ndarray:
    """One-pole LPF on white noise, then z-normalise."""
    cutoff = max(0.5, cutoff_hz)
    a = np.exp(-2 * np.pi * cutoff / sr).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    state = 0.0
    for i in range(n):
        x = float(rng.standard_normal())
        state = a * state + (1 - a) * x
        out[i] = state
    out = (out - out.mean()) / (out.std() + 1e-9)
    return out


# ====================================================================
# Synthesis: all PyTorch, fully differentiable
# ====================================================================
def _contact_area(angle_deg: torch.Tensor) -> torch.Tensor:
    """A = (1 + cos θ) / 2, where θ is the contact angle."""
    return (1 + torch.cos(angle_deg * torch.pi / 180.0)) / 2


def synth_rolling_droplet_diff(p: RollingDropletParamsT) -> torch.Tensor:
    """Generate the rolling droplet sound, fully differentiable.

    Returns a 1-D float32 tensor of length p._n_samples.
    """
    sr = p._sr
    n = p._n_samples
    device = p.radius_mm.device
    dtype = p.radius_mm.dtype
    t_idx = torch.arange(n, device=device, dtype=dtype) / sr

    # Wetting
    contact_a = _contact_area(p.contact_angle_deg)
    one_minus_contact = 1.0 - contact_a

    # ----- Layer A: body Minnaert sustained -----
    f_minnaert = 3.26 / (p.radius_mm * 1e-3)
    # FM wobble from path_roughness
    fm_depth = 0.03 + 0.05 * p.path_roughness
    f_inst_body = f_minnaert * (1 + fm_depth * p.body_fm)
    phase_body = 2 * torch.pi * torch.cumsum(f_inst_body, dim=0) / sr
    # AM envelope (sostenida 0.75±0.25)
    am_env_body = torch.clamp(0.75 + 0.25 * p.am_lfo, min=0.0)
    # Impedance gain from mean t60 (proxy for surface acoustic impedance)
    mean_t60 = torch.clamp(p.surface_t60s_s.mean() / 0.6, max=1.0)
    impedance_gain = 0.3 + 0.7 * mean_t60
    wetting_gain = 0.4 + 0.6 * one_minus_contact
    visc_atten = (1.0 - 0.6 * p.viscosity)
    body_amp = 0.7 * visc_atten * impedance_gain * wetting_gain
    body = body_amp * am_env_body * torch.sin(phase_body)

    # ----- Layer B: Helmholtz cavity -----
    cavity_vol_factor = 0.3 + 0.7 * one_minus_contact
    f_cavity_base = torch.clamp(p.surface_modes_hz[0] * 0.6, min=120.0, max=1200.0)
    f_cavity = f_cavity_base / torch.sqrt(cavity_vol_factor)
    phase_cavity = 2 * torch.pi * f_cavity * t_idx + torch.pi / 2
    am_env_cavity = torch.clamp(0.75 + 0.25 * p.am_lfo, min=0.0)
    cavity_amp = 0.5 * visc_atten * (0.5 + 0.5 * one_minus_contact)
    cavity = cavity_amp * am_env_cavity * torch.sin(phase_cavity)

    # ----- Layer C: Rayleigh shape modes (exact physics) -----
    r_m = p.radius_mm * 1e-3
    rho = _RHO_WATER
    rayleigh = torch.zeros(n, device=device, dtype=dtype)
    base_amp = 0.4 * (0.4 + 0.6 * p.path_roughness)
    for k, mode in enumerate(range(2, 7)):
        ml = mode * (mode - 1) * (mode + 2)
        omega2 = ml * p.surface_tension_n_m / (rho * r_m ** 3)
        f_mode = torch.sqrt(torch.clamp(omega2, min=1e-9)) / (2 * torch.pi)
        # Skip modes above Nyquist
        if float(f_mode.detach()) >= sr / 2 - 50:
            continue
        mode_amp = base_amp / mode
        trig = 0.5 + 0.5 * p.rayleigh_triggers[k]
        phase_mode = 2 * torch.pi * f_mode * t_idx
        rayleigh = rayleigh + mode_amp * trig * torch.sin(phase_mode)

    # ----- Layer E: surface ringing (material voice) -----
    # Continuous bank of surface modes with t60-derived sustain
    gains_n = torch.softmax(p.surface_gains, dim=0)
    M = p.surface_modes_hz.shape[0]
    t60_factor = torch.clamp(p.surface_t60s_s.mean() / 0.6, max=1.0)
    sustain_gain = 0.2 + 0.8 * t60_factor
    ring_base = 0.55 * visc_atten * sustain_gain
    ring = torch.zeros(n, device=device, dtype=dtype)
    fm_depth_ring = 0.005 + 0.015 * p.path_roughness
    for m in range(M):
        fc = p.surface_modes_hz[m]
        if float(fc.detach()) >= sr / 2 - 100:
            continue
        f_inst = fc * (1 + fm_depth_ring * p.ring_fms[m])
        phase = 2 * torch.pi * torch.cumsum(f_inst, dim=0) / sr
        am_env = torch.clamp(0.7 + 0.3 * p.am_lfo, min=0.0)
        ring = ring + ring_base * gains_n[m] * am_env * torch.sin(phase)

    # ----- Layer F: microbubble cloud -----
    micro = torch.zeros(n, device=device, dtype=dtype)
    Nmicro = p.microbubble_radii_mm.shape[0]
    micro_base_amp = 0.5 * visc_atten / np.sqrt(max(Nmicro, 1))
    parent_r_thirds = p.radius_mm / 3.0
    am_env_micro = torch.clamp(0.7 + 0.3 * p.am_lfo, min=0.0)
    for b in range(Nmicro):
        r_b = p.microbubble_radii_mm[b]  # buffer (no grad through this)
        f_b = 3.26 / (r_b * 1e-3)
        if float(f_b.detach()) >= sr / 2 - 200:
            continue
        amp_weight = (r_b / parent_r_thirds) ** 1.5
        b_amp = micro_base_amp * amp_weight
        onset = int(p.microbubble_onsets[b].item())
        # decay_n based on the FIXED bubble radius (buffer), not current
        # learnable radius_mm (which could become NaN at exact-zero loss).
        # Damping coupled to radius via van den Doel (e-folding tau = 1/d),
        # so small microbubbles decay faster than large ones.
        r_b_buf = float(p.microbubble_radii_mm[b].item())
        decay_n = max(200, int(_bubble_efold_tau_ms(r_b_buf) * 1e-3 * sr))
        # Build oscillator from onset to end (differentiable through f_b and amp)
        idxs = torch.arange(n - onset, device=device, dtype=dtype)
        decay = torch.exp(-idxs / decay_n)
        phase = p.microbubble_phases[b] + 2 * torch.pi * f_b * idxs / sr
        signal = b_amp * decay * am_env_micro[onset:] * torch.sin(phase)
        # Zero-pad onset
        padded = torch.cat([torch.zeros(onset, device=device, dtype=dtype), signal])
        micro = micro + padded

    # ----- Layer G: stick-slip events -----
    stick = torch.zeros(n, device=device, dtype=dtype)
    visc_stick = 1.0 - 0.7 * p.viscosity
    stick_gain = visc_stick * p.path_roughness
    Nstick = p.stick_positions.shape[0]
    max_burst = p.stick_noise.shape[1]
    for k in range(Nstick):
        start = int(p.stick_positions[k].item())
        burst_len = int(p.stick_burst_lens[k].item())
        if start + burst_len > n:
            burst_len = n - start
        if burst_len <= 0:
            continue
        # Envelope (decaying)
        idxs = torch.arange(burst_len, device=device, dtype=dtype)
        env = torch.exp(-4 * idxs / burst_len)
        # Pre-baked filtered noise × differentiable amplitude
        noise_slice = p.stick_noise[k, :burst_len]
        stick[start:start + burst_len] = (
            stick[start:start + burst_len]
            + stick_gain * p.stick_amplitudes[k] * env * noise_slice
        )

    # ----- Mix continuous layers -----
    out = (
        p.body_resonance_mix * body
        + p.cavity_mix * cavity
        + p.rayleigh_mix * rayleigh
        + p.surface_ring_mix * ring
        + p.microbubble_mix * micro
        + p.stickslip_mix * stick
    )

    # ----- Layer D: shimmer (post-process AM) -----
    shimmer_mult = torch.clamp(1.0 + p.shimmer_depth * p.shimmer_lfo, min=0.0)
    out = out * shimmer_mult

    return out
