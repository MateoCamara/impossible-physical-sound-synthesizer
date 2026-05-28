// Modal synthesis in Web Audio API: bank of biquad resonators excited by
// a short noise burst. Replicates a subset of impossible_mix.physics.modal.
//
// Material profiles mirror the Python PROFILES dict by name. Each profile
// declares fundamental frequency, harmonic spacing, t60, n_modes, and a
// spectral tilt that decides relative gain of each mode.

export const MODAL_PROFILES = {
  metal:  { fundamental_hz: 900,  spacing: 1.85, damping_ms: 800, n_modes: 8,
            tilt_db_oct: -3.0, inharmonicity: 0.05 },
  rock:   { fundamental_hz: 350,  spacing: 2.30, damping_ms: 80,  n_modes: 6,
            tilt_db_oct: -3.0, inharmonicity: 0.60 },
  wood:   { fundamental_hz: 280,  spacing: 1.70, damping_ms: 200, n_modes: 5,
            tilt_db_oct: -3.0, inharmonicity: 0.20 },
  glass:  { fundamental_hz: 1800, spacing: 1.95, damping_ms: 1200, n_modes: 10,
            tilt_db_oct: -1.5, inharmonicity: 0.02 },
  earth:  { fundamental_hz: 120,  spacing: 2.00, damping_ms: 30,  n_modes: 4,
            tilt_db_oct: -3.0, inharmonicity: 0.80 },
  fabric: { fundamental_hz: 200,  spacing: 2.50, damping_ms: 15,  n_modes: 3,
            tilt_db_oct: -3.0, inharmonicity: 0.50 },
  rubber: { fundamental_hz: 160,  spacing: 2.20, damping_ms: 30,  n_modes: 4,
            tilt_db_oct: -3.0, inharmonicity: 0.40 },
  bone:   { fundamental_hz: 520,  spacing: 1.90, damping_ms: 120, n_modes: 5,
            tilt_db_oct: -3.0, inharmonicity: 0.25 },
  ice:    { fundamental_hz: 1500, spacing: 1.92, damping_ms: 900, n_modes: 7,
            tilt_db_oct: -1.5, inharmonicity: 0.10 },
  chitin: { fundamental_hz: 950,  spacing: 2.10, damping_ms: 180, n_modes: 6,
            tilt_db_oct: -3.0, inharmonicity: 0.30 },
  plasma: { fundamental_hz: 400,  spacing: 2.10, damping_ms: 400, n_modes: 12,
            tilt_db_oct: 1.5,  inharmonicity: 0.95 },
};

function rand(seed) {
  // Mulberry32 PRNG (deterministic seeds for reproducibility)
  let t = (seed | 0);
  return function () {
    t = (t + 0x6D2B79F5) | 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function modalFrequencies(profile, seed = 0) {
  const rng = rand(seed);
  const freqs = [];
  for (let k = 0; k < profile.n_modes; k++) {
    const base = profile.fundamental_hz * Math.pow(profile.spacing, k);
    const jitter = (rng() * 2 - 1) * profile.inharmonicity;
    freqs.push(base * (1 + 0.3 * jitter));
  }
  return freqs;
}

function gainCurve(n, tilt_db_oct) {
  const gains = [];
  for (let k = 0; k < n; k++) {
    gains.push(Math.pow(10, (tilt_db_oct * k) / 20));
  }
  const sum = gains.reduce((a, b) => a + b, 0) || 1;
  return gains.map((g) => g / sum);
}

/**
 * Render a modal impact into an AudioBuffer.
 *
 * @param {AudioContext} ctx
 * @param {object} opts
 *   profile_name: key into MODAL_PROFILES
 *   duration_s: total length of the rendered audio
 *   impact_strength: 0..1
 *   sharpness: 0.3..3 (lower = softer attack)
 *   velocity: 0.5..2 (modulates strength and shifts pitch slightly)
 *   damping_anisotropy: 0..1 (higher = upper modes die faster)
 *   excitation_shape: 'felt' | 'wood' | 'steel' | 'impulse'
 *   seed: int
 * @returns {Promise<AudioBuffer>}
 */
export async function renderModalImpact(ctx, opts) {
  const {
    profile_name = "metal",
    duration_s = 2.0,
    impact_strength = 0.9,
    sharpness = 1.0,
    velocity = 1.0,
    damping_anisotropy = 0.5,
    excitation_shape = "felt",
    seed = 0,
  } = opts;
  const profile = MODAL_PROFILES[profile_name];
  if (!profile) throw new Error(`Unknown profile ${profile_name}`);

  const sr = ctx.sampleRate;
  const n = Math.floor(duration_s * sr);
  const offline = new OfflineAudioContext(1, n, sr);
  const rng = rand(seed);

  // Build exciter buffer (short noise burst with shape envelope).
  let excLen;
  if (excitation_shape === "felt") {
    excLen = Math.max(8, Math.floor((0.005 / Math.max(sharpness, 0.1)) * sr));
  } else if (excitation_shape === "wood") {
    excLen = Math.max(2, Math.floor((0.0008 / Math.max(sharpness, 0.1)) * sr));
  } else if (excitation_shape === "steel") {
    excLen = Math.max(2, Math.floor((0.0003 / Math.max(sharpness, 0.1)) * sr));
  } else {
    excLen = 2;
  }
  const excBuf = offline.createBuffer(1, excLen, sr);
  const exc = excBuf.getChannelData(0);
  const effStrength = impact_strength * Math.max(0.3, Math.min(2.0, velocity));
  for (let i = 0; i < excLen; i++) {
    let env = 1;
    if (excitation_shape === "felt") {
      env = 0.5 * (1 - Math.cos((2 * Math.PI * i) / excLen));
    } else {
      env = Math.exp((-4 * i) / excLen);
    }
    exc[i] = env * (0.7 + 0.6 * (rng() - 0.5)) * effStrength;
  }

  // Schedule one source per mode, each routed through a peaking BiquadFilter.
  const source = offline.createBufferSource();
  source.buffer = excBuf;

  const freqsBase = modalFrequencies(profile, seed);
  const pitchFactor = 1.0 + 0.04 * (velocity - 1.0);
  const gains = gainCurve(profile.n_modes, profile.tilt_db_oct);

  const sum = offline.createGain();
  sum.gain.value = 0.5;
  sum.connect(offline.destination);

  for (let k = 0; k < profile.n_modes; k++) {
    const fh = freqsBase[k] * pitchFactor;
    if (fh <= 0 || fh >= sr / 2 - 50) continue;
    const rel = fh / profile.fundamental_hz;
    const anisoExp = 0.3 + 0.6 * damping_anisotropy;
    const t60 = (profile.damping_ms / 1000) / Math.max(Math.pow(rel, anisoExp), 1.0);
    // Convert t60 (-60 dB time) into BiquadFilter Q approx: Q ≈ pi * fh * t60.
    const Q = Math.max(0.5, Math.PI * fh * t60);
    const filt = offline.createBiquadFilter();
    filt.type = "bandpass";
    filt.frequency.value = fh;
    filt.Q.value = Q;
    const modeGain = offline.createGain();
    modeGain.gain.value = gains[k] * 20.0; // boost bandpass output
    source.connect(filt).connect(modeGain).connect(sum);
  }

  source.start(0);
  const rendered = await offline.startRendering();
  return rendered;
}

/**
 * Render a modal roll: rapid train of damped modal impulses to simulate
 * a rigid body rolling on a surface (each impulse = one micro-contact).
 * Mirrors impossible_mix.physics.modal.synth_modal_roll.
 *
 * @param {AudioContext} ctx
 * @param {object} opts
 *   profile_name, duration_s, rate_hz, jitter, strength, seed
 * @returns {Promise<AudioBuffer>}
 */
export async function renderModalRoll(ctx, opts) {
  const {
    profile_name = "metal",
    duration_s = 2.0,
    rate_hz = 20.0,
    jitter = 0.3,
    strength = 0.6,
    seed = 0,
  } = opts;
  const profile = MODAL_PROFILES[profile_name];
  if (!profile) throw new Error(`Unknown profile ${profile_name}`);
  const sr = ctx.sampleRate;
  const n = Math.floor(duration_s * sr);
  const offline = new OfflineAudioContext(1, n, sr);
  const rng = rand(seed);

  // Build a dense noise buffer with impulses at quasi-periodic times.
  const noiseBuf = offline.createBuffer(1, n, sr);
  const noiseData = noiseBuf.getChannelData(0);
  const period = sr / Math.max(rate_hz, 0.1);
  let t = 0;
  while (t < n) {
    const idx = Math.floor(t);
    if (idx >= n) break;
    const burstLen = Math.max(2, Math.floor(0.0008 * sr));
    const amp = strength * (0.6 + 0.4 * rng());
    for (let i = 0; i < burstLen; i++) {
      if (idx + i < n) noiseData[idx + i] += (rng() * 2 - 1) * amp;
    }
    const off = period * (1 + jitter * (rng() * 2 - 1));
    t += Math.max(period * 0.1, off);
  }
  const source = offline.createBufferSource();
  source.buffer = noiseBuf;

  // Modal bandpass bank (same as renderModalImpact)
  const freqsBase = modalFrequencies(profile, seed);
  const gains = gainCurve(profile.n_modes, profile.tilt_db_oct);
  const sum = offline.createGain();
  sum.gain.value = 0.4;
  sum.connect(offline.destination);
  for (let k = 0; k < profile.n_modes; k++) {
    const fh = freqsBase[k];
    if (fh <= 0 || fh >= sr / 2 - 50) continue;
    const t60 = profile.damping_ms / 1000;
    const Q = Math.max(0.5, Math.PI * fh * t60);
    const filt = offline.createBiquadFilter();
    filt.type = "bandpass";
    filt.frequency.value = fh;
    filt.Q.value = Q;
    const modeGain = offline.createGain();
    modeGain.gain.value = gains[k] * 15.0;
    source.connect(filt).connect(modeGain).connect(sum);
  }
  source.start(0);
  return await offline.startRendering();
}

