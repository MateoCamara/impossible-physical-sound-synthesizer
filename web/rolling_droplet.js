// Rolling droplet in Web Audio: quasi-periodic train of drip events.
//
// Instead of orchestrating one OfflineAudioContext per drip (drip.js uses
// that path because it returns a complete buffer), here we write each
// contact's signal directly into a shared output AudioBuffer. Same
// physics as Python rolling_droplet (Minnaert chirp + decay + surface
// modal tail), but inlined for efficiency with 50-300 contacts per clip.

const SURFACE_PROFILES = {
  fabric:  { modes: [180, 320],                    gains: [0.7, 0.3],            t60_ms: 8 },
  wood:    { modes: [280, 720, 1450, 2400],        gains: [0.4, 0.3, 0.2, 0.1],  t60_ms: 80 },
  ceramic: { modes: [1100, 2400, 4800, 7200],      gains: [0.35, 0.3, 0.2, 0.15], t60_ms: 350 },
  glass:   { modes: [1800, 4200, 7100, 9800],      gains: [0.3, 0.3, 0.25, 0.15], t60_ms: 600 },
  metal:   { modes: [900, 2200, 5100, 8800],       gains: [0.35, 0.3, 0.2, 0.15], t60_ms: 900 },
  stone:   { modes: [380, 880, 1800],              gains: [0.5, 0.3, 0.2],       t60_ms: 60 },
};

function rand(seed) {
  let t = seed | 0;
  return function () {
    t = (t + 0x6D2B79F5) | 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function minnaertHz(radius_mm) {
  return 3.26 / (Math.max(radius_mm, 0.1) * 1e-3);
}

/**
 * Write one drip "event" directly into the destination Float32Array
 * starting at sampleStart. Mutates dst in place.
 *
 * This is a CPU-side, math-only renderer (no OfflineAudioContext). It is
 * the engine of renderRollingDroplet but can be reused elsewhere.
 *
 * @param {Float32Array} dst
 * @param {number} sampleStart
 * @param {number} sr
 * @param {object} opts
 *   radius_mm, viscosity, surface_profile, velocity_factor, seed
 */
function writeDripInline(dst, sampleStart, sr, opts) {
  const {
    radius_mm = 2.0,
    viscosity = 0.0,
    surface_profile = "ceramic",
    velocity_factor = 1.0,
    seed = 0,
  } = opts;
  const surf = SURFACE_PROFILES[surface_profile] || SURFACE_PROFILES.ceramic;
  const rng = rand(seed);

  const fM = minnaertHz(radius_mm);
  const fStart = fM * 0.45;
  const fEnd = fM * 1.6 * (1.0 + 0.15 * (rng() * 2 - 1));
  const chirpDurMs = (15 + 8 * radius_mm) * (1 + 1.5 * viscosity) *
                      (0.7 + 0.6 / Math.max(velocity_factor, 0.3));
  const chirpDurS = chirpDurMs / 1000;
  const decayMs = (50 + 30 * radius_mm) * (1 - 0.6 * viscosity);
  const totalS = chirpDurS + decayMs / 1000;
  const totalN = Math.max(8, Math.floor(totalS * sr));
  if (sampleStart >= dst.length) return;

  // 1) Chirp: exponential frequency ramp + envelope
  const attackN = Math.max(2, Math.floor(0.001 * sr));
  const chirpN = Math.max(8, Math.floor(chirpDurS * sr));
  let phase = 0;
  const ratio = Math.max(1.001, fEnd / Math.max(fStart, 1));
  const log_ratio = Math.log(ratio);
  for (let i = 0; i < totalN; i++) {
    const dst_idx = sampleStart + i;
    if (dst_idx >= dst.length) break;
    // Frequency at sample i (exponential ramp clamped after chirpN)
    const t = i / sr;
    const t_norm = Math.min(1.0, t / Math.max(chirpDurS, 1e-4));
    const f = fStart * Math.exp(log_ratio * t_norm);
    phase += 2 * Math.PI * f / sr;
    // Envelope: short attack then exponential decay
    let env;
    if (i < attackN) {
      env = Math.pow(i / attackN, 0.7);
    } else {
      env = Math.exp(-4 * (1 - viscosity * 0.5) * (i - attackN) / chirpN);
    }
    dst[dst_idx] += 0.7 * velocity_factor * env * Math.sin(phase);
  }

  // 2) Surface modal tail: damped sinusoids per mode
  const tau_surf = (surf.t60_ms / 1000) / 6.907755;
  const decay_surf = Math.exp(-1 / (tau_surf * sr));  // per-sample multiplier
  for (let m = 0; m < surf.modes.length; m++) {
    const fc = surf.modes[m];
    if (fc <= 0 || fc >= sr / 2 - 100) continue;
    const g = surf.gains[m] * 0.4 * velocity_factor * (0.3 + 0.7 * 0.5);
    // We compute the impulse response analytically: noise burst at t=0
    // feeds a single-mode resonator. For inline efficiency we generate
    // a damped sinusoid directly.
    const omega = 2 * Math.PI * fc / sr;
    let amp = g;
    for (let i = 0; i < totalN; i++) {
      const dst_idx = sampleStart + i;
      if (dst_idx >= dst.length) break;
      dst[dst_idx] += amp * Math.sin(omega * i);
      amp *= decay_surf;
    }
  }
}

/**
 * Render a rolling droplet (quasi-periodic train of drips) into a new
 * AudioBuffer.
 *
 * @param {BaseAudioContext} ctx
 * @param {object} opts
 *   radius_mm, viscosity, surface_profile
 *   roll_velocity_hz: contacts per second
 *   path_roughness: 0=metronome, 1=heavy jitter
 *   duration_s, seed
 * @returns {AudioBuffer}
 */
export function renderRollingDroplet(ctx, opts) {
  const {
    radius_mm = 2.0,
    viscosity = 0.0,
    surface_profile = "ceramic",
    roll_velocity_hz = 14.0,
    path_roughness = 0.35,
    duration_s = 5.0,
    seed = 0,
  } = opts;
  const sr = ctx.sampleRate;
  const nTotal = Math.floor(duration_s * sr);
  const buf = ctx.createBuffer(1, nTotal, sr);
  const data = buf.getChannelData(0);
  const rng = rand(seed);
  const periodSamples = sr / Math.max(roll_velocity_hz, 0.1);

  let t = 0;
  let count = 0;
  while (t < nTotal) {
    const start = Math.floor(t);
    if (start >= nTotal) break;
    const velocity = Math.max(0.4, Math.min(1.8,
      1.0 + (rng() * 2 - 1) * (0.25 + 0.4 * path_roughness)));
    writeDripInline(data, start, sr, {
      radius_mm, viscosity, surface_profile,
      velocity_factor: velocity, seed: seed + count * 7,
    });
    const offset = periodSamples * (1 + path_roughness * (rng() * 2 - 1) * 0.7);
    t += Math.max(periodSamples * 0.1, offset);
    count++;
    if (count > 600) break;  // safety cap
  }

  // Peak-normalise
  let peak = 0;
  for (let i = 0; i < nTotal; i++) if (Math.abs(data[i]) > peak) peak = Math.abs(data[i]);
  if (peak > 0.95) {
    const scale = 0.95 / peak;
    for (let i = 0; i < nTotal; i++) data[i] *= scale;
  }
  return buf;
}
