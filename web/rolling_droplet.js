// Rolling droplet in Web Audio: quasi-periodic train of drip events.
//
// Instead of orchestrating one OfflineAudioContext per drip (drip.js uses
// that path because it returns a complete buffer), here we write each
// contact's signal directly into a shared output AudioBuffer. Same
// physics as Python rolling_droplet (Minnaert chirp + decay + surface
// modal tail), but inlined for efficiency with 50-300 contacts per clip.

const SURFACE_PROFILES = {
  fabric:  { modes: [180, 320],                    gains: [0.7, 0.3],            t60_ms: 8,   click_color: [500, 2500] },
  wood:    { modes: [280, 720, 1450, 2400],        gains: [0.4, 0.3, 0.2, 0.1],  t60_ms: 80,  click_color: [800, 5000] },
  ceramic: { modes: [1100, 2400, 4800, 7200],      gains: [0.35, 0.3, 0.2, 0.15], t60_ms: 350, click_color: [2000, 9000] },
  glass:   { modes: [1800, 4200, 7100, 9800],      gains: [0.3, 0.3, 0.25, 0.15], t60_ms: 600, click_color: [3000, 10000] },
  metal:   { modes: [900, 2200, 5100, 8800],       gains: [0.35, 0.3, 0.2, 0.15], t60_ms: 900, click_color: [3500, 11000] },
  stone:   { modes: [380, 880, 1800],              gains: [0.5, 0.3, 0.2],       t60_ms: 60,  click_color: [1000, 5000] },
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
 * Continuous rolling rumble layer (anti 'tacatacataca').
 *
 * Mirrors impossible_mix.physics.droplet._continuous_roll_layer: bandpass-
 * filtered noise excited through the surface modal bank, modulated by a
 * slow velocity envelope derived from path_roughness + roll_velocity_hz.
 * Implemented with an OfflineAudioContext + BiquadFilter chain so it
 * doesn't need inline IIR code in JS.
 *
 * @param {number} sr
 * @param {number} duration_s
 * @param {object} opts
 *   surface_profile, viscosity, roll_velocity_hz, path_roughness, seed,
 *   body_resonance_strength
 * @returns {Promise<AudioBuffer>}
 */
async function renderContinuousRollLayer(sr, duration_s, opts) {
  const {
    surface_profile = "ceramic",
    viscosity = 0.0,
    roll_velocity_hz = 14.0,
    path_roughness = 0.35,
    seed = 0,
    body_resonance_strength = 0.6,
  } = opts;
  const surf = SURFACE_PROFILES[surface_profile] || SURFACE_PROFILES.ceramic;
  const n = Math.max(64, Math.floor(duration_s * sr));
  const rng = rand(seed + 12345);

  const offline = new OfflineAudioContext(1, n, sr);

  // 1) Pre-bake noise * velocity envelope (one-pole LPF on noise for the LFO)
  const noiseBuf = offline.createBuffer(1, n, sr);
  const data = noiseBuf.getChannelData(0);
  const baseLevel = 0.4 + 0.6 * Math.min(roll_velocity_hz / 25.0, 1.0);
  const lfoCutoff = Math.max(0.5, 2.0 + 6.0 * path_roughness);
  const a = Math.exp(-2 * Math.PI * lfoCutoff / sr);
  let lfoState = 0;
  for (let i = 0; i < n; i++) {
    const x = rng() * 2 - 1;
    lfoState = a * lfoState + (1 - a) * x;
    const env = Math.max(0, Math.min(1.3, baseLevel + 0.3 * path_roughness * lfoState));
    data[i] = (rng() * 2 - 1) * env;
  }

  // 2) Click-color bandpass: material's broadband character
  const src = offline.createBufferSource();
  src.buffer = noiseBuf;
  const [clLo, clHiRaw] = surf.click_color;
  const clHi = Math.min(clHiRaw, sr / 2 - 200);
  const clCenter = Math.sqrt(clLo * clHi);
  const clQ = Math.max(0.4, clCenter / Math.max(1, clHi - clLo));
  const bp = offline.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = clCenter;
  bp.Q.value = clQ;

  // 3) Body modal bank: series of peaking filters per mode (boosts each
  // resonance without zeroing out the others — they share the chain)
  let node = bp;
  for (let m = 0; m < surf.modes.length; m++) {
    const fc = surf.modes[m];
    if (fc <= 0 || fc >= sr / 2 - 100) continue;
    const peak = offline.createBiquadFilter();
    peak.type = "peaking";
    peak.frequency.value = fc;
    peak.Q.value = 6.0;
    peak.gain.value = (6.0 + 8.0 * surf.gains[m]) * body_resonance_strength;
    node.connect(peak);
    node = peak;
  }

  // 4) Output gain (viscosity attenuates, like in Python)
  const outGain = offline.createGain();
  outGain.gain.value = 0.45 * (1.0 - 0.5 * viscosity);
  node.connect(outGain).connect(offline.destination);
  src.connect(bp);
  src.start();
  return await offline.startRendering();
}


/**
 * Render a rolling droplet (quasi-periodic train of drips) into a new
 * AudioBuffer. Combines a discrete drip-event train with a continuous
 * rumble layer that fills in the silences between contacts so the
 * percept is "rrrrr with ticks" rather than "tacatacataca".
 *
 * @param {BaseAudioContext} ctx
 * @param {object} opts
 *   radius_mm, viscosity, surface_profile
 *   roll_velocity_hz: contacts per second
 *   path_roughness: 0=metronome, 1=heavy jitter
 *   continuous_layer_mix: 0=only discrete drips, 1=heavy rumble
 *   body_resonance_strength: how loud the modal body response is in the rumble
 *   duration_s, seed
 * @returns {Promise<AudioBuffer>}
 */
export async function renderRollingDroplet(ctx, opts) {
  const {
    radius_mm = 2.0,
    viscosity = 0.0,
    surface_profile = "ceramic",
    roll_velocity_hz = 14.0,
    path_roughness = 0.35,
    continuous_layer_mix = 0.4,
    body_resonance_strength = 0.6,
    duration_s = 5.0,
    seed = 0,
  } = opts;
  const sr = ctx.sampleRate;
  const nTotal = Math.floor(duration_s * sr);
  const buf = ctx.createBuffer(1, nTotal, sr);
  const data = buf.getChannelData(0);
  const rng = rand(seed);
  const periodSamples = sr / Math.max(roll_velocity_hz, 0.1);

  // 1) Discrete drip-event train (inline)
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

  // 2) Continuous rolling-rumble layer, modulated by RMS of drip train
  if (continuous_layer_mix > 0.01) {
    const layerBuf = await renderContinuousRollLayer(sr, duration_s, {
      surface_profile, viscosity, roll_velocity_hz, path_roughness, seed,
      body_resonance_strength,
    });
    const layer = layerBuf.getChannelData(0);

    // Sliding-window RMS of drip-train data (40 ms)
    const winN = Math.max(1, Math.floor(0.04 * sr));
    let sumSq = 0;
    for (let i = 0; i < Math.min(winN, nTotal); i++) sumSq += data[i] * data[i];
    const env = new Float32Array(nTotal);
    let maxRms = 0;
    for (let i = 0; i < nTotal; i++) {
      env[i] = Math.sqrt(Math.max(0, sumSq) / winN);
      if (env[i] > maxRms) maxRms = env[i];
      sumSq -= data[i] * data[i];
      if (i + winN < nTotal) sumSq += data[i + winN] * data[i + winN];
    }
    const floorVal = 0.3 + 0.5 * Math.min(roll_velocity_hz / 25.0, 1.0);
    const maxRmsInv = 1.0 / (maxRms + 1e-9);
    const layerLen = Math.min(layer.length, nTotal);
    for (let i = 0; i < layerLen; i++) {
      const rmsNorm = env[i] * maxRmsInv;
      const envelope = floorVal + (1.0 - floorVal) * rmsNorm;
      data[i] += layer[i] * envelope * continuous_layer_mix;
    }
  }

  // 3) Peak-normalise
  let peak = 0;
  for (let i = 0; i < nTotal; i++) if (Math.abs(data[i]) > peak) peak = Math.abs(data[i]);
  if (peak > 0.95) {
    const scale = 0.95 / peak;
    for (let i = 0; i < nTotal; i++) data[i] *= scale;
  }
  return buf;
}
