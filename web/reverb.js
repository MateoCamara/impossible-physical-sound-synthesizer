// Reverb in Web Audio: synthetic IRs (noise * exp decay + early reflections)
// rendered into AudioBuffers, then convolved via ConvolverNode in an
// OfflineAudioContext so the wet/dry mix can be produced as a single buffer.

export const IR_PRESETS = {
  dry:         { t60_s: 0.05, pre_delay_ms: 0,  early_reflections_ms: [],            early_gain: 0,    color_lpf_hz: 20000 },
  small_room:  { t60_s: 0.4,  pre_delay_ms: 5,  early_reflections_ms: [8, 14, 22, 33, 48],     early_gain: 0.6,  color_lpf_hz: 8000 },
  medium_hall: { t60_s: 1.3,  pre_delay_ms: 20, early_reflections_ms: [28, 45, 68, 95, 130],   early_gain: 0.5,  color_lpf_hz: 5000 },
  cathedral:   { t60_s: 4.5,  pre_delay_ms: 40, early_reflections_ms: [55, 90, 140, 220, 310], early_gain: 0.35, color_lpf_hz: 3000 },
  cave:        { t60_s: 2.8,  pre_delay_ms: 25, early_reflections_ms: [40, 80, 135, 200, 290], early_gain: 0.45, color_lpf_hz: 2500 },
  exterior:    { t60_s: 0.2,  pre_delay_ms: 2,  early_reflections_ms: [120, 250],              early_gain: 0.15, color_lpf_hz: 6000 },
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

/**
 * Build a synthetic IR buffer.
 * @param {BaseAudioContext} ctx
 * @param {string} preset_name
 * @param {number} seed
 * @returns {AudioBuffer}
 */
export function generateIR(ctx, preset_name = "medium_hall", seed = 0) {
  const p = IR_PRESETS[preset_name];
  if (!p) throw new Error(`Unknown IR preset ${preset_name}`);
  const sr = ctx.sampleRate;
  const total_s = p.pre_delay_ms / 1000 + p.t60_s * 1.5;
  const total_n = Math.max(8, Math.floor(total_s * sr));
  const buf = ctx.createBuffer(1, total_n, sr);
  const data = buf.getChannelData(0);
  const rng = rand(seed);
  const pre_n = Math.floor((p.pre_delay_ms / 1000) * sr);

  // Early reflections
  for (const ms of p.early_reflections_ms) {
    const idx = pre_n + Math.floor((ms / 1000) * sr);
    if (idx < total_n) {
      const sign = rng() > 0.3 ? 1 : -1;
      data[idx] += sign * p.early_gain * (0.5 + 0.5 * rng());
    }
  }

  // Late tail: white noise × exponential decay
  const last_er = p.early_reflections_ms.length
    ? Math.max(...p.early_reflections_ms)
    : 10;
  const tail_start = pre_n + Math.floor((last_er / 1000) * sr);
  if (tail_start < total_n) {
    const decay_factor = -6.907755 / (p.t60_s * sr + 1e-6);
    for (let i = tail_start; i < total_n; i++) {
      const noise = (rng() * 2 - 1);
      const env = Math.exp(decay_factor * (i - tail_start));
      data[i] += noise * env * 0.6;
    }
  }

  // Single-pole LPF for warmth (color_lpf_hz)
  if (p.color_lpf_hz < sr / 2 - 100) {
    const rc = 1 / (2 * Math.PI * p.color_lpf_hz);
    const dt = 1 / sr;
    const alpha = dt / (rc + dt);
    let y = 0;
    for (let i = 0; i < total_n; i++) {
      y = y + alpha * (data[i] - y);
      data[i] = y;
    }
  }

  // Normalise IR to peak ~0.3 to avoid clipping after convolution
  let peak = 0;
  for (let i = 0; i < total_n; i++) if (Math.abs(data[i]) > peak) peak = Math.abs(data[i]);
  if (peak > 1e-9) {
    const scale = 0.3 / peak;
    for (let i = 0; i < total_n; i++) data[i] *= scale;
  }
  return buf;
}


/**
 * Apply a reverb IR to a dry buffer offline and return the wet result.
 * @param {AudioContext} liveCtx  the live AudioContext (for sampleRate)
 * @param {AudioBuffer} dry
 * @param {AudioBuffer} ir
 * @param {number} mix  0..1 dry/wet blend
 * @returns {Promise<AudioBuffer>}
 */
export async function applyReverb(liveCtx, dry, ir, mix = 0.4) {
  const sr = dry.sampleRate;
  const len = dry.length + ir.length;
  const offline = new OfflineAudioContext(1, len, sr);

  const drySrc = offline.createBufferSource();
  drySrc.buffer = dry;

  const dryGain = offline.createGain();
  dryGain.gain.value = 1 - mix;
  drySrc.connect(dryGain).connect(offline.destination);

  // Wet path: dry -> convolver -> wetGain -> output
  const convolver = offline.createConvolver();
  convolver.buffer = ir;
  const wetGain = offline.createGain();
  wetGain.gain.value = mix;
  drySrc.connect(convolver).connect(wetGain).connect(offline.destination);

  drySrc.start();
  const rendered = await offline.startRendering();
  // Trim back to dry.length for convenience (caller can still hear the tail
  // by passing trim=false at the call site; here we keep tail to preserve
  // natural ambience).
  return rendered;
}
