// Friction/scrape in Web Audio.
//
// Pipeline mirrors a simplified version of impossible_mix.physics.friction:
//   1. White noise (pre-generated buffer)
//   2. Velocity envelope (smoothed random LFO; per-sample gain modulation)
//   3. Bandpass filter whose centre/width depend on surface_hardness
//   4. Body resonator (BiquadFilter peaking at body_freq_hz)
//   5. Optional roughness: micro-spikes added to the velocity envelope
//
// Like granular, the heavy lifting is done by writing the output buffer
// in JS rather than scheduling many AudioNodes.

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
 * Render a scrape/drag event into an AudioBuffer.
 *
 * @param {BaseAudioContext} ctx
 * @param {object} opts
 *   surface_hardness: 0..1 -> shifts bandpass centre 700..6000 Hz
 *   velocity_mean:    0..1 -> overall amplitude
 *   velocity_jitter:  0..1 -> wiggle amount of the velocity envelope
 *   roughness:        0..1 -> micro-spike density
 *   pressure:         0..1 -> output scale
 *   body_freq_hz:     resonant frequency of the material body
 *   body_q:           Q of the body peaking filter
 *   duration_s
 *   seed
 * @returns {Promise<AudioBuffer>}
 */
export async function renderScrape(ctx, opts) {
  const {
    surface_hardness = 0.5,
    velocity_mean = 0.6,
    velocity_jitter = 0.4,
    roughness = 0.3,
    pressure = 0.6,
    body_freq_hz = 1200.0,
    body_q = 4.0,
    duration_s = 3.0,
    seed = 0,
  } = opts;

  const sr = ctx.sampleRate;
  const n = Math.floor(duration_s * sr);

  // 1) Noise buffer with embedded velocity envelope + micro-spikes
  const noiseBuf = ctx.createBuffer(1, n, sr);
  const noiseData = noiseBuf.getChannelData(0);
  const rng = rand(seed);

  // Velocity envelope: low-frequency smoothed random (one-pole LPF on noise)
  const cutoffHz = 1 + 8 * velocity_jitter;
  const a = Math.exp(-2 * Math.PI * cutoffHz / sr);
  let env = 0.5;
  const envSig = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = rng() * 2 - 1;
    env = a * env + (1 - a) * x;
    envSig[i] = velocity_mean + 0.4 * velocity_jitter * env;
    if (envSig[i] < 0) envSig[i] = 0;
    if (envSig[i] > 1.5) envSig[i] = 1.5;
  }

  // Sprinkle roughness micro-spikes onto the envelope
  if (roughness > 0.05) {
    const spikeRate = 8 + 80 * roughness;
    const nSpikes = Math.floor(duration_s * spikeRate);
    for (let k = 0; k < nSpikes; k++) {
      const idx = Math.floor(rng() * (n - 1));
      const amp = (1 + 1.5 * roughness) * (0.5 + 0.5 * rng());
      envSig[idx] = Math.min(1.5, envSig[idx] * amp);
    }
  }

  // Fill noise buffer modulated by envelope
  for (let i = 0; i < n; i++) {
    noiseData[i] = (rng() * 2 - 1) * envSig[i];
  }

  // 2) Bandpass + body resonator via OfflineAudioContext + BiquadFilterNodes
  const offline = new OfflineAudioContext(1, n, sr);
  const src = offline.createBufferSource();
  src.buffer = noiseBuf;

  // Bandpass whose centre rises with surface_hardness
  const bpCentre = 1100 + 4900 * surface_hardness;
  const bp = offline.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = bpCentre;
  bp.Q.value = 1.5 + 2 * surface_hardness;

  // Body resonator (peaking)
  const body = offline.createBiquadFilter();
  body.type = "peaking";
  body.frequency.value = body_freq_hz;
  body.gain.value = 12;
  body.Q.value = Math.max(0.5, body_q);

  const outGain = offline.createGain();
  outGain.gain.value = 0.3 + 0.7 * pressure;

  src.connect(bp).connect(body).connect(outGain).connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();

  // Peak-normalise to avoid clipping after reverb
  const ch = rendered.getChannelData(0);
  let peak = 0;
  for (let i = 0; i < ch.length; i++) if (Math.abs(ch[i]) > peak) peak = Math.abs(ch[i]);
  if (peak > 0.95) {
    const scale = 0.95 / peak;
    for (let i = 0; i < ch.length; i++) ch[i] *= scale;
  }
  return rendered;
}
