// Friction/scrape in Web Audio.
//
// Pipeline mirrors impossible_mix.physics.friction.synth_scrape:
//   1. White noise bandpass-filtered by the surface click_color
//   2. Velocity envelope with stick-slip dynamics (discrete slip bumps)
//   3. Multi-mode body response: peaking filters per surface mode
//   4. Roughness micro-impacts: real short grain spikes modulated by
//      local velocity
//
// Uses OfflineAudioContext for the bandpass + modal chain; the velocity
// envelope and micro-spikes are baked into the noise buffer beforehand.

const SURFACE_PROFILES = {
  fabric:  { modes: [180, 320],               gains: [0.7, 0.3],            t60_ms: 8,   click_color: [500, 2500] },
  wood:    { modes: [280, 720, 1450, 2400],   gains: [0.4, 0.3, 0.2, 0.1], t60_ms: 80,  click_color: [800, 5000] },
  ceramic: { modes: [1100, 2400, 4800, 7200], gains: [0.35, 0.3, 0.2, 0.15], t60_ms: 350, click_color: [2000, 9000] },
  glass:   { modes: [1800, 4200, 7100, 9800], gains: [0.3, 0.3, 0.25, 0.15], t60_ms: 600, click_color: [3000, 10000] },
  metal:   { modes: [900, 2200, 5100, 8800],  gains: [0.35, 0.3, 0.2, 0.15], t60_ms: 900, click_color: [3500, 11000] },
  stone:   { modes: [380, 880, 1800],         gains: [0.5, 0.3, 0.2],       t60_ms: 60,  click_color: [1000, 5000] },
  water:   { modes: [420, 900],               gains: [0.7, 0.3],            t60_ms: 25,  click_color: [400, 2500] },
  rubber:  { modes: [110, 250],               gains: [0.7, 0.3],            t60_ms: 12,  click_color: [200, 1500] },
  leather: { modes: [240, 480, 900],          gains: [0.5, 0.3, 0.2],       t60_ms: 25,  click_color: [400, 2200] },
  mud:     { modes: [150, 320],               gains: [0.6, 0.4],            t60_ms: 20,  click_color: [200, 1500] },
  ice:     { modes: [2000, 4400, 7800],       gains: [0.4, 0.35, 0.25],     t60_ms: 400, click_color: [3000, 10000] },
  plastic: { modes: [520, 1100, 2400],        gains: [0.45, 0.35, 0.20],    t60_ms: 60,  click_color: [1500, 7000] },
  cork:    { modes: [380, 780],               gains: [0.6, 0.4],            t60_ms: 35,  click_color: [800, 3500] },
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

function resolveSurface(profile, hardness) {
  if (profile && SURFACE_PROFILES[profile]) return SURFACE_PROFILES[profile];
  const order = ["rubber", "fabric", "cork", "leather", "mud", "wood",
                 "ceramic", "plastic", "stone", "ice", "glass", "metal"];
  const idx = Math.min(Math.round(hardness * (order.length - 1)), order.length - 1);
  return SURFACE_PROFILES[order[idx]];
}

/**
 * Render a scrape/drag event into an AudioBuffer.
 *
 * @param {BaseAudioContext} ctx
 * @param {object} opts
 *   surface_profile: key into SURFACE_PROFILES (overrides hardness)
 *   surface_hardness: 0..1 -> selects surface if no profile given
 *   velocity_mean:    0..1 -> overall amplitude
 *   velocity_jitter:  0..1 -> wiggle amount of the velocity envelope
 *   stick_slip:       0..1 -> strength of discrete slip events
 *   stick_slip_rate:  slips/s (default 30)
 *   roughness:        0..1 -> micro-spike density
 *   pressure:         0..1 -> output scale
 *   duration_s
 *   seed
 * @returns {Promise<AudioBuffer>}
 */
export async function renderScrape(ctx, opts) {
  const {
    surface_profile = null,
    surface_hardness = 0.5,
    velocity_mean = 0.6,
    velocity_jitter = 0.4,
    stick_slip = 0.3,
    stick_slip_rate = 30.0,
    roughness = 0.3,
    pressure = 0.6,
    duration_s = 3.0,
    seed = 0,
  } = opts;

  const sr = ctx.sampleRate;
  const n = Math.floor(duration_s * sr);
  const surf = resolveSurface(surface_profile, surface_hardness);
  const rng = rand(seed);

  // 1) Velocity envelope: slow random LFO (one-pole LPF)
  const cutoffHz = Math.max(0.5, 3 + 12 * velocity_jitter);
  const a = Math.exp(-2 * Math.PI * cutoffHz / sr);
  let envState = 0.5;
  const envSig = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = rng() * 2 - 1;
    envState = a * envState + (1 - a) * x;
    envSig[i] = Math.max(0, Math.min(1.5,
      velocity_mean + 0.4 * velocity_jitter * envState));
  }

  // 2) Stick-slip: discrete bumps superimposed on the envelope
  if (stick_slip > 0.05) {
    const slipRate = stick_slip_rate * (0.3 + 1.5 * stick_slip) * (0.5 + velocity_mean);
    const nSlips = Math.floor(slipRate * duration_s);
    const attackN = Math.max(2, Math.floor(0.0008 * sr));
    for (let s = 0; s < nSlips; s++) {
      const idx = Math.floor(rng() * (n - 100));
      const decayN = Math.max(10, Math.floor((0.003 + 0.012 * rng()) * sr));
      const bumpLen = attackN + decayN;
      const amp = (0.4 + 0.6 * rng()) * stick_slip;
      for (let i = 0; i < bumpLen; i++) {
        const di = idx + i;
        if (di >= n) break;
        const bVal = i < attackN
          ? i / attackN
          : Math.exp(-4 * (i - attackN) / decayN);
        const newVal = bVal * amp + velocity_mean * 0.5;
        if (newVal > envSig[di]) envSig[di] = newVal;
      }
    }
  }

  // 3) Roughness micro-impacts: short noise spikes modulated by local velocity
  const microBuf = new Float32Array(n);
  if (roughness > 0.1) {
    const grainRate = 30 + 250 * roughness;
    const nGrains = Math.floor(duration_s * grainRate);
    for (let g = 0; g < nGrains; g++) {
      const idx = Math.floor(rng() * (n - 10));
      const grainAmp = (0.2 + 0.6 * rng()) * roughness * envSig[idx];
      for (let i = 0; i < 6; i++) {
        if (idx + i < n) microBuf[idx + i] += (rng() * 2 - 1) * grainAmp;
      }
    }
  }

  // 4) Bake noise × velocity envelope into the source buffer
  const noiseBuf = ctx.createBuffer(1, n, sr);
  const noiseData = noiseBuf.getChannelData(0);
  for (let i = 0; i < n; i++) {
    noiseData[i] = (rng() * 2 - 1) * envSig[i] + microBuf[i];
  }

  // 5) Filter chain: click-color bandpass → multi-mode body peaking → gain
  const offline = new OfflineAudioContext(1, n, sr);
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

  // Multi-mode body: peaking filters per surface mode
  let node = bp;
  for (let m = 0; m < surf.modes.length; m++) {
    const fc = surf.modes[m];
    if (fc <= 0 || fc >= sr / 2 - 100) continue;
    const peak = offline.createBiquadFilter();
    peak.type = "peaking";
    peak.frequency.value = fc;
    peak.Q.value = 5.0;
    peak.gain.value = 6.0 + 8.0 * surf.gains[m];
    node.connect(peak);
    node = peak;
  }

  const outGain = offline.createGain();
  outGain.gain.value = 0.3 + 0.7 * pressure;
  node.connect(outGain).connect(offline.destination);
  src.connect(bp);
  src.start();
  const rendered = await offline.startRendering();

  // Peak-normalise
  const ch = rendered.getChannelData(0);
  let peak = 0;
  for (let i = 0; i < ch.length; i++) if (Math.abs(ch[i]) > peak) peak = Math.abs(ch[i]);
  if (peak > 0.95) {
    const scale = 0.95 / peak;
    for (let i = 0; i < ch.length; i++) ch[i] *= scale;
  }
  return rendered;
}
