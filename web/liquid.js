// Liquid events (splash, pour) in Web Audio.
//
// Mirrors impossible_mix.physics.liquid: splash is a burst of bubbles with
// an exponential onset followed by subsurface agitation; pour is a dense
// drip train with a turbulence bed. Both reuse writeDripInline from the
// rolling_droplet module.

import { writeDripInline } from "./rolling_droplet.js";

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
 * Render a splash event: initial whoosh + cascading bubble burst + subsurface bed.
 *
 * @param {BaseAudioContext} ctx
 * @param {object} opts
 *   intensity: 0..1
 *   bubble_size_mean_mm: radius of typical bubbles
 *   bubble_size_var: 0..1 size variation
 *   n_bubbles: total bubble count
 *   spread_ms: temporal spread of bubble onset
 *   viscosity: 0..1
 *   duration_s, seed
 * @returns {AudioBuffer}
 */
export function renderSplash(ctx, opts) {
  const {
    intensity = 0.7,
    bubble_size_mean_mm = 3.0,
    bubble_size_var = 0.6,
    n_bubbles = 30,
    spread_ms = 80.0,
    viscosity = 0.1,
    duration_s = 2.0,
    seed = 0,
  } = opts;
  const sr = ctx.sampleRate;
  const n = Math.floor(duration_s * sr);
  const buf = ctx.createBuffer(1, n, sr);
  const data = buf.getChannelData(0);
  const rng = rand(seed);

  // 1) Pre-impact whoosh: swept bandpass noise
  const preN = Math.floor(0.025 * sr);
  for (let i = 0; i < preN && i < n; i++) {
    const env = Math.sin(Math.PI * i / preN);
    data[i] += (rng() * 2 - 1) * intensity * env * 0.5;
  }

  // 2) Cascading bubbles with exponentially decaying onset rate
  const totalBubbles = Math.floor(n_bubbles * intensity);
  const spreadSamples = Math.floor(spread_ms / 1000 * sr);
  for (let i = 0; i < totalBubbles; i++) {
    // Exponential distribution: more dense at the start
    const raw = -Math.log(1 - rng() * 0.999) * spreadSamples * 0.3;
    const offset = Math.min(Math.floor(raw), spreadSamples);
    const progress = i / Math.max(totalBubbles, 1);
    // Smaller bubbles first (microbubbles), larger later
    const radiusGrowth = 0.4 + 1.6 * progress;
    let radius = bubble_size_mean_mm * radiusGrowth *
                 (1 + bubble_size_var * (rng() * 1.7 - 0.5));
    radius = Math.max(0.3, radius);
    const velocity = Math.max(0.4, Math.min(1.5,
      1.0 + 0.3 * (1 - progress) + (rng() * 2 - 1) * 0.2));
    const amp = (0.25 + 0.75 * rng()) * intensity * (0.5 + 0.5 * (1 - progress * 0.5));
    writeDripInline(data, offset, sr, {
      radius_mm: radius, viscosity, surface_profile: "water",
      velocity_factor: velocity * amp, seed: seed + i * 3,
      capillary_ringing: 0.4, bounce_amount: 0.0, bounce_chain_length: 0,
      bounce_decay: 0.5,
    });
  }

  // 3) Subsurface bed: low-frequency agitation noise with exponential decay
  const bedStart = preN;
  const bedN = n - bedStart;
  if (bedN > 100) {
    for (let i = 0; i < bedN; i++) {
      const bedEnv = Math.exp(-4 * i / bedN);
      data[bedStart + i] += (rng() * 2 - 1) * 0.03 * intensity * bedEnv;
    }
  }

  // Peak-normalise
  let peak = 0;
  for (let i = 0; i < n; i++) if (Math.abs(data[i]) > peak) peak = Math.abs(data[i]);
  if (peak > 0.95) {
    const scale = 0.95 / peak;
    for (let i = 0; i < n; i++) data[i] *= scale;
  }
  return buf;
}

/**
 * Render a pour event: dense drip train + turbulence bed.
 *
 * @param {BaseAudioContext} ctx
 * @param {object} opts
 *   flow_rate: 0..1 (higher = denser drips)
 *   bubble_size_mean_mm: typical drop radius
 *   viscosity: 0..1
 *   duration_s, seed
 * @returns {AudioBuffer}
 */
export function renderPour(ctx, opts) {
  const {
    flow_rate = 0.6,
    bubble_size_mean_mm = 1.5,
    viscosity = 0.1,
    duration_s = 3.0,
    seed = 0,
  } = opts;
  const sr = ctx.sampleRate;
  const n = Math.floor(duration_s * sr);
  const buf = ctx.createBuffer(1, n, sr);
  const data = buf.getChannelData(0);
  const rng = rand(seed);

  // Dense drip train
  const dripRate = 30 + 200 * flow_rate;
  const periodSamples = sr / dripRate;
  let t = 0;
  let count = 0;
  while (t < n) {
    let radius = bubble_size_mean_mm * (1 + 0.4 * (rng() * 1.3 - 0.5));
    radius = Math.max(0.3, radius);
    const amp = 0.4 + 0.5 * rng();
    const start = Math.floor(t);
    if (start >= n) break;
    writeDripInline(data, start, sr, {
      radius_mm: radius, viscosity, surface_profile: "water",
      velocity_factor: amp, seed: seed + count * 5,
      capillary_ringing: 0.3, bounce_amount: 0.0, bounce_chain_length: 0,
      bounce_decay: 0.5,
    });
    t += periodSamples * (1 + 0.6 * (rng() * 2 - 1) * 0.7);
    count++;
    if (count > 2000) break;
  }

  // Turbulence bed: mid-frequency bandpass noise
  for (let i = 0; i < n; i++) {
    data[i] += (rng() * 2 - 1) * 0.05 * flow_rate;
  }

  // Peak-normalise
  let peak = 0;
  for (let i = 0; i < n; i++) if (Math.abs(data[i]) > peak) peak = Math.abs(data[i]);
  if (peak > 0.95) {
    const scale = 0.95 / peak;
    for (let i = 0; i < n; i++) data[i] *= scale;
  }
  return buf;
}
