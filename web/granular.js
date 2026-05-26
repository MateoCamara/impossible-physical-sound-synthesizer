// Granular flow in Web Audio: stochastic cloud of short modal impacts.
//
// Implementation note: instead of scheduling N OscillatorNodes (which scales
// poorly to N=200+), we build the AudioBuffer manually in JS. Each grain is
// a damped sinusoid mixed into the buffer with a per-grain frequency and
// velocity, mirroring the Python physics.granular module.

export const GRAIN_PROFILES = {
  pebble:         { base_freq_hz: 900,  spread_octaves: 0.7, damping_ms: 22,  inharmonicity: 0.40 },
  fine_gravel:    { base_freq_hz: 1600, spread_octaves: 0.9, damping_ms: 12,  inharmonicity: 0.50 },
  coarse_gravel:  { base_freq_hz: 550,  spread_octaves: 0.6, damping_ms: 35,  inharmonicity: 0.45 },
  sand:           { base_freq_hz: 3500, spread_octaves: 1.2, damping_ms: 4,   inharmonicity: 0.70 },
  crushed_glass:  { base_freq_hz: 2800, spread_octaves: 0.8, damping_ms: 120, inharmonicity: 0.10 },
  ice_shards:     { base_freq_hz: 3200, spread_octaves: 1.0, damping_ms: 80,  inharmonicity: 0.15 },
  snow_crunch:    { base_freq_hz: 2200, spread_octaves: 1.5, damping_ms: 3,   inharmonicity: 0.85 },
  ash:            { base_freq_hz: 4500, spread_octaves: 1.3, damping_ms: 2,   inharmonicity: 0.90 },
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
 * Render a granular flow into an AudioBuffer.
 *
 * @param {AudioContext|OfflineAudioContext} ctx
 * @param {object} opts
 *   profile_name: key into GRAIN_PROFILES
 *   density_hz: average grains per second
 *   density_jitter: 0..1 randomness of inter-grain intervals
 *   spread_octaves_override: optional override of profile.spread_octaves
 *   damping_ms_override:     optional override of profile.damping_ms
 *   energy_mean: 0..1
 *   cluster_factor: 0..1 (groups grains around anchors instead of uniform)
 *   duration_s
 *   seed
 * @returns {AudioBuffer}
 */
export function renderGranularFlow(ctx, opts) {
  const {
    profile_name = "pebble",
    density_hz = 80.0,
    density_jitter = 0.7,
    spread_octaves_override = null,
    damping_ms_override = null,
    energy_mean = 0.6,
    cluster_factor = 0.4,
    duration_s = 3.0,
    seed = 0,
  } = opts;

  const profile = GRAIN_PROFILES[profile_name];
  if (!profile) throw new Error(`Unknown grain profile ${profile_name}`);
  const spread = spread_octaves_override ?? profile.spread_octaves;
  const damping = damping_ms_override ?? profile.damping_ms;

  const sr = ctx.sampleRate;
  const n_total = Math.floor(duration_s * sr);
  const buf = ctx.createBuffer(1, n_total, sr);
  const data = buf.getChannelData(0);
  const rng = rand(seed);

  // Cluster anchors
  const n_anchors = Math.max(2, Math.floor(duration_s * 2));
  const anchors = new Array(n_anchors).fill(0).map((_, i) => (i / (n_anchors - 1)) * n_total);
  const anchor_width = (sr / Math.max(density_hz, 0.5)) * 3 * (1 - cluster_factor);

  const period_samples = sr / Math.max(density_hz, 0.5);
  const grain_dur_ms = Math.max(8, 5 * damping);
  const grain_n = Math.max(8, Math.floor((grain_dur_ms / 1000) * sr));
  const tau = (damping / 1000) / 6.907755;
  const decay = new Float32Array(grain_n);
  for (let i = 0; i < grain_n; i++) decay[i] = Math.exp(-(i / sr) / tau);

  let t = 0;
  let grain_count = 0;
  while (t < n_total) {
    // Position with optional clustering
    let sample_pos;
    if (cluster_factor > 0.05) {
      const nearest = anchors.reduce((best, a) => Math.abs(a - t) < Math.abs(best - t) ? a : best, anchors[0]);
      const jitter = anchor_width > 0 ? (rng() * 2 - 1) * anchor_width : 0;
      sample_pos = Math.max(0, Math.min(n_total - 1, Math.floor(nearest + jitter)));
    } else {
      sample_pos = Math.floor(t);
    }
    // Velocity per grain
    const vel = Math.max(0.2, Math.min(2.0, 1.0 + (rng() * 2 - 1) * (0.3 + 0.4 * 0.5)));
    const amp = vel * energy_mean * (0.5 + rng() * 0.8);
    // Frequency: base * 2 ** (jitter * spread)
    const jitter_oct = (rng() * 2 - 1);
    const freq = profile.base_freq_hz * Math.pow(2, jitter_oct * spread);
    // Mix grain into buffer
    const omega = 2 * Math.PI * freq / sr;
    for (let i = 0; i < grain_n; i++) {
      const dst = sample_pos + i;
      if (dst >= n_total) break;
      data[dst] += amp * decay[i] * Math.sin(omega * i);
    }
    t += period_samples * (1 + density_jitter * (rng() * 2 - 1));
    grain_count++;
    if (grain_count > 1500) break;  // safety cap
  }

  // Peak-normalise
  let peak = 0;
  for (let i = 0; i < n_total; i++) if (Math.abs(data[i]) > peak) peak = Math.abs(data[i]);
  if (peak > 0.95) {
    const scale = 0.95 / peak;
    for (let i = 0; i < n_total; i++) data[i] *= scale;
  }
  return buf;
}
