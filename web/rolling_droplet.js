// Rolling droplet in Web Audio: quasi-periodic train of drip events.
//
// Instead of orchestrating one OfflineAudioContext per drip (drip.js uses
// that path because it returns a complete buffer), here we write each
// contact's signal directly into a shared output AudioBuffer. Same
// physics as Python rolling_droplet (Minnaert chirp + decay + surface
// modal tail), but inlined for efficiency with 50-300 contacts per clip.

// Per-material voicing presets: multipliers applied on top of the user's
// mix sliders so each surface has a clearly recognisable sonic signature.
// Conceptually: "fabric should be muffled, metal should ring, stone should
// rumble low". These are perceptual hyper-parameters tuned by ear.
const MATERIAL_VOICING = {
  fabric:  { body_mul: 0.30, cavity_mul: 0.50, ring_mul: 0.40, discrete_mul: 1.0 },
  wood:    { body_mul: 0.60, cavity_mul: 0.80, ring_mul: 0.80, discrete_mul: 1.2 },
  ceramic: { body_mul: 1.00, cavity_mul: 0.70, ring_mul: 1.10, discrete_mul: 1.0 },
  glass:   { body_mul: 1.20, cavity_mul: 0.50, ring_mul: 1.30, discrete_mul: 1.0 },
  metal:   { body_mul: 1.10, cavity_mul: 0.40, ring_mul: 1.50, discrete_mul: 0.9 },
  stone:   { body_mul: 0.40, cavity_mul: 1.20, ring_mul: 0.70, discrete_mul: 1.1 },
  water:   { body_mul: 0.50, cavity_mul: 0.80, ring_mul: 0.50, discrete_mul: 0.8 },
  rubber:  { body_mul: 0.25, cavity_mul: 0.40, ring_mul: 0.30, discrete_mul: 0.7 },
  leather: { body_mul: 0.45, cavity_mul: 0.70, ring_mul: 0.50, discrete_mul: 0.9 },
  mud:     { body_mul: 0.20, cavity_mul: 0.60, ring_mul: 0.20, discrete_mul: 0.5 },
  ice:     { body_mul: 1.00, cavity_mul: 0.60, ring_mul: 1.40, discrete_mul: 1.0 },
  plastic: { body_mul: 0.70, cavity_mul: 0.60, ring_mul: 0.90, discrete_mul: 1.0 },
  cork:    { body_mul: 0.35, cavity_mul: 0.60, ring_mul: 0.40, discrete_mul: 0.8 },
};
const DEFAULT_VOICING = { body_mul: 1.0, cavity_mul: 1.0, ring_mul: 1.0, discrete_mul: 1.0 };

const SURFACE_PROFILES = {
  fabric:  { modes: [180, 320],                    gains: [0.7, 0.3],            t60_ms: 8,   click_color: [500, 2500] },
  wood:    { modes: [280, 720, 1450, 2400],        gains: [0.4, 0.3, 0.2, 0.1],  t60_ms: 80,  click_color: [800, 5000] },
  ceramic: { modes: [1100, 2400, 4800, 7200],      gains: [0.35, 0.3, 0.2, 0.15], t60_ms: 350, click_color: [2000, 9000] },
  glass:   { modes: [1800, 4200, 7100, 9800],      gains: [0.3, 0.3, 0.25, 0.15], t60_ms: 600, click_color: [3000, 10000] },
  metal:   { modes: [900, 2200, 5100, 8800],       gains: [0.35, 0.3, 0.2, 0.15], t60_ms: 900, click_color: [3500, 11000] },
  stone:   { modes: [380, 880, 1800],              gains: [0.5, 0.3, 0.2],       t60_ms: 60,  click_color: [1000, 5000] },
  water:   { modes: [420, 900],                    gains: [0.7, 0.3],            t60_ms: 25,  click_color: [400, 2500] },
  rubber:  { modes: [110, 250],                    gains: [0.7, 0.3],            t60_ms: 12,  click_color: [200, 1500] },
  leather: { modes: [240, 480, 900],               gains: [0.5, 0.3, 0.2],       t60_ms: 25,  click_color: [400, 2200] },
  mud:     { modes: [150, 320],                    gains: [0.6, 0.4],            t60_ms: 20,  click_color: [200, 1500] },
  ice:     { modes: [2000, 4400, 7800],            gains: [0.4, 0.35, 0.25],     t60_ms: 400, click_color: [3000, 10000] },
  plastic: { modes: [520, 1100, 2400],             gains: [0.45, 0.35, 0.20],    t60_ms: 60,  click_color: [1500, 7000] },
  cork:    { modes: [380, 780],                    gains: [0.6, 0.4],            t60_ms: 35,  click_color: [800, 3500] },
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

export function minnaertHz(radius_mm) {
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
 *   radius_mm, viscosity, surface_profile, velocity_factor, seed,
 *   capillary_ringing, bounce_amount, bounce_chain_length, bounce_decay
 */
export function writeDripInline(dst, sampleStart, sr, opts) {
  const {
    radius_mm = 2.0,
    viscosity = 0.0,
    surface_profile = "ceramic",
    velocity_factor = 1.0,
    seed = 0,
    capillary_ringing = 0.5,
    bounce_amount = 0.35,
    bounce_chain_length = 1,
    bounce_decay = 0.55,
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
  const chirpN = Math.max(8, Math.floor(chirpDurS * sr));
  const decayN = Math.floor(decayMs / 1000 * sr);
  const totalN = chirpN + decayN + Math.floor(0.08 * sr);
  if (sampleStart >= dst.length) return;

  // 1) Material-coloured onset click: very short bandpass-flavoured noise
  // burst using surf.click_color. This gives each material a recognisable
  // "click" character (metal=bright/high, wood=dull/mid, fabric=muffled/low).
  // Implemented as filtered noise via a one-pole BP approximation: feed
  // white noise through a resonator centred at sqrt(clLo*clHi).
  const onsetN = Math.max(4, Math.floor(0.003 * sr));  // 3 ms
  const [clLo, clHi] = surf.click_color;
  const clCenter = Math.sqrt(clLo * clHi);
  const clBw = Math.max(50, clHi - clLo);
  // 2-pole resonator (Direct Form II Transposed): biquad bandpass coeffs
  const omega = 2 * Math.PI * clCenter / sr;
  const alpha = Math.sin(omega) * (clBw / clCenter) / 2;  // bandwidth-driven
  const cosw = Math.cos(omega);
  const b0 = alpha, b1 = 0, b2 = -alpha;
  const a0 = 1 + alpha, a1 = -2 * cosw, a2 = 1 - alpha;
  const nb0 = b0 / a0, nb1 = b1 / a0, nb2 = b2 / a0;
  const na1 = a1 / a0, na2 = a2 / a0;
  let z1 = 0, z2 = 0;
  for (let i = 0; i < onsetN; i++) {
    const dst_idx = sampleStart + i;
    if (dst_idx >= dst.length) break;
    const noise = rng() * 2 - 1;
    // Biquad filter: y[n] = b0*x[n] + z1
    const y = nb0 * noise + z1;
    z1 = nb1 * noise - na1 * y + z2;
    z2 = nb2 * noise - na2 * y;
    // Envelope: short attack, exponential decay over onsetN samples
    const env = Math.exp(-3 * i / onsetN);
    dst[dst_idx] += 0.5 * velocity_factor * env * y;
  }

  // 2) Chirp: exponential frequency ramp + envelope
  const attackN = Math.max(2, Math.floor(0.001 * sr));
  let phase = 0;
  const ratio = Math.max(1.001, fEnd / Math.max(fStart, 1));
  const log_ratio = Math.log(ratio);
  for (let i = 0; i < totalN; i++) {
    const dst_idx = sampleStart + i;
    if (dst_idx >= dst.length) break;
    const t = i / sr;
    const t_norm = Math.min(1.0, t / Math.max(chirpDurS, 1e-4));
    const f = fStart * Math.exp(log_ratio * t_norm);
    phase += 2 * Math.PI * f / sr;
    let env;
    if (i < attackN) {
      env = Math.pow(i / attackN, 0.7);
    } else {
      env = Math.exp(-4 * (1 - viscosity * 0.5) * (i - attackN) / chirpN);
    }
    dst[dst_idx] += 0.7 * velocity_factor * env * Math.sin(phase);
  }

  // 2) Capillary ringing: short oscillation of the liquid film at ~2.2x Minnaert
  if (capillary_ringing > 0.05) {
    const fCap = fEnd * 2.2;
    const capN = Math.max(8, Math.floor(0.012 * sr));
    const clickN = Math.max(2, Math.floor(0.0015 * sr));
    for (let i = 0; i < capN; i++) {
      const dst_idx = sampleStart + clickN + i;
      if (dst_idx >= dst.length) break;
      const capEnv = Math.exp(-6 * i / capN);
      dst[dst_idx] += 0.25 * capillary_ringing * velocity_factor *
                      capEnv * Math.sin(2 * Math.PI * fCap * i / sr);
    }
  }

  // 3) Surface modal tail: damped sinusoids per mode
  const tau_surf = (surf.t60_ms / 1000) / 6.907755;
  const decay_surf = Math.exp(-1 / (tau_surf * sr));
  for (let m = 0; m < surf.modes.length; m++) {
    const fc = surf.modes[m];
    if (fc <= 0 || fc >= sr / 2 - 100) continue;
    const g = surf.gains[m] * 0.4 * velocity_factor * (0.3 + 0.7 * 0.5);
    const omega = 2 * Math.PI * fc / sr;
    let amp = g;
    for (let i = 0; i < totalN; i++) {
      const dst_idx = sampleStart + i;
      if (dst_idx >= dst.length) break;
      dst[dst_idx] += amp * Math.sin(omega * i);
      amp *= decay_surf;
    }
  }

  // 4) Bounce chain: geometric-decay mini-chirps after the main contact
  if (bounce_amount > 0.05 && bounce_chain_length >= 1) {
    const bounceDelayBase = (20 + 12 * radius_mm) * (1 + 0.5 * viscosity) / 1000;
    let cumOffset = 0;
    let bAmp = bounce_amount * 0.4;
    for (let k = 0; k < bounce_chain_length; k++) {
      const shrink = Math.pow(Math.sqrt(bounce_decay), k);
      const delayN = Math.floor(bounceDelayBase * sr * shrink);
      cumOffset += delayN;
      if (sampleStart + cumOffset >= dst.length - 50) break;
      const miniN = Math.max(20, Math.floor(chirpN / (2 + k)));
      let miniPhase = 0;
      for (let i = 0; i < miniN; i++) {
        const dst_idx = sampleStart + cumOffset + i;
        if (dst_idx >= dst.length) break;
        const tNorm = i / Math.max(miniN - 1, 1);
        const miniF = fStart * Math.exp(log_ratio * tNorm);
        miniPhase += 2 * Math.PI * miniF / sr;
        const miniEnv = Math.exp(-(5 + k) * i / miniN);
        dst[dst_idx] += bAmp * miniEnv * Math.sin(miniPhase);
      }
      bAmp *= bounce_decay;
    }
  }

  // 5) Bubble pop: short bandpass noise burst (low viscosity only)
  if (viscosity < 0.4 && capillary_ringing > 0.1) {
    const popIdx = sampleStart + chirpN + Math.floor(decayN * 0.4);
    const popLen = 40;
    if (popIdx < dst.length - popLen) {
      for (let i = 0; i < popLen; i++) {
        const dst_idx = popIdx + i;
        if (dst_idx >= dst.length) break;
        const noise = rng() * 2 - 1;
        const popEnv = Math.exp(-8 * i / popLen);
        dst[dst_idx] += noise * 0.15 * (1 - viscosity) * velocity_factor * popEnv;
      }
    }
  }
}

// ====================================================================
// Continuous-contact layers: the "water roller" character
// ====================================================================
//
// A rolling droplet is NOT a series of impacts — it's a continuous
// contact of a deformable liquid mass against a surface. Four layers
// model the physics of this sustained contact:
//
//   A) Body resonance:  Minnaert tone sustained throughout the roll,
//                       AM-modulated by rolling speed, FM-wobbled by
//                       path roughness (the droplet deforms as it rolls).
//   B) Cavity:          Helmholtz-like rumble of the trapped air pocket
//                       between droplet bottom and surface (200-600 Hz).
//   C) Sloshing:        Subharmonic wobble of the liquid mass deforming.
//   D) Shimmer:         Slow random AM applied to A+B+C, the "alive"
//                       quality of moving water (capillary ripples).
//
// These layers are the PROTAGONISTS (60-80% energy). The discrete drip
// events become subtle texture (20-40%, attenuated from 0.7 to ~0.25).

/** Generate a slowly-varying random LFO using one-pole LPF on white noise. */
function slowLFO(n, sr, cutoffHz, rng) {
  const a = Math.exp(-2 * Math.PI * cutoffHz / sr);
  const out = new Float32Array(n);
  let state = 0;
  let sum = 0, sumSq = 0;
  for (let i = 0; i < n; i++) {
    const x = rng() * 2 - 1;
    state = a * state + (1 - a) * x;
    out[i] = state;
    sum += state;
    sumSq += state * state;
  }
  // Normalize to unit variance, zero mean
  const mean = sum / n;
  const variance = sumSq / n - mean * mean;
  const std = Math.sqrt(Math.max(variance, 1e-9));
  for (let i = 0; i < n; i++) out[i] = (out[i] - mean) / std;
  return out;
}

// ====================================================================
// Wetting / contact angle physics
// ====================================================================
// Contact angle θ (in degrees) governs how much the droplet wets the
// surface:
//   θ < 90°  hydrophilic (spreads, slides) → long contact, strong coupling
//   θ ≈ 90°  neutral
//   θ > 90°  hydrophobic (beads, bounces) → short contact, weak coupling
//   θ ≈ 180° lotus / superhydrophobic → almost no contact
//
// Contact area A ∝ (1 + cos(θ))/2  — fraction of droplet touching surface.
// At θ=0°  A=1.0 (fully spread); at θ=180° A=0.0 (lotus, no contact).
function contactAreaFactor(angle_deg) {
  const rad = angle_deg * Math.PI / 180;
  return (1 + Math.cos(rad)) / 2;
}

/** Layer A: sustained Minnaert body resonance with deformation FM/AM.
 *  Amplitude couples to surface impedance × (1 - contactArea) — at low
 *  contact angle (hydrophilic) the droplet is squashed against the
 *  surface and damps strongly; at high angle (lotus) the droplet barely
 *  touches and rings freely. */
function renderBodyResonanceLayer(sr, n, opts) {
  const { radius_mm, viscosity, roll_velocity_hz, path_roughness, seed,
          surface_profile, contact_angle_deg = 110 } = opts;
  const surf = SURFACE_PROFILES[surface_profile] || SURFACE_PROFILES.ceramic;
  const rng = rand(seed + 7001);
  const fM = 3.26 / (Math.max(radius_mm, 0.1) * 1e-3);
  const amHz = 2 + 6 * Math.min(roll_velocity_hz / 25, 1);
  const am = slowLFO(n, sr, amHz, rng);
  const fm = slowLFO(n, sr, amHz * 0.6, rng);
  const fmDepth = 0.03 + 0.05 * path_roughness;
  const t60Factor = Math.min(1, surf.t60_ms / 600);
  const impedanceGain = 0.3 + 0.7 * t60Factor;
  // Wetting: less contact (lotus) → less damping → more body resonance
  const contactArea = contactAreaFactor(contact_angle_deg);
  const wettingGain = 0.4 + 0.6 * (1 - contactArea);  // [0.4 spread → 1.0 lotus]
  const viscAtten = (1.0 - 0.6 * viscosity) * impedanceGain * wettingGain;
  const out = new Float32Array(n);
  let phase = 0;
  const baseAmp = 0.7 * viscAtten;
  for (let i = 0; i < n; i++) {
    const fInst = fM * (1 + fmDepth * fm[i]);
    phase += 2 * Math.PI * fInst / sr;
    const amVal = 0.75 + 0.25 * am[i];
    out[i] = baseAmp * Math.max(0, amVal) * Math.sin(phase);
  }
  return out;
}

/** Layer B: Helmholtz cavity — wetting modulates BOTH frequency
 *  (cavity volume scales with (1 - contactArea)) AND amplitude. */
function renderCavityResonanceLayer(sr, n, opts) {
  const { surface_profile, viscosity, roll_velocity_hz, seed,
          contact_angle_deg = 110 } = opts;
  const surf = SURFACE_PROFILES[surface_profile] || SURFACE_PROFILES.ceramic;
  const rng = rand(seed + 7002);
  const minMode = Math.min(...surf.modes);
  // Cavity volume scales with (1 - contactArea): more wetting = smaller
  // pocket = higher frequency
  const contactArea = contactAreaFactor(contact_angle_deg);
  const cavityVolumeFactor = 0.3 + 0.7 * (1 - contactArea);  // [0.3 wet, 1.0 lotus]
  // Helmholtz f ∝ 1/sqrt(V) → fCavity decreases as cavity volume grows
  const baseFCavity = Math.max(120, Math.min(1200, minMode * 0.6));
  const fCavity = baseFCavity / Math.sqrt(cavityVolumeFactor);
  const amHz = 2 + 6 * Math.min(roll_velocity_hz / 25, 1);
  const am = slowLFO(n, sr, amHz, rng);
  const viscAtten = 1.0 - 0.6 * viscosity;
  // Cavity amplitude proportional to cavity volume
  const cavityAmpGain = 0.5 + 0.5 * (1 - contactArea);  // [0.5 wet, 1.0 lotus]
  const out = new Float32Array(n);
  let phase = Math.PI / 2;
  const baseAmp = 0.5 * viscAtten * cavityAmpGain;
  for (let i = 0; i < n; i++) {
    phase += 2 * Math.PI * fCavity / sr;
    const amVal = 0.75 + 0.25 * am[i];
    out[i] = baseAmp * Math.max(0, amVal) * Math.sin(phase);
  }
  return out;
}

/** NEW Layer F: multi-bubble microbubble cloud.
 *  As the droplet rolls, it entrains N microbubbles of varying radii.
 *  Each radius gives its own Minnaert tone. We model this as a sum
 *  of N partials with log-normal radius distribution centred at
 *  parent_r/3 (microbubbles are 30% of parent radius on average).
 *  Higher contact_angle (lotus) → more bubbles entrained at contact;
 *  higher viscosity → fewer, slower-decaying bubbles. */
function renderMicroBubbleCloudLayer(sr, n, opts) {
  const { radius_mm, viscosity, roll_velocity_hz, path_roughness, seed,
          contact_angle_deg = 110 } = opts;
  const rng = rand(seed + 7006);
  // Number of microbubbles: scales with contact (more contact = more
  // entrained), reduced by viscosity (honey doesn't froth)
  const contactArea = contactAreaFactor(contact_angle_deg);
  const baseN = 6 + Math.floor(14 * contactArea * (1 - viscosity));
  const nBubbles = Math.max(2, baseN);
  const amHz = 2 + 6 * Math.min(roll_velocity_hz / 25, 1);
  const am = slowLFO(n, sr, amHz, rng);
  const out = new Float32Array(n);
  // Parent Minnaert as ceiling (smaller bubbles have higher freq)
  const parentFM = 3.26 / (Math.max(radius_mm, 0.1) * 1e-3);
  const viscAtten = 1.0 - 0.6 * viscosity;
  // Total energy is spread across N bubbles → individual amplitude smaller
  const baseAmp = 0.5 * viscAtten / Math.sqrt(nBubbles);
  for (let b = 0; b < nBubbles; b++) {
    // Log-normal radius: log r ~ N(log(parent_r/3), σ=0.6)
    // r_b = parent_r/3 * exp(0.6 * N(0,1))
    const u1 = Math.max(1e-9, rng()), u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const rBubble = (radius_mm / 3) * Math.exp(0.6 * z);
    const rBubbleSafe = Math.max(0.05, Math.min(radius_mm * 0.8, rBubble));
    const fBubble = 3.26 / (rBubbleSafe * 1e-3);
    if (fBubble >= sr / 2 - 200) continue;
    // Damping time for this bubble (shorter for smaller bubbles)
    // life ~ exp(-i/decay_n), decay_n proportional to r²
    const decayN = Math.max(200, Math.floor(0.05 * sr * (rBubbleSafe / radius_mm) ** 2));
    // Random phase + random onset within first 30% of clip
    const onset = Math.floor(rng() * n * 0.3);
    let phase = 2 * Math.PI * rng();
    // Amplitude weighted by ~r^1.5 (larger bubbles louder)
    const ampWeight = Math.pow(rBubbleSafe / (radius_mm / 3), 1.5);
    const bAmp = baseAmp * ampWeight * (0.5 + 0.5 * rng());
    // Slow FM modulation specific to this bubble (uncorrelated)
    const bubbleFm = slowLFO(n, sr, amHz * (0.5 + rng()), rng);
    for (let i = onset; i < n; i++) {
      const t = i - onset;
      // Exponential decay envelope, re-triggered periodically by AM
      const decay = Math.exp(-t / decayN);
      const reTrigger = Math.max(0, 0.7 + 0.3 * am[i]);
      const fInst = fBubble * (1 + 0.02 * bubbleFm[i]);
      phase += 2 * Math.PI * fInst / sr;
      out[i] += bAmp * decay * reTrigger * Math.sin(phase);
    }
  }
  return out;
}

/** Layer E (NEW): continuous surface ringing. For each mode of the surface,
 *  a sustained damped sinusoid at fc[k] with amplitude proportional to
 *  gains[k] × t60-derived sustain. Excited continuously by the same rolling
 *  LFO that drives body/cavity. This is the "material voice" — what makes
 *  metal sing and wood thud when rolled on. */
function renderSurfaceRingingLayer(sr, n, opts) {
  const { surface_profile, viscosity, roll_velocity_hz, path_roughness, seed } = opts;
  const surf = SURFACE_PROFILES[surface_profile] || SURFACE_PROFILES.ceramic;
  const rng = rand(seed + 7005);
  const amHz = 2 + 6 * Math.min(roll_velocity_hz / 25, 1);
  const am = slowLFO(n, sr, amHz, rng);
  // Per-mode FM wobble (each mode jitters independently — adds liveness)
  const fmDepth = 0.005 + 0.015 * path_roughness;
  const viscAtten = 1.0 - 0.6 * viscosity;
  // t60-derived sustain factor: long t60 (metal) → sustained ring,
  // short t60 (fabric) → barely audible.
  const t60Factor = Math.min(1, surf.t60_ms / 600);
  const sustainGain = 0.2 + 0.8 * t60Factor;
  const out = new Float32Array(n);
  const baseAmp = 0.55 * viscAtten * sustainGain;
  for (let m = 0; m < surf.modes.length; m++) {
    const fc = surf.modes[m];
    if (fc <= 0 || fc >= sr / 2 - 100) continue;
    const g = surf.gains[m];
    // Each mode has its own slow FM (uncorrelated with body/cavity)
    const modeFm = slowLFO(n, sr, amHz * (0.7 + 0.4 * m / surf.modes.length), rng);
    let phase = 2 * Math.PI * rng();  // random phase per mode
    for (let i = 0; i < n; i++) {
      const fInst = fc * (1 + fmDepth * modeFm[i]);
      phase += 2 * Math.PI * fInst / sr;
      const amVal = 0.7 + 0.3 * am[i];
      out[i] += baseAmp * g * Math.max(0, amVal) * Math.sin(phase);
    }
  }
  return out;
}

/** Layer C: Rayleigh shape oscillation modes of a free liquid sphere.
 *  Replaces the old heuristic "sloshing" with the exact physical formula:
 *
 *      ω_n² = n(n-1)(n+2) × σ / (ρ × r³)
 *
 *  where n=2,3,4,... is the mode number, σ is surface tension (water:
 *  0.072 N/m), ρ is density (water: 1000 kg/m³), r is droplet radius.
 *
 *  For a 2 mm water droplet:  f_2 ≈ 52 Hz, f_3 ≈ 95 Hz, f_4 ≈ 143 Hz,
 *  f_5 ≈ 196 Hz, f_6 ≈ 250 Hz — a real subharmonic series, much lower
 *  than the old heuristic (which sat at 650-980 Hz).
 *
 *  Reference: Rayleigh, Lord (1879) "On the capillary phenomena of jets".
 *  Excitation amplitude scales with path_roughness (a rougher surface
 *  perturbs the droplet shape more). Damping decreases with viscosity
 *  (honey stops wobbling faster; water keeps wobbling for ~100 ms). */
function renderRayleighModesLayer(sr, n, opts) {
  const { radius_mm, viscosity, path_roughness, seed, surface_tension = 0.072 } = opts;
  const rng = rand(seed + 7003);
  const r = Math.max(0.1, radius_mm) * 1e-3;     // m
  const rho = 1000.0;                              // kg/m³ (water)
  const sigma = surface_tension;                   // N/m
  const out = new Float32Array(n);
  const baseAmp = 0.4 * (0.4 + 0.6 * path_roughness);
  // Higher viscosity → faster damping (less ring time)
  const dampRate = 5 + 20 * viscosity;             // Hz of envelope decay
  for (let mode = 2; mode <= 6; mode++) {
    const omega2 = mode * (mode - 1) * (mode + 2) * sigma / (rho * r * r * r);
    if (omega2 <= 0) continue;
    const fMode = Math.sqrt(omega2) / (2 * Math.PI);
    if (fMode >= sr / 2 - 50) continue;
    // Mode amplitude falls off with mode number (n=2 dominant)
    const modeAmp = baseAmp / mode;
    // Re-excite mode periodically (path bumps) — use a slow chaotic LFO
    const triggerHz = 2 + 8 * path_roughness;
    const triggers = slowLFO(n, sr, triggerHz, rng);
    let phase = 2 * Math.PI * rng();
    for (let i = 0; i < n; i++) {
      phase += 2 * Math.PI * fMode / sr;
      // AM envelope from triggers, decayed by viscosity
      const trig = 0.5 + 0.5 * triggers[i];
      out[i] += modeAmp * trig * Math.sin(phase);
    }
  }
  return out;
}

/** NEW Layer G: rolling stick-slip.
 *  Even pure rolling has micro-events: surface asperities are briefly
 *  captured by capillary forces, then released. Each release is a
 *  tiny impact filtered by the surface's click_color. Engagement rate
 *  scales with path_roughness × roll_velocity_hz.
 *
 *  Reference: Persson, B.N.J. (2001) "Theory of rubber friction and
 *  contact mechanics"; modified for liquid rolling. */
function renderRollingStickSlipLayer(sr, n, opts) {
  const { surface_profile, viscosity, roll_velocity_hz, path_roughness, seed,
          contact_angle_deg = 110 } = opts;
  const surf = SURFACE_PROFILES[surface_profile] || SURFACE_PROFILES.ceramic;
  const rng = rand(seed + 7007);
  // Rate of asperity engagement events per second
  const rateHz = (5 + 60 * path_roughness) * (0.5 + roll_velocity_hz / 20);
  const nEvents = Math.floor(rateHz * n / sr);
  if (nEvents < 1) return new Float32Array(n);
  // Contact wettability: low contact angle = more sticking = more events
  const contactArea = contactAreaFactor(contact_angle_deg);
  const stickProb = 0.3 + 0.6 * contactArea;
  // Material's bandpass for the click colour
  const [clLo, clHi] = surf.click_color;
  const clCenter = Math.sqrt(clLo * Math.min(clHi, sr / 2 - 200));
  const clBw = Math.max(50, Math.min(clHi, sr / 2 - 200) - clLo);
  // Biquad bandpass coefficients (same as writeDripInline onset)
  const omega = 2 * Math.PI * clCenter / sr;
  const alpha = Math.sin(omega) * (clBw / clCenter) / 2;
  const cosw = Math.cos(omega);
  const a0 = 1 + alpha;
  const nb0 = alpha / a0;
  const nb2 = -alpha / a0;
  const na1 = -2 * cosw / a0;
  const na2 = (1 - alpha) / a0;
  // Velocity attenuation: viscous fluids slip more smoothly
  const viscAtten = 1.0 - 0.7 * viscosity;
  const out = new Float32Array(n);
  for (let e = 0; e < nEvents; e++) {
    if (rng() > stickProb) continue;  // not every event triggers
    const start = Math.floor(rng() * (n - 100));
    const burstLen = Math.max(8, Math.floor((0.0003 + 0.0008 * rng()) * sr));
    const amp = (0.15 + 0.4 * rng()) * path_roughness * viscAtten;
    let z1 = 0, z2 = 0;
    for (let i = 0; i < burstLen; i++) {
      const idx = start + i;
      if (idx >= n) break;
      const noise = rng() * 2 - 1;
      // Biquad direct form II transposed (bandpass)
      const y = nb0 * noise + z1;
      z1 = -na1 * y + z2;
      z2 = nb2 * noise - na2 * y;
      const env = Math.exp(-4 * i / burstLen);
      out[idx] += amp * env * y;
    }
  }
  return out;
}

/** Layer D: spectral shimmer (slow random AM on the combined continuous mix). */
function applyShimmer(data, n, sr, depth, seed) {
  if (depth < 0.01) return;
  const rng = rand(seed + 7004);
  // Random LFO at 5-15 Hz
  const shHz = 5 + 10 * rng();
  const lfo = slowLFO(n, sr, shHz, rng);
  for (let i = 0; i < n; i++) {
    // Multiplier oscillates around 1, scaled by depth
    const mult = 1 + depth * lfo[i];
    data[i] *= Math.max(0, mult);
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
 * Render a rolling droplet as a continuous liquid mass rolling on a surface.
 *
 * Architecture (NEW): continuous-contact layers are protagonists, discrete
 * drip events are subtle texture.
 *   - Body resonance (Minnaert tone sustained throughout the roll)
 *   - Cavity (Helmholtz of trapped air pocket droplet/surface)
 *   - Sloshing (subharmonic wobble of the liquid mass)
 *   - Shimmer (slow random AM = capillary ripple texture)
 *   - Noise-based rumble (material click_color filtered through modal bank)
 *   - Discrete drip ticks (attenuated to ~25%, just texture)
 *
 * @param {BaseAudioContext} ctx
 * @param {object} opts — see slider definitions in index.html
 * @returns {Promise<AudioBuffer>}
 */
export async function renderRollingDroplet(ctx, opts) {
  const {
    radius_mm = 2.0,
    viscosity = 0.0,
    surface_profile = "ceramic",
    roll_velocity_hz = 14.0,
    path_roughness = 0.35,
    // New "water roller" continuous-layer mix levels:
    body_resonance_mix = 0.7,
    cavity_mix = 0.4,
    slosh_mix = 0.3,
    shimmer_depth = 0.2,
    surface_ring_mix = 0.7,
    // NEW physics layers
    microbubble_mix = 0.5,        // multi-bubble microbubble cloud
    rayleigh_mix = 0.35,           // exact Rayleigh shape oscillations
    stickslip_mix = 0.4,           // rolling friction stick-slip texture
    contact_angle_deg = 110,       // wetting: 30=hydrophilic, 170=lotus
    surface_tension_n_m = 0.072,   // N/m (water = 0.072, oils ~0.03)
    // Existing noise-based rumble and surface modal body:
    continuous_layer_mix = 0.75,
    body_resonance_strength = 0.6,
    discrete_mix = 0.5,
    capillary_ringing = 0.5,
    bounce_amount = 0.35,
    bounce_chain_length = 1,
    bounce_decay = 0.55,
    inter_event_variability = 0.6,
    drying_factor = 0.0,
    duration_s = 5.0,
    seed = 0,
  } = opts;
  // Wetting modulates bounce: lotus surface → bounces much more,
  // hydrophilic surface → barely bounces.
  const _contactArea = contactAreaFactor(contact_angle_deg);
  const wettingBounce = 0.5 + 1.5 * (1 - _contactArea);  // [0.5 wet → 2.0 lotus]
  const effective_bounce_amount = bounce_amount * wettingBounce;
  // Wetting also affects capillary ringing (more contact = more film deform)
  const effective_capillary_ringing = capillary_ringing * (0.5 + 0.7 * _contactArea);
  const sr = ctx.sampleRate;
  const nTotal = Math.floor(duration_s * sr);
  const buf = ctx.createBuffer(1, nTotal, sr);
  const data = buf.getChannelData(0);
  const rng = rand(seed);
  const periodSamples = sr / Math.max(roll_velocity_hz, 0.1);

  // Per-material voicing: each surface multiplies the user's mix levels
  // so that fabric is muffled, metal sings, etc., without the user
  // needing to tune sliders per material.
  const voicing = MATERIAL_VOICING[surface_profile] || DEFAULT_VOICING;

  // ============================================================
  // 1) Continuous-contact layers (protagonists)
  // ============================================================
  // Layer A: sustained Minnaert body resonance (impedance + wetting coupled)
  if (body_resonance_mix > 0.01) {
    const bodyLayer = renderBodyResonanceLayer(sr, nTotal, {
      radius_mm, viscosity, roll_velocity_hz, path_roughness, seed,
      surface_profile, contact_angle_deg,
    });
    const mix = body_resonance_mix * voicing.body_mul;
    for (let i = 0; i < nTotal; i++) data[i] += bodyLayer[i] * mix;
  }

  // Layer B: Helmholtz cavity resonance (wetting-modulated volume)
  if (cavity_mix > 0.01) {
    const cavityLayer = renderCavityResonanceLayer(sr, nTotal, {
      surface_profile, viscosity, roll_velocity_hz, seed, contact_angle_deg,
    });
    const mix = cavity_mix * voicing.cavity_mul;
    for (let i = 0; i < nTotal; i++) data[i] += cavityLayer[i] * mix;
  }

  // Layer C: Rayleigh shape oscillation modes (physical, replaces heuristic sloshing)
  if (rayleigh_mix > 0.01) {
    const rayleighLayer = renderRayleighModesLayer(sr, nTotal, {
      radius_mm, viscosity, path_roughness, seed,
      surface_tension: surface_tension_n_m,
    });
    for (let i = 0; i < nTotal; i++) data[i] += rayleighLayer[i] * rayleigh_mix;
  }

  // Legacy slosh_mix: still supported for backward compat but defaults low
  if (slosh_mix > 0.01) {
    const sloshLayer = renderSloshingLayer(sr, nTotal, {
      radius_mm, path_roughness, seed,
    });
    for (let i = 0; i < nTotal; i++) data[i] += sloshLayer[i] * slosh_mix;
  }

  // Layer F (NEW): microbubble cloud — N=8-20 entrained microbubbles
  // give the body a "fizzy, alive" quality. More for lotus (more
  // entrainment), less for honey (no froth).
  if (microbubble_mix > 0.01) {
    const microLayer = renderMicroBubbleCloudLayer(sr, nTotal, {
      radius_mm, viscosity, roll_velocity_hz, path_roughness, seed,
      contact_angle_deg,
    });
    for (let i = 0; i < nTotal; i++) data[i] += microLayer[i] * microbubble_mix;
  }

  // Layer E: continuous surface ringing (material voice)
  if (surface_ring_mix > 0.01) {
    const ringLayer = renderSurfaceRingingLayer(sr, nTotal, {
      surface_profile, viscosity, roll_velocity_hz, path_roughness, seed,
    });
    const mix = surface_ring_mix * voicing.ring_mul;
    for (let i = 0; i < nTotal; i++) data[i] += ringLayer[i] * mix;
  }

  // Layer G (NEW): rolling stick-slip — micro-impacts from asperity
  // engagement. Adds the "scratching" texture characteristic of rolling
  // on rough surfaces.
  if (stickslip_mix > 0.01) {
    const stickLayer = renderRollingStickSlipLayer(sr, nTotal, {
      surface_profile, viscosity, roll_velocity_hz, path_roughness, seed,
      contact_angle_deg,
    });
    for (let i = 0; i < nTotal; i++) data[i] += stickLayer[i] * stickslip_mix;
  }

  // Layer D: spectral shimmer (post-process the continuous mix)
  applyShimmer(data, nTotal, sr, shimmer_depth, seed);

  // ============================================================
  // 2) Noise-based rumble layer (material click_color + modal bank)
  // ============================================================
  if (continuous_layer_mix > 0.01) {
    const layerBuf = await renderContinuousRollLayer(sr, duration_s, {
      surface_profile, viscosity, roll_velocity_hz, path_roughness, seed,
      body_resonance_strength,
    });
    const layer = layerBuf.getChannelData(0);
    const layerLen = Math.min(layer.length, nTotal);
    // Continuous gain (not RMS-gated by drip train): the rumble is
    // present as long as the droplet is rolling, not just at impacts.
    const velocityFloor = 0.4 + 0.6 * Math.min(roll_velocity_hz / 25, 1);
    for (let i = 0; i < layerLen; i++) {
      data[i] += layer[i] * velocityFloor * continuous_layer_mix;
    }
  }

  // ============================================================
  // 3) Discrete drip ticks (texture on top of the continuous body)
  // ============================================================
  if (discrete_mix > 0.01) {
    const tickBuf = new Float32Array(nTotal);
    const nVariants = 1 + Math.round(inter_event_variability * 5);
    let t = 0;
    let count = 0;
    while (t < nTotal) {
      const start = Math.floor(t);
      if (start >= nTotal) break;
      const velSpread = 0.25 + 0.4 * path_roughness * (0.5 + 0.5 * inter_event_variability);
      const velocity = Math.max(0.4, Math.min(1.8,
        1.0 + (rng() * 2 - 1) * velSpread));
      const useFreshSeed = inter_event_variability > 0.1 &&
                           Math.abs(velocity - 1.0) > 0.25;
      const evtSeed = useFreshSeed
        ? seed + start
        : seed + 100 + (count % nVariants);
      writeDripInline(tickBuf, start, sr, {
        radius_mm, viscosity, surface_profile,
        velocity_factor: velocity, seed: evtSeed,
        capillary_ringing: effective_capillary_ringing,
        bounce_amount: effective_bounce_amount,
        bounce_chain_length, bounce_decay,
      });
      const offset = periodSamples * (1 + path_roughness * (rng() * 2 - 1) * 0.7);
      t += Math.max(periodSamples * 0.1, offset);
      count++;
      if (count > 600) break;
    }
    // Mix ticks with per-material voicing multiplier
    const tickMix = discrete_mix * voicing.discrete_mul;
    for (let i = 0; i < nTotal; i++) data[i] += tickBuf[i] * tickMix;
  }

  // ============================================================
  // 4) Drying tail
  // ============================================================
  if (drying_factor > 0.05) {
    const half = Math.floor(nTotal / 2);
    for (let i = half; i < nTotal; i++) {
      const ramp = (i - half) / (nTotal - half);
      const dryEnv = 1.0 - drying_factor * (1 - Math.exp(-3 * ramp));
      data[i] *= dryEnv;
    }
  }

  // ============================================================
  // 5) Fade-in/out (avoid filter edge transients)
  // ============================================================
  const fadeN = Math.min(Math.floor(0.02 * sr), Math.floor(nTotal / 8));
  if (fadeN > 4) {
    for (let i = 0; i < fadeN; i++) {
      const w = 0.5 * (1 - Math.cos(Math.PI * i / fadeN));
      data[i] *= w;
      data[nTotal - 1 - i] *= w;
    }
  }

  // ============================================================
  // 6) Peak-normalise
  // ============================================================
  let peak = 0;
  for (let i = 0; i < nTotal; i++) if (Math.abs(data[i]) > peak) peak = Math.abs(data[i]);
  if (peak > 0.95) {
    const scale = 0.95 / peak;
    for (let i = 0; i < nTotal; i++) data[i] *= scale;
  }
  return buf;
}
