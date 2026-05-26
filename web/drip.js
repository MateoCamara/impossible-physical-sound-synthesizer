// Single drip event in Web Audio API: a Minnaert bubble chirp + short
// noise click + optional surface modal tail + bubble pop. Mirrors a
// simplified subset of impossible_mix.physics.droplet.synth_drip_event.

const SURFACE_PROFILES = {
  fabric: { modes: [180, 320],
            gains: [0.7, 0.3], t60_ms: 8,  click_lo: 500,  click_hi: 2500 },
  wood:   { modes: [280, 720, 1450, 2400],
            gains: [0.4, 0.3, 0.2, 0.1], t60_ms: 80, click_lo: 800, click_hi: 5000 },
  ceramic:{ modes: [1100, 2400, 4800, 7200],
            gains: [0.35, 0.3, 0.2, 0.15], t60_ms: 350, click_lo: 2000, click_hi: 9000 },
  glass:  { modes: [1800, 4200, 7100, 9800],
            gains: [0.3, 0.3, 0.25, 0.15], t60_ms: 600, click_lo: 3000, click_hi: 10000 },
  metal:  { modes: [900, 2200, 5100, 8800],
            gains: [0.35, 0.3, 0.2, 0.15], t60_ms: 900, click_lo: 3500, click_hi: 11000 },
  stone:  { modes: [380, 880, 1800],
            gains: [0.5, 0.3, 0.2], t60_ms: 60, click_lo: 1000, click_hi: 5000 },
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
 * Render a single drip event into an AudioBuffer.
 *
 * @param {AudioContext} ctx
 * @param {object} opts
 *   radius_mm: 0.5..6
 *   viscosity: 0..1 (1 = honey-like, slower chirp, shorter decay)
 *   surface_profile: key into SURFACE_PROFILES
 *   capillary_ringing: 0..1
 *   velocity_factor: 0.5..2
 *   duration_s
 *   seed
 * @returns {Promise<AudioBuffer>}
 */
export async function renderDripEvent(ctx, opts) {
  const {
    radius_mm = 2.0,
    viscosity = 0.0,
    surface_profile = "ceramic",
    capillary_ringing = 0.5,
    velocity_factor = 1.0,
    duration_s = 0.6,
    seed = 0,
  } = opts;
  const surf = SURFACE_PROFILES[surface_profile] || SURFACE_PROFILES.ceramic;
  const sr = ctx.sampleRate;
  const n = Math.floor(duration_s * sr);
  const offline = new OfflineAudioContext(1, n, sr);
  const rng = rand(seed);

  // Chirp: simulate Minnaert ascend by sweeping an oscillator's frequency
  // from f_start to f_end during chirp_duration_ms.
  const fM = minnaertHz(radius_mm);
  const fStart = fM * 0.45;
  const fEnd = fM * 1.6 * (1.0 + 0.15 * (rng() * 2 - 1));
  const chirpDurMs = (15 + 8 * radius_mm) * (1 + 1.5 * viscosity) *
                      (0.7 + 0.6 / Math.max(velocity_factor, 0.3));
  const chirpDurS = chirpDurMs / 1000;
  const decayMs = (50 + 30 * radius_mm) * (1 - 0.6 * viscosity);

  const osc = offline.createOscillator();
  osc.type = "sine";
  osc.frequency.value = fStart;
  osc.frequency.exponentialRampToValueAtTime(Math.max(20, fEnd), chirpDurS);

  const chirpGain = offline.createGain();
  chirpGain.gain.value = 0;
  // attack
  chirpGain.gain.setValueAtTime(0, 0);
  chirpGain.gain.linearRampToValueAtTime(0.7, 0.001);
  // decay during chirp
  chirpGain.gain.exponentialRampToValueAtTime(0.001,
                                              chirpDurS + decayMs / 1000);

  osc.connect(chirpGain).connect(offline.destination);

  // Click (short bandpass noise burst at t=0)
  const clickLen = Math.max(20, Math.floor(0.0015 * sr));
  const clickBuf = offline.createBuffer(1, clickLen, sr);
  const click = clickBuf.getChannelData(0);
  for (let i = 0; i < clickLen; i++) click[i] = (rng() * 2 - 1) * 0.6 * velocity_factor;
  const clickSrc = offline.createBufferSource();
  clickSrc.buffer = clickBuf;
  const clickFilt = offline.createBiquadFilter();
  clickFilt.type = "bandpass";
  clickFilt.frequency.value = (surf.click_lo + surf.click_hi) / 2;
  clickFilt.Q.value = (surf.click_lo + surf.click_hi) / (surf.click_hi - surf.click_lo);
  clickSrc.connect(clickFilt).connect(offline.destination);

  // Surface modal tail: short noise excitation through bandpass at each mode.
  const tailLen = Math.max(50, Math.floor(0.001 * sr));
  const tailBuf = offline.createBuffer(1, tailLen, sr);
  const tail = tailBuf.getChannelData(0);
  for (let i = 0; i < tailLen; i++) tail[i] = (rng() * 2 - 1) * 0.3;
  for (let k = 0; k < surf.modes.length; k++) {
    const fc = surf.modes[k];
    if (fc <= 0 || fc >= sr / 2 - 100) continue;
    const tailSrc = offline.createBufferSource();
    tailSrc.buffer = tailBuf;
    const bp = offline.createBiquadFilter();
    bp.type = "bandpass";
    bp.frequency.value = fc;
    bp.Q.value = Math.max(1, Math.PI * fc * surf.t60_ms / 1000);
    const tg = offline.createGain();
    tg.gain.value = surf.gains[k] * 0.5 * velocity_factor;
    tailSrc.connect(bp).connect(tg).connect(offline.destination);
    tailSrc.start(0);
  }

  // Bubble pop (low-viscosity finisher): short bandpass noise burst near
  // the end of the chirp.
  if (viscosity < 0.4 && capillary_ringing > 0.1) {
    const popDelay = chirpDurS + 0.4 * (decayMs / 1000);
    const popLen = 40;
    const popBuf = offline.createBuffer(1, popLen, sr);
    const pop = popBuf.getChannelData(0);
    for (let i = 0; i < popLen; i++) {
      pop[i] = (rng() * 2 - 1) * Math.exp((-8 * i) / popLen) *
               0.2 * (1 - viscosity) * velocity_factor;
    }
    const popSrc = offline.createBufferSource();
    popSrc.buffer = popBuf;
    const popFilt = offline.createBiquadFilter();
    popFilt.type = "bandpass";
    popFilt.frequency.value = 2000;
    popFilt.Q.value = 3;
    popSrc.connect(popFilt).connect(offline.destination);
    popSrc.start(popDelay);
  }

  osc.start(0);
  osc.stop(chirpDurS + decayMs / 1000 + 0.05);
  clickSrc.start(0);

  const rendered = await offline.startRendering();
  return rendered;
}
