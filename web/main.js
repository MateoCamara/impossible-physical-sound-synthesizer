import { MODAL_PROFILES, renderModalImpact } from "./modal.js";
import { renderDripEvent } from "./drip.js";
import { GRAIN_PROFILES, renderGranularFlow } from "./granular.js";
import { IR_PRESETS, generateIR, applyReverb } from "./reverb.js";

// Single shared AudioContext (created on first user gesture)
let ctx = null;
function ensureCtx() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

async function maybeReverb(buf) {
  // If the global Reverb panel has a non-dry preset and mix > 0, apply it.
  const preset = document.getElementById("global-reverb-preset").value;
  const mix = parseFloat(document.getElementById("global-reverb-mix").value);
  if (preset === "dry" || mix <= 0.001) return buf;
  const ir = generateIR(ctx, preset, 42);
  return await applyReverb(ctx, buf, ir, mix);
}

function playBuffer(buf) {
  const c = ensureCtx();
  const src = c.createBufferSource();
  src.buffer = buf;
  src.connect(c.destination);
  src.start();
}

function bindSlider(rangeId, labelId, fmt = (v) => v) {
  const range = document.getElementById(rangeId);
  const label = document.getElementById(labelId);
  function update() { label.textContent = fmt(parseFloat(range.value)); }
  range.addEventListener("input", update);
  update();
}

// ------- Modal tab -------
bindSlider("modal-duration", "modal-duration-val", (v) => `${v.toFixed(1)} s`);
bindSlider("modal-strength", "modal-strength-val", (v) => v.toFixed(2));
bindSlider("modal-sharpness", "modal-sharpness-val", (v) => v.toFixed(2));
bindSlider("modal-velocity", "modal-velocity-val", (v) => v.toFixed(2));
bindSlider("modal-anisotropy", "modal-anisotropy-val", (v) => v.toFixed(2));

document.getElementById("modal-render").addEventListener("click", async () => {
  const c = ensureCtx();
  const opts = {
    profile_name: document.getElementById("modal-profile").value,
    duration_s: parseFloat(document.getElementById("modal-duration").value),
    impact_strength: parseFloat(document.getElementById("modal-strength").value),
    sharpness: parseFloat(document.getElementById("modal-sharpness").value),
    velocity: parseFloat(document.getElementById("modal-velocity").value),
    damping_anisotropy: parseFloat(document.getElementById("modal-anisotropy").value),
    excitation_shape: document.getElementById("modal-excitation").value,
    seed: parseInt(document.getElementById("modal-seed").value, 10) || 0,
  };
  try {
    let buf = await renderModalImpact(c, opts);
    buf = await maybeReverb(buf);
    playBuffer(buf);
  } catch (e) {
    alert("Modal render failed: " + e);
  }
});

// ------- Drip tab -------
bindSlider("drip-radius", "drip-radius-val", (v) => `${v.toFixed(2)} mm`);
bindSlider("drip-viscosity", "drip-viscosity-val", (v) => v.toFixed(2));
bindSlider("drip-velocity", "drip-velocity-val", (v) => v.toFixed(2));
bindSlider("drip-capillary", "drip-capillary-val", (v) => v.toFixed(2));
bindSlider("drip-duration", "drip-duration-val", (v) => `${v.toFixed(2)} s`);

document.getElementById("drip-render").addEventListener("click", async () => {
  const c = ensureCtx();
  const opts = {
    radius_mm: parseFloat(document.getElementById("drip-radius").value),
    viscosity: parseFloat(document.getElementById("drip-viscosity").value),
    surface_profile: document.getElementById("drip-surface").value,
    capillary_ringing: parseFloat(document.getElementById("drip-capillary").value),
    velocity_factor: parseFloat(document.getElementById("drip-velocity").value),
    duration_s: parseFloat(document.getElementById("drip-duration").value),
    seed: parseInt(document.getElementById("drip-seed").value, 10) || 0,
  };
  try {
    let buf = await renderDripEvent(c, opts);
    buf = await maybeReverb(buf);
    playBuffer(buf);
  } catch (e) {
    alert("Drip render failed: " + e);
  }
});

// ------- Granular tab -------
bindSlider("gran-density", "gran-density-val", (v) => `${v.toFixed(0)} /s`);
bindSlider("gran-jitter", "gran-jitter-val", (v) => v.toFixed(2));
bindSlider("gran-cluster", "gran-cluster-val", (v) => v.toFixed(2));
bindSlider("gran-energy", "gran-energy-val", (v) => v.toFixed(2));
bindSlider("gran-spread", "gran-spread-val", (v) => `${v.toFixed(2)} oct`);
bindSlider("gran-duration", "gran-duration-val", (v) => `${v.toFixed(1)} s`);

document.getElementById("gran-render").addEventListener("click", async () => {
  const c = ensureCtx();
  const opts = {
    profile_name: document.getElementById("gran-profile").value,
    density_hz: parseFloat(document.getElementById("gran-density").value),
    density_jitter: parseFloat(document.getElementById("gran-jitter").value),
    cluster_factor: parseFloat(document.getElementById("gran-cluster").value),
    energy_mean: parseFloat(document.getElementById("gran-energy").value),
    spread_octaves_override: parseFloat(document.getElementById("gran-spread").value),
    duration_s: parseFloat(document.getElementById("gran-duration").value),
    seed: parseInt(document.getElementById("gran-seed").value, 10) || 0,
  };
  try {
    let buf = renderGranularFlow(c, opts);
    buf = await maybeReverb(buf);
    playBuffer(buf);
  } catch (e) {
    alert("Granular render failed: " + e);
  }
});

// ------- Global reverb panel -------
bindSlider("global-reverb-mix", "global-reverb-mix-val", (v) => v.toFixed(2));

// ------- Tab switching -------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`panel-${target}`).classList.add("active");
  });
});
