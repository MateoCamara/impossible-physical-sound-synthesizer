import { MODAL_PROFILES, renderModalImpact } from "./modal.js";
import { renderDripEvent } from "./drip.js";

// Single shared AudioContext (created on first user gesture)
let ctx = null;
function ensureCtx() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

function playBuffer(buf) {
  const c = ensureCtx();
  const src = c.createBufferSource();
  src.buffer = buf;
  src.connect(c.destination);
  src.start();
}

// Helper: build a slider with live label
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
    const buf = await renderModalImpact(c, opts);
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
    const buf = await renderDripEvent(c, opts);
    playBuffer(buf);
  } catch (e) {
    alert("Drip render failed: " + e);
  }
});

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
