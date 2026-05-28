// Main entry for the 2D pad demo.
//
// Wires together SoundPad2D + PadCanvas + composer + UI controls.

import { SoundPad2D } from "./pad2d.js";
import { PadCanvas } from "./pad_canvas.js";
import { renderRollingDroplet } from "./rolling_droplet.js";
import {
  composeImpossible, ALL_MATERIALS, ALL_INTERACTIONS, GENERATORS, PRESETS,
} from "./composer.js";

// ====================================================================
// Audio context
// ====================================================================
let ctx = null;
function ensureCtx() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

// ====================================================================
// Parameter spaces — each describes a 2D map and the renderFn that
// generates an AudioBuffer at each grid cell.
// ====================================================================
function makeRollingDropletSpace() {
  return {
    xParam: "roll_velocity_hz",
    yParam: "surface_hardness_proxy",  // we map this to a profile choice below
    xRange: [4, 30],
    yRange: [0, 1],
    xLabel: "Rolling speed (Hz)",
    yLabel: "Surface hardness",
    gridSize: parseInt(document.getElementById("grid-size").value, 10),
    durationS: 2.5,
    fixedParams: {
      radius_mm: 2.5,
      viscosity: 0.1,
      path_roughness: 0.35,
      body_resonance_mix: 0.7,
      cavity_mix: 0.4,
      slosh_mix: 0.3,
      shimmer_depth: 0.2,
      continuous_layer_mix: 0.75,
      discrete_mix: 0.3,
    },
    renderFn: async (ctx, params) => {
      // Map surface_hardness_proxy [0..1] to a surface profile name
      const surfaces = ["fabric", "cork", "wood", "ceramic", "stone", "glass", "metal"];
      const idx = Math.min(surfaces.length - 1,
                            Math.floor(params.surface_hardness_proxy * surfaces.length));
      return renderRollingDroplet(ctx, {
        ...params,
        surface_profile: surfaces[idx],
      });
    },
  };
}

function makeRollingContinuityGranularitySpace() {
  return {
    xParam: "continuity_proxy",
    yParam: "granularity_proxy",
    xRange: [0, 1],
    yRange: [0, 1],
    xLabel: "Continuity (impulsive → continuous)",
    yLabel: "Granularity (smooth → rough)",
    gridSize: parseInt(document.getElementById("grid-size").value, 10),
    durationS: 2.5,
    fixedParams: { radius_mm: 2.5, viscosity: 0.1, surface_profile: "ceramic" },
    renderFn: async (ctx, params) => {
      // Map continuity → roll_velocity + discrete_mix; granularity → path_roughness
      const cont = params.continuity_proxy;
      const gran = params.granularity_proxy;
      return renderRollingDroplet(ctx, {
        radius_mm: 2.5,
        viscosity: 0.1,
        surface_profile: "ceramic",
        roll_velocity_hz: 6 + 24 * cont,
        path_roughness: 0.15 + 0.7 * gran,
        body_resonance_mix: 0.4 + 0.6 * cont,
        cavity_mix: 0.4,
        slosh_mix: 0.2 + 0.4 * gran,
        shimmer_depth: 0.1 + 0.3 * cont,
        continuous_layer_mix: 0.4 + 0.7 * cont,
        discrete_mix: 0.6 - 0.5 * cont,
        duration_s: params.duration_s,
        seed: params.seed,
      });
    },
  };
}

function makeImpossibleMixSpace() {
  const baseMat = document.getElementById("base-material").value;
  const baseInt = document.getElementById("base-interaction").value;
  const overlayMat = document.getElementById("overlay-material").value;
  const overlayInt = document.getElementById("overlay-interaction").value;
  return {
    xParam: "overlay_weight_x",
    yParam: "wetness_y",
    xRange: [0, 1.0],
    yRange: [0, 1.0],
    xLabel: `Overlay weight (${overlayMat}×${overlayInt})`,
    yLabel: "Wetness",
    gridSize: parseInt(document.getElementById("grid-size").value, 10),
    durationS: 2.5,
    fixedParams: {
      _baseMat: baseMat, _baseInt: baseInt,
      _overlayMat: overlayMat, _overlayInt: overlayInt,
    },
    renderFn: async (ctx, params) => {
      const modifiers = {
        wetness: params.wetness_y,
        granularity: parseFloat(document.getElementById("k-granularity").value),
        rigidity: parseFloat(document.getElementById("k-rigidity").value),
        resonance: parseFloat(document.getElementById("k-resonance").value),
        continuity: parseFloat(document.getElementById("k-continuity").value),
      };
      return composeImpossible(ctx, {
        baseMaterial: params._baseMat,
        baseInteraction: params._baseInt,
        overlayMaterial: params._overlayMat,
        overlayInteraction: params._overlayInt,
        overlayWeight: params.overlay_weight_x,
        modifiers,
        duration_s: params.duration_s,
        seed: params.seed,
      });
    },
  };
}

const SPACE_BUILDERS = {
  rolling_droplet: makeRollingDropletSpace,
  rolling_xy_continuity: makeRollingContinuityGranularitySpace,
  impossible_mix: makeImpossibleMixSpace,
};

// ====================================================================
// State
// ====================================================================
let pad = null;
let canvas = null;
let analyser = null;
let analyserBuf = null;

// ====================================================================
// UI helpers
// ====================================================================
function bindSlider(id, labelId, fmt = (v) => v.toFixed(2)) {
  const range = document.getElementById(id);
  const label = document.getElementById(labelId);
  const upd = () => { label.textContent = fmt(parseFloat(range.value)); };
  range.addEventListener("input", upd);
  upd();
}

// Universal knobs
bindSlider("overlay-weight", "overlay-weight-val");
bindSlider("k-wetness", "k-wetness-val");
bindSlider("k-granularity", "k-granularity-val");
bindSlider("k-rigidity", "k-rigidity-val");
bindSlider("k-resonance", "k-resonance-val");
bindSlider("k-continuity", "k-continuity-val");
bindSlider("master-vol", "master-vol-val");

// Material/interaction dropdowns
function populateSelect(id, options, defaultVal) {
  const sel = document.getElementById(id);
  options.forEach(o => {
    const opt = document.createElement("option");
    opt.value = o; opt.textContent = o;
    if (o === defaultVal) opt.selected = true;
    sel.appendChild(opt);
  });
}
populateSelect("base-material", ALL_MATERIALS, "rock");
populateSelect("base-interaction", ALL_INTERACTIONS, "impact");
populateSelect("overlay-material", ALL_MATERIALS, "liquid");
populateSelect("overlay-interaction", ALL_INTERACTIONS, "splash");

// Master volume
document.getElementById("master-vol").addEventListener("input", (e) => {
  if (pad) pad.setVolume(parseFloat(e.target.value));
});

// ====================================================================
// Pad initialization
// ====================================================================
async function initPad() {
  const c = ensureCtx();
  // Tear down previous pad
  if (pad) pad.stop();
  if (canvas) canvas.destroy();

  const spaceKey = document.getElementById("space-select").value;
  const space = SPACE_BUILDERS[spaceKey]();
  pad = new SoundPad2D(c, space, (frac, label) => {
    document.getElementById("progress-fill").style.width = `${frac * 100}%`;
    document.getElementById("progress-label").textContent = label;
  });
  // Show progress overlay
  const ovl = document.getElementById("progress-overlay");
  ovl.classList.add("active");
  document.getElementById("progress-title").textContent =
    `Pre-rendering ${space.gridSize}×${space.gridSize} grid…`;

  const t0 = performance.now();
  await pad.prerender();
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
  document.getElementById("progress-title").textContent = `Ready (${elapsed}s)`;
  setTimeout(() => ovl.classList.remove("active"), 400);

  // Wire up canvas
  const canvasEl = document.getElementById("pad");
  canvas = new PadCanvas(canvasEl, pad);

  // Set up analyser for the mini spectrum
  if (!analyser) {
    analyser = c.createAnalyser();
    analyser.fftSize = 512;
    analyserBuf = new Uint8Array(analyser.frequencyBinCount);
    drawMiniSpec();
  }
  pad.setVolume(parseFloat(document.getElementById("master-vol").value));
  pad.start(analyser);
  analyser.connect(c.destination);
}

document.getElementById("init-btn").addEventListener("click", initPad);
document.getElementById("stop-btn").addEventListener("click", () => {
  if (pad) pad.stop();
});

// Change space → tear down (user must re-init)
document.getElementById("space-select").addEventListener("change", () => {
  if (pad) {
    pad.stop();
    document.getElementById("progress-overlay").classList.add("active");
    document.getElementById("progress-title").textContent = "Click \"Initialize\" to re-render grid";
    document.getElementById("progress-fill").style.width = "0%";
    document.getElementById("progress-label").textContent = "(space changed)";
  }
});

// ====================================================================
// Presets
// ====================================================================
document.querySelectorAll(".preset-btn").forEach(btn => {
  btn.addEventListener("click", async () => {
    const presetKey = btn.dataset.preset;
    const preset = PRESETS[presetKey];
    if (!preset) return;
    // Apply preset to UI
    document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("base-material").value = preset.baseMaterial;
    document.getElementById("base-interaction").value = preset.baseInteraction;
    if (preset.overlayMaterial) {
      document.getElementById("overlay-material").value = preset.overlayMaterial;
      document.getElementById("overlay-interaction").value = preset.overlayInteraction || preset.baseInteraction;
      document.getElementById("overlay-weight").value = preset.overlayWeight ?? 0.5;
      document.getElementById("overlay-weight-val").textContent = (preset.overlayWeight ?? 0.5).toFixed(2);
    }
    // Apply knob modifiers
    const mods = preset.modifiers || {};
    Object.entries(mods).forEach(([k, v]) => {
      const el = document.getElementById(`k-${k}`);
      if (el) {
        el.value = v;
        document.getElementById(`k-${k}-val`).textContent = v.toFixed(2);
      }
    });
    // Auto-switch to impossible_mix space if overlay defined
    if (preset.overlayMaterial) {
      document.getElementById("space-select").value = "impossible_mix";
    } else {
      document.getElementById("space-select").value = "rolling_droplet";
    }
    // Re-initialize
    await initPad();
  });
});

// ====================================================================
// Mini spectrum visualization
// ====================================================================
function drawMiniSpec() {
  const cv = document.getElementById("mini-spec");
  const cctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  function frame() {
    if (!analyser) return;
    const rect = cv.getBoundingClientRect();
    if (cv.width !== rect.width * dpr) {
      cv.width = rect.width * dpr;
      cv.height = rect.height * dpr;
    }
    cctx.fillStyle = "#0a0e1a";
    cctx.fillRect(0, 0, cv.width, cv.height);
    analyser.getByteFrequencyData(analyserBuf);
    const n = analyserBuf.length;
    const bw = cv.width / n;
    for (let i = 0; i < n; i++) {
      const v = analyserBuf[i] / 255;
      const h = v * cv.height;
      const hue = 30 + 180 * (1 - i / n);
      cctx.fillStyle = `hsl(${hue}, 70%, ${30 + 40 * v}%)`;
      cctx.fillRect(i * bw, cv.height - h, bw + 0.5, h);
    }
    requestAnimationFrame(frame);
  }
  frame();
}

// ====================================================================
// URL state persistence (knobs + space + base/overlay encoded in hash)
// ====================================================================
function readUrlState() {
  if (!window.location.hash) return;
  try {
    const params = new URLSearchParams(window.location.hash.slice(1));
    const setIf = (id, key) => {
      const v = params.get(key);
      if (v !== null) {
        const el = document.getElementById(id);
        if (el) {
          el.value = v;
          const valSpan = document.getElementById(`${id}-val`);
          if (valSpan) valSpan.textContent = parseFloat(v).toFixed(2);
        }
      }
    };
    setIf("k-wetness", "w");
    setIf("k-granularity", "g");
    setIf("k-rigidity", "r");
    setIf("k-resonance", "q");
    setIf("k-continuity", "c");
    setIf("overlay-weight", "ow");
    const sp = params.get("s");
    if (sp) document.getElementById("space-select").value = sp;
    const bm = params.get("bm");
    if (bm) document.getElementById("base-material").value = bm;
    const bi = params.get("bi");
    if (bi) document.getElementById("base-interaction").value = bi;
    const om = params.get("om");
    if (om) document.getElementById("overlay-material").value = om;
    const oi = params.get("oi");
    if (oi) document.getElementById("overlay-interaction").value = oi;
  } catch (e) {
    console.warn("Bad URL state:", e);
  }
}

function writeUrlState() {
  const params = new URLSearchParams();
  params.set("s", document.getElementById("space-select").value);
  params.set("bm", document.getElementById("base-material").value);
  params.set("bi", document.getElementById("base-interaction").value);
  params.set("om", document.getElementById("overlay-material").value);
  params.set("oi", document.getElementById("overlay-interaction").value);
  params.set("ow", document.getElementById("overlay-weight").value);
  params.set("w", document.getElementById("k-wetness").value);
  params.set("g", document.getElementById("k-granularity").value);
  params.set("r", document.getElementById("k-rigidity").value);
  params.set("q", document.getElementById("k-resonance").value);
  params.set("c", document.getElementById("k-continuity").value);
  history.replaceState(null, "", `#${params.toString()}`);
}

// Update URL on any control change (debounced)
let urlSaveTimer = null;
document.querySelectorAll("input, select").forEach(el => {
  el.addEventListener("input", () => {
    clearTimeout(urlSaveTimer);
    urlSaveTimer = setTimeout(writeUrlState, 300);
  });
});

// Read URL state at startup
readUrlState();

// ====================================================================
// Keyboard shortcuts
// ====================================================================
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === " ") {
    e.preventDefault();
    if (pad) {
      if (pad.playing) pad.stop();
      else { pad.start(analyser); analyser.connect(ctx.destination); }
    }
  } else if (e.key === "ArrowLeft" && pad) {
    pad.moveTo(pad.x - 0.05, pad.y);
  } else if (e.key === "ArrowRight" && pad) {
    pad.moveTo(pad.x + 0.05, pad.y);
  } else if (e.key === "ArrowUp" && pad) {
    pad.moveTo(pad.x, pad.y + 0.05);
  } else if (e.key === "ArrowDown" && pad) {
    pad.moveTo(pad.x, pad.y - 0.05);
  }
});
