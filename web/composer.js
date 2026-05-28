// Impossible composer: physics-informed two-layer mixer.
//
// Mirrors impossible_mix/physics/composer.py: GENERATORS table maps
// (material, interaction) -> render function, with 5 universal knobs
// (wetness, granularity, rigidity, resonance, continuity) mapped to
// generator-specific physical parameters.
//
// composeImpossible(base, overlay, weight, modifiers) renders both
// layers and mixes them with overlay_weight.

import { renderModalImpact, renderModalRoll } from "./modal.js";
import { renderDripEvent } from "./drip.js";
import { renderGranularFlow } from "./granular.js";
import { renderRollingDroplet } from "./rolling_droplet.js";
import { renderScrape } from "./friction.js";
import { renderSplash, renderPour } from "./liquid.js";

// ====================================================================
// Knob → generator-param mappings (one per primitive)
// ====================================================================

function _modalImpactParams(spec) {
  const profileMap = {
    rock: "rock", metal: "metal", wood: "wood", glass: "glass",
    earth: "earth", fabric: "fabric", rubber: "rubber", bone: "bone",
    ice: "ice", chitin: "chitin", plasma: "plasma",
  };
  return {
    profile_name: profileMap[spec.material] || "rock",
    duration_s: spec.duration_s,
    impact_strength: 0.6 + 0.4 * spec.resonance,
    sharpness: 1.0 + 1.5 * spec.rigidity,
    velocity: 1.0,
    damping_anisotropy: 0.3 + 0.5 * spec.resonance,
    excitation_shape: spec.rigidity > 0.6 ? "steel" : (spec.rigidity > 0.3 ? "wood" : "felt"),
    seed: spec.seed,
  };
}

function _modalRollParams(spec) {
  const profileMap = {
    rock: "rock", metal: "metal", wood: "wood", glass: "glass",
    earth: "earth", fabric: "fabric", rubber: "rubber", bone: "bone",
    ice: "ice", chitin: "chitin", plasma: "plasma",
  };
  return {
    profile_name: profileMap[spec.material] || "rock",
    duration_s: spec.duration_s,
    rate_hz: 10 + 30 * (1 - spec.continuity * 0.5),
    jitter: 0.3 + 0.5 * (1 - spec.continuity),
    strength: 0.6,
    seed: spec.seed,
  };
}

function _scrapeParams(spec) {
  const surfMap = {
    metal: "metal", wood: "wood", glass: "glass", ceramic: "ceramic",
    fabric: "fabric", stone: "stone", rubber: "rubber", ice: "ice",
    rock: "stone",  // rock → stone surface profile
  };
  return {
    surface_profile: surfMap[spec.material] || null,
    surface_hardness: spec.rigidity,
    velocity_mean: 0.5 + 0.4 * spec.continuity,
    velocity_jitter: 0.3 + 0.4 * (1 - spec.continuity),
    stick_slip: 0.2 + 0.5 * (1 - spec.continuity),
    roughness: spec.granularity,
    pressure: 0.6,
    duration_s: spec.duration_s,
    seed: spec.seed,
  };
}

function _dripParams(spec) {
  // Single drip in the middle of the clip
  return {
    radius_mm: 2.0 * (1 + spec.rigidity * 0.5),
    viscosity: 0.5 * (1 - spec.wetness * 0.5),
    surface_profile: "ceramic",
    capillary_ringing: 0.5,
    velocity_factor: 1.0,
    duration_s: Math.min(spec.duration_s, 0.6),
    seed: spec.seed,
  };
}

function _splashParams(spec) {
  return {
    intensity: 0.6 + 0.4 * spec.wetness,
    bubble_size_mean_mm: 2.0 + 3.0 * spec.rigidity,
    bubble_size_var: 0.6,
    n_bubbles: Math.floor(20 + 40 * spec.wetness),
    spread_ms: 100 - 50 * spec.continuity,
    viscosity: 0.3 * (1 - spec.wetness * 0.5),
    duration_s: spec.duration_s,
    seed: spec.seed,
  };
}

function _pourParams(spec) {
  return {
    flow_rate: 0.4 + 0.6 * spec.continuity,
    bubble_size_mean_mm: 1.5,
    viscosity: 0.3 * (1 - spec.wetness * 0.5),
    duration_s: spec.duration_s,
    seed: spec.seed,
  };
}

function _rollingDropletParams(spec) {
  const surfMap = {
    metal: "metal", wood: "wood", glass: "glass", ceramic: "ceramic",
    fabric: "fabric", stone: "stone", ice: "ice",
    rock: "stone",
  };
  return {
    radius_mm: 1.5 + 2.5 * (1 - spec.rigidity),
    viscosity: Math.max(0, 0.6 - 0.5 * spec.wetness),
    surface_profile: surfMap[spec.material] || "ceramic",
    roll_velocity_hz: 8 + 20 * spec.continuity,
    path_roughness: 0.2 + 0.5 * spec.granularity,
    // Use the "water roller" defaults (continuous protagonists)
    body_resonance_mix: 0.7,
    cavity_mix: 0.4,
    slosh_mix: 0.3,
    shimmer_depth: 0.2,
    continuous_layer_mix: 0.75,
    discrete_mix: 0.3,
    duration_s: spec.duration_s,
    seed: spec.seed,
  };
}

function _granularParams(spec) {
  const grainMap = {
    rock: "coarse_gravel", metal: "fine_gravel", gravel: "pebble",
    earth: "sand", ice: "ice_shards",
  };
  return {
    profile_name: grainMap[spec.material] || "pebble",
    density_hz: 30 + 200 * spec.continuity,
    density_jitter: 0.4 + 0.5 * (1 - spec.continuity),
    cluster_factor: 0.3 + 0.4 * (1 - spec.continuity),
    energy_mean: 0.5,
    spread_octaves_override: 0.6 + 0.5 * spec.granularity,
    duration_s: spec.duration_s,
    seed: spec.seed,
  };
}

// ====================================================================
// Generators: (material, interaction) -> render function returning AudioBuffer
// ====================================================================

const GENERATOR_FNS = {
  solid_impact: async (ctx, spec) => renderModalImpact(ctx, _modalImpactParams(spec)),
  solid_roll:   async (ctx, spec) => renderModalRoll(ctx, _modalRollParams(spec)),
  scrape:       async (ctx, spec) => renderScrape(ctx, _scrapeParams(spec)),
  drip:         async (ctx, spec) => renderDripEvent(ctx, _dripParams(spec)),
  splash:       async (ctx, spec) => renderSplash(ctx, _splashParams(spec)),
  pour:         async (ctx, spec) => renderPour(ctx, _pourParams(spec)),
  rolling_droplet: async (ctx, spec) => renderRollingDroplet(ctx, _rollingDropletParams(spec)),
  granular:     async (ctx, spec) => renderGranularFlow(ctx, _granularParams(spec)),
};

export const GENERATORS = {
  // solids
  "rock|impact":   "solid_impact",  "metal|impact":  "solid_impact",
  "wood|impact":   "solid_impact",  "glass|impact":  "solid_impact",
  "rubber|impact": "solid_impact",  "bone|impact":   "solid_impact",
  "ice|impact":    "solid_impact",  "chitin|impact": "solid_impact",
  "plasma|impact": "solid_impact",
  "rock|roll":     "solid_roll",    "metal|roll":    "solid_roll",
  "rubber|roll":   "solid_roll",    "ice|roll":      "solid_roll",
  "plasma|roll":   "solid_roll",
  "rock|scrape":   "scrape",        "metal|scrape":  "scrape",
  "wood|scrape":   "scrape",        "glass|scrape":  "scrape",
  "rubber|scrape": "scrape",        "bone|scrape":   "scrape",
  "ice|scrape":    "scrape",        "chitin|scrape": "scrape",
  "fabric|drag":   "scrape",        "rock|drag":     "scrape",
  // liquids
  "liquid|drip":   "drip",
  "liquid|splash": "splash",
  "liquid|impact": "splash",
  "liquid|pour":   "pour",
  "liquid|roll":   "rolling_droplet",  // ★ THE impossible
  // granular
  "gravel|step":   "granular",
  "gravel|roll":   "granular",
  "gravel|scrape": "granular",
  "gravel|pour":   "granular",
  "earth|step":    "granular",
};

export const ALL_MATERIALS = [
  "rock", "metal", "wood", "glass", "earth", "fabric",
  "rubber", "bone", "ice", "chitin", "plasma",
  "liquid", "gravel",
];

export const ALL_INTERACTIONS = [
  "impact", "roll", "scrape", "drag", "drip", "splash", "pour", "step",
];

// ====================================================================
// Public API
// ====================================================================

/**
 * Default spec with the 5 universal knobs.
 */
export function defaultSpec(overrides = {}) {
  return {
    material: "rock",
    interaction: "impact",
    duration_s: 5.0,
    seed: 0,
    wetness: 0.0,
    granularity: 0.0,
    rigidity: 0.5,
    resonance: 0.5,
    continuity: 0.5,
    ...overrides,
  };
}

/** Compose a single layer. */
export async function compose(ctx, spec) {
  const key = `${spec.material}|${spec.interaction}`;
  let genName = GENERATORS[key];
  if (!genName) {
    // Fallback: pick the canonical interaction for this material
    const fallback = {
      liquid: "splash", gravel: "step", fabric: "drag",
    }[spec.material] || "impact";
    const fbKey = `${spec.material}|${fallback}`;
    genName = GENERATORS[fbKey];
    if (!genName) {
      throw new Error(`No generator for ${spec.material} × ${spec.interaction}`);
    }
    spec = { ...spec, interaction: fallback };
  }
  return GENERATOR_FNS[genName](ctx, spec);
}

/**
 * Compose an "impossible" sound: base layer + overlay layer mixed with weight.
 *
 * @param {AudioContext} ctx
 * @param {object} opts
 *   baseMaterial, baseInteraction
 *   overlayMaterial, overlayInteraction (optional — if absent, returns base only)
 *   overlayWeight: 0..1+
 *   modifiers: { wetness, granularity, rigidity, resonance, continuity }
 *   duration_s, seed
 * @returns {Promise<AudioBuffer>}
 */
export async function composeImpossible(ctx, opts) {
  const {
    baseMaterial = "rock",
    baseInteraction = "impact",
    overlayMaterial = null,
    overlayInteraction = null,
    overlayWeight = 0.5,
    modifiers = {},
    duration_s = 5.0,
    seed = 0,
  } = opts;

  const baseSpec = defaultSpec({
    material: baseMaterial, interaction: baseInteraction,
    duration_s, seed, ...modifiers,
  });
  const baseBuf = await compose(ctx, baseSpec);

  if (!overlayMaterial) return baseBuf;

  const overlaySpec = defaultSpec({
    material: overlayMaterial,
    interaction: overlayInteraction || baseInteraction,
    duration_s, seed: seed + 1, ...modifiers,
  });
  const overlayBuf = await compose(ctx, overlaySpec);

  // Mix sample-wise
  const sr = ctx.sampleRate;
  const n = Math.min(baseBuf.length, overlayBuf.length);
  const outBuf = ctx.createBuffer(1, n, sr);
  const out = outBuf.getChannelData(0);
  const bd = baseBuf.getChannelData(0);
  const od = overlayBuf.getChannelData(0);
  let peak = 0;
  for (let i = 0; i < n; i++) {
    const v = bd[i] + overlayWeight * od[i];
    out[i] = v;
    if (Math.abs(v) > peak) peak = Math.abs(v);
  }
  if (peak > 0.95) {
    const scale = 0.95 / peak;
    for (let i = 0; i < n; i++) out[i] *= scale;
  }
  return outBuf;
}

// ====================================================================
// Preset "famous" impossible combinations
// ====================================================================

export const PRESETS = {
  "rolling-droplet": {
    label: "Rolling droplet on ceramic",
    baseMaterial: "liquid", baseInteraction: "roll",
    overlayMaterial: null,
    modifiers: { wetness: 0.7, continuity: 0.6, rigidity: 0.5, granularity: 0.3 },
  },
  "liquid-rock": {
    label: "Liquid rock impact",
    baseMaterial: "rock", baseInteraction: "impact",
    overlayMaterial: "liquid", overlayInteraction: "splash", overlayWeight: 0.6,
    modifiers: { wetness: 0.8, resonance: 0.4, rigidity: 0.7 },
  },
  "wet-gravel": {
    label: "Wet gravel scrape",
    baseMaterial: "gravel", baseInteraction: "scrape",
    overlayMaterial: "liquid", overlayInteraction: "pour", overlayWeight: 0.35,
    modifiers: { wetness: 0.6, granularity: 0.9, continuity: 0.5 },
  },
  "plasma-drop": {
    label: "Plasma drop on metal",
    baseMaterial: "plasma", baseInteraction: "roll",
    overlayMaterial: "liquid", overlayInteraction: "splash", overlayWeight: 0.4,
    modifiers: { wetness: 0.5, resonance: 0.9, rigidity: 0.3 },
  },
  "mercury-ceramic": {
    label: "Mercury rolling on ceramic",
    baseMaterial: "liquid", baseInteraction: "roll",
    overlayMaterial: "metal", overlayInteraction: "roll", overlayWeight: 0.3,
    modifiers: { wetness: 0.4, rigidity: 0.8, continuity: 0.7 },
  },
  "ice-lava": {
    label: "Ice drop in lava",
    baseMaterial: "ice", baseInteraction: "impact",
    overlayMaterial: "liquid", overlayInteraction: "splash", overlayWeight: 0.5,
    modifiers: { wetness: 0.6, rigidity: 0.85, resonance: 0.7 },
  },
};
