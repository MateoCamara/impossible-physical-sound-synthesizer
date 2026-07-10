# WebAudio demo — physics engine in the browser

A zero-dependency port of the modal, drip, granular, rolling-droplet,
friction, liquid (splash/pour) and reverb primitives to the Web Audio
API. No backend, no neural networks. Click a slider, hit **Render**,
listen.

## Run locally

```bash
cd web
python3 -m http.server 8080
# open http://localhost:8080
```

## Deploy publicly

Drag the `web/` folder into [Netlify Drop](https://app.netlify.com/drop)
or push it to a GitHub Pages branch. Pure HTML+JS.

## Files

| File | Purpose |
|---|---|
| `index.html` | UI with seven tabs (Modal / Drip / Granular / Rolling / Friction / Splash / Pour) + global reverb panel. |
| `main.js` | Wiring, AudioContext bootstrapping, slider bindings, reverb pass-through. |
| `modal.js` | `MODAL_PROFILES` + `renderModalImpact(ctx, opts)`. |
| `drip.js` | Surface profiles (13 materials) + `renderDripEvent(ctx, opts)`. |
| `granular.js` | `GRAIN_PROFILES` (8 minerals) + `renderGranularFlow(ctx, opts)`. |
| `rolling_droplet.js` | `renderRollingDroplet(ctx, opts)` — continuous sustained-contact model (body resonance, cavity, Rayleigh modes, microbubble cloud, surface ringing, stick-slip) layered under attenuated discrete drip events; also exports `writeDripInline` reused by `liquid.js`. |
| `friction.js` | `renderScrape(ctx, opts)` — noise + velocity LFO + stick-slip bumps + bandpass + body resonator. |
| `liquid.js` | `renderSplash(ctx, opts)` / `renderPour(ctx, opts)` — bubble bursts and dense drip trains built on `writeDripInline`. |
| `reverb.js` | `IR_PRESETS` + `generateIR(ctx, preset, seed)` + `applyReverb(ctx, dry, ir, mix)`. |

## Parity with Python

| Python | JS |
|---|---|
| `impossible_mix.physics.modal.synth_modal_impact` | `modal.js → renderModalImpact` |
| `impossible_mix.physics.modal.PROFILES` | `modal.js → MODAL_PROFILES` |
| `impossible_mix.physics.droplet.synth_drip_event` | `drip.js → renderDripEvent` |
| `impossible_mix.physics.droplet.SURFACE_PROFILES` | `drip.js → SURFACE_PROFILES` (13 surfaces) |
| `impossible_mix.physics.droplet.synth_rolling_droplet` | `rolling_droplet.js → renderRollingDroplet` |
| `impossible_mix.physics.friction.synth_scrape` | `friction.js → renderScrape` (stick-slip bumps implemented) |
| `impossible_mix.physics.granular.synth_granular_flow` | `granular.js → renderGranularFlow` |
| `impossible_mix.physics.granular.GRAIN_PROFILES` | `granular.js → GRAIN_PROFILES` (8 minerals) |
| `impossible_mix.physics.liquid.synth_splash` | `liquid.js → renderSplash` |
| `impossible_mix.physics.liquid.synth_pour` | `liquid.js → renderPour` |
| `impossible_mix.physics.reverb.generate_ir` | `reverb.js → generateIR` |
| `impossible_mix.physics.reverb.apply_reverb` | `reverb.js → applyReverb` (uses `ConvolverNode`) |
| `impossible_mix.physics.composer.compose_impossible` | — (multi-layer composition stays in Python) |
| `impossible_mix.physics.diff.*` | — (DDSP-style fitting stays in Python) |

## Implementation notes

- Modal voices use peaking `BiquadFilterNode`s with `Q ≈ π · f · t60` to
  approximate the Python resonators without recursive IIR code.
- The drip chirp uses `OscillatorNode.frequency.exponentialRampToValueAtTime`
  for the Minnaert sweep; modal tails go through bandpass filters.
- Granular and rolling-droplet renderers **build the output `AudioBuffer`
  manually in JS** instead of scheduling hundreds of nodes. Granular
  sums damped sinusoids per grain; rolling-droplet sums several
  continuous, per-sample-synthesised layers (body resonance, Helmholtz
  cavity, Rayleigh shape modes, microbubble cloud, surface ringing,
  rolling stick-slip) — a rolling droplet is modelled as sustained
  contact, not a series of discrete impacts — with attenuated discrete
  drip events layered on top as texture. This scales smoothly and keeps
  render times tolerable in the browser.
- Friction uses an `OfflineAudioContext` with a pre-generated noise
  buffer modulated by an embedded velocity envelope, fed through a
  bandpass + peaking body filter.
- Reverb is `ConvolverNode` with a synthetic IR (noise · exp-decay +
  discrete early reflections + warm-tilt LPF colour) generated inline.
- A deterministic Mulberry32 PRNG mirrors the seed argument from Python.
