# WebAudio demo — physics engine in the browser

A zero-dependency port of the modal, drip, granular, rolling-droplet,
friction and reverb primitives to the Web Audio API. No backend, no
neural networks. Click a slider, hit **Render**, listen.

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
| `index.html` | UI with five tabs (Modal / Drip / Granular / Rolling / Friction) + global reverb panel. |
| `main.js` | Wiring, AudioContext bootstrapping, slider bindings, reverb pass-through. |
| `modal.js` | `MODAL_PROFILES` + `renderModalImpact(ctx, opts)`. |
| `drip.js` | Surface profiles + `renderDripEvent(ctx, opts)`. |
| `granular.js` | `GRAIN_PROFILES` (8 minerals) + `renderGranularFlow(ctx, opts)`. |
| `rolling_droplet.js` | `renderRollingDroplet(ctx, opts)` — quasi-periodic drip train with inline buffer mixing. |
| `friction.js` | `renderScrape(ctx, opts)` — noise + velocity LFO + bandpass + body resonator. |
| `reverb.js` | `IR_PRESETS` + `generateIR(ctx, preset, seed)` + `applyReverb(ctx, dry, ir, mix)`. |

## Parity with Python

| Python | JS |
|---|---|
| `impossible_mix.physics.modal.synth_modal_impact` | `modal.js → renderModalImpact` |
| `impossible_mix.physics.modal.PROFILES` | `modal.js → MODAL_PROFILES` |
| `impossible_mix.physics.droplet.synth_drip_event` | `drip.js → renderDripEvent` |
| `impossible_mix.physics.droplet.SURFACE_PROFILES` | `drip.js → SURFACE_PROFILES` (6 surfaces) |
| `impossible_mix.physics.droplet.synth_rolling_droplet` | `rolling_droplet.js → renderRollingDroplet` |
| `impossible_mix.physics.friction.synth_scrape` | `friction.js → renderScrape` (no stick-slip in JS) |
| `impossible_mix.physics.granular.synth_granular_flow` | `granular.js → renderGranularFlow` |
| `impossible_mix.physics.granular.GRAIN_PROFILES` | `granular.js → GRAIN_PROFILES` (8 minerals) |
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
  manually in JS** (damped sinusoids summed per grain or per contact)
  instead of scheduling hundreds of nodes. This scales smoothly past
  200 grains and keeps render times tolerable in the browser.
- Friction uses an `OfflineAudioContext` with a pre-generated noise
  buffer modulated by an embedded velocity envelope, fed through a
  bandpass + peaking body filter.
- Reverb is `ConvolverNode` with a synthetic IR (noise · exp-decay +
  discrete early reflections + warm-tilt LPF colour) generated inline.
- A deterministic Mulberry32 PRNG mirrors the seed argument from Python.
