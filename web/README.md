# WebAudio MVP — modal & drip in the browser

A zero-dependency port of the modal-impact and drip-event primitives to
the Web Audio API. No backend, no neural networks. Click a slider, hit
**Render**, listen.

## Run locally

```bash
cd web
python3 -m http.server 8080
# open http://localhost:8080
```

## Deploy publicly

Drag the `web/` folder into [Netlify Drop](https://app.netlify.com/drop)
or push it to a GitHub Pages branch. Pure HTML+JS, ~10 KB total.

## Files

| File | Purpose |
|---|---|
| `index.html` | UI with three tabs (Modal / Drip / Granular) + global reverb panel. |
| `main.js` | Wiring, AudioContext bootstrapping, slider bindings, reverb pass-through. |
| `modal.js` | `MODAL_PROFILES` + `renderModalImpact(ctx, opts)`. |
| `drip.js` | Surface profiles + `renderDripEvent(ctx, opts)`. |
| `granular.js` | `GRAIN_PROFILES` (8 minerals) + `renderGranularFlow(ctx, opts)`. |
| `reverb.js` | `IR_PRESETS` + `generateIR(ctx, preset, seed)` + `applyReverb(ctx, dry, ir, mix)`. |

## Parity with Python

This MVP covers a strict subset of the Python engine. The table maps the
JS functions to their Python counterparts:

| Python | JS |
|---|---|
| `impossible_mix.physics.modal.synth_modal_impact` | `modal.js → renderModalImpact` |
| `impossible_mix.physics.modal.PROFILES` | `modal.js → MODAL_PROFILES` |
| `impossible_mix.physics.droplet.synth_drip_event` | `drip.js → renderDripEvent` |
| `impossible_mix.physics.droplet.SURFACE_PROFILES` | `drip.js → SURFACE_PROFILES` (subset of 6 surfaces) |
| `impossible_mix.physics.granular.synth_granular_flow` | `granular.js → renderGranularFlow` |
| `impossible_mix.physics.granular.GRAIN_PROFILES` | `granular.js → GRAIN_PROFILES` (8 minerals) |
| `impossible_mix.physics.reverb.generate_ir` | `reverb.js → generateIR` |
| `impossible_mix.physics.reverb.apply_reverb` | `reverb.js → applyReverb` (uses `ConvolverNode`) |
| `impossible_mix.physics.droplet.synth_rolling_droplet` | — (not yet in JS) |
| `impossible_mix.physics.composer.compose_impossible` | — |

## Implementation notes

- Each modal impact runs inside an `OfflineAudioContext` to render off
  the audio clock; the result is then played via a `BufferSource`. This
  avoids the timing wobble of triggering many `BiquadFilter` voices on
  the live context.
- The biquad `Q` in each modal voice is set from t60 with the
  approximation $Q \approx \pi \cdot f_h \cdot t_{60}$, which gives a
  similar decay to the Python resonator without needing a full IIR.
- The drip event uses `OscillatorNode.frequency.exponentialRampToValueAtTime`
  for the Minnaert chirp, plus short `BufferSource` clicks and modal
  tails routed through bandpass filters.
- A deterministic Mulberry32 PRNG mirrors the seed argument from Python.
