# A cookbook of impossible-sound recipes

This document is a hands-on companion to the paper. Each recipe pairs a
description with a parameter set that reproduces the sound on the
proposed engine. Run any code block as-is from the repository root.

The framework exposes the following primitive layers and exotic generators:

| Primitive | Module | Key parameters |
|---|---|---|
| Modal impact | `physics.modal` | profile, excitation_shape, velocity, damping_anisotropy, coupling |
| Modal roll  | `physics.modal` | profile, rate_hz, jitter |
| Friction (scrape/drag) | `physics.friction` | surface_profile, roughness, stick_slip, velocity_mean, pressure |
| Granular flow | `physics.granular` | grain_profile, density_hz, cluster_factor, surface_coupling |
| Drip event | `physics.droplet` | droplet_radius_mm, viscosity, surface_profile, bounce_chain_length, capillary_ringing |
| Rolling droplet | `physics.droplet` | + roll_velocity_hz, path_roughness, drying_factor |
| Splash | `physics.liquid` | intensity, bubble_size_mean_mm, viscosity, spread_ms |
| Pour | `physics.liquid` | flow_rate, bubble_size_mean_mm, viscosity |
| Rain | `physics.exotic` | intensity, drop_size_mm, wind_strength, gust_rate_hz |
| Fire | `physics.exotic` | intensity, crackle_density |
| Thunder | `physics.exotic` | distance, intensity |
| Glass break | `physics.exotic` | n_shards |
| Ocean wave | `physics.exotic` | breaking_intensity |

Material catalogs:

| Surface profiles | Grain profiles | Modal profiles |
|---|---|---|
| fabric, wood, ceramic, glass, metal, stone, water | pebble, fine_gravel, coarse_gravel, sand, crushed_glass, broken_ceramic, basalt, ice | metal, rock, wood, glass, earth, fabric |

Excitation shapes: `default | felt | wood | steel | brush | impulse`

---

## Recipe 1: a single drop on ceramic

```python
from impossible_mix.physics.droplet import DropletParams, synth_drip_event
p = DropletParams(droplet_radius_mm=2.0, viscosity=0.0,
                   surface_profile="ceramic",
                   bounce_chain_length=3, bounce_decay=0.55,
                   capillary_ringing=0.6,
                   duration_s=0.5, seed=42)
wav = synth_drip_event(p, sr=44100)
```

## Recipe 2: rolling droplet, drying surface

A droplet rolling on glass that progressively dries: the liquid texture
fades over the second half of the clip while the rolling pattern remains.

```python
from impossible_mix.physics.droplet import DropletParams, synth_rolling_droplet
p = DropletParams(droplet_radius_mm=2.2, viscosity=0.1,
                   surface_profile="glass", roll_velocity_hz=18,
                   path_roughness=0.4, drying_factor=0.8,
                   duration_s=6.0, seed=42)
wav = synth_rolling_droplet(p)
```

## Recipe 3: mercury rolling on metal (impossible)

Small dense droplet that bounces irregularly on a metallic plate.

```python
from impossible_mix.physics.droplet_presets import get_preset
from impossible_mix.physics.droplet import synth_rolling_droplet
p = get_preset("mercury", duration_s=5.0)
wav = synth_rolling_droplet(p)
```

## Recipe 4: bell strike with felt mallet

A glass-like bell, struck with a soft mallet, with strong inter-modal
coupling for organic beating.

```python
from impossible_mix.physics.modal import PROFILES, synth_modal_impact
wav = synth_modal_impact(PROFILES["glass"], sr=44100,
                          duration_s=3.0, impact_strength=0.9,
                          excitation_shape="felt", coupling=0.6,
                          damping_anisotropy=0.7, velocity=1.2)
```

## Recipe 5: heavy rain in a windy night

Dense drip cloud whose density rises and falls with wind gusts. The
underlying bed gains a wind whoosh.

```python
from impossible_mix.physics.exotic import synth_rain
wav = synth_rain(duration_s=8.0, intensity=0.8, drop_size_mm=1.2,
                  wind_strength=0.7, gust_rate_hz=0.4)
```

## Recipe 6: gravel scrape under water (impossible)

Coarse gravel on a stone surface, overlaid with a liquid pour to create
a wet scraping texture.

```python
from impossible_mix.physics.composer import compose_impossible
wav = compose_impossible(
    base_material="gravel", base_interaction="scrape",
    overlay_material="liquid", overlay_interaction="pour",
    overlay_weight=0.4,
    modifiers=dict(wetness=0.7, granularity=0.9),
    duration_s=5.0, seed=42,
)
```

## Recipe 7: a story — droplet falls, rolls, splashes, in a cathedral

Cinematic event sequence + spatial placement + cathedral reverb.

```python
from impossible_mix.physics.sequences import droplet_story
from impossible_mix.physics.spatial import place_source
from impossible_mix.physics.reverb import generate_ir, apply_reverb
sr = 44100
story = droplet_story(duration_s=8.0, seed=42)
stereo = place_source(story, sr, distance_m=4.0, pan=0.1)
ir = generate_ir("cathedral", sr=sr, seed=42)
wet = apply_reverb(stereo, ir, mix=0.45)
```

## Recipe 8: a droplet traversing the stereo field

Per-event panning + distance: a sequence of events placed independently
in space.

```python
from impossible_mix.physics.sequences import Sequence, evt_drip, evt_rolling, evt_splash
seq = Sequence(duration_s=8.0)
seq.add_at(0.3, evt_drip(radius_mm=2.0), gain=0.9, pan=-0.7, distance_m=1.5)
seq.add_at(1.0, evt_rolling(duration_s=4.0, surface_hardness=0.5),
           gain=0.8, pan=-0.3, distance_m=1.0)
seq.add_at(5.0, evt_splash(intensity=0.6), gain=0.7, pan=0.4, distance_m=2.0)
stereo = seq.render()  # (T, 2) float32
```

## Recipe 9: lava drop on flesh (impossible)

A viscous droplet (lava preset) hitting a soft surface (modelled as
fabric). Combines a slow chirp with a soft modal tail.

```python
from impossible_mix.physics.droplet import DropletParams, synth_drip_event
p = DropletParams(droplet_radius_mm=4.0, viscosity=0.9,
                   surface_profile="fabric",
                   roll_velocity_hz=2, path_roughness=0.1,
                   bounce_chain_length=2, bounce_decay=0.4,
                   capillary_ringing=0.1, duration_s=1.5, seed=42)
wav = synth_drip_event(p, sr=44100)
```

## Recipe 10: fire crackling on a winter night

Wood burning, with modal crackles (fibre fractures) over a warm bed.
The chimney atmosphere is added by an additional room reverb (`small_room`).

```python
from impossible_mix.physics.exotic import synth_fire
from impossible_mix.physics.reverb import generate_ir, apply_reverb
sr = 44100
fire = synth_fire(duration_s=10.0, intensity=0.65, crackle_density=0.7)
ir = generate_ir("small_room", sr=sr)
wav = apply_reverb(fire, ir, mix=0.25)
```

---

## Recipe 11: bell with selective damping (multi-band)

A bell where the fundamental rings for 1.5 s but the upper partials die
in 100 ms — produces a "muted bell" effect impossible with global damping.

```python
from impossible_mix.physics.modal import synth_modal_impact, PROFILES
wav = synth_modal_impact(
    PROFILES["metal"], sr=44100, duration_s=3.0,
    impact_strength=0.9, excitation_shape="felt",
    t60_per_mode={0: 1.5, 1: 0.15, 2: 0.10, 3: 0.05, 4: 0.05},
)
```

## Recipe 12: bell with custom ADSR envelope

An ADSR envelope on the exciter (instead of the hardcoded `felt`/`steel`
shapes) gives full control over attack/release independently of the
modal bank.

```python
from impossible_mix.physics.modal import synth_modal_impact, PROFILES, EnvelopeParams
env = EnvelopeParams(attack_ms=30, hold_ms=20, release_ms=80, sustain_db=-3)
wav = synth_modal_impact(
    PROFILES["glass"], sr=44100, duration_s=3.0,
    envelope_params=env, coupling=0.4,
)
```

## Recipe 13: a single drip frozen into a 5 s drone

Granular freezer turns a 400 ms drip event into a sustained 5 s drone
by repeating a small grain with Hann cross-fade.

```python
from impossible_mix.physics.droplet import DropletParams, synth_drip_event
from impossible_mix.physics.timewarp import FreezerParams, synth_granular_freezer
src = synth_drip_event(DropletParams(droplet_radius_mm=2.0,
                                       surface_profile="ceramic",
                                       duration_s=0.4), sr=44100)
drone = synth_granular_freezer(
    src, FreezerParams(freeze_start_s=0.08, grain_ms=120,
                       output_duration_s=5.0, overlap=0.6),
    sr=44100,
)
```

## Recipe 14: slow-motion splash (½× time stretch)

Stretching the audio to half-speed without altering pitch reveals the
cascade of bubbles that a normal-speed splash hides.

```python
from impossible_mix.physics.liquid import SplashParams, synth_splash
from impossible_mix.physics.timewarp import TimeStretchParams, apply_time_stretch
src = synth_splash(SplashParams(intensity=0.8, n_bubbles=40, duration_s=2.0),
                    sr=44100)
slow = apply_time_stretch(src, TimeStretchParams(rate=0.5, method="librosa"),
                            sr=44100)
```

## Recipe 15: droplet inside St Andrews chapel (real IR)

Requires running `python scripts/24_download_irs.py` first.

```python
from impossible_mix.physics.sequences import droplet_story
from impossible_mix.physics.reverb import get_ir, apply_reverb
from impossible_mix.physics.spatial import place_source
dry = droplet_story(duration_s=8.0, seed=42)
ir = get_ir("st_andrews_chapel", sr=44100)
stereo = place_source(dry, 44100, distance_m=5.0, pan=0.1)
wet = apply_reverb(stereo, ir, mix=0.55)
```

## Recipe 16: a droplet sonata in 5 lines of DSL

The text-driven DSL parses a recipe and assembles the corresponding
`Sequence`. Per-event `pan` and `distance_m` make the output stereo
automatically.

```python
from impossible_mix.physics.sequences import parse_dsl
recipe = """
# Droplet sonata
drip(radius_mm=1.5) @ 0.3s gain=0.8 pan=-0.6
drip(radius_mm=2.0) @ 0.7s gain=0.9 pan=-0.2
roll(duration_s=2.5, roll_velocity_hz=14) @ 1.2s distance_m=1.0
splash(intensity=0.7, n_bubbles=30) @ 4.0s pan=0.4 distance_m=2.0
impact(material=rock, rigidity=0.7) @ 5.6s gain=0.6
"""
wav = parse_dsl(recipe).render()  # stereo float32
```

Supported DSL functions: `drip`, `roll`, `splash`, `impact`, `pour`. Each
line accepts named arguments (`radius_mm=2`, `material=rock`, etc.) plus
optional `gain`, `pan` and `distance_m` extras for the event placement.

## Recipe 17: play the engine from a MIDI keyboard

See `MIDI_INPUT.md` for full instructions. The short version:

```bash
pip install -e ".[midi]"
python scripts/25_midi_input.py --list-ports
python scripts/25_midi_input.py --mode drip --material ceramic
```

Each `note_on` triggers a drip event whose `radius_mm` follows the MIDI
note (each octave doubles the radius) and whose `velocity_factor` is set
by the MIDI velocity. The cache keeps latency under ~10 ms by re-using
pre-rendered wavs per `(note, velocity bin)`.

## Recipe 18: WebAudio demo — modal & drip in the browser

The `web/` folder ships a zero-dependency port of `synth_modal_impact`
and `synth_drip_event` to the Web Audio API. Two tabs, sliders, no
backend. Deploy by dragging the folder to Netlify Drop or by:

```bash
cd web && python3 -m http.server 8080
# open http://localhost:8080
```

Parity table (JS ↔ Python):

| Python | JS |
|---|---|
| `synth_modal_impact` | `renderModalImpact` (modal.js) |
| `synth_drip_event` | `renderDripEvent` (drip.js) |
| `MODAL_PROFILES` | `MODAL_PROFILES` (verbatim) |
| `SURFACE_PROFILES` | subset of 6 surfaces |

The JS port uses `BiquadFilterNode` (Q derived from t60 by
$Q \approx \pi \cdot f \cdot t_{60}$) and `OscillatorNode` chirps with
`exponentialRampToValueAtTime` for the Minnaert sweep — the result is
audibly close to the Python without any DSP code beyond what the browser
provides natively. Future steps: granular and reverb (Web Audio has
`ConvolverNode` for free).

## Recipe 19: bouncing rubber ball on metal plate

Rubber's low fundamental and short decay tame the brightness of a metal
surface — the ball is heard, the plate is felt.

```python
from impossible_mix.physics.composer import compose_impossible
wav = compose_impossible(
    base_material="rubber", base_interaction="roll",
    overlay_material="metal", overlay_interaction="impact",
    overlay_weight=0.30,
    modifiers=dict(rigidity=0.45, resonance=0.5, continuity=0.6),
    duration_s=4.0, seed=42,
)
```

## Recipe 20: ice cracking shards

Stochastic granular flow with the `ice_shards` profile, scattered over a
hard surface. Higher `density_hz` packs more cracks in the same window.

```python
from impossible_mix.physics.granular import GranularParams, synth_granular_flow
p = GranularParams(grain_profile="ice_shards", surface_profile="ice",
                    density_hz=70, density_jitter=0.5, cluster_factor=0.6,
                    energy_mean=0.7, duration_s=4.0, seed=42)
wav = synth_granular_flow(p, sr=44100)
```

## Recipe 21: plasma drop on a metal plate (impossible)

Uses the new `plasma_drop` droplet preset (high path roughness, metallic
surface contact). The drop rolls erratically and leaves a multi-modal
ringing on the surface.

```python
from impossible_mix.physics.droplet_presets import get_preset
from impossible_mix.physics.droplet import synth_rolling_droplet
p = get_preset("plasma_drop", duration_s=5.0, seed=42)
# Override surface to metal for a bright ring
p.surface_profile = "metal"
wav = synth_rolling_droplet(p, sr=44100)
```

## Recipe 22: squelchy mud splash

Splash whose drops use the `mud_drop` preset (very viscous, soft surface
hardness). The resulting splash is dense, low-pitched, and short — the
classic Foley "footstep in deep mud".

```python
from impossible_mix.physics.liquid import SplashParams, synth_splash
wav = synth_splash(
    SplashParams(intensity=0.8, n_bubbles=35,
                  bubble_size_mean_mm=4.0, viscosity=0.85,
                  spread_ms=70, duration_s=3.0, seed=42),
    sr=44100,
)
```

## How to extend the cookbook

Most recipes follow the same skeleton:

1. Pick a primitive (drip, modal, granular, friction, splash, pour).
2. Choose a surface or grain profile that matches the *material* on which
   the event happens.
3. Tune the high-level knobs (wetness, granularity, rigidity, resonance,
   continuity) on the controller, or call the primitive directly.
4. Optionally compose with an overlay (`compose_impossible`).
5. Optionally route through `place_source` and `apply_reverb` for stereo
   placement and ambience.

A future extension will provide a **differentiable** path through the same
operations (a DDSP-style implementation in PyTorch), enabling gradient-based
parameter inference from real recordings while preserving the interpretable
knobs documented here.
