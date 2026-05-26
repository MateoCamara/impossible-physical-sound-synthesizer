# MIDI input — play the physics engine live

`scripts/25_midi_input.py` lets you trigger drip events and modal impacts
from a MIDI keyboard in real time. Latency is kept low by caching pre-
rendered wavs per `(note, velocity bin)`.

## Install the optional dependencies

```bash
pip install -e ".[midi]"
```

This adds `mido`, `python-rtmidi` and `sounddevice` to your environment.

## Quick start

```bash
# 1. List available MIDI ports (connect your keyboard first, or launch
#    a virtual MIDI piano like VMPK).
python scripts/25_midi_input.py --list-ports

# 2. Play droplet events on a ceramic surface
python scripts/25_midi_input.py --mode drip --material ceramic --port "MyKeyboard"

# 3. Play a glass bell with modal impacts
python scripts/25_midi_input.py --mode modal --material glass --port "MyKeyboard"
```

If you don't pass `--port`, the first available port is used.

## Mapping

| MIDI note | Effect (drip mode) | Effect (modal mode) |
|---|---|---|
| 36 (C2) | small radius (~0.5 mm) | low-tuned modal |
| 48 (C3) | 1 mm radius | one octave below the profile fundamental |
| 60 (C4) | 2 mm radius | profile fundamental |
| 72 (C5) | 4 mm radius | one octave above the profile |

Velocity 1..127 maps to a continuous `velocity_factor` 0.3..2.0 that
modulates amplitude, brightness and (in modal mode) a small pitch
shift (~4 % per unit).

## Materials

**Drip mode** (`surface_profile`): `fabric, wood, ceramic, glass, metal, stone, water`.

**Modal mode** (`MaterialModalProfile` in `physics/modal.py`): `metal, rock, wood, glass, earth, fabric`.

## Troubleshooting

- **No ports listed** → your OS does not see a MIDI device. On Linux,
  `lsusb` or `aconnect -l` will show MIDI hardware; if you only have a
  software keyboard, install `vmpk` and start it.
- **High latency** → reduce `sounddevice` buffer size with
  `sd.default.latency = 'low'` at the top of the script.
- **Silence after a few notes** → check that the cache `(note, vbin)` is
  growing; if not, your audio backend may be muted.

## Cookbook recipe

```python
# Example: render and play a single note programmatically (no MIDI)
import sounddevice as sd
from impossible_mix.config import SAMPLE_RATE
from impossible_mix.physics.droplet import DropletParams, synth_drip_event
p = DropletParams(droplet_radius_mm=2.0, surface_profile="ceramic",
                   duration_s=0.6, seed=42)
wav = synth_drip_event(p, SAMPLE_RATE, velocity_factor=1.0)
sd.play(wav, SAMPLE_RATE, blocking=True)
```
