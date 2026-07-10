# How does a rolling droplet sound?

Physics-informed parametric framework for the synthesis of impossible sounds
(material × interaction combinations that do not occur in nature: rolling
droplets, wet rock impacts, gravel scrapes under water).

Submitted to **Tecniacústica 2026** (double-blind).

## Why "physics-informed"?

Generative SFX based on pretrained neural codecs (EnCodec, RAVE) produces
spectral variation when its latent is perturbed, but the changes rarely
translate into perceptually identifiable shifts ("more liquid", "more
rocky"): the latent direction has no anchor in physical attributes.

This framework takes the opposite route. Each primitive layer has a
**direct physical interpretation** (modal resonator banks for material
bodies, Minnaert resonance for bubbles, stochastic granular events for
gravel, friction models for scrapes). Each knob ("wetness", "granularity",
"rigidity", "resonance", "continuity") maps to a physical parameter such
as droplet radius, viscosity, surface hardness, rolling velocity, or path
roughness. A "more X" control therefore produces an acoustic
transformation **coherent with the named property**.

## Repository layout

```
impossible_mix/
├── physics/                    Physics-informed synthesis engine
│   ├── modal.py                Modal resonators (rock, metal, wood, glass...)
│   ├── friction.py             Scrape/drag via velocity-modulated noise + resonance
│   ├── granular.py             Gravel/earth via stochastic micro-impact clouds
│   ├── liquid.py               Splash, pour
│   ├── droplet.py              Rolling droplet (Minnaert-based bubble chirps)
│   └── composer.py             Generic composer; supports impossible overlays
├── physics_controller.py       High-level controller with physical knobs
├── metrics/quality.py          Cheap acoustic verdict (good / suspicious / garbage)
├── methods/                    Legacy neural baselines (A, B, D, baselines)
├── encoders/                   EnCodec wrapper used by neural baselines
└── data/                       Corpus loaders and labels schema

scripts/
├── 00_parse_ulfc.py            Parse ULTIMATE-FOOTSTEP-COLLECTION (baseline corpus)
├── 01_freesound_download.py    Download Freesound previews to fill gaps
├── 02_prepare_corpus.py        Resample/normalize to 44.1 kHz mono 5 s
├── 03_extract_embeddings.py    Cache EnCodec embeddings (neural baseline)
├── 04_merge_labels.py          Merge corpus into single labels.csv
├── 05_train_heads.py           Train Method-B attribute heads (neural baseline)
├── 06_generate_method_a.py     Neural baseline A
├── 07_evaluate_intermediateness.py
├── 08_generate_method_b.py     Neural baseline B
├── 09_apply_method_d.py        DSP postproc baseline D
├── 10_quality_check.py         Batch verdict over outputs/
├── 11_sweep.py                 Neural latent-direction sweep (controller)
├── 12_physics_sweep.py         PHYSICS sweep CLI
├── 13_final_stimuli.py         Render 24 stimuli for perceptual test
├── 14_monotonicity.py          Spearman test per (scene, knob, metric)
├── 15_figures.py               Figs 2, 3, 4 (spectrograms, heatmap, curves)
├── 16_build_perceptual_form.py Self-contained HTML perceptual test
├── 17_analyze_perceptual.py    Analyze responses CSV
├── 18_compare_physics_vs_neural.py    Master table + Fig 5
├── 19_neural_monotonicity.py   Monotonicity of neural sweeps
├── 20_build_interactive_demo.py    Interactive demo with sliders (135 audios)
├── 21_extra_impossibles.py     Extra impossible combos (lava_footstep, boiling_water...)
├── 22_gradio_app.py            Interactive Gradio demo (rolling droplet, impossible scenes, evolving knobs, exotic sounds)
├── 24_download_irs.py          Download real CC-licensed IRs (OpenAIR), with synthetic fallback
├── 25_midi_input.py            Play the physics engine live from a MIDI keyboard
├── 26_inverse_drip_fitting.py          Inverse fitting: drip event params via differentiable DDSP
├── 27_inverse_modal_fitting.py         Inverse fitting: modal impact (K resonant modes) via DDSP
├── 28_inverse_granular_fitting.py      Inverse fitting: granular flow profile via DDSP
├── 29_inverse_reverb_fitting.py        Inverse fitting: recover a room IR from dry/wet pair via DDSP
├── 30_inverse_friction_fitting.py      Inverse fitting: friction/scrape params via DDSP
└── 31_inverse_rolling_droplet_fitting.py   Inverse fitting: rolling droplet params via DDSP

Scripts 26-31 (inverse fitting) persist each run's recovered vs.
ground-truth params, errors and loss history to
`results/diff_fits/<engine>/params.json` via
`impossible_mix/utils/fit_io.py::save_fit_report`, instead of leaving
the numbers only in stdout.

perceptual_test/
├── stimuli/                    24 final stimuli (3 combos × 8 variants)
├── stimuli_manifest.csv        Metadata + per-clip metrics
├── form/                       Self-contained perceptual test (HTML+JS)
├── interactive_demo/           Interactive sliders demo (HTML+JS+135 audios)
└── responses/                  CSVs from listeners (collected here)

results/
├── monotonicity/               Spearman tables and summary
├── comparison/                 Master comparison (physics vs neural baselines)
├── method_a/                   Neural baseline outputs and metrics
├── method_b/                   Neural baseline outputs and metrics
└── quality_report.csv          Verdict for all generated audio

figures/                        F2-F5 PNGs for the paper

manuscript/
├── paper_draft.md              Full draft (≈6 pages)
├── abstract_v1.md ... v3.md    Abstract iterations
└── paper_outline.md            Section-by-section outline
```

## Quickstart

### 1. Install

```bash
# Requires Python 3.10+ and uv (https://docs.astral.sh/uv/)
uv venv
uv pip install --python .venv/bin/python -e ".[clap,encodec,fad,stats,notebooks]"
cp .env.example .env  # add FREESOUND_API_KEY if running the data pipeline
```

### 2. Render the 24 paper stimuli

```bash
.venv/bin/python scripts/13_final_stimuli.py
```
Produces `perceptual_test/stimuli/*.wav` and `stimuli_manifest.csv`.

### 3. Reproduce the paper tables and figures

```bash
.venv/bin/python scripts/14_monotonicity.py        # Table monotonicity + CSV summary
.venv/bin/python scripts/15_figures.py             # F2, F3, F4
.venv/bin/python scripts/18_compare_physics_vs_neural.py  # Master table + F5
```

### 4. Try the interactive demo

```bash
cd perceptual_test/interactive_demo
python3 -m http.server 8765
# open http://localhost:8765
```

### 5. Use the controller from Python

```python
from impossible_mix.physics_controller import PhysicsController

ctrl = PhysicsController(seed=42, duration_s=5.0)

# Generic API: any (material, interaction) supported by the composer
ctrl.set_scene(material="liquid", interaction="roll")
wav = ctrl.render(wetness=0.8, granularity=0.3, continuity=0.7)

# Impossible combos: base + overlay
ctrl.set_scene(material="rock", interaction="impact",
               overlay_material="liquid", overlay_interaction="splash",
               overlay_weight=0.55)
wav = ctrl.render(wetness=0.9, resonance=0.3)

# Sweep a knob
steps = ctrl.sweep("wetness", [0.0, 0.25, 0.5, 0.75, 1.0])
```

### 6. Use the CLI

```bash
# List available knobs / scenes
.venv/bin/python scripts/12_physics_sweep.py --list-scenes
.venv/bin/python scripts/12_physics_sweep.py --list-knobs

# Sweep wetness on the rolling droplet
.venv/bin/python scripts/12_physics_sweep.py \
  --material liquid --interaction roll \
  --more wetness --from 0 --to 1 --steps 7

# Sweep with an overlay
.venv/bin/python scripts/12_physics_sweep.py \
  --material rock --interaction impact \
  --overlay-material liquid --overlay-interaction splash --overlay-weight 0.55 \
  --more wetness --from 0 --to 1 --steps 7
```

## Re-running the neural baseline (optional)

The neural baselines (A, B) require the EnCodec embeddings on a curated
corpus. The data pipeline is included for full reproducibility but is
*not* needed to reproduce the physics-only results of the paper.

```bash
.venv/bin/python scripts/01_freesound_download.py   # ~5 min, needs API key
.venv/bin/python scripts/02_prepare_corpus.py       # ~30 s
.venv/bin/python scripts/03_extract_embeddings.py   # ~27 min on CPU
.venv/bin/python scripts/04_merge_labels.py
.venv/bin/python scripts/05_train_heads.py          # ~6 min for 200 epochs
.venv/bin/python scripts/06_generate_method_a.py    # ~5 min for full grid
.venv/bin/python scripts/08_generate_method_b.py
.venv/bin/python scripts/19_neural_monotonicity.py
```

## Conducting the perceptual study

1. Build the form: `.venv/bin/python scripts/16_build_perceptual_form.py`
2. Deploy `perceptual_test/form/` to any static host (Netlify Drop, GitHub Pages, S3).
3. Listeners fill it and download a CSV with their numeric answers.
4. Collect CSVs in `perceptual_test/responses/` and run:

```bash
.venv/bin/python scripts/17_analyze_perceptual.py
```

Produces summary, identification rates, Friedman tests, and a boxplot.

## Hardware / compute notes

The full pipeline runs on **CPU** in under 10 min (excluding optional
neural baseline training). No GPU is required. The physics engine uses
NumPy + SciPy only.

The interactive demo is 45 MB of pre-rendered WAVs + HTML/JS; no server,
no backend.

## Datasets used

- **Ultimate Footstep Collection** (locally `ULTIMATE-FOOTSTEP-COLLECTION/`):
  used to seed the neural baseline corpus.
- **Freesound API**: used to fill the gaps (liquid, gravel, wood, fabric
  events not present in the footstep collection). CC0 / CC-BY only.

The physics framework itself does not depend on any dataset.

## License

Code: MIT. Data: see per-clip license in `data/raw/freesound/manifest.json`.

## Citation

```bibtex
@inproceedings{rolling_droplet_2026,
  title={{How does a rolling droplet sound?} A physics-informed parametric framework for the synthesis of impossible sounds},
  author={Anonymous},
  booktitle={Proceedings of Tecniac\'ustica 2026},
  year={2026},
  note={Double-blind submission}
}
```
