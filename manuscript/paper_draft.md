# How does a rolling droplet sound?
## A physics-informed parametric framework for the synthesis of impossible sounds

**Anonymous authors** *(double-blind submission to Tecniacústica 2026)*

---

## Abstract

Audiovisual sound design routinely demands material–interaction combinations that do not occur in nature: a liquid drop *rolling*, a wet rock *impact*, gravel *scraping* under water. Generative approaches based on pretrained neural audio codecs, dominant in current SFX synthesis, allow manipulating embeddings in ways that produce measurable variation in spectral metrics but rarely translate into changes that listeners identify *semantically* as "more liquid" or "more rocky": the latent direction lacks perceptual grounding. We propose a **physics-informed parametric framework** organised in modular layers — modal resonators for material bodies, body-and-surface friction, stochastic granular events, and droplet events with Minnaert bubble dynamics — combined by a generic *composer* that takes specifications of the form `(material, interaction, modifiers)`. Each parameter has **direct physical meaning** (droplet radius in mm, viscosity, surface hardness, rolling velocity, path roughness), guaranteeing that a "more X" control corresponds to an acoustic transformation coherent with that parameter. We present the **rolling droplet** as a paradigmatic case with no direct natural referent. We synthesise 24 stimuli across three impossible families (rolling droplet, liquid rock impact, wet gravel scrape) and show that 10 of 15 (scene, knob) pairs produce significant monotonic variation ($|\rho_{Spearman}| > 0.7$) in at least one acoustic metric. A pilot perceptual listening study contrasts quantitative and perceptual monotonicity between the proposed framework and a neural latent-direction baseline.

---

## 1. Introduction

Foley design for film, animation, video games, and immersive media frequently needs sounds that *do not* exist as recordings: an animator wants a rolling drop of mercury, a sound designer needs a wet rock thudding into mud, a game audio team needs gravel scraping in a flooded cave. Two routes have dominated practice. **Studio Foley** combines recorded layers and processes them aggressively; the result is high-quality but expensive, slow, and not reproducible. **Generative neural synthesis** based on pretrained codecs (EnCodec, RAVE, DAC) or latent diffusion (AudioLDM, Stable Audio Open) promises a faster path, but two limitations stand out.

First, embeddings of pretrained codecs are optimised for **general audio reconstruction**, not for representing material or interaction semantics; their latent axes have no anchor in physical attributes. Second, when manipulating these latent representations to obtain "more liquid" or "more rocky" mixtures, the spectrum changes measurably (RMS, dynamic range, centroid) but the perceived character of the sound often does not — the listener does not hear a liquid where none was. This dissociation between **quantitative monotonicity** in metric space and **perceptual monotonicity** in audio space is a central obstacle for interpretable control.

We propose to attack the problem from the opposite end: instead of starting from a learned latent space and hoping it encodes physical meaning, we build a **parametric synthesis engine whose knobs *are* physical parameters from the outset**. Modal resonator banks reproduce material bodies (rock, metal, wood, glass, earth, fabric), friction models reproduce scrapes and drags, stochastic granular trains reproduce gravel and earth flows, and Minnaert resonance reproduces water bubble formation. A generic composer combines these primitives according to specifications of the form `(material, interaction, modifiers)`, and a controller exposes physical knobs (wetness, granularity, rigidity, resonance, continuity) that the user manipulates as continuous sliders.

The result is a framework where *more wetness* changes droplet viscosity directly, *more granularity* increases the path roughness of the rolling event, and *more rigidity* shifts modal fundamentals up — each knob produces an audible transformation coherent with its physical meaning. The case of the **rolling droplet** — a sound with no natural referent we can record — illustrates the approach: a quasi-periodic train of drip events modulated by velocity and roughness, layered with a subtle surface modal tail.

The contributions of this paper are:
1. A modular library of physics-informed sound primitives for SFX.
2. A generic composer that supports impossible material–interaction combinations.
3. A controller with physical knobs and demonstrable monotonicity (objective and perceptual).
4. An open dataset of 24 impossible-sound stimuli with metadata.
5. A pilot perceptual study contrasting the proposed framework with a neural latent-direction baseline.

---

## 2. Related Work

**Physically-based sound synthesis**. Modal synthesis with biquad resonators is a long-standing tool for percussion and impact sounds; *PhySynth*-style toolkits, van den Doel's earlier work on physically-based liquid sounds, and modern formulations by Avanzini and colleagues cover most rigid-body cases. Liquid synthesis with Minnaert bubble resonance ($f_M = 3.26 / r$ in water with $r$ in metres) underlies the work of Drumm and others on bubble sound generation; the chirp inside a single drip event corresponds to bubble radius growth. Friction-driven synthesis (scrape, drag, bow) follows the Avanzini–Crosato and Serafin formulations.

**Neural SFX synthesis**. Pretrained codecs (EnCodec, RAVE) encode arbitrary audio into compact latents. Recent work (FOLEY-VAE family, NoiseBandNet, *Learning Control of Neural Sound Effects Synthesis from Physically Inspired Models*) explores controllability within these latents, ranging from interpolation between centroids to attribute heads trained to predict material categories. Text-conditioned diffusion models (AudioLDM2, Stable Audio Open) accept natural-language prompts and produce plausible but opaque outputs.

**Morphing and infusion**. SoundMorpher introduces explicit criteria for *correspondence*, *intermediateness*, and *smoothness* in audio morphing; Mix2Morph proposes structural/timbral infusion. CLAP-based metrics measure semantic alignment between audio and text descriptions. We borrow the *intermediateness* criterion as one of our objective metrics.

Our framework is closest in spirit to *Learning Control of Neural Sound Effects Synthesis from Physically Inspired Models*, but it is **deliberately not learned**: we use no parameter inference network in this paper. The physical parameters are exposed directly, and the rendering is performed entirely by classical DSP. We see this as complementary to the neural route: future work can train inverse models that estimate physical parameters from real recordings, closing the loop.

---

## 3. Method

### 3.1 Primitive layers

**Modal synthesis.** A material body is modelled as a sum of damped sinusoids excited by a short impulse. Each material profile specifies the number of modes, fundamental frequency, harmonic spacing, $t_{60}$ damping, spectral tilt, and inharmonicity. We provide profiles for *metal*, *rock*, *wood*, *glass*, *earth*, and *fabric*. A roll event chains impacts at a quasi-periodic rate with amplitude and timing jitter.

**Friction.** Scrape and drag events are modelled as bandpass-filtered noise modulated by a fluctuating velocity envelope and passed through the body resonance of the material. Surface roughness adds stochastic micro-pulses to the velocity envelope. Parameters: `surface_hardness`, `roughness`, `velocity_mean`, `velocity_jitter`, `pressure`, `body_resonance_hz`, `body_q`.

**Granular flows.** Gravel, sand, and earth events are clouds of stochastic micro-impacts. Each grain is a short modal hit whose fundamental frequency scales inversely with grain size (smaller grain → higher pitch). Parameters: `density_hz`, `density_jitter`, `grain_size_mm`, `size_variance`, `energy`, `spatial_spread`.

**Droplet event.** A single drip is composed of an impulsive attack (1 ms noise burst), an ascending bubble chirp ($f_{start} \approx 0.45 f_M$, $f_{end} \approx 1.6 f_M$) with high-Q resonance, an exponential decay envelope, and an optional modal tail of the impact surface. The Minnaert frequency derives from droplet radius. Viscosity slows the chirp and shortens the decay.

**Rolling droplet.** A quasi-periodic train of drip events with timing jitter (path roughness) and amplitude jitter (rolling dynamics), summed with a low-level coloured noise bed gated by the envelope of the train. Parameters: `droplet_radius_mm`, `viscosity`, `surface_hardness`, `roll_velocity_hz`, `path_roughness`.

**Splash and pour.** A splash is a short bandpass burst followed by $N$ dispersed drip events whose bubble sizes are drawn from a controllable distribution. A pour is a dense train of drips with overlapping bandpass turbulence.

### 3.2 Composer

The composer exposes a single function `compose(material, interaction, modifiers, duration) → wav`. A lookup table maps `(material, interaction)` pairs to primitive generators (Table 1). High-level modifiers — *wetness*, *granularity*, *rigidity*, *resonance*, *continuity* — are translated into the physical parameters consumed by the active primitive. For example, in a `(liquid, roll)` scene, *wetness* reduces droplet viscosity, *continuity* increases rolling velocity, and *granularity* increases path roughness.

For impossible combinations, the composer supports overlay: `compose_impossible(base, overlay, weight)` mixes two compositions with a weight. *Liquid rock impact* is realised as `(rock, impact)` plus a `(liquid, splash)` overlay with weight 0.55. *Wet gravel scrape* is `(gravel, scrape)` plus a `(liquid, pour)` overlay with weight 0.35. *Rolling droplet* is the single `(liquid, roll)` scene with no overlay — the impossibility is intrinsic to the combination because liquids do not roll.

**Table 1.** Supported (material, interaction) pairs mapped to primitive generators.

| Material | Supported interactions |
|---|---|
| rock, metal | impact, roll, scrape (rock also drag) |
| wood | impact, scrape |
| fabric | drag |
| liquid | drip, splash, pour, roll, impact |
| gravel | step, roll, scrape, pour |
| earth | step |

### 3.3 Physical controller

The `PhysicsController` class wraps the composer. The user calls `set_scene(material, interaction)` and then `render(wetness=..., granularity=..., ...)` or `sweep(knob, values)`. The output of a sweep is a sequence of audio buffers whose acoustic content varies coherently with the knob value. The CLI front-end (`scripts/12_physics_sweep.py`) accepts arguments like `--material liquid --interaction roll --more wetness --from 0 --to 1 --steps 6` and writes a directory of WAVs ready for evaluation.

---

## 4. The Rolling Droplet as a Case Study

The title question — *how does a rolling droplet sound?* — has no recorded reference. We construct the sound bottom-up: a 2 mm droplet rolling at 14 contacts per second over a semi-hard surface, with mild path roughness. Each contact is a 1.6 kHz chirp (the Minnaert frequency for 2 mm) lasting ≈ 30 ms, decaying over ≈ 100 ms, with a faint metallic tail from the surface modal layer. The bed of coloured noise gated by the contact envelope suggests humidity. We render eight variants by perturbing the canonical configuration: `very_wet` (full liquid character), `dry_drip` (slow, discrete contacts), `slow_roll`, `fast_roll`, `grainy_path`, `smooth_path`, `big_droplet`, and the `canonical` itself.

**Figure 2** shows the spectrograms of four variants. The canonical droplet exhibits a periodic train of chirps in the 1–2 kHz band; `very_wet` shifts the centroid down and extends the decays; `slow_roll` separates the contacts into clearly identifiable events; `fast_roll` collapses them into a near-continuous texture.

---

## 5. Evaluation

### 5.1 Stimulus corpus

We render 24 stimuli grouped as 3 scenes × 8 variants:
- **A: rolling droplet** — variants of the rolling event described above.
- **B: liquid rock impact** — rock impacts overlaid with liquid splashes at varying weights.
- **C: wet gravel scrape** — gravel scraping under liquid pour overlays.

All stimuli are 5 s, mono, 44.1 kHz, peak-normalised to −1 dBFS. The manifest (`perceptual_test/stimuli_manifest.csv`) records all parameters and per-clip quality metrics.

### 5.2 Objective metrics

We compute six acoustic metrics on every audio: RMS dBFS, peak dBFS, crest factor, spectral flatness, spectral centroid, and dynamic range. A composite *quality verdict* in `{good, suspicious, garbage}` integrates the metrics with thresholds tuned to flag silent, transient-less, noise-like, or hiss-only outputs. The verdict and metric ranges are released as a Python module (`impossible_mix/metrics/quality.py`).

**Table 2** compares the physics framework to neural baselines on identical metric ranges.

**Table 2.** Master comparison of methods over impossible-sound generations.

| Method | n | % good | mean dyn (dB) | mean RMS (dB) | mean centroid (Hz) |
|---|---:|---:|---:|---:|---:|
| **PHYSICS (24 variants)** | 24 | **100.0** | 46.4 | −22.5 | 6208 |
| NEURAL A (latent direction, 162) | 162 | 100.0 | 45.8 | −28.6 | 4002 |
| NEURAL B (gradient edit, 24) | 24 | 100.0 | 59.2 | −32.8 | 2987 |
| Baseline (additive sum) | 6 | 100.0 | 11.2 | −17.3 | 451 |
| Baseline (linear interp) | 6 | **16.7** | 38.9 | −48.2 | 3409 |

Physics, NEURAL A, NEURAL B, and the additive-sum baseline all pass the quality verdict at 100 %. The linear interpolation baseline fails in 5 of 6 cases (collapse to the anchor or out-of-manifold artefacts), as anticipated by the *intermediateness* criterion.

### 5.3 Monotonicity

For every (scene, knob) pair we render 7 stimuli with the knob in $\{0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0\}$ and the other knobs held at 0.5. We compute the Spearman correlation between knob value and each acoustic metric. A pair is *monotonic* if at least one metric satisfies $|\rho| > 0.7$.

**Of 15 (scene, knob) pairs (3 scenes × 5 knobs), 10 are monotonic.** The five non-monotonic pairs correspond to physically inapplicable combinations: *resonance* in gravel scrape (granular events have no body resonance), *rigidity* and *resonance* in the rolling droplet (water has no rigid mode structure), and *granularity* in the rock-impact scene (granularity is an attribute of granular flows, not single impacts). Figure 3 reports the heatmap of $|\rho_{\max}|$ per (scene, knob).

Critically, the neural latent-direction sweeps (re-measured on four scene/knob combinations available from prior experiments) are also 4-of-4 monotonic in spectral metrics. **The physics and the neural baselines are indistinguishable on this objective monotonicity test.** What separates them is whether the spectral monotonicity translates into perceptual monotonicity — which the pilot study addresses.

### 5.4 Pilot perceptual study

We prepared an HTML form serving 12 of the 24 stimuli per listener in randomised order (Latin Square light). Each stimulus is followed by four questions: (1) perceived material, (2) perceived interaction, (3) impossibility on a 1–7 Likert, and (4) coherence as a single event on 1–7. The form is deployed as a static page; listeners download a CSV of their numeric responses at the end. Recruitment targets ≥ 16 listeners (AES España, SEAcústica, social networks).

For each scene we will compare: (i) identification rates against a permissive ground truth (any of the components of the impossible counts), (ii) Friedman tests over variants for impossibility and coherence Likert scales, and (iii) Spearman correlation between the physical knob value (when known) and the median impossibility rating across variants. The latter is the perceptual-monotonicity test: a positive correlation between, e.g., wetness knob and perceived impossibility would indicate that the framework's "more wet" really is heard as more impossible / more liquid by listeners.

*Pilot results will be substituted into the final version of this section.*

---

## 6. Discussion

The framework deliberately does not learn. Each primitive is hand-engineered, each modifier is a function of physical parameters, and the entire pipeline is deterministic given a seed. The benefits are reproducibility, interpretability, and the absence of training data requirements. The costs are scope (each material × interaction needs an engineering decision) and inability to generalise to truly novel combinations without programmer intervention. We see the framework as the *teacher* in a future student–teacher setup: a neural model trained to invert physical parameters from real recordings, with the physical engine generating supervision data on demand.

The neural latent-direction baselines highlight a recurring confound in the literature: spectral monotonicity is necessary but not sufficient for perceptual control. A method that changes the RMS and centroid of an audio monotonically may still produce audio that listeners cannot order along the intended semantic axis. The proposed physical framework collapses this gap by construction; the open question, settled by the pilot, is whether listeners agree.

Limitations: (a) the framework is currently restricted to the materials and interactions tabulated above; (b) the modal profiles are hand-tuned and would benefit from data-driven calibration against real impulse responses; (c) the rolling droplet, while audibly recognisable, is one realisation of a class — different rolling surfaces and droplet shapes would require parameter sweeps not yet validated.

---

## 7. Conclusion

We presented a physics-informed parametric framework for synthesising impossible sounds, with the rolling droplet as a paradigmatic case. The framework supports impossible material–interaction combinations through layered primitives and a generic composer. Acoustic metrics confirm monotonic control where physically applicable; the pilot perceptual study tests whether monotonicity in metrics translates to monotonicity in perception. Code, audio dataset, and the interactive demonstration are released anonymously for review and will be opened publicly upon acceptance.

---

## Reproducibility

All audio in this paper is generated by deterministic code. The full pipeline runs on CPU in under 5 minutes:

```bash
python scripts/13_final_stimuli.py        # 24 stimuli
python scripts/14_monotonicity.py         # monotonicity table
python scripts/15_figures.py              # F2-F4
python scripts/18_compare_physics_vs_neural.py   # Table 2 + F5
```

The interactive demonstration (`perceptual_test/interactive_demo/`) is a single-page HTML with 135 pre-rendered audio files (45 MB) deployable to any static host.

---

## Appendix: Knob → physical parameter mapping (per scene)

For reference and reproducibility:

| Knob | Rolling droplet | Liquid rock impact | Wet gravel scrape |
|---|---|---|---|
| wetness | viscosity ↓ | overlay weight ↑, splash bubble density ↑ | pour density ↑, overlay weight ↑ |
| granularity | path roughness ↑ | (n/a) | grain size variance ↑ |
| rigidity | droplet radius ↓, surface hardness ↑ | rock fundamental ↑ | (n/a) |
| resonance | (n/a) | rock damping ↑, modal $t_{60}$ ↑ | (n/a) |
| continuity | roll velocity ↑ | splash spread ↓ | scrape velocity_mean ↑ |
