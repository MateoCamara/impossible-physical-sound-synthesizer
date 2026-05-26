"""Demo interactiva Gradio con motor fisico en backend.

Cuatro pestanas:
  1. Rolling droplet — knobs en vivo, con presets de gotas (water, mercury, lava, oil...)
  2. Impossible scenes — combos base x overlay
  3. Evolving — knobs como curvas temporales (rampas, LFO)
  4. Exotic — rain, fire, thunder, glass_break, ocean_wave

Run:
    .venv/bin/python scripts/22_gradio_app.py
    -> http://localhost:7860
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impossible_mix.config import SAMPLE_RATE
from impossible_mix.physics_controller import KNOBS, PhysicsController
from impossible_mix.physics.droplet import synth_rolling_droplet
from impossible_mix.physics.droplet_presets import PRESETS, get_preset
from impossible_mix.physics.modulation import KnobCurve, compose_evolving
from impossible_mix.physics.exotic import (
    synth_rain, synth_fire, synth_thunder,
    synth_glass_break, synth_ocean_wave,
)
from impossible_mix.physics.sequences import (
    droplet_story, mercury_drama, lava_step_into_water, parse_dsl,
)
from impossible_mix.physics.spatial import place_source
from impossible_mix.physics.reverb import generate_ir, apply_reverb, PRESETS as IR_PRESETS


def to_gradio_audio(wav: np.ndarray, sr: int = SAMPLE_RATE):
    """Gradio Audio espera (sr, np.array int16 o float32)."""
    return (sr, wav.astype(np.float32))


# ====================================================================
# Tab 1: Rolling droplet with physical knobs + preset selector
# ====================================================================
def render_droplet(preset: str, radius_mm: float, viscosity: float,
                   surface_hardness: float, roll_velocity_hz: float,
                   path_roughness: float, duration_s: float, seed: int):
    # Si preset es "custom", usar los sliders. Si no, usar los del preset.
    if preset != "custom":
        p = get_preset(preset, duration_s=duration_s, seed=seed)
    else:
        from impossible_mix.physics.droplet import DropletParams
        p = DropletParams(droplet_radius_mm=radius_mm, viscosity=viscosity,
                          surface_hardness=surface_hardness,
                          roll_velocity_hz=roll_velocity_hz,
                          path_roughness=path_roughness,
                          duration_s=duration_s, seed=seed)
    wav = synth_rolling_droplet(p, SAMPLE_RATE)
    return to_gradio_audio(wav)


# ====================================================================
# Tab 2: Impossible scenes (compose_impossible)
# ====================================================================
def render_impossible(scene: str, wetness: float, granularity: float,
                       rigidity: float, resonance: float, continuity: float,
                       overlay_weight: float, duration_s: float, seed: int):
    ctrl = PhysicsController(seed=seed, duration_s=duration_s)
    if scene == "rolling_droplet":
        ctrl.set_scene(material="liquid", interaction="roll")
    elif scene == "liquid_rock_impact":
        ctrl.set_scene(material="rock", interaction="impact",
                       overlay_material="liquid", overlay_interaction="splash",
                       overlay_weight=overlay_weight)
    elif scene == "wet_gravel_scrape":
        ctrl.set_scene(material="gravel", interaction="scrape",
                       overlay_material="liquid", overlay_interaction="pour",
                       overlay_weight=overlay_weight)
    elif scene == "muddy_metal_roll":
        ctrl.set_scene(material="metal", interaction="roll",
                       overlay_material="liquid", overlay_interaction="pour",
                       overlay_weight=overlay_weight)
    elif scene == "soggy_wood_impact":
        ctrl.set_scene(material="wood", interaction="impact",
                       overlay_material="liquid", overlay_interaction="splash",
                       overlay_weight=overlay_weight)
    else:
        ctrl.set_scene(material="rock", interaction="impact")
    wav = ctrl.render(wetness=wetness, granularity=granularity,
                       rigidity=rigidity, resonance=resonance, continuity=continuity)
    return to_gradio_audio(wav)


# ====================================================================
# Tab 3: Evolving (knob curves)
# ====================================================================
def render_evolving(scene: str,
                    knob_to_modulate: str,
                    curve_type: str,
                    curve_start: float, curve_end: float,
                    lfo_freq: float,
                    other_value: float,
                    duration_s: float, seed: int):
    ctrl = PhysicsController(seed=seed, duration_s=duration_s)
    if scene == "rolling_droplet":
        ctrl.set_scene(material="liquid", interaction="roll")
    elif scene == "wet_gravel_scrape":
        ctrl.set_scene(material="gravel", interaction="scrape",
                       overlay_material="liquid", overlay_interaction="pour",
                       overlay_weight=0.35)
    else:
        ctrl.set_scene(material="rock", interaction="impact",
                       overlay_material="liquid", overlay_interaction="splash",
                       overlay_weight=0.55)

    if curve_type == "ramp_up":
        curve = KnobCurve.ramp(curve_start, curve_end)
    elif curve_type == "ramp_down":
        curve = KnobCurve.ramp(curve_end, curve_start)
    elif curve_type == "exp_decay":
        curve = KnobCurve.exp_decay(curve_end, tau_norm=0.3, floor=curve_start)
    elif curve_type == "lfo_sine":
        curve = KnobCurve.lfo(lfo_freq, lo=curve_start, hi=curve_end)
    else:
        curve = KnobCurve.constant((curve_start + curve_end) / 2)

    # otros knobs constantes
    curves = {k: KnobCurve.constant(other_value) for k in KNOBS}
    curves[knob_to_modulate] = curve
    wav = compose_evolving(ctrl, curves, duration_s=duration_s, sr=SAMPLE_RATE)
    return to_gradio_audio(wav)


# ====================================================================
# Tab 4: Exotic
# ====================================================================
def render_exotic(kind: str, p1: float, p2: float, duration_s: float, seed: int):
    if kind == "rain":
        wav = synth_rain(duration_s=duration_s, intensity=p1, drop_size_mm=p2, seed=seed)
    elif kind == "fire":
        wav = synth_fire(duration_s=duration_s, intensity=p1, crackle_density=p2, seed=seed)
    elif kind == "thunder":
        wav = synth_thunder(duration_s=duration_s, distance=p1, intensity=p2, seed=seed)
    elif kind == "glass_break":
        wav = synth_glass_break(duration_s=duration_s, n_shards=int(p1 * 60), seed=seed)
    elif kind == "ocean_wave":
        wav = synth_ocean_wave(duration_s=duration_s, breaking_intensity=p1, seed=seed)
    else:
        wav = np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)
    return to_gradio_audio(wav)


# ====================================================================
# UI
# ====================================================================
PRESET_NAMES = ["custom"] + list(PRESETS.keys())

with gr.Blocks(title="How does a rolling droplet sound?") as app:
    gr.Markdown(
        "# How does a rolling droplet sound? \n"
        "Interactive demo of a **physics-informed parametric synthesis framework** "
        "for impossible sounds. Move the sliders — each parameter has direct "
        "physical meaning. Audio is rendered live on CPU."
    )

    # ------- Tab 1: Rolling droplet --------
    with gr.Tab("Rolling droplet"):
        gr.Markdown(
            "**The starring case.** A liquid droplet rolling, modelled as a "
            "quasi-periodic train of bubble-formation events (Minnaert resonance) "
            "over a surface modal layer. Try the preset materials or design your own."
        )
        with gr.Row():
            preset = gr.Dropdown(PRESET_NAMES, value="water", label="Preset")
            duration_d = gr.Slider(2.0, 10.0, value=5.0, step=0.5, label="Duration (s)")
            seed_d = gr.Number(value=42, label="Seed", precision=0)
        with gr.Row():
            radius = gr.Slider(0.3, 6.0, value=2.0, step=0.1, label="Droplet radius (mm)")
            viscosity = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Viscosity (0=water, 1=honey)")
            surface_hard = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Surface hardness")
        with gr.Row():
            velocity = gr.Slider(2.0, 40.0, value=14.0, step=1.0, label="Roll velocity (contacts/s)")
            rough = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Path roughness")
        btn_d = gr.Button("Render droplet", variant="primary")
        audio_d = gr.Audio(label="Output", autoplay=True)
        gr.Markdown("**Try**: switch the preset to *mercury* (impossible — dense metallic small droplet) "
                    "or *lava* (impossible — viscous deep bubble). The same engine covers all cases.")

        btn_d.click(render_droplet,
                    inputs=[preset, radius, viscosity, surface_hard, velocity, rough, duration_d, seed_d],
                    outputs=audio_d)

    # ------- Tab 2: Impossible scenes --------
    with gr.Tab("Impossible scenes"):
        gr.Markdown(
            "**Generic composer.** Mix any (material × interaction) plus an optional "
            "liquid overlay to create impossible sounds. The five physical knobs map "
            "to the active primitives of each scene."
        )
        with gr.Row():
            scene = gr.Dropdown(
                ["rolling_droplet", "liquid_rock_impact", "wet_gravel_scrape",
                 "muddy_metal_roll", "soggy_wood_impact"],
                value="liquid_rock_impact", label="Scene"
            )
            duration_i = gr.Slider(2.0, 10.0, value=5.0, step=0.5, label="Duration (s)")
            seed_i = gr.Number(value=42, label="Seed", precision=0)
            overlay_w = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="Overlay weight")
        with gr.Row():
            wet_i = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Wetness")
            gran_i = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Granularity")
            rig_i = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Rigidity")
        with gr.Row():
            res_i = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Resonance")
            cont_i = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Continuity")
        btn_i = gr.Button("Render impossible scene", variant="primary")
        audio_i = gr.Audio(label="Output", autoplay=True)

        btn_i.click(render_impossible,
                    inputs=[scene, wet_i, gran_i, rig_i, res_i, cont_i,
                            overlay_w, duration_i, seed_i],
                    outputs=audio_i)

    # ------- Tab 3: Evolving --------
    with gr.Tab("Evolving (knob curves)"):
        gr.Markdown(
            "**Time-varying control.** A knob can follow a curve through the clip, "
            "producing evolving sounds: a droplet that gradually gets wetter, a rolling "
            "event whose surface oscillates, a scrape that decays. Rendered with "
            "overlap-add over 0.6 s windows."
        )
        with gr.Row():
            scene_e = gr.Dropdown(
                ["rolling_droplet", "liquid_rock_impact", "wet_gravel_scrape"],
                value="rolling_droplet", label="Scene"
            )
            knob_e = gr.Dropdown(list(KNOBS), value="wetness", label="Knob to modulate")
            curve_e = gr.Dropdown(
                ["ramp_up", "ramp_down", "exp_decay", "lfo_sine"],
                value="ramp_up", label="Curve type"
            )
        with gr.Row():
            cstart = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Curve start / min")
            cend = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Curve end / max")
            lfo_f = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="LFO cycles/clip (sine only)")
            other_v = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Other knobs constant at")
        with gr.Row():
            duration_e = gr.Slider(4.0, 12.0, value=8.0, step=0.5, label="Duration (s)")
            seed_e = gr.Number(value=42, label="Seed", precision=0)
        btn_e = gr.Button("Render evolving", variant="primary")
        audio_e = gr.Audio(label="Output", autoplay=True)
        gr.Markdown(
            "**Cinematic examples**: rolling_droplet × wetness ramp_up (drying surface), "
            "wet_gravel_scrape × continuity ramp_down (gradual stop), "
            "liquid_rock_impact × resonance lfo_sine (oscillating room)."
        )

        btn_e.click(render_evolving,
                    inputs=[scene_e, knob_e, curve_e, cstart, cend, lfo_f, other_v,
                            duration_e, seed_e],
                    outputs=audio_e)

    # ------- Tab 4: Exotic --------
    with gr.Tab("Exotic sounds"):
        gr.Markdown(
            "**Extended physics primitives.** Same engine reused for natural complex events: "
            "rain (drip clouds), fire (filtered noise + crackles), thunder (modal rumble), "
            "glass break (high-mode cascade), ocean wave (3-phase envelope)."
        )
        with gr.Row():
            kind = gr.Dropdown(
                ["rain", "fire", "thunder", "glass_break", "ocean_wave"],
                value="rain", label="Kind"
            )
            duration_x = gr.Slider(2.0, 10.0, value=5.0, step=0.5, label="Duration (s)")
            seed_x = gr.Number(value=42, label="Seed", precision=0)
        with gr.Row():
            p1 = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                           label="P1 (rain:intensity / fire:intensity / thunder:distance / glass:shards / ocean:break_intensity)")
            p2 = gr.Slider(0.0, 5.0, value=1.2, step=0.1,
                           label="P2 (rain:drop_mm / fire:crackle / thunder:intensity / unused for glass+ocean)")
        btn_x = gr.Button("Render exotic", variant="primary")
        audio_x = gr.Audio(label="Output", autoplay=True)

        btn_x.click(render_exotic,
                    inputs=[kind, p1, p2, duration_x, seed_x],
                    outputs=audio_x)

    # ------- Tab 5: Cinematic --------
    with gr.Tab("Cinematic (sequences + space + reverb)"):
        gr.Markdown(
            "**Full cinematic pipeline.** Choose a pre-composed event sequence "
            "(a droplet that falls, rolls, and splashes), place it in space "
            "(distance + stereo pan), and convolve it with a synthetic impulse "
            "response (small room ... cathedral). The combination produces "
            "stereo, immersive impossible-sound clips ready for film/games."
        )
        with gr.Row():
            story = gr.Dropdown(
                ["droplet_story", "mercury_drama", "lava_step_into_water"],
                value="droplet_story", label="Cinematic sequence"
            )
            duration_c = gr.Slider(5.0, 12.0, value=8.0, step=0.5, label="Duration (s)")
            seed_c = gr.Number(value=42, label="Seed", precision=0)
        with gr.Row():
            distance_c = gr.Slider(0.5, 50.0, value=4.0, step=0.5, label="Distance (m)")
            pan_c = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="Pan (-1=L, +1=R)")
            with_delay = gr.Checkbox(value=False, label="Include propagation delay")
        with gr.Row():
            ir_preset = gr.Dropdown(list(IR_PRESETS.keys()), value="medium_hall", label="Space / IR")
            reverb_mix = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Reverb mix")
        btn_c = gr.Button("Render cinematic", variant="primary")
        audio_c = gr.Audio(label="Stereo output", autoplay=True)
        gr.Markdown(
            "**Examples**: *droplet_story* + *cathedral* + distance 4m → as if a droplet fell in a chapel. "
            "*mercury_drama* + *cave* + pan −0.5 → metallic chase in a tunnel. "
            "*lava_step_into_water* + *exterior* → outdoor field recording feel."
        )

        def render_cinematic(story_name, duration, seed, distance, pan, with_delay_flag, ir_name, mix):
            if story_name == "droplet_story":
                wav = droplet_story(duration_s=duration, seed=int(seed))
            elif story_name == "mercury_drama":
                wav = mercury_drama(duration_s=duration, seed=int(seed))
            else:
                wav = lava_step_into_water(duration_s=duration, seed=int(seed))
            stereo = place_source(wav, SAMPLE_RATE, distance_m=distance, pan=pan,
                                   with_delay=with_delay_flag)
            ir = generate_ir(ir_name, sr=SAMPLE_RATE, seed=int(seed))
            wet_stereo = apply_reverb(stereo, ir, mix=mix)
            return (SAMPLE_RATE, wet_stereo.astype(np.float32))

        btn_c.click(render_cinematic,
                    inputs=[story, duration_c, seed_c, distance_c, pan_c, with_delay,
                            ir_preset, reverb_mix],
                    outputs=audio_c)

    # ------- Tab 6: DSL --------
    with gr.Tab("DSL (text recipes)"):
        gr.Markdown(
            "**Text-driven event sequencing.** Write your scene as a small "
            "recipe and the engine assembles it. Each line is one event: "
            "`<func>(arg=value, ...) @ <time>s [gain=... pan=... distance_m=...]`. "
            "Functions: `drip, roll, splash, impact, pour`. Lines starting with "
            "`#` are comments."
        )
        default_recipe = """\
# Droplet sonata in 5 lines
drip(radius_mm=1.5) @ 0.3s gain=0.8 pan=-0.6
drip(radius_mm=2.0) @ 0.7s gain=0.9 pan=-0.2
roll(duration_s=2.5, roll_velocity_hz=14) @ 1.2s pan=0.0 distance_m=1.0
splash(intensity=0.7, n_bubbles=30) @ 4.0s pan=0.4 distance_m=2.0
impact(material=rock, rigidity=0.7) @ 5.6s gain=0.6
"""
        dsl_input = gr.Textbox(value=default_recipe, label="DSL recipe", lines=8)
        btn_dsl = gr.Button("Parse & render", variant="primary")
        audio_dsl = gr.Audio(label="Output (auto-detects stereo)", autoplay=True)
        dsl_info = gr.Markdown()

        def render_dsl(recipe: str):
            try:
                seq = parse_dsl(recipe)
                wav = seq.render()
                msg = f"OK · {len(seq.events)} events · duration {seq.duration_s:.1f}s · "
                msg += "stereo" if wav.ndim == 2 else "mono"
                return (SAMPLE_RATE, wav.astype(np.float32)), msg
            except ValueError as e:
                return None, f"**Parse error**: {e}"

        btn_dsl.click(render_dsl, inputs=dsl_input, outputs=[audio_dsl, dsl_info])

    gr.Markdown(
        "---\n"
        "**About**: All audio is rendered procedurally by a physics-informed parametric "
        "engine. No neural networks involved in the synthesis path. Submitted to "
        "**Tecniacústica 2026** (double-blind)."
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=False,
               theme=gr.themes.Soft(primary_hue="blue"))
