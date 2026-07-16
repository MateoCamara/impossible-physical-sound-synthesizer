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
    synth_burning_water, synth_glass_thunder,
    synth_mercury_rain, synth_fabric_bell,
)
from impossible_mix.physics.sequences import (
    droplet_story, mercury_drama, lava_step_into_water, parse_dsl,
    plasma_meteor_strike, wax_droplets_into_silk, rubber_through_glass,
    mud_avalanche, ice_drop_in_lava,
)
from impossible_mix.physics.spatial import place_source
from impossible_mix.physics.reverb import generate_ir, apply_reverb, PRESETS as IR_PRESETS


def to_gradio_audio(wav: np.ndarray, sr: int = SAMPLE_RATE):
    """Gradio Audio espera (sr, np.array int16 o float32)."""
    return (sr, wav.astype(np.float32))


# ====================================================================
# Tab 1: Rolling droplet with physical knobs + preset selector
# ====================================================================
def render_v5_live(radius_mm, viscosity, surface_hardness, roll_velocity_hz,
                   path_roughness, rev_wobble_depth, asperities, density_mul,
                   pattern_drift, core_mix, accent_gain, fusion_auto, fusion,
                   profile_floor, smoothness, noise_darkness, tonal_mix,
                   sing_mix, duration_s, seed):
    """Render on-release de la pestana Rolling v5 (live): todos los dials."""
    from impossible_mix.physics.droplet import DropletParams
    p = DropletParams(
        droplet_radius_mm=radius_mm, viscosity=viscosity,
        surface_hardness=surface_hardness, roll_velocity_hz=roll_velocity_hz,
        path_roughness=path_roughness,
        rev_wobble_depth=rev_wobble_depth,
        asperities_per_rev=(None if int(asperities) == 0 else int(asperities)),
        contact_density_mul=density_mul, pattern_drift=pattern_drift,
        continuous_core_mix=core_mix, accent_gain=accent_gain,
        fusion=(None if fusion_auto else fusion),
        profile_floor=profile_floor, smoothness=smoothness,
        noise_darkness=noise_darkness, tonal_mix=tonal_mix, sing_mix=sing_mix,
        duration_s=duration_s, seed=int(seed))
    return to_gradio_audio(synth_rolling_droplet(p, SAMPLE_RATE))


_UI_CAPTURE_DIR = Path("escucha_AB/ui_capturas")


def save_v5_capture(*args):
    """Guarda el render actual de la pestana v5 como wav numerado."""
    import time
    from impossible_mix.utils import save_wav
    sr, wav = render_v5_live(*args)
    _UI_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    n = len(list(_UI_CAPTURE_DIR.glob("*.wav")))
    path = _UI_CAPTURE_DIR / f"{n:03d}_{int(time.time())}.wav"
    save_wav(path, wav, sr)
    return str(path)


# --- Nuevos imposibles v6: (funcion, etiqueta_p1, (min,max,def), etiqueta_p2, (min,max,def)) ---
_V6_FX = {
    "burning_water (agua ardiendo)": (
        lambda d, p1, p2, s: synth_burning_water(duration_s=d, intensity=p1,
                                                 drip_rate_hz=p2, seed=s),
        "Intensidad", (0.1, 1.0, 0.7), "Gotas/s", (2.0, 20.0, 8.0)),
    "glass_thunder (trueno de cristal)": (
        lambda d, p1, p2, s: synth_glass_thunder(duration_s=d, distance=p1,
                                                 ring_gain=p2, seed=s),
        "Distancia", (0.0, 1.0, 0.4), "Resonancia vidrio", (0.0, 1.2, 0.6)),
    "mercury_rain (lluvia de mercurio)": (
        lambda d, p1, p2, s: synth_mercury_rain(duration_s=d, intensity=p1,
                                                drop_radius_mm=p2, seed=s),
        "Intensidad", (0.1, 1.0, 0.6), "Radio gota (mm)", (0.4, 2.0, 0.9)),
    "fabric_bell (campana de tela)": (
        lambda d, p1, p2, s: synth_fabric_bell(duration_s=d, size=p1,
                                               softness=p2, seed=s),
        "Tamaño campana", (0.1, 1.0, 0.5), "Suavidad (tela)", (0.0, 1.0, 0.7)),
}


def render_v6_fx(effect: str, p1: float, p2: float, duration_s: float, seed: int):
    fn, *_ = _V6_FX[effect]
    return to_gradio_audio(fn(duration_s, p1, p2, int(seed)))


def _v6_slider_labels(effect: str):
    _, l1, r1, l2, r2 = _V6_FX[effect]
    return (gr.update(label=l1, minimum=r1[0], maximum=r1[1], value=r1[2]),
            gr.update(label=l2, minimum=r2[0], maximum=r2[1], value=r2[2]))


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

    # ------- Tab 1b: Rolling v5 (live) --------
    with gr.Tab("Rolling v5 (live)"):
        gr.Markdown(
            "**Motor v5 completo, casi tiempo real.** Cada slider re-renderiza "
            "al soltarlo (~0,5 s) y reproduce. Los defaults son la receta "
            "consolidada (canica continua mojada). *Asperezas/vuelta = 0* usa "
            "el auto (4 + 4·rugosidad); *fusion auto* la deriva de la velocidad."
        )
        with gr.Row():
            v5_radius = gr.Slider(0.3, 6.0, value=2.2, step=0.1, label="Radio gota (mm)")
            v5_visc = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Viscosidad")
            v5_hard = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Dureza superficie")
            v5_vel = gr.Slider(2.0, 40.0, value=14.0, step=1.0, label="Velocidad (rev-contactos/s)")
            v5_rough = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Rugosidad trayectoria")
        with gr.Row():
            v5_wob = gr.Slider(0.0, 0.6, value=0.22, step=0.02, label="Wobble por vuelta")
            v5_asp = gr.Slider(0, 12, value=0, step=1, label="Asperezas/vuelta (0=auto)")
            v5_dens = gr.Slider(0.5, 4.0, value=2.0, step=0.25, label="Densidad de contactos ×")
            v5_drift = gr.Slider(0.0, 0.3, value=0.05, step=0.01, label="Precesión del patrón")
        with gr.Row():
            v5_core = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Núcleo continuo")
            v5_acc = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Acentos discretos")
            v5_fauto = gr.Checkbox(value=True, label="Fusión auto (por velocidad)")
            v5_fus = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Fusión manual")
            v5_floor = gr.Slider(0.0, 0.6, value=0.18, step=0.02, label="Suelo del perfil")
        with gr.Row():
            v5_smooth = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Suavidad (anti-aspereza)")
            v5_dark = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Oscuridad del ruido")
            v5_tonal = gr.Slider(0.0, 1.2, value=0.4, step=0.05, label="Zumbido de canica")
            v5_sing = gr.Slider(0.0, 1.2, value=0.4, step=0.05, label="Canto de copa")
        with gr.Row():
            v5_dur = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label="Duración (s)")
            v5_seed = gr.Number(value=42, label="Seed", precision=0)
            v5_btn = gr.Button("Render", variant="primary")
            v5_save = gr.Button("Guardar wav")
        v5_audio = gr.Audio(label="Output", autoplay=True)
        v5_saved = gr.Textbox(label="Último guardado", interactive=False)

        _v5_inputs = [v5_radius, v5_visc, v5_hard, v5_vel, v5_rough, v5_wob,
                      v5_asp, v5_dens, v5_drift, v5_core, v5_acc, v5_fauto,
                      v5_fus, v5_floor, v5_smooth, v5_dark, v5_tonal, v5_sing,
                      v5_dur, v5_seed]
        for _c in _v5_inputs[:-2]:
            if hasattr(_c, "release"):
                _c.release(render_v5_live, inputs=_v5_inputs, outputs=v5_audio)
            else:
                _c.change(render_v5_live, inputs=_v5_inputs, outputs=v5_audio)
        v5_btn.click(render_v5_live, inputs=_v5_inputs, outputs=v5_audio)
        v5_save.click(save_v5_capture, inputs=_v5_inputs, outputs=v5_saved)

    # ------- Tab 1c: Nuevos imposibles (v6) --------
    with gr.Tab("Nuevos imposibles"):
        gr.Markdown(
            "**Cuatro fenómenos físicamente imposibles nuevos**: fuego cuyas "
            "chispas son gotas, cielo de vidrio que truena, llovizna metálica "
            "que tintinea, y una campana de metal con amortiguamiento de tela. "
            "Los sliders se re-etiquetan según el efecto; re-render al soltar."
        )
        with gr.Row():
            fx_sel = gr.Dropdown(list(_V6_FX.keys()),
                                 value=list(_V6_FX.keys())[0], label="Efecto")
            fx_dur = gr.Slider(2.0, 10.0, value=5.0, step=0.5, label="Duración (s)")
            fx_seed = gr.Number(value=42, label="Seed", precision=0)
        with gr.Row():
            fx_p1 = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Intensidad")
            fx_p2 = gr.Slider(2.0, 20.0, value=8.0, step=0.5, label="Gotas/s")
        fx_btn = gr.Button("Render", variant="primary")
        fx_audio = gr.Audio(label="Output", autoplay=True)

        fx_sel.change(_v6_slider_labels, inputs=fx_sel, outputs=[fx_p1, fx_p2])
        _fx_inputs = [fx_sel, fx_p1, fx_p2, fx_dur, fx_seed]
        for _c in (fx_p1, fx_p2):
            _c.release(render_v6_fx, inputs=_fx_inputs, outputs=fx_audio)
        fx_sel.change(render_v6_fx, inputs=_fx_inputs, outputs=fx_audio)
        fx_btn.click(render_v6_fx, inputs=_fx_inputs, outputs=fx_audio)

    # ------- Tab 1d: Chimera lab (v10, el metodo de fusion ganador) --------
    with gr.Tab("Chimera lab"):
        gr.Markdown(
            "**Fusión de identidad por chimera auditiva** (Smith, Delgutte & "
            "Oxenham, *Nature* 2002): el padre A pone la **dinámica** "
            "(envolvente), el padre B pone la **materia** (estructura fina) — "
            "una sola señal por construcción. El nº de bandas es el dial de "
            "identidad: pocas → domina la materia; muchas → domina la dinámica."
        )
        from impossible_mix.physics.blend import auditory_chimera as _chimera
        from impossible_mix.physics.blend_recipes import (
            CHIMERA_PAIRS as _PAIRS, CHIMERA_PARENTS as _PARENTS,
            chimera_parent as _parent)

        def render_chimera_ui(a, b, n_bands, duration_s, seed):
            wa = _parent(a, duration_s, int(seed))
            wb = _parent(b, duration_s, int(seed) + 17)
            w = _chimera(wa, wb, SAMPLE_RATE, n_bands=int(n_bands))
            peak = float(np.abs(w).max() + 1e-9)
            return to_gradio_audio((w * (0.9 / peak)).astype(np.float32))

        with gr.Row():
            ch_a = gr.Dropdown(list(_PARENTS), value="fuego",
                               label="Padre A (dinámica / envolvente)")
            ch_b = gr.Dropdown(list(_PARENTS), value="vidrio",
                               label="Padre B (materia / estructura fina)")
            ch_nb = gr.Slider(1, 32, value=16, step=1, label="Bandas (dial de identidad)")
        with gr.Row():
            ch_dur = gr.Slider(2.0, 10.0, value=6.0, step=0.5, label="Duración (s)")
            ch_seed = gr.Number(value=42, label="Seed", precision=0)
            ch_btn = gr.Button("Render", variant="primary")
        with gr.Row():
            _pair_names = [p[0] for p in _PAIRS]
            ch_preset = gr.Dropdown(_pair_names, value=_pair_names[1],
                                    label="Parejas curadas (cargan A/B/bandas)")
        ch_audio = gr.Audio(label="Output", autoplay=True)

        def _load_pair(name):
            for pname, a, b, nb in _PAIRS:
                if pname == name:
                    return a, b, nb
            return "fuego", "vidrio", 16

        _ch_inputs = [ch_a, ch_b, ch_nb, ch_dur, ch_seed]
        ch_preset.change(_load_pair, inputs=ch_preset, outputs=[ch_a, ch_b, ch_nb])
        ch_preset.change(render_chimera_ui, inputs=_ch_inputs, outputs=ch_audio)
        ch_nb.release(render_chimera_ui, inputs=_ch_inputs, outputs=ch_audio)
        for _c in (ch_a, ch_b):
            _c.change(render_chimera_ui, inputs=_ch_inputs, outputs=ch_audio)
        ch_btn.click(render_chimera_ui, inputs=_ch_inputs, outputs=ch_audio)

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
        _CINEMATIC_FNS = {
            "droplet_story": droplet_story,
            "mercury_drama": mercury_drama,
            "lava_step_into_water": lava_step_into_water,
            "plasma_meteor_strike": plasma_meteor_strike,
            "wax_droplets_into_silk": wax_droplets_into_silk,
            "rubber_through_glass": rubber_through_glass,
            "mud_avalanche": mud_avalanche,
            "ice_drop_in_lava": ice_drop_in_lava,
        }
        with gr.Row():
            story = gr.Dropdown(
                list(_CINEMATIC_FNS.keys()),
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
            "*plasma_meteor_strike* + *exterior* → otherworldly meteor in the open. "
            "*ice_drop_in_lava* + *cave* → impossible thermal contrast. "
            "*mud_avalanche* + *medium_hall* → epic geological event."
        )

        def render_cinematic(story_name, duration, seed, distance, pan, with_delay_flag, ir_name, mix):
            fn = _CINEMATIC_FNS.get(story_name, droplet_story)
            # Some recipes have their own intrinsic timing and may ignore duration.
            try:
                wav = fn(duration_s=duration, seed=int(seed))
            except TypeError:
                wav = fn(seed=int(seed))
            # If the sequence already returned stereo (per-event pan), reduce to mono
            # for place_source's air-absorption/distance pipeline, then re-pan.
            if wav.ndim == 2:
                wav_mono = wav.mean(axis=-1).astype(np.float32)
            else:
                wav_mono = wav
            stereo = place_source(wav_mono, SAMPLE_RATE, distance_m=distance, pan=pan,
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
