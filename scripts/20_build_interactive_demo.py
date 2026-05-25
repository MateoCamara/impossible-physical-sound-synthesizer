"""Demo interactivo HTML con sliders sobre una grid pre-renderizada de la
gota rodante. Distribuible en cualquier static host.

Grid: 3 knobs (wetness, granularity, continuity) x N valores cada uno =
3D de audios. El HTML reproduce el archivo correspondiente al estado
actual de los sliders.

Tambien incluye demos one-shot de las otras 2 escenas (liquid_rock_impact
y wet_gravel_scrape) con un slider mas pequeno.

Uso:
    python scripts/20_build_interactive_demo.py
    cd perceptual_test/interactive_demo && python3 -m http.server 8765
"""
from __future__ import annotations

import json
import shutil
import sys
from itertools import product
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import SAMPLE_RATE
from impossible_mix.physics_controller import PhysicsController


OUT = Path("perceptual_test/interactive_demo")
AUDIO_DIR = OUT / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Grid principal: rolling droplet con 3 knobs
WET_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]
GRAN_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]
CONT_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]


def gen_rolling_droplet_grid() -> list[dict]:
    """3D grid de la gota rodante. 5x5x5 = 125 audios. Tiempo ~ 2-3 min."""
    ctrl = PhysicsController(seed=42, duration_s=4.0)
    ctrl.set_scene(material="liquid", interaction="roll")
    entries = []
    total = len(WET_VALS) * len(GRAN_VALS) * len(CONT_VALS)
    i = 0
    for wet, gran, cont in product(WET_VALS, GRAN_VALS, CONT_VALS):
        i += 1
        wav = ctrl.render(wetness=wet, granularity=gran, continuity=cont)
        fname = f"droplet_w{wet:.2f}_g{gran:.2f}_c{cont:.2f}.wav"
        sf.write(str(AUDIO_DIR / fname), wav, SAMPLE_RATE)
        entries.append(dict(wetness=wet, granularity=gran, continuity=cont, file=fname))
        if i % 25 == 0:
            print(f"  droplet grid: {i}/{total}")
    return entries


def gen_secondary_demo(
    scene_name: str,
    setup_kwargs: dict,
    knob: str,
    values: list[float],
    out_dir: Path,
) -> list[dict]:
    ctrl = PhysicsController(seed=42, duration_s=4.0)
    ctrl.set_scene(**setup_kwargs)
    entries = []
    for v in values:
        wav = ctrl.render(**{knob: v})
        fname = f"{scene_name}_{knob}{v:.2f}.wav"
        sf.write(str(out_dir / fname), wav, SAMPLE_RATE)
        entries.append({"value": v, "file": fname})
    return entries


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Interactive demo: How does a rolling droplet sound?</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 20px auto; padding: 0 16px; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.6em; color: #1565c0; }
h2 { font-size: 1.1em; margin-top: 28px; color: #444; }
.panel { background: white; border: 1px solid #ddd; border-radius: 10px; padding: 20px 24px; margin: 18px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.slider-row { display: flex; align-items: center; gap: 14px; margin: 10px 0; }
.slider-row label { min-width: 130px; font-weight: 600; }
.slider-row input[type=range] { flex: 1; }
.slider-row .value { min-width: 50px; text-align: right; font-family: monospace; color: #1565c0; font-weight: 600; }
audio { width: 100%; margin-top: 12px; }
.physical { font-size: 0.85em; color: #666; margin-top: 4px; padding-left: 144px; }
.help { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin: 14px 0; font-size: 0.95em; }
.scene-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
.scene-tabs button { background: white; border: 1px solid #ccc; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.95em; }
.scene-tabs button.active { background: #1565c0; color: white; border-color: #1565c0; }
.scene { display: none; }
.scene.active { display: block; }
.tagline { color: #555; font-style: italic; }
.preset-btns { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.preset-btns button { background: #eef5ff; border: 1px solid #b8d4ff; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.85em; }
.preset-btns button:hover { background: #d5e5ff; }
</style>
</head>
<body>

<h1>How does a rolling droplet sound?</h1>
<p class="tagline">An interactive demo of physics-informed parametric synthesis for impossible sounds.</p>

<div class="help">
  Drag the sliders to explore the parameter space. Audio is pre-rendered for each grid point. Each slider corresponds to a <strong>physical parameter</strong> with direct acoustic meaning. Use headphones.
</div>

<div class="scene-tabs">
  <button class="active" onclick="showScene('droplet')">Rolling droplet</button>
  <button onclick="showScene('rock')">Liquid rock impact</button>
  <button onclick="showScene('gravel')">Wet gravel scrape</button>
</div>

<!-- === ROLLING DROPLET === -->
<div id="scene-droplet" class="scene active panel">
  <h2>Rolling droplet (3 physical knobs)</h2>

  <div class="preset-btns">
    Presets:
    <button onclick="setDroplet(0.5, 0.5, 0.5)">canonical</button>
    <button onclick="setDroplet(1.0, 0.5, 0.5)">very wet</button>
    <button onclick="setDroplet(0.25, 0.5, 0.25)">dry slow drip</button>
    <button onclick="setDroplet(0.5, 1.0, 0.75)">grainy fast roll</button>
    <button onclick="setDroplet(0.75, 0.0, 1.0)">smooth fast slick</button>
  </div>

  <div class="slider-row">
    <label>Wetness</label>
    <input id="wet" type="range" min="0" max="1" step="0.25" value="0.5" oninput="updateDroplet()">
    <span class="value" id="wet-val">0.50</span>
  </div>
  <div class="physical">viscosity of the droplet (more wet = less viscous, faster bubbles)</div>

  <div class="slider-row">
    <label>Granularity</label>
    <input id="gran" type="range" min="0" max="1" step="0.25" value="0.5" oninput="updateDroplet()">
    <span class="value" id="gran-val">0.50</span>
  </div>
  <div class="physical">path roughness (more granular = more jitter between contacts)</div>

  <div class="slider-row">
    <label>Continuity</label>
    <input id="cont" type="range" min="0" max="1" step="0.25" value="0.5" oninput="updateDroplet()">
    <span class="value" id="cont-val">0.50</span>
  </div>
  <div class="physical">rolling velocity (more continuous = more contacts per second)</div>

  <audio id="droplet-audio" controls autoplay></audio>
  <div id="droplet-file" style="font-size:0.8em;color:#999;margin-top:6px"></div>
</div>

<!-- === LIQUID ROCK IMPACT === -->
<div id="scene-rock" class="scene panel">
  <h2>Liquid rock impact (1 knob: wetness)</h2>
  <p class="tagline">A rock impact augmented with a liquid splash overlay. Higher wetness = thicker liquid character.</p>
  <div class="slider-row">
    <label>Wetness</label>
    <input id="rock-wet" type="range" min="0" max="1" step="0.25" value="0.5" oninput="updateRock()">
    <span class="value" id="rock-wet-val">0.50</span>
  </div>
  <audio id="rock-audio" controls></audio>
  <div id="rock-file" style="font-size:0.8em;color:#999;margin-top:6px"></div>
</div>

<!-- === WET GRAVEL SCRAPE === -->
<div id="scene-gravel" class="scene panel">
  <h2>Wet gravel scrape (1 knob: wetness)</h2>
  <p class="tagline">Gravel scraping under an overlay of liquid pour. Higher wetness = more flooded character.</p>
  <div class="slider-row">
    <label>Wetness</label>
    <input id="gravel-wet" type="range" min="0" max="1" step="0.25" value="0.5" oninput="updateGravel()">
    <span class="value" id="gravel-wet-val">0.50</span>
  </div>
  <audio id="gravel-audio" controls></audio>
  <div id="gravel-file" style="font-size:0.8em;color:#999;margin-top:6px"></div>
</div>

<!-- === Footer === -->
<div class="panel" style="font-size:0.9em;color:#555">
  <strong>What you are hearing.</strong> The audio is synthesized procedurally
  by a physics-informed parametric engine. There is no neural network in the
  audio path. Each slider modulates a physical parameter (viscosity, path
  roughness, rolling velocity) and the engine rebuilds the sound. The
  underlying primitives are: modal resonators for surfaces, Minnaert
  resonance for bubble formation, stochastic granular events, and friction
  models for scrapes.
</div>

<script>
const DROPLET = __DROPLET_JSON__;
const ROCK = __ROCK_JSON__;
const GRAVEL = __GRAVEL_JSON__;

function showScene(name) {
  document.querySelectorAll('.scene').forEach(s => s.classList.remove('active'));
  document.getElementById('scene-' + name).classList.add('active');
  document.querySelectorAll('.scene-tabs button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
}

function findClosest(grid, query) {
  let best = null, bestDist = Infinity;
  for (const e of grid) {
    let d = 0;
    for (const k in query) d += Math.abs(e[k] - query[k]);
    if (d < bestDist) { bestDist = d; best = e; }
  }
  return best;
}

function updateDroplet() {
  const wet = parseFloat(document.getElementById('wet').value);
  const gran = parseFloat(document.getElementById('gran').value);
  const cont = parseFloat(document.getElementById('cont').value);
  document.getElementById('wet-val').textContent = wet.toFixed(2);
  document.getElementById('gran-val').textContent = gran.toFixed(2);
  document.getElementById('cont-val').textContent = cont.toFixed(2);
  const e = findClosest(DROPLET, {wetness: wet, granularity: gran, continuity: cont});
  const audio = document.getElementById('droplet-audio');
  const newSrc = 'audio/' + e.file;
  if (!audio.src.endsWith(e.file)) {
    audio.src = newSrc;
    audio.play().catch(()=>{});
  }
  document.getElementById('droplet-file').textContent = e.file;
}

function setDroplet(w, g, c) {
  document.getElementById('wet').value = w;
  document.getElementById('gran').value = g;
  document.getElementById('cont').value = c;
  updateDroplet();
}

function updateRock() {
  const v = parseFloat(document.getElementById('rock-wet').value);
  document.getElementById('rock-wet-val').textContent = v.toFixed(2);
  let best = ROCK[0];
  let bestD = Math.abs(best.value - v);
  for (const e of ROCK) {
    if (Math.abs(e.value - v) < bestD) { best = e; bestD = Math.abs(e.value - v); }
  }
  const audio = document.getElementById('rock-audio');
  if (!audio.src.endsWith(best.file)) {
    audio.src = 'audio/' + best.file;
    audio.play().catch(()=>{});
  }
  document.getElementById('rock-file').textContent = best.file;
}

function updateGravel() {
  const v = parseFloat(document.getElementById('gravel-wet').value);
  document.getElementById('gravel-wet-val').textContent = v.toFixed(2);
  let best = GRAVEL[0];
  let bestD = Math.abs(best.value - v);
  for (const e of GRAVEL) {
    if (Math.abs(e.value - v) < bestD) { best = e; bestD = Math.abs(e.value - v); }
  }
  const audio = document.getElementById('gravel-audio');
  if (!audio.src.endsWith(best.file)) {
    audio.src = 'audio/' + best.file;
    audio.play().catch(()=>{});
  }
  document.getElementById('gravel-file').textContent = best.file;
}

// Init
updateDroplet();
updateRock();
updateGravel();
</script>

</body>
</html>
"""


def main() -> int:
    print("=== Generating rolling droplet grid (125 audios) ===")
    droplet = gen_rolling_droplet_grid()

    print("\n=== Generating liquid rock impact sweep ===")
    rock = gen_secondary_demo(
        "rock", dict(material="rock", interaction="impact",
                     overlay_material="liquid", overlay_interaction="splash",
                     overlay_weight=0.55),
        "wetness", [0.0, 0.25, 0.5, 0.75, 1.0],
        AUDIO_DIR,
    )

    print("\n=== Generating wet gravel scrape sweep ===")
    gravel = gen_secondary_demo(
        "gravel", dict(material="gravel", interaction="scrape",
                       overlay_material="liquid", overlay_interaction="pour",
                       overlay_weight=0.35),
        "wetness", [0.0, 0.25, 0.5, 0.75, 1.0],
        AUDIO_DIR,
    )

    html = (HTML_TEMPLATE
            .replace("__DROPLET_JSON__", json.dumps(droplet))
            .replace("__ROCK_JSON__", json.dumps(rock))
            .replace("__GRAVEL_JSON__", json.dumps(gravel)))
    (OUT / "index.html").write_text(html)
    (OUT / "README.md").write_text(
        "# Interactive demo: How does a rolling droplet sound?\n\n"
        "Local: `cd perceptual_test/interactive_demo && python3 -m http.server 8765`\n"
        "Then open http://localhost:8765\n\n"
        "Remote: upload this folder to any static host (Netlify, GitHub Pages, S3).\n"
    )
    n_audios = len(list(AUDIO_DIR.glob("*.wav")))
    print(f"\nGenerated {n_audios} audios + index.html in {OUT}")
    print(f"Total size: {sum(f.stat().st_size for f in AUDIO_DIR.glob('*.wav')) / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
