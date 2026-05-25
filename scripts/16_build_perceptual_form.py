"""Genera un formulario web autocontenido (single HTML) para el test perceptual.

El formulario:
  - Sirve los 24 estimulos en orden aleatorizado por oyente (Latin Square light).
  - Cada estimulo: audio reproducible + 4 preguntas.
  - Al final, descarga un CSV local con las respuestas.

Para distribucion: el HTML referencia los wavs por ruta relativa, asi que
o bien (a) se sube todo a un servidor estatico (Netlify/GitHub Pages) o
(b) el oyente abre el index.html localmente desde la carpeta perceptual_test/.

Para anonimato y registro centralizado, el plan es subirlo a Netlify
y que el CSV se envie a un endpoint (Formspree o similar). Como MVP del
sprint, el formulario descarga el CSV al final.

Uso:
    python scripts/16_build_perceptual_form.py
    # Abrir perceptual_test/form/index.html y servir con:
    python -m http.server 8000  # desde perceptual_test/form
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("perceptual_test/form")
OUT.mkdir(parents=True, exist_ok=True)

# Copiar los 24 wavs al subdirectorio audio/
audio_dir = OUT / "audio"
audio_dir.mkdir(exist_ok=True)
for w in Path("perceptual_test/stimuli").glob("*.wav"):
    shutil.copy(w, audio_dir / w.name)

# Leer manifest para metadata
manifest_path = Path("perceptual_test/stimuli_manifest.csv")
stimuli = []
with manifest_path.open() as fh:
    for r in csv.DictReader(fh):
        stimuli.append({
            "file": Path(r["path"]).name,
            "combo": r["combo"],
            "variant": r["variant"],
        })

MATERIALS = ["wood", "metal", "rock", "fabric", "earth", "liquid", "gravel", "other", "not sure"]
INTERACTIONS = ["impact", "scrape", "roll", "drip", "splash", "pour", "drag", "step", "not sure"]

HTML = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Listening test: impossible sounds</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 760px; margin: 24px auto; padding: 0 16px; color: #222; }
h1 { font-size: 1.4em; }
.stimulus { border: 1px solid #ddd; padding: 16px 18px; margin: 16px 0; border-radius: 8px; background: #fafafa; }
audio { width: 100%; margin: 8px 0; }
fieldset { border: 0; padding: 0; margin: 12px 0; }
label { display: inline-block; padding: 2px 6px; }
.likert label { padding: 4px 10px; border: 1px solid #ccc; margin: 2px; border-radius: 4px; cursor: pointer; }
.likert input[type=radio] { display: none; }
.likert input[type=radio]:checked + span { background: #5e8de2; color: white; padding: 4px 8px; border-radius: 3px; }
button { background: #2e6fd2; color: white; border: 0; padding: 10px 24px; font-size: 1em; border-radius: 6px; cursor: pointer; }
button:hover { background: #1c54a8; }
.progress { font-weight: bold; color: #2e6fd2; }
.intro { background: #eef5ff; padding: 12px 16px; border-radius: 6px; }
.warn { color: #b71c1c; font-size: 0.9em; }
select { font-size: 0.95em; padding: 4px; }
</style>
</head>
<body>

<h1>Listening test: impossible sounds</h1>
<div class=\"intro\">
  <p>This study explores the perception of synthesized sound events that combine
  materials and physical interactions in ways that do not occur in nature
  (rolling drops, liquid rocks, wet gravel). For each sound, please report what
  you perceive. No technical expertise is required; trust your ear.</p>
  <p><strong>~7 minutes</strong>. Use headphones if possible. You will hear
  <strong>12 of 24</strong> short clips (5 s each).</p>
  <p class=\"warn\">No personal data is collected. A CSV with your numeric
  answers will be offered for download at the end so you can email it to the
  experimenter.</p>
</div>

<p>Listener ID (any short string; helps deduplicate):
  <input id=\"listener\" type=\"text\" placeholder=\"e.g., L01\" size=\"10\" required>
</p>
<p>Years of audio experience (Foley / mixing / instrument):
  <select id=\"experience\">
    <option value=\"0\">None</option>
    <option value=\"1\">&lt; 1</option>
    <option value=\"3\">1-5</option>
    <option value=\"7\">5-10</option>
    <option value=\"15\">&gt; 10</option>
  </select>
</p>

<button id=\"start\" onclick=\"begin()\">Start</button>

<div id=\"test\" style=\"display:none\"></div>

<script>
const STIMULI = __STIMULI_JSON__;
const MATERIALS = __MATERIALS_JSON__;
const INTERACTIONS = __INTERACTIONS_JSON__;

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function begin() {
  const listener = document.getElementById('listener').value.trim();
  if (!listener) { alert('Please enter a listener ID'); return; }
  document.getElementById('start').style.display = 'none';
  document.getElementById('listener').disabled = true;
  document.getElementById('experience').disabled = true;
  const subset = shuffle(STIMULI.slice()).slice(0, 12);
  buildTest(subset);
}

function buildTest(subset) {
  const div = document.getElementById('test');
  div.style.display = '';
  subset.forEach((s, idx) => {
    div.appendChild(makeBlock(s, idx, subset.length));
  });
  const submit = document.createElement('button');
  submit.textContent = 'Submit and download CSV';
  submit.onclick = () => collect(subset);
  div.appendChild(submit);
}

function makeBlock(s, idx, total) {
  const block = document.createElement('div');
  block.className = 'stimulus';
  block.innerHTML = `
    <p class=\"progress\">Sound ${idx+1} of ${total}</p>
    <audio controls src=\"audio/${s.file}\"></audio>
    <fieldset>
      <label>What MATERIAL do you hear?</label>
      <select name=\"mat_${idx}\">
        ${MATERIALS.map(m => `<option value=\"${m}\">${m}</option>`).join('')}
      </select>
    </fieldset>
    <fieldset>
      <label>What INTERACTION do you hear?</label>
      <select name=\"int_${idx}\">
        ${INTERACTIONS.map(m => `<option value=\"${m}\">${m}</option>`).join('')}
      </select>
    </fieldset>
    <fieldset class=\"likert\">
      <label><strong>How hybrid/impossible does it sound?</strong> (1 = sounds totally natural, 7 = clearly impossible)</label><br>
      ${[1,2,3,4,5,6,7].map(v => `<label><input type=\"radio\" name=\"imp_${idx}\" value=\"${v}\" required><span>${v}</span></label>`).join('')}
    </fieldset>
    <fieldset class=\"likert\">
      <label><strong>How coherent is it as a single event?</strong> (1 = collage of unrelated sounds, 7 = single coherent event)</label><br>
      ${[1,2,3,4,5,6,7].map(v => `<label><input type=\"radio\" name=\"coh_${idx}\" value=\"${v}\" required><span>${v}</span></label>`).join('')}
    </fieldset>
  `;
  return block;
}

function collect(subset) {
  const listener = document.getElementById('listener').value.trim();
  const experience = document.getElementById('experience').value;
  const rows = [];
  rows.push(['listener','experience','stim_file','combo','variant','material_pred','interaction_pred','impossibility_likert','coherence_likert']);
  subset.forEach((s, idx) => {
    const mat = document.querySelector(`select[name=mat_${idx}]`).value;
    const intx = document.querySelector(`select[name=int_${idx}]`).value;
    const imp = document.querySelector(`input[name=imp_${idx}]:checked`);
    const coh = document.querySelector(`input[name=coh_${idx}]:checked`);
    if (!imp || !coh) { alert('Please complete all Likert questions before submitting'); return; }
    rows.push([listener, experience, s.file, s.combo, s.variant, mat, intx, imp.value, coh.value]);
  });
  const csv = rows.map(r => r.map(v => `\"${String(v).replace(/\"/g,'\"\"')}\"`).join(',')).join('\\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `responses_${listener}_${Date.now()}.csv`;
  a.click();
  alert('Thank you! Please email the downloaded CSV file to the experimenter.');
}
</script>

</body>
</html>
"""

html = (HTML
        .replace("__STIMULI_JSON__", json.dumps(stimuli))
        .replace("__MATERIALS_JSON__", json.dumps(MATERIALS))
        .replace("__INTERACTIONS_JSON__", json.dumps(INTERACTIONS)))
(OUT / "index.html").write_text(html)

# README cortito para servir
(OUT / "README.md").write_text("""\
# Listening test form

Local:
```
cd perceptual_test/form
python -m http.server 8000
```
Then open http://localhost:8000

For remote distribution: upload the contents of this folder to any static
host (Netlify drop, GitHub Pages, S3+CloudFront). Audio is referenced by
relative paths inside `audio/`.
""")

print(f"Generated {OUT / 'index.html'}")
print(f"Audios copied: {len(list(audio_dir.glob('*.wav')))}")
print(f"\nServe with:  cd {OUT} && python -m http.server 8000")
