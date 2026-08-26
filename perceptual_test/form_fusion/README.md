# Test de escucha: fusion de sonidos

Formulario A/B pareado: suma vs fusion ganadora, 3 parejas x 2 condiciones.
Lee los estimulos de `perceptual_test/fusion_manifest.csv`, que genera
`scripts/39_fusion_search.py --freeze` (junto con los wav en
`perceptual_test/stimuli_fusion/`). No existen hasta que alguien corre
`--freeze` con `GANADORES_V12` rellena.

## Servir el test real

Dos formas validas (el formulario prueba el manifest en el mismo
directorio primero, y un nivel por encima despues):

**(a) servidor con raiz en `perceptual_test/`** (no copia nada):
```
python -m http.server -d perceptual_test 8000
```
Abrir `http://localhost:8000/form_fusion/`.

**(b) copiar el manifest y los estimulos dentro de esta carpeta** (mismo
patron que `scripts/16_build_perceptual_form.py` hace con `form/audio/`):
```
cp perceptual_test/fusion_manifest.csv perceptual_test/form_fusion/
cp -r perceptual_test/stimuli_fusion perceptual_test/form_fusion/
cd perceptual_test/form_fusion && python -m http.server 8000
```
Abrir `http://localhost:8000/`. Util para subir la carpeta entera a un
hosting estatico (Netlify, GitHub Pages, ...).

## Modo demo (revision del formulario sin estimulos reales)

`index.html?demo=1` carga 6 filas SINTETICAS embebidas en el propio HTML
(`DEMO_ROWS`) para poder revisar visualmente los tres items y el CSV
exportado sin depender de que `--freeze` haya corrido. El listener id
descargado siempre empieza por `DEMO_`; `scripts/40_analyze_fusion_test.py`
descarta ese prefijo si aparece por error en `responses_fusion/`. No envies
ese CSV al experimentador.

## Recogida de respuestas

El oyente descarga un CSV al terminar (`respuestas_fusion_<listener>_<ts>.csv`)
y lo guarda en `perceptual_test/responses_fusion/`. Analisis:
`PYTHONPATH=. .venv/bin/python scripts/40_analyze_fusion_test.py`.
