# Envío Tecniacústica 2026

Carpeta con el material listo para enviar al **57º Congreso Español
de Acústica · XIII Congreso Ibérico de Acústica · TECNIACÚSTICA 2026**
(Granada, 21–23 octubre 2026).

## Estructura

```
manuscript/tecniacustica/
├── abstract.txt          Texto del abstract para el formulario online
├── paper.tex             Paper completo en español (≈8 páginas)
├── tecniacustica.cls     Plantilla LaTeX no oficial (reproduce el look
│                          del 53º Congreso, Elche 2022)
└── README.md             Este fichero
```

## Compilar el paper

```bash
cd manuscript/tecniacustica
pdflatex paper.tex
pdflatex paper.tex      # 2ª pasada para resolver referencias
```

El PDF resultante reproduce el formato visual del congreso (header
verde con texto del congreso, A4 una columna, secciones numéricas en
mayúsculas, ortografía española).

## Sustituir por la plantilla oficial cuando aparezca

`tecniacustica.cls` es un **esqueleto razonable** mientras la
organización no publique la plantilla oficial. Cuando aparezca en
[tecniacustica.es](https://www.tecniacustica.es/TECNIACUSTICA2026/)
o en [sea-acustica.es](https://www.sea-acustica.es/):

1. Descargar la plantilla oficial (LaTeX o Word).
2. Si es Word: pegar el contenido de `paper.tex` sección a sección,
   conservando formato de la plantilla.
3. Si es LaTeX: cambiar `\documentclass{tecniacustica}` por
   `\documentclass{oficial}` y revisar comandos.

## Logo del congreso

`tecniacustica.cls` deja preparado un `\includegraphics` en el header
comentado. Cuando se descargue el logo oficial, guardarlo en esta
carpeta como `tecniacustica-logo.png` y descomentar la línea.

## Doble ciego — checklist antes de enviar

```bash
# Estas líneas deben dar cero resultados en el abstract y paper:
grep -i "mateo\|camara\|upm\|alumnos" abstract.txt paper.tex
grep -i "github\.com/MateoCamara\|our previous\|nuestro trabajo previo" paper.tex
```

El paper incluye `[Doble ciego: autores y afiliaciones se añaden tras
revisión]` en el bloque de autores. Sustituir tras aceptación.

## Datos importantes del congreso

- **Lugar**: Granada, España
- **Fechas**: 21–23 octubre 2026
- **Fecha límite abstract**: pendiente confirmación (revisar web del congreso)
- **Fecha límite paper completo**: pendiente confirmación
- **Idiomas admitidos**: español, portugués, inglés
- **Sesiones objetivo recomendadas**:
  - A09-1 ML/IA en Acústica
  - A16-1 Procesamiento y Aprendizaje para Señales Acústicas
  - A24-1 Acústica Virtual

## Figuras

El paper referencia figuras en `../../figures/`:
- `F2_rolling_droplet_spectrograms.png`
- `F3_monotonicity_heatmap.png`
- `F4_sweep_curves.png`
- `F6_fusion.png`

`F5_master_comparison.png` se retiró del paper en la Tarea 7 (Nivel 2,
recorte de páginas): sus cifras ya estaban en `tab:master`, la tabla
maestra de la Sección de Evaluación, así que la figura era redundante.
El fichero PNG se conserva en `figures/` por si se reutiliza en otro
sitio, pero ya no se cita desde `paper.tex`.

Si las regeneras con `scripts/15_figures.py` o
`scripts/18_compare_physics_vs_neural.py`, las referencias del paper
las recogen automáticamente.

## PACS y palabras clave

PACS del paper (verificados 2026-08-26 contra el listado oficial AIP/JASA
"Appendix to 43: Acoustics", PACS 2010):
- **43.60.Lq** Acoustic imaging, displays, pattern recognition, feature extraction
- **43.58.Ta** Computers and computer programs in acoustics
- **43.60.Uv** Model-based signal processing

Las anotaciones anteriores de 43.60.Lq ("Speech and Music Perception ·
Computer algorithms, synthesis") y 43.58.Ta ("Instrumentation · Digital
signal processing") eran incorrectas frente al listado oficial; quedan
corregidas arriba. El código perceptual **43.66.Lj** (Hearing ·
Psychoacoustics — oficialmente "Perceptual effects of sound") se retiró
al eliminar el estudio perceptual piloto del paper (Tarea 7) y se
sustituyó por **43.60.Uv**, que encaja con el procesado basado en
modelos físicos que vertebra tanto el motor de síntesis como la
recuperación diferenciable de parámetros.

Nota: el `abstract.txt` enviado al congreso (meses antes que el paper
completo) sí menciona el estudio piloto perceptual. **No se reenvía**
— es el comportamiento normal del proceso de envío — y por tanto queda
una diferencia intencionada entre lo enviado como abstract y el paper
final.

Palabras clave: *síntesis de sonido, Foley, física-DSP, sonidos
imposibles, DDSP, inversión de parámetros*.
