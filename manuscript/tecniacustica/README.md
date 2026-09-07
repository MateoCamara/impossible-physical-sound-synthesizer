# Envío Tecniacústica 2026

Material de envío al **57º Congreso Español de Acústica · XIV Congreso
Ibérico de Acústica · TECNIACÚSTICA 2026** (Granada, 21–23 de octubre de 2026).

## Datos del congreso verificados el 2026-09-07

Fuente: `tecniacustica.es/TECNIACUSTICA2026/comunicaciones/normativa` y
`.../comunicaciones/plantillas-finales`.

- **Fecha límite del paper completo:** la normativa dice «nueva fecha! 14 de
  septiembre de 2026». La página de fechas importantes seguía diciendo 7 de
  septiembre. Ante la duda, secretaría técnica: tecniacustica@viajeseci.es.
- **No es doble ciego.** «Todos los autores deben indicar nombre, apellidos y
  filiación». El PDF va directo al Libro de Actas «sin posibilidad de ser
  editado posteriormente». Los abstracts se aceptaron en junio.
- **Extensión: entre 4 y 8 páginas**, figuras, tablas y referencias incluidas.
  Resumen y abstract de 250 palabras como máximo. Palabras clave en español
  e inglés, obligatorias.
- **Plantilla oficial obligatoria.** «No se publicará en el libro de actas
  ningún artículo que no se ajuste a las plantillas suministradas». Solo se
  admite PDF sin proteger, máximo 10 MB.
- Idiomas admitidos: español, portugués, inglés.
- Sesiones sugeridas: A09-1 (ML/IA en Acústica), A16-1 (Procesamiento y
  Aprendizaje para Señales Acústicas), A24-1 (Acústica Virtual).

## Estructura

```
manuscript/tecniacustica/
├── paper.tex               Paper completo en español (8 páginas)
├── tecniacustica2026.sty   Plantilla oficial (publicada el 30/06/2026). NO tocar.
├── Tecni_Banner.pdf        Banner de la portada (plantilla oficial)
├── footer_logo_sea.png     Logos del pie de la portada (plantilla oficial)
├── logo_spacustica_1.png
├── abstract.txt            Texto del abstract enviado en mayo (ya aceptado)
└── README.md               Este fichero
```

La plantilla oficial viene de `LaTeX_TecniAcustica2026_Templates-ES.zip`
(web del congreso). Fija Times 11 pt, márgenes 25/35 mm, portada exclusiva
en la página 1 (banner, título, autores, afiliaciones, resumen, abstract,
palabras clave) y el cuerpo desde la página 2. La bibliografía va en estilo
IEEE numérico a tamaño normal, sin columnas.

La clase no oficial `tecniacustica.cls` (imitación de Elche 2022) se retiró
el 2026-09-07; está en el historial de git si hiciera falta.

## Compilar

No hay LaTeX instalado en la máquina. Con docker, montando `figures` aparte
porque el paper usa rutas `../../figures/`:

```bash
docker run --rm -v "$PWD/manuscript/tecniacustica:/w" -v "$PWD/figures:/figures" -w /w \
  texlive/texlive sh -c 'pdflatex -interaction=nonstopmode paper.tex >/dev/null 2>&1; \
                         pdflatex -interaction=nonstopmode paper.tex 2>&1 | tail -2'
```

La imagen `texlive/texlive:latest-small` no sirve (faltan `titlesec` y
`tikz`). El número de páginas sale en la última línea de pdflatex; el paper
tiene que dar 8.

## Checklist antes de enviar

```bash
# Autores, afiliación y correo reales (la plantilla los imprime en la portada):
grep -n "PENDIENTE" paper.tex                       # debe dar cero
# Nada de restos de la versión anónima:
grep -n -i "doble ciego\|anónim" paper.tex           # debe dar cero
# 8 páginas y cero referencias sin resolver:
grep -a -E "Output written|undefined" paper.log
```

## Figuras

El paper incluye `../../figures/F3_monotonicity_heatmap.png` (mapa de calor
de monotonía) y `../../figures/F6_fusion.png` (FCI frente a factor de cresta,
un solo panel). Se regeneran con `scripts/15_figures.py` y
`scripts/41_fusion_evidence.py`; los PNG no llevan título incrustado, el pie
lo pone LaTeX.

`F2_rolling_droplet_spectrograms.png` (espectrogramas de la gota rodante)
está regenerada en español y sin título, pero no cabe en las 8 páginas con la
plantilla oficial: incluirla cuesta unas 8 líneas más de lo que hay.
`F5_master_comparison.png` y `F4_sweep_curves.png` no se citan.

## PACS y palabras clave

PACS (verificados contra el listado oficial AIP/JASA, PACS 2010):
**43.60.Lq** (pattern recognition, feature extraction), **43.58.Ta**
(computers and computer programs in acoustics), **43.60.Uv** (model-based
signal processing). El código perceptual 43.66.Lj se retiró al eliminar el
estudio piloto del paper.

Nota: el `abstract.txt` enviado en mayo sí menciona el estudio piloto
perceptual. No se reenvía, es el flujo normal del congreso, y queda una
diferencia intencionada entre el abstract y el paper final.

Palabras clave: *síntesis de sonido, Foley, física-DSP, sonidos imposibles,
DDSP, inversión de parámetros* / *sound synthesis, Foley, physics-DSP,
impossible sounds, DDSP, parameter inversion*.
