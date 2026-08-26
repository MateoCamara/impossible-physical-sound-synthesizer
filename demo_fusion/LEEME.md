# Demo de fusión con sliders (`demo_fusion/`)

Generada por `scripts/43_demo_fusion_web.py` a partir de `demo_fusion/manifest.json`
(456 clips Opus: 432 de rejilla + 24 de
referencia, commit `d0731d4` de `scripts/42_fusion_set.py` -- este script NO regenera
el audio, solo construye la página que lo consume). Vuelve a ejecutar el generador
si cambia `manifest.json`; si no, `index.html`/`fusionpad.js` quedan desactualizados
respecto al audio real (usa `--check` para detectarlo, ver más abajo).

## Cómo servirla (importante: "sin wifi" no significa `file://`)

**Ruta principal, totalmente offline:**
```
python3 -m http.server -d demo_fusion 8000
```
y abrir `http://localhost:8000/`. `fetch` (necesario para decodificar los clips
bajo demanda) funciona con el wifi apagado siempre que sea `http://localhost` --
no hace ninguna petición externa (sin CDN, sin fuentes remotas, sin analítica).

**Si alguien abre `index.html` directamente con doble clic (`file://`):** `fetch`
falla ahí en Chrome y Safari (origen opaco), así que la página lo detecta
(`location.protocol === 'file:'`) y degrada sola a conmutación de `<audio src>`
sin crossfade, con un aviso visible en pantalla que recuerda el comando de arriba.
Sigue siendo utilizable para enseñar algo en una emergencia, pero sin la
interpolación continua del eje de color.

No copies los 21 MB de audio en base64 dentro del HTML/JS: es peor que servir la
carpeta (carga todo de golpe, nada de decodificación bajo demanda). Los ficheros
`.opus` viven en `demo_fusion/audio/` (no versionado en git, ~21 MB, regenerable
con `PYTHONPATH=. .venv/bin/python scripts/42_fusion_set.py`).

## Los dos ejes: NO son intercambiables

- **`color_mix` (9 valores grabados, 0 a 1): eje continuo, con crossfade real.**
  El slider se mueve de forma continua y la página reproduce simultáneamente los
  2 clips vecinos de la rejilla con ganancias interpoladas (2 `BufferSourceNode` +
  2 `GainNode`, igual que `web/pad2d.js` pero en 1-D). Está verificado en
  `impossible_mix/physics/blend.py` (bloque ~938-980) que la contribución por
  banda `parts[k]` NO depende de `color_mix` -- solo cambian los pesos
  `band_w` -- así que mezclar linealmente dos salidas vecinas da otro miembro
  EXACTO de la misma familia de chimeras, sin interferencia de fase.
- **`n_bands` (8 valores: 2, 4, 6, 8, 12, 16, 24, 32): eje discreto, con conmutación.**
  NUNCA se interpola entre dos valores de `n_bands`: cambia el banco de filtros
  entero, así que mezclar dos sonaría a dos chimeras simultáneas, no a una
  intermedia. El slider tiene muescas (`<datalist>` + los 8 números reales
  siempre visibles bajo la barra). Al conmutar, la página preserva la posición
  de reproducción (el offset dentro del bucle de 6 s) y aplica ~20 ms de fundido
  **solo como antichasquidos** -- eso NO es interpolación entre familias.

## Honestidades (norma del proyecto: lo que se sabe que es aproximado, se dice)

1. **El crossfade interpola aritméticamente los pesos de ganancia; `color_mix`
   es una interpolación geométrica (dominio logarítmico, ver
   `blend.py:966-967`: `exp((1-color_mix)*log(rms_a) + color_mix*log(rms_b))`).
   Una posición intermedia del slider de color NO equivale al `color_mix` que su
   etiqueta numérica sugeriría.** Por eso el eje se etiqueta "color: dinámica ↔
   materia" (una descripción cualitativa) y el número exacto de `color_mix`
   solo se muestra cuando el slider está, con tolerancia de punto flotante,
   sobre una de las 9 posiciones realmente grabadas; en cualquier otra posición
   la página muestra "crossfade entre X e Y (aritmético, no es un color_mix
   real)".
2. **Las 4 referencias (`baseline_v11`, `suma_ancla`, `sin_alinear`, `plana`)
   quedan fuera de la ruta de crossfade.** Son botones A/B independientes, no
   posiciones de slider: la igualación de sonoridad les aplica a veces un
   limitador no lineal (`metodo_igualacion="soft_limit"`, ver más abajo) por su
   cresta patológica -- de hecho `baseline_v11` de `trueno_hecho_de_agua` y de
   `trueno_hecho_de_canica` llegan a ~47 dB de cresta, el síntoma medido de "dos
   capas" que este trabajo intenta corregir. La afirmación de exactitud del
   punto anterior se hace SOLO sobre la rejilla principal (432 puntos), nunca
   sobre estas 4 referencias por pareja (24 puntos).
3. **Dentro de la propia rejilla principal, 5 celdas rompen la exactitud del
   crossfade.** La propiedad de "combinación lineal exacta, sin interferencia
   de fase" (honestidad 1) depende de que la igualación de sonoridad de AMBOS
   clips vecinos haya sido una escala lineal (`metodo_igualacion` "linear" o
   "linear_capped": `audio_final = factor * audio_crudo`, un escalar puro que
   preserva la forma de onda). En el manifiesto real hay 5 celdas de
   `trueno_hecho_de_canica` con `metodo_igualacion="soft_limit"` (compresión
   `tanh`, no lineal, porque su cresta cruda era demasiado patológica para el
   recorte lineal):
   - `trueno_hecho_de_canica`: n_bands=12, color_mix=0.875
   - `trueno_hecho_de_canica`: n_bands=12, color_mix=1.000
   - `trueno_hecho_de_canica`: n_bands=16, color_mix=1.000
   - `trueno_hecho_de_canica`: n_bands=24, color_mix=1.000
   - `trueno_hecho_de_canica`: n_bands=32, color_mix=1.000
   Cuando el punto activo o su vecino de crossfade cae en una de estas celdas,
   `fusionpad.js` lo detecta en tiempo real (compara `metodo_igualacion` del
   vecino contra `"soft_limit"`) y muestra un aviso junto al slider de color.
   El resto de la rejilla (427 de las 432 celdas) sí cumple la exactitud sin
   reservas.

## Restricción de memoria: decodificación bajo demanda, no precarga

72 clips de rejilla x 6 s x 44 100 Hz x 4 bytes (Float32 decodificado) ≈ 76 MB
por pareja; las 6 parejas a la vez serían ~457 MB. `fusionpad.js`
(`FusionPad1D`) decodifica solo los 2 clips vecinos que hacen falta para la
posición actual de los sliders, cachea por pareja (`Map` fichero→AudioBuffer)
y VACÍA ese cache al cambiar de pareja (`loadPareja` llama a `buffers.clear()`).
"Vaciar" aquí significa soltar las referencias -- JavaScript no tiene
liberación explícita de memoria, así que lo que se garantiza es que el código
ya no retiene el `AudioBuffer`; el recolector de basura del navegador hace el
resto en su propio momento, no de forma instantánea. **No precargar las 6
parejas de golpe "para que vaya más fluido"**: es exactamente el patrón que
esta restricción está pensada para evitar, y en un portátil modesto de sala
de congreso puede ser la diferencia entre que la demo funcione o se cuelgue.

## Panel A/B

Junto a los sliders, cuatro botones (`<audio controls>` nativos, funcionan
igual en modo servidor y en modo `file://`) reproducen las 4 referencias de la
pareja activa, con su `crest_db` y `SSO` del manifiesto al lado. Los "vecinos"
del punto activo de la rejilla (los dos clips que se están crossfadeando)
muestran sus propios `crest_db`/`SSO` debajo de los sliders para comparar en
la sala, con el sonido delante: la cresta alta es el síntoma medido de "dos
sonidos superpuestos" en vez de un objeto fundido. No se muestra un número de
cresta "del blend": el crest factor no es un promedio ponderado simple de sus
dos extremos (es una métrica no lineal sobre la forma de onda), así que
inventar uno sería falsa precisión -- se muestran los dos valores reales
medidos en los extremos que se están mezclando.

## Verificación

```
PYTHONPATH=. .venv/bin/python scripts/43_demo_fusion_web.py --check
```
Sin navegador: valida que `index.html`/`fusionpad.js` en disco NO están
desactualizados respecto a `manifest.json` (recalcula el índice y lo compara
byte a byte con el que hay horneado en `fusionpad.js`), que los 456 clips
referenciados existen en `demo_fusion/audio/`, que los 6 selectores de pareja
resuelven, que las 4 referencias están en cada una, y que no hay ninguna URL
externa en el HTML ni en el JS.

## Convenciones

Español en la interfaz; sin tildes en identificadores. Determinismo: el HTML y
el JS generados dependen solo de `manifest.json`, nunca de aleatoriedad ni del
reloj.
