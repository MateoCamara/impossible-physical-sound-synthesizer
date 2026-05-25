# Paper outline — Tecniacústica 2026 (4-8 páginas, 7 sept 2026)

**Título**: How does a rolling droplet sound? Un marco paramétrico físicamente informado para la síntesis de sonidos imposibles

## 1. Introducción (~3/4 pag)
- Problema: el diseño Foley requiere combinaciones materia-interacción que no existen físicamente (gota rodante, roca líquida, paso sobre lava).
- Limitación de los enfoques generativos neurales (codecs preentrenados como EnCodec): el espacio latente captura identidad acústica pero **no soporta ejes semánticos de material/interacción**; manipular la media latente aplasta la dinámica, manipular la secuencia preserva calidad pero no produce transformaciones perceptuales coherentes (los oyentes no escuchan "más líquido" al mover hacia el centroide de líquido).
- Propuesta: marco paramétrico físicamente informado por capas modulares, con knobs de **significado físico directo**. Caso paradigmático: la gota rodante.
- Contribución: (i) primitivas físicas (modal, fricción, granular, drip/burbuja), (ii) composer genérico (material, interacción, modificadores), (iii) controlador con knobs continuos, (iv) caso de estudio de la gota rodante + dos combinaciones imposibles adicionales.

## 2. Trabajo relacionado (~1/2 pag)
- Síntesis modal y modelos físicos para SFX (van den Doel, Avanzini, Cook).
- Síntesis de líquidos: Minnaert resonance para burbujas (Drumm 2010), splash/pour modeling.
- Modelos friction para scrape/drag (Avanzini-Crosato, Serafin).
- Generative SFX neural: comparación cualitativa con encoders preentrenados (EnCodec, RAVE) y modelos de difusión texto-audio.
- Mix2Morph / SoundMorpher como antecedentes de "sound infusion" y morphing.

## 3. Método (~2 pags)
### 3.1 Primitivas físicas
- **Modal**: banco de resonadores con perfiles materiales (metal, rock, wood, glass, earth, fabric). Parámetros: nº modos, frecuencia fundamental, espaciado armónico, damping (t60), inharmonicidad, forma espectral.
- **Friction**: ruido bandpass modulado por envolvente de velocidad + banco resonante del cuerpo. Parámetros: hardness, roughness, velocidad media+jitter, presión, body resonance+Q.
- **Granular**: nube estocástica de impactos modales cortos. Parámetros: densidad, jitter, tamaño de grano+varianza, energía, dispersión espacial.
- **Drip/burbuja**: pulso + chirp ascendente Minnaert + cola modal de superficie. Parámetros: radio de gota, viscosidad, dureza superficie. Fórmula clave: $f_M = 3.26 / r$ (Minnaert).
- **Splash/pour**: composición de N drips dispersos temporalmente.
- **Rolling droplet**: tren cuasi-periódico de drips con jitter (path roughness) e intensidad variable. Parámetros: radius, viscosity, surface_hardness, roll_velocity, path_roughness.

### 3.2 Composer genérico
- API `compose(material, interaction, modifiers, duration) -> wav`.
- Tabla `(material, interaction) -> generador` para combinaciones convencionales.
- Modificadores comunes (wetness, granularity, rigidity, resonance, continuity) mapeados a parámetros físicos primitivos.
- `compose_impossible(base, overlay, weight)` para sonidos imposibles: capa principal + capa overlay con peso.

### 3.3 Controlador físico con knobs
- API `PhysicsController.sweep(knob, values) -> [steps]`.
- Cada sweep produce variación monotónica medible y perceptualmente coherente.
- Comparación con sweep en latente EnCodec (baseline negativo).

## 4. Caso de estudio: la gota rodante (~1 pag)
- Pregunta-gancho del título: "How does a rolling droplet sound?"
- Descomposición física: tren cuasi-periódico de eventos drip × superficie modal × modulación de velocidad.
- 8 variantes del estímulo (canonical, very_wet, dry_drip, slow_roll, fast_roll, grainy_path, smooth_path, big_droplet).
- Espectrogramas comparados.
- Sweep monotónico de cada parámetro físico → cambio audible.

## 5. Evaluación (~1.5 pag)
### 5.1 Métricas objetivas
- RMS, peak, dynamic range, log-spec distance, spectral flatness, centroid.
- Verdict automático (good/suspicious/garbage) sobre 230+ generaciones del corpus + baselines neurales.
- Comparativa: marco físico = 100% good; baseline EnCodec edición latente = 98% good pero sin cambio perceptual; baseline interp neural = 8% good.
- Tabla maestra: (método × combo × métrica).

### 5.2 Test perceptual (n=16-24)
- 24 estímulos = 3 combos imposibles × 8 variantes.
- Diseño Latin Square: cada oyente escucha 12, balanceado.
- Preguntas: (1) material percibido, (2) interacción percibida, (3) Likert 1-7 "cuán híbrido/imposible", (4) Likert 1-7 "cuán coherente como evento único".
- Análisis: Friedman + Wilcoxon-Holm.

## 6. Resultados (~1.5 pag)
- Sweep monotonía: |Δ_metric / Δ_knob| > 0 en todos los knobs del controlador físico.
- Tabla maestra (objetiva): el marco físico domina en `controllability_score` y `coherence_score` frente a edición latente neural.
- Test perceptual: las variantes del rolling_droplet se ordenan correctamente por % de oyentes que identifican "más liquid", "más grano", etc.
- Figura UMAP/PCA del espacio paramétrico físico para los 24 estímulos.

## 7. Discusión (~3/4 pag)
- Limitaciones: el marco es **deliberadamente paramétrico** — no aprende de datos. Esto es ventaja (reproducible, interpretable) y limitación (no escala a categorías nuevas sin ingeniería del generador).
- Comparación abierta: los modelos de difusión text-to-audio (AudioLDM2, Stable Audio Open) producen ejemplos plausibles pero opacos y no reproducibles; el marco físico complementa estos sistemas como capa de control.
- Aplicación inmediata: diseño Foley para animación/videojuego, prototipado rápido, accesibilidad sonora.

## 8. Conclusión (~1/4 pag)
- Síntesis paramétrica físicamente informada es viable y produce sonidos imposibles con control interpretable.
- La gota rodante es un caso pedagógico que demuestra el marco completo.
- Trabajo futuro: aprendizaje inverso de parámetros físicos desde audio real (puente a la rama neural).

## Figuras objetivo (5 figuras + 2 tablas)
- F1: arquitectura del composer (bloques).
- F2: descomposición de un evento drip (impulso + chirp + tail).
- F3: tren rolling droplet con espectrograma marcado por eventos.
- F4: comparativa espectrogramas físico vs EnCodec edición vs interp neural (negativo).
- F5: boxplot Likert por knob/variante.
- T1: métricas objetivas por método.
- T2: resultados perceptuales.

## Huecos a cerrar entre 29 mayo y 7 sept
- [ ] Test perceptual a n=24+ (vs piloto n=16 del abstract).
- [ ] Modelos de difusión como comparación texto-audio.
- [ ] Aprendizaje inverso: estimar parámetros físicos desde audio real con las cabezas ya entrenadas.
- [ ] Generalizar más combinaciones imposibles (paso sobre lava, viento de madera, etc.).
- [ ] Visualización interactiva del controller para demo en sesión.
