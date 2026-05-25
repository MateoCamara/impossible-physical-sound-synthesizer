# Abstract v3 — Tecniacústica 2026 (doble ciego, 250 palabras)

## Título tentativo
**How does a rolling droplet sound? Un marco paramétrico físicamente informado para la síntesis de sonidos imposibles**

## Resumen (≈ 249 palabras)

El diseño sonoro audiovisual demanda combinaciones materia-interacción que no existen en la naturaleza: una gota líquida rodando, un impacto rocoso húmedo, grava raspando bajo agua. Los enfoques generativos basados en codificadores neurales preentrenados, dominantes en síntesis de efectos, permiten manipular embeddings de forma que producen variación medible en métricas espectrales pero raramente se traducen a un cambio que el oyente identifique semánticamente como "más líquido" o "más rocoso": la dirección latente carece de anclaje perceptual.

Este trabajo propone un **marco paramétrico físicamente informado** organizado en capas modulares —síntesis modal de resonadores materiales, fricción de cuerpo y superficie, granular estocástico, eventos de gota con dinámica de Minnaert para burbujas— combinadas por un *composer* genérico que admite especificaciones de la forma `(material, interacción, modificadores)`. Cada parámetro tiene **significado físico directo** (radio de gota en milímetros, viscosidad, dureza de superficie, velocidad de rodadura, rugosidad de trayectoria), garantizando que un control "más X" corresponda a una transformación acústica coherente con dicho parámetro.

Como caso paradigmático presentamos la **gota rodante**, sin referente natural directo, construida como tren cuasi-periódico de eventos drip modulados por velocidad y rugosidad. Generamos 24 estímulos cubriendo tres familias imposibles (gota rodante, impacto rocoso líquido, raspado de grava húmeda). En 75 mediciones (3 escenas × 5 knobs × 5 métricas) **10 de 15 pares (escena, knob)** producen variación monotónica significativa ($|\rho_{Spearman}|>0.7$); los pares no monotónicos corresponden a propiedades no aplicables al material. Un test perceptual piloto contrastará monotonía cuantitativa y monotonía perceptual entre el marco paramétrico y un baseline neural de control. El marco abre vías de diseño sonoro asistido con controles interpretables y reproducibles.

---

## Notas para iteración

- Conteo: ~249 palabras.
- **Cambio v2→v3**: matizada la limitación neural ("monotonía espectral sí, semántica no"), añadida la frase sobre "monotonía cuantitativa vs perceptual" que el test perceptual debe validar. Esto refuerza la necesidad del test perceptual aunque sea piloto.
- **Anti-checklist doble ciego**: revisar antes de enviar — sin "nuestro trabajo previo", sin nombres propios, sin "FOLEY-VAE", sin "Intellimixer".
- Sesiones objetivo: A09-1 (ML/IA en Acústica) o A24-1 (Acústica Virtual).
- Ground truth para juzgar éxito: cuando lleguen ≥16 respuestas perceptuales, sustituir "contrastará" por la cifra real (`identification_acc = X%, impossibility_mean = Y/7`).
