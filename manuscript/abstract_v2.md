# Abstract v2 — Tecniacústica 2026 (doble ciego, 250 palabras)

## Título tentativo
**How does a rolling droplet sound? Un marco paramétrico físicamente informado para la síntesis de sonidos imposibles**

## Resumen (≈ 250 palabras)

El diseño sonoro audiovisual demanda combinaciones materia-interacción que no existen en la naturaleza: una gota líquida rodando, un impacto rocoso húmedo, grava raspando bajo agua. Los enfoques generativos basados en codificadores neurales preentrenados, dominantes en síntesis de efectos, capturan identidad acústica pero no soportan **control perceptual interpretable**: mover el embedding hacia el "centroide de líquido" rara vez produce un cambio que el oyente identifique como "más líquido".

Este trabajo propone un **marco paramétrico físicamente informado** organizado en capas modulares —síntesis modal de resonadores materiales, fricción de cuerpo y superficie, granular estocástico, eventos de gota con dinámica de Minnaert para burbujas— combinadas por un *composer* genérico que admite especificaciones de la forma `(material, interacción, modificadores)`. Cada parámetro tiene **significado físico directo** (radio de gota en milímetros, viscosidad, dureza de superficie, velocidad de rodadura, rugosidad de la trayectoria), garantizando que un control "más X" corresponda a una transformación acústica coherente.

Como caso paradigmático presentamos la **gota rodante**, sin referente natural directo, construida como tren cuasi-periódico de eventos drip modulados por velocidad y rugosidad. Generamos 24 estímulos cubriendo tres familias imposibles (gota rodante, impacto rocoso líquido, raspado de grava húmeda), ocho variantes cada una. Sobre 75 mediciones (3 escenas × 5 knobs × 5 métricas acústicas), **10 de 15 pares (escena, knob) producen variación monotónica significativa** ($|\rho_{Spearman}|>0.7$); los pares no monotónicos corresponden a propiedades físicamente no aplicables (e.g., resonancia en grava). Un test perceptual piloto (n≥16) evalúa coherencia y "imposibilidad" subjetiva. El marco abre vías de diseño sonoro asistido con controles acústicamente interpretables y reproducibles, complementando los enfoques generativos opacos.

---

## Notas para iteración

- Conteo: ~248 palabras.
- **Anti-checklist doble ciego**: revisar antes de enviar — sin "nuestro trabajo previo", sin nombres propios, sin "FOLEY-VAE", sin "Intellimixer".
- Cambio v1→v2: insertado el dato concreto "10 de 15 pares producen variación monotónica significativa (|ρ|>0.7)" para sustentar empíricamente la afirmación de control interpretable.
- Sesiones: A09-1 (ML/IA en Acústica) o A24-1 (Acústica Virtual).
