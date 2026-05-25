# Abstract v1 — Tecniacústica 2026 (doble ciego, 250 palabras)

## Título tentativo
**How does a rolling droplet sound? Un marco paramétrico físicamente informado para la síntesis de sonidos imposibles**

## Resumen (250 palabras)

El diseño sonoro audiovisual demanda combinaciones materia-interacción que no existen en la naturaleza —una gota líquida rodando, un impacto rocoso húmedo, grava raspando bajo agua—. Los enfoques generativos basados en codificadores neurales preentrenados, dominantes en síntesis de efectos sonoros, tienden a aplastar la dinámica temporal al manipular el espacio latente y, sobre todo, no soportan **control perceptual interpretable**: mover el embedding hacia el "centroide de líquido" rara vez produce un cambio que el oyente identifique como "más líquido".

Este trabajo propone un **marco paramétrico físicamente informado** organizado en capas modulares —síntesis modal de resonadores materiales, fricción de cuerpo y superficie, granular estocástico, eventos de gota con dinámica de Minnaert para burbujas— combinadas por un *composer* genérico que admite especificaciones de la forma `(material, interacción, modificadores)`. Cada parámetro tiene **significado físico directo** (radio de gota en milímetros, viscosidad, dureza de superficie, velocidad de rodadura, rugosidad), garantizando que un control `+más X` corresponda a una transformación acústica coherente.

Como caso de estudio paradigmático presentamos la **gota rodante**, un sonido sin referente natural construido como tren cuasi-periódico de eventos drip moduladas por velocidad y rugosidad. Generamos 24 estímulos cubriendo tres familias imposibles (gota rodante, impacto rocoso líquido, raspado de grava húmeda) en ocho variantes cada una. Métricas objetivas (RMS, rango dinámico, distancia log-espectral, distancia a centroides en espacios latentes neurales de referencia) confirman variación monotónica con cada knob. Un test perceptual piloto (n≥16) evalúa coherencia perceptual y "imposibilidad" subjetiva. El marco abre vías de diseño sonoro asistido con controles acústicamente interpretables y reproducibles, complementando los enfoques generativos basados en embeddings opacos.

---

## Notas para iteración (no incluir en envío)

- **Conteo aproximado**: 247 palabras.
- **Anti-checklist doble ciego**: revisar antes de enviar — sin "nuestro trabajo previo", sin nombres propios, sin "FOLEY-VAE", sin "Intellimixer".
- **Cita externa permitida en el cuerpo**: "los enfoques generativos basados en codificadores neurales preentrenados" (suficientemente vago).
- **Sesión preferida**: A09-1 (ML/IA en Acústica) o A24-1 (Acústica Virtual). El término "physically informed" se reconoce en ambas.
- **Pivotes vs propuesta inicial**:
  - El enfoque latente neural (FOLEY-VAE, RAVE, EnCodec) ya no es el eje del paper sino una **comparación de referencia**.
  - "How does a rolling droplet sound?" es la frase-gancho. Funciona como subtítulo o sección.
  - El método físico garantiza control perceptual; los embeddings neurales se mencionan como motivación/baseline negativa.
