# Decisión de encoder — Día 1

## Hallazgos

1. **No hay checkpoint FOLEY-VAE 2023 disponible**. Solo el PDF del paper en `/home/atorsaev/projects/acreditacion_aneca/.../Paper_2023_arXiv_P13_FoleyVAE.pdf`. El modelo entrenado no está en local.
2. **GPU local no fiable** (sobrecalentamiento). Política: ideación y código aquí; entrenamientos pesados se migran al otro ordenador.
3. **Corpus ULFC** disponible y masivo (~10k chunks), parseado en `labels.csv` con 720 clips balanceados.

## Decisión

**Encoder primario:** EnCodec 24 kHz preentrenado (HuggingFace) → no requiere entrenamiento, corre en CPU, latente continuo de dim 128 sobre `encoder.encoder(x)`.

**Encoder secundario para reentrenar (en otro ordenador):** RAVE v2 oficial sobre el subset de 720 clips ULFC + extensiones Freesound del Día 2. Entrenamiento ~5-8h en GPU media. Se hace cuando tengamos los embeddings de EnCodec validados (Día 3-4).

**Encoder terciario para CLAP-similarity:** LAION-CLAP fused-base preentrenado, usado únicamente como **evaluador** (no para mezcla). Permite calcular `CLAP_sim(audio_hibrido, prompt='rolling drop')` como métrica objetiva.

## Consecuencias para el plan

- Día 2: extracción de embeddings con EnCodec sobre 720 ULFC + ~200 Freesound + ~30 propios → `data/embeddings/encodec.pt`. También CLAP para métricas → `data/embeddings/clap_audio.pt`.
- Día 3: método A se prototipa sobre embedding EnCodec.
- Día 4-5: si EnCodec produce mezclas defendibles, no necesitamos RAVE. Si suenan a artefactos de quantización, lanzamos RAVE en otro ordenador.
- Decisión irrevocable: FAD computado con backbone VGGish (no CLAP-FAD ni EnCodec-FAD) por consistencia con literatura previa.
