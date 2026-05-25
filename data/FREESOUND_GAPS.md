# Gaps que Día 2 debe cubrir con Freesound + grabación propia

El subset ULFC (`labels.csv`, 720 clips) cubre **{rock, metal, earth, other} × {step, impact, scrape}** — todo material sólido y seco, interacciones percusivas/de fricción. Para construir el espacio "imposible" necesitamos cruzar con los siguientes huecos.

## Materiales ausentes (objetivo ~40-50 clips cada uno)

| Material | Términos de búsqueda Freesound | Notas |
|---|---|---|
| **liquid** | `water drop`, `water splash`, `pouring water`, `water hitting`, `liquid impact` | Imprescindible — eje del "wet" |
| **gravel** | `gravel walking`, `gravel rolling`, `pebbles falling`, `crunch gravel` | Granularidad alta |
| **wood** | `wood impact`, `wood knock`, `wood plank fall`, `wooden footstep` | Resonancia media |
| **fabric** | `cloth rustle`, `fabric drag`, `silk slide`, `cotton movement` | Granularidad baja, rigidez baja |

## Interacciones ausentes (objetivo ~30-40 clips cada una)

| Interacción | Términos Freesound | Materiales con los que combinar |
|---|---|---|
| **roll** | `ball rolling`, `marble rolling`, `barrel roll`, `stone rolling` | metal, rock, wood (ancla "rolling drop" = roll × liquid imposible) |
| **drip** | `single drop`, `water dripping`, `dripping faucet` | liquid (puro) — anclaje del eje wetness |
| **splash** | `water splash`, `puddle splash`, `liquid impact` | liquid + cualquier sólido |
| **pour** | `pouring water`, `pouring sand`, `liquid pouring` | liquid, gravel |
| **drag** | `dragging chair`, `cloth dragging`, `heavy drag` | fabric, wood (paralelo a scrape) |

## Restricciones de descarga

- Duración: 2–8 s (después se recorta a 5 s en `data/processed/`)
- Sample rate ≥ 44.1 kHz
- Licencia: CC0 o CC-BY
- Mono o convertible (descartar field recordings densos)
- Sin voz, sin música, sin layers obvios

## Grabación propia (30–60 clips, móvil/REAPER)

Cubre interacciones poco frecuentes en Freesound y permite controlar humedad:

- Gota cayendo sobre madera/metal/papel (3 materiales × 5 tomas)
- Granos rodando sobre cartón (10 tomas)
- Frotar dedo mojado vs seco sobre cristal/madera (5 + 5)
- Verter agua/azúcar/arroz (3 × 3)
- Bola pequeña rodando sobre superficies (10)

## Combinaciones objetivo "imposibles" para evaluación

Las 3 anclas obligatorias (luego se amplía día 7):

1. **`rolling drop`** = roll + liquid (eje wet con estructura temporal de rodadura)
2. **`liquid rock impact`** = impact + rock × liquid (impacto duro humedecido)
3. **`wet gravel scrape`** = scrape + gravel × liquid (granularidad continua mojada)

Estas 3 son las que evaluamos en Día 3 (primer lote) y aparecen en abstract.
