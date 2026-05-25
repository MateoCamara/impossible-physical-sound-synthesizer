"""Taxonomia cerrada de material e interaccion + esquema de etiquetado."""
from __future__ import annotations

from enum import Enum
from typing import Final


class Material(str, Enum):
    WOOD = "wood"
    METAL = "metal"
    ROCK = "rock"
    FABRIC = "fabric"
    EARTH = "earth"
    LIQUID = "liquid"   # nuevo vs FOLEY-VAE 2023
    GRAVEL = "gravel"   # nuevo vs FOLEY-VAE 2023
    OTHER = "other"


class Interaction(str, Enum):
    IMPACT = "impact"
    SCRAPE = "scrape"
    ROLL = "roll"
    DRIP = "drip"
    SPLASH = "splash"
    POUR = "pour"
    DRAG = "drag"
    STEP = "step"


MODIFIERS: Final[tuple[str, ...]] = (
    "wetness",      # 1=seco, 5=empapado
    "rigidity",     # 1=blando, 5=rigido
    "resonance",    # 1=mate, 5=resonante con cola larga
    "granularity",  # 1=continuo, 5=granular/discreto
    "continuity",   # 1=impulso unico, 5=continuo
)

LABELS_COLUMNS: Final[tuple[str, ...]] = (
    "clip_id",
    "source",       # ulfc | freesound | own
    "parent_id",    # null si original, clip_id padre si augmentation
    "material",
    "interaction",
    *MODIFIERS,
    "notes",
)


# ULFC = Ultimate Footstep Collection (corpus base, antes FOLEY-VAE 2023).
# Codigos en filename -> (Material, Interaction, weak modifiers por defecto).
# Cuando un codigo de ULFC no encaja en Interaction (p.ej. JOG/RUN), se mapea
# a STEP con continuity alta; SCRAPE1-5 colapsan a SCRAPE.
ULFC_MATERIAL_MAP: Final[dict[str, Material]] = {
    "ASPH": Material.ROCK,      # asfalto -> roca dura
    "CONC": Material.ROCK,      # hormigon -> roca dura
    "DIAMOND": Material.METAL,  # diamond plate
    "GRATE": Material.METAL,    # rejilla metalica
    "METAL1": Material.METAL,
    "DIRT": Material.EARTH,
    "GRASS": Material.OTHER,    # hierba: hibrido fabric+earth; revisar dia 4
}

ULFC_ACTION_MAP: Final[dict[str, Interaction]] = {
    "WALK": Interaction.STEP,
    "SLOW": Interaction.STEP,    # token aislado en "WALK XX SLOW"
    "JOG": Interaction.STEP,
    "RUN": Interaction.STEP,
    "STOMP": Interaction.IMPACT,
    "LAND": Interaction.IMPACT,
    "STAIR": Interaction.STEP,
    "SCRAPE1": Interaction.SCRAPE,
    "SCRAPE2": Interaction.SCRAPE,
    "SCRAPE3": Interaction.SCRAPE,
    "SCRAPE4": Interaction.SCRAPE,
    "SCRAPE5": Interaction.SCRAPE,
    "SCUFF": Interaction.SCRAPE,
}


# Huecos que el corpus ULFC NO cubre y que dia 2 debe rellenar via Freesound:
#   Materiales: wood, fabric (textil real), liquid, gravel
#   Interacciones: roll, drip, splash, pour, drag
# El espacio "imposible" se construye precisamente cruzando estos huecos
# con los materiales/interacciones del corpus existente.
ULFC_GAPS_MATERIALS: Final[tuple[Material, ...]] = (
    Material.WOOD,
    Material.FABRIC,
    Material.LIQUID,
    Material.GRAVEL,
)
ULFC_GAPS_INTERACTIONS: Final[tuple[Interaction, ...]] = (
    Interaction.ROLL,
    Interaction.DRIP,
    Interaction.SPLASH,
    Interaction.POUR,
    Interaction.DRAG,
)
