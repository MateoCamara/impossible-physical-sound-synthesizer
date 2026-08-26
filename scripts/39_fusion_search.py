"""Barrido de configuraciones de fusion (v12-F4): rankea por FCI dentro de
cada pareja de padres, guarda TODOS los candidatos con sus metricas a CSV
(hoy esos numeros solo viven en stdout de las iteraciones V8-V11) y sirve
un top-5 corto + baseline + suma-ancla a escucha humana, con RMS igualado.

Sigue el patron de scripts/37_chimera_catalog.py y scripts/38_demo_congreso.py:
script numerado, --check con smokes, determinismo con seeds fijas.

Uso:
    PYTHONPATH=. .venv/bin/python scripts/39_fusion_search.py --check
    PYTHONPATH=. .venv/bin/python scripts/39_fusion_search.py --pair trueno_hecho_de_agua
    PYTHONPATH=. .venv/bin/python scripts/39_fusion_search.py            # barrido completo
    PYTHONPATH=. .venv/bin/python scripts/39_fusion_search.py --freeze   # estimulos T5

Notas de diseno importantes (repetidas en el informe de la tarea):

- El grid de la etapa 1 es el PRODUCTO CARTESIANO COMPLETO de los 4 ejes que
  pide el brief (align x n_bands x color_mix x warp = 2x8x5x2 = 160 celdas
  de method="chimera" por pareja), no un subconjunto. El brief estima "~100
  celdas" en la seccion de coste, pero esa es una aproximacion de esa misma
  seccion, no una restriccion del grid -- se sigue la lectura literal de los
  4 ejes.
- render_fusion() SIEMPRE re-mide register_hz (2-3 veces por render con
  align=True) por diseno documentado en fusion_chain.py (para detectar
  alineaciones fallidas en silencio). Medido en vivo: ~0.9s por llamada, y
  con align=True fijo dentro de una pareja el valor medido es SIEMPRE el
  mismo (el objetivo depende solo del padre-ancla, no de n_bands/color_mix/
  warp). Sin mitigacion, esto solo ya cuesta ~3-4 min por pareja SOLO en
  registro redundante. Este script instala un memo por identidad de objeto
  sobre fusion_chain.register_hz (ver _register_hz_memoized): NO toca
  fusion_chain.py en disco, es reversible, y esta verificado bit-identico
  en --check. Igual para analysis._sso (0.9s/llamada, constante dentro de
  una pareja para cada align on/off): ver _composite_fusion_fast.
- La reconstruccion de (parent_a, parent_b_aligned, parent_b_original) para
  composite_fusion (_metric_parents) reutiliza fusion_chain._cached_parent
  (privada por convencion, publica de facto entre los ficheros de
  impossible_mix.physics -- mismo patron que fusion_chain.py reusa
  blend_recipes._render_drips) en vez de reimplementar la logica de
  alineacion: usa el MISMO cache y el moved_parent/register_target_hz que
  render_fusion ya calculo, asi que nunca puede divergir de lo que
  realmente entro en el blend.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import impossible_mix.physics.fusion_chain as _fc_module  # noqa: E402
from impossible_mix.physics.analysis import (  # noqa: E402
    CompositeFusionReport, _band, _bri, _dop, _mci, _sso, band_envelopes,
    composite_fusion, crest_factor_db,
)
from impossible_mix.physics.blend_recipes import (  # noqa: E402
    CHIMERA_PARENTS_V3, chimera_bands_heuristic, chimera_parent_v3, chimera_parent_v4,
)
from impossible_mix.physics.fusion_chain import (  # noqa: E402
    PHYSICAL_BODIES, FusionSpec, _cached_parent, render_fusion,
)
from impossible_mix.utils import save_wav  # noqa: E402

SR = 44_100
SEED = 42
DUR_GRID = 6.0
OUT_RESULTS = Path("results/fusion_search")
OUT_LISTEN = Path("escucha_AB/fusion_v12")
OUT_FREEZE = Path("perceptual_test/stimuli_fusion")
OUT_FREEZE_MANIFEST = Path("perceptual_test/fusion_manifest.csv")

PAREJAS_V12 = [
    ("trueno_hecho_de_agua",    "trueno", "goteo"),
    ("fuego_hecho_de_vidrio",   "fuego",  "vidrio"),
    ("canica_hecha_de_fuego",   "canica", "fuego"),
    ("trueno_hecho_de_canica",  "trueno", "canica"),
    ("oceano_hecho_de_campana", "oceano", "campana_tela"),
    ("goteo_hecho_de_campana",  "goteo",  "campana_tela"),
]

N_BANDS_GRID = (2, 4, 6, 8, 12, 16, 24, 32)
COLOR_MIX_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
VOCODER_N_BANDS = (8, 12, 16)
ORPHAN_FLOOR_DB = -25.0
METRIC_N_BANDS = 12  # fijo (default de composite_fusion), NO el n_bands del
                     # candidato: mide coherencia con una resolucion constante
                     # y comparable entre TODAS las filas, sea cual sea su
                     # propio n_bands de fusion.

# ====================================================================
# RULING del coordinador (2026-08-26, tras revisar el pilotaje): el FCI
# esta ANTICORRELADO con el objetivo. Medido en el CSV real del pilotaje de
# trueno_hecho_de_agua (171 candidatos del pool): corr(fci, crest_db) =
# +0.68 -- el FCI premia la cresta alta, que es el sintoma medido del
# defecto de "dos capas" (audio sano 10-20 dB; la config rota de la demo
# V11 da 49.4 dB). Entre los candidatos de cresta sana (<22 dB), el que
# GANA por FCI es method="suma" (fci=0.208) -- la propia ancla de "dos
# sonidos superpuestos" le gana a TODAS las chimeras genuinas de esa franja
# (la mejor chimera sana solo llega a fci=0.116). Diagnostico: MCI (peso
# 0.55) premia "que las bandas respiren juntas" en 2-16 Hz, y eso lo
# consigue tanto una chimera que funde de verdad como una suma con
# envolvente compartida o una chimera patologica (e_norm~=1 plano en las
# bandas sin A) -- el mismo fallo de saturacion que ya hundio a
# stream_unity_index en V8 (avisado en su propio docstring).
#
# Consecuencia: el FCI y sus 4 componentes SIGUEN calculandose y
# persistiendose integros en el CSV -- es un resultado NEGATIVO citable
# para el paper -- pero DEJAN de decidir que se exporta a escucha_AB. La
# seleccion del lote de escucha usa en su lugar: (a) un filtro DURO de
# cresta sana, (b) cobertura de DIVERSIDAD del espacio de parametros entre
# los supervivientes (ver _select_diverse), no una metrica. El smoke3
# (auto-calibracion intra-pareja, --check) YA habia detectado sintomas de
# esto antes de este diagnostico: n_bands=4 (validado de oido) quedo
# ULTIMO (8/8) en el ranking FCI de trueno_hecho_de_agua, y n_bands=16
# (validado de oido) quedo ULTIMO (8/8) en fuego_hecho_de_vidrio -- el
# smoke fallo limpio, sin que se tocaran los pesos para forzarlo a pasar
# (ver informe de la tarea).
CREST_HEALTHY_DB = 25.0  # filtro duro: por encima de esto, cresta patologica.
N_LISTEN_DIVERSE = 8  # candidatos por diversidad; + baseline_v11 + suma_ancla
                      # (SIEMPRE incluidos como referencias fijas) = 10
                      # clips/pareja, el extremo superior del rango 8-10
                      # que pidio el coordinador ("mas no se escucha bien").

# ====================================================================
# GANADORES_V12: se rellena tras la escucha de escucha_AB/fusion_v12/
# ====================================================================
# Formato: lista de EXACTAMENTE 3 tuplas (pareja, FusionSpec) -- una por
# cada pareja del mini-test T5 (3 parejas x {suma, ganadora} = 6 estimulos).
# `pareja` debe ser uno de los nombres de PAREJAS_V12; el FusionSpec debe
# tener env_parent/fine_parent coherentes con esa pareja. Ejemplo (NO
# activo, solo ilustrativo del formato):
#
#   GANADORES_V12 = [
#       ("trueno_hecho_de_agua", FusionSpec(
#           env_parent="trueno", fine_parent="goteo", n_bands=4,
#           align=True, warp=True, color_mix=0.25, method="chimera",
#           duration_s=6.0, seed=42)),
#       ("fuego_hecho_de_vidrio", FusionSpec(...)),
#       ("oceano_hecho_de_campana", FusionSpec(...)),
#   ]
#
# Se entrega VACIA a proposito: los ganadores los elige el usuario de oido
# sobre escucha_AB/fusion_v12/. --freeze con la constante vacia falla con
# un mensaje claro (ver freeze()), no con un traceback.
GANADORES_V12: list[tuple[str, FusionSpec]] = []


# ====================================================================
# Memoizacion de fusion_chain.register_hz (ver docstring del modulo)
# ====================================================================
_REGISTER_HZ_MEMO: dict = {}
_ORIG_REGISTER_HZ = _fc_module.register_hz


def _register_hz_memoized(w, sr, n_bands=24, lo=60.0, hi=8000.0):
    """Envoltorio de memo sobre fusion_chain.register_hz (parchea el
    ATRIBUTO del modulo importado, no el fichero fusion_chain.py en disco).

    register_hz es puro (sin estado, sin aleatoriedad): memorizar por
    identidad de objeto (id(w)) es seguro mientras el array siga vivo, y en
    este barrido SIEMPRE sigue vivo (vive dentro de `cache`, que se
    mantiene durante toda una pareja). Ver _check_register_hz_memo() en
    --check para la verificacion de bit-identidad frente al register_hz
    original.
    """
    key = (id(w), w.shape, sr, n_bands, lo, hi)
    if key not in _REGISTER_HZ_MEMO:
        _REGISTER_HZ_MEMO[key] = _ORIG_REGISTER_HZ(w, sr, n_bands=n_bands, lo=lo, hi=hi)
    return _REGISTER_HZ_MEMO[key]


_fc_module.register_hz = _register_hz_memoized


# ====================================================================
# Reconstruccion de padres para la metrica + FCI memoizado en SSO
# ====================================================================

def _metric_parents(spec: FusionSpec, sr: int, cache: dict, meta: dict):
    """(parent_a, parent_b_aligned, parent_b_original) tal y como los uso
    render_fusion para ESTE candidato -- reconstruidos reusando el mismo
    `cache` (via fusion_chain._cached_parent, mismas claves) y el
    moved_parent/register_target_hz que render_fusion ya devolvio en meta,
    para que nunca puedan divergir del audio que realmente entro en el
    blend. parent_b_aligned NO incluye el warp (solo la alineacion de
    registro): composite_fusion.__doc__ define b_aligned como "el B
    re-renderizado en el registro de A si hubo alineacion", no como el B
    warpeado."""
    seed_env = spec.seed
    seed_fine = spec.seed + 17
    a = _cached_parent(cache, spec.env_parent, spec.duration_s, seed_env, sr, None, 1.0)
    b_original = _cached_parent(cache, spec.fine_parent, spec.duration_s, seed_fine, sr, None, 1.0)
    b_aligned = b_original
    moved_parent = meta.get("moved_parent")
    if moved_parent is not None:
        moved_seed = seed_fine if moved_parent == spec.fine_parent else seed_env
        moved_audio = _cached_parent(cache, moved_parent, spec.duration_s, moved_seed, sr,
                                     meta["register_target_hz"], 1.0)
        if moved_parent == spec.fine_parent:
            b_aligned = moved_audio
        else:
            a = moved_audio
    n = min(len(a), len(b_aligned))
    return a[:n], b_aligned[:n], b_original


def _composite_fusion_fast(blend: np.ndarray, parent_a: np.ndarray,
                           parent_b_aligned: np.ndarray, parent_b_original: np.ndarray,
                           sr: int, sso_cache: dict,
                           n_bands: int = METRIC_N_BANDS) -> CompositeFusionReport:
    """Igual que analysis.composite_fusion(), pero memoiza _sso por
    (id(parent_a), id(parent_b_aligned)): SSO no depende del blend, solo de
    los padres, y con METRIC_N_BANDS fijo hay como mucho 2 pares (a,b)
    distintos por pareja (align False/align True) -- medido en vivo, _sso
    cuesta ~0.9s por llamada (band_envelopes x2), el componente MAS caro de
    los 4; sin memo se recalcularia ~170 veces por pareja el MISMO numero.

    Reimplementa la formula del FCI (0.55*mci - 0.30*dop - 0.15*bri) en vez
    de llamar a composite_fusion() directamente -- duplicacion deliberada,
    verificada bit-a-bit contra composite_fusion() en
    _check_metric_fast_matches_reference() (--check): si analysis.py
    cambia la formula sin que este fichero se actualice, ese smoke lo
    detecta.
    """
    mci = _mci(blend, sr, n_bands)
    key = (id(parent_a), id(parent_b_aligned), n_bands)
    if key not in sso_cache:
        sso_cache[key] = _sso(parent_a, parent_b_aligned, sr, n_bands)
    sso = sso_cache[key]
    dop = _dop(blend, sr)
    bri = _bri(blend, parent_a, parent_b_original, sr)
    fci = 0.55 * mci - 0.30 * dop - 0.15 * bri
    return CompositeFusionReport(mci=mci, sso=sso, dop=dop, bri=bri, fci=fci)


# ====================================================================
# Diagnosticos adicionales pedidos por el brief
# ====================================================================

def _count_orphan_bands_a(parent_a_used: np.ndarray, sr: int, n_bands: int,
                          floor_db: float = ORPHAN_FLOOR_DB) -> int:
    """Cuantas de las n_bands del filterbank de CHIMERA (lo=80, hi=8820,
    los mismos limites que auditory_chimera/auditory_chimera_colored) tienen
    a parent_a por debajo de floor_db relativo a su banda mas fuerte -- la
    medida directa del mecanismo de "dos capas" (esas bandas emiten B crudo,
    ver fusion_chain.py modulo doc)."""
    envs, _ = band_envelopes(parent_a_used, sr, n_bands=n_bands, lo=80.0, hi=8820.0)
    rms = np.sqrt((envs.astype(np.float64) ** 2).mean(axis=1))
    peak_rms = float(rms.max()) + 1e-18
    db_rel = 20.0 * np.log10((rms + 1e-18) / peak_rms)
    return int(np.sum(db_rel < floor_db))


def _band_rms_db(w: np.ndarray, sr: int, band: tuple[float, float]) -> float:
    x = _band(w, sr, band)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return 20.0 * np.log10(rms + 1e-12)


# ====================================================================
# Igualacion de sonoridad (RMS, con techo de pico) para escucha humana
# ====================================================================
_LISTEN_TARGET_RMS = 0.04  # ~ -28 dBFS: a crest ~28 dB (tipico sano en este
                           # dominio, ver progress.md) el pico queda en
                           # ~0.95, justo en el techo -- margen deliberado.
_LISTEN_PEAK_CEILING = 0.95
_SOFT_LIMIT_DB_TOLERANCE = 6.0  # si el recorte lineal (lo que pide el
                                # brief) deja el RMS final > 6 dB por debajo
                                # del objetivo, se cambia a un limitador
                                # suave en vez de bajar todo el clip (ver
                                # progress.md Ruling 13: sin esto, un clip de
                                # cresta patologica -49.4dB en V11- queda
                                # ~23-28dB por debajo del resto y el oyente
                                # compara volumen, no fusion).


def _match_loudness(w: np.ndarray, target_rms: float = _LISTEN_TARGET_RMS,
                    peak_ceiling: float = _LISTEN_PEAK_CEILING) -> tuple[np.ndarray, dict]:
    """Iguala `w` a target_rms (lineal) con techo de pico peak_ceiling.

    Metodo primario (el que pide el brief): escala TODO el clip a
    target_rms; si el pico se pasa del techo, baja el clip entero (recorte
    lineal proporcional, preserva la forma de onda) y lo anota. Si ese
    recorte lineal deja el RMS final a mas de _SOFT_LIMIT_DB_TOLERANCE dB
    por debajo del objetivo (clips de cresta patologica, ver constante de
    arriba), usa en su lugar un limitador suave (tanh): sube al RMS
    objetivo y comprime SOLO los picos que superan el techo, sin bajar el
    cuerpo del clip. Devuelve (audio, dict-de-diagnostico); el dict entero
    va al CSV (factor_rms_aplicado, metodo_igualacion, pico_recortado,
    rms_final_db_vs_objetivo)."""
    x = w.astype(np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    factor = target_rms / rms
    scaled = x * factor
    peak = float(np.abs(scaled).max())
    if peak <= peak_ceiling:
        rms_final = float(np.sqrt(np.mean(scaled ** 2)))
        return scaled.astype(np.float32), {
            "factor_rms_aplicado": factor, "metodo_igualacion": "linear",
            "pico_recortado": False,
            "rms_final_db_vs_objetivo": 20.0 * np.log10((rms_final + 1e-12) / target_rms),
        }
    factor_cap = factor * (peak_ceiling / peak)
    capped = x * factor_cap
    rms_final_capped = float(np.sqrt(np.mean(capped ** 2)))
    dev_db = 20.0 * np.log10((rms_final_capped + 1e-12) / target_rms)
    if dev_db >= -_SOFT_LIMIT_DB_TOLERANCE:
        return capped.astype(np.float32), {
            "factor_rms_aplicado": factor_cap, "metodo_igualacion": "linear_capped",
            "pico_recortado": True, "rms_final_db_vs_objetivo": dev_db,
        }
    soft = np.tanh(scaled / peak_ceiling) * peak_ceiling
    rms_final_soft = float(np.sqrt(np.mean(soft ** 2)))
    return soft.astype(np.float32), {
        "factor_rms_aplicado": factor, "metodo_igualacion": "soft_limit",
        "pico_recortado": True,
        "rms_final_db_vs_objetivo": 20.0 * np.log10((rms_final_soft + 1e-12) / target_rms),
    }


# ====================================================================
# Descriptor autodescriptivo (nombres de fichero + columna CSV)
# ====================================================================

def _descriptor(spec: FusionSpec) -> str:
    """p.ej. 'align-on__nb8__mix0.50__warp-on' (chimera) o
    'plana__align-off__nb8__warp-off' o 'suma' o 'vocoder__align-off__nb12__warp-off'."""
    parts = []
    if spec.method != "chimera":
        parts.append({"chimera_plana": "plana", "vocoder": "vocoder",
                      "suma": "suma"}[spec.method])
    if spec.method == "suma":
        return "suma"
    parts.append(f"align-{'on' if spec.align else 'off'}")
    parts.append(f"nb{spec.n_bands}")
    if spec.method == "chimera":
        mix = "V11" if spec.color_mix is None else f"{spec.color_mix:.2f}"
        parts.append(f"mix{mix}")
    parts.append(f"warp-{'on' if spec.warp else 'off'}")
    if spec.stats_finish:
        parts.append("stats-on")
    return "__".join(parts)


# ====================================================================
# Grid etapa 1
# ====================================================================

def _grid_candidates(env: str, fine: str) -> list[tuple[str, FusionSpec | None, str]]:
    """(grupo, spec, notas) para la etapa 1 de una pareja env/fine."""
    out: list[tuple[str, FusionSpec | None, str]] = []
    for align, nb, mix, warp in itertools.product(
            (False, True), N_BANDS_GRID, COLOR_MIX_GRID, (False, True)):
        out.append(("chimera_grid", FusionSpec(
            env_parent=env, fine_parent=fine, n_bands=nb, align=align, warp=warp,
            color_mix=mix, method="chimera", duration_s=DUR_GRID, seed=SEED), ""))
    for nb in N_BANDS_GRID:
        out.append(("chimera_plana", FusionSpec(
            env_parent=env, fine_parent=fine, n_bands=nb, align=False, warp=False,
            method="chimera_plana", duration_s=DUR_GRID, seed=SEED), ""))
    nb_heur = chimera_bands_heuristic(env)
    out.append(("baseline_v11", FusionSpec(
        env_parent=env, fine_parent=fine, n_bands=nb_heur, align=False, warp=False,
        color_mix=None, method="chimera", duration_s=DUR_GRID, seed=SEED), ""))
    out.append(("suma_ancla", FusionSpec(
        env_parent=env, fine_parent=fine, align=False, warp=False,
        method="suma", duration_s=DUR_GRID, seed=SEED), ""))
    if fine in PHYSICAL_BODIES:
        for nb in VOCODER_N_BANDS:
            out.append(("vocoder", FusionSpec(
                env_parent=env, fine_parent=fine, n_bands=nb, align=False, warp=False,
                method="vocoder", duration_s=DUR_GRID, seed=SEED), ""))
    else:
        out.append(("vocoder_omitido", None,
                    f"vocoder omitido: fine_parent={fine!r} no tiene BodySpec "
                    f"en PHYSICAL_BODIES ({sorted(PHYSICAL_BODIES)})"))
    return out


# ====================================================================
# CSV: esquema comun a todas las filas (pareja/resumen/round-trip)
# ====================================================================
FIELDNAMES = [
    "pareja", "grupo", "etapa", "descriptor", "notas",
    "env_parent", "fine_parent", "method", "n_bands", "align", "warp",
    "color_mix", "a_floor_db", "stats_finish", "duration_s", "seed",
    "mci", "sso", "dop", "bri", "fci",
    "register_target_hz", "register_achieved_hz", "register_error_hz",
    "register_before_move_hz", "register_clamped", "moved_parent", "align_noop",
    "warp_method", "n_anchors", "n_onsets_matched", "n_onsets_total", "warp_max_rate_dev",
    "parametros_inertes",
    "n_bandas_huerfanas_a", "crest_db", "rms_grave_db", "rms_aguda_db",
    "exportado", "rank_export", "wav_path",
    "factor_rms_aplicado", "metodo_igualacion", "pico_recortado",
    "rms_final_db_vs_objetivo", "rank_etapa1_ref", "render_s",
]


def _empty_row() -> dict:
    return {k: "" for k in FIELDNAMES}


def _spec_to_row(spec: FusionSpec) -> dict:
    return {
        "env_parent": spec.env_parent, "fine_parent": spec.fine_parent,
        "method": spec.method, "n_bands": spec.n_bands, "align": spec.align,
        "warp": spec.warp, "color_mix": "" if spec.color_mix is None else spec.color_mix,
        "a_floor_db": spec.a_floor_db, "stats_finish": spec.stats_finish,
        "duration_s": spec.duration_s, "seed": spec.seed,
    }


def _process_candidate(pareja: str, grupo: str, spec: FusionSpec, notas: str,
                       cache: dict, sso_cache: dict, sr: int = SR) -> dict:
    t0 = time.time()
    w, meta = render_fusion(spec, sr=sr, cache=cache)
    a_used, b_aligned, b_original = _metric_parents(spec, sr, cache, meta)
    rep = _composite_fusion_fast(w, a_used, b_aligned, b_original, sr, sso_cache)
    crest = crest_factor_db(w)
    orphan = "" if spec.method == "suma" else _count_orphan_bands_a(a_used, sr, spec.n_bands)
    row = _empty_row()
    row.update(pareja=pareja, grupo=grupo, etapa=(2 if spec.stats_finish else 1),
              descriptor=_descriptor(spec), notas=notas)
    row.update(_spec_to_row(spec))
    row.update(mci=rep.mci, sso=rep.sso, dop=rep.dop, bri=rep.bri, fci=rep.fci,
              register_target_hz=meta.get("register_target_hz"),
              register_achieved_hz=meta.get("register_achieved_hz"),
              register_error_hz=meta.get("register_error_hz"),
              register_before_move_hz=meta.get("register_before_move_hz"),
              register_clamped=meta.get("register_clamped"),
              moved_parent=meta.get("moved_parent"), align_noop=meta.get("align_noop"),
              warp_method=meta.get("warp_method", ""), n_anchors=meta.get("n_anchors", ""),
              n_onsets_matched=meta.get("n_onsets_matched", ""),
              n_onsets_total=meta.get("n_onsets_total", ""),
              warp_max_rate_dev=meta.get("warp_max_rate_dev", ""),
              parametros_inertes=";".join(meta.get("parametros_inertes", [])),
              n_bandas_huerfanas_a=orphan, crest_db=crest,
              rms_grave_db=_band_rms_db(w, sr, (40.0, 500.0)),
              rms_aguda_db=_band_rms_db(w, sr, (1500.0, 20000.0)),
              render_s=time.time() - t0)
    return row


def _spec_from_row(row: dict) -> FusionSpec:
    """Reconstruye un FusionSpec desde una fila EN MEMORIA de `rows` (con
    tipos Python reales: bool/float/None), no desde una fila releida de un
    CSV en disco (donde todo es str y bool("False") seria un bug: eso NO
    ocurre en este script -- _spec_from_row solo se llama sobre filas que
    acaban de salir de _process_candidate, nunca sobre filas leidas con
    csv.DictReader)."""
    return FusionSpec(
        env_parent=row["env_parent"], fine_parent=row["fine_parent"],
        n_bands=int(row["n_bands"]), align=bool(row["align"]), warp=bool(row["warp"]),
        color_mix=(None if row["color_mix"] == "" else float(row["color_mix"])),
        a_floor_db=float(row["a_floor_db"]), stats_finish=bool(row["stats_finish"]),
        method=row["method"], duration_s=float(row["duration_s"]), seed=int(row["seed"]))


def _fmt_cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return round(v, 6)
    return v


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt_cell(r.get(k, "")) for k in FIELDNAMES})


def _write_resumen(results_dir: Path) -> Path | None:
    csvs = sorted(p for p in results_dir.glob("*.csv") if p.name != "resumen.csv")
    if not csvs:
        return None
    out_path = results_dir / "resumen.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
        writer.writeheader()
        for p in csvs:
            with open(p, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    writer.writerow(row)
    print(f"resumen: {out_path} ({len(csvs)} parejas)")
    return out_path


# ====================================================================
# Seleccion del lote de escucha: filtro de cresta + cobertura de
# diversidad (Ruling del coordinador -- el FCI NO decide esto, ver arriba)
# ====================================================================

def _select_diverse(survivors: list[dict], n_target: int) -> list[dict]:
    """Elige hasta n_target filas de `survivors` maximizando la COBERTURA
    del espacio de parametros (method, n_bands, color_mix, align, warp) en
    vez de rankear por una metrica.

    Greedy determinista de cobertura marginal maxima: en cada paso anade la
    fila SIN elegir que introduce mas valores de eje TODAVIA no
    representados en la seleccion; empates se rompen por el orden de
    entrada (que llega ordenado por align/n_bands/color_mix/warp desde
    _grid_candidates, asi que el resultado es reproducible sin
    aleatoriedad). No es optimo (cobertura maxima es NP-dificil en
    general), pero es simple, determinista y suficiente para el objetivo
    declarado: "que el usuario oiga un abanico representativo, no cinco
    variantes casi identicas"."""
    if len(survivors) <= n_target:
        return list(survivors)

    def axis_tokens(r: dict) -> set:
        return {("method", r["method"]), ("n_bands", r["n_bands"]),
               ("color_mix", r["color_mix"]), ("align", r["align"]),
               ("warp", r["warp"])}

    chosen: list[dict] = []
    covered: set = set()
    remaining = list(range(len(survivors)))
    while len(chosen) < n_target and remaining:
        best_i, best_gain = remaining[0], -1
        for i in remaining:
            gain = len(axis_tokens(survivors[i]) - covered)
            if gain > best_gain:
                best_i, best_gain = i, gain
        chosen.append(survivors[best_i])
        covered |= axis_tokens(survivors[best_i])
        remaining.remove(best_i)
    return chosen


# ====================================================================
# run_pair: etapa 1 + etapa 2 (top-5 FCI, solo para stats_finish) + lote
# de escucha (filtro de cresta + diversidad) + CSV
# ====================================================================

def run_pair(pareja: str, env: str, fine: str, results_dir: Path, listen_dir: Path,
            sr: int = SR, verbose: bool = True) -> dict:
    t_start = time.time()
    cache: dict = {}
    sso_cache: dict = {}
    rows: list[dict] = []

    candidates = _grid_candidates(env, fine)
    for grupo, spec, notas in candidates:
        if spec is None:
            row = _empty_row()
            row.update(pareja=pareja, grupo=grupo, notas=notas, etapa=1)
            rows.append(row)
            continue
        rows.append(_process_candidate(pareja, grupo, spec, notas, cache, sso_cache, sr))
    t_grid = time.time()

    def _num(rs, key):
        out = []
        for r in rs:
            v = r[key]
            if v == "" or v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            out.append(v)
        return out

    chimera_rows = [r for r in rows if r["grupo"] == "chimera_grid"]
    pool_rows = [r for r in rows if r["grupo"] in ("chimera_grid", "chimera_plana", "vocoder")]
    stds = {
        "chimera_grid": {k: (float(np.std(_num(chimera_rows, k)))
                             if _num(chimera_rows, k) else float("nan"))
                         for k in ("mci", "sso", "dop", "bri", "fci")},
        "pool_completo": {k: (float(np.std(_num(pool_rows, k)))
                              if _num(pool_rows, k) else float("nan"))
                         for k in ("mci", "sso", "dop", "bri", "fci")},
    }

    sso_vals = sorted({round(v, 8) for v in _num(pool_rows, "sso")})
    align_noop_true = any(r["align_noop"] is True for r in rows if r["align_noop"] != "")
    n_nan_fci = sum(1 for r in pool_rows
                    if r["fci"] == "" or (isinstance(r["fci"], float) and np.isnan(r["fci"])))

    ranked_fci = sorted(
        (r for r in pool_rows
         if not (r["fci"] == "" or (isinstance(r["fci"], float) and np.isnan(r["fci"])))),
        key=lambda r: r["fci"], reverse=True)
    top5_fci = ranked_fci[:5]

    # --- etapa 2: stats_finish=True SOLO sobre el top-5 POR FCI de la
    # etapa 1 (asi lo pide el brief original; el Ruling del coordinador de
    # mas abajo solo cambia que se EXPORTA a escucha_AB, no esto) ---
    t_stats0 = time.time()
    for i, r in enumerate(top5_fci, 1):
        spec2 = replace(_spec_from_row(r), stats_finish=True)
        row2 = _process_candidate(pareja, "stats_finish_top5", spec2, "", cache, sso_cache, sr)
        row2["rank_etapa1_ref"] = i
        rows.append(row2)
    t_stats1 = time.time()

    # --- lote de escucha: filtro de cresta sana + cobertura de diversidad
    # (Ruling del coordinador -- el FCI ya NO decide esto, ver constantes
    # CREST_HEALTHY_DB / N_LISTEN_DIVERSE arriba: esta anticorrelado con el
    # objetivo, medido corr(fci,crest_db)=+0.68 en este mismo pilotaje) +
    # baseline_v11 y suma_ancla SIEMPRE como referencias fijas ---
    survivors = [r for r in pool_rows if r["crest_db"] != "" and r["crest_db"] < CREST_HEALTHY_DB]
    n_survivors_crest = len(survivors)
    crest_filter_vacio = n_survivors_crest == 0
    if crest_filter_vacio:
        # Ningun candidato paso el filtro: en vez de exportar un lote
        # vacio, se cae a los N_LISTEN_DIVERSE de MENOR cresta (los menos
        # malos), anotado explicitamente para que quede claro que esta
        # pareja no tiene NINGUNA configuracion sana en el grid actual.
        survivors = sorted((r for r in pool_rows if r["crest_db"] != ""),
                           key=lambda r: r["crest_db"])[:N_LISTEN_DIVERSE]
    seleccion = _select_diverse(survivors, N_LISTEN_DIVERSE)

    baseline_row = next(r for r in rows if r["grupo"] == "baseline_v11")
    suma_row = next(r for r in rows if r["grupo"] == "suma_ancla")
    export_targets = [(f"muestra{i}", r) for i, r in enumerate(seleccion, 1)] + [
        ("baseline_v11", baseline_row), ("suma_ancla", suma_row)]
    pareja_dir = listen_dir / pareja
    pareja_dir.mkdir(parents=True, exist_ok=True)
    export_table = []
    for tag, r in export_targets:
        spec_e = _spec_from_row(r)
        w_e, _meta_e = render_fusion(spec_e, sr=sr, cache=cache)
        w9, info = _match_loudness(w_e)
        fname = f"{tag}__{r['descriptor']}.wav"
        path = pareja_dir / fname
        save_wav(path, w9, sr)
        r["exportado"] = True
        r["rank_export"] = tag
        r["wav_path"] = str(path)
        r["factor_rms_aplicado"] = info["factor_rms_aplicado"]
        r["metodo_igualacion"] = info["metodo_igualacion"]
        r["pico_recortado"] = info["pico_recortado"]
        r["rms_final_db_vs_objetivo"] = info["rms_final_db_vs_objetivo"]
        export_table.append({"tag": tag, "descriptor": r["descriptor"], "fci": r["fci"],
                             "mci": r["mci"], "dop": r["dop"], "bri": r["bri"],
                             "crest_db": r["crest_db"], **info, "fname": fname})

    csv_path = results_dir / f"{pareja}.csv"
    _write_csv(csv_path, rows)

    t_end = time.time()
    summary = {
        "pareja": pareja, "n_candidatos_etapa1": len(candidates),
        "n_filas_total": len(rows),
        "t_grid_s": t_grid - t_start, "t_stats_s": t_stats1 - t_stats0,
        "t_total_s": t_end - t_start,
        "std": stds, "sso_valores_unicos": sso_vals,
        "align_noop_alguna_vez": align_noop_true, "n_nan_fci": n_nan_fci,
        "n_chimera_grid": len(chimera_rows), "n_pool": len(pool_rows),
        "n_survivors_crest": n_survivors_crest,
        "crest_filter_vacio": crest_filter_vacio,
        "export_table": export_table, "csv_path": str(csv_path),
    }
    if verbose:
        _print_pilot_summary(summary)
    return summary


def _print_pilot_summary(s: dict) -> None:
    print(f"\n=== {s['pareja']} ===")
    print(f"  candidatos etapa1: {s['n_candidatos_etapa1']} "
         f"(chimera_grid={s['n_chimera_grid']}, pool={s['n_pool']}), "
         f"filas totales (con etapa2): {s['n_filas_total']}")
    print(f"  tiempo grid etapa1: {s['t_grid_s']:.1f}s | "
         f"stats_finish etapa2 (5 candidatos): {s['t_stats_s']:.1f}s | "
         f"total pareja: {s['t_total_s']:.1f}s")
    print("  std por componente (chimera_grid puro / pool completo):")
    for k in ("mci", "sso", "dop", "bri", "fci"):
        print(f"    {k}: {s['std']['chimera_grid'][k]:.4f} / {s['std']['pool_completo'][k]:.4f}")
    print(f"  sso valores unicos en el pool: {s['sso_valores_unicos']} "
         f"(esperado: <=2 por pareja)")
    print(f"  align_noop=True en alguna fila: {s['align_noop_alguna_vez']} (esperado: False)")
    print(f"  fci=NaN en el pool: {s['n_nan_fci']}")
    print(f"  supervivientes filtro cresta<{CREST_HEALTHY_DB}dB: {s['n_survivors_crest']}/"
         f"{s['n_pool']}{'  (VACIO -- fallback a los de menor cresta)' if s['crest_filter_vacio'] else ''}")
    print("  lote de escucha (filtro cresta + diversidad, NO ranking por FCI -- ver Ruling):")
    for e in s["export_table"]:
        print(f"    {e['tag']:12s} fci={e['fci']:.4f} crest_db={e['crest_db']:.1f} "
             f"metodo_igualacion={e['metodo_igualacion']} "
             f"rms_final_db_vs_objetivo={e['rms_final_db_vs_objetivo']:.2f} -> {e['fname']}")
    print(f"  CSV: {s['csv_path']}")


# ====================================================================
# LEEME.md del lote de escucha
# ====================================================================

def _write_leeme(listen_dir: Path, summaries: list[dict]) -> None:
    parts = ["""# Barrido de fusion v12 -- lote de escucha por pareja

## La metrica compuesta (FCI) FALLO como criterio de seleccion -- tu oido es el arbitro

El barrido calcula un indice compuesto (FCI = 0.55*mci - 0.30*dop - 0.15*bri)
pensado para rankear candidatos automaticamente. Al analizar el CSV del
pilotaje (`trueno_hecho_de_agua`, 171 candidatos) resulto que el FCI esta
ANTICORRELADO con lo que buscamos: `corr(fci, crest_db) = +0.68` -- el FCI
premia la cresta alta (pico aislado, sin cuerpo), que es justo el sintoma
medido del defecto de "dos capas" (audio sano 10-20 dB; la config rota de la
demo V11 llega a 49.4 dB). Entre los candidatos de cresta SANA (<22 dB), el
que gana por FCI es la suma ponderada (`suma_ancla`, fci=0.208) -- la propia
ancla de "dos sonidos superpuestos" le gana por FCI a TODAS las chimeras
genuinas de esa franja (la mejor solo llega a 0.116). Motivo probable: el
componente MCI (peso 0.55) premia que las bandas "respiren juntas" en 2-16 Hz,
y eso lo consigue tanto una fusion real como una suma con envolvente
compartida o una chimera patologica con bandas de A ausentes -- el mismo
fallo que ya hundio a `stream_unity_index` en V8.

**Consecuencia practica: el FCI y sus 4 componentes se calculan y guardan
integros en el CSV (es un resultado negativo citable para el paper), pero
NO deciden que hay en estas carpetas.** La seleccion de cada lote es:

1. Filtro DURO: descarta cualquier candidato con `crest_db >= 25` (cresta
   patologica -- el sintoma medido del defecto).
2. Entre los que pasan el filtro, cobertura de DIVERSIDAD del espacio de
   parametros (distintos `n_bands`, distintos `color_mix`, con y sin
   `align`, con y sin `warp`, y los metodos alternativos `chimera_plana`/
   `vocoder` cuando aplican) -- NO una metrica. El objetivo es que oigas un
   abanico representativo, no cinco variantes casi identicas.
3. SIEMPRE se incluyen, como referencias fijas y etiquetadas como tales:
   `baseline_v11` (exactamente la demo actual: coloreada, sin alinear, sin
   suelo activo -- el "hoy") y `suma_ancla` (los mismos dos padres sumados
   sin ningun mecanismo de fusion -- el "antes", dos sonidos superpuestos a
   proposito).

Cada carpeta `<pareja>/` trae hasta 10 clips: `muestra1`..`muestra8`
(seleccion por cobertura, NO ordenados por calidad -- no asumas que
`muestra1` es mejor que `muestra8`) + `baseline_v11` + `suma_ancla`.

Todos los clips estan igualados en RMS, no solo en pico: la sonoridad del
blend varia hasta 20 dB entre configuraciones (color_mix=0 vs color_mix=1),
y comparar solo por pico habria dejado al oyente juzgando volumen en vez de
fusion. Si un clip tenia una cresta patologica, se uso un limitador suave en
vez de bajar todo el clip (columna `metodo_igualacion` en el CSV); esta
anotado por si se nota en el oido.

## Que escuchar

La pregunta de siempre: **¿oyes UN objeto/evento imposible, o dos sonidos
superpuestos?**

- Compara cada `muestraN` contra `baseline_v11` (¿mejora la fusion respecto
  a lo que hay hoy en la demo?) y contra `suma_ancla` (el suelo: si una
  muestra suena como esto, es tan mala como no fundir nada).
- El nombre del fichero es autodescriptivo: `align-on/off` (¿se movio el
  registro de un padre?), `nbN` (numero de bandas del filterbank),
  `mixX.XX` (0.00 = color puro de la dinamica A, 1.00 = el comportamiento
  de la demo actual), `warp-on/off` (¿se reagendaron los eventos de la
  materia a los picos de la dinamica?).

## Que necesito de vuelta

Por cada pareja:

1. Un ranking DE OIDO (aunque sea parcial) de las muestras, y si alguna
   cruza el umbral de "esto es UN objeto". Como el FCI no aporta orden
   fiable, este ranking es la unica fuente de verdad que vamos a tener.
2. Vetos: lo que suene mal o roto, para no repetirlo en el siguiente
   barrido.
3. Si tienes que elegir UN ganador por pareja para el mini-test perceptual
   (3 parejas), dilo explicitamente -- alimenta `GANADORES_V12` en
   `scripts/39_fusion_search.py`.

Los numeros completos (los 4 componentes de la metrica + FCI + diagnosticos,
de TODOS los candidatos del barrido, no solo estos 10 por pareja) estan en
`results/fusion_search/<pareja>.csv` y `results/fusion_search/resumen.csv`.

## Resumen por pareja
"""]
    for s in summaries:
        parts.append(f"\n### {s['pareja']}\n")
        if s["crest_filter_vacio"]:
            parts.append(
                f"**Aviso**: NINGUN candidato de esta pareja bajo de "
                f"{CREST_HEALTHY_DB} dB de cresta; el lote de abajo son los "
                f"{s['n_survivors_crest'] or N_LISTEN_DIVERSE} de MENOR cresta "
                f"(los menos malos, no candidatos sanos).\n\n")
        else:
            parts.append(f"Supervivientes del filtro de cresta: "
                        f"{s['n_survivors_crest']}/{s['n_pool']}.\n\n")
        parts.append("| clip | fci (diagnostico, NO decide) | mci | dop | bri | "
                    "crest_db | igualacion |\n")
        parts.append("|---|---|---|---|---|---|---|\n")
        for e in s["export_table"]:
            parts.append(f"| {e['tag']} | {e['fci']:.3f} | {e['mci']:.3f} | "
                        f"{e['dop']:.3f} | {e['bri']:.3f} | {e['crest_db']:.1f} | "
                        f"{e['metodo_igualacion']} |\n")
    listen_dir.mkdir(parents=True, exist_ok=True)
    (listen_dir / "LEEME.md").write_text("".join(parts), encoding="utf-8")
    print(f"LEEME: {listen_dir / 'LEEME.md'}")


# ====================================================================
# --freeze: estimulos del mini-test perceptual (T5)
# ====================================================================
# Semilla FIJA solo para el ORDEN de los ficheros opacos (independiente de
# SEED, que sigue gobernando el audio) -- el mismo GANADORES_V12 produce
# siempre el mismo mapeo est_NN <-> (pareja, condicion), reproducible.
_FREEZE_SHUFFLE_SEED = 12345


def _freeze_stimuli(ganadores: list[tuple[str, FusionSpec]], out_dir: Path,
                    manifest_path: Path, sr: int = SR) -> int:
    """Renderiza los 6 estimulos (3 parejas x {suma, ganadora}) de
    GANADORES_V12 con nombres de fichero OPACOS (est_01.wav..est_06.wav),
    en un orden barajado con semilla fija, y un manifest aparte que
    contiene el mapeo real.

    CEGAMIENTO DELIBERADO -- no "arreglar" poniendo nombres descriptivos:
    el formulario perceptual (T5, perceptual_test/form_fusion/) sirve estos
    WAV en un <audio src="...">, asi que el nombre de fichero queda expuesto
    en el DOM -- cualquier oyente con clic derecho / Inspeccionar elemento /
    la pestana de red vería literalmente la palabra "ganadora" o el nombre
    de la pareja si el fichero se llamara "<pareja>__<condicion>.wav". El
    test es a doble ciego para un envio a paper: esa fuga invalidaria
    metodologicamente los juicios. Por eso el nombre en disco no lleva ni
    pareja ni condicion, y el ORDEN tampoco seria seguro sin barajar (si
    todos los "suma" fueran est_01/03/05 y todas las "ganadora" est_02/04/06,
    el patron se descubre con dos o tres estimulos escuchados). El mapeo
    real (pareja, condicion, path, config_json -- trazabilidad total para el
    analisis posterior) vive SOLO en `manifest_path`, que el oyente no ve.
    """
    if not ganadores:
        print(
            "ERROR: GANADORES_V12 esta vacia.\n\n"
            "Rellena la constante GANADORES_V12 en scripts/39_fusion_search.py con "
            "EXACTAMENTE 3 tuplas (pareja, FusionSpec) tras escuchar "
            "escucha_AB/fusion_v12/ y elegir el ganador de cada una de esas 3 "
            "parejas para el mini-test perceptual. El formato exacto (con un "
            "ejemplo) esta documentado justo encima de la constante en este "
            "fichero.", file=sys.stderr)
        return 1
    if len(ganadores) != 3:
        print(f"ERROR: GANADORES_V12 tiene {len(ganadores)} entradas, se esperan "
             "exactamente 3 (una por pareja del mini-test T5).", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}

    items: list[tuple[str, str, FusionSpec]] = []  # (pareja, condicion, spec)
    for pareja, spec in ganadores:
        env, fine = spec.env_parent, spec.fine_parent
        spec_suma = FusionSpec(env_parent=env, fine_parent=fine, method="suma",
                               duration_s=6.0, seed=SEED)
        items.append((pareja, "suma", spec_suma))
        items.append((pareja, "ganadora", spec))

    order = list(range(len(items)))
    random.Random(_FREEZE_SHUFFLE_SEED).shuffle(order)

    rows = []
    for i, idx in enumerate(order, 1):
        pareja, condicion, s = items[idx]
        w, _meta = render_fusion(s, sr=sr, cache=cache)
        w9, _info = _match_loudness(w, peak_ceiling=0.9)
        fname = f"est_{i:02d}.wav"
        path = out_dir / fname
        save_wav(path, w9, sr)
        rows.append({"pareja": pareja, "condicion": condicion, "path": str(path),
                    "config_json": json.dumps(asdict(s))})
        print(f"  {fname}  <- {pareja}/{condicion} (oculto al oyente)")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pareja", "condicion", "path", "config_json"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"manifest: {manifest_path} ({len(rows)} filas)")
    return 0


def freeze(out_dir: Path, manifest_path: Path) -> int:
    return _freeze_stimuli(GANADORES_V12, out_dir, manifest_path)


# ====================================================================
# --check: smokes
# ====================================================================

def _check_v4_equals_v3() -> list[str]:
    failures = []
    for name in CHIMERA_PARENTS_V3:
        a3 = chimera_parent_v3(name, 3.0, SEED, SR)
        a4 = chimera_parent_v4(name, 3.0, SEED, SR)
        if not np.array_equal(a3, a4):
            failures.append(name)
    print(f"smoke1 (chimera_parent_v4 == v3, {len(CHIMERA_PARENTS_V3)} padres): "
         f"{'OK' if not failures else 'FALLOS: ' + str(failures)}")
    return failures


def _check_methods_and_mechanisms() -> list[str]:
    failures = []
    specs = {
        "method=chimera": FusionSpec(env_parent="trueno", fine_parent="goteo", n_bands=4,
                                     method="chimera", duration_s=3.0, seed=SEED),
        "method=chimera_plana": FusionSpec(env_parent="trueno", fine_parent="goteo", n_bands=4,
                                           method="chimera_plana", duration_s=3.0, seed=SEED),
        "method=vocoder": FusionSpec(env_parent="fuego", fine_parent="vidrio", n_bands=12,
                                     method="vocoder", duration_s=3.0, seed=SEED),
        "method=suma": FusionSpec(env_parent="trueno", fine_parent="goteo",
                                  method="suma", duration_s=3.0, seed=SEED),
        "mecanismo=align": FusionSpec(env_parent="trueno", fine_parent="goteo", n_bands=4,
                                      align=True, method="chimera", duration_s=3.0, seed=SEED),
        "mecanismo=warp": FusionSpec(env_parent="trueno", fine_parent="goteo", n_bands=4,
                                     warp=True, method="chimera", duration_s=3.0, seed=SEED),
        "mecanismo=color_mix": FusionSpec(env_parent="trueno", fine_parent="goteo", n_bands=4,
                                          color_mix=0.5, method="chimera", duration_s=3.0,
                                          seed=SEED),
        "mecanismo=stats_finish": FusionSpec(env_parent="trueno", fine_parent="goteo", n_bands=4,
                                             stats_finish=True, method="chimera",
                                             duration_s=3.0, seed=SEED),
    }
    for label, spec in specs.items():
        cache: dict = {}
        w1, _m1 = render_fusion(spec, sr=SR, cache=cache)
        w2, _m2 = render_fusion(spec, sr=SR, cache=cache)
        finite = bool(np.isfinite(w1).all())
        peak = float(np.abs(w1).max())
        bounded = peak <= 0.9501
        deterministic = np.array_equal(w1, w2)
        ok = finite and bounded and deterministic
        print(f"  {label}: finite={finite} peak={peak:.4f} bounded={bounded} "
             f"det={deterministic} {'OK' if ok else 'FALLO'}")
        if not ok:
            failures.append(label)
    print(f"smoke2 (metodos+mecanismos): {'OK' if not failures else 'FALLOS: ' + str(failures)}")
    return failures


def _check_metric_fast_matches_reference() -> list[str]:
    """_composite_fusion_fast (con memo de SSO y formula del FCI duplicada)
    debe dar EXACTAMENTE lo mismo que analysis.composite_fusion() directo,
    con y sin alineacion -- guarda contra que la formula/pesos de analysis.py
    cambien sin que este fichero se entere."""
    failures = []
    for align in (False, True):
        cache: dict = {}
        sso_cache: dict = {}
        spec = FusionSpec(env_parent="trueno", fine_parent="vidrio", n_bands=6,
                          align=align, method="chimera", duration_s=3.0, seed=SEED)
        w, meta = render_fusion(spec, sr=SR, cache=cache)
        a_used, b_aligned, b_original = _metric_parents(spec, SR, cache, meta)
        fast = _composite_fusion_fast(w, a_used, b_aligned, b_original, SR, sso_cache)
        ref = composite_fusion(w, a_used, b_aligned, b_original, SR)
        ok = (fast.mci == ref.mci and fast.sso == ref.sso and fast.dop == ref.dop
             and fast.bri == ref.bri and fast.fci == ref.fci)
        print(f"  align={align}: fast={fast} ref={ref} {'OK' if ok else 'FALLO'}")
        if not ok:
            failures.append(f"align={align}")
    print(f"smoke2b (_composite_fusion_fast == composite_fusion): "
         f"{'OK' if not failures else 'FALLOS: ' + str(failures)}")
    return failures


def _check_register_hz_memo() -> list[str]:
    """Bit-identidad de render_fusion con register_hz memoizado vs original."""
    spec = FusionSpec(env_parent="trueno", fine_parent="vidrio", n_bands=6, align=True,
                      warp=False, method="chimera", duration_s=3.0, seed=SEED)
    w_memo, meta_memo = render_fusion(spec, sr=SR, cache={})
    _fc_module.register_hz = _ORIG_REGISTER_HZ
    try:
        w_orig, meta_orig = render_fusion(spec, sr=SR, cache={})
    finally:
        _fc_module.register_hz = _register_hz_memoized
    ok = (np.array_equal(w_memo, w_orig)
         and meta_memo.get("register_achieved_hz") == meta_orig.get("register_achieved_hz"))
    print(f"smoke2c (register_hz memoizado == original): {'OK' if ok else 'FALLO'}")
    return [] if ok else ["register_hz_memo"]


def _check_autocalibracion() -> list[str]:
    """Ruling 7 (progress.md): con align y color fijos en la config de V11
    (align=False, color_mix=None), el FCI debe rankear el n_bands validado
    de oido en el top-2 de los n_bands de ESA pareja. Autoridad: 6s (la
    duracion real del grid), no los 3s del resto de --check -- a 3s el
    trueno apenas da eventos para DOP/BRI y un fallo seria inintepretable.
    Se reporta TAMBIEN a 3s para comparar, pero el criterio de aprobado/
    fallado es el de 6s."""
    failures = []
    targets = [("trueno_hecho_de_agua", "trueno", "goteo", 4),
              ("fuego_hecho_de_vidrio", "fuego", "vidrio", 16)]
    for dur in (6.0, 3.0):
        print(f"  -- duracion {dur}s --")
        for pareja, env, fine, nb_esperado in targets:
            cache: dict = {}
            sso_cache: dict = {}
            results = []
            for nb in N_BANDS_GRID:
                spec = FusionSpec(env_parent=env, fine_parent=fine, n_bands=nb, align=False,
                                  warp=False, color_mix=None, method="chimera",
                                  duration_s=dur, seed=SEED)
                w, meta = render_fusion(spec, sr=SR, cache=cache)
                a_used, b_aligned, b_original = _metric_parents(spec, SR, cache, meta)
                rep = _composite_fusion_fast(w, a_used, b_aligned, b_original, SR, sso_cache)
                results.append((nb, rep.fci))
            ranked = sorted(results, key=lambda t: t[1], reverse=True)
            top2 = [nb for nb, _ in ranked[:2]]
            pass_ = nb_esperado in top2
            print(f"    {pareja}: esperado nb={nb_esperado}, ranking fci={ranked}, "
                 f"top2={top2}, {'OK' if pass_ else 'FALLO'}")
            if dur == 6.0 and not pass_:
                failures.append(f"{pareja}@6s")
    print(f"smoke3 (auto-calibracion @6s, autoridad): "
         f"{'OK' if not failures else 'FALLOS: ' + str(failures)}")
    return failures


def _check_csv_roundtrip() -> list[str]:
    row = _empty_row()
    row.update({k: "1" for k in ("pareja", "grupo", "method")})
    row.update(mci=0.123456, fci=-0.5, register_target_hz=None, align=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.csv"
        _write_csv(path, [row])
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            rows_back = list(reader)
    ok = (cols == FIELDNAMES and len(rows_back) == 1
         and rows_back[0]["pareja"] == "1" and rows_back[0]["mci"] == "0.123456"
         and rows_back[0]["register_target_hz"] == "")
    print(f"smoke4 (round-trip CSV): {'OK' if ok else 'FALLO'}")
    return [] if ok else ["csv_roundtrip"]


def _check_freeze_empty_fails_clean() -> list[str]:
    ok_empty = (_freeze_stimuli([], Path("/dev/null"), Path("/dev/null")) == 1)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "stimuli"
        manifest = Path(tmp) / "manifest.csv"
        # 3 ganadores de prueba (el formato real que --freeze exige): una
        # tupla (pareja, FusionSpec) por cada una de 3 parejas distintas.
        fake = [
            ("trueno_hecho_de_agua", FusionSpec(
                env_parent="trueno", fine_parent="goteo", n_bands=4, align=True,
                warp=True, color_mix=0.25, method="chimera", duration_s=3.0, seed=SEED)),
            ("fuego_hecho_de_vidrio", FusionSpec(
                env_parent="fuego", fine_parent="vidrio", n_bands=16, align=False,
                warp=False, color_mix=0.5, method="chimera", duration_s=3.0, seed=SEED)),
            ("oceano_hecho_de_campana", FusionSpec(
                env_parent="oceano", fine_parent="campana_tela", n_bands=12, align=False,
                warp=False, color_mix=0.5, method="chimera", duration_s=3.0, seed=SEED)),
        ]
        rc = _freeze_stimuli(fake, out_dir, manifest)
        wavs = sorted(out_dir.glob("*.wav"))
        names = [p.name for p in wavs]
        ok_fake = (rc == 0 and len(wavs) == 6 and manifest.exists())

        # Regresion del cegamiento pedido por el coordinador: el nombre en
        # disco NO debe delatar pareja ni condicion, y debe ser el patron
        # opaco est_NN.wav exacto.
        leaky_tokens = ("suma", "ganadora", "trueno", "fuego", "oceano",
                        "goteo", "vidrio", "campana")
        ok_opaque = all(
            p.stem.startswith("est_") and not any(tok in p.name for tok in leaky_tokens)
            for p in wavs)

        # Regresion de "no todos los suma primero": con GANADORES_V12 real
        # (siempre suma antes que ganadora por pareja en `items`), el orden
        # SIN barajar seria exactamente suma,ganadora,suma,ganadora,... --
        # comprobamos que el manifest (que preserva el orden de escritura,
        # es decir el orden est_01..est_06) no reproduce ese patron trivial.
        with open(manifest, "r", encoding="utf-8") as f:
            condiciones_en_orden = [row["condicion"] for row in csv.DictReader(f)]
        patron_trivial = ["suma", "ganadora"] * 3
        ok_shuffled = condiciones_en_orden != patron_trivial
    ok = ok_empty and ok_fake and ok_opaque and ok_shuffled
    print(f"smoke5 (--freeze vacio falla limpio={ok_empty}, 3 ganadores de "
         f"prueba -> 6 wavs+manifest={ok_fake}, nombres opacos={ok_opaque} "
         f"{names}, orden no trivial={ok_shuffled} {condiciones_en_orden}): "
         f"{'OK' if ok else 'FALLO'}")
    return [] if ok else ["freeze"]


def check() -> int:
    all_failures = []
    all_failures += _check_v4_equals_v3()
    all_failures += _check_methods_and_mechanisms()
    all_failures += _check_metric_fast_matches_reference()
    all_failures += _check_register_hz_memo()
    all_failures += _check_autocalibracion()
    all_failures += _check_csv_roundtrip()
    all_failures += _check_freeze_empty_fails_clean()
    if all_failures:
        print(f"\n{len(all_failures)} fallos: {all_failures}")
        return 1
    print("\ntodos los smokes OK")
    return 0


# ====================================================================
# main
# ====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="smokes (clips de 3s salvo smoke3)")
    ap.add_argument("--pair", type=str, default=None,
                    help="ejecuta solo esta pareja (pilotaje antes del barrido completo)")
    ap.add_argument("--freeze", action="store_true",
                    help="congela estimulos de GANADORES_V12 para el mini-test T5")
    ap.add_argument("--results-dir", type=Path, default=OUT_RESULTS)
    ap.add_argument("--listen-dir", type=Path, default=OUT_LISTEN)
    ap.add_argument("--freeze-dir", type=Path, default=OUT_FREEZE)
    ap.add_argument("--freeze-manifest", type=Path, default=OUT_FREEZE_MANIFEST)
    args = ap.parse_args()

    if args.check:
        return check()
    if args.freeze:
        return freeze(args.freeze_dir, args.freeze_manifest)
    if args.pair:
        match = next(((n, e, f) for n, e, f in PAREJAS_V12 if n == args.pair), None)
        if match is None:
            print(f"pareja desconocida: {args.pair!r}. Validas: "
                 f"{[n for n, _, _ in PAREJAS_V12]}", file=sys.stderr)
            return 1
        run_pair(*match, args.results_dir, args.listen_dir)
        _write_resumen(args.results_dir)
        return 0

    summaries = []
    for name, env, fine in PAREJAS_V12:
        summaries.append(run_pair(name, env, fine, args.results_dir, args.listen_dir))
    _write_resumen(args.results_dir)
    _write_leeme(args.listen_dir, summaries)
    total_t = sum(s["t_total_s"] for s in summaries)
    print(f"\nTOTAL: {total_t:.1f}s ({total_t / 60.0:.1f} min) para {len(summaries)} parejas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
