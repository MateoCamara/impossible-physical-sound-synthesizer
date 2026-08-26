"""Evidencia trazable + figura del paper (tarea 6 / F1) para el trabajo de
fusion v12-F4/F5 (scripts/39_fusion_search.py).

Los hallazgos de la fusion hoy solo viven en docstrings y en stdout de
scripts/39_fusion_search.py. El paper promete en la seccion de
Reproducibilidad que sus numeros son trazables a artefactos versionados.
Este script persiste esa evidencia (CSV) y produce la figura F6 (dos
paneles) sin volver a barrer el grid completo (results/fusion_search/
resumen.csv ya esta commiteado y verificado, 1066 filas x 47 columnas,
6 parejas x scripts/39_fusion_search.py::PAREJAS_V12) -- solo hace los
renders CORTOS que hacen falta para medidas por banda que resumen.csv no
guarda a ese nivel de detalle (RMS por banda individual, correlacion de
envolvente por region, cresta a una duracion distinta de la del grid).

Genera en results/fusion_search/:
  bandas_huerfanas.csv       RMS por banda del padre A (dB rel. al pico),
                              marca de huerfana, para cada pareja x n_bands
                              del grid x align on/off.
  correlacion_envolvente.csv Correlacion de envolvente del blend con cada
                              padre, por region (graves/medios/agudos),
                              chimera_plana vs. chimera coloreada barriendo
                              color_mix.
  baseline_v11_crest.csv     Cresta del baseline V11 (config real de
                              resumen.csv: n_bands=chimera_bands_heuristic,
                              align=False, warp=False, color_mix=None) a
                              6s (duracion del grid) y a 8s.
  evidencia_paper.csv        Indice de afirmaciones citables: id_claim,
                              seccion, texto_claim, valor, unidad,
                              fuente_csv, filtro_o_fila -- auditable con
                              --check.

Genera en figures/: F6_fusion.png (2 paneles, un solo flotante).

Reutiliza scripts/39_fusion_search.py (PAREJAS_V12, N_BANDS_GRID,
COLOR_MIX_GRID, ORPHAN_FLOOR_DB, DUR_GRID, _count_orphan_bands_a,
_band_rms_db, _metric_parents) via importlib -- "39_fusion_search" no es
un identificador Python valido (empieza por digito), asi que no se puede
`import` de forma literal; se carga el fichero por ruta con
importlib.util.spec_from_file_location. Importar ese modulo ejecuta su
monkeypatch de fusion_chain.register_hz (memo verificado bit-identico en
su propio --check) como efecto secundario deliberado y documentado ahi.

Uso:
    PYTHONPATH=. .venv/bin/python scripts/41_fusion_evidence.py --check
    PYTHONPATH=. .venv/bin/python scripts/41_fusion_evidence.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Estilo paper: copiado INTEGRO de scripts/15_figures.py:22-33 (no se toca
# ese fichero -- depende de otros artefactos y obligaria a regenerarlos).
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 140,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impossible_mix.physics.analysis import band_envelopes, _max_corr, crest_factor_db  # noqa: E402
from impossible_mix.physics.blend_recipes import chimera_bands_heuristic  # noqa: E402
from impossible_mix.physics.fusion_chain import FusionSpec, render_fusion  # noqa: E402


def _load_script39():
    """Carga scripts/39_fusion_search.py por ruta (ver docstring del
    modulo: "39_fusion_search" no es un identificador Python valido para
    un `import` normal). Ejecutar el modulo aplica su monkeypatch de
    register_hz -- documentado, reversible, verificado bit-identico."""
    path = Path(__file__).resolve().parent / "39_fusion_search.py"
    spec = importlib.util.spec_from_file_location("_script39_fusion_search", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S39 = _load_script39()
PAREJAS_V12 = S39.PAREJAS_V12
N_BANDS_GRID = S39.N_BANDS_GRID
COLOR_MIX_GRID = S39.COLOR_MIX_GRID
ORPHAN_FLOOR_DB = S39.ORPHAN_FLOOR_DB
DUR_GRID = S39.DUR_GRID
_count_orphan_bands_a = S39._count_orphan_bands_a
_band_rms_db = S39._band_rms_db
_metric_parents = S39._metric_parents

SR = 44_100
SEED = 42
RESULTS_DIR = Path("results/fusion_search")
FIG_DIR = Path("figures")
RESUMEN_PATH = RESULTS_DIR / "resumen.csv"
BANDAS_PATH = RESULTS_DIR / "bandas_huerfanas.csv"
CORR_PATH = RESULTS_DIR / "correlacion_envolvente.csv"
BASELINE_CREST_PATH = RESULTS_DIR / "baseline_v11_crest.csv"
EVIDENCIA_PATH = RESULTS_DIR / "evidencia_paper.csv"
FIG_PATH = FIG_DIR / "F6_fusion.png"

# Filterbank de CHIMERA (auditory_chimera / auditory_chimera_colored):
# mismos limites que usa _count_orphan_bands_a (scripts/39_fusion_search.py).
CHIMERA_LO_HZ = 80.0
CHIMERA_HI_HZ = 8820.0

# Regiones de correlacion de envolvente (mismos limites que rms_grave_db/
# rms_aguda_db en scripts/39_fusion_search.py para graves/agudos; medios
# rellena el hueco entre ambas).
REGIONS = {"graves": (40.0, 500.0), "medios": (500.0, 1500.0), "agudos": (1500.0, 20000.0)}
CORR_MAX_LAG_S = 0.020  # 20 ms, igual que fusion_index en analysis.py

BASELINE_DURATIONS = (6.0, 8.0)  # 6s = DUR_GRID (resumen.csv); 8s = duracion
                                  # de los clips "estrellas" de scripts/38_demo_congreso.py

# (pareja, padre movido, id_claim) -- unica fuente para el sobredisparo de
# registro, reusada por evidencia_paper.csv Y por la figura (panel b/pie),
# para que ambos no puedan divergir si resumen.csv cambiara.
OVERSHOOT_SPECS = [
    ("fuego_hecho_de_vidrio", "vidrio", "overshoot_vidrio"),
    ("trueno_hecho_de_canica", "canica", "overshoot_canica_registro_bajo"),
    ("canica_hecha_de_fuego", "canica", "overshoot_canica_registro_alto"),
    ("goteo_hecho_de_campana", "campana_tela", "overshoot_campana_tela_registro_bajo"),
    ("oceano_hecho_de_campana", "campana_tela", "overshoot_campana_tela_registro_alto"),
    ("trueno_hecho_de_agua", "goteo", "overshoot_goteo"),
]


def compute_overshoots(resumen: pd.DataFrame) -> dict[str, dict]:
    """(achieved-target)/target*100 por padre movido, para cada entrada de
    OVERSHOOT_SPECS -- unica fuente de verdad (ver comentario junto a
    OVERSHOOT_SPECS)."""
    aligned = resumen[resumen["align"] == True].drop_duplicates(  # noqa: E712
        subset=["pareja", "moved_parent", "register_target_hz", "register_achieved_hz"])
    out: dict[str, dict] = {}
    for pareja, moved, claim_id in OVERSHOOT_SPECS:
        row = aligned[(aligned["pareja"] == pareja) & (aligned["moved_parent"] == moved)].iloc[0]
        target, achieved = float(row["register_target_hz"]), float(row["register_achieved_hz"])
        out[claim_id] = {"pareja": pareja, "moved": moved, "target": target,
                         "achieved": achieved, "pct": (achieved - target) / target * 100.0}
    return out


def compute_clamped_stats(resumen: pd.DataFrame) -> tuple[int, int]:
    """(n_candidatos con align=True, de esos cuantos con register_clamped=True)."""
    n_aligned = int((resumen["align"] == True).sum())  # noqa: E712
    n_clamped = int(((resumen["align"] == True) & (resumen["register_clamped"] == True)).sum())  # noqa: E712
    return n_aligned, n_clamped


# ====================================================================
# Carga de resumen.csv con tipos reales
# ====================================================================

def load_resumen() -> pd.DataFrame:
    df = pd.read_csv(RESUMEN_PATH)
    for c in ("align", "warp", "stats_finish", "register_clamped", "align_noop",
              "exportado", "pico_recortado"):
        df[c] = df[c].map({"True": True, "False": False, True: True, False: False, np.nan: False})
    return df


# ====================================================================
# bandas_huerfanas.csv
# ====================================================================

def build_bandas_huerfanas() -> pd.DataFrame:
    """Para cada pareja x align(False/True) x n_bands del grid: RMS por
    banda del padre A REALMENTE USADO (a_used de _metric_parents, que
    difiere del env_parent crudo cuando align mueve al padre-ancla en vez
    de al padre-materia -- ver docstring de _metric_parents), en dB
    relativos a su banda mas fuerte, y la marca de huerfana (< ORPHAN_
    FLOOR_DB). Usa _band_rms_db (RMS del padre bandpaseado, no de la
    envolvente) sobre los mismos bordes log-espaciados que band_envelopes
    usaria para ese n_bands -- reproduce casi exactamente el docstring de
    blend.py:903-905 (ver informe de la tarea). Cruza cada grupo contra
    _count_orphan_bands_a (envolvente, el que ya vive en resumen.csv como
    n_bandas_huerfanas_a) como columna de auditoria: ambos metodos miden
    "presencia de A por banda" pero no son identicos (RMS de envolvente
    suavizada+rectificada frente a RMS de la senal bandpaseada cruda), y
    pueden discrepar en 1-2 bandas a n_bands alto -- se deja visible en el
    CSV en vez de forzar que coincidan."""
    rows: list[dict] = []
    for pareja, env, fine in PAREJAS_V12:
        cache: dict = {}
        for align in (False, True):
            rep_spec = FusionSpec(env_parent=env, fine_parent=fine, n_bands=N_BANDS_GRID[0],
                                  align=align, warp=False, color_mix=None, method="chimera",
                                  duration_s=DUR_GRID, seed=SEED)
            w, meta = render_fusion(rep_spec, sr=SR, cache=cache)
            a_used, _b_aligned, _b_original = _metric_parents(rep_spec, SR, cache, meta)
            for n_bands in N_BANDS_GRID:
                edges = np.geomspace(CHIMERA_LO_HZ, CHIMERA_HI_HZ, n_bands + 1)
                db_abs = np.array([_band_rms_db(a_used, SR, (float(edges[k]), float(edges[k + 1])))
                                   for k in range(n_bands)])
                db_rel = db_abs - db_abs.max()
                es_huerfana = db_rel < ORPHAN_FLOOR_DB
                n_huerfanas_bandpass = int(es_huerfana.sum())
                n_huerfanas_ref_envolvente = int(_count_orphan_bands_a(a_used, SR, n_bands))
                for k in range(n_bands):
                    rows.append({
                        "pareja": pareja, "env_parent": env, "fine_parent": fine,
                        "align": align, "n_bands": n_bands, "banda_idx": k,
                        "centro_hz": round(float(np.sqrt(edges[k] * edges[k + 1])), 2),
                        "edge_lo_hz": round(float(edges[k]), 2),
                        "edge_hi_hz": round(float(edges[k + 1]), 2),
                        "rms_db_abs": round(float(db_abs[k]), 4),
                        "rms_db_rel_pico": round(float(db_rel[k]), 4),
                        "es_huerfana": bool(es_huerfana[k]),
                        "n_huerfanas_bandpass_grupo": n_huerfanas_bandpass,
                        "n_huerfanas_ref_envolvente_grupo": n_huerfanas_ref_envolvente,
                        "coincide_con_referencia_grupo": n_huerfanas_bandpass == n_huerfanas_ref_envolvente,
                    })
    return pd.DataFrame(rows)


# ====================================================================
# correlacion_envolvente.csv
# ====================================================================

def _region_envelope(w: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    envs, _ = band_envelopes(w, sr, n_bands=1, lo=lo, hi=hi)
    return envs[0].astype(np.float64)


def build_correlacion_envolvente() -> pd.DataFrame:
    """Correlacion de envolvente (band_envelopes + _max_corr, ventana de
    +-20ms igual que fusion_index en analysis.py) del blend con cada padre
    crudo, por region (graves <500Hz, medios 500-1500Hz, agudos >1500Hz),
    comparando method="chimera_plana" contra method="chimera" barriendo
    color_mix en COLOR_MIX_GRID -- align=False, warp=False fijos (aisla el
    efecto del color de las otras dos palancas), n_bands=chimera_bands_
    heuristic(env) (la misma resolucion que usa baseline_v11 en el grid).
    color_mix=1.0 es numericamente casi identico a color_mix=None (color_
    from="b", el default real de auditory_chimera_colored y por tanto de
    baseline_v11 -- verificado en el informe de la tarea, diferencia <1e-3
    en los casos probados)."""
    rows: list[dict] = []
    configs = [("chimera_plana", None)] + [("chimera", cm) for cm in COLOR_MIX_GRID]
    max_lag = int(CORR_MAX_LAG_S * SR)
    for pareja, env, fine in PAREJAS_V12:
        nb = chimera_bands_heuristic(env)
        cache: dict = {}
        a_spec = FusionSpec(env_parent=env, fine_parent=fine, n_bands=nb, duration_s=DUR_GRID,
                            seed=SEED)
        w0, meta0 = render_fusion(a_spec, sr=SR, cache=cache)
        a_used, b_aligned, b_original = _metric_parents(a_spec, SR, cache, meta0)
        env_a = {r: _region_envelope(a_used, SR, *bd) for r, bd in REGIONS.items()}
        env_b = {r: _region_envelope(b_original, SR, *bd) for r, bd in REGIONS.items()}
        for method, cm in configs:
            spec = FusionSpec(env_parent=env, fine_parent=fine, n_bands=nb, align=False,
                              warp=False, color_mix=cm, method=method, duration_s=DUR_GRID,
                              seed=SEED)
            w, _meta = render_fusion(spec, sr=SR, cache=cache)
            for region, bd in REGIONS.items():
                e_w = _region_envelope(w, SR, *bd)
                n = min(len(e_w), len(env_a[region]), len(env_b[region]))
                corr_a = _max_corr(e_w[:n], env_a[region][:n], max_lag)
                corr_b = _max_corr(e_w[:n], env_b[region][:n], max_lag)
                rows.append({
                    "pareja": pareja, "env_parent": env, "fine_parent": fine, "n_bands": nb,
                    "method": method, "color_mix": ("" if cm is None else cm),
                    "region": region, "region_lo_hz": bd[0], "region_hi_hz": bd[1],
                    "corr_con_a": round(corr_a, 6), "corr_con_b": round(corr_b, 6),
                })
    return pd.DataFrame(rows)


# ====================================================================
# baseline_v11_crest.csv
# ====================================================================

def build_baseline_v11_crest() -> pd.DataFrame:
    """Cresta (crest_factor_db) del baseline V11 EXACTO (misma FusionSpec
    que el grupo baseline_v11 de resumen.csv: n_bands=chimera_bands_
    heuristic(env), align=False, warp=False, color_mix=None, method=
    chimera, seed=42) a duration_s=6.0 (DUR_GRID, ya en resumen.csv, se
    re-mide aqui para tener ambas duraciones en el mismo artefacto) y a
    duration_s=8.0 (duracion de los clips "estrellas" de scripts/38_demo_
    congreso.py). Cierra (con el numero medido, no a ciegas) la
    discrepancia conocida entre "47.5 dB en el grid a 6s" y "49.4 dB
    citado de la demo a 8s" -- ver informe de la tarea para el detalle de
    que combinaciones de n_bands/duracion se probaron y ninguna reprodujo
    49.4 dB."""
    rows: list[dict] = []
    for pareja, env, fine in PAREJAS_V12:
        nb = chimera_bands_heuristic(env)
        for dur in BASELINE_DURATIONS:
            spec = FusionSpec(env_parent=env, fine_parent=fine, n_bands=nb, align=False,
                              warp=False, color_mix=None, method="chimera", duration_s=dur,
                              seed=SEED)
            w, _meta = render_fusion(spec, sr=SR, cache={})
            rows.append({
                "pareja": pareja, "env_parent": env, "fine_parent": fine, "n_bands": nb,
                "duration_s": dur, "crest_db": round(crest_factor_db(w), 4),
            })
    return pd.DataFrame(rows)


# ====================================================================
# evidencia_paper.csv
# ====================================================================

def build_evidencia_paper(resumen: pd.DataFrame, bandas: pd.DataFrame,
                          baseline_crest: pd.DataFrame, corr_env: pd.DataFrame) -> pd.DataFrame:
    """Una fila por afirmacion citable del paper. `filtro_o_fila` es una
    expresion booleana evaluable con DataFrame.query() sobre `fuente_csv`
    (pandas, columnas con sus nombres literales) -- --check la evalua de
    verdad contra el fichero en disco (ver check_evidencia_resuelve). Salvo
    la fila explicitamente marcada como "0 candidatos" (register_clamped),
    donde CERO coincidencias ES la afirmacion, todas las demas deben
    resolver a >=1 fila real."""
    claims: list[dict] = []

    def add(id_claim, seccion, texto, valor, unidad, fuente, filtro):
        claims.append({"id_claim": id_claim, "seccion": seccion, "texto_claim": texto,
                       "valor": valor, "unidad": unidad, "fuente_csv": fuente,
                       "filtro_o_fila": filtro})

    # --- 1. Solape espectral (SSO) sin alinear -> alineando, las 6 parejas ---
    sso_tbl = (resumen.groupby(["pareja", "align"])["sso"].agg(["nunique", "mean"]))
    assert (sso_tbl["nunique"] == 1).all(), "sso no es constante por (pareja,align) en resumen.csv"
    for pareja, _env, _fine in PAREJAS_V12:
        off = float(sso_tbl.loc[(pareja, False), "mean"])
        on = float(sso_tbl.loc[(pareja, True), "mean"])
        add(f"sso_off_{pareja}", "alineacion_registro",
           f"Solape espectral (SSO) de {pareja} SIN alinear registro.",
           round(off, 6), "sso (0-1, Bhattacharyya)", "resumen.csv",
           f"pareja == '{pareja}' and align == False")
        add(f"sso_on_{pareja}", "alineacion_registro",
           f"Solape espectral (SSO) de {pareja} alineando registro.",
           round(on, 6), "sso (0-1, Bhattacharyya)", "resumen.csv",
           f"pareja == '{pareja}' and align == True")

    # --- 2. Correlacion global FCI-crest_db (el fallo de la metrica) ---
    sub = resumen.dropna(subset=["fci", "crest_db"])
    corr_fci_crest = float(sub["fci"].corr(sub["crest_db"]))
    add("corr_fci_crest_global", "metrica_fci",
       f"Correlacion de Pearson entre FCI y crest_db sobre los "
       f"{len(sub)} candidatos con FCI definido (6 parejas): el FCI premia "
       f"la cresta alta, el sintoma medido del defecto de 'dos capas'.",
       round(corr_fci_crest, 6), "r de Pearson", "resumen.csv", "fci.notna()")

    # --- 3. Bandas huerfanas: verificacion del docstring de blend.py y
    # cobertura por pareja a su n_bands heuristico ---
    ref_rows = bandas[(bandas["pareja"] == "trueno_hecho_de_agua") & (~bandas["align"])
                      & (bandas["n_bands"] == 6) & (bandas["es_huerfana"])]
    valores = ";".join(f"{v:.2f}" for v in sorted(ref_rows["rms_db_rel_pico"], reverse=True))
    add("orphan_bands_trueno_nb6_docstring", "bandas_huerfanas",
       "Verificacion del docstring de blend.py:903-905 ('A esta a -56,-85,"
       "-94.6 dB en 3 de las 6 bandas', trueno como padre A -- la cuenta "
       "no depende del padre B, asi que aplica igual a trueno_hecho_de_agua "
       "y trueno_hecho_de_canica): valor medido de nuevo con _band_rms_db, "
       f"{len(ref_rows)} de 6 bandas huerfanas.",
       valores, "dB rel. al pico de la banda mas fuerte (por banda huerfana)",
       "bandas_huerfanas.csv",
       "pareja == 'trueno_hecho_de_agua' and align == False and n_bands == 6 and es_huerfana == True")

    for pareja, env, _fine in PAREJAS_V12:
        nb = chimera_bands_heuristic(env)
        grp = bandas[(bandas["pareja"] == pareja) & (~bandas["align"]) & (bandas["n_bands"] == nb)]
        n_h = int(grp["n_huerfanas_bandpass_grupo"].iloc[0])
        add(f"orphan_bands_{pareja}_nb_heuristico", "bandas_huerfanas",
           f"Bandas huerfanas del padre A de {pareja} (sin alinear) al "
           f"n_bands heuristico usado por baseline_v11 (n_bands={nb}).",
           n_h, f"bandas huerfanas de {nb} (umbral {ORPHAN_FLOOR_DB:.0f} dB)",
           "bandas_huerfanas.csv",
           f"pareja == '{pareja}' and align == False and n_bands == {nb}")

    # --- 4. Cresta del baseline V11 (6s y 8s), trueno_hecho_de_agua ---
    b6 = resumen[(resumen["pareja"] == "trueno_hecho_de_agua") & (resumen["grupo"] == "baseline_v11")]
    crest_6s = float(b6["crest_db"].iloc[0])
    add("crest_baseline_v11_trueno_agua_6s", "reproducibilidad",
       "Cresta del baseline V11 (trueno_hecho_de_agua) a 6s -- duracion "
       "del grid de resumen.csv.",
       round(crest_6s, 4), "dB (crest factor)", "resumen.csv",
       "pareja == 'trueno_hecho_de_agua' and grupo == 'baseline_v11'")

    b8 = baseline_crest[(baseline_crest["pareja"] == "trueno_hecho_de_agua")
                        & (baseline_crest["duration_s"] == 8.0)]
    crest_8s = float(b8["crest_db"].iloc[0])
    add("crest_baseline_v11_trueno_agua_8s", "reproducibilidad",
       "Cresta del baseline V11 (trueno_hecho_de_agua) a 8s -- misma "
       "duracion que los clips 'estrellas' de scripts/38_demo_congreso.py. "
       "NO reproduce los 49.4 dB citados de la demo (se probo tambien "
       "n_bands=4 y n_bands=8 a 6s/8s: maximo medido 47.71 dB) -- "
       "discrepancia sin cerrar, reportada tal cual, ver informe de la tarea.",
       round(crest_8s, 4), "dB (crest factor)", "baseline_v11_crest.csv",
       "pareja == 'trueno_hecho_de_agua' and duration_s == 8.0")

    # --- 5. Sobredisparos de registro por padre movido (align=True) ---
    overshoots = compute_overshoots(resumen)
    for claim_id, o in overshoots.items():
        add(claim_id, "alineacion_registro",
           f"Sobredisparo de registro al mover a '{o['moved']}' en {o['pareja']}: "
           f"objetivo {o['target']:.1f} Hz, logrado {o['achieved']:.1f} Hz.",
           round(o["pct"], 2), "% sobre el objetivo (register_target_hz)", "resumen.csv",
           f"pareja == '{o['pareja']}' and align == True and moved_parent == '{o['moved']}'")

    # --- 6. Ningun candidato topo con el limite de rango de alineacion ---
    n_aligned, n_clamped = compute_clamped_stats(resumen)
    add("register_clamped_cero_de_494", "alineacion_registro",
       f"Candidatos con align=True cuyo register_clamped=True (tope de "
       f"rango de alineacion), de {n_aligned} candidatos con align=True: "
       f"el error de alineacion no es por recorte de rango.",
       n_clamped, f"candidatos (de {n_aligned})", "resumen.csv",
       "align == True and register_clamped == True")

    # --- 7. Correlacion de envolvente (graves) plana vs. coloreada: el
    # hallazgo del brief ("la coloreada sigue a A con 0.121 en graves frente
    # a 0.655 de la plana") NO se reproduce como UNA comparacion dentro de
    # una misma pareja -- ver informe de la tarea. Lo que se mide, medido de
    # nuevo y verificado con --check, es dos hallazgos DISTINTOS en dos
    # parejas distintas (fuego_hecho_de_vidrio: coloreada sigue MENOS a A
    # que la plana, como dice el brief; goteo_hecho_de_campana: coloreada
    # sigue MAS a A que la plana, direccion contraria) -- se citan ambos
    # tal cual, sin fundirlos en una sola afirmacion.
    for pareja in ("fuego_hecho_de_vidrio", "goteo_hecho_de_campana"):
        plana = corr_env[(corr_env["pareja"] == pareja) & (corr_env["region"] == "graves")
                         & (corr_env["method"] == "chimera_plana")].iloc[0]
        color = corr_env[(corr_env["pareja"] == pareja) & (corr_env["region"] == "graves")
                         & (corr_env["method"] == "chimera") & (corr_env["color_mix"] == 1.0)].iloc[0]
        add(f"corr_envolvente_graves_{pareja}_plana", "correlacion_envolvente",
           f"Correlacion de envolvente (banda graves <500Hz) del blend con el "
           f"padre A, chimera_plana, {pareja}.",
           round(float(plana["corr_con_a"]), 4), "correlacion (max +-20ms)",
           "correlacion_envolvente.csv",
           f"pareja == '{pareja}' and region == 'graves' and method == 'chimera_plana'")
        add(f"corr_envolvente_graves_{pareja}_coloreada", "correlacion_envolvente",
           f"Correlacion de envolvente (banda graves <500Hz) del blend con el "
           f"padre A, chimera coloreada (color_mix=1.0, ~V11), {pareja}. "
           + ("Coincide con la direccion del brief (coloreada sigue MENOS a A "
              "que la plana)." if pareja == "fuego_hecho_de_vidrio" else
              "Direccion CONTRARIA a la citada en el brief para esta pareja "
              "(coloreada sigue MAS a A que la plana, no menos) -- el 0,655 "
              "del brief es la plana de ESTA pareja, no la coloreada; el "
              "0,121 del brief es la coloreada de fuego_hecho_de_vidrio, no "
              "de esta pareja. El brief mezclaba dos parejas en una sola "
              "afirmacion."),
           round(float(color["corr_con_a"]), 4), "correlacion (max +-20ms)",
           "correlacion_envolvente.csv",
           f"pareja == '{pareja}' and region == 'graves' and method == 'chimera' "
           f"and color_mix == 1.0")

    return pd.DataFrame(claims)


# ====================================================================
# Figura F6 -- dos paneles, un flotante
# ====================================================================

# Marcadores/estilos por pareja: distinguibles en escala de grises (forma +
# trazo), no solo por color -- pedido explicito del brief para figuras a
# tamano de una columna.
_PAREJA_STYLE = {
    "trueno_hecho_de_agua":    dict(marker="o", ls="-",  color="#1b1b1b"),
    "fuego_hecho_de_vidrio":   dict(marker="s", ls="--", color="#555555"),
    "canica_hecha_de_fuego":   dict(marker="^", ls="-.", color="#8a8a8a"),
    "trueno_hecho_de_canica":  dict(marker="D", ls=":",  color="#1b1b1b"),
    "oceano_hecho_de_campana": dict(marker="v", ls="-",  color="#c1272d"),  # EMPEORA: resaltada
    "goteo_hecho_de_campana":  dict(marker="P", ls="--", color="#8a8a8a"),
}
_PAREJA_LABEL = {
    "trueno_hecho_de_agua": "trueno x agua",
    "fuego_hecho_de_vidrio": "fuego x vidrio",
    "canica_hecha_de_fuego": "canica x fuego",
    "trueno_hecho_de_canica": "trueno x canica",
    "oceano_hecho_de_campana": "oceano x campana",
    "goteo_hecho_de_campana": "goteo x campana",
}


def _es(x: float, nd: int = 2, signed: bool = False) -> str:
    """Formatea un numero con coma decimal (convencion del paper en
    espanol) -- p.ej. _es(0.4216, 2, signed=True) -> '+0,42'."""
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace(".", ",")


def _comma_formatter(v, _pos=None) -> str:
    return _es(v, 2)


def _panel_a(ax, resumen: pd.DataFrame, corr_fci_crest: float) -> None:
    sub = resumen.dropna(subset=["fci", "crest_db"])
    ax.axhspan(10, 20, color="#2e7d32", alpha=0.12, zorder=0, label="cresta sana (10-20 dB)")
    others = sub[~sub["grupo"].isin(["suma_ancla", "baseline_v11"])]
    ax.scatter(others["fci"], others["crest_db"], s=6, color="#9a9a9a", alpha=0.35,
              linewidths=0, zorder=1, label=f"candidatos (n={len(sub)})")
    for grupo, marker, fc, label in (
            ("suma_ancla", "*", "#1b5e20", "suma_ancla (6)"),
            ("baseline_v11", "X", "#c1272d", "baseline_v11 (6)")):
        g = sub[sub["grupo"] == grupo]
        ax.scatter(g["fci"], g["crest_db"], s=85, marker=marker, facecolor=fc,
                  edgecolor="black", linewidths=0.6, zorder=3, label=label)
    # Solo se etiqueta con nombre el punto citado explicitamente en el texto
    # (baseline_v11 de trueno_hecho_de_agua, ~47dB de cresta, el caso
    # patologico de "dos capas" que ancla la lectura del panel) -- marcar
    # los 12 puntos (suma_ancla/baseline_v11) sin nombrar cada uno evita el
    # amontonamiento de 6+6 etiquetas que se pisaban entre si.
    cited = sub[(sub["pareja"] == "trueno_hecho_de_agua") & (sub["grupo"] == "baseline_v11")].iloc[0]
    ax.annotate(f"baseline_v11\ntrueno x agua\n({_es(cited['crest_db'], 1)} dB)",
               xy=(cited["fci"], cited["crest_db"]),
               xycoords="data", xytext=(0.50, 0.72), textcoords="axes fraction",
               fontsize=6.3, ha="center", va="center", color="#c1272d",
               bbox=dict(boxstyle="round", fc="white", ec="#c1272d", lw=0.6, alpha=0.95),
               arrowprops=dict(arrowstyle="-", color="#c1272d", lw=0.7))
    ax.set_ylim(top=sub["crest_db"].max() + 4)
    ax.margins(x=0.12)
    ax.xaxis.set_major_formatter(_comma_formatter)
    ax.set_xlabel("FCI (indice compuesto de fusion)")
    ax.set_ylabel("crest factor (dB)")
    ax.set_title(f"(a) el FCI premia la cresta alta ($r$ = {_es(corr_fci_crest, 2, signed=True)})",
                fontsize=10)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.9, borderaxespad=0.3)


def _panel_b(ax, resumen: pd.DataFrame) -> None:
    x = [0, 1]
    for pareja, _env, _fine in PAREJAS_V12:
        g = resumen[(resumen["pareja"] == pareja)].groupby("align")["sso"].mean()
        y = [float(g.loc[False]), float(g.loc[True])]
        st = _PAREJA_STYLE[pareja]
        emphasize = pareja == "oceano_hecho_de_campana"
        ax.plot(x, y, marker=st["marker"], ls=st["ls"], color=st["color"],
               lw=2.4 if emphasize else 1.4, ms=7 if emphasize else 5.5,
               zorder=3 if emphasize else 2, label=_PAREJA_LABEL[pareja])
        ax.annotate(_PAREJA_LABEL[pareja], (1, y[1]), fontsize=6.3, xytext=(5, 0),
                   textcoords="offset points", va="center",
                   fontweight="bold" if emphasize else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(["sin alinear", "alineando"])
    ax.set_xlim(-0.15, 1.95)
    ax.margins(y=0.15)
    ax.yaxis.set_major_formatter(_comma_formatter)
    ax.set_ylabel("SSO (solape espectral)")
    ax.set_title("(b) alinear registro mejora el solape -- salvo si ya solapaban",
                fontsize=10)


def _panel_footer(ax, overshoots: dict[str, dict], n_aligned: int, n_clamped: int) -> None:
    """Pie de figura (fuera de los ejes de datos, en su propio subplot sin
    marco, a todo el ancho): los sobredisparos de registro por padre y el
    conteo de candidatos que topo con el limite de rango -- LEIDOS del
    mismo `overshoots`/`n_aligned`/`n_clamped` que build_evidencia_paper()
    calcula desde resumen.csv (compute_overshoots/compute_clamped_stats),
    nunca literales aparte: si resumen.csv cambiara, la figura no puede
    quedarse citando un numero viejo."""
    ax.axis("off")
    o = overshoots
    note = (
       f"Sobredisparo del padre movido: vidrio "
       f"{_es(o['overshoot_vidrio']['pct'], 1, True)}%; campana_tela "
       f"{_es(o['overshoot_campana_tela_registro_bajo']['pct'], 1, True)}%/"
       f"{_es(o['overshoot_campana_tela_registro_alto']['pct'], 1, True)}%; canica "
       f"{_es(o['overshoot_canica_registro_bajo']['pct'], 1, True)}% (registro bajo)/"
       f"{_es(o['overshoot_canica_registro_alto']['pct'], 1, True)}% (alto); goteo "
       f"{_es(o['overshoot_goteo']['pct'], 1, True)}%. {n_clamped}/{n_aligned} candidatos "
       f"con align=True topo con el limite de rango (register_clamped)."
    )
    wrapped = "\n".join(textwrap.wrap(note, width=118))
    ax.text(0.5, 0.5, wrapped, transform=ax.transAxes, fontsize=7.2,
           va="center", ha="center",
           bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb", alpha=0.9))


def make_figure(resumen: pd.DataFrame, corr_fci_crest: float) -> None:
    """Layout horizontal (2 paneles lado a lado, 1 fila x 2 columnas): el
    paper es a una columna ancha (~16cm util en A4), no a dos columnas
    estrechas -- un flotante apilado verticalmente se comia media pagina
    (pedido de la revision del coordinador). El pie con los sobredisparos
    va en una franja fina a todo el ancho, debajo de ambos paneles."""
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    overshoots = compute_overshoots(resumen)
    n_aligned, n_clamped = compute_clamped_stats(resumen)
    fig = plt.figure(figsize=(10.4, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.14))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_foot = fig.add_subplot(gs[1, :])
    _panel_a(ax_a, resumen, corr_fci_crest)
    _panel_b(ax_b, resumen)
    _panel_footer(ax_foot, overshoots, n_aligned, n_clamped)
    # Sin titulo general dentro de la imagen: el pie de figura ("Fig. 6 ...")
    # va en LaTeX via \caption{}, fuera del PNG -- ponerlo aqui tambien
    # lo duplicaria (pedido de la revision del coordinador). Los titulos
    # (a)/(b) de cada panel si se quedan.
    # metadata explicita y vacia de campos libres: ningun backend debe
    # filtrar ruta/usuario de la maquina en el PNG (paper a doble ciego).
    fig.savefig(FIG_PATH, metadata={"Software": "matplotlib", "Author": "", "Comment": ""})
    plt.close(fig)
    print(f"Figura: {FIG_PATH}")


# ====================================================================
# Escritura
# ====================================================================

def generate() -> dict:
    resumen = load_resumen()
    bandas = build_bandas_huerfanas()
    corr_env = build_correlacion_envolvente()
    baseline_crest = build_baseline_v11_crest()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bandas.to_csv(BANDAS_PATH, index=False)
    corr_env.to_csv(CORR_PATH, index=False)
    baseline_crest.to_csv(BASELINE_CREST_PATH, index=False)

    evidencia = build_evidencia_paper(resumen, bandas, baseline_crest, corr_env)
    evidencia.to_csv(EVIDENCIA_PATH, index=False)

    corr_fci_crest = float(evidencia.loc[evidencia["id_claim"] == "corr_fci_crest_global",
                                         "valor"].iloc[0])
    make_figure(resumen, corr_fci_crest)

    print(f"bandas_huerfanas.csv: {len(bandas)} filas -> {BANDAS_PATH}")
    print(f"correlacion_envolvente.csv: {len(corr_env)} filas -> {CORR_PATH}")
    print(f"baseline_v11_crest.csv: {len(baseline_crest)} filas -> {BASELINE_CREST_PATH}")
    print(f"evidencia_paper.csv: {len(evidencia)} filas -> {EVIDENCIA_PATH}")
    return {"resumen": resumen, "bandas": bandas, "corr_env": corr_env,
           "baseline_crest": baseline_crest, "evidencia": evidencia}


# ====================================================================
# --check
# ====================================================================

def check_sso_nunique(resumen: pd.DataFrame) -> list[str]:
    """Guardarraíl 1 del brief: sso tiene nunique==1 por (pareja,align) --
    contra reintroducir el bug de sso_cache por id() (V12-F4 arreglo 2)."""
    bad = []
    g = resumen.groupby(["pareja", "align"])["sso"].nunique()
    for (pareja, align), n in g.items():
        ok = n == 1
        print(f"  sso nunique pareja={pareja} align={align}: {n} {'OK' if ok else 'FALLO'}")
        if not ok:
            bad.append(f"{pareja}/{align}")
    print(f"check1 (sso nunique==1 por pareja,align): {'OK' if not bad else 'FALLOS: ' + str(bad)}")
    return bad


def check_corr_fci_crest(resumen: pd.DataFrame, evidencia: pd.DataFrame) -> list[str]:
    """Guardarraíl 2: corr(fci,crest_db) recomputado desde resumen.csv
    coincide con el valor citado en evidencia_paper.csv a 1e-4."""
    sub = resumen.dropna(subset=["fci", "crest_db"])
    recomputed = float(sub["fci"].corr(sub["crest_db"]))
    cited = float(evidencia.loc[evidencia["id_claim"] == "corr_fci_crest_global", "valor"].iloc[0])
    ok = abs(recomputed - cited) < 1e-4
    print(f"  recomputado={recomputed:.6f} citado={cited:.6f} diff={abs(recomputed - cited):.2e}")
    print(f"check2 (corr(fci,crest_db) recomputado == citado a 1e-4): {'OK' if ok else 'FALLO'}")
    return [] if ok else ["corr_fci_crest"]


def check_evidencia_resuelve() -> list[str]:
    """Guardarraíl 3: cada id_claim de evidencia_paper.csv resuelve a
    filas reales de su fuente_csv aplicando su filtro (pandas .query()).
    La unica excepcion es la fila que afirma "cero candidatos" (register_
    clamped): ahi CERO coincidencias es la propia afirmacion, no un fallo
    de resolucion."""
    evidencia = pd.read_csv(EVIDENCIA_PATH)
    fuentes = {"resumen.csv": load_resumen(), "bandas_huerfanas.csv": pd.read_csv(BANDAS_PATH),
              "correlacion_envolvente.csv": pd.read_csv(CORR_PATH),
              "baseline_v11_crest.csv": pd.read_csv(BASELINE_CREST_PATH)}
    bad = []
    for _, row in evidencia.iterrows():
        df = fuentes[row["fuente_csv"]]
        try:
            matched = df.query(row["filtro_o_fila"])
        except Exception as exc:  # noqa: BLE001
            print(f"  {row['id_claim']}: filtro invalido ({exc}) FALLO")
            bad.append(row["id_claim"])
            continue
        expect_zero = row["id_claim"] == "register_clamped_cero_de_494"
        ok = (len(matched) == 0) if expect_zero else (len(matched) >= 1)
        print(f"  {row['id_claim']}: {len(matched)} filas en {row['fuente_csv']} "
             f"{'OK' if ok else 'FALLO'}")
        if not ok:
            bad.append(row["id_claim"])
    print(f"check3 (cada id_claim resuelve filas reales): {'OK' if not bad else 'FALLOS: ' + str(bad)}")
    return bad


def check_figure() -> list[str]:
    """Guardarraíl 4: la figura existe, es un PNG valido y no esta vacia."""
    bad = []
    if not FIG_PATH.exists():
        print(f"  {FIG_PATH} no existe FALLO")
        return ["figure_missing"]
    data = FIG_PATH.read_bytes()
    is_png = data[:8] == b"\x89PNG\r\n\x1a\n"
    non_empty = len(data) > 1000
    print(f"  {FIG_PATH}: {len(data)} bytes, PNG valido={is_png}, no vacio={non_empty}")
    if not (is_png and non_empty):
        bad.append("figure_invalid")
    print(f"check4 (figura PNG valida y no vacia): {'OK' if not bad else 'FALLOS: ' + str(bad)}")
    return bad


def check_no_leaks() -> list[str]:
    """Anonimato (doble ciego): ningun CSV generado contiene rutas
    absolutas ni el usuario de la maquina."""
    import getpass
    bad = []
    user = getpass.getuser()
    for p in (BANDAS_PATH, CORR_PATH, BASELINE_CREST_PATH, EVIDENCIA_PATH):
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        leaks = []
        if "/home/" in text:
            leaks.append("/home/")
        if user and user in text:
            leaks.append(f"usuario:{user}")
        print(f"  {p}: {'OK' if not leaks else 'FUGA: ' + str(leaks)}")
        if leaks:
            bad.append(str(p))
    print(f"check5 (sin rutas absolutas ni usuario en los CSV): {'OK' if not bad else 'FALLOS: ' + str(bad)}")
    return bad


def check() -> int:
    print("(--check genera los artefactos primero -- son la entrada de los guardarrailes)")
    generate()
    resumen = load_resumen()
    evidencia = pd.read_csv(EVIDENCIA_PATH)
    all_failures: list[str] = []
    all_failures += check_sso_nunique(resumen)
    all_failures += check_corr_fci_crest(resumen, evidencia)
    all_failures += check_evidencia_resuelve()
    all_failures += check_figure()
    all_failures += check_no_leaks()
    if all_failures:
        print(f"\n{len(all_failures)} fallos: {all_failures}")
        return 1
    print("\ntodos los checks OK")
    return 0


# ====================================================================
# main
# ====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="genera + guardarrailes de auditoria")
    args = ap.parse_args()
    if args.check:
        return check()
    generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
