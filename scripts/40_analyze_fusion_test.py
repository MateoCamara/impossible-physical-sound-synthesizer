"""Analisis del mini-test perceptual A/B de fusion sonora (suma vs ganadora).

Cada CSV que descarga perceptual_test/form_fusion/index.html tiene formato:
  listener,experience,stim_file,pareja,condicion,unidad,fidelidad_mos,credibilidad_mos

donde `unidad` in {"un_objeto","dos_sonidos"} y `fidelidad_mos`/`credibilidad_mos`
son MOS 1..5. Este script:
  - Carga todos los CSV de perceptual_test/responses_fusion/.
  - Deduplica por oyente (si el mismo listener envio mas de un fichero, se
    queda con el mas reciente y avisa de los descartados).
  - Descarta oyentes con prefijo DEMO_/SINTETICO_ (datos de prueba del
    formulario o del propio --check de este script; nunca deben colarse
    en un analisis real).
  - Valida rangos y descarta filas invalidas informando de ello.
  - Calcula, por (pareja, condicion): proporcion de "un objeto" con IC-95
    de Wilson, media de fidelidad +- IC-95 (t de Student), media de
    credibilidad +- IC-95 (t), y n_listeners. -> resumen.csv
  - Calcula contrastes pareados suma vs fusion por pareja (McNemar exacto
    para unidad, Wilcoxon signed-rank para cada MOS). -> contrastes.csv
  - Guarda un boxplot/barplot PNG (backend Agg, patron de scripts/17).

*** LIMITACION ESTADISTICA (leer antes de citar nada de contrastes.csv) ***
El diseno tiene 3 parejas x 2 condiciones x 10-15 oyentes. Es una n MUY
pequena para un contraste pareado: con tan pocos oyentes, McNemar exacto y
Wilcoxon signed-rank estan claramente infrapotenciados (con menos de ~6
diferencias no nulas, ni el mejor caso posible -todas las diferencias del
mismo signo- llega a p<0.05 en el signed-rank exacto; ver la columna
`*_wilcoxon_p_min_posible` de contrastes.csv, calculada para el n real de
cada fila). Por eso TODAS las columnas de p-valor de contrastes.csv llevan
el sufijo `_PRELIMINAR` en el propio nombre de columna: no son el resultado
citable del mini-test, y un p~=0.05 en esta n NO debe convertirse en una
claim del paper (que ademas va a doble ciego). El resultado PRINCIPAL y
citable son los intervalos de confianza de resumen.csv (Wilson 95% para la
proporcion de "un objeto", t de Student 95% para las medias MOS): funcionan
razonablemente con n pequena (se ensanchan, no rompen) y no dependen de que
el efecto cruce un umbral binario de significacion.

Recordatorio de diseno (no es un bug si aparece): la hipotesis central del
trabajo es que alinear el registro de los dos padres mejora que se perciba
UN objeto pero puede destruir la identidad de la materia. Si el resultado
es "fusion sube unidad y BAJA fidelidad", eso ES el resultado, no un fallo
del pipeline ni del analisis.

Uso:
    PYTHONPATH=. .venv/bin/python scripts/40_analyze_fusion_test.py
    PYTHONPATH=. .venv/bin/python scripts/40_analyze_fusion_test.py --check
    PYTHONPATH=. .venv/bin/python scripts/40_analyze_fusion_test.py --responses /otro/path --out /otro/out
"""
from __future__ import annotations

import argparse
import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import t as t_dist
from scipy.stats import wilcoxon

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


RESPONSES_DIR = Path("perceptual_test/responses_fusion")
OUT_DIR = Path("results/perceptual_fusion")

REQUIRED_COLUMNS = ["listener", "experience", "stim_file", "pareja", "condicion",
                    "unidad", "fidelidad_mos", "credibilidad_mos"]
UNIDAD_VALUES = {"un_objeto", "dos_sonidos"}
MOS_VALUES = {1, 2, 3, 4, 5}

# Prefijos que marcan datos NO reales: el modo demo del formulario
# (perceptual_test/form_fusion/index.html?demo=1) fuerza listener="DEMO_...",
# y el fixture sintetico de este mismo script usa "SINTETICO_...". Si
# aparecen en perceptual_test/responses_fusion/ (p.ej. alguien envia el CSV
# de demo por error) se descartan con un aviso en vez de contaminar el
# resumen citable.
FAKE_LISTENER_PREFIXES = ("DEMO_", "SINTETICO_")

Z_95 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """IC-95 de Wilson para una proporcion binomial k/n.

    Preferido sobre el IC normal (Wald) para n pequena o proporciones
    cerca de 0/1 -- justo el regimen de este test (n=10-15 oyentes).
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / denom
    margen = (z * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))) / denom
    lo = max(0.0, centro - margen)
    hi = min(1.0, centro + margen)
    return (float(lo), float(hi))


def t_ci_mean(values) -> tuple[float, float, float, int]:
    """Media +- IC-95 (t de Student). Devuelve (media, lo, hi, n).

    Con n<2 el IC no esta definido (hace falta al menos 2 puntos para
    estimar una desviacion tipica muestral): lo=hi=nan.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    media = float(arr.mean())
    if n < 2:
        return (media, float("nan"), float("nan"), n)
    sem = arr.std(ddof=1) / np.sqrt(n)
    margen = float(t_dist.ppf(0.975, df=n - 1) * sem)
    return (media, media - margen, media + margen, n)


# ====================================================================
# Carga, deduplicado y validacion
# ====================================================================

_TS_RE = re.compile(r"(\d{10,})\.csv$")


def _recency_key(path: Path) -> float:
    """Clave de orden para deduplicar por oyente: si el nombre de fichero
    trae un timestamp epoch-ms (patron del formulario,
    respuestas_fusion_<listener>_<epoch_ms>.csv), se usa ese numero; si no
    coincide el patron, se cae al mtime del fichero en disco.
    """
    m = _TS_RE.search(path.name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return path.stat().st_mtime


def load_all_responses(d: Path) -> pd.DataFrame:
    """Carga todos los CSV de `d`. Ficheros a los que les falte alguna
    columna requerida se descartan enteros, avisando. No deduplica ni
    valida rangos (eso lo hacen dedupe_by_listener/validate_ranges).
    """
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(d.glob("*.csv"))
    if not csv_files:
        print(f"!! No hay CSVs en {d}. Esperando respuestas del formulario.")
        return pd.DataFrame()
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"!! Error leyendo {f}: {e}")
            continue
        faltan = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if faltan:
            print(f"!! {f.name}: faltan columnas {faltan}; se descarta el fichero entero.")
            continue
        df["src_file"] = f.name
        df["_recency"] = _recency_key(f)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["listener"] = out["listener"].astype(str).str.strip()
    return out


def dedupe_by_listener(df: pd.DataFrame) -> pd.DataFrame:
    """Un oyente = una sesion. Si el mismo `listener` aparece en mas de un
    fichero fuente (reenvio, correccion, etc.), nos quedamos con TODAS las
    filas del fichero mas reciente para ese oyente (ver _recency_key) y
    descartamos las del resto, avisando. No se promedian sesiones repetidas.
    """
    if df.empty:
        return df
    idx_ganador = df.groupby("listener")["_recency"].idxmax()
    ganador_file = df.loc[idx_ganador].set_index("listener")["src_file"]
    df = df.copy()
    df["_ganador_file"] = df["listener"].map(ganador_file)
    keep_mask = df["src_file"] == df["_ganador_file"]

    descartadas = df.loc[~keep_mask, ["listener", "src_file"]].drop_duplicates()
    if not descartadas.empty:
        print(f"!! Deduplicando por oyente: se descartan envios mas antiguos de "
              f"{len(descartadas)} (oyente, fichero):")
        for _, r in descartadas.iterrows():
            print(f"   - oyente={r['listener']!r} fichero={r['src_file']!r}")
    return df.loc[keep_mask].drop(columns=["_recency", "_ganador_file"])


def drop_fake_listeners(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta filas cuyo listener empiece por un prefijo de datos NO
    reales (ver FAKE_LISTENER_PREFIXES), avisando en voz alta.
    """
    if df.empty:
        return df
    prefijos = tuple(p.upper() for p in FAKE_LISTENER_PREFIXES)
    es_fake = df["listener"].str.upper().str.startswith(prefijos)
    if es_fake.any():
        oyentes = sorted(df.loc[es_fake, "listener"].unique())
        print(f"!! AVISO: {int(es_fake.sum())} fila(s) de oyentes con prefijo de datos "
              f"NO reales (modo demo del formulario o fixture sintetico de --check): "
              f"{oyentes}. Se descartan del analisis. Si esto aparece con datos "
              f"genuinos hay una contaminacion que investigar antes de fiarse de "
              f"resumen.csv.")
    return df.loc[~es_fake].copy()


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Valida `unidad` en UNIDAD_VALUES y los MOS en 1..5 (enteros).
    Descarta filas invalidas informando cuantas y por que motivo.

    Tambien protege el denominador del IC de Wilson: si tras deduplicar
    por oyente sigue habiendo mas de una fila por (listener,pareja,
    condicion) -que no deberia pasar, cada oyente responde cada estimulo
    una vez- el n de Wilson se inflaria en silencio. Se avisa y se
    conserva solo la primera fila de cada grupo.
    """
    if df.empty:
        return df
    out = df.copy()
    out["fidelidad_mos"] = pd.to_numeric(out["fidelidad_mos"], errors="coerce")
    out["credibilidad_mos"] = pd.to_numeric(out["credibilidad_mos"], errors="coerce")

    mask_unidad = out["unidad"].isin(UNIDAD_VALUES)
    mask_fid = out["fidelidad_mos"].isin(MOS_VALUES)
    mask_cred = out["credibilidad_mos"].isin(MOS_VALUES)
    mask_ok = mask_unidad & mask_fid & mask_cred

    n_malas = int((~mask_ok).sum())
    if n_malas:
        print(f"!! Descartando {n_malas} fila(s) invalida(s):")
        print(f"   - unidad fuera de {UNIDAD_VALUES}: {int((~mask_unidad).sum())}")
        print(f"   - fidelidad_mos fuera de 1..5: {int((~mask_fid).sum())}")
        print(f"   - credibilidad_mos fuera de 1..5: {int((~mask_cred).sum())}")

    out = out.loc[mask_ok].copy()

    dup_mask = out.duplicated(subset=["listener", "pareja", "condicion"], keep="first")
    if dup_mask.any():
        print(f"!! AVISO: {int(dup_mask.sum())} fila(s) duplicada(s) de "
              f"(listener,pareja,condicion) tras deduplicar por oyente -- posible "
              f"bug en el CSV de origen. Nos quedamos con la primera de cada grupo "
              f"para no inflar el denominador del IC de Wilson.")
        out = out.loc[~dup_mask].copy()

    return out.drop(columns=["src_file"], errors="ignore")


# ====================================================================
# Resultado principal: resumen con intervalos de confianza
# ====================================================================

RESUMEN_COLS = [
    "pareja", "condicion", "n_listeners", "prop_un_objeto",
    "prop_un_objeto_ic95_low", "prop_un_objeto_ic95_high",
    "fidelidad_mos_mean", "fidelidad_mos_ic95_low", "fidelidad_mos_ic95_high",
    "credibilidad_mos_mean", "credibilidad_mos_ic95_low", "credibilidad_mos_ic95_high",
]


def resumen(df: pd.DataFrame) -> pd.DataFrame:
    """Por (pareja, condicion): proporcion de 'un objeto' con IC-95 de
    Wilson, media de fidelidad +- IC-95 (t), media de credibilidad +-
    IC-95 (t), y n_listeners. RESULTADO PRINCIPAL Y CITABLE del mini-test.
    """
    filas = []
    for (pareja, condicion), sub in df.groupby(["pareja", "condicion"]):
        n = len(sub)
        k = int((sub["unidad"] == "un_objeto").sum())
        p_lo, p_hi = wilson_ci(k, n)
        fid_m, fid_lo, fid_hi, _ = t_ci_mean(sub["fidelidad_mos"])
        cred_m, cred_lo, cred_hi, _ = t_ci_mean(sub["credibilidad_mos"])
        filas.append(dict(
            pareja=pareja, condicion=condicion, n_listeners=n,
            prop_un_objeto=round(k / n, 4) if n else float("nan"),
            prop_un_objeto_ic95_low=round(p_lo, 4),
            prop_un_objeto_ic95_high=round(p_hi, 4),
            fidelidad_mos_mean=round(fid_m, 3),
            fidelidad_mos_ic95_low=round(fid_lo, 3),
            fidelidad_mos_ic95_high=round(fid_hi, 3),
            credibilidad_mos_mean=round(cred_m, 3),
            credibilidad_mos_ic95_low=round(cred_lo, 3),
            credibilidad_mos_ic95_high=round(cred_hi, 3),
        ))
    return (pd.DataFrame(filas, columns=RESUMEN_COLS)
            .sort_values(["pareja", "condicion"]).reset_index(drop=True))


# ====================================================================
# Contrastes pareados -- PRELIMINARES (ver docstring del modulo)
# ====================================================================

CONTRASTES_COLS = [
    "pareja", "condicion_a", "condicion_b", "n_pares",
    "unidad_discordantes_b", "unidad_discordantes_c", "unidad_mcnemar_p_PRELIMINAR",
    "fidelidad_n_no_cero", "fidelidad_wilcoxon_stat_PRELIMINAR",
    "fidelidad_wilcoxon_p_PRELIMINAR", "fidelidad_wilcoxon_p_min_posible",
    "credibilidad_n_no_cero", "credibilidad_wilcoxon_stat_PRELIMINAR",
    "credibilidad_wilcoxon_p_PRELIMINAR", "credibilidad_wilcoxon_p_min_posible",
    "nota",
]


def _wilcoxon_min_p(n_no_cero: int) -> float:
    """Techo de potencia: el menor p de dos colas que el signed-rank
    EXACTO puede dar con `n_no_cero` diferencias no nulas, si TODAS
    tuvieran el mismo signo (el mejor caso posible para este n). Sirve
    para leer si un p_PRELIMINAR observado siquiera podia haber sido
    significativo con esta n.
    """
    if n_no_cero <= 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p_min = wilcoxon(np.ones(n_no_cero))
    return float(p_min)


def contrastes(df: pd.DataFrame) -> pd.DataFrame:
    """Contrastes pareados suma vs fusion por pareja: McNemar EXACTO
    (unidad, via binomtest exacto sobre los pares discordantes) y
    Wilcoxon signed-rank (cada MOS). *** PRELIMINARES ***: ver el
    docstring del modulo -- no citar sin leerlo.
    """
    filas = []
    for pareja, sub in df.groupby("pareja"):
        condiciones = sorted(sub["condicion"].unique())
        fila = dict(pareja=pareja)
        notas = []

        if len(condiciones) != 2:
            fila.update(condicion_a=None, condicion_b=None, n_pares=0)
            notas.append(f"se esperaban 2 condiciones, hay {len(condiciones)}: {condiciones}")
            fila["nota"] = "; ".join(notas)
            filas.append(fila)
            continue

        ca, cb = condiciones
        wa = sub[sub["condicion"] == ca].set_index("listener")
        wb = sub[sub["condicion"] == cb].set_index("listener")
        listeners_comunes = sorted(set(wa.index) & set(wb.index))
        n_pares = len(listeners_comunes)
        fila.update(condicion_a=ca, condicion_b=cb, n_pares=n_pares)

        if n_pares == 0:
            notas.append("sin oyentes con ambas condiciones para esta pareja")
            fila["nota"] = "; ".join(notas)
            filas.append(fila)
            continue

        a_un = (wa.loc[listeners_comunes, "unidad"] == "un_objeto").to_numpy()
        b_un = (wb.loc[listeners_comunes, "unidad"] == "un_objeto").to_numpy()
        b_disc = int(np.sum(a_un & ~b_un))  # condicion_a=un_objeto, condicion_b=dos_sonidos
        c_disc = int(np.sum(~a_un & b_un))  # condicion_a=dos_sonidos, condicion_b=un_objeto
        fila["unidad_discordantes_b"] = b_disc
        fila["unidad_discordantes_c"] = c_disc
        if b_disc + c_disc == 0:
            fila["unidad_mcnemar_p_PRELIMINAR"] = float("nan")
            notas.append("mcnemar: sin pares discordantes en unidad")
        else:
            pv = binomtest(min(b_disc, c_disc), b_disc + c_disc, 0.5,
                            alternative="two-sided").pvalue
            fila["unidad_mcnemar_p_PRELIMINAR"] = round(float(pv), 4)

        for metric, nombre in [("fidelidad_mos", "fidelidad"), ("credibilidad_mos", "credibilidad")]:
            va = wa.loc[listeners_comunes, metric].to_numpy(dtype=float)
            vb = wb.loc[listeners_comunes, metric].to_numpy(dtype=float)
            diffs = va - vb
            n_no_cero = int(np.sum(diffs != 0))
            fila[f"{nombre}_n_no_cero"] = n_no_cero
            if n_no_cero == 0:
                fila[f"{nombre}_wilcoxon_stat_PRELIMINAR"] = float("nan")
                fila[f"{nombre}_wilcoxon_p_PRELIMINAR"] = float("nan")
                fila[f"{nombre}_wilcoxon_p_min_posible"] = float("nan")
                notas.append(f"wilcoxon {nombre}: todas las diferencias son 0")
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stat, pv = wilcoxon(va, vb, zero_method="wilcox")
            fila[f"{nombre}_wilcoxon_stat_PRELIMINAR"] = round(float(stat), 3)
            fila[f"{nombre}_wilcoxon_p_PRELIMINAR"] = round(float(pv), 4)
            fila[f"{nombre}_wilcoxon_p_min_posible"] = round(_wilcoxon_min_p(n_no_cero), 4)

        fila["nota"] = "; ".join(notas)
        filas.append(fila)

    out = pd.DataFrame(filas)
    for c in CONTRASTES_COLS:
        if c not in out.columns:
            out[c] = None
    return out[CONTRASTES_COLS]


# ====================================================================
# Figura
# ====================================================================

def plot_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    """3 filas x N parejas: proporcion de 'un objeto' (barra + IC95
    Wilson), boxplot de fidelidad y boxplot de credibilidad. Backend Agg,
    mismo patron que scripts/17_analyze_perceptual.py.
    """
    parejas = sorted(df["pareja"].unique())
    n = max(1, len(parejas))
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 10), squeeze=False)
    for c, pareja in enumerate(parejas):
        sub = df[df["pareja"] == pareja]
        condiciones = sorted(sub["condicion"].unique())

        ax0 = axes[0, c]
        props, los, his = [], [], []
        for cond in condiciones:
            s = sub[sub["condicion"] == cond]
            k = int((s["unidad"] == "un_objeto").sum())
            p_lo, p_hi = wilson_ci(k, len(s))
            props.append(k / len(s) if len(s) else float("nan"))
            los.append(p_lo)
            his.append(p_hi)
        x = np.arange(len(condiciones))
        yerr = [[max(0.0, p - lo) for p, lo in zip(props, los)],
                [max(0.0, hi - p) for p, hi in zip(props, his)]]
        ax0.bar(x, props, yerr=yerr, capsize=4, color="#5e8de2")
        ax0.set_xticks(x)
        ax0.set_xticklabels(condiciones, rotation=20)
        ax0.set_ylim(0, 1.05)
        ax0.set_ylabel("prop. 'un objeto' (IC95 Wilson)")
        ax0.set_title(pareja)

        for r, metric, label in [(1, "fidelidad_mos", "Fidelidad"),
                                  (2, "credibilidad_mos", "Credibilidad")]:
            ax = axes[r, c]
            data = [sub[sub["condicion"] == cond][metric].dropna().to_numpy()
                    for cond in condiciones]
            if not any(len(d) for d in data):
                ax.text(0.5, 0.5, "sin datos", ha="center", va="center")
                continue
            ax.boxplot(data, tick_labels=condiciones, showmeans=True)
            ax.set_ylim(0.5, 5.5)
            ax.set_ylabel(f"{label} (MOS 1-5)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ====================================================================
# --check: fixture sintetico en memoria
# ====================================================================

def _fixture(n_listeners: int, seed: int, escenario: str) -> pd.DataFrame:
    """Fixture SINTETICO en memoria para --check. NUNCA se escribe en
    perceptual_test/responses_fusion/: los listener id llevan el prefijo
    'SINTETICO_' (drop_fake_listeners() los filtraria si aparecieran ahi).

    escenario="monotona": la fusion sube en las 3 metricas -- el ejemplo
    literal del brief ("mayor proporcion de 'un objeto' y MOS mas altos").
    escenario="tension": la fusion sube unidad pero BAJA fidelidad -- la
    tension central del trabajo (alinear el registro de los dos padres
    mejora que se perciba UN objeto pero puede destruir la identidad de
    la materia). Prueba que el pipeline recupera un efecto de direccion
    MIXTA, no solo "todo mejora a la vez".
    """
    if escenario not in ("monotona", "tension"):
        raise ValueError(f"escenario desconocido: {escenario!r}")
    rng = np.random.default_rng(seed)
    parejas = ["SINTETICA_par1", "SINTETICA_par2", "SINTETICA_par3"]
    filas = []
    for i in range(n_listeners):
        listener = f"SINTETICO_L{i:02d}"
        experience = int(rng.integers(0, 4))
        for pareja in parejas:
            for condicion, es_fusion in [("suma", False), ("fusion", True)]:
                if escenario == "monotona":
                    p_un = 0.75 if es_fusion else 0.25
                    fid_mu = 4.0 if es_fusion else 2.3
                    cred_mu = 3.8 if es_fusion else 2.6
                else:  # tension
                    p_un = 0.80 if es_fusion else 0.20
                    fid_mu = 2.2 if es_fusion else 4.0  # invertido a proposito
                    cred_mu = 3.3 if es_fusion else 3.1  # sin efecto fuerte, a proposito
                unidad = "un_objeto" if rng.random() < p_un else "dos_sonidos"
                fid = int(np.clip(round(rng.normal(fid_mu, 0.6)), 1, 5))
                cred = int(np.clip(round(rng.normal(cred_mu, 0.6)), 1, 5))
                filas.append(dict(
                    listener=listener, experience=experience,
                    stim_file=f"{pareja}__{condicion}.wav",
                    pareja=pareja, condicion=condicion,
                    unidad=unidad, fidelidad_mos=fid, credibilidad_mos=cred,
                ))
    return pd.DataFrame(filas)


def _check_escenario(escenario: str, esperar_fidelidad_sube: bool) -> bool:
    df = _fixture(n_listeners=20, seed=42, escenario=escenario)

    r = resumen(df)
    ok_cols_r = list(r.columns) == RESUMEN_COLS

    c = contrastes(df)
    ok_cols_c = list(c.columns) == CONTRASTES_COLS

    ok_direccion = True
    for pareja in r["pareja"].unique():
        fusion = r[(r["pareja"] == pareja) & (r["condicion"] == "fusion")].iloc[0]
        suma = r[(r["pareja"] == pareja) & (r["condicion"] == "suma")].iloc[0]
        if not (fusion["prop_un_objeto"] > suma["prop_un_objeto"]):
            ok_direccion = False
        if esperar_fidelidad_sube:
            if not (fusion["fidelidad_mos_mean"] > suma["fidelidad_mos_mean"]):
                ok_direccion = False
        else:
            if not (fusion["fidelidad_mos_mean"] < suma["fidelidad_mos_mean"]):
                ok_direccion = False

    ok_plot = True
    with tempfile.TemporaryDirectory() as td:
        try:
            out_png = Path(td) / "boxplot_check.png"
            plot_boxplot(df, out_png)
            ok_plot = out_png.exists() and out_png.stat().st_size > 0
        except Exception as e:
            print(f"!! plot_boxplot fallo en escenario={escenario}: {e}")
            ok_plot = False

    ok = ok_cols_r and ok_cols_c and ok_direccion and ok_plot
    print(f"  escenario={escenario}: cols_resumen={'OK' if ok_cols_r else 'FALLO'} "
          f"cols_contrastes={'OK' if ok_cols_c else 'FALLO'} "
          f"direccion={'OK' if ok_direccion else 'FALLO'} plot={'OK' if ok_plot else 'FALLO'}")
    return ok


def _check_datos_sucios() -> bool:
    """Fixture SUCIO a mano (no via _fixture()) para ejercitar en un solo
    paso las ramas degeneradas/defensivas que el fixture "limpio" de
    _check_escenario nunca toca: filas fuera de rango (unidad y MOS), una
    fila duplicada de (listener,pareja,condicion) que SOBREVIVE a
    dedupe_by_listener (porque esta en el mismo fichero simulado, no en
    ficheros distintos -- el guardarraiz de Wilson lo tiene que atrapar en
    validate_ranges), y una pareja con CERO pares discordantes en unidad Y
    todas las diferencias de MOS a cero (fuerza el guard de binomtest con
    n=0 y la rama "todas las diferencias son 0" de wilcoxon). Listener ids
    reales (sin prefijo fake) para que drop_fake_listeners no los quite
    antes de que validate_ranges pueda actuar.
    """
    filas = [
        # (listener, pareja, condicion, unidad, fidelidad_mos, credibilidad_mos)
        ("REAL01", "P1", "suma",    "un_objeto",   3, 3),  # valido
        ("REAL01", "P1", "suma",    "dos_sonidos", 1, 1),  # duplicado de (REAL01,P1,suma) -> se descarta
        ("REAL01", "P1", "fusion",  "un_objeto",   4, 4),  # valido
        ("REAL02", "P1", "suma",    "quizas",      3, 3),  # unidad invalida -> se descarta
        ("REAL02", "P1", "fusion",  "un_objeto",   4, 4),  # valido
        ("REAL03", "P1", "suma",    "un_objeto",   3, 3),  # valido
        ("REAL03", "P1", "fusion",  "un_objeto",   7, 0),  # MOS fuera de 1..5 (7 y 0) -> se descarta
        ("REAL04", "P2", "suma",    "un_objeto",   3, 3),  # valido; identica a su fila de fusion
        ("REAL04", "P2", "fusion",  "un_objeto",   3, 3),  # sin discordancia, sin diferencia de MOS
        ("REAL05", "P2", "suma",    "dos_sonidos", 2, 2),  # valido; identica a su fila de fusion
        ("REAL05", "P2", "fusion",  "dos_sonidos", 2, 2),  # idem
    ]
    df = pd.DataFrame(filas, columns=["listener", "pareja", "condicion", "unidad",
                                       "fidelidad_mos", "credibilidad_mos"])
    df["experience"] = 0
    df["stim_file"] = df["pareja"] + "__" + df["condicion"] + ".wav"
    df["src_file"] = "sucio.csv"  # un unico fichero simulado: dedupe_by_listener no debe tocar nada
    df["_recency"] = 1.0

    df = dedupe_by_listener(df)
    ok_dedupe_noop = len(df) == len(filas)

    df = drop_fake_listeners(df)
    ok_no_fake_dropped = len(df) == len(filas)

    df = validate_ranges(df)
    # Esperado: 2 filas fuera de rango (unidad "quizas"; MOS 7 y 0) + 1 fila
    # duplicada de (listener,pareja,condicion) = 3 descartadas, quedan 8.
    ok_n_final = len(df) == 8

    r = resumen(df)
    c = contrastes(df)

    fila_p2 = c[c["pareja"] == "P2"].iloc[0]
    ok_p2 = bool(
        fila_p2["n_pares"] == 2
        and pd.isna(fila_p2["unidad_mcnemar_p_PRELIMINAR"])
        and "sin pares discordantes" in fila_p2["nota"]
        and fila_p2["fidelidad_n_no_cero"] == 0
        and pd.isna(fila_p2["fidelidad_wilcoxon_p_PRELIMINAR"])
        and "fidelidad" in fila_p2["nota"] and "diferencias son 0" in fila_p2["nota"]
    )

    fila_p1 = c[c["pareja"] == "P1"].iloc[0]
    ok_p1 = bool(fila_p1["n_pares"] == 1)  # solo REAL01 conserva ambas condiciones tras limpiar

    ok = (ok_dedupe_noop and ok_no_fake_dropped and ok_n_final and ok_p1 and ok_p2
          and not r.empty and not c.empty)
    print(f"  datos_sucios: dedupe_noop={'OK' if ok_dedupe_noop else 'FALLO'} "
          f"fake_noop={'OK' if ok_no_fake_dropped else 'FALLO'} "
          f"n_tras_validar={'OK' if ok_n_final else 'FALLO'}({len(df)}/8) "
          f"P1_n_pares={'OK' if ok_p1 else 'FALLO'} "
          f"P2_degenerado={'OK' if ok_p2 else 'FALLO'}")
    return ok


def check() -> int:
    print("Fixture SINTETICO en memoria (nunca se escribe en el repo).")
    ok1 = _check_escenario("monotona", esperar_fidelidad_sube=True)
    ok2 = _check_escenario("tension", esperar_fidelidad_sube=False)

    df_fake = _fixture(5, 7, "monotona")
    filtrado = drop_fake_listeners(df_fake)
    ok3 = filtrado.empty
    print(f"  drop_fake_listeners() filtra el fixture completo (defensa en profundidad): "
          f"{'OK' if ok3 else 'FALLO'}")

    print("Fixture SUCIO a mano (rangos invalidos, duplicado en el mismo fichero, "
          "pareja degenerada sin discordancia ni diferencia de MOS):")
    ok4 = _check_datos_sucios()

    ok = ok1 and ok2 and ok3 and ok4
    print("smoke:", "OK" if ok else "FALLO")
    return 0 if ok else 1


# ====================================================================
# main
# ====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                     help="Smoke test con fixture sintetico en memoria; no toca disco real.")
    ap.add_argument("--responses", type=Path, default=RESPONSES_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.check:
        return check()

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_all_responses(args.responses)
    if df.empty:
        print("Sin datos. El analisis correra cuando haya CSVs en", args.responses)
        return 0

    df = dedupe_by_listener(df)
    df = drop_fake_listeners(df)
    df = validate_ranges(df)
    if df.empty:
        print("Sin filas validas tras la limpieza.")
        return 0

    print(f"Total filas validas: {len(df)} | oyentes unicos: {df['listener'].nunique()} "
          f"| parejas: {sorted(df['pareja'].unique())}")

    r = resumen(df)
    r.to_csv(args.out / "resumen.csv", index=False)
    print("\n=== Resumen por (pareja, condicion) -- RESULTADO PRINCIPAL (IC-95) ===")
    print(r.to_string(index=False))

    c = contrastes(df)
    c.to_csv(args.out / "contrastes.csv", index=False)
    print("\n=== Contrastes pareados -- *** PRELIMINARES, infrapotenciados: no citar sin leer "
          "el docstring del modulo *** ===")
    print(c.to_string(index=False))

    plot_boxplot(df, args.out / "boxplot.png")
    print(f"\nCSVs y figura en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
