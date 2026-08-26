"""Tarea 8 (F3): set grande de fusiones para la web de demo con sliders.

Genera `demo_fusion/` con 456 clips (76 por cada una de las 6 parejas de
`PAREJAS_V12`, ver `scripts/39_fusion_search.py:81`) mas un `manifest.json`
que es la INTERFAZ que consumira la web de la tarea siguiente: cada clip
lleva sus metricas al lado del fichero de audio, para que la web pueda
mostrar el numero junto al sonido.

Rejilla por pareja (76 clips):
  - 72 de la rejilla principal: n_bands en {2,4,6,8,12,16,24,32} x color_mix
    en {0, 0.125, 0.25, ..., 1.0} (9 valores, paso 0.125), align=True,
    warp=False, method="chimera", duration_s=6.0, seed=42.
  - 4 referencias en la celda ancla determinista
    (chimera_bands_heuristic(env_parent), color_mix=0.5):
      baseline_v11  -- config exacta de la demo V11: align=False,
                       color_mix=None, n_bands=heuristica, method=chimera.
      suma_ancla    -- method="suma" (align=True: SI afecta al audio, ver
                       nota de align en "suma" mas abajo).
      sin_alinear   -- la MISMA celda ancla (n_bands=heuristica,
                       color_mix=0.5, method=chimera) con align=False: el
                       A/B limpio del mecanismo de alineacion.
      plana         -- method="chimera_plana" en el n_bands ancla,
                       align=True (solo cambia el metodo respecto a la
                       celda ancla).

Decision documentada sobre align en las 4 referencias (el brief solo dice
"sin alinear" para baseline_v11 y nombra sin_alinear explicitamente para el
otro; suma_ancla y plana no mencionan align, asi que se lee como "todo lo
demas queda en el valor de la celda ancla", que es align=True -- la rejilla
principal completa fija align=True). Importante: align NO es inerte para
method="suma" (mueve el padre movil en el paso 1 de render_fusion ANTES de
la formula 0.6a+0.6b), asi que suma_ancla con align=True es audio
genuinamente distinto de un suma con align=False -- no es una decision solo
de etiquetado.

Reutilizacion (no reinventa la cadena de fusion, ver brief):
  - FusionSpec / render_fusion / _cached_parent de
    impossible_mix.physics.fusion_chain (motor terminado y verificado, NO
    se toca).
  - _match_loudness, _metric_parents, _composite_fusion_fast,
    _register_hz_memoized (aplicado como efecto secundario de importar el
    modulo) de scripts/39_fusion_search.py, cargado por ruta via
    importlib.util.spec_from_file_location -- "39_fusion_search" no es un
    identificador Python valido (empieza por digito), mismo patron ya
    usado en scripts/41_fusion_evidence.py (_load_script39). Importar ese
    modulo instala el memo de fusion_chain.register_hz sobre el ATRIBUTO
    del modulo importado (no toca fusion_chain.py en disco); sin el memo,
    con align=True fijo, register_hz se remide 2-3 veces por render y
    cuesta ~3-4 min extra por pareja en balde (ver docstring de
    _register_hz_memoized en 39). sso_cache (memo de analysis._sso, ~0.9s/
    llamada) se crea NUEVO por pareja aqui, igual que en run_pair de 39 --
    nunca se comparte entre parejas (evita el bug de colision por id()
    documentado en 39, ya corregido alli con clave por datos).

Metricas (crest_db/sso/fci) se miden SIEMPRE sobre el audio CRUDO que sale
de render_fusion, ANTES de _match_loudness -- igual que
39:_process_candidate. Esto es lo que hace posible el check de regresion:
_match_loudness cambia el audio (linear/linear_capped preservan la forma de
onda y por tanto el crest, pero soft_limit (tanh) NO -- comprime picos, baja
el crest_db). Medir post-match_loudness rompería el check de regresion para
cualquier clip que caiga en la rama soft_limit.

Compresion a Opus: libsndfile/soundfile solo aceptan Opus a 8k/12k/16k/24k/
48k Hz (probado en vivo: 44100 lanza LibsndfileError). El motor renderiza a
44100 (fusion_chain.SR), asi que el pipeline es
render(44100) -> match_loudness -> resample_poly a 48000 -> sf.write OGG/
OPUS. Verificado en vivo (ver informe de la tarea): sobre audio REAL de
este motor (no ruido blanco, que es peor caso y da ~-0.49dB, al filo de la
tolerancia) el redondeo completo resample+opus+decode da ~0.004dB de
diferencia de RMS -- muy por debajo de los 0.5dB del check 3. Por eso este
script NO escribe un WAV maestro intermedio (el brief lo permite: "o
escribe Opus directamente si el round-trip es fiel").

Uso:
    PYTHONPATH=. .venv/bin/python scripts/42_fusion_set.py --check
    PYTHONPATH=. .venv/bin/python scripts/42_fusion_set.py --pair trueno_hecho_de_agua
    PYTHONPATH=. .venv/bin/python scripts/42_fusion_set.py              # set completo (~25 min serie)
    PYTHONPATH=. .venv/bin/python scripts/42_fusion_set.py --jobs 6     # ~5 min, paralelo por pareja
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from impossible_mix.physics.analysis import crest_factor_db  # noqa: E402
from impossible_mix.physics.blend_recipes import chimera_bands_heuristic  # noqa: E402
from impossible_mix.physics.fusion_chain import FusionSpec, render_fusion  # noqa: E402


def _load_script39():
    """Carga scripts/39_fusion_search.py por ruta (no es un identificador
    Python valido para un `import` normal -- empieza por digito). Mismo
    patron que scripts/41_fusion_evidence.py::_load_script39. Ejecutar el
    modulo aplica su monkeypatch de fusion_chain.register_hz (memo,
    verificado bit-identico en el --check de 39) como efecto secundario
    deliberado."""
    path = Path(__file__).resolve().parent / "39_fusion_search.py"
    spec = importlib.util.spec_from_file_location("_script39_fusion_search", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S39 = _load_script39()
PAREJAS_V12 = S39.PAREJAS_V12
_metric_parents = S39._metric_parents
_composite_fusion_fast = S39._composite_fusion_fast
_match_loudness = S39._match_loudness

SR = 44_100
SR_OPUS = 48_000
SEED = 42
DUR = 6.0
N_BANDS_GRID = (2, 4, 6, 8, 12, 16, 24, 32)
COLOR_MIX_GRID = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
# Subconjunto que SI esta en resumen.csv (barrido previo, 5 valores) --
# es el que habilita el check de regresion gratis (ver brief).
REGRESSION_COLOR_MIX = (0.0, 0.25, 0.5, 0.75, 1.0)

OUT_DIR = Path("demo_fusion")
RESUMEN_CSV = Path("results/fusion_search/resumen.csv")

MANIFEST_VERSION = 1

# Contexto de multiprocessing fijado explicitamente a "fork" (no confiar en
# el default de la plataforma): el modulo scripts/39_fusion_search.py se
# carga por ruta bajo el nombre sintetico "_script39_fusion_search" (ver
# _load_script39), que ningun proceso hijo podria RE-importar por su cuenta
# si arrancase en frio (metodo "spawn"/"forkserver") -- solo "fork" hereda
# el modulo ya cargado por copia de memoria del proceso padre.
_MP_CTX = multiprocessing.get_context("fork")


# ====================================================================
# Rejilla de una pareja: 72 celdas de la rejilla principal + 4 referencias
# ====================================================================

def _clip_specs(env: str, fine: str) -> list[tuple[str, str, FusionSpec]]:
    """(grupo, etiqueta, spec) de los 76 clips de una pareja env/fine.

    grupo: "grid" (72) o "referencia" (4). etiqueta: identificador estable
    usado como nombre de fichero (sin extension) -- "nbN__mixM.MMM" para la
    rejilla, el nombre de la referencia ("baseline_v11", etc.) para las 4
    ancla."""
    out: list[tuple[str, str, FusionSpec]] = []
    for nb in N_BANDS_GRID:
        for mix in COLOR_MIX_GRID:
            spec = FusionSpec(env_parent=env, fine_parent=fine, n_bands=nb,
                              align=True, warp=False, color_mix=mix,
                              method="chimera", duration_s=DUR, seed=SEED)
            out.append(("grid", f"nb{nb}__mix{mix:.3f}", spec))

    nb_heur = chimera_bands_heuristic(env)
    refs = [
        ("baseline_v11", FusionSpec(
            env_parent=env, fine_parent=fine, n_bands=nb_heur, align=False,
            warp=False, color_mix=None, method="chimera", duration_s=DUR, seed=SEED)),
        ("suma_ancla", FusionSpec(
            env_parent=env, fine_parent=fine, n_bands=nb_heur, align=True,
            warp=False, method="suma", duration_s=DUR, seed=SEED)),
        ("sin_alinear", FusionSpec(
            env_parent=env, fine_parent=fine, n_bands=nb_heur, align=False,
            warp=False, color_mix=0.5, method="chimera", duration_s=DUR, seed=SEED)),
        ("plana", FusionSpec(
            env_parent=env, fine_parent=fine, n_bands=nb_heur, align=True,
            warp=False, method="chimera_plana", duration_s=DUR, seed=SEED)),
    ]
    for nombre, spec in refs:
        out.append(("referencia", nombre, spec))
    return out


def _filename(grupo: str, etiqueta: str) -> str:
    return f"ref__{etiqueta}.opus" if grupo == "referencia" else f"{etiqueta}.opus"


# ====================================================================
# Compresion a Opus (resample 44100 -> 48000, ver docstring del modulo)
# ====================================================================
_RESAMPLE_UP = SR_OPUS // gcd(SR, SR_OPUS)
_RESAMPLE_DOWN = SR // gcd(SR, SR_OPUS)


def _to_opus_bytes_rms(w: np.ndarray, sr_in: int = SR, sr_out: int = SR_OPUS) -> np.ndarray:
    """Remuestrea `w` de sr_in a sr_out (48000, unico rate cercano a 44100
    que Opus soporta) y recorta a [-1,1] -- mismo recorte que save_wav."""
    y = resample_poly(w.astype(np.float64), _RESAMPLE_UP, _RESAMPLE_DOWN).astype(np.float32)
    return np.clip(y, -1.0, 1.0)


def _write_opus(path: Path, w: np.ndarray, sr_in: int = SR, sr_out: int = SR_OPUS) -> None:
    y = _to_opus_bytes_rms(w, sr_in, sr_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr_out, format="OGG", subtype="OPUS")


# ====================================================================
# Procesado de un clip: metricas sobre el render crudo, luego igualacion
# de sonoridad para la escucha
# ====================================================================

def _process_clip(pareja: str, grupo: str, etiqueta: str, spec: FusionSpec,
                  cache: dict, sso_cache: dict, sr: int = SR) -> tuple[dict, np.ndarray]:
    w, meta = render_fusion(spec, sr=sr, cache=cache)

    # Metricas SOBRE EL RENDER CRUDO (antes de match_loudness) -- asi las
    # midio resumen.csv, ver docstring del modulo.
    crest_db = crest_factor_db(w)
    a_used, b_aligned, b_original = _metric_parents(spec, sr, cache, meta)
    rep = _composite_fusion_fast(w, a_used, b_aligned, b_original, sr, sso_cache, spec, meta)

    w_listen, info = _match_loudness(w)

    inertes = set(meta.get("parametros_inertes", []))
    n_bands_out = None if "n_bands" in inertes else spec.n_bands
    color_mix_out = None if "color_mix" in inertes else spec.color_mix

    row = {
        "id": etiqueta if grupo == "grid" else f"ref__{etiqueta}",
        "pareja": pareja, "grupo": grupo,
        "nombre_referencia": etiqueta if grupo == "referencia" else None,
        "env_parent": spec.env_parent, "fine_parent": spec.fine_parent,
        "n_bands": n_bands_out, "color_mix": color_mix_out,
        "align": spec.align, "method": spec.method,
        "crest_db": round(float(crest_db), 6),
        "sso": round(float(rep.sso), 6),
        "fci": round(float(rep.fci), 6),
        "factor_rms": round(float(info["factor_rms_aplicado"]), 6),
        "metodo_igualacion": info["metodo_igualacion"],
        "pico_recortado": bool(info["pico_recortado"]),
        "fichero": f"audio/{pareja}/{_filename(grupo, etiqueta)}",
    }
    return row, w_listen


# ====================================================================
# Check de regresion contra resumen.csv (ver brief, check 1)
# ====================================================================

def _load_resumen_index(path: Path = RESUMEN_CSV) -> dict[tuple, float]:
    """(pareja, n_bands, color_mix redondeado a 6 decimales) -> crest_db,
    SOLO para las filas que coinciden con la config de la rejilla principal
    de esta tarea (chimera_grid, align=True, warp=False, method=chimera,
    duration_s=6.0, seed=42) -- las UNICAS bit-comparables."""
    idx: dict[tuple, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not (row["grupo"] == "chimera_grid" and row["align"] == "True"
                    and row["warp"] == "False" and row["method"] == "chimera"
                    and row["duration_s"] == "6.0" and row["seed"] == "42"
                    and row["crest_db"] != ""):
                continue
            key = (row["pareja"], int(row["n_bands"]), round(float(row["color_mix"]), 6))
            idx[key] = float(row["crest_db"])
    return idx


def _check_regression_cell(pareja: str, spec: FusionSpec, crest_db: float,
                           resumen_idx: dict, tol: float = 1e-6) -> str | None:
    """None si OK / no aplica (color_mix fuera del subconjunto barrido);
    string de fallo si diverge o falta en resumen.csv."""
    if spec.color_mix not in REGRESSION_COLOR_MIX:
        return None
    key = (pareja, spec.n_bands, round(spec.color_mix, 6))
    ref = resumen_idx.get(key)
    if ref is None:
        return f"{key}: no encontrado en resumen.csv"
    diff = abs(crest_db - ref)
    if diff > tol:
        return f"{key}: crest_db={crest_db:.6f} vs resumen.csv={ref:.6f} (diff={diff:.2e})"
    return None


# ====================================================================
# run_pareja: los 76 clips de una pareja (grid + referencias)
# ====================================================================

def run_pareja(pareja: str, env: str, fine: str, out_dir: Path,
               check_regression: bool = True, verbose: bool = True,
               specs: list[tuple[str, str, FusionSpec]] | None = None) -> dict:
    """`specs` por defecto es None -> _clip_specs(env, fine) (los 76 clips
    reales de la pareja). Parametro de override SOLO para tests
    (_check_jobs_deterministic pasa un subconjunto de 4 clips para comparar
    serie vs. paralelo sin pagar el coste de las 76 celdas x2)."""
    t0 = time.time()
    cache: dict = {}
    sso_cache: dict = {}
    resumen_idx = _load_resumen_index() if check_regression else {}
    if specs is None:
        specs = _clip_specs(env, fine)

    audio_dir = out_dir / "audio" / pareja
    rows: list[dict] = []
    regression_failures: list[str] = []
    n_regression_checked = 0

    for grupo, etiqueta, spec in specs:
        row, w_listen = _process_clip(pareja, grupo, etiqueta, spec, cache, sso_cache)
        _write_opus(audio_dir / _filename(grupo, etiqueta), w_listen)
        rows.append(row)

        if check_regression and grupo == "grid":
            # row["crest_db"] ya viene redondeado a 6 decimales (igual que
            # resumen.csv, escrito por _fmt_cell de 39 con el mismo
            # redondeo) -- comparar ambos valores ya redondeados con
            # tolerancia 1e-6 es equivalente a comparar los originales sin
            # redondear.
            fail = _check_regression_cell(pareja, spec, row["crest_db"], resumen_idx)
            if spec.color_mix in REGRESSION_COLOR_MIX:
                n_regression_checked += 1
            if fail:
                regression_failures.append(fail)

    t_total = time.time() - t0
    summary = {
        "pareja": pareja, "env_parent": env, "fine_parent": fine,
        "n_bands_heuristica": chimera_bands_heuristic(env),
        "rows": rows, "t_total_s": t_total,
        "n_regression_checked": n_regression_checked,
        "regression_failures": regression_failures,
    }
    if verbose:
        estado = "OK" if not regression_failures else f"FALLOS: {len(regression_failures)}"
        print(f"  {pareja}: {len(rows)} clips en {t_total:.1f}s | regresion "
             f"{n_regression_checked} celdas: {estado}")
        for f in regression_failures:
            print(f"    FALLO regresion: {f}")
    return summary


# ====================================================================
# Manifest.json
# ====================================================================

def _build_manifest(summaries: list[dict]) -> dict:
    clips = []
    parejas_meta = []
    for s in summaries:
        clips.extend(s["rows"])
        parejas_meta.append({
            "pareja": s["pareja"], "env_parent": s["env_parent"],
            "fine_parent": s["fine_parent"],
            "n_bands_heuristica": s["n_bands_heuristica"],
        })
    return {
        "meta": {
            "version": MANIFEST_VERSION,
            "sr_audio": SR_OPUS,
            "sr_render": SR,
            "duration_s": DUR,
            "seed": SEED,
            "n_bands_grid": list(N_BANDS_GRID),
            "color_mix_grid": list(COLOR_MIX_GRID),
            "align_grid": True,
            "warp_grid": False,
            "method_grid": "chimera",
            "target_rms_lineal": float(S39._LISTEN_TARGET_RMS),
            "n_parejas": len(summaries),
            "n_clips_total": len(clips),
        },
        "parejas": parejas_meta,
        "clips": clips,
    }


def _write_manifest(manifest: dict, out_dir: Path) -> Path:
    path = out_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


# ====================================================================
# Biyeccion manifest <-> disco (una sola funcion, usada en --check contra
# un tempdir sintetico y al final del set completo contra demo_fusion/)
# ====================================================================

def check_bijection(manifest: dict, out_dir: Path) -> list[str]:
    """Todo clip del manifiesto existe en disco Y todo .opus en disco esta
    en el manifiesto."""
    failures: list[str] = []
    manifest_files = {(out_dir / c["fichero"]).resolve() for c in manifest["clips"]}
    disk_files = {p.resolve() for p in (out_dir / "audio").rglob("*.opus")} \
        if (out_dir / "audio").exists() else set()
    missing_on_disk = manifest_files - disk_files
    missing_in_manifest = disk_files - manifest_files
    if missing_on_disk:
        failures.append(f"en manifiesto pero no en disco ({len(missing_on_disk)}): "
                        f"{sorted(str(p) for p in missing_on_disk)[:5]}")
    if missing_in_manifest:
        failures.append(f"en disco pero no en manifiesto ({len(missing_in_manifest)}): "
                        f"{sorted(str(p) for p in missing_in_manifest)[:5]}")
    return failures


# ====================================================================
# --check: smokes rapidos
# ====================================================================

def _check_regression_sample() -> list[str]:
    """Regresion (check 1 del brief) sobre una MUESTRA rapida: 2 celdas por
    pareja (extremos de n_bands, ambas dentro de REGRESSION_COLOR_MIX),
    duracion 6.0s real (DUR_GRID) -- NO se puede acortar la duracion aqui,
    porque el punto es reproducir bit-a-bit valores medidos a 6s en
    resumen.csv; acortarla invalidaria la comparacion. La verificacion
    COMPLETA de las 240 celdas compartidas se hace en el set completo
    (gratis: ya se computa crest_db para las 456), ver run_pareja."""
    resumen_idx = _load_resumen_index()
    failures = []
    muestras = [(2, 0.0), (32, 1.0)]
    for pareja, env, fine in PAREJAS_V12:
        cache: dict = {}
        sso_cache: dict = {}
        for nb, mix in muestras:
            spec = FusionSpec(env_parent=env, fine_parent=fine, n_bands=nb, align=True,
                              warp=False, color_mix=mix, method="chimera",
                              duration_s=DUR, seed=SEED)
            w, meta = render_fusion(spec, sr=SR, cache=cache)
            crest_db = crest_factor_db(w)
            fail = _check_regression_cell(pareja, spec, crest_db, resumen_idx)
            if fail:
                failures.append(fail)
    print(f"smoke1 (regresion contra resumen.csv, muestra de "
         f"{len(muestras) * len(PAREJAS_V12)} celdas @ 6s reales): "
         f"{'OK' if not failures else 'FALLOS: ' + str(failures)}")
    return failures


def _check_determinism() -> list[str]:
    spec = FusionSpec(env_parent="trueno", fine_parent="goteo", n_bands=8, align=True,
                      warp=False, color_mix=0.375, method="chimera", duration_s=1.5, seed=SEED)
    w1, _m1 = render_fusion(spec, sr=SR, cache={})
    w2, _m2 = render_fusion(spec, sr=SR, cache={})
    ok = np.array_equal(w1, w2)
    if ok:
        wl1, _ = _match_loudness(w1)
        wl2, _ = _match_loudness(w2)
        ok = np.array_equal(wl1, wl2)
    print(f"smoke2 (determinismo: re-render + match_loudness bit-identico): "
         f"{'OK' if ok else 'FALLO'}")
    return [] if ok else ["determinismo"]


def _check_opus_roundtrip() -> list[str]:
    """RMS del Opus decodificado dentro de 0.5dB del audio original -- sobre
    un render REAL de este motor (no ruido blanco: medido en vivo, el ruido
    blanco da ~-0.49dB, al filo de la tolerancia, por ser el peor caso
    posible para un codec perceptual; audio real de este motor da ~0.004dB,
    ver docstring del modulo)."""
    spec = FusionSpec(env_parent="fuego", fine_parent="vidrio", n_bands=12, align=True,
                      warp=False, color_mix=0.625, method="chimera", duration_s=3.0, seed=SEED)
    w, _meta = render_fusion(spec, sr=SR, cache={})
    w_listen, _info = _match_loudness(w)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roundtrip.opus"
        _write_opus(path, w_listen)
        decoded, sr_decoded = sf.read(str(path), dtype="float32")
    rms_orig = float(np.sqrt(np.mean(w_listen.astype(np.float64) ** 2)))
    rms_dec = float(np.sqrt(np.mean(decoded.astype(np.float64) ** 2)))
    diff_db = 20.0 * np.log10((rms_dec + 1e-12) / (rms_orig + 1e-12))
    ok = sr_decoded == SR_OPUS and abs(diff_db) <= 0.5
    print(f"smoke3 (round-trip Opus, sr={sr_decoded}, diff RMS={diff_db:.4f}dB, "
         f"tolerancia 0.5dB): {'OK' if ok else 'FALLO'}")
    return [] if ok else ["opus_roundtrip"]


def _check_bijection_logic() -> list[str]:
    """Ejercita check_bijection contra un tempdir sintetico: 2 clips reales
    en disco + manifiesto que declara 3 (uno de mas, "falta en disco") + 1
    fichero extra en disco no declarado ("sobra en disco") -- ambos casos
    de fallo deben detectarse, y el caso feliz (manifiesto == disco) debe
    pasar limpio."""
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        (out_dir / "audio" / "p1").mkdir(parents=True)
        for name in ("a.opus", "b.opus"):
            (out_dir / "audio" / "p1" / name).write_bytes(b"\x00")

        manifest_ok = {"clips": [
            {"fichero": "audio/p1/a.opus"}, {"fichero": "audio/p1/b.opus"},
        ]}
        f_ok = check_bijection(manifest_ok, out_dir)
        if f_ok:
            failures.append(f"caso feliz deberia pasar limpio, dio: {f_ok}")

        manifest_falta = {"clips": [
            {"fichero": "audio/p1/a.opus"}, {"fichero": "audio/p1/b.opus"},
            {"fichero": "audio/p1/c.opus"},
        ]}
        f_falta = check_bijection(manifest_falta, out_dir)
        if not f_falta:
            failures.append("no detecto clip del manifiesto ausente en disco")

        (out_dir / "audio" / "p1" / "sobra.opus").write_bytes(b"\x00")
        f_sobra = check_bijection(manifest_ok, out_dir)
        if not f_sobra:
            failures.append("no detecto fichero en disco ausente del manifiesto")

    print(f"smoke4 (logica de biyeccion: feliz OK, detecta falta-en-disco, "
         f"detecta sobra-en-disco): {'OK' if not failures else 'FALLOS: ' + str(failures)}")
    return failures


def _mini_specs(env: str, fine: str) -> list[tuple[str, str, FusionSpec]]:
    """Subconjunto de 4 clips (2 de rejilla + 2 referencias, duracion corta)
    de los 76 reales de una pareja -- SOLO para _check_jobs_deterministic,
    para ejercitar ambos caminos (grid/referencia) sin pagar el coste de
    las 76 celdas x2 (serie+paralelo)."""
    nb_heur = chimera_bands_heuristic(env)
    return [
        ("grid", "nb4__mix0.250", FusionSpec(
            env_parent=env, fine_parent=fine, n_bands=4, align=True, warp=False,
            color_mix=0.25, method="chimera", duration_s=1.5, seed=SEED)),
        ("grid", "nb16__mix0.750", FusionSpec(
            env_parent=env, fine_parent=fine, n_bands=16, align=True, warp=False,
            color_mix=0.75, method="chimera", duration_s=1.5, seed=SEED)),
        ("referencia", "suma_ancla", FusionSpec(
            env_parent=env, fine_parent=fine, n_bands=nb_heur, align=True, warp=False,
            method="suma", duration_s=1.5, seed=SEED)),
        ("referencia", "sin_alinear", FusionSpec(
            env_parent=env, fine_parent=fine, n_bands=nb_heur, align=False, warp=False,
            color_mix=0.5, method="chimera", duration_s=1.5, seed=SEED)),
    ]


def _check_jobs_deterministic() -> list[str]:
    """--jobs es opcional y debe dar el MISMO resultado que la ejecucion en
    serie (brief: "si anades paralelismo, que sea determinista"). Compara,
    para 2 parejas y un subconjunto corto de 4 clips (ver _mini_specs), las
    filas producidas por run_pareja() llamado directamente (equivalente a
    --jobs 1) frente al mismo run_pareja ejecutado dentro de un
    ProcessPoolExecutor (equivalente a --jobs>1) -- cada pareja es
    independiente (cache propio), asi que el resultado no deberia depender
    de en que proceso se calculo."""
    failures = []
    pares = PAREJAS_V12[:2]
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        serie = [run_pareja(p, e, f, out_dir, check_regression=False, verbose=False,
                            specs=_mini_specs(e, f))
                for p, e, f in pares]
    with tempfile.TemporaryDirectory() as tmp2:
        out_dir2 = Path(tmp2)
        with ProcessPoolExecutor(max_workers=2, mp_context=_MP_CTX) as ex:
            paralelo = list(ex.map(
                _run_pareja_worker,
                [(p, e, f, out_dir2, _mini_specs(e, f)) for p, e, f in pares]))
    for s_serie, s_par in zip(serie, paralelo):
        crest_serie = [r["crest_db"] for r in s_serie["rows"]]
        crest_par = [r["crest_db"] for r in s_par["rows"]]
        if crest_serie != crest_par:
            failures.append(f"{s_serie['pareja']}: crest_db difiere entre serie y paralelo")
    print(f"smoke5 (--jobs determinista: serie == paralelo en "
         f"{len(pares)} parejas x {len(_mini_specs('trueno','goteo'))} clips): "
         f"{'OK' if not failures else 'FALLOS: ' + str(failures)}")
    return failures


def check() -> int:
    all_failures = []
    all_failures += _check_regression_sample()
    all_failures += _check_determinism()
    all_failures += _check_opus_roundtrip()
    all_failures += _check_bijection_logic()
    all_failures += _check_jobs_deterministic()
    if all_failures:
        print(f"\n{len(all_failures)} fallos: {all_failures}")
        return 1
    print("\ntodos los smokes OK")
    return 0


# ====================================================================
# Paralelismo opcional por pareja (ver docstring de _check_jobs_deterministic)
# ====================================================================

def _run_pareja_worker(args: tuple) -> dict:
    """`args` es (pareja, env, fine, out_dir) para el uso real (--jobs>1 en
    main()) o (pareja, env, fine, out_dir, specs) para el subconjunto de
    prueba de _check_jobs_deterministic."""
    pareja, env, fine, out_dir = args[0], args[1], args[2], args[3]
    specs = args[4] if len(args) > 4 else None
    return run_pareja(pareja, env, fine, out_dir, check_regression=False, verbose=False,
                      specs=specs)


# ====================================================================
# main
# ====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="smokes rapidos")
    ap.add_argument("--pair", type=str, default=None,
                    help="genera solo esta pareja")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parejas en paralelo (proceso por pareja, opcional; "
                         "1 = serie, por defecto)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.check:
        return check()

    parejas = PAREJAS_V12
    if args.pair:
        match = [(n, e, f) for n, e, f in PAREJAS_V12 if n == args.pair]
        if not match:
            print(f"pareja desconocida: {args.pair!r}. Validas: "
                 f"{[n for n, _, _ in PAREJAS_V12]}", file=sys.stderr)
            return 1
        parejas = match

    t_start = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=_MP_CTX) as ex:
            summaries_unordered = list(ex.map(
                _run_pareja_worker,
                [(n, e, f, args.out_dir) for n, e, f in parejas]))
        # reordena a PAREJAS_V12 (el orden de finalizacion del pool no lo
        # respeta) + aplica el check de regresion en el proceso principal
        # (barato: solo lookups de diccionario, no vuelve a renderizar).
        by_pareja = {s["pareja"]: s for s in summaries_unordered}
        resumen_idx = _load_resumen_index()
        summaries = []
        for n, e, f in parejas:
            s = by_pareja[n]
            fails = []
            n_checked = 0
            for grupo, etiqueta, spec in _clip_specs(e, f):
                if grupo != "grid" or spec.color_mix not in REGRESSION_COLOR_MIX:
                    continue
                row = next(r for r in s["rows"] if r["id"] == etiqueta)
                n_checked += 1
                fail = _check_regression_cell(n, spec, row["crest_db"], resumen_idx)
                if fail:
                    fails.append(fail)
            s["n_regression_checked"] = n_checked
            s["regression_failures"] = fails
            estado = "OK" if not fails else f"FALLOS: {len(fails)}"
            print(f"  {n}: {len(s['rows'])} clips en {s['t_total_s']:.1f}s | "
                 f"regresion {n_checked} celdas: {estado}")
            summaries.append(s)
    else:
        summaries = [run_pareja(n, e, f, args.out_dir) for n, e, f in parejas]

    manifest = _build_manifest(summaries)
    manifest_path = _write_manifest(manifest, args.out_dir)

    n_checked_total = sum(s["n_regression_checked"] for s in summaries)
    fails_total = [f for s in summaries for f in s["regression_failures"]]
    bijection_fails = check_bijection(manifest, args.out_dir)

    # Tripwire barato: len(N_BANDS_GRID) x len(REGRESSION_COLOR_MIX) celdas
    # compartidas por pareja procesada -- si el numero real es MENOR, algo
    # dejo de llegar a la comparacion en silencio (no lo detectaria
    # fails_total, que solo ve divergencias, no ausencias de check).
    n_esperado = len(N_BANDS_GRID) * len(REGRESSION_COLOR_MIX) * len(parejas)
    if n_checked_total != n_esperado:
        fails_total = fails_total + [
            f"n_regression_checked={n_checked_total} != esperado={n_esperado}"]

    t_total = time.time() - t_start
    print(f"\nmanifest: {manifest_path} ({len(manifest['clips'])} clips)")
    print(f"regresion completa: {n_checked_total}/{n_esperado} celdas comparadas contra "
         f"resumen.csv, {len(fails_total)} fallos")
    print(f"biyeccion manifiesto<->disco: {'OK' if not bijection_fails else bijection_fails}")
    print(f"TOTAL: {t_total:.1f}s ({t_total / 60.0:.1f} min) para "
         f"{len(manifest['clips'])} clips")
    return 1 if (fails_total or bijection_fails) else 0


if __name__ == "__main__":
    raise SystemExit(main())
