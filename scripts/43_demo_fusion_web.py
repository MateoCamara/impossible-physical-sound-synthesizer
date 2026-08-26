"""Tarea 9 (F4): web de demostracion con sliders sobre el set de la tarea 8.

Genera `demo_fusion/index.html` + `demo_fusion/fusionpad.js` +
`demo_fusion/LEEME.md` a partir de `demo_fusion/manifest.json` (456 clips
Opus, commit d0731d4, NO se regenera aqui -- ver `scripts/42_fusion_set.py`).
Este script no renderiza audio ni toca `demo_fusion/audio/`: solo construye
la pagina que lo consume.

Patron: `scripts/38_demo_congreso.py` (pagina estatica generada desde
Python). Diferencia clave: aqui la "rejilla" no se pre-renderiza en el
navegador (como hace `web/pad2d.js` con `renderFn`) sino que se decodifica
bajo demanda desde ficheros Opus ya existentes -- ver docstring de
`fusionpad.js` en `_FUSIONPAD_JS_TEMPLATE` para el porque de la restriccion
de memoria (72 clips x 6s x 44.1kHz x 4B ~= 76 MB por pareja).

Los dos ejes de la rejilla (ver brief de la tarea, seccion "Los dos ejes"):
  - `color_mix` (9 valores): eje CONTINUO, con crossfade real (2 fuentes,
    ganancias interpoladas linealmente). Verificado en `impossible_mix/
    physics/blend.py` (~938-980) que la contribucion por banda `parts[k]`
    NO depende de `color_mix` -- solo cambian los pesos `band_w` -- asi que
    mezclar linealmente dos salidas vecinas da otro miembro EXACTO de la
    misma familia (sin interferencia de fase), siempre que la igualacion de
    sonoridad de ambos clips haya sido una escala lineal (metodo "linear" o
    "linear_capped": `x * factor`, preserva forma de onda). Excepcion real
    encontrada en el manifiesto real (no hipotetica): 5 celdas de la
    rejilla de `trueno_hecho_de_canica` (nb in {12,16,24,32} en mix=1.0, mas
    nb=12 en mix=0.875) usan `metodo_igualacion="soft_limit"` (tanh, NO
    lineal) porque su cresta era demasiado patologica para el recorte
    lineal -- ver `_soft_limit_grid_cells()` mas abajo, que las detecta
    dinamicamente contra el manifiesto real en vez de darlas por sentado.
    Para esas celdas la propiedad de exactitud NO se cumple: la pagina lo
    avisa en pantalla (badge junto al slider) cuando el punto activo o su
    vecino de crossfade cae en una de ellas.
  - `n_bands` (8 valores): eje DISCRETO, con conmutacion (nunca
    interpolacion: cambia el banco de filtros entero). Al conmutar se
    preserva la posicion de reproduccion y se aplica un fundido corto
    (~20ms) solo como antichasquidos.

Uso:
    PYTHONPATH=. .venv/bin/python scripts/43_demo_fusion_web.py --check
    PYTHONPATH=. .venv/bin/python scripts/43_demo_fusion_web.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path("demo_fusion")
MANIFEST_NAME = "manifest.json"

REFERENCIA_NOMBRES = ("baseline_v11", "suma_ancla", "sin_alinear", "plana")

# Nombres bonitos en espaniol (CON tildes -- son texto de interfaz, no
# identificadores) para los padres fisicos que aparecen en las 6 parejas de
# `manifest.json["parejas"]`. Si algun dia se ariade una pareja con un padre
# nuevo, cae al nombre crudo (sin tilde) en vez de romper.
PRETTY_PARENT = {
    "trueno": "el trueno",
    "goteo": "el goteo",
    "fuego": "el fuego",
    "vidrio": "el vidrio",
    "canica": "la canica",
    "oceano": "el océano",
    "campana_tela": "la campana de tela",
}


# ======================================================================
# Manifiesto -> indice por pareja (unica fuente de verdad que se hornea en
# fusionpad.js; index.html NO duplica esta informacion, la lee en runtime
# desde FUSION_MANIFEST)
# ======================================================================

def load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        raise SystemExit(
            f"ERROR: no existe {path}. Este script NO genera el audio ni el "
            f"manifiesto (eso es scripts/42_fusion_set.py, tarea 8, commit "
            f"d0731d4) -- solo consume lo que ya este ahi.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _frase_explicacion(env_parent: str, fine_parent: str) -> str:
    env_label = PRETTY_PARENT.get(env_parent, env_parent)
    fine_label = PRETTY_PARENT.get(fine_parent, fine_parent)
    return (f"La dinámica (ritmo, ataques) la pone {env_label}. "
            f"La materia (timbre, textura fina) la pone {fine_label}.")


def build_pareja_index(manifest: dict) -> dict:
    """{pareja_id: {titulo, env_parent, fine_parent, frase,
    n_bands_heuristica, grid: {nb_str: {mix_str: rec}}, referencias:
    {nombre: rec}}} -- rec siempre trae fichero/crest_db/sso/
    metodo_igualacion/pico_recortado; en referencias ademas n_bands."""
    meta_by_pareja = {p["pareja"]: p for p in manifest["parejas"]}
    by_pareja: dict[str, dict] = {}
    for pareja_id, pmeta in meta_by_pareja.items():
        env, fine = pmeta["env_parent"], pmeta["fine_parent"]
        by_pareja[pareja_id] = {
            "id": pareja_id,
            "titulo": pareja_id.replace("_", " ").capitalize(),
            "env_parent": env,
            "fine_parent": fine,
            "frase": _frase_explicacion(env, fine),
            "n_bands_heuristica": pmeta["n_bands_heuristica"],
            "grid": {},
            "referencias": {},
        }
    for c in manifest["clips"]:
        entry = by_pareja[c["pareja"]]
        rec = {
            "fichero": c["fichero"],
            "crest_db": c["crest_db"],
            "sso": c["sso"],
            "metodo_igualacion": c["metodo_igualacion"],
            "pico_recortado": c["pico_recortado"],
        }
        if c["grupo"] == "grid":
            nb_key = str(c["n_bands"])
            mix_key = f'{c["color_mix"]:.3f}'
            entry["grid"].setdefault(nb_key, {})[mix_key] = rec
        else:
            rec["n_bands"] = c["n_bands"]
            entry["referencias"][c["nombre_referencia"]] = rec
    return by_pareja


def _soft_limit_grid_cells(pareja_index: dict) -> list[tuple[str, str, str]]:
    """(pareja_id, n_bands_str, mix_str) de toda celda de REJILLA (no
    referencia) cuya igualacion de sonoridad fue "soft_limit" -- las unicas
    donde el crossfade deja de ser una combinacion lineal exacta (ver
    docstring del modulo). Recorrido determinista (orden de insercion de
    dict, que en Python 3.7+ es orden de insercion == orden de
    manifest["clips"])."""
    out = []
    for pareja_id, entry in pareja_index.items():
        for nb_key, by_mix in entry["grid"].items():
            for mix_key, rec in by_mix.items():
                if rec["metodo_igualacion"] == "soft_limit":
                    out.append((pareja_id, nb_key, mix_key))
    return out


# ======================================================================
# fusionpad.js -- estructura segun web/pad2d.js (mismo ciclo de vida
# BufferSourceNode/GainNode, misma forma prerender->crossfade, mismo patron
# de progreso), pero 1-D (2 fuentes, no 4; interpolacion lineal, no
# bilineal), con fetch+decodeAudioData en vez de sintesis, y el eje
# discreto (n_bands) fuera de la interpolacion. NO se modifica
# web/pad2d.js: es de otra demo y sigue en uso.
# ======================================================================

_FUSIONPAD_JS_TEMPLATE = r'''// fusionpad.js -- motor de audio + interfaz de la demo de fusion con
// sliders (Tarea 9 / F4). GENERADO por scripts/43_demo_fusion_web.py a
// partir de demo_fusion/manifest.json -- NO EDITAR A MANO (los cambios se
// pierden en la proxima ejecucion del generador). Si algo aqui esta mal,
// arreglar la plantilla en el script, no este fichero.
//
// Estructura (ver docstring de web/pad2d.js, del que esta tomado el ciclo
// de vida BufferSourceNode/GainNode + patron prerender->crossfade->progreso
// -- pero 1-D: 2 fuentes con interpolacion lineal en vez de 4 con
// bilineal, y decodeAudioData bajo demanda en vez de una funcion de
// sintesis):
//   1. FUSION_MANIFEST: datos horneados desde manifest.json (fichero,
//      crest_db, sso, metodo_igualacion por celda). Unica fuente de verdad
//      que consume esta pagina -- index.html no duplica nada de esto.
//   2. Funciones puras (indice, vecindario de crossfade, deteccion de
//      protocolo) -- exportadas al final via module.exports SI se ejecuta
//      bajo Node (typeof module !== "undefined"), para poder probarlas con
//      "node" sin navegador. En un <script> de navegador normal ese bloque
//      no hace nada (module no existe ahi).
//   3. FusionPad1D: motor Web Audio (fetch + decodeAudioData, 2 fuentes
//      para el crossfade de color_mix, conmutacion con posicion preservada
//      para n_bands). Requiere que fetch() funcione -- ver aviso de
//      "servir sin wifi" mas abajo.
//   4. Modo alternativo file:// (fetch falla ahi en Chrome y Safari):
//      conmutacion simple de <audio src>, SIN crossfade, con aviso visible
//      en pantalla.
//   5. Cableado de la interfaz (DOMContentLoaded), guardado tras un
//      "typeof document !== 'undefined'" para que cargar este fichero con
//      Node (para probar el punto 2) no intente tocar el DOM.
//
// RESTRICCION DE MEMORIA (real, medida): 72 clips x 6s x 44100Hz x 4 bytes
// (Float32) ~= 76 MB decodificados por pareja; las 6 parejas a la vez
// serian ~457 MB. Por eso este fichero decodifica SOLO los 2 clips vecinos
// que hacen falta para la posicion actual del slider, cachea por pareja
// (Map fichero->AudioBuffer), y VACIA ese cache al cambiar de pareja
// (FusionPad1D.loadPareja llama a buffers.clear()). "Vaciar" aqui quiere
// decir soltar las referencias: JavaScript no tiene liberacion explicita de
// memoria, asi que lo que se garantiza es que nada en este codigo retiene
// ya el AudioBuffer -- el recolector de basura del navegador se encarga
// del resto en su propio momento, no es instantaneo. NO precargar las 6
// parejas de golpe "para que vaya mas fluido": es exactamente el patron que
// esta pensado para evitar.

"use strict";

// ==== INICIO DATOS GENERADOS (no editar a mano) ====
const FUSION_MANIFEST = __FUSION_MANIFEST_JSON__;
// ==== FIN DATOS GENERADOS ====

const N_BANDS_GRID = FUSION_MANIFEST.meta.n_bands_grid;
const COLOR_MIX_GRID = FUSION_MANIFEST.meta.color_mix_grid;
const PAREJA_IDS = Object.keys(FUSION_MANIFEST.parejas);
const REFERENCIA_ORDEN = ["baseline_v11", "suma_ancla", "sin_alinear", "plana"];
const REFERENCIA_ETIQUETAS = {
  baseline_v11: "Baseline V11 (lo que hay hoy en la demo del congreso)",
  suma_ancla: "Suma (dos sonidos superpuestos, a propósito)",
  sin_alinear: "Sin alinear (mismo punto, sin alinear registro)",
  plana: "Plana (chimera plana, sin recoloreado)",
};

// ====================================================================
// Funciones puras (sin DOM, sin Web Audio) -- estas son las que se
// prueban directamente con Node contra el manifiesto real.
// ====================================================================

/** Recorta v al rango [0,1]; NaN se trata como 0 (nunca debe pasar desde
 * un <input type=range>, pero mejor un valor valido que romper el resto). */
function clamp01(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/** Formatea un valor de color_mix EXACTAMENTE como las claves horneadas en
 * FUSION_MANIFEST (3 decimales) -- los 9 valores de la rejilla son
 * multiplos de 1/8, exactos en binario64, asi que toFixed(3) es estable. */
function keyMix(v) {
  return v.toFixed(3);
}

/** Vecindario de crossfade para una posicion continua v en [0,1] sobre la
 * rejilla `colorGrid` (9 valores, paso 0.125). Devuelve {loIdx, hiIdx, lo,
 * hi, t} con loIdx/hiIdx SIEMPRE dentro de [0, colorGrid.length-1] y t en
 * [0,1] -- incluidos los extremos v=0 y v=1 y cualquier v fuera de rango
 * (se recorta primero). t=0 significa "exactamente en lo" (posicion
 * renderizada real); t=1 significa "exactamente en hi". */
function resolveColorNeighbors(colorGrid, v) {
  const vv = clamp01(v);
  const nSeg = colorGrid.length - 1;
  const pos = vv * nSeg;
  let idx = Math.floor(pos);
  if (idx >= nSeg) idx = nSeg - 1;
  if (idx < 0) idx = 0;
  const t = pos - idx;
  return { loIdx: idx, hiIdx: idx + 1, lo: colorGrid[idx], hi: colorGrid[idx + 1], t };
}

/** true si `v` cae (con tolerancia de punto flotante) exactamente sobre
 * una de las 9 posiciones renderizadas -- ahi, y SOLO ahi, tiene sentido
 * mostrar el color_mix real como numero (ver honestidad 1 del LEEME). */
function esPosicionRenderizada(neigh, eps = 1e-6) {
  return neigh.t < eps || neigh.t > 1 - eps;
}

/** Clip de rejilla (pareja, n_bands, color_mix) -> registro o undefined si
 * no existe (nunca deberia pasar para las 432 combinaciones reales, pero
 * la funcion no asume nada: la llamante decide que hacer). */
function lookupGridClip(parejaObj, nBands, mixKeyStr) {
  const porBanda = parejaObj.grid[String(nBands)];
  if (!porBanda) return undefined;
  return porBanda[mixKeyStr];
}

/** Clip de referencia (una de las 4 fijas) o undefined. */
function lookupReferencia(parejaObj, nombre) {
  return parejaObj.referencias[nombre];
}

/** true si el protocolo de la pagina es file:// -- ahi fetch() falla en
 * Chrome y Safari (CORS de origen opaco), asi que hay que degradar. */
function isFileProtocol(protocol) {
  return protocol === "file:";
}

/** "server" (fetch+decodeAudioData, crossfade real) o "file" (conmutacion
 * de <audio src>, sin crossfade) segun el protocolo de la pagina. */
function chooseMode(protocol) {
  return isFileProtocol(protocol) ? "file" : "server";
}

/** Indice de la rejilla de color_mix mas cercano a v -- lo unico que existe
 * en modo "file" (no hay crossfade, solo conmutacion al punto renderizado
 * mas proximo). */
function nearestColorIndex(colorGrid, v) {
  const vv = clamp01(v);
  const nSeg = colorGrid.length - 1;
  let idx = Math.round(vv * nSeg);
  if (idx < 0) idx = 0;
  if (idx > nSeg) idx = nSeg;
  return idx;
}

function fmtDb(x) {
  return (x === null || x === undefined || !Number.isFinite(x)) ? "--" : x.toFixed(1) + " dB";
}

function fmtSso(x) {
  return (x === null || x === undefined || !Number.isFinite(x)) ? "--" : x.toFixed(3);
}

// ====================================================================
// FusionPad1D -- motor Web Audio, modo "server" (fetch + decodeAudioData)
// ====================================================================

class FusionPad1D {
  /**
   * @param {AudioContext} ctx
   * @param {{onProgress?: (activo:boolean)=>void, onReadout?: (info:object)=>void}} [callbacks]
   */
  constructor(ctx, callbacks = {}) {
    this.ctx = ctx;
    this.onProgress = callbacks.onProgress || null;
    this.onReadout = callbacks.onReadout || null;
    this.buffers = new Map(); // fichero -> AudioBuffer, cache POR PAREJA (ver loadPareja)
    this.pareja = null;
    this.nBandsIdx = 0;
    this.colorMix = 0.5;
    this.playing = false;
    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = 0.85;
    this.loSource = null; this.hiSource = null;
    this.loGain = null; this.hiGain = null;
    this._loStartTime = 0; this._hiStartTime = 0;
    this._curNBands = null; this._curLoKey = null; this._curHiKey = null;
    this._lastNeighbor = resolveColorNeighbors(COLOR_MIX_GRID, this.colorMix);
    this._pending = 0;
    // Guardia anti-desorden: setColorMix/setNBandsIndex NO se esperan
    // (`await`) desde los listeners "input" del slider (dispararia una
    // decodificacion por cada evento y bloquearia el arrastre), asi que
    // pueden solaparse dos llamadas a _swapTo en vuelo a la vez. Si la
    // decodificacion del segmento MAS ANTIGUO tarda mas que la del MAS
    // NUEVO (tipico: el nuevo puede venir de cache, el antiguo no), sin
    // esta guardia la respuesta antigua llegaria DESPUES y pisaria el
    // estado con datos obsoletos -- el deslizador mostraria una posicion y
    // sonaria otra, sin ningun error visible. _swapSeq se incrementa en
    // cada llamada; solo la invocacion MAS RECIENTE aplica su resultado.
    this._swapSeq = 0;
  }

  /** Cambia de pareja: para todo, VACIA el cache de AudioBuffers de la
   * pareja anterior (ver nota de memoria arriba del fichero) y resetea los
   * sliders a una posicion razonable (n_bands = heuristica de esta pareja,
   * color_mix = 0.5). */
  loadPareja(parejaObj) {
    this.stop();
    this.buffers.clear();
    this.pareja = parejaObj;
    const heurIdx = N_BANDS_GRID.indexOf(parejaObj.n_bands_heuristica);
    this.nBandsIdx = heurIdx >= 0 ? heurIdx : Math.floor(N_BANDS_GRID.length / 2);
    this.colorMix = 0.5;
    this._lastNeighbor = resolveColorNeighbors(COLOR_MIX_GRID, this.colorMix);
    this._emitReadout();
  }

  _setProgress(active) {
    this._pending += active ? 1 : -1;
    if (this._pending < 0) this._pending = 0;
    if (this.onProgress) this.onProgress(this._pending > 0);
  }

  async ensureBuffer(fichero) {
    if (this.buffers.has(fichero)) return this.buffers.get(fichero);
    this._setProgress(true);
    try {
      const resp = await fetch(fichero);
      if (!resp.ok) {
        throw new Error("fusionpad: no se pudo cargar " + fichero + " (HTTP " + resp.status + ")");
      }
      const arr = await resp.arrayBuffer();
      const buf = await this.ctx.decodeAudioData(arr);
      this.buffers.set(fichero, buf);
      return buf;
    } finally {
      this._setProgress(false);
    }
  }

  /** Decodifica (si hace falta) los 2 clips vecinos en `nBandsVal` y
   * cambia las fuentes activas a ellos, preservando la posicion de
   * reproduccion (offset dentro del loop, leido de la fuente que estaba
   * sonando DESPUES de esperar la decodificacion -- no antes, para no
   * arrastrar la latencia de la decodificacion como si fuera silencio) y
   * aplicando un fundido de `fadeMs` (20ms para conmutacion de n_bands =
   * SOLO antichasquidos, no interpolacion; algo mas para el crossfade real
   * de color_mix, donde el fundido corto es ademas el propio mecanismo). */
  async _swapTo(nBandsVal, neigh, fadeMs) {
    const seq = ++this._swapSeq;
    const loClip = lookupGridClip(this.pareja, nBandsVal, keyMix(neigh.lo));
    const hiClip = lookupGridClip(this.pareja, nBandsVal, keyMix(neigh.hi));
    if (!loClip || !hiClip) {
      throw new Error("fusionpad: celda de rejilla no encontrada (" + this.pareja.id +
                       ", n_bands=" + nBandsVal + ")");
    }
    const [loBuf, hiBuf] = await Promise.all([
      this.ensureBuffer(loClip.fichero), this.ensureBuffer(hiClip.fichero),
    ]);
    // Si otra llamada a _swapTo empezo DESPUES de esta (this._swapSeq ya
    // avanzo) o el pad se paro mientras decodificabamos, esta respuesta
    // esta obsoleta: no toques el estado ni crees fuentes nuevas (ver
    // comentario de _swapSeq en el constructor).
    if (seq !== this._swapSeq || !this.playing) return;

    const now = this.ctx.currentTime;
    let refStart = this._loStartTime;
    if (this.hiGain && this.loGain && this.hiGain.gain.value > this.loGain.gain.value) {
      refStart = this._hiStartTime;
    }
    const dur = loBuf.duration;
    const offset = dur > 0 ? ((now - refStart) % dur + dur) % dur : 0;

    const oldLoSrc = this.loSource, oldHiSrc = this.hiSource;
    const oldLoGain = this.loGain, oldHiGain = this.hiGain;

    const loSrc = this.ctx.createBufferSource(); loSrc.buffer = loBuf; loSrc.loop = true;
    const hiSrc = this.ctx.createBufferSource(); hiSrc.buffer = hiBuf; hiSrc.loop = true;
    const loGain = this.ctx.createGain(); loGain.gain.value = 0;
    const hiGain = this.ctx.createGain(); hiGain.gain.value = 0;
    loSrc.connect(loGain).connect(this.masterGain);
    hiSrc.connect(hiGain).connect(this.masterGain);
    loSrc.start(now, offset);
    hiSrc.start(now, offset);

    const fadeS = Math.max(0.001, fadeMs / 1000);
    loGain.gain.linearRampToValueAtTime(1 - neigh.t, now + fadeS);
    hiGain.gain.linearRampToValueAtTime(neigh.t, now + fadeS);

    [oldLoGain, oldHiGain].forEach((g) => {
      if (!g) return;
      g.gain.cancelScheduledValues(now);
      g.gain.setValueAtTime(g.gain.value, now);
      g.gain.linearRampToValueAtTime(0, now + fadeS);
    });
    const oldSrcs = [oldLoSrc, oldHiSrc];
    const oldGains = [oldLoGain, oldHiGain];
    setTimeout(() => {
      oldSrcs.forEach((s) => { if (s) { try { s.stop(); } catch (e) { /* ya parada */ } s.disconnect(); } });
      oldGains.forEach((g) => { if (g) g.disconnect(); });
    }, fadeMs + 30);

    this.loSource = loSrc; this.hiSource = hiSrc;
    this.loGain = loGain; this.hiGain = hiGain;
    this._loStartTime = now - offset; this._hiStartTime = now - offset;
    this._curNBands = nBandsVal;
    this._curLoKey = keyMix(neigh.lo); this._curHiKey = keyMix(neigh.hi);
  }

  async start() {
    if (this.playing || !this.pareja) return;
    this.playing = true;
    this.masterGain.connect(this.ctx.destination);
    const nBandsVal = N_BANDS_GRID[this.nBandsIdx];
    const neigh = resolveColorNeighbors(COLOR_MIX_GRID, this.colorMix);
    this._lastNeighbor = neigh;
    await this._swapTo(nBandsVal, neigh, 30);
  }

  stop() {
    this.playing = false;
    [this.loSource, this.hiSource].forEach((s) => {
      if (s) { try { s.stop(); } catch (e) { /* ya parada */ } s.disconnect(); }
    });
    [this.loGain, this.hiGain].forEach((g) => { if (g) g.disconnect(); });
    this.loSource = this.hiSource = this.loGain = this.hiGain = null;
    this._curNBands = null; this._curLoKey = null; this._curHiKey = null;
    try { this.masterGain.disconnect(); } catch (e) { /* ya desconectado */ }
  }

  setVolume(v) {
    this.masterGain.gain.value = clamp01(v);
  }

  /** Slider continuo de color_mix. Si el segmento (par de vecinos) no
   * cambia, solo reajusta las ganancias (crossfade suave, sin recrear
   * fuentes); si cambia de segmento, decodifica bajo demanda y hace el
   * intercambio preservando posicion. */
  async setColorMix(v) {
    this.colorMix = clamp01(v);
    const neigh = resolveColorNeighbors(COLOR_MIX_GRID, this.colorMix);
    this._lastNeighbor = neigh;
    this._emitReadout();
    if (!this.playing) return;
    const nBandsVal = N_BANDS_GRID[this.nBandsIdx];
    const loKey = keyMix(neigh.lo), hiKey = keyMix(neigh.hi);
    if (nBandsVal === this._curNBands && loKey === this._curLoKey && hiKey === this._curHiKey) {
      // El par que ya esta sonando es el correcto para esta posicion del
      // slider: cualquier _swapTo todavia en vuelo (p.ej. arrastrar A->B->A
      // rapido, donde la decodificacion de B resuelve DESPUES de volver a
      // A) esta pidiendo un segmento que ya hemos abandonado. Invalidarlo
      // aqui (incrementar _swapSeq) para que su guardia lo descarte al
      // resolver -- si no, B llegaria mas tarde y pisaria este estado
      // correcto con uno obsoleto (ver comentario de _swapSeq en el
      // constructor).
      ++this._swapSeq;
      const now = this.ctx.currentTime;
      this.loGain.gain.cancelScheduledValues(now);
      this.loGain.gain.setValueAtTime(this.loGain.gain.value, now);
      this.loGain.gain.linearRampToValueAtTime(1 - neigh.t, now + 0.03);
      this.hiGain.gain.cancelScheduledValues(now);
      this.hiGain.gain.setValueAtTime(this.hiGain.gain.value, now);
      this.hiGain.gain.linearRampToValueAtTime(neigh.t, now + 0.03);
      return;
    }
    await this._swapTo(nBandsVal, neigh, 30);
  }

  /** Slider discreto de n_bands (indice 0..7, NUNCA interpolado -- ver
   * docstring del modulo). Conmuta preservando posicion + ~20ms de
   * antichasquidos. */
  async setNBandsIndex(idx) {
    const clamped = Math.max(0, Math.min(N_BANDS_GRID.length - 1, idx));
    if (clamped === this.nBandsIdx) return;
    this.nBandsIdx = clamped;
    this._emitReadout();
    if (!this.playing) return;
    const nBandsVal = N_BANDS_GRID[this.nBandsIdx];
    const neigh = this._lastNeighbor || resolveColorNeighbors(COLOR_MIX_GRID, this.colorMix);
    await this._swapTo(nBandsVal, neigh, 20);
  }

  _emitReadout() {
    if (!this.onReadout || !this.pareja) return;
    this.onReadout({
      pareja: this.pareja,
      nBandsIdx: this.nBandsIdx,
      nBandsVal: N_BANDS_GRID[this.nBandsIdx],
      colorMix: this.colorMix,
      neighbor: this._lastNeighbor,
    });
  }
}

// ====================================================================
// Modo file:// -- SIN crossfade (ver docstring del modulo): un solo
// <audio> conmutando de src al punto de rejilla renderizado mas cercano.
// ====================================================================

function crearControladorFallback(audioEl) {
  let parejaObj = null;
  let nBandsIdx = 0;
  let colorMix = 0.5;
  let ficheroActual = null;

  function parejaActiva(p) {
    parejaObj = p;
    const heurIdx = N_BANDS_GRID.indexOf(p.n_bands_heuristica);
    nBandsIdx = heurIdx >= 0 ? heurIdx : Math.floor(N_BANDS_GRID.length / 2);
    colorMix = 0.5;
    ficheroActual = null;
    actualizar();
  }

  function actualizar() {
    if (!parejaObj) return;
    const nBandsVal = N_BANDS_GRID[nBandsIdx];
    const idx = nearestColorIndex(COLOR_MIX_GRID, colorMix);
    const mixVal = COLOR_MIX_GRID[idx];
    const clip = lookupGridClip(parejaObj, nBandsVal, keyMix(mixVal));
    if (!clip || clip.fichero === ficheroActual) return;
    ficheroActual = clip.fichero;
    const wasPlaying = !audioEl.paused;
    const prevTime = audioEl.currentTime || 0;
    const onReady = () => {
      audioEl.removeEventListener("loadedmetadata", onReady);
      try {
        audioEl.currentTime = Math.min(prevTime, Math.max(0, (audioEl.duration || 0) - 0.05));
      } catch (e) { /* metadata aun no lista en algun navegador: ignorar */ }
      if (wasPlaying) audioEl.play().catch(() => {});
    };
    audioEl.addEventListener("loadedmetadata", onReady);
    audioEl.src = clip.fichero;
    audioEl.load();
    return clip;
  }

  return {
    seleccionarPareja: parejaActiva,
    setNBandsIndex(idx) { nBandsIdx = Math.max(0, Math.min(N_BANDS_GRID.length - 1, idx)); return actualizar(); },
    setColorMix(v) { colorMix = clamp01(v); return actualizar(); },
    get audioEl() { return audioEl; },
  };
}

// ====================================================================
// Cableado de interfaz -- solo se ejecuta si hay DOM real (navegador). Al
// cargar este fichero con Node (para probar las funciones puras de
// arriba), `document` no existe y este bloque entero se salta.
// ====================================================================

function initPage() {
  const modo = chooseMode(window.location.protocol);
  const fileWarning = document.getElementById("file-warning");
  if (modo === "file") fileWarning.classList.remove("hidden");

  let ctx = null;
  let pad = null;
  let parejaActualId = PAREJA_IDS[0];
  const fallback = modo === "file" ? crearControladorFallback(document.getElementById("fallback-audio")) : null;

  const nbandsSlider = document.getElementById("slider-nbands");
  const colormixSlider = document.getElementById("slider-colormix");
  const nbandsReadout = document.getElementById("nbands-readout");
  const colormixReadout = document.getElementById("colormix-readout");
  const colormixWarning = document.getElementById("colormix-warning");
  const explicacion = document.getElementById("explicacion");
  const activeMetrics = document.getElementById("active-metrics");
  const progressEl = document.getElementById("progress-indicator");
  const playBtn = document.getElementById("play-btn");
  const stopBtn = document.getElementById("stop-btn");
  const volSlider = document.getElementById("master-vol");
  const abGrid = document.getElementById("ab-grid");
  const parejaNav = document.getElementById("pareja-selector");
  const nbandsTickLabels = document.getElementById("nbands-tick-labels");

  // Etiquetas fijas bajo el slider discreto (numero REAL de bandas -- ver
  // brief: "etiquetado con el numero real de bandas"; datalist por si el
  // navegador dibuja las muescas, pero no nos fiamos de que se vea
  // proyectado, asi que ademas hay texto real).
  N_BANDS_GRID.forEach((nb) => {
    const span = document.createElement("span");
    span.textContent = String(nb);
    nbandsTickLabels.appendChild(span);
  });

  function currentGridWarning(readoutInfo) {
    const nBandsVal = readoutInfo.nBandsVal;
    const loClip = lookupGridClip(readoutInfo.pareja, nBandsVal, keyMix(readoutInfo.neighbor.lo));
    const hiClip = lookupGridClip(readoutInfo.pareja, nBandsVal, keyMix(readoutInfo.neighbor.hi));
    return (loClip && loClip.metodo_igualacion === "soft_limit") ||
           (hiClip && hiClip.metodo_igualacion === "soft_limit");
  }

  /** Construye el objeto {pareja, nBandsIdx, nBandsVal, colorMix, neighbor}
   * que espera actualizarLecturas(), leyendo el estado ACTUAL de los
   * sliders -- usado en modo file:// (donde no hay FusionPad1D emitiendo
   * lecturas via onReadout) para que cada cambio de slider actualice el
   * texto en pantalla, no solo el audio. */
  function infoLecturaActual(parejaObj) {
    const colorMixVal = parseFloat(colormixSlider.value);
    return {
      pareja: parejaObj,
      nBandsIdx: parseInt(nbandsSlider.value, 10),
      nBandsVal: N_BANDS_GRID[parseInt(nbandsSlider.value, 10)],
      colorMix: colorMixVal,
      neighbor: resolveColorNeighbors(COLOR_MIX_GRID, colorMixVal),
    };
  }

  function actualizarLecturas(info) {
    nbandsReadout.textContent = "Bandas del filtro: " + info.nBandsVal +
      (info.nBandsVal === info.pareja.n_bands_heuristica ? " (heurística de esta pareja)" : "");
    const neigh = info.neighbor;
    if (modo === "file") {
      // Modo degradado: NO hay crossfade (ver LEEME) -- el slider conmuta
      // al punto de rejilla grabado mas cercano, asi que el texto tiene
      // que decir eso, no describir una interpolacion que no esta pasando.
      const idxCercano = nearestColorIndex(COLOR_MIX_GRID, info.colorMix);
      const valCercano = COLOR_MIX_GRID[idxCercano];
      colormixReadout.textContent = "modo sin servidor: conmutado al punto grabado más cercano, " +
        "color_mix = " + valCercano.toFixed(3) + " (sin crossfade en este modo, ver aviso de arriba)";
    } else if (esPosicionRenderizada(neigh)) {
      const exacto = neigh.t < 0.5 ? neigh.lo : neigh.hi;
      colormixReadout.textContent = "color_mix real = " + exacto.toFixed(3) + " (posición grabada)";
    } else {
      const pctHi = Math.round(neigh.t * 100);
      colormixReadout.textContent = "crossfade entre " + neigh.lo.toFixed(3) + " y " + neigh.hi.toFixed(3) +
        " (" + (100 - pctHi) + "% / " + pctHi + "% -- interpolación aritmética, NO es un color_mix real, ver LEEME)";
    }
    if (currentGridWarning(info)) {
      colormixWarning.textContent = "⚠ este tramo pasa por un limitador no lineal en la igualación de sonoridad: " +
        "aquí el crossfade deja de ser exacto (ver LEEME, sección de honestidades).";
      colormixWarning.classList.remove("hidden");
    } else {
      colormixWarning.classList.add("hidden");
    }
    explicacion.textContent = info.pareja.frase;

    // Las metricas mostradas tienen que coincidir con lo que REALMENTE
    // suena, no con "los dos vecinos de la formula" -- en modo file:// y en
    // cualquier posicion exacta del slider de color solo hay UN clip
    // audible (ganancia 1 en uno, 0 en el otro); listar los dos sin peso
    // sugeriria dos contribuciones simultaneas cuando solo hay una. En el
    // crossfade real (posicion intermedia) SI suenan los dos a la vez, y
    // ahi se muestra el peso de cada uno por la misma razon que no se
    // inventa un crest_db del blend: dos numeros desnudos sugeririan dos
    // contribuciones iguales.
    const nBandsVal = info.nBandsVal;
    const partes = [];
    if (modo === "file") {
      const idxCercano = nearestColorIndex(COLOR_MIX_GRID, info.colorMix);
      const valCercano = COLOR_MIX_GRID[idxCercano];
      const clip = lookupGridClip(info.pareja, nBandsVal, keyMix(valCercano));
      if (clip) partes.push("punto grabado " + valCercano.toFixed(3) + " (el unico que suena en este modo): " +
        "crest=" + fmtDb(clip.crest_db) + ", SSO=" + fmtSso(clip.sso));
    } else if (esPosicionRenderizada(neigh)) {
      const exacto = neigh.t < 0.5 ? neigh.lo : neigh.hi;
      const clip = lookupGridClip(info.pareja, nBandsVal, keyMix(exacto));
      if (clip) partes.push("punto " + exacto.toFixed(3) + " (el unico que suena aqui, ganancia 1): " +
        "crest=" + fmtDb(clip.crest_db) + ", SSO=" + fmtSso(clip.sso));
    } else {
      const loClip = lookupGridClip(info.pareja, nBandsVal, keyMix(neigh.lo));
      const hiClip = lookupGridClip(info.pareja, nBandsVal, keyMix(neigh.hi));
      const pctHi = Math.round(neigh.t * 100);
      if (loClip) partes.push("vecino " + neigh.lo.toFixed(3) + " (" + (100 - pctHi) + "% de ganancia): " +
        "crest=" + fmtDb(loClip.crest_db) + ", SSO=" + fmtSso(loClip.sso));
      if (hiClip) partes.push("vecino " + neigh.hi.toFixed(3) + " (" + pctHi + "% de ganancia): " +
        "crest=" + fmtDb(hiClip.crest_db) + ", SSO=" + fmtSso(hiClip.sso));
    }
    activeMetrics.textContent = "Punto activo de la rejilla (métricas medidas sobre el audio crudo, antes de igualar sonoridad) -- " +
      partes.join(" · ");
  }

  function construirPanelAB(parejaObj) {
    abGrid.innerHTML = "";
    REFERENCIA_ORDEN.forEach((nombre) => {
      const rec = lookupReferencia(parejaObj, nombre);
      const card = document.createElement("div");
      card.className = "ab-card";
      const h3 = document.createElement("h3");
      h3.textContent = REFERENCIA_ETIQUETAS[nombre] || nombre;
      const metric = document.createElement("div");
      metric.className = "metric";
      metric.textContent = rec ? ("crest = " + fmtDb(rec.crest_db) + " · SSO = " + fmtSso(rec.sso)) : "(sin datos)";
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      if (rec) audio.src = rec.fichero;
      audio.addEventListener("play", () => { if (pad) pad.stop(); if (fallback) fallback.audioEl.pause(); });
      card.appendChild(h3);
      card.appendChild(metric);
      card.appendChild(audio);
      abGrid.appendChild(card);
    });
  }

  function seleccionarPareja(parejaId) {
    const parejaObj = FUSION_MANIFEST.parejas[parejaId];
    parejaActualId = parejaId;
    Array.from(parejaNav.children).forEach((btn) => {
      btn.classList.toggle("activo", btn.dataset.pareja === parejaId);
    });
    nbandsSlider.value = String(N_BANDS_GRID.indexOf(parejaObj.n_bands_heuristica) >= 0 ?
      N_BANDS_GRID.indexOf(parejaObj.n_bands_heuristica) : Math.floor(N_BANDS_GRID.length / 2));
    colormixSlider.value = "0.5";
    construirPanelAB(parejaObj);
    if (modo === "server") {
      if (pad) pad.loadPareja(parejaObj);
      actualizarLecturas(infoLecturaActual(parejaObj));
    } else {
      fallback.seleccionarPareja(parejaObj);
      actualizarLecturas(infoLecturaActual(parejaObj));
    }
  }

  PAREJA_IDS.forEach((id, i) => {
    const p = FUSION_MANIFEST.parejas[id];
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pareja-btn";
    btn.dataset.pareja = id;
    btn.textContent = p.titulo;
    btn.addEventListener("click", () => seleccionarPareja(id));
    parejaNav.appendChild(btn);
    if (i === 0) btn.classList.add("activo");
  });

  function ensureCtx() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      pad = new FusionPad1D(ctx, {
        onProgress: (activo) => progressEl.classList.toggle("hidden", !activo),
        onReadout: actualizarLecturas,
      });
      pad.loadPareja(FUSION_MANIFEST.parejas[parejaActualId]);
      // El usuario puede haber movido los sliders ANTES del primer Play
      // (no habia pad todavia para recibir esos eventos "input"):
      // sincroniza el pad recien creado con la posicion actual en pantalla
      // en vez de arrancar siempre en los valores por defecto de
      // loadPareja. playing=false aqui (loadPareja llama a stop()), asi
      // que estas llamadas solo actualizan estado + lectura, no decodifican.
      pad.setNBandsIndex(parseInt(nbandsSlider.value, 10));
      pad.setColorMix(parseFloat(colormixSlider.value));
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  if (modo === "server") {
    playBtn.addEventListener("click", async () => {
      ensureCtx();
      if (fallback) fallback.audioEl.pause();
      await pad.start();
    });
    stopBtn.addEventListener("click", () => { if (pad) pad.stop(); });
    nbandsSlider.addEventListener("input", () => {
      // Antes del primer Play, pad todavia no existe (el AudioContext se
      // crea con un gesto real del usuario, ver ensureCtx): en ese caso
      // solo actualizamos el texto en pantalla, no hay nada que decodificar
      // todavia. ensureCtx() sincroniza el pad con estos sliders al crearlo.
      if (pad) pad.setNBandsIndex(parseInt(nbandsSlider.value, 10));
      else actualizarLecturas(infoLecturaActual(FUSION_MANIFEST.parejas[parejaActualId]));
    });
    colormixSlider.addEventListener("input", () => {
      if (pad) pad.setColorMix(parseFloat(colormixSlider.value));
      else actualizarLecturas(infoLecturaActual(FUSION_MANIFEST.parejas[parejaActualId]));
    });
    volSlider.addEventListener("input", () => { if (pad) pad.setVolume(parseFloat(volSlider.value)); });
  } else {
    // Modo file://: sin AudioContext, sin crossfade -- conmutacion directa
    // del <audio> de fallback. Los botones de play/stop controlan ese
    // elemento nativo.
    playBtn.addEventListener("click", () => { fallback.audioEl.play().catch(() => {}); });
    stopBtn.addEventListener("click", () => { fallback.audioEl.pause(); });
    nbandsSlider.addEventListener("input", () => {
      fallback.setNBandsIndex(parseInt(nbandsSlider.value, 10));
      actualizarLecturas(infoLecturaActual(FUSION_MANIFEST.parejas[parejaNav.querySelector(".activo").dataset.pareja]));
    });
    colormixSlider.addEventListener("input", () => {
      fallback.setColorMix(parseFloat(colormixSlider.value));
      actualizarLecturas(infoLecturaActual(FUSION_MANIFEST.parejas[parejaNav.querySelector(".activo").dataset.pareja]));
    });
    volSlider.addEventListener("input", () => { fallback.audioEl.volume = parseFloat(volSlider.value); });
  }

  seleccionarPareja(PAREJA_IDS[0]);
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", initPage);
}

// ====================================================================
// Exportacion SOLO para pruebas con Node (ver scripts/43_demo_fusion_web.py
// y el informe de la tarea: "node <script> contra el manifiesto real", sin
// navegador). En un <script src="fusionpad.js"> normal de navegador
// `module` no existe, así que este bloque nunca se ejecuta ahí.
// ====================================================================
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    FUSION_MANIFEST, N_BANDS_GRID, COLOR_MIX_GRID, PAREJA_IDS,
    clamp01, keyMix, resolveColorNeighbors, esPosicionRenderizada,
    lookupGridClip, lookupReferencia, isFileProtocol, chooseMode,
    nearestColorIndex, fmtDb, fmtSso, FusionPad1D, crearControladorFallback,
  };
}
'''


def render_fusionpad_js(pareja_index: dict, manifest: dict) -> str:
    fusion_manifest_obj = {
        "meta": {
            "n_bands_grid": manifest["meta"]["n_bands_grid"],
            "color_mix_grid": manifest["meta"]["color_mix_grid"],
            "duracion_s": manifest["meta"]["duration_s"],
            "sr_audio": manifest["meta"]["sr_audio"],
        },
        "parejas": pareja_index,
    }
    blob = json.dumps(fusion_manifest_obj, ensure_ascii=False, indent=None, sort_keys=False)
    return _FUSIONPAD_JS_TEMPLATE.replace("__FUSION_MANIFEST_JSON__", blob)


# ======================================================================
# index.html -- oscuro, una columna, tipografia grande (proyeccion en
# sala). Espiritu de demo_congreso/index.html, pero con controles reales
# (sliders + panel A/B) en vez de una lista de <audio>.
# ======================================================================

_INDEX_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>De dos sonidos a un objeto imposible — demo de fusión</title>
<style>
:root {
  --bg: #0a0d12;
  --panel: #141a23;
  --panel2: #1c2531;
  --text: #f4f6f8;
  --muted: #b7c2d0;
  --dinamica: #ff9248;
  --materia: #4fd1ff;
  --warn-bg: #3a2a12;
  --warn-text: #ffd485;
  --border: #2a3242;
  --ok: #6fdc8c;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; padding: 0 1.2rem 4rem;
  background: var(--bg); color: var(--text);
  font-family: system-ui, "Segoe UI", Roboto, sans-serif;
  font-size: clamp(16px, 1.1vw + 0.6rem, 20px);
  line-height: 1.5;
}
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size: clamp(1.6rem, 1rem + 2vw, 2.6rem); margin: 1.4rem 0 0.3rem; }
h2 { font-size: clamp(1.2rem, 0.9rem + 1vw, 1.6rem); margin: 2.2rem 0 0.6rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
h3 { font-size: 1.05rem; margin: 0 0 0.3rem; }
p.lead { color: var(--muted); font-size: 1.05em; max-width: 62ch; }
.hidden { display: none !important; }

#file-warning {
  background: var(--warn-bg); color: var(--warn-text);
  border: 1px solid #6b4d1a; border-radius: 10px;
  padding: 1rem 1.2rem; margin: 1rem 0; font-size: 1rem;
}
#file-warning code { background: #00000055; padding: 0.1rem 0.4rem; border-radius: 4px; }

nav#pareja-selector { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 1rem 0 1.6rem; }
.pareja-btn {
  background: var(--panel); color: var(--text); border: 1px solid var(--border);
  border-radius: 999px; padding: 0.6rem 1.1rem; font-size: 1rem; cursor: pointer;
}
.pareja-btn.activo { background: linear-gradient(90deg, var(--dinamica), var(--materia)); color: #0a0d12; font-weight: 600; border-color: transparent; }

section { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.4rem; }

#explicacion { font-size: 1.15em; margin: 0 0 1.2rem; }

.slider-block { margin: 1.4rem 0; }
.slider-block label { display: block; font-size: 1.02rem; color: var(--muted); margin-bottom: 0.5rem; }
input[type=range] { width: 100%; height: 2.2rem; accent-color: var(--materia); cursor: pointer; }
#slider-colormix { accent-color: var(--dinamica); }
.axis-endpoints { display: flex; justify-content: space-between; font-size: 0.9rem; color: var(--muted); margin-top: 0.2rem; }
.axis-endpoints .izq { color: var(--dinamica); } .axis-endpoints .der { color: var(--materia); }
#nbands-tick-labels { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--muted); margin-top: 0.2rem; padding: 0 0.2rem; }
.value-readout { margin-top: 0.5rem; font-size: 1.02rem; font-variant-numeric: tabular-nums; }
#colormix-warning { margin-top: 0.5rem; background: var(--warn-bg); color: var(--warn-text); border-radius: 8px; padding: 0.5rem 0.8rem; font-size: 0.95rem; }

.transport { display: flex; flex-wrap: wrap; align-items: center; gap: 1rem; margin: 1.2rem 0; }
.transport button {
  font-size: 1.05rem; padding: 0.6rem 1.3rem; border-radius: 10px; border: 1px solid var(--border);
  background: var(--panel2); color: var(--text); cursor: pointer;
}
#play-btn { border-color: var(--ok); }
.transport .vol-block { display: flex; align-items: center; gap: 0.5rem; min-width: 180px; }
.transport .vol-block input { width: 140px; height: 1.6rem; }
#progress-indicator { color: var(--materia); font-size: 0.95rem; }

#active-metrics { color: var(--muted); font-size: 0.95rem; margin-top: 0.6rem; }

.ab-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-top: 1rem; }
.ab-card { background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }
.ab-card .metric { color: var(--muted); font-size: 0.9rem; margin: 0.3rem 0 0.6rem; }
.ab-card audio { width: 100%; }

footer { color: var(--muted); font-size: 0.9rem; margin-top: 2rem; }
footer code { background: var(--panel2); padding: 0.15rem 0.45rem; border-radius: 5px; }

@media (max-width: 600px) {
  section { padding: 1rem; }
  .transport { flex-direction: column; align-items: stretch; }
}
</style>
</head>
<body>
<div class="wrap">
  <h1>De dos sonidos a un objeto imposible</h1>
  <p class="lead">Fusión de identidad por chimera auditiva: un padre pone la <b>dinámica</b>
  (envolvente/ritmo), otro pone la <b>materia</b> (estructura fina/timbre) — una sola señal por
  construcción, no dos sonidos sonando a la vez. Mueve los sliders y compáralo con el panel A/B
  para oír la diferencia.</p>

  <div id="file-warning" class="hidden">
    <b>Modo degradado (sin servidor):</b> esta página se abrió con <code>file://</code>, donde el
    navegador bloquea la carga de audio por <code>fetch</code>. Los sliders siguen funcionando,
    pero conmutan al punto de rejilla grabado más cercano <b>sin crossfade</b> (con un pequeño
    corte al cambiar). Para la experiencia completa (crossfade real de color, offline, sin wifi):
    abre una terminal en la carpeta del proyecto y ejecuta
    <code>python3 -m http.server -d demo_fusion 8000</code>, luego visita
    <code>http://localhost:8000/</code>.
  </div>

  <nav id="pareja-selector" role="tablist" aria-label="Selector de pareja"></nav>

  <section id="pad-section">
    <p id="explicacion"></p>

    <div class="slider-block">
      <label for="slider-nbands">Bandas del filtro — eje discreto, se conmuta (nunca se interpola entre bandas)</label>
      <input type="range" id="slider-nbands" min="0" max="7" step="1" value="4" list="nbands-ticks">
      <datalist id="nbands-ticks">
        <option value="0"></option><option value="1"></option><option value="2"></option>
        <option value="3"></option><option value="4"></option><option value="5"></option>
        <option value="6"></option><option value="7"></option>
      </datalist>
      <div id="nbands-tick-labels"></div>
      <div id="nbands-readout" class="value-readout">—</div>
    </div>

    <div class="slider-block">
      <label for="slider-colormix">Color: dinámica ↔ materia — eje continuo, con crossfade real</label>
      <input type="range" id="slider-colormix" min="0" max="1" step="0.001" value="0.5" list="colormix-ticks">
      <datalist id="colormix-ticks">
        <option value="0"></option><option value="0.125"></option><option value="0.25"></option>
        <option value="0.375"></option><option value="0.5"></option><option value="0.625"></option>
        <option value="0.75"></option><option value="0.875"></option><option value="1"></option>
      </datalist>
      <div class="axis-endpoints"><span class="izq">dinámica</span><span class="der">materia</span></div>
      <div id="colormix-readout" class="value-readout">—</div>
      <div id="colormix-warning" class="hidden"></div>
    </div>

    <div class="transport">
      <button id="play-btn" type="button">▶ Reproducir</button>
      <button id="stop-btn" type="button">■ Detener</button>
      <div class="vol-block">
        <label for="master-vol">Volumen</label>
        <input type="range" id="master-vol" min="0" max="1" step="0.02" value="0.85">
      </div>
      <div id="progress-indicator" class="hidden">decodificando…</div>
    </div>
    <div id="active-metrics"></div>
    <audio id="fallback-audio" preload="none" loop></audio>
  </section>

  <section id="ab-section">
    <h2>Panel A/B — el argumento del paper, con el sonido delante</h2>
    <p class="lead">Cuatro referencias fijas de la pareja activa (fuera de la rejilla de
    crossfade: la igualación de sonoridad les aplica a veces un limitador no lineal). Compara su
    <code>crest_db</code> y <code>SSO</code> (medidos sobre el audio crudo, antes de igualar
    sonoridad — igual que el punto activo de arriba, los números son comparables entre sí) con los
    del punto activo: la cresta alta separa lo defectuoso ("dos capas") de lo sano.</p>
    <div class="ab-grid" id="ab-grid"></div>
  </section>

  <footer>
    Servir sin wifi: <code>python3 -m http.server -d demo_fusion 8000</code> desde la raíz del
    proyecto, luego abrir <code>http://localhost:8000/</code>. Detalles, honestidades y
    limitaciones conocidas: <code>demo_fusion/LEEME.md</code>.
  </footer>
</div>
<noscript>Esta demo necesita JavaScript (Web Audio API) para reproducir sonido.</noscript>
<script src="fusionpad.js"></script>
</body>
</html>
'''


def render_index_html() -> str:
    return _INDEX_HTML_TEMPLATE


# ======================================================================
# LEEME.md
# ======================================================================

def render_leeme(manifest: dict, pareja_index: dict) -> str:
    soft_cells = _soft_limit_grid_cells(pareja_index)
    soft_lines = "\n".join(
        f"   - `{p}`: n_bands={nb}, color_mix={mix}"
        for p, nb, mix in soft_cells) or "   (ninguna en este manifiesto)"
    n_parejas = len(pareja_index)
    n_grid = n_parejas * len(manifest["meta"]["n_bands_grid"]) * len(manifest["meta"]["color_mix_grid"])
    n_ref = n_parejas * len(REFERENCIA_NOMBRES)
    return f"""# Demo de fusión con sliders (`demo_fusion/`)

Generada por `scripts/43_demo_fusion_web.py` a partir de `demo_fusion/manifest.json`
({manifest["meta"]["n_clips_total"]} clips Opus: {n_grid} de rejilla + {n_ref} de
referencia, commit `d0731d4` de `scripts/42_fusion_set.py` -- este script NO regenera
el audio, solo construye la página que lo consume). Vuelve a ejecutar el generador
si cambia `manifest.json`; si no, `index.html`/`fusionpad.js` quedan desactualizados
respecto al audio real (usa `--check` para detectarlo, ver más abajo).

## Cómo servirla (importante: "sin wifi" no significa `file://`)

**Ruta principal, totalmente offline:**
```
python3 -m http.server -d demo_fusion 8000
```
y abrir `http://localhost:8000/`. `fetch` (necesario para decodificar los clips
bajo demanda) funciona con el wifi apagado siempre que sea `http://localhost` --
no hace ninguna petición externa (sin CDN, sin fuentes remotas, sin analítica).

**Si alguien abre `index.html` directamente con doble clic (`file://`):** `fetch`
falla ahí en Chrome y Safari (origen opaco), así que la página lo detecta
(`location.protocol === 'file:'`) y degrada sola a conmutación de `<audio src>`
sin crossfade, con un aviso visible en pantalla que recuerda el comando de arriba.
Sigue siendo utilizable para enseñar algo en una emergencia, pero sin la
interpolación continua del eje de color.

No copies los 21 MB de audio en base64 dentro del HTML/JS: es peor que servir la
carpeta (carga todo de golpe, nada de decodificación bajo demanda). Los ficheros
`.opus` viven en `demo_fusion/audio/` (no versionado en git, ~21 MB, regenerable
con `PYTHONPATH=. .venv/bin/python scripts/42_fusion_set.py`).

## Los dos ejes: NO son intercambiables

- **`color_mix` (9 valores grabados, 0 a 1): eje continuo, con crossfade real.**
  El slider se mueve de forma continua y la página reproduce simultáneamente los
  2 clips vecinos de la rejilla con ganancias interpoladas (2 `BufferSourceNode` +
  2 `GainNode`, igual que `web/pad2d.js` pero en 1-D). Está verificado en
  `impossible_mix/physics/blend.py` (bloque ~938-980) que la contribución por
  banda `parts[k]` NO depende de `color_mix` -- solo cambian los pesos
  `band_w` -- así que mezclar linealmente dos salidas vecinas da otro miembro
  EXACTO de la misma familia de chimeras, sin interferencia de fase.
- **`n_bands` (8 valores: {", ".join(str(n) for n in manifest["meta"]["n_bands_grid"])}): eje discreto, con conmutación.**
  NUNCA se interpola entre dos valores de `n_bands`: cambia el banco de filtros
  entero, así que mezclar dos sonaría a dos chimeras simultáneas, no a una
  intermedia. El slider tiene muescas (`<datalist>` + los 8 números reales
  siempre visibles bajo la barra). Al conmutar, la página preserva la posición
  de reproducción (el offset dentro del bucle de 6 s) y aplica ~20 ms de fundido
  **solo como antichasquidos** -- eso NO es interpolación entre familias.

## Honestidades (norma del proyecto: lo que se sabe que es aproximado, se dice)

1. **El crossfade interpola aritméticamente los pesos de ganancia; `color_mix`
   es una interpolación geométrica (dominio logarítmico, ver
   `blend.py:966-967`: `exp((1-color_mix)*log(rms_a) + color_mix*log(rms_b))`).
   Una posición intermedia del slider de color NO equivale al `color_mix` que su
   etiqueta numérica sugeriría.** Por eso el eje se etiqueta "color: dinámica ↔
   materia" (una descripción cualitativa) y el número exacto de `color_mix`
   solo se muestra cuando el slider está, con tolerancia de punto flotante,
   sobre una de las 9 posiciones realmente grabadas; en cualquier otra posición
   la página muestra "crossfade entre X e Y (aritmético, no es un color_mix
   real)".
2. **Las 4 referencias (`baseline_v11`, `suma_ancla`, `sin_alinear`, `plana`)
   quedan fuera de la ruta de crossfade.** Son botones A/B independientes, no
   posiciones de slider: la igualación de sonoridad les aplica a veces un
   limitador no lineal (`metodo_igualacion="soft_limit"`, ver más abajo) por su
   cresta patológica -- de hecho `baseline_v11` de `trueno_hecho_de_agua` y de
   `trueno_hecho_de_canica` llegan a ~47 dB de cresta, el síntoma medido de "dos
   capas" que este trabajo intenta corregir. La afirmación de exactitud del
   punto anterior se hace SOLO sobre la rejilla principal (432 puntos), nunca
   sobre estas 4 referencias por pareja (24 puntos).
3. **Dentro de la propia rejilla principal, 5 celdas rompen la exactitud del
   crossfade.** La propiedad de "combinación lineal exacta, sin interferencia
   de fase" (honestidad 1) depende de que la igualación de sonoridad de AMBOS
   clips vecinos haya sido una escala lineal (`metodo_igualacion` "linear" o
   "linear_capped": `audio_final = factor * audio_crudo`, un escalar puro que
   preserva la forma de onda). En el manifiesto real hay 5 celdas de
   `trueno_hecho_de_canica` con `metodo_igualacion="soft_limit"` (compresión
   `tanh`, no lineal, porque su cresta cruda era demasiado patológica para el
   recorte lineal):
{soft_lines}
   Cuando el punto activo o su vecino de crossfade cae en una de estas celdas,
   `fusionpad.js` lo detecta en tiempo real (compara `metodo_igualacion` del
   vecino contra `"soft_limit"`) y muestra un aviso junto al slider de color.
   El resto de la rejilla (427 de las 432 celdas) sí cumple la exactitud sin
   reservas.

## Restricción de memoria: decodificación bajo demanda, no precarga

72 clips de rejilla x 6 s x 44 100 Hz x 4 bytes (Float32 decodificado) ≈ 76 MB
por pareja; las 6 parejas a la vez serían ~457 MB. `fusionpad.js`
(`FusionPad1D`) decodifica solo los 2 clips vecinos que hacen falta para la
posición actual de los sliders, cachea por pareja (`Map` fichero→AudioBuffer)
y VACÍA ese cache al cambiar de pareja (`loadPareja` llama a `buffers.clear()`).
"Vaciar" aquí significa soltar las referencias -- JavaScript no tiene
liberación explícita de memoria, así que lo que se garantiza es que el código
ya no retiene el `AudioBuffer`; el recolector de basura del navegador hace el
resto en su propio momento, no de forma instantánea. **No precargar las 6
parejas de golpe "para que vaya más fluido"**: es exactamente el patrón que
esta restricción está pensada para evitar, y en un portátil modesto de sala
de congreso puede ser la diferencia entre que la demo funcione o se cuelgue.

## Panel A/B

Junto a los sliders, cuatro botones (`<audio controls>` nativos, funcionan
igual en modo servidor y en modo `file://`) reproducen las 4 referencias de la
pareja activa, con su `crest_db` y `SSO` del manifiesto al lado. Los "vecinos"
del punto activo de la rejilla (los dos clips que se están crossfadeando)
muestran sus propios `crest_db`/`SSO` debajo de los sliders para comparar en
la sala, con el sonido delante: la cresta alta es el síntoma medido de "dos
sonidos superpuestos" en vez de un objeto fundido. No se muestra un número de
cresta "del blend": el crest factor no es un promedio ponderado simple de sus
dos extremos (es una métrica no lineal sobre la forma de onda), así que
inventar uno sería falsa precisión -- se muestran los dos valores reales
medidos en los extremos que se están mezclando.

## Verificación

```
PYTHONPATH=. .venv/bin/python scripts/43_demo_fusion_web.py --check
```
Sin navegador: valida que `index.html`/`fusionpad.js` en disco NO están
desactualizados respecto a `manifest.json` (recalcula el índice y lo compara
byte a byte con el que hay horneado en `fusionpad.js`), que los 456 clips
referenciados existen en `demo_fusion/audio/`, que los 6 selectores de pareja
resuelven, que las 4 referencias están en cada una, y que no hay ninguna URL
externa en el HTML ni en el JS.

## Convenciones

Español en la interfaz; sin tildes en identificadores. Determinismo: el HTML y
el JS generados dependen solo de `manifest.json`, nunca de aleatoriedad ni del
reloj.
"""


# ======================================================================
# --check
# ======================================================================

_FUSION_MANIFEST_RE = re.compile(
    r"const FUSION_MANIFEST = (\{.*\});\s*\n// ==== FIN DATOS GENERADOS", re.S)


def _extract_baked_manifest(js_text: str) -> dict:
    m = _FUSION_MANIFEST_RE.search(js_text)
    if not m:
        raise ValueError("no se encontro el bloque 'const FUSION_MANIFEST = ...;' "
                          "delimitado por los marcadores INICIO/FIN DATOS GENERADOS")
    return json.loads(m.group(1))


def check(out_dir: Path) -> int:
    failures: list[str] = []
    manifest = load_manifest(out_dir)
    fresh_index = build_pareja_index(manifest)
    fresh_fusion_manifest = {
        "meta": {
            "n_bands_grid": manifest["meta"]["n_bands_grid"],
            "color_mix_grid": manifest["meta"]["color_mix_grid"],
            "duracion_s": manifest["meta"]["duration_s"],
            "sr_audio": manifest["meta"]["sr_audio"],
        },
        "parejas": fresh_index,
    }

    html_path = out_dir / "index.html"
    js_path = out_dir / "fusionpad.js"
    if not html_path.exists() or not js_path.exists():
        print(f"AVISO: {html_path} o {js_path} no existen todavia -- ejecuta el "
              f"generador sin --check primero. Comprobando solo la logica del indice "
              f"contra el manifiesto (sin frescura de disco).")
        baked = fresh_fusion_manifest
    else:
        js_text = js_path.read_text(encoding="utf-8")
        try:
            baked = _extract_baked_manifest(js_text)
        except ValueError as e:
            failures.append(f"fusionpad.js: {e}")
            baked = None

    # smoke 1: frescura -- lo horneado en disco == lo que produce el manifiesto AHORA
    if baked is not None:
        if baked != fresh_fusion_manifest:
            failures.append("demo_fusion/fusionpad.js esta DESACTUALIZADO respecto a "
                             "manifest.json (el indice horneado no coincide con uno "
                             "recien derivado) -- vuelve a ejecutar el generador")
        print(f"smoke1 (frescura fusionpad.js vs manifest.json): "
              f"{'OK' if baked == fresh_fusion_manifest else 'FALLO'}")

    # smoke 2: bijeccion manifiesto -> disco (todo fichero referenciado existe)
    all_ficheros = []
    for p in fresh_index.values():
        for by_mix in p["grid"].values():
            all_ficheros.extend(rec["fichero"] for rec in by_mix.values())
        all_ficheros.extend(rec["fichero"] for rec in p["referencias"].values())
    missing = [f for f in all_ficheros if not (out_dir / f).exists()]
    if missing:
        if not (out_dir / "audio").exists():
            failures.append(
                f"demo_fusion/audio/ no existe (falta TODO el audio, {len(missing)} "
                f"clips) -- no se versiona en git; regenera con "
                f"'PYTHONPATH=. .venv/bin/python scripts/42_fusion_set.py'")
        else:
            failures.append(f"{len(missing)} clips referenciados en el manifiesto no "
                            f"existen en disco, p.ej.: {missing[:5]}")
    print(f"smoke2 (bijeccion manifiesto->disco, {len(all_ficheros)} clips esperados): "
          f"{'OK' if not missing else 'FALLOS: ' + str(len(missing))}")

    # smoke 3: 6 selectores de pareja resuelven
    parejas_esperadas = set(fresh_index.keys())
    parejas_encontradas = set(baked["parejas"].keys()) if baked else set()
    ok3 = parejas_encontradas == parejas_esperadas and len(parejas_esperadas) == 6
    if not ok3:
        failures.append(f"parejas: esperadas {sorted(parejas_esperadas)}, "
                        f"encontradas {sorted(parejas_encontradas)}")
    print(f"smoke3 (6 selectores de pareja resuelven): {'OK' if ok3 else 'FALLO'}")

    # smoke 4: 4 referencias por pareja + 72 celdas de rejilla por pareja
    ok4 = True
    for pareja_id, entry in fresh_index.items():
        refs = set(entry["referencias"].keys())
        if refs != set(REFERENCIA_NOMBRES):
            failures.append(f"{pareja_id}: referencias = {sorted(refs)}, esperado "
                            f"{sorted(REFERENCIA_NOMBRES)}")
            ok4 = False
        n_grid_cells = sum(len(by_mix) for by_mix in entry["grid"].values())
        n_nb = len(entry["grid"])
        if n_grid_cells != 72 or n_nb != len(manifest["meta"]["n_bands_grid"]):
            failures.append(f"{pareja_id}: rejilla tiene {n_nb} n_bands x celdas "
                            f"totales {n_grid_cells} (esperado "
                            f"{len(manifest['meta']['n_bands_grid'])} x 72)")
            ok4 = False
    print(f"smoke4 (4 referencias + 72 celdas de rejilla por pareja, 6 parejas): "
          f"{'OK' if ok4 else 'FALLO'}")

    # smoke 5: cero URLs externas. "http://localhost..." NO cuenta -- es texto
    # instructivo del propio comando de servir la carpeta (obligatorio que
    # aparezca), no una peticion real a un host externo.
    url_re = re.compile(r"https?://(?!localhost\b)\S+")
    ok5 = True
    for path in (html_path, js_path):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        hits = url_re.findall(text)
        if hits:
            failures.append(f"{path}: {len(hits)} URL(s) externas encontradas: {hits}")
            ok5 = False
    print(f"smoke5 (cero URLs externas en index.html/fusionpad.js, localhost excluido): "
          f"{'OK' if ok5 else 'FALLO'}")

    # smoke 6: todo document.getElementById("...") que usa fusionpad.js
    # existe como id="..." en index.html. La interfaz de pareja/rejilla se
    # construye en runtime desde FUSION_MANIFEST (smoke3/4 ya validan ESE
    # indice horneado, no marcado de index.html); este smoke cubre la otra
    # mitad del contrato: que el HTML estatico tenga todos los ganchos que
    # el JS espera. Un id renombrado en la plantilla de HTML haria que
    # initPage() reviente con un elemento null en cuanto alguien la abra --
    # sin esto, nada lo detecta antes de proyectarlo en la sala.
    ok6 = True
    if js_path.exists() and html_path.exists():
        js_text = js_path.read_text(encoding="utf-8")
        html_text = html_path.read_text(encoding="utf-8")
        ids_js = set(re.findall(r'getElementById\("([^"]+)"\)', js_text))
        ids_html = set(re.findall(r'\bid="([^"]+)"', html_text))
        faltan = sorted(ids_js - ids_html)
        if faltan:
            failures.append(f"ids pedidos por fusionpad.js pero ausentes en index.html: {faltan}")
            ok6 = False
    print(f"smoke6 (todo getElementById de fusionpad.js existe en index.html): "
          f"{'OK' if ok6 else 'FALLO'}")

    # smoke 7: la rejilla que index.html hornea a mano (max=7 del slider de
    # n_bands, 8+9 <option> de los dos <datalist>) coincide con el tamano
    # real de la rejilla del manifiesto. index.html NO lee estos numeros de
    # FUSION_MANIFEST en runtime (son marcado estatico, a proposito, para
    # que el slider tenga limites correctos incluso antes de que cargue
    # fusionpad.js) -- si la rejilla del manifiesto cambiase de tamano algun
    # dia, esto falla en vez de dejar un slider que no llega a la ultima
    # banda o un datalist con muescas de mas.
    n_nb = len(manifest["meta"]["n_bands_grid"])
    n_mix = len(manifest["meta"]["color_mix_grid"])
    ok7 = n_nb == 8 and n_mix == 9
    if not ok7:
        failures.append(f"tamano de rejilla inesperado: n_bands_grid tiene {n_nb} valores "
                        f"(index.html asume 8, slider-nbands max=7), color_mix_grid tiene "
                        f"{n_mix} valores (index.html asume 9 <option> en el datalist) -- "
                        f"si esto cambia hay que actualizar _INDEX_HTML_TEMPLATE a mano")
    print(f"smoke7 (rejilla del manifiesto = 8 n_bands x 9 color_mix, lo que index.html asume "
          f"a mano): {'OK' if ok7 else 'FALLO'}")

    if failures:
        print(f"\n{len(failures)} fallos: {failures}")
        return 1
    print("\ntodos los smokes OK")
    return 0


# ======================================================================
# render (uso normal, sin --check)
# ======================================================================

def render(out_dir: Path) -> int:
    manifest = load_manifest(out_dir)
    pareja_index = build_pareja_index(manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(render_index_html(), encoding="utf-8")
    (out_dir / "fusionpad.js").write_text(render_fusionpad_js(pareja_index, manifest), encoding="utf-8")
    (out_dir / "LEEME.md").write_text(render_leeme(manifest, pareja_index), encoding="utf-8")

    print(f"escrito: {out_dir}/index.html")
    print(f"escrito: {out_dir}/fusionpad.js")
    print(f"escrito: {out_dir}/LEEME.md")
    print(f"({len(pareja_index)} parejas, {manifest['meta']['n_clips_total']} clips en el manifiesto)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="smokes rapidos, sin navegador")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    return check(args.out_dir) if args.check else render(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
