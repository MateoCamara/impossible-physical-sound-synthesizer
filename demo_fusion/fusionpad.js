// fusionpad.js -- motor de audio + interfaz de la demo de fusion con
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
const FUSION_MANIFEST = {"meta": {"n_bands_grid": [2, 4, 6, 8, 12, 16, 24, 32], "color_mix_grid": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], "duracion_s": 6.0, "sr_audio": 48000}, "parejas": {"trueno_hecho_de_agua": {"id": "trueno_hecho_de_agua", "titulo": "Trueno hecho de agua", "env_parent": "trueno", "fine_parent": "goteo", "frase": "La dinámica (ritmo, ataques) la pone el trueno. La materia (timbre, textura fina) la pone el goteo.", "n_bands_heuristica": 6, "grid": {"2": {"0.000": {"fichero": "audio/trueno_hecho_de_agua/nb2__mix0.000.opus", "crest_db": 17.928086, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_agua/nb2__mix0.125.opus", "crest_db": 17.928055, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_agua/nb2__mix0.250.opus", "crest_db": 17.927997, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_agua/nb2__mix0.375.opus", "crest_db": 17.927889, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_agua/nb2__mix0.500.opus", "crest_db": 17.927682, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_agua/nb2__mix0.625.opus", "crest_db": 17.927273, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_agua/nb2__mix0.750.opus", "crest_db": 17.926421, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_agua/nb2__mix0.875.opus", "crest_db": 17.924514, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_agua/nb2__mix1.000.opus", "crest_db": 17.919868, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}}, "4": {"0.000": {"fichero": "audio/trueno_hecho_de_agua/nb4__mix0.000.opus", "crest_db": 18.357497, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_agua/nb4__mix0.125.opus", "crest_db": 18.442976, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_agua/nb4__mix0.250.opus", "crest_db": 18.508981, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_agua/nb4__mix0.375.opus", "crest_db": 18.558671, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_agua/nb4__mix0.500.opus", "crest_db": 18.592838, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_agua/nb4__mix0.625.opus", "crest_db": 18.613429, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_agua/nb4__mix0.750.opus", "crest_db": 18.623038, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_agua/nb4__mix0.875.opus", "crest_db": 18.623312, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_agua/nb4__mix1.000.opus", "crest_db": 18.614711, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}}, "6": {"0.000": {"fichero": "audio/trueno_hecho_de_agua/nb6__mix0.000.opus", "crest_db": 18.420487, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_agua/nb6__mix0.125.opus", "crest_db": 18.600989, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_agua/nb6__mix0.250.opus", "crest_db": 18.688399, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_agua/nb6__mix0.375.opus", "crest_db": 18.686868, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_agua/nb6__mix0.500.opus", "crest_db": 18.612463, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_agua/nb6__mix0.625.opus", "crest_db": 18.490264, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_agua/nb6__mix0.750.opus", "crest_db": 18.344561, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_agua/nb6__mix0.875.opus", "crest_db": 18.191602, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_agua/nb6__mix1.000.opus", "crest_db": 18.049447, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}}, "8": {"0.000": {"fichero": "audio/trueno_hecho_de_agua/nb8__mix0.000.opus", "crest_db": 18.408203, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_agua/nb8__mix0.125.opus", "crest_db": 18.520616, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_agua/nb8__mix0.250.opus", "crest_db": 18.563954, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_agua/nb8__mix0.375.opus", "crest_db": 18.520816, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_agua/nb8__mix0.500.opus", "crest_db": 18.384058, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_agua/nb8__mix0.625.opus", "crest_db": 18.159775, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_agua/nb8__mix0.750.opus", "crest_db": 17.865388, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_agua/nb8__mix0.875.opus", "crest_db": 17.516604, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_agua/nb8__mix1.000.opus", "crest_db": 17.129896, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}}, "12": {"0.000": {"fichero": "audio/trueno_hecho_de_agua/nb12__mix0.000.opus", "crest_db": 17.520542, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_agua/nb12__mix0.125.opus", "crest_db": 17.868332, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_agua/nb12__mix0.250.opus", "crest_db": 18.126398, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_agua/nb12__mix0.375.opus", "crest_db": 18.280595, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_agua/nb12__mix0.500.opus", "crest_db": 18.339248, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_agua/nb12__mix0.625.opus", "crest_db": 18.320626, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_agua/nb12__mix0.750.opus", "crest_db": 18.252055, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_agua/nb12__mix0.875.opus", "crest_db": 18.326174, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_agua/nb12__mix1.000.opus", "crest_db": 18.443314, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}}, "16": {"0.000": {"fichero": "audio/trueno_hecho_de_agua/nb16__mix0.000.opus", "crest_db": 18.209142, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_agua/nb16__mix0.125.opus", "crest_db": 18.626775, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_agua/nb16__mix0.250.opus", "crest_db": 18.964834, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_agua/nb16__mix0.375.opus", "crest_db": 19.206314, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_agua/nb16__mix0.500.opus", "crest_db": 19.349492, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_agua/nb16__mix0.625.opus", "crest_db": 19.409986, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_agua/nb16__mix0.750.opus", "crest_db": 19.41272, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_agua/nb16__mix0.875.opus", "crest_db": 19.382484, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_agua/nb16__mix1.000.opus", "crest_db": 19.337474, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}}, "24": {"0.000": {"fichero": "audio/trueno_hecho_de_agua/nb24__mix0.000.opus", "crest_db": 18.808132, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_agua/nb24__mix0.125.opus", "crest_db": 18.53516, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_agua/nb24__mix0.250.opus", "crest_db": 18.08033, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_agua/nb24__mix0.375.opus", "crest_db": 17.442512, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_agua/nb24__mix0.500.opus", "crest_db": 17.255046, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_agua/nb24__mix0.625.opus", "crest_db": 17.267762, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_agua/nb24__mix0.750.opus", "crest_db": 17.487437, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_agua/nb24__mix0.875.opus", "crest_db": 17.603584, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_agua/nb24__mix1.000.opus", "crest_db": 19.539745, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}}, "32": {"0.000": {"fichero": "audio/trueno_hecho_de_agua/nb32__mix0.000.opus", "crest_db": 19.393991, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_agua/nb32__mix0.125.opus", "crest_db": 19.191339, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_agua/nb32__mix0.250.opus", "crest_db": 18.855001, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_agua/nb32__mix0.375.opus", "crest_db": 19.110005, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_agua/nb32__mix0.500.opus", "crest_db": 19.232953, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_agua/nb32__mix0.625.opus", "crest_db": 19.239947, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_agua/nb32__mix0.750.opus", "crest_db": 19.154248, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_agua/nb32__mix0.875.opus", "crest_db": 18.989778, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_agua/nb32__mix1.000.opus", "crest_db": 22.426284, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false}}}, "referencias": {"baseline_v11": {"fichero": "audio/trueno_hecho_de_agua/ref__baseline_v11.opus", "crest_db": 47.458203, "sso": 0.369222, "metodo_igualacion": "soft_limit", "pico_recortado": true, "n_bands": 6}, "suma_ancla": {"fichero": "audio/trueno_hecho_de_agua/ref__suma_ancla.opus", "crest_db": 14.719993, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": null}, "sin_alinear": {"fichero": "audio/trueno_hecho_de_agua/ref__sin_alinear.opus", "crest_db": 19.358114, "sso": 0.369222, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 6}, "plana": {"fichero": "audio/trueno_hecho_de_agua/ref__plana.opus", "crest_db": 18.402643, "sso": 0.743855, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 6}}}, "fuego_hecho_de_vidrio": {"id": "fuego_hecho_de_vidrio", "titulo": "Fuego hecho de vidrio", "env_parent": "fuego", "fine_parent": "vidrio", "frase": "La dinámica (ritmo, ataques) la pone el fuego. La materia (timbre, textura fina) la pone el vidrio.", "n_bands_heuristica": 16, "grid": {"2": {"0.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb2__mix0.000.opus", "crest_db": 30.124801, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.125": {"fichero": "audio/fuego_hecho_de_vidrio/nb2__mix0.125.opus", "crest_db": 30.151184, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.250": {"fichero": "audio/fuego_hecho_de_vidrio/nb2__mix0.250.opus", "crest_db": 30.173759, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.375": {"fichero": "audio/fuego_hecho_de_vidrio/nb2__mix0.375.opus", "crest_db": 30.19307, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.500": {"fichero": "audio/fuego_hecho_de_vidrio/nb2__mix0.500.opus", "crest_db": 30.209586, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.625": {"fichero": "audio/fuego_hecho_de_vidrio/nb2__mix0.625.opus", "crest_db": 30.223713, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.750": {"fichero": "audio/fuego_hecho_de_vidrio/nb2__mix0.750.opus", "crest_db": 30.235798, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.875": {"fichero": "audio/fuego_hecho_de_vidrio/nb2__mix0.875.opus", "crest_db": 30.246139, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb2__mix1.000.opus", "crest_db": 30.254991, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}}, "4": {"0.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb4__mix0.000.opus", "crest_db": 30.94475, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.125": {"fichero": "audio/fuego_hecho_de_vidrio/nb4__mix0.125.opus", "crest_db": 30.795396, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.250": {"fichero": "audio/fuego_hecho_de_vidrio/nb4__mix0.250.opus", "crest_db": 30.611371, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.375": {"fichero": "audio/fuego_hecho_de_vidrio/nb4__mix0.375.opus", "crest_db": 30.387535, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.500": {"fichero": "audio/fuego_hecho_de_vidrio/nb4__mix0.500.opus", "crest_db": 30.119036, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.625": {"fichero": "audio/fuego_hecho_de_vidrio/nb4__mix0.625.opus", "crest_db": 29.856829, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.750": {"fichero": "audio/fuego_hecho_de_vidrio/nb4__mix0.750.opus", "crest_db": 29.857912, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.875": {"fichero": "audio/fuego_hecho_de_vidrio/nb4__mix0.875.opus", "crest_db": 29.830526, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb4__mix1.000.opus", "crest_db": 29.774436, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}}, "6": {"0.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb6__mix0.000.opus", "crest_db": 32.164041, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.125": {"fichero": "audio/fuego_hecho_de_vidrio/nb6__mix0.125.opus", "crest_db": 31.580016, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.250": {"fichero": "audio/fuego_hecho_de_vidrio/nb6__mix0.250.opus", "crest_db": 30.908577, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.375": {"fichero": "audio/fuego_hecho_de_vidrio/nb6__mix0.375.opus", "crest_db": 30.24966, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.500": {"fichero": "audio/fuego_hecho_de_vidrio/nb6__mix0.500.opus", "crest_db": 29.87504, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.625": {"fichero": "audio/fuego_hecho_de_vidrio/nb6__mix0.625.opus", "crest_db": 30.19575, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.750": {"fichero": "audio/fuego_hecho_de_vidrio/nb6__mix0.750.opus", "crest_db": 30.479339, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.875": {"fichero": "audio/fuego_hecho_de_vidrio/nb6__mix0.875.opus", "crest_db": 30.711516, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb6__mix1.000.opus", "crest_db": 30.89852, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}}, "8": {"0.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb8__mix0.000.opus", "crest_db": 32.368856, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.125": {"fichero": "audio/fuego_hecho_de_vidrio/nb8__mix0.125.opus", "crest_db": 32.040547, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.250": {"fichero": "audio/fuego_hecho_de_vidrio/nb8__mix0.250.opus", "crest_db": 31.637738, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.375": {"fichero": "audio/fuego_hecho_de_vidrio/nb8__mix0.375.opus", "crest_db": 31.146583, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.500": {"fichero": "audio/fuego_hecho_de_vidrio/nb8__mix0.500.opus", "crest_db": 30.82455, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.625": {"fichero": "audio/fuego_hecho_de_vidrio/nb8__mix0.625.opus", "crest_db": 30.580394, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.750": {"fichero": "audio/fuego_hecho_de_vidrio/nb8__mix0.750.opus", "crest_db": 30.279729, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.875": {"fichero": "audio/fuego_hecho_de_vidrio/nb8__mix0.875.opus", "crest_db": 29.928641, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb8__mix1.000.opus", "crest_db": 29.539232, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}}, "12": {"0.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb12__mix0.000.opus", "crest_db": 32.477456, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.125": {"fichero": "audio/fuego_hecho_de_vidrio/nb12__mix0.125.opus", "crest_db": 32.281824, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.250": {"fichero": "audio/fuego_hecho_de_vidrio/nb12__mix0.250.opus", "crest_db": 32.037683, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.375": {"fichero": "audio/fuego_hecho_de_vidrio/nb12__mix0.375.opus", "crest_db": 31.73623, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.500": {"fichero": "audio/fuego_hecho_de_vidrio/nb12__mix0.500.opus", "crest_db": 31.36951, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.625": {"fichero": "audio/fuego_hecho_de_vidrio/nb12__mix0.625.opus", "crest_db": 31.142325, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.750": {"fichero": "audio/fuego_hecho_de_vidrio/nb12__mix0.750.opus", "crest_db": 30.816508, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.875": {"fichero": "audio/fuego_hecho_de_vidrio/nb12__mix0.875.opus", "crest_db": 30.394824, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb12__mix1.000.opus", "crest_db": 30.57067, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}}, "16": {"0.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb16__mix0.000.opus", "crest_db": 32.06896, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.125": {"fichero": "audio/fuego_hecho_de_vidrio/nb16__mix0.125.opus", "crest_db": 31.675646, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.250": {"fichero": "audio/fuego_hecho_de_vidrio/nb16__mix0.250.opus", "crest_db": 31.215334, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.375": {"fichero": "audio/fuego_hecho_de_vidrio/nb16__mix0.375.opus", "crest_db": 30.683728, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.500": {"fichero": "audio/fuego_hecho_de_vidrio/nb16__mix0.500.opus", "crest_db": 30.182569, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.625": {"fichero": "audio/fuego_hecho_de_vidrio/nb16__mix0.625.opus", "crest_db": 29.880298, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.750": {"fichero": "audio/fuego_hecho_de_vidrio/nb16__mix0.750.opus", "crest_db": 30.193857, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.875": {"fichero": "audio/fuego_hecho_de_vidrio/nb16__mix0.875.opus", "crest_db": 30.606988, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb16__mix1.000.opus", "crest_db": 30.959484, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}}, "24": {"0.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb24__mix0.000.opus", "crest_db": 31.866091, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.125": {"fichero": "audio/fuego_hecho_de_vidrio/nb24__mix0.125.opus", "crest_db": 31.558653, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.250": {"fichero": "audio/fuego_hecho_de_vidrio/nb24__mix0.250.opus", "crest_db": 31.235258, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.375": {"fichero": "audio/fuego_hecho_de_vidrio/nb24__mix0.375.opus", "crest_db": 31.051688, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.500": {"fichero": "audio/fuego_hecho_de_vidrio/nb24__mix0.500.opus", "crest_db": 30.827706, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.625": {"fichero": "audio/fuego_hecho_de_vidrio/nb24__mix0.625.opus", "crest_db": 30.56272, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.750": {"fichero": "audio/fuego_hecho_de_vidrio/nb24__mix0.750.opus", "crest_db": 30.262136, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.875": {"fichero": "audio/fuego_hecho_de_vidrio/nb24__mix0.875.opus", "crest_db": 29.93694, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb24__mix1.000.opus", "crest_db": 29.601417, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}}, "32": {"0.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb32__mix0.000.opus", "crest_db": 32.374313, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.125": {"fichero": "audio/fuego_hecho_de_vidrio/nb32__mix0.125.opus", "crest_db": 32.14057, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.250": {"fichero": "audio/fuego_hecho_de_vidrio/nb32__mix0.250.opus", "crest_db": 31.857515, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.375": {"fichero": "audio/fuego_hecho_de_vidrio/nb32__mix0.375.opus", "crest_db": 31.518245, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.500": {"fichero": "audio/fuego_hecho_de_vidrio/nb32__mix0.500.opus", "crest_db": 31.119115, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.625": {"fichero": "audio/fuego_hecho_de_vidrio/nb32__mix0.625.opus", "crest_db": 30.662989, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.750": {"fichero": "audio/fuego_hecho_de_vidrio/nb32__mix0.750.opus", "crest_db": 30.161703, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "0.875": {"fichero": "audio/fuego_hecho_de_vidrio/nb32__mix0.875.opus", "crest_db": 29.636015, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/fuego_hecho_de_vidrio/nb32__mix1.000.opus", "crest_db": 29.112319, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true}}}, "referencias": {"baseline_v11": {"fichero": "audio/fuego_hecho_de_vidrio/ref__baseline_v11.opus", "crest_db": 29.521972, "sso": 0.746012, "metodo_igualacion": "linear_capped", "pico_recortado": true, "n_bands": 16}, "suma_ancla": {"fichero": "audio/fuego_hecho_de_vidrio/ref__suma_ancla.opus", "crest_db": 23.596146, "sso": 0.792658, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": null}, "sin_alinear": {"fichero": "audio/fuego_hecho_de_vidrio/ref__sin_alinear.opus", "crest_db": 30.627612, "sso": 0.746012, "metodo_igualacion": "linear_capped", "pico_recortado": true, "n_bands": 16}, "plana": {"fichero": "audio/fuego_hecho_de_vidrio/ref__plana.opus", "crest_db": 29.293521, "sso": 0.792658, "metodo_igualacion": "linear_capped", "pico_recortado": true, "n_bands": 16}}}, "canica_hecha_de_fuego": {"id": "canica_hecha_de_fuego", "titulo": "Canica hecha de fuego", "env_parent": "canica", "fine_parent": "fuego", "frase": "La dinámica (ritmo, ataques) la pone la canica. La materia (timbre, textura fina) la pone el fuego.", "n_bands_heuristica": 16, "grid": {"2": {"0.000": {"fichero": "audio/canica_hecha_de_fuego/nb2__mix0.000.opus", "crest_db": 15.345636, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/canica_hecha_de_fuego/nb2__mix0.125.opus", "crest_db": 15.334982, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/canica_hecha_de_fuego/nb2__mix0.250.opus", "crest_db": 15.32411, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/canica_hecha_de_fuego/nb2__mix0.375.opus", "crest_db": 15.313034, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/canica_hecha_de_fuego/nb2__mix0.500.opus", "crest_db": 15.301765, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/canica_hecha_de_fuego/nb2__mix0.625.opus", "crest_db": 15.290314, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/canica_hecha_de_fuego/nb2__mix0.750.opus", "crest_db": 15.278693, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/canica_hecha_de_fuego/nb2__mix0.875.opus", "crest_db": 15.266914, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/canica_hecha_de_fuego/nb2__mix1.000.opus", "crest_db": 15.254986, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}}, "4": {"0.000": {"fichero": "audio/canica_hecha_de_fuego/nb4__mix0.000.opus", "crest_db": 14.805498, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/canica_hecha_de_fuego/nb4__mix0.125.opus", "crest_db": 15.003637, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/canica_hecha_de_fuego/nb4__mix0.250.opus", "crest_db": 15.594044, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/canica_hecha_de_fuego/nb4__mix0.375.opus", "crest_db": 16.17424, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/canica_hecha_de_fuego/nb4__mix0.500.opus", "crest_db": 16.731488, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/canica_hecha_de_fuego/nb4__mix0.625.opus", "crest_db": 17.252911, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/canica_hecha_de_fuego/nb4__mix0.750.opus", "crest_db": 17.726705, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/canica_hecha_de_fuego/nb4__mix0.875.opus", "crest_db": 18.143446, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/canica_hecha_de_fuego/nb4__mix1.000.opus", "crest_db": 18.587083, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}}, "6": {"0.000": {"fichero": "audio/canica_hecha_de_fuego/nb6__mix0.000.opus", "crest_db": 16.485764, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/canica_hecha_de_fuego/nb6__mix0.125.opus", "crest_db": 17.912664, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/canica_hecha_de_fuego/nb6__mix0.250.opus", "crest_db": 19.36227, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/canica_hecha_de_fuego/nb6__mix0.375.opus", "crest_db": 20.801707, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/canica_hecha_de_fuego/nb6__mix0.500.opus", "crest_db": 22.192218, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/canica_hecha_de_fuego/nb6__mix0.625.opus", "crest_db": 23.490998, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/canica_hecha_de_fuego/nb6__mix0.750.opus", "crest_db": 24.655919, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/canica_hecha_de_fuego/nb6__mix0.875.opus", "crest_db": 25.676879, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/canica_hecha_de_fuego/nb6__mix1.000.opus", "crest_db": 26.524325, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}}, "8": {"0.000": {"fichero": "audio/canica_hecha_de_fuego/nb8__mix0.000.opus", "crest_db": 17.267414, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/canica_hecha_de_fuego/nb8__mix0.125.opus", "crest_db": 18.874709, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/canica_hecha_de_fuego/nb8__mix0.250.opus", "crest_db": 20.421338, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/canica_hecha_de_fuego/nb8__mix0.375.opus", "crest_db": 21.878349, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/canica_hecha_de_fuego/nb8__mix0.500.opus", "crest_db": 23.216456, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/canica_hecha_de_fuego/nb8__mix0.625.opus", "crest_db": 24.409727, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/canica_hecha_de_fuego/nb8__mix0.750.opus", "crest_db": 25.440137, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/canica_hecha_de_fuego/nb8__mix0.875.opus", "crest_db": 26.301268, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/canica_hecha_de_fuego/nb8__mix1.000.opus", "crest_db": 26.999308, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}}, "12": {"0.000": {"fichero": "audio/canica_hecha_de_fuego/nb12__mix0.000.opus", "crest_db": 17.540863, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/canica_hecha_de_fuego/nb12__mix0.125.opus", "crest_db": 18.962099, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/canica_hecha_de_fuego/nb12__mix0.250.opus", "crest_db": 20.309926, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/canica_hecha_de_fuego/nb12__mix0.375.opus", "crest_db": 21.556653, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/canica_hecha_de_fuego/nb12__mix0.500.opus", "crest_db": 22.6717, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/canica_hecha_de_fuego/nb12__mix0.625.opus", "crest_db": 23.802152, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/canica_hecha_de_fuego/nb12__mix0.750.opus", "crest_db": 24.858091, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/canica_hecha_de_fuego/nb12__mix0.875.opus", "crest_db": 25.732086, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/canica_hecha_de_fuego/nb12__mix1.000.opus", "crest_db": 26.429156, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}}, "16": {"0.000": {"fichero": "audio/canica_hecha_de_fuego/nb16__mix0.000.opus", "crest_db": 17.973874, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/canica_hecha_de_fuego/nb16__mix0.125.opus", "crest_db": 19.361064, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/canica_hecha_de_fuego/nb16__mix0.250.opus", "crest_db": 20.661308, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/canica_hecha_de_fuego/nb16__mix0.375.opus", "crest_db": 21.842709, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/canica_hecha_de_fuego/nb16__mix0.500.opus", "crest_db": 22.873312, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/canica_hecha_de_fuego/nb16__mix0.625.opus", "crest_db": 23.910099, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/canica_hecha_de_fuego/nb16__mix0.750.opus", "crest_db": 24.868342, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/canica_hecha_de_fuego/nb16__mix0.875.opus", "crest_db": 25.646683, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/canica_hecha_de_fuego/nb16__mix1.000.opus", "crest_db": 26.25997, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}}, "24": {"0.000": {"fichero": "audio/canica_hecha_de_fuego/nb24__mix0.000.opus", "crest_db": 18.341551, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/canica_hecha_de_fuego/nb24__mix0.125.opus", "crest_db": 20.080759, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/canica_hecha_de_fuego/nb24__mix0.250.opus", "crest_db": 21.712286, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/canica_hecha_de_fuego/nb24__mix0.375.opus", "crest_db": 23.260506, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/canica_hecha_de_fuego/nb24__mix0.500.opus", "crest_db": 24.690651, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/canica_hecha_de_fuego/nb24__mix0.625.opus", "crest_db": 25.908486, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/canica_hecha_de_fuego/nb24__mix0.750.opus", "crest_db": 26.906563, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/canica_hecha_de_fuego/nb24__mix0.875.opus", "crest_db": 27.696014, "sso": 0.850496, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/canica_hecha_de_fuego/nb24__mix1.000.opus", "crest_db": 28.3025, "sso": 0.850496, "metodo_igualacion": "linear_capped", "pico_recortado": true}}, "32": {"0.000": {"fichero": "audio/canica_hecha_de_fuego/nb32__mix0.000.opus", "crest_db": 19.16131, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/canica_hecha_de_fuego/nb32__mix0.125.opus", "crest_db": 20.683368, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/canica_hecha_de_fuego/nb32__mix0.250.opus", "crest_db": 22.287035, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/canica_hecha_de_fuego/nb32__mix0.375.opus", "crest_db": 23.748114, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/canica_hecha_de_fuego/nb32__mix0.500.opus", "crest_db": 25.037471, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/canica_hecha_de_fuego/nb32__mix0.625.opus", "crest_db": 26.131832, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/canica_hecha_de_fuego/nb32__mix0.750.opus", "crest_db": 27.019147, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/canica_hecha_de_fuego/nb32__mix0.875.opus", "crest_db": 27.701555, "sso": 0.850496, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/canica_hecha_de_fuego/nb32__mix1.000.opus", "crest_db": 28.194358, "sso": 0.850496, "metodo_igualacion": "linear_capped", "pico_recortado": true}}}, "referencias": {"baseline_v11": {"fichero": "audio/canica_hecha_de_fuego/ref__baseline_v11.opus", "crest_db": 27.017958, "sso": 0.847642, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 16}, "suma_ancla": {"fichero": "audio/canica_hecha_de_fuego/ref__suma_ancla.opus", "crest_db": 19.420179, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": null}, "sin_alinear": {"fichero": "audio/canica_hecha_de_fuego/ref__sin_alinear.opus", "crest_db": 24.71219, "sso": 0.847642, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 16}, "plana": {"fichero": "audio/canica_hecha_de_fuego/ref__plana.opus", "crest_db": 15.791149, "sso": 0.850496, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 16}}}, "trueno_hecho_de_canica": {"id": "trueno_hecho_de_canica", "titulo": "Trueno hecho de canica", "env_parent": "trueno", "fine_parent": "canica", "frase": "La dinámica (ritmo, ataques) la pone el trueno. La materia (timbre, textura fina) la pone la canica.", "n_bands_heuristica": 6, "grid": {"2": {"0.000": {"fichero": "audio/trueno_hecho_de_canica/nb2__mix0.000.opus", "crest_db": 18.942168, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_canica/nb2__mix0.125.opus", "crest_db": 18.942532, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_canica/nb2__mix0.250.opus", "crest_db": 18.943362, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_canica/nb2__mix0.375.opus", "crest_db": 18.945241, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_canica/nb2__mix0.500.opus", "crest_db": 18.949415, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_canica/nb2__mix0.625.opus", "crest_db": 18.958264, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_canica/nb2__mix0.750.opus", "crest_db": 18.974743, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_canica/nb2__mix0.875.opus", "crest_db": 18.992751, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_canica/nb2__mix1.000.opus", "crest_db": 21.400154, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}}, "4": {"0.000": {"fichero": "audio/trueno_hecho_de_canica/nb4__mix0.000.opus", "crest_db": 19.457053, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_canica/nb4__mix0.125.opus", "crest_db": 19.512737, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_canica/nb4__mix0.250.opus", "crest_db": 19.502298, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_canica/nb4__mix0.375.opus", "crest_db": 19.423202, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_canica/nb4__mix0.500.opus", "crest_db": 19.282558, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_canica/nb4__mix0.625.opus", "crest_db": 19.094227, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_canica/nb4__mix0.750.opus", "crest_db": 18.873355, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_canica/nb4__mix0.875.opus", "crest_db": 23.011776, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_canica/nb4__mix1.000.opus", "crest_db": 31.393959, "sso": 0.376244, "metodo_igualacion": "linear_capped", "pico_recortado": true}}, "6": {"0.000": {"fichero": "audio/trueno_hecho_de_canica/nb6__mix0.000.opus", "crest_db": 19.174599, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_canica/nb6__mix0.125.opus", "crest_db": 19.227198, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_canica/nb6__mix0.250.opus", "crest_db": 19.221752, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_canica/nb6__mix0.375.opus", "crest_db": 19.133235, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_canica/nb6__mix0.500.opus", "crest_db": 18.933488, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_canica/nb6__mix0.625.opus", "crest_db": 18.589038, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_canica/nb6__mix0.750.opus", "crest_db": 18.073896, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_canica/nb6__mix0.875.opus", "crest_db": 20.471861, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_canica/nb6__mix1.000.opus", "crest_db": 27.337469, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}}, "8": {"0.000": {"fichero": "audio/trueno_hecho_de_canica/nb8__mix0.000.opus", "crest_db": 17.377107, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_canica/nb8__mix0.125.opus", "crest_db": 17.277728, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_canica/nb8__mix0.250.opus", "crest_db": 17.061104, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_canica/nb8__mix0.375.opus", "crest_db": 16.675303, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_canica/nb8__mix0.500.opus", "crest_db": 16.205457, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_canica/nb8__mix0.625.opus", "crest_db": 16.846526, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_canica/nb8__mix0.750.opus", "crest_db": 17.408061, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_canica/nb8__mix0.875.opus", "crest_db": 25.949671, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/trueno_hecho_de_canica/nb8__mix1.000.opus", "crest_db": 33.37824, "sso": 0.376244, "metodo_igualacion": "linear_capped", "pico_recortado": true}}, "12": {"0.000": {"fichero": "audio/trueno_hecho_de_canica/nb12__mix0.000.opus", "crest_db": 17.944726, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_canica/nb12__mix0.125.opus", "crest_db": 18.102069, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_canica/nb12__mix0.250.opus", "crest_db": 18.186703, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_canica/nb12__mix0.375.opus", "crest_db": 18.156438, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_canica/nb12__mix0.500.opus", "crest_db": 17.939393, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_canica/nb12__mix0.625.opus", "crest_db": 18.479856, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_canica/nb12__mix0.750.opus", "crest_db": 26.778969, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_canica/nb12__mix0.875.opus", "crest_db": 34.070948, "sso": 0.376244, "metodo_igualacion": "soft_limit", "pico_recortado": true}, "1.000": {"fichero": "audio/trueno_hecho_de_canica/nb12__mix1.000.opus", "crest_db": 39.745141, "sso": 0.376244, "metodo_igualacion": "soft_limit", "pico_recortado": true}}, "16": {"0.000": {"fichero": "audio/trueno_hecho_de_canica/nb16__mix0.000.opus", "crest_db": 18.124887, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_canica/nb16__mix0.125.opus", "crest_db": 18.152288, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_canica/nb16__mix0.250.opus", "crest_db": 18.085855, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_canica/nb16__mix0.375.opus", "crest_db": 17.907732, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_canica/nb16__mix0.500.opus", "crest_db": 18.210461, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_canica/nb16__mix0.625.opus", "crest_db": 20.89987, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_canica/nb16__mix0.750.opus", "crest_db": 27.20657, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_canica/nb16__mix0.875.opus", "crest_db": 33.049018, "sso": 0.376244, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/trueno_hecho_de_canica/nb16__mix1.000.opus", "crest_db": 37.413102, "sso": 0.376244, "metodo_igualacion": "soft_limit", "pico_recortado": true}}, "24": {"0.000": {"fichero": "audio/trueno_hecho_de_canica/nb24__mix0.000.opus", "crest_db": 18.782036, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_canica/nb24__mix0.125.opus", "crest_db": 18.809843, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_canica/nb24__mix0.250.opus", "crest_db": 18.703003, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_canica/nb24__mix0.375.opus", "crest_db": 18.438143, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_canica/nb24__mix0.500.opus", "crest_db": 18.275362, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_canica/nb24__mix0.625.opus", "crest_db": 18.233146, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_canica/nb24__mix0.750.opus", "crest_db": 24.908667, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_canica/nb24__mix0.875.opus", "crest_db": 31.751143, "sso": 0.376244, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/trueno_hecho_de_canica/nb24__mix1.000.opus", "crest_db": 36.783508, "sso": 0.376244, "metodo_igualacion": "soft_limit", "pico_recortado": true}}, "32": {"0.000": {"fichero": "audio/trueno_hecho_de_canica/nb32__mix0.000.opus", "crest_db": 19.713524, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/trueno_hecho_de_canica/nb32__mix0.125.opus", "crest_db": 19.811035, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/trueno_hecho_de_canica/nb32__mix0.250.opus", "crest_db": 19.817245, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/trueno_hecho_de_canica/nb32__mix0.375.opus", "crest_db": 19.715395, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/trueno_hecho_de_canica/nb32__mix0.500.opus", "crest_db": 19.4598, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/trueno_hecho_de_canica/nb32__mix0.625.opus", "crest_db": 19.251842, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/trueno_hecho_de_canica/nb32__mix0.750.opus", "crest_db": 26.171461, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/trueno_hecho_de_canica/nb32__mix0.875.opus", "crest_db": 32.56082, "sso": 0.376244, "metodo_igualacion": "linear_capped", "pico_recortado": true}, "1.000": {"fichero": "audio/trueno_hecho_de_canica/nb32__mix1.000.opus", "crest_db": 37.24645, "sso": 0.376244, "metodo_igualacion": "soft_limit", "pico_recortado": true}}}, "referencias": {"baseline_v11": {"fichero": "audio/trueno_hecho_de_canica/ref__baseline_v11.opus", "crest_db": 47.285335, "sso": 0.084442, "metodo_igualacion": "soft_limit", "pico_recortado": true, "n_bands": 6}, "suma_ancla": {"fichero": "audio/trueno_hecho_de_canica/ref__suma_ancla.opus", "crest_db": 16.44836, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": null}, "sin_alinear": {"fichero": "audio/trueno_hecho_de_canica/ref__sin_alinear.opus", "crest_db": 17.98175, "sso": 0.084442, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 6}, "plana": {"fichero": "audio/trueno_hecho_de_canica/ref__plana.opus", "crest_db": 19.187141, "sso": 0.376244, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 6}}}, "oceano_hecho_de_campana": {"id": "oceano_hecho_de_campana", "titulo": "Oceano hecho de campana", "env_parent": "oceano", "fine_parent": "campana_tela", "frase": "La dinámica (ritmo, ataques) la pone el océano. La materia (timbre, textura fina) la pone la campana de tela.", "n_bands_heuristica": 16, "grid": {"2": {"0.000": {"fichero": "audio/oceano_hecho_de_campana/nb2__mix0.000.opus", "crest_db": 17.661402, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/oceano_hecho_de_campana/nb2__mix0.125.opus", "crest_db": 17.967466, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/oceano_hecho_de_campana/nb2__mix0.250.opus", "crest_db": 18.216306, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/oceano_hecho_de_campana/nb2__mix0.375.opus", "crest_db": 18.416572, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/oceano_hecho_de_campana/nb2__mix0.500.opus", "crest_db": 18.576564, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/oceano_hecho_de_campana/nb2__mix0.625.opus", "crest_db": 18.703724, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/oceano_hecho_de_campana/nb2__mix0.750.opus", "crest_db": 18.804437, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/oceano_hecho_de_campana/nb2__mix0.875.opus", "crest_db": 18.884018, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/oceano_hecho_de_campana/nb2__mix1.000.opus", "crest_db": 18.946807, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}}, "4": {"0.000": {"fichero": "audio/oceano_hecho_de_campana/nb4__mix0.000.opus", "crest_db": 17.234093, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/oceano_hecho_de_campana/nb4__mix0.125.opus", "crest_db": 17.510843, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/oceano_hecho_de_campana/nb4__mix0.250.opus", "crest_db": 17.727515, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/oceano_hecho_de_campana/nb4__mix0.375.opus", "crest_db": 17.892448, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/oceano_hecho_de_campana/nb4__mix0.500.opus", "crest_db": 18.013527, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/oceano_hecho_de_campana/nb4__mix0.625.opus", "crest_db": 18.097865, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/oceano_hecho_de_campana/nb4__mix0.750.opus", "crest_db": 18.15174, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/oceano_hecho_de_campana/nb4__mix0.875.opus", "crest_db": 18.180631, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/oceano_hecho_de_campana/nb4__mix1.000.opus", "crest_db": 18.189291, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}}, "6": {"0.000": {"fichero": "audio/oceano_hecho_de_campana/nb6__mix0.000.opus", "crest_db": 17.055613, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/oceano_hecho_de_campana/nb6__mix0.125.opus", "crest_db": 17.160085, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/oceano_hecho_de_campana/nb6__mix0.250.opus", "crest_db": 17.203, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/oceano_hecho_de_campana/nb6__mix0.375.opus", "crest_db": 17.191331, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/oceano_hecho_de_campana/nb6__mix0.500.opus", "crest_db": 17.131458, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/oceano_hecho_de_campana/nb6__mix0.625.opus", "crest_db": 17.208515, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/oceano_hecho_de_campana/nb6__mix0.750.opus", "crest_db": 17.303396, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/oceano_hecho_de_campana/nb6__mix0.875.opus", "crest_db": 17.380643, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/oceano_hecho_de_campana/nb6__mix1.000.opus", "crest_db": 17.440054, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}}, "8": {"0.000": {"fichero": "audio/oceano_hecho_de_campana/nb8__mix0.000.opus", "crest_db": 17.092821, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/oceano_hecho_de_campana/nb8__mix0.125.opus", "crest_db": 17.182025, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/oceano_hecho_de_campana/nb8__mix0.250.opus", "crest_db": 17.196062, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/oceano_hecho_de_campana/nb8__mix0.375.opus", "crest_db": 17.142802, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/oceano_hecho_de_campana/nb8__mix0.500.opus", "crest_db": 17.029126, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/oceano_hecho_de_campana/nb8__mix0.625.opus", "crest_db": 16.861105, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/oceano_hecho_de_campana/nb8__mix0.750.opus", "crest_db": 16.644468, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/oceano_hecho_de_campana/nb8__mix0.875.opus", "crest_db": 16.632737, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/oceano_hecho_de_campana/nb8__mix1.000.opus", "crest_db": 16.752981, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}}, "12": {"0.000": {"fichero": "audio/oceano_hecho_de_campana/nb12__mix0.000.opus", "crest_db": 17.003352, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/oceano_hecho_de_campana/nb12__mix0.125.opus", "crest_db": 17.277884, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/oceano_hecho_de_campana/nb12__mix0.250.opus", "crest_db": 17.492762, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/oceano_hecho_de_campana/nb12__mix0.375.opus", "crest_db": 17.646345, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/oceano_hecho_de_campana/nb12__mix0.500.opus", "crest_db": 17.730349, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/oceano_hecho_de_campana/nb12__mix0.625.opus", "crest_db": 17.743347, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/oceano_hecho_de_campana/nb12__mix0.750.opus", "crest_db": 17.685731, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/oceano_hecho_de_campana/nb12__mix0.875.opus", "crest_db": 18.011889, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/oceano_hecho_de_campana/nb12__mix1.000.opus", "crest_db": 18.319624, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}}, "16": {"0.000": {"fichero": "audio/oceano_hecho_de_campana/nb16__mix0.000.opus", "crest_db": 17.898293, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/oceano_hecho_de_campana/nb16__mix0.125.opus", "crest_db": 17.842678, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/oceano_hecho_de_campana/nb16__mix0.250.opus", "crest_db": 17.699755, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/oceano_hecho_de_campana/nb16__mix0.375.opus", "crest_db": 17.48135, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/oceano_hecho_de_campana/nb16__mix0.500.opus", "crest_db": 17.64766, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/oceano_hecho_de_campana/nb16__mix0.625.opus", "crest_db": 17.851499, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/oceano_hecho_de_campana/nb16__mix0.750.opus", "crest_db": 17.947405, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/oceano_hecho_de_campana/nb16__mix0.875.opus", "crest_db": 17.947267, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/oceano_hecho_de_campana/nb16__mix1.000.opus", "crest_db": 17.874509, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}}, "24": {"0.000": {"fichero": "audio/oceano_hecho_de_campana/nb24__mix0.000.opus", "crest_db": 18.026112, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/oceano_hecho_de_campana/nb24__mix0.125.opus", "crest_db": 18.139434, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/oceano_hecho_de_campana/nb24__mix0.250.opus", "crest_db": 18.225819, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/oceano_hecho_de_campana/nb24__mix0.375.opus", "crest_db": 18.322172, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/oceano_hecho_de_campana/nb24__mix0.500.opus", "crest_db": 18.420852, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/oceano_hecho_de_campana/nb24__mix0.625.opus", "crest_db": 18.529417, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/oceano_hecho_de_campana/nb24__mix0.750.opus", "crest_db": 18.560804, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/oceano_hecho_de_campana/nb24__mix0.875.opus", "crest_db": 18.516875, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/oceano_hecho_de_campana/nb24__mix1.000.opus", "crest_db": 18.404423, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}}, "32": {"0.000": {"fichero": "audio/oceano_hecho_de_campana/nb32__mix0.000.opus", "crest_db": 18.781945, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/oceano_hecho_de_campana/nb32__mix0.125.opus", "crest_db": 18.788293, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/oceano_hecho_de_campana/nb32__mix0.250.opus", "crest_db": 18.874014, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/oceano_hecho_de_campana/nb32__mix0.375.opus", "crest_db": 18.93234, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/oceano_hecho_de_campana/nb32__mix0.500.opus", "crest_db": 18.919786, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/oceano_hecho_de_campana/nb32__mix0.625.opus", "crest_db": 18.898787, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/oceano_hecho_de_campana/nb32__mix0.750.opus", "crest_db": 19.089143, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/oceano_hecho_de_campana/nb32__mix0.875.opus", "crest_db": 19.210061, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/oceano_hecho_de_campana/nb32__mix1.000.opus", "crest_db": 19.27565, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false}}}, "referencias": {"baseline_v11": {"fichero": "audio/oceano_hecho_de_campana/ref__baseline_v11.opus", "crest_db": 19.469162, "sso": 0.709075, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 16}, "suma_ancla": {"fichero": "audio/oceano_hecho_de_campana/ref__suma_ancla.opus", "crest_db": 20.192104, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": null}, "sin_alinear": {"fichero": "audio/oceano_hecho_de_campana/ref__sin_alinear.opus", "crest_db": 19.739092, "sso": 0.709075, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 16}, "plana": {"fichero": "audio/oceano_hecho_de_campana/ref__plana.opus", "crest_db": 17.264022, "sso": 0.54929, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 16}}}, "goteo_hecho_de_campana": {"id": "goteo_hecho_de_campana", "titulo": "Goteo hecho de campana", "env_parent": "goteo", "fine_parent": "campana_tela", "frase": "La dinámica (ritmo, ataques) la pone el goteo. La materia (timbre, textura fina) la pone la campana de tela.", "n_bands_heuristica": 6, "grid": {"2": {"0.000": {"fichero": "audio/goteo_hecho_de_campana/nb2__mix0.000.opus", "crest_db": 21.476301, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/goteo_hecho_de_campana/nb2__mix0.125.opus", "crest_db": 21.470933, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/goteo_hecho_de_campana/nb2__mix0.250.opus", "crest_db": 21.465561, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/goteo_hecho_de_campana/nb2__mix0.375.opus", "crest_db": 21.460187, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/goteo_hecho_de_campana/nb2__mix0.500.opus", "crest_db": 21.45481, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/goteo_hecho_de_campana/nb2__mix0.625.opus", "crest_db": 21.449431, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/goteo_hecho_de_campana/nb2__mix0.750.opus", "crest_db": 21.44405, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/goteo_hecho_de_campana/nb2__mix0.875.opus", "crest_db": 21.438666, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/goteo_hecho_de_campana/nb2__mix1.000.opus", "crest_db": 21.433281, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}}, "4": {"0.000": {"fichero": "audio/goteo_hecho_de_campana/nb4__mix0.000.opus", "crest_db": 22.410108, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/goteo_hecho_de_campana/nb4__mix0.125.opus", "crest_db": 22.54146, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/goteo_hecho_de_campana/nb4__mix0.250.opus", "crest_db": 22.675896, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/goteo_hecho_de_campana/nb4__mix0.375.opus", "crest_db": 22.813254, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/goteo_hecho_de_campana/nb4__mix0.500.opus", "crest_db": 22.953346, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/goteo_hecho_de_campana/nb4__mix0.625.opus", "crest_db": 23.095959, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/goteo_hecho_de_campana/nb4__mix0.750.opus", "crest_db": 23.240851, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/goteo_hecho_de_campana/nb4__mix0.875.opus", "crest_db": 23.387755, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/goteo_hecho_de_campana/nb4__mix1.000.opus", "crest_db": 23.536379, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}}, "6": {"0.000": {"fichero": "audio/goteo_hecho_de_campana/nb6__mix0.000.opus", "crest_db": 23.600028, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/goteo_hecho_de_campana/nb6__mix0.125.opus", "crest_db": 23.9359, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/goteo_hecho_de_campana/nb6__mix0.250.opus", "crest_db": 24.264036, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/goteo_hecho_de_campana/nb6__mix0.375.opus", "crest_db": 24.58484, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/goteo_hecho_de_campana/nb6__mix0.500.opus", "crest_db": 24.898601, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/goteo_hecho_de_campana/nb6__mix0.625.opus", "crest_db": 25.205483, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/goteo_hecho_de_campana/nb6__mix0.750.opus", "crest_db": 25.505531, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/goteo_hecho_de_campana/nb6__mix0.875.opus", "crest_db": 25.79868, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/goteo_hecho_de_campana/nb6__mix1.000.opus", "crest_db": 26.084767, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}}, "8": {"0.000": {"fichero": "audio/goteo_hecho_de_campana/nb8__mix0.000.opus", "crest_db": 22.929389, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/goteo_hecho_de_campana/nb8__mix0.125.opus", "crest_db": 23.389671, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/goteo_hecho_de_campana/nb8__mix0.250.opus", "crest_db": 23.784536, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/goteo_hecho_de_campana/nb8__mix0.375.opus", "crest_db": 24.12713, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/goteo_hecho_de_campana/nb8__mix0.500.opus", "crest_db": 24.428879, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/goteo_hecho_de_campana/nb8__mix0.625.opus", "crest_db": 24.699017, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/goteo_hecho_de_campana/nb8__mix0.750.opus", "crest_db": 24.944685, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/goteo_hecho_de_campana/nb8__mix0.875.opus", "crest_db": 25.171254, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/goteo_hecho_de_campana/nb8__mix1.000.opus", "crest_db": 25.382706, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}}, "12": {"0.000": {"fichero": "audio/goteo_hecho_de_campana/nb12__mix0.000.opus", "crest_db": 23.402391, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/goteo_hecho_de_campana/nb12__mix0.125.opus", "crest_db": 23.944246, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/goteo_hecho_de_campana/nb12__mix0.250.opus", "crest_db": 24.403207, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/goteo_hecho_de_campana/nb12__mix0.375.opus", "crest_db": 24.794018, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/goteo_hecho_de_campana/nb12__mix0.500.opus", "crest_db": 25.125422, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/goteo_hecho_de_campana/nb12__mix0.625.opus", "crest_db": 25.400677, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/goteo_hecho_de_campana/nb12__mix0.750.opus", "crest_db": 25.630763, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/goteo_hecho_de_campana/nb12__mix0.875.opus", "crest_db": 25.824822, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/goteo_hecho_de_campana/nb12__mix1.000.opus", "crest_db": 25.989706, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}}, "16": {"0.000": {"fichero": "audio/goteo_hecho_de_campana/nb16__mix0.000.opus", "crest_db": 23.681468, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/goteo_hecho_de_campana/nb16__mix0.125.opus", "crest_db": 24.403279, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/goteo_hecho_de_campana/nb16__mix0.250.opus", "crest_db": 25.026326, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/goteo_hecho_de_campana/nb16__mix0.375.opus", "crest_db": 25.558661, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/goteo_hecho_de_campana/nb16__mix0.500.opus", "crest_db": 26.014479, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/goteo_hecho_de_campana/nb16__mix0.625.opus", "crest_db": 26.409194, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/goteo_hecho_de_campana/nb16__mix0.750.opus", "crest_db": 26.75623, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/goteo_hecho_de_campana/nb16__mix0.875.opus", "crest_db": 27.065658, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/goteo_hecho_de_campana/nb16__mix1.000.opus", "crest_db": 27.344115, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}}, "24": {"0.000": {"fichero": "audio/goteo_hecho_de_campana/nb24__mix0.000.opus", "crest_db": 22.396594, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/goteo_hecho_de_campana/nb24__mix0.125.opus", "crest_db": 22.978849, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/goteo_hecho_de_campana/nb24__mix0.250.opus", "crest_db": 23.50102, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/goteo_hecho_de_campana/nb24__mix0.375.opus", "crest_db": 23.966711, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/goteo_hecho_de_campana/nb24__mix0.500.opus", "crest_db": 24.607709, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/goteo_hecho_de_campana/nb24__mix0.625.opus", "crest_db": 25.403893, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/goteo_hecho_de_campana/nb24__mix0.750.opus", "crest_db": 26.145592, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/goteo_hecho_de_campana/nb24__mix0.875.opus", "crest_db": 26.822233, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/goteo_hecho_de_campana/nb24__mix1.000.opus", "crest_db": 27.424324, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}}, "32": {"0.000": {"fichero": "audio/goteo_hecho_de_campana/nb32__mix0.000.opus", "crest_db": 22.617241, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.125": {"fichero": "audio/goteo_hecho_de_campana/nb32__mix0.125.opus", "crest_db": 23.384767, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.250": {"fichero": "audio/goteo_hecho_de_campana/nb32__mix0.250.opus", "crest_db": 24.072873, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.375": {"fichero": "audio/goteo_hecho_de_campana/nb32__mix0.375.opus", "crest_db": 24.666237, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.500": {"fichero": "audio/goteo_hecho_de_campana/nb32__mix0.500.opus", "crest_db": 25.160431, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.625": {"fichero": "audio/goteo_hecho_de_campana/nb32__mix0.625.opus", "crest_db": 25.562545, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.750": {"fichero": "audio/goteo_hecho_de_campana/nb32__mix0.750.opus", "crest_db": 25.887856, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "0.875": {"fichero": "audio/goteo_hecho_de_campana/nb32__mix0.875.opus", "crest_db": 26.200267, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}, "1.000": {"fichero": "audio/goteo_hecho_de_campana/nb32__mix1.000.opus", "crest_db": 26.5216, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false}}}, "referencias": {"baseline_v11": {"fichero": "audio/goteo_hecho_de_campana/ref__baseline_v11.opus", "crest_db": 28.033032, "sso": 0.650676, "metodo_igualacion": "linear_capped", "pico_recortado": true, "n_bands": 6}, "suma_ancla": {"fichero": "audio/goteo_hecho_de_campana/ref__suma_ancla.opus", "crest_db": 23.289602, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": null}, "sin_alinear": {"fichero": "audio/goteo_hecho_de_campana/ref__sin_alinear.opus", "crest_db": 26.099664, "sso": 0.650676, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 6}, "plana": {"fichero": "audio/goteo_hecho_de_campana/ref__plana.opus", "crest_db": 21.984371, "sso": 0.82002, "metodo_igualacion": "linear", "pico_recortado": false, "n_bands": 6}}}}};
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
