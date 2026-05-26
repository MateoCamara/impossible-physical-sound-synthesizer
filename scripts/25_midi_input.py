"""Toca el motor fisico con un teclado MIDI en vivo.

Mapea note_on -> {drip event | modal impact} parametrizado por la nota
y la velocity. Usa un cache pequeno de wavs pre-renderizados por
(modo, material, nota) para mantener latencia baja (~10 ms).

Instalacion del extra:
    pip install -e ".[midi]"

Uso:
    python scripts/25_midi_input.py --list-ports                       # lista puertos
    python scripts/25_midi_input.py --mode drip --port "MyKeyboard"    # toca drips
    python scripts/25_midi_input.py --mode modal --material glass      # campana
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from impossible_mix.config import SAMPLE_RATE
from impossible_mix.physics.droplet import (
    DropletParams,
    SURFACE_PROFILES,
    synth_drip_event,
)
from impossible_mix.physics.modal import PROFILES as MODAL_PROFILES, synth_modal_impact


def midi_note_to_radius_mm(note: int) -> float:
    """Mapea note MIDI a radio de gota. Note 48 (C3) = 1mm; +12 semitonos
    duplica el radio (gota mas grave). Rango usable ~36..72."""
    return max(0.3, 1.0 * (2 ** ((note - 48) / 12.0)))


def midi_note_to_modal_freq_factor(note: int) -> float:
    """Mapea note MIDI a factor de transposicion modal. Note 60 (C4) = 1.0;
    +12 semitonos = factor 2.0. Aplicado al perfil base via velocity."""
    return 2 ** ((note - 60) / 12.0)


def list_ports() -> list[str]:
    import mido
    return list(mido.get_input_names())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-ports", action="store_true",
                    help="Lista puertos MIDI disponibles y sale")
    ap.add_argument("--mode", choices=("drip", "modal"), default="drip",
                    help="Tipo de evento a disparar")
    ap.add_argument("--port", default=None,
                    help="Nombre del puerto MIDI (default: primero disponible)")
    ap.add_argument("--material", default=None,
                    help="Material: ceramic|glass|metal|rock|wood|fabric (drip), "
                         "metal|rock|wood|glass|earth|fabric (modal)")
    ap.add_argument("--viscosity", type=float, default=0.0,
                    help="Viscosidad para mode=drip")
    args = ap.parse_args()

    # Imports condicionales: solo si esta el extra [midi]
    try:
        import mido
        import sounddevice as sd
    except ImportError:
        print("ERROR: faltan dependencias del extra [midi]")
        print("Instalar:  pip install -e \".[midi]\"")
        return 1

    if args.list_ports:
        ports = list_ports()
        if not ports:
            print("No hay puertos MIDI disponibles. Conecta un teclado o "
                   "lanza un teclado virtual (e.g. VMPK / Midi Virtual Piano).")
        else:
            print("Puertos MIDI:")
            for p in ports:
                print(f"  {p}")
        return 0

    ports = list_ports()
    if not ports:
        print("ERROR: ningun puerto MIDI disponible. Use --list-ports.")
        return 1
    port_name = args.port if args.port else ports[0]
    if port_name not in ports:
        print(f"ERROR: puerto {port_name!r} no encontrado. Disponibles: {ports}")
        return 1

    # Default material por modo
    if args.material is None:
        args.material = "ceramic" if args.mode == "drip" else "metal"

    # Validar
    if args.mode == "drip" and args.material not in SURFACE_PROFILES:
        print(f"ERROR: surface profile {args.material!r} no existe. "
               f"Disponibles: {sorted(SURFACE_PROFILES)}")
        return 1
    if args.mode == "modal" and args.material not in MODAL_PROFILES:
        print(f"ERROR: modal profile {args.material!r} no existe. "
               f"Disponibles: {sorted(MODAL_PROFILES)}")
        return 1

    print(f"\nMode: {args.mode}  material: {args.material}  port: {port_name}")
    print(f"Listening for note_on... (Ctrl+C to exit)\n")

    # Cache de wavs por (note, velocity_bin)
    cache: dict[tuple[int, int], np.ndarray] = {}

    def synth_for_note(note: int, velocity: int) -> np.ndarray:
        # Cuantizar velocity en bins de 16 para reducir cache
        vbin = (velocity // 16) * 16
        key = (note, vbin)
        if key in cache:
            return cache[key]
        vel_factor = max(0.3, vbin / 127.0 * 2.0)
        if args.mode == "drip":
            p = DropletParams(
                droplet_radius_mm=midi_note_to_radius_mm(note),
                viscosity=args.viscosity,
                surface_profile=args.material,
                roll_velocity_hz=1, path_roughness=0,
                duration_s=0.6, seed=note,
            )
            wav = synth_drip_event(p, SAMPLE_RATE, velocity_factor=vel_factor)
        else:
            # modal: transponer el perfil base via factor en velocity
            # (la API actual no soporta override de freq, asi que escalamos
            # impact_strength y velocity para dar variacion)
            wav = synth_modal_impact(
                MODAL_PROFILES[args.material], SAMPLE_RATE,
                duration_s=1.5, impact_strength=0.9,
                velocity=vel_factor * midi_note_to_modal_freq_factor(note),
                excitation_shape="felt" if vbin < 64 else "wood",
            )
        cache[key] = wav
        return wav

    inport = mido.open_input(port_name)
    try:
        for msg in inport:
            if msg.type == "note_on" and msg.velocity > 0:
                wav = synth_for_note(msg.note, msg.velocity)
                # Reproducir asincrono (no bloquea siguiente nota)
                sd.play(wav, SAMPLE_RATE, blocking=False)
    except KeyboardInterrupt:
        print("\nBye.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
