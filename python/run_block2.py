#!/usr/bin/env python
"""Block 2, quick test: horizontal or vertical stripes at every spot of a Block 1 map,
static first, then 20 and 38 Hz. See specvis/procedure/block2.py.

    python\\specvis.bat block2 --participant P01 --scale "Lab screen" --tracker
    python/specvis.sh block2 --no-map --windowed          (a try-out on the Mac, no map)

--participant (as typed in Block 1) finds that person's latest finished Block 1 run;
--eye L/R picks the eye's run; --block1 <folder> picks one run by hand.

--block1 is the folder of a finished Block 1 run (its summary.csv and session.json): each
spot gets its own threshold contrast, "not seen" spots and the blind spot the highest.
The eye, screen, distance and background are taken from that run. With --tracker,
Block 1's setup runs first (head position, calibration, accuracy check; no perimetry).

Saves to ``python/output/block2/<participant>_<eye>_<date>_<time>/``: trials.csv (every
trial), summary.csv (per spot and condition: shown, seen, right, wrong), overall.csv
(true positives, false presses on catch trials) and session.json. Set the screen to its highest refresh rate first (Windows:
Settings > Display > Advanced display); the rates shown follow from it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from specvis.procedure.block2 import (Block2Settings, results_text, run_block2,  # noqa: E402
                                      write_outputs)
from specvis.procedure.fixation import FIX_MOUSE, FIX_NONE, FIX_TRACKER  # noqa: E402
from specvis.procedure.perimetry import background_for, default_calibration  # noqa: E402
from specvis.procedure.session import SessionError  # noqa: E402

OUTPUT = HERE / "output" / "block2"
PERIMETRY = HERE / "output" / "perimetry"


def latest_block1(participant: str, eye=None, root: pathlib.Path = PERIMETRY):
    """This participant's latest Block 1 run that finished (not stopped early), for
    ``eye`` if given. Folders are <participant>_<eye>_<date>_<time>."""
    import json
    best = None
    for eye_ in ([eye] if eye else ["L", "R"]):
        for d in root.glob(f"{participant}_{eye_}_*"):
            rest = d.name[len(participant) + 3:]
            if not (len(rest) == 15 and rest[8] == "_" and rest.replace("_", "").isdigit()):
                continue
            try:
                meta = json.loads((d / "session.json").read_text())
            except (OSError, ValueError):
                continue
            if meta.get("aborted") or not (d / "summary.csv").exists():
                continue
            if best is None or rest > best[0]:
                best = (rest, d)
    return best[1] if best else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default="pilot")
    ap.add_argument("--eye", choices=("L", "R"), default=None,
                    help="the tested eye; cover the other one (default: from Block 1, else R)")
    ap.add_argument("--distance-mm", type=float, default=570.0)
    ap.add_argument("--screen-width-mm", type=float, default=542.3,
                    help="the lab BenQ: 542.3 (MacBook 14\": 302)")
    ap.add_argument("--screen-height-mm", type=float, default=305.0,
                    help="the lab BenQ: 305 (0 = from the width and the pixels)")
    ap.add_argument("--scale", default="",
                    help='the ColorCAL scale to use, e.g. "Lab screen" (none: an assumed screen)')
    ap.add_argument("--flat-field", default="",
                    help="a flat-field (screen evenness) map's name or id, from "
                         "specvis flatfield (needs --scale); default: none")
    ap.add_argument("--block1", default="",
                    help="folder of the Block 1 run whose map to use (default: this "
                         "participant's latest finished Block 1 run, for --eye if given)")
    ap.add_argument("--no-map", action="store_true",
                    help="no Block 1 map: 4 quadrants + blind spot (try-outs only)")
    ap.add_argument("--train-s", type=float, default=2.0, help="seconds each patch is up")
    ap.add_argument("--no-blind-spot", action="store_true")
    ap.add_argument("--forced-choice", action="store_true",
                    help="a guess on every trial (for blindsight); no answer = shown again")
    ap.add_argument("--rating", action="store_true",
                    help="after each trial: 1 = saw nothing, 2 = something, 3 = clearly")
    ap.add_argument("--no-setup", action="store_true",
                    help="skip the eye-tracker calibration")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--tracker", action="store_true", help="record gaze with the Tobii")
    group.add_argument("--mouse", action="store_true",
                       help="the mouse pointer stands in for the eye (try-outs)")
    ap.add_argument("--windowed", action="store_true", help="not full screen (try-outs only)")
    args = ap.parse_args()
    if not args.block1 and not args.no_map:
        found = latest_block1(args.participant, args.eye)
        if found is None:
            print(f'No finished Block 1 run for participant "{args.participant}"'
                  + (f", {args.eye} eye" if args.eye else "") + f" in {PERIMETRY}.\n"
                  "Give --participant as in Block 1, or --block1 <folder>, or --no-map.")
            return 1
        args.block1 = str(found)
    print(f"Block 1 run: {args.block1 or 'none (--no-map)'}", flush=True)

    settings = Block2Settings(
        participant_id=args.participant, eye=args.eye or "R",
        viewing_distance_mm=args.distance_mm, screen_width_mm=args.screen_width_mm,
        screen_height_mm=args.screen_height_mm, train_ms=1000.0 * args.train_s,
        blind_spot=not args.no_blind_spot, setup=not args.no_setup,
        block1_folder=args.block1 or None, forced_choice=args.forced_choice,
        awareness_rating=args.rating,
        fixation_monitoring=FIX_TRACKER if args.tracker else (FIX_MOUSE if args.mouse
                                                              else FIX_NONE))
    if args.block1:
        from specvis.procedure.block2 import load_block1, settings_from_block1
        places, _ = load_block1(args.block1)
        settings = settings_from_block1(settings)
        print(f"Block 1 map: {len(places)} places, {settings.eye} eye, "
              f"{settings.viewing_distance_mm:.0f} mm", flush=True)
    if args.scale:
        from specvis.app.store import ScaleStore
        found = [s for s in ScaleStore().all() if s.name == args.scale]
        if not found:
            print(f'No ColorCAL scale called "{args.scale}". Known: '
                  + ", ".join(f'"{s.name}"' for s in ScaleStore().all()))
            return 1
        calibration = found[0].calibration()
    else:
        calibration = default_calibration(settings.perimetry_settings())
        print("No --scale: contrast and background are for an ASSUMED screen, not measured.")

    flat_map = None
    if args.flat_field:
        from specvis.core.flatfield import FlatFieldStore
        flat_map = FlatFieldStore().find(args.flat_field)
        if flat_map is None:
            print(f'No flat-field map called "{args.flat_field}". Known: '
                  + ", ".join(f'"{m.name}"' for m in FlatFieldStore().all()))
            return 1
        if not calibration.calibrated:
            print("--flat-field needs --scale (a measured luminance scale).")
            return 1

    from specvis.stimulus.psychopy_backend import PsychoPyBackend
    backend = PsychoPyBackend(fullscreen=not args.windowed)
    background = background_for(settings.perimetry_settings(), calibration)[1]
    backend.open(1400, 900, background_brightness=background)
    try:
        print(f"Display {backend.size_px[0]}x{backend.size_px[1]} px, "
              f"measured {backend.refresh_hz:.2f} Hz", flush=True)
        if flat_map is not None:
            from specvis.core.flatfield import FlatField
            ff = FlatField(flat_map, calibration, background, *backend.size_px)
            backend.set_flat_field(ff)
            print(f"flat-field correction: {flat_map.describe()} ({ff.surface_kind})",
                  flush=True)
        else:
            print("flat-field correction: none", flush=True)
        try:
            outcome = run_block2(settings, backend, calibration=calibration)
        except SessionError as error:
            print(f"Cannot run: {error}", flush=True)
            backend.show_screen(f"Cannot run Block 2:\n\n{error}\n\nPress SPACE.")
            return 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = write_outputs(outcome, OUTPUT / f"{settings.participant_id}_{settings.eye}_{stamp}")
        print("\n" + results_text(outcome) + f"\n\nSaved to {folder}", flush=True)
        try:
            from tools.block2_report import report
            report(folder)
            print("Figures and results_std.csv written there too.", flush=True)
        except Exception as error:                 # the data are saved; figures are extra
            print(f"(figures not made: {error})", flush=True)
        backend.show_screen(results_text(outcome) + "\n\nPress SPACE to close.")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
