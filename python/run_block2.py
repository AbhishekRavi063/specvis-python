#!/usr/bin/env python
"""Block 2, quick test: horizontal or vertical, at 9 reversal rates, in 4 quadrants and
the natural blind spot. See specvis/procedure/block2.py.

    python\\specvis.bat block2 --participant P01 --eye R --scale "Lab screen" --tracker
    python/specvis.sh block2 --eye R --windowed          (a try-out on the Mac)

Starts with Block 1's setup: with --tracker, head position, calibration of the tested
eye and the accuracy check; then the blind-spot search (press SPACE when a flash is
seen), so the blind-spot patch goes where this person does not see.

Saves to ``python/output/block2/<participant>_<eye>_<date>_<time>/``: trials.csv (every
patch, with the asked-for and the shown rate), summary.csv (right answers per place and
rate) and session.json. Set the screen to its highest refresh rate first (Windows:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--participant", default="pilot")
    ap.add_argument("--eye", choices=("L", "R"), default="R",
                    help="the tested eye; cover the other one")
    ap.add_argument("--distance-mm", type=float, default=570.0)
    ap.add_argument("--screen-width-mm", type=float, default=542.3,
                    help="the lab BenQ: 542.3 (MacBook 14\": 302)")
    ap.add_argument("--screen-height-mm", type=float, default=305.0,
                    help="the lab BenQ: 305 (0 = from the width and the pixels)")
    ap.add_argument("--scale", default="",
                    help='the ColorCAL scale to use, e.g. "Lab screen" (none: an assumed screen)')
    ap.add_argument("--train-s", type=float, default=5.0, help="seconds of flicker per patch")
    ap.add_argument("--no-blind-spot", action="store_true")
    ap.add_argument("--no-setup", action="store_true",
                    help="skip the calibration and blind-spot search (average blind spot)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--tracker", action="store_true", help="record gaze with the Tobii")
    group.add_argument("--mouse", action="store_true",
                       help="the mouse pointer stands in for the eye (try-outs)")
    ap.add_argument("--windowed", action="store_true", help="not full screen (try-outs only)")
    args = ap.parse_args()

    settings = Block2Settings(
        participant_id=args.participant, eye=args.eye,
        viewing_distance_mm=args.distance_mm, screen_width_mm=args.screen_width_mm,
        screen_height_mm=args.screen_height_mm, train_ms=1000.0 * args.train_s,
        blind_spot=not args.no_blind_spot, setup=not args.no_setup,
        fixation_monitoring=FIX_TRACKER if args.tracker else (FIX_MOUSE if args.mouse
                                                              else FIX_NONE))
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

    from specvis.stimulus.psychopy_backend import PsychoPyBackend
    backend = PsychoPyBackend(fullscreen=not args.windowed)
    backend.open(1400, 900, background_brightness=background_for(
        settings.perimetry_settings(), calibration)[1])
    try:
        print(f"Display {backend.size_px[0]}x{backend.size_px[1]} px, "
              f"measured {backend.refresh_hz:.2f} Hz", flush=True)
        try:
            outcome = run_block2(settings, backend, calibration=calibration)
        except SessionError as error:
            print(f"Cannot run: {error}", flush=True)
            backend.show_screen(f"Cannot run Block 2:\n\n{error}\n\nPress SPACE.")
            return 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = write_outputs(outcome, OUTPUT / f"{settings.participant_id}_{settings.eye}_{stamp}")
        print("\n" + results_text(outcome) + f"\n\nSaved to {folder}", flush=True)
        backend.show_screen(results_text(outcome) + "\n\nPress SPACE to close.")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
