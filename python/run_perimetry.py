#!/usr/bin/env python
"""Run one eye of monocular perimetry: settings dialog, the test, the map.

    python/specvis.sh perimetry

Saves to ``python/output/perimetry/<participant>_<eye>_<date>_<time>/``:

* ``trials.csv``   -- every presentation, including catch trials
* ``summary.csv``  -- one row per point: threshold, category, trials, starting guess
* ``session.json`` -- settings, screen, checks (false presses, misses, blind spot)
* ``map.png``      -- the map: seen / weak / blind

With pilot mode off, the normal table in ``python/output/perimetry/prior_table.csv``
(made by ``python/specvis.sh build-table``) is used as the starting guess, if it exists.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from specvis.procedure.perimetry import (CENTRAL, FINISH_KEY, FULL,  # noqa: E402
                                         OTHER_EYE_KEY, REPEAT_KEY, PerimetrySettings,
                                         results_marks, results_text, run_perimetry,
                                         write_outputs)
from specvis.procedure.prior_table import PriorTable  # noqa: E402
from specvis.procedure.session import SessionError  # noqa: E402

OUTPUT = HERE / "output" / "perimetry"
LAST_SETTINGS = OUTPUT / "last_settings.json"
PRIOR_TABLE = OUTPUT / "prior_table.csv"


def _defaults() -> dict:
    values = {
        "participant_id": "PILOT01",
        "eye_tested": ["R", "L"],
        "pilot_mode": True,
        "grid": [FULL, CENTRAL],
        "screen_width_cm": 30.2,
        "viewing_distance_cm": 35.0,
        "fullscreen": True,
    }
    try:
        last = json.loads(LAST_SETTINGS.read_text())
    except (OSError, ValueError):
        return values
    for key, value in last.items():
        if key not in values:
            continue
        if isinstance(values[key], list):
            if value in values[key]:
                values[key] = [value] + [v for v in values[key] if v != value]
        else:
            values[key] = value
    return values


def ask_settings():
    from psychopy import gui

    info = _defaults()
    dlg = gui.DlgFromDict(
        info, title="ReFreM perimetry (one eye)",
        order=["participant_id", "eye_tested", "pilot_mode", "grid", "screen_width_cm",
               "viewing_distance_cm", "fullscreen"],
        tip={"eye_tested": "the OPEN eye; cover the other one",
             "pilot_mode": "on for the first 3-5 healthy people (builds the normal table)",
             "grid": "full = +/-21 x +/-13 deg (72 points); central = +/-13 deg",
             "screen_width_cm": "visible image width, measured (14-inch MacBook Pro: 30.2)",
             "viewing_distance_cm": "eye to screen, measured; MacBook 35, lab monitor 57"})
    if not dlg.OK:
        return None, None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LAST_SETTINGS.write_text(json.dumps(info, indent=2))
    settings = PerimetrySettings(
        participant_id=str(info["participant_id"]).strip() or "pilot",
        eye=info["eye_tested"], extent=info["grid"], pilot_mode=bool(info["pilot_mode"]),
        screen_width_mm=float(info["screen_width_cm"]) * 10.0,
        viewing_distance_mm=float(info["viewing_distance_cm"]) * 10.0)
    return settings, bool(info["fullscreen"])


def main() -> int:
    from specvis.stimulus.psychopy_backend import PsychoPyBackend

    settings, fullscreen = ask_settings()
    if settings is None:
        print("Cancelled.")
        return 0

    table = PriorTable.load(PRIOR_TABLE) if PRIOR_TABLE.exists() else None
    if settings.pilot_mode:
        print("PILOT MODE: every point measured on its own (no starting guesses).")
    else:
        print("Starting guesses: " + (f"normal table {table.source} + neighbours"
                                      if table else "neighbours only (no normal table yet)"))

    backend = PsychoPyBackend(fullscreen=fullscreen)
    backend.open(1400, 900, background_brightness=0.5)
    maps = []
    try:
        print(f"Display {backend.size_px[0]}x{backend.size_px[1]} px, "
              f"measured {backend.refresh_hz:.2f} Hz", flush=True)
        if not fullscreen:
            print("WINDOWED: positions assume the window is the full screen width. "
                  "Use full screen for data.", flush=True)
        while True:
            print(f"\n=== {settings.participant_id}: {settings.eye} eye ===", flush=True)
            try:
                outcome = run_perimetry(settings, backend, prior_table=table,
                                        on_trial=lambda r: r["note"] and print(
                                            f"  trial {r['trial']}: {r['kind']} "
                                            f"({r['x_deg']}, {r['y_deg']}) -> {r['note']}",
                                            flush=True))
            except SessionError as error:
                print(f"Cannot run: {error}", flush=True)
                backend.show_screen(f"Cannot run this test:\n\n{error}\n\nPress SPACE.")
                return 1

            stamp = time.strftime("%Y%m%d_%H%M%S")
            folder = write_outputs(
                outcome, OUTPUT / f"{settings.participant_id}_{settings.eye}_{stamp}", table)
            maps.append(folder / "map.png")
            print()
            print(results_text(outcome).rsplit("\n", 1)[0])
            print(f"\nSaved to {folder}", flush=True)

            key = backend.show_screen(results_text(outcome, offer_keys=True),
                                      results_marks(outcome, table),
                                      keys=[FINISH_KEY, OTHER_EYE_KEY, REPEAT_KEY])
            if key == OTHER_EYE_KEY:
                settings = dataclasses.replace(settings, eye="L" if settings.eye == "R" else "R")
            elif key == REPEAT_KEY:
                settings = dataclasses.replace(settings)
            else:
                break
    finally:
        backend.close()
        for path in maps:
            open_file(path)
    return 0


def open_file(path: pathlib.Path) -> None:
    """Show a saved map in the system viewer (Preview on a Mac)."""
    import os
    import subprocess

    if not path.exists():
        return
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("win"):
            os.startfile(str(path))           # noqa: S606 -- a file we just wrote
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError as error:
        print(f"Could not open {path}: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
