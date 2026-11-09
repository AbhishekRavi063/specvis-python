#!/usr/bin/env python
"""Run one threshold session: settings dialog, the session, the results map.

    python/specvis.sh run

Saves to ``python/output/<participant>_<date>_<time>/``:

* ``trials.csv`` -- one row per presentation, the Java's format
* ``gabor_presentations.csv`` -- requested vs rendered contrast, the Java's format
* ``summary.csv`` -- one row per location: status, threshold, trials
* ``session.json`` -- settings, measured refresh, actual reversal rate, dropped frames

Timing is software self-report until a photodiode has checked it, and contrast is
nominal until the screen has been measured with a photometer. Both are written into
every session folder.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from specvis.procedure.session import (BISECTION, FOUR_TWO, SessionError,  # noqa: E402
                                       SessionSettings, results_marks, results_text,
                                       run_session, write_outputs)

OUTPUT = HERE / "output"
LAST_SETTINGS = OUTPUT / "last_settings.json"

RATES = {"6 reversals/s": 6.0, "10 reversals/s": 10.0, "30 reversals/s": 30.0,
         "static (no reversal)": 0.0}


def _defaults() -> dict:
    values = {
        "participant_id": "demo",
        "eye": ["both", "L", "R"],
        "method": [FOUR_TWO, BISECTION],
        "reversal_rate": list(RATES),
        "stimulus_ms": 1000,
        "eccentricity_deg": 5.0,
        "screen_width_cm": 30.2,
        "viewing_distance_cm": 57.0,
        "fullscreen": True,
    }
    try:
        last = json.loads(LAST_SETTINGS.read_text())
    except (OSError, ValueError):
        return values
    for key, value in last.items():
        if key not in values:
            continue
        if isinstance(values[key], list):     # put the last choice first
            if value in values[key]:
                values[key] = [value] + [v for v in values[key] if v != value]
        else:
            values[key] = value
    return values


def ask_settings():
    from psychopy import gui

    info = _defaults()
    dlg = gui.DlgFromDict(
        info, title="ReFreM threshold session",
        order=["participant_id", "eye", "method", "reversal_rate", "stimulus_ms",
               "eccentricity_deg", "screen_width_cm", "viewing_distance_cm", "fullscreen"],
        tip={"screen_width_cm": "visible image width, measured (14-inch MacBook Pro: 30.2)",
             "viewing_distance_cm": "eye to screen, measured",
             "eccentricity_deg": "the four locations sit at (+/-e, +/-e) degrees"})
    if not dlg.OK:
        return None, None
    OUTPUT.mkdir(exist_ok=True)
    LAST_SETTINGS.write_text(json.dumps(info, indent=2))

    settings = SessionSettings(
        participant_id=str(info["participant_id"]).strip() or "demo",
        eye=info["eye"], method=info["method"],
        reversal_hz=RATES[info["reversal_rate"]],
        train_ms=float(info["stimulus_ms"]),
        eccentricity_degrees=float(info["eccentricity_deg"]),
        screen_width_mm=float(info["screen_width_cm"]) * 10.0,
        viewing_distance_mm=float(info["viewing_distance_cm"]) * 10.0,
    )
    return settings, bool(info["fullscreen"])


def main() -> int:
    from specvis.stimulus.psychopy_backend import PsychoPyBackend

    settings, fullscreen = ask_settings()
    if settings is None:
        print("Cancelled.")
        return 0

    backend = PsychoPyBackend(fullscreen=fullscreen)
    backend.open(1400, 900, background_brightness=0.5)
    try:
        print(f"Display {backend.size_px[0]}x{backend.size_px[1]} px, "
              f"measured {backend.refresh_hz:.2f} Hz", flush=True)
        if not fullscreen:
            print("WINDOWED: degrees assume the window is the full screen width, so "
                  "positions and sizes are only approximate. Use full screen for data.",
                  flush=True)
        try:
            outcome = run_session(settings, backend)
        except SessionError as error:
            print(f"Cannot run: {error}", flush=True)
            backend.show_screen(f"Cannot run this session:\n\n{error}\n\nPress SPACE.")
            return 1

        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = write_outputs(outcome, OUTPUT / f"{settings.participant_id}_{stamp}")

        print()
        print(results_text(outcome).replace("\nPress SPACE to finish.", ""))
        for o in outcome.locations:
            value = (f"{o.threshold_contrast:.3f} ({o.threshold_db:.1f} dB)"
                     if o.result.valid else o.status.upper())
            print(f"  location {o.location.index} {o.location.quadrant:12s} "
                  f"({o.location.degrees_x:+.0f}, {o.location.degrees_y:+.0f}) deg: "
                  f"{value}, {o.result.trials} trials")
        print(f"\nSaved to {folder}", flush=True)

        backend.show_screen(results_text(outcome), results_marks(outcome, backend.size_px))
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
