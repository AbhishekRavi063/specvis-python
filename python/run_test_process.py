#!/usr/bin/env python
"""The stimulus side of the app: runs in its own process, so PsychoPy's full-screen
window and the app's Qt window never share an event loop (on a Mac they cannot).

    run_test_process.py test  JOB.json   one eye of perimetry; writes JOB.result.json
    run_test_process.py check JOB.json   the screen check: a box of known size to measure

``JOB.json`` holds ``{"settings": {...}, "screen": 0, "fullscreen": true}``.
The result file holds ``{"folder": ..., "valid": ...}`` or ``{"error": ...}``.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from specvis.app.store import PERIMETRY_DIR, PRIOR_TABLE_FILE  # noqa: E402
from specvis.core.luminance import LuminanceCalibration  # noqa: E402
from specvis.procedure.perimetry import (background_for, default_calibration,  # noqa: E402
                                         field_half_degrees, fit_problem,
                                         results_marks, results_text, run_perimetry,
                                         settings_from_dict, write_outputs)
from specvis.procedure.prior_table import PriorTable  # noqa: E402
from specvis.procedure.session import SessionError  # noqa: E402

CHECK_BOX_MM = 100.0


def _log(row: dict) -> None:
    if row["note"]:
        print(f"trial {row['trial']}: {row['kind']} ({row['x_deg']}, {row['y_deg']}) "
              f"-> {row['note']}", flush=True)


def settings_and_calibration(job: dict):
    """The settings, and the luminance scale chosen in step 2 (or the assumed display).

    The background is the set cd/m^2 (10, the Humphrey's) on the measured scale, or on
    an assumed display when there is no scale. With ``background_cd_m2`` 0 the old
    Specvis way is kept: the background is the brightness % of the scale. The highest
    contrast is limited to what the screen can show around the background.
    """
    settings = settings_from_dict(job["settings"])
    scale = job.get("scale")
    if not scale:
        calibration = default_calibration(settings)
    else:
        if len(scale.get("table") or []) == 101:       # measured by the ColorCAL tool
            calibration = LuminanceCalibration.from_table(
                scale["table"], f"scale {scale.get('name', '?')} (ColorCAL, measured)")
        else:
            calibration = LuminanceCalibration.from_measurements(
                scale["measurements"], f"scale {scale.get('name', '?')} (photometer)")
        if settings.background_cd_m2 <= 0:
            settings = dataclasses.replace(settings, mean_luminance=float(
                calibration.luminance_at(settings.background_brightness_pct)))
    mean, _ = background_for(settings, calibration)
    limit = calibration.max_achievable_contrast(mean)
    zest = dataclasses.replace(settings.zest,
                               max_contrast=max(min(settings.zest.max_contrast, limit), 0.002))
    settings = dataclasses.replace(settings, zest=zest,
                                   high_contrast=min(settings.high_contrast, zest.max_contrast),
                                   blind_spot_contrast=min(settings.blind_spot_contrast,
                                                           zest.max_contrast))
    return settings, calibration


def run_test(job: dict, result_path: pathlib.Path) -> int:
    from specvis.stimulus.psychopy_backend import PsychoPyBackend

    settings, calibration = settings_and_calibration(job)
    table = PriorTable.load(PRIOR_TABLE_FILE) if PRIOR_TABLE_FILE.exists() else None
    backend = PsychoPyBackend(screen=int(job.get("screen", 0)),
                              fullscreen=bool(job.get("fullscreen", True)))
    _, background = background_for(settings, calibration)
    backend.open(1400, 900, background_brightness=background)
    try:
        print(f"display {backend.size_px[0]}x{backend.size_px[1]} px, "
              f"{backend.refresh_hz:.2f} Hz; luminance: {calibration.source}", flush=True)
        try:
            outcome = run_perimetry(settings, backend, calibration=calibration,
                                    prior_table=table, on_trial=_log)
        except SessionError as error:
            backend.show_screen(f"Cannot run this test:\n\n{error}\n\nPress SPACE.")
            result_path.write_text(json.dumps({"error": str(error)}))
            return 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = write_outputs(outcome,
                               PERIMETRY_DIR / f"{settings.participant_id}_{settings.eye}_{stamp}",
                               table)
        result_path.write_text(json.dumps({"folder": str(folder), "valid": outcome.valid}))
        print(f"saved {folder}", flush=True)
        backend.show_screen(results_text(outcome).replace("Press SPACE to finish.",
                                                          "Press SPACE to go back to the app."),
                            results_marks(outcome, table))
    finally:
        backend.close()
    return 0


def run_check(job: dict) -> int:
    """A box that must measure exactly 10 cm, and the field the settings imply."""
    from psychopy import event, visual
    from specvis.stimulus.psychopy_backend import PsychoPyBackend

    settings = settings_from_dict(job["settings"])
    backend = PsychoPyBackend(screen=int(job.get("screen", 0)),
                              fullscreen=bool(job.get("fullscreen", True)))
    backend.open(1400, 900, background_brightness=0.5)
    try:
        w_px, h_px = backend.size_px
        win, scale = backend._win, backend._scale
        box_px = CHECK_BOX_MM * w_px / settings.screen_width_mm / scale   # in points
        hx, hy = field_half_degrees(settings, w_px, h_px)
        problem = fit_problem(settings, w_px, h_px)
        visual.Rect(win, width=box_px, height=box_px / 2, units="pix", lineColor="white",
                    fillColor=None, lineWidth=3).draw()
        text = (f"SCREEN CHECK.  This box should be exactly {CHECK_BOX_MM / 10:.0f} cm wide "
                f"(and {CHECK_BOX_MM / 20:.0f} cm tall).\n"
                "Measure it with a ruler. If it is not, correct the screen width:\n"
                "true width = entered width x 10 / measured box width (cm).\n\n"
                f"Entered: width {settings.screen_width_mm / 10:.1f} cm, distance "
                f"{settings.viewing_distance_mm / 10:.1f} cm, {w_px}x{h_px} px, "
                f"{backend.refresh_hz:.1f} Hz -> covers +/-{hx:.1f} x +/-{hy:.1f} deg.\n"
                + ("The grid FITS." if not problem else "The grid DOES NOT FIT: " + problem)
                + "\n\nPress SPACE to close.")
        visual.TextStim(win, text=text, units="pix", height=h_px / scale * 0.022,
                        wrapWidth=w_px / scale * 0.85, color="white",
                        pos=(0, h_px / scale * 0.3)).draw()
        backend.draw_fixation((w_px / 2.0, h_px / 2.0), 30.0)
        win.flip()
        event.waitKeys(keyList=["space", "escape"])
        print(json.dumps({"width_px": w_px, "height_px": h_px, "refresh_hz": backend.refresh_hz,
                          "half_degrees": [hx, hy], "fits": problem is None}), flush=True)
    finally:
        backend.close()
    return 0


def main(argv) -> int:
    if len(argv) != 2 or argv[0] not in ("test", "check"):
        print(__doc__)
        return 2
    job_path = pathlib.Path(argv[1])
    job = json.loads(job_path.read_text())
    if argv[0] == "check":
        return run_check(job)
    return run_test(job, job_path.with_suffix(".result.json"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
