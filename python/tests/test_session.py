"""A whole session, run against simulated observers with no window open.

These prove the session *logic* -- interleaving, staircases, catch trials, boundaries,
the files written -- never the timing, which is photodiode work.

The simulated observer looks only at what the backend is asked to draw: the rendered
Michelson contrast of the patch and where it is. It never sees the staircase.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import random
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.core.gabor import GaborParameters, measure_contrast  # noqa: E402
from specvis.core.luminance import LuminanceCalibration  # noqa: E402
from specvis.procedure.session import (BISECTION, FOUR_TWO, SessionError,  # noqa: E402
                                       SessionSettings, results_marks, run_session,
                                       write_outputs)
from specvis.stimulus.backend import NullBackend  # noqa: E402
from specvis.threshold.staircase42 import contrast_to_db  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


class Observer(NullBackend):
    """Sees a patch when its rendered contrast reaches this location's threshold."""

    def __init__(self, thresholds_by_x_sign=None, see_all=False, see_none=False,
                 abort_after=None):
        super().__init__(refresh_hz=120.0)
        self.thresholds = thresholds_by_x_sign or {}
        self.see_all, self.see_none = see_all, see_none
        self.abort_after = abort_after
        self.presented = []

    def present_train(self, polarities, frames_per_reversal, total_frames,
                      fixation=None, response_keys=()):
        report = super().present_train(polarities, frames_per_reversal, total_frames,
                                       fixation, response_keys)
        b = polarities[0].brightness
        rendered = (b.max() - b.min()) / (b.max() + b.min())
        centre = polarities[0].centre_px
        self.presented.append((centre, rendered))
        if self.abort_after is not None and len(self.presented) > self.abort_after:
            report.keys = [("escape", 10.0)]
            return report
        if self.see_all:
            seen = True
        elif self.see_none or rendered < 1e-9:
            seen = False
        else:
            side = (centre[0] > self.size_px[0] / 2, centre[1] > self.size_px[1] / 2)
            seen = rendered >= self.thresholds[side]
        if seen:
            report.keys = [("space", 420.0)]
        return report

    def wait_for_key(self, keys, timeout_ms, fixation=None, clear_events=True):
        return (None, None)


def opened(backend, w=3024, h=1964):
    backend.open(w, h, 0.5)
    return backend


settings = SessionSettings(participant_id="sim", screen_width_mm=302.0,
                           viewing_distance_mm=570.0)
cal = LuminanceCalibration.uncalibrated()

# Rendered contrast is below the requested carrier contrast (the Gaussian envelope is
# below 1 at the carrier's extrema). Thresholds below are set on the rendered axis, so
# convert to the requested axis for comparison.
base = GaborParameters(2.0, 4.0, 90.0, 0.0, 2.0 / 6.0, 50.0, 0.5)
screen_ppd = 3024 / (2 * math.degrees(math.atan(151 / 570)))
k = measure_contrast(base, cal, screen_ppd, screen_ppd) / 0.5

print("4-2 staircase, deterministic observer, a different threshold per quadrant")
true_rendered = {(False, False): 0.12, (True, False): 0.20,
                 (True, True): 0.30, (False, True): 0.45}
obs = opened(Observer(true_rendered))
out = run_session(settings, obs, cal, rng=random.Random(1), show_instructions=False)
require(not out.aborted, "session completes")
require(all(o.result.valid for o in out.locations), "all four locations valid")
for o in out.locations:
    centre = out.screen.flat_geometry().to_px(o.location.degrees_x, o.location.degrees_y)
    side = (centre[0] > 3024 / 2, centre[1] > 1964 / 2)
    expected = true_rendered[side] / k
    err = abs(contrast_to_db(o.threshold_contrast) - contrast_to_db(expected))
    require(err <= 2.5, f"location {o.location.index}: estimate {o.threshold_contrast:.3f} "
                        f"within 2.5 dB of {expected:.3f} (off by {err:.2f} dB)")
require(all(4 <= o.result.trials <= 20 for o in out.locations), "4-20 trials per location")
require(obs.trains and obs.trains[0] == (20, 120),
        "6 reversals/s at 120 Hz: 20 frames per reversal, 120-frame (1 s) train")

main = [e for e in out.trial_log.entries if e.block == "main"]
last_trial = {}
for i, e in enumerate(main):
    last_trial[e.stimulus_index] = i
bad_repeats = [i for i in range(1, len(main))
               if main[i].stimulus_index == main[i - 1].stimulus_index
               and any(last >= i for loc, last in last_trial.items()
                       if loc != main[i].stimulus_index)]
require(not bad_repeats,
        "no location shown twice running while another location is still pending")
require(sum(e.block == "practice" for e in out.trial_log.entries) == 2, "2 practice rows")
require(sum(o.result.trials for o in out.locations) == len(main),
        "practice and catch trials never reach a staircase")

print("never sees anything")
out = run_session(settings, opened(Observer(see_none=True)), cal,
                  rng=random.Random(2), show_instructions=False)
require(all(o.status == "ceiling" and not o.result.valid for o in out.locations),
        "every location reported CEILING, invalid")
require(all(math.isnan(o.threshold_contrast) for o in out.locations),
        "no threshold value for an invalid location")

print("sees everything, blanks included")
out = run_session(settings, opened(Observer(see_all=True)), cal,
                  rng=random.Random(3), show_instructions=False)
require(all(o.status == "floor" for o in out.locations), "every location reported FLOOR")
require(out.blank_trials > 0 and out.false_alarms == out.blank_trials,
        f"every blank counted as a false alarm ({out.false_alarms}/{out.blank_trials})")

print("Specvis bisection on the same display")
bis = SessionSettings(participant_id="sim", method=BISECTION)
out = run_session(bis, opened(Observer(true_rendered)), cal,
                  rng=random.Random(4), show_instructions=False)
require(all(o.result.valid for o in out.locations), "all four locations valid")
require(all(3 <= o.result.trials <= 4 for o in out.locations), "3-4 presentations each")
ladder = {c / 100 * 0.8 for c in (10, 13, 17, 23, 31, 42, 56, 74, 100)}
require(all(any(abs(o.threshold_contrast - c) < 1e-9 for c in ladder) for o in out.locations),
        "every threshold is a ladder rung x max contrast (section 3.7)")

print("aborting")
out = run_session(settings, opened(Observer(true_rendered, abort_after=6)), cal,
                  rng=random.Random(5), show_instructions=False)
require(out.aborted, "ESC during a stimulus stops the session")
require(any(o.status == "not finished" for o in out.locations),
        "unfinished locations reported as not finished")

print("output files")
with tempfile.TemporaryDirectory() as tmp:
    out = run_session(settings, opened(Observer(true_rendered)), cal,
                      rng=random.Random(6), show_instructions=False)
    folder = write_outputs(out, pathlib.Path(tmp) / "session")
    names = sorted(p.name for p in folder.iterdir())
    require(names == ["gabor_presentations.csv", "session.json", "summary.csv",
                      "trials.csv"], f"four files written: {names}")
    trials_text = (folder / "trials.csv").read_text()
    header = [l for l in trials_text.splitlines() if not l.startswith("#")][0]
    require(header.split(",")[0] == "trial" and len(header.split(",")) == 15,
            "trials.csv has the Java's 15 columns")
    rows = list(csv.DictReader((folder / "summary.csv").open()))
    require(len(rows) == 4 and all(r["status"] == "valid" for r in rows),
            "summary.csv: one valid row per location")
    meta = json.loads((folder / "session.json").read_text())
    require(meta["luminance_calibrated"] is False and "UNCALIBRATED" in meta["luminance_scale"],
            "session.json labels the luminance scale uncalibrated")
    require("not photodiode-validated" in meta["timing_note"], "and the timing as self-report")
    require(len(results_marks(out, (3024, 1964))) == 4, "results map has four marks")

print("settings the display cannot honour")
try:
    run_session(SessionSettings(eccentricity_degrees=30.0), opened(Observer(true_rendered)),
                cal, show_instructions=False)
    require(False, "30 deg locations on a 30 cm screen at 57 cm are refused")
except SessionError as e:
    require("does not fit" in str(e), "30 deg locations on a 30 cm screen at 57 cm are refused")

sys.exit(0 if ok else 1)
