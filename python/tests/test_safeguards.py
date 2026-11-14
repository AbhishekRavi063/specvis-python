"""Gaze-contingent fixation (Tobii eye tracker, or the mouse standing in for it) --
against simulated observers, no window open."""

from __future__ import annotations

import csv
import json
import math
import pathlib
import random
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.procedure.fixation import (FIX_BLIND_SPOT, FIX_MOUSE, FIX_TRACKER,  # noqa: E402
                                        FixationMonitor, ScriptedMonitor, judge_patch)
from specvis.procedure.perimetry import (BLIND, SEEN, WEAK, FULL, ZEST_FULL,  # noqa: E402
                                         PerimetrySettings, categorise, run_perimetry,
                                         write_outputs)
from specvis.procedure.prior_table import build_prior_table  # noqa: E402
from specvis.procedure.session import SessionError  # noqa: E402
from sim_observer import (DIST_MM, H, W, WIDTH_MM, Observer, healthy,  # noqa: E402
                          right_hemianopia, run, seen_contrast)

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


def peek(i: int):
    """Every 4th patch, the eye glances 10 deg to the right (towards a right blind half)."""
    return (10.0, 0.0) if i % 4 == 3 else (0.0, 0.0)


class GazeObserver(Observer):
    """Sees with the retina: a patch's position relative to where the eye looks."""

    def __init__(self, threshold, rng, gaze=lambda i: (0.0, 0.0), **kw):
        super().__init__(threshold, rng, **kw)
        self.gaze, self.n = gaze, 0

    def present_train(self, polarities, frames_per_reversal, total_frames,
                      fixation=None, response_keys=()):
        gx, gy = self.gaze(self.n)
        self.n += 1
        real = self.threshold
        self.threshold = lambda px, py: real(px - gx, py - gy)
        try:
            return super().present_train(polarities, frames_per_reversal, total_frames,
                                         fixation, response_keys)
        finally:
            self.threshold = real


def session(threshold, seed, monitor=None, observer=None, table=None, show=False,
            **overrides):
    settings = PerimetrySettings(participant_id=f"SAFE{seed}", eye="R",
                                 screen_width_mm=WIDTH_MM, viewing_distance_mm=DIST_MM,
                                 pilot_mode=table is None, extent=FULL, **overrides)
    obs = observer or GazeObserver(threshold, random.Random(seed))
    obs.open(W, H, 0.5)
    return run_perimetry(settings, obs, prior_table=table, rng=random.Random(seed),
                         show_instructions=show, fixation_monitor=monitor), obs


print("0. a normal table for the runs below")
folder = pathlib.Path(tempfile.mkdtemp())
for i in range(4):
    o, _ = run(healthy, 300 + i, pid=f"N{i}", settings_overrides=dict(strategy=ZEST_FULL))
    write_outputs(o, folder / f"N{i}")
table, _ = build_prior_table(sorted(folder.glob("*/summary.csv")))

print("1. judging one patch from its gaze samples")
require(judge_patch([(0.5, 0.2), (1.0, -1.0)], 2.5)[0], "eye on the + -> kept")
require(not judge_patch([(0.5, 0.2), (4.0, 0.0)], 2.5)[0], "eye 4 deg away -> thrown away")
require(not judge_patch([None, None, (0.0, 0.0)], 2.5)[0], "mostly blinks -> thrown away")

print("2. the blind-spot check cannot see a patient who looks into a right blind half")
out, _ = session(right_hemianopia, 21, observer=GazeObserver(right_hemianopia, random.Random(21),
                                                              gaze=peek),
                 table=table, fixation_monitoring=FIX_BLIND_SPOT)
right = [p for p in out.points if p.x_deg > 0]
fake = sum(categorise(out, p, table) in (SEEN, WEAK) for p in right)
require(out.checks["fixation"], "the blind-spot check still passes (its spot is in the blind half)")
require(fake >= 3, f"... while {fake} right-side points are wrongly called seen or weak")

print("3. the eye tracker throws those trials away, and the map comes out right")
out, _ = session(right_hemianopia, 21, monitor=ScriptedMonitor(peek),
                 observer=GazeObserver(right_hemianopia, random.Random(21), gaze=peek),
                 table=table, fixation_monitoring=FIX_TRACKER, max_fixation_loss_rate=0.3)
right = [p for p in out.points if p.x_deg > 0]
cats = [categorise(out, p, table) for p in right]
left_ok = sum(categorise(out, p, table) == SEEN for p in out.points if p.x_deg < 0)
require(not any(c == SEEN for c in cats) and sum(c == WEAK for c in cats) <= 1,
        f"right side: {sum(c == BLIND for c in cats)}/{len(right)} blind, none called normal "
        f"(at most 1 weak from a chance press)")
require(left_ok >= 30, f"left side still seen ({left_ok} points)")
rate = out.fixation_loss_rate
require(0.2 < rate < 0.3, f"fixation losses measured directly: {100 * rate:.0f}% of patches")
require(out.counts["blind_spot"] == 0, "no blind-spot trials with the tracker")
breaks = [r for r in out.trial_rows if r["note"].startswith("FIXATION_BREAK")]
require(len(breaks) == out.counts["fixation_breaks"], "every thrown-away trial is in trials.csv")
d = write_outputs(out, pathlib.Path(tempfile.mkdtemp()) / "tracker", table)
require((d / "gaze.csv").exists(), "gaze.csv written")
with (d / "trials.csv").open() as f:
    cols = next(csv.reader(f))
require({"gaze_max_deg", "gaze_valid_pct"} <= set(cols), "trials.csv has gaze columns")
s = out.settings
require(abs(sum(s.catch_rates()) - (s.p_blank + s.p_high + s.p_blind_spot)) < 1e-9,
        "the blind-spot share goes to the other two checks")

print("4. a session where the eye wanders too often is not valid")
out, _ = session(healthy, 22, monitor=ScriptedMonitor(lambda i: (5.0, 0.0) if i % 2 else (0, 0)),
                 fixation_monitoring=FIX_TRACKER)
require(not out.checks["fixation"] and not out.valid,
        f"half the patches thrown away ({100 * out.fixation_loss_rate:.0f}%) -> not valid")


class LateMonitor(FixationMonitor):
    """No gaze for the first 400 polls, then on the +: the reminder screen must appear."""
    kind, watches_gaze = FIX_TRACKER, True

    def __init__(self):
        super().__init__()
        self.polls = 0

    def current(self):
        self.polls += 1
        return None if self.polls < 400 else (0.0, 0.0)

    def between(self, t0, t1):
        return [(0.0, 0.0)]


obs = GazeObserver(healthy, random.Random(23))
out, obs = session(healthy, 23, monitor=LateMonitor(), observer=obs,
                   fixation_monitoring=FIX_TRACKER)
require(out.counts.get("fixation_reminders", 0) >= 1 and
        any("look at the" in t.lower() for t in obs.screens),
        "no fixation for 5 s -> a 'look at the +' reminder")

print("5. the mouse stand-in needs a display that reports the mouse")
try:
    session(healthy, 24, fixation_monitoring=FIX_MOUSE)
    require(False, "refused without a mouse")
except SessionError as e:
    require("mouse" in str(e).lower(), "refused, with a clear message")
try:
    import tobii_research  # noqa: F401
    print("  (Tobii SDK installed: not checking the missing-SDK message)")
except ImportError:
    try:
        session(healthy, 25, fixation_monitoring=FIX_TRACKER)
        require(False, "eye tracker refused without the SDK")
    except SessionError as e:
        require("tracker-bridge" in str(e) and "tracker-setup" in str(e),
                "eye tracker without the SDK and no bridge -> says what to start")

print("6. eye-tracker check before the test")
obs = GazeObserver(healthy, random.Random(29))
out, obs = session(healthy, 29, observer=obs, monitor=ScriptedMonitor(lambda i: (0, 0)),
                   fixation_monitoring=FIX_TRACKER, show=True)
require(any("Eye tracker check" in t for t in obs.screens), "the check screen is shown")
require(out.gaze_check.get("passed") is True, "eye-tracker check at the start passed")
d = write_outputs(out, pathlib.Path(tempfile.mkdtemp()) / "check")
meta = json.loads((d / "session.json").read_text())
require(meta["gaze_check"]["passed"] and "scripted" in meta["fixation_monitor"],
        "session.json records the gaze check and the monitor")

sys.exit(0 if ok else 1)
