"""Block 2 quick test (17 Nov): every place x orientation x rate once, the rates as whole
frames, the blind spot on the tested eye's side, answers scored, gaze recorded."""

from __future__ import annotations

import collections
import csv
import json
import pathlib
import random
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from specvis.procedure.block2 import (BLIND_SPOT, HORIZONTAL, VERTICAL, Block2Settings,  # noqa: E402
                                      plan_trials, rate_table, results_text, run_block2,
                                      write_outputs)
from specvis.procedure.fixation import ScriptedMonitor  # noqa: E402
from specvis.procedure.session import SessionError  # noqa: E402
from specvis.stimulus.backend import NullBackend  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok    " if condition else "  FAIL  ") + what)
    ok = ok and bool(condition)


class Person(NullBackend):
    """Answers right outside the blind spot, guesses inside it; notes the trains."""
    def __init__(self, refresh=239.76, seed=1):
        super().__init__(refresh_hz=refresh)
        self.rng = random.Random(seed)
        self.shown = []

    def present_train(self, polarities, frames_per_reversal, total_frames, fixation=None,
                      response_keys=()):
        self.shown.append((polarities, frames_per_reversal, total_frames))
        return super().present_train(polarities, frames_per_reversal, total_frames,
                                     fixation, response_keys)

    def wait_for_key(self, keys, timeout_ms, fixation=None, clear_events=True):
        if "left" not in keys:
            return None, None                       # a pause: no key
        trial = self.current
        if trial is None or trial.place == BLIND_SPOT:
            return self.rng.choice(["left", "right"]), 600.0
        return ("left" if trial.orientation == HORIZONTAL else "right"), 500.0


print("1. the plan")
s = Block2Settings(eye="R")
trials = plan_trials(s, random.Random(1))
combos = collections.Counter((t.place, t.orientation, t.requested_hz) for t in trials)
require(len(trials) == 90 and set(combos.values()) == {1},
        "5 places x 2 orientations x 9 rates = 90 patches, each once")
require(s.places()[BLIND_SPOT] == (15.5, -1.5)
        and Block2Settings(eye="L").places()[BLIND_SPOT] == (-15.5, -1.5),
        "the blind spot: 15.5 deg temporal (right for the right eye), 1.5 deg below")
require({p for p in s.places() if p != BLIND_SPOT}
        == {"upper right", "upper left", "lower left", "lower right"}
        and all(abs(x) == abs(y) == 5.0 for p, (x, y) in s.places().items() if p != BLIND_SPOT),
        "4 quadrants at (+/-5, +/-5) deg")
try:
    Block2Settings(eye="both").validate()
    require(False, "both eyes refused")
except SessionError:
    require(True, "both eyes refused (the blind spot is blind only with one eye covered)")

print("2. rates are whole frames, and both numbers are kept")
t240 = {r["requested_hz"]: r for r in rate_table(s.frequencies_hz, 240.0)}
t120 = {r["requested_hz"]: r for r in rate_table(s.frequencies_hz, 120.0)}
require(t240[20.0]["exact"] and t240[20.0]["frames_per_reversal"] == 12
        and t240[100.0]["shown_hz"] == 120.0 and not t240[100.0]["exact"]
        and t240[71.0]["shown_hz"] == 80.0 and t240[38.0]["shown_hz"] == 40.0,
        "240 Hz: 20 exact (12 frames); 100 -> 120, 71 -> 80, 38 -> 40")
require(t120[25.0]["shown_hz"] == 24.0 and t120[100.0]["frames_per_reversal"] == 1,
        "120 Hz: 25 -> 24; 100 -> 120 (1 frame, the fastest possible)")

print("3. a run")
person = Person()
orig = person.present_train
order = plan_trials(Block2Settings(eye="R"), random.Random(7))
it = iter(order)


def tracking_train(polarities, frames_per_reversal, total_frames, fixation=None,
                   response_keys=()):
    person.current = next(it)
    return orig(polarities, frames_per_reversal, total_frames, fixation, response_keys)


person.current = None
person.present_train = tracking_train
person.open(1920, 1080, 0.5)
o = run_block2(Block2Settings(eye="R", setup=False), person, rng=random.Random(7))
require(not o.aborted and len(o.rows) == 90 and len(person.shown) == 90,
        f"90 patches shown and logged ({len(o.rows)})")
require(all(t == round(5.0 * 239.76) for _, _, t in person.shown)
        and all(len(p) == 2 for p, _, _ in person.shown),
        "each patch: 5 s of frames, two opposite polarities (pattern reversal)")
row = o.rows[0]
require(all(k in row for k in ("requested_hz", "shown_hz", "frames_per_reversal", "orientation",
                               "answer", "correct", "place", "x_deg", "y_deg", "contrast")),
        "trials.csv has the asked-for and shown rate, place, orientation, answer, right/wrong")
by_place = collections.defaultdict(list)
for r in o.rows:
    by_place[r["place"]].append(r["correct"])
require(all(all(v) for p, v in by_place.items() if p != BLIND_SPOT)
        and 0 < sum(by_place[BLIND_SPOT]) < 18,
        f"scored: all right in the quadrants, guesses in the blind spot "
        f"({sum(by_place[BLIND_SPOT])}/18)")
require(o.contrast > 0.9, f"full contrast on the assumed screen ({o.contrast:.0%})")
require("chance 50%" in results_text(o) and "blind spot" in results_text(o),
        "results: percent right per place, against chance")
with tempfile.TemporaryDirectory() as tmp:
    folder = write_outputs(o, pathlib.Path(tmp) / "B")
    rows = list(csv.DictReader(open(folder / "trials.csv")))
    summary = list(csv.DictReader(open(folder / "summary.csv")))
    meta = json.loads((folder / "session.json").read_text())
    require(len(rows) == 90 and len(summary) == 45 and meta["rates"]
            and meta["measured_refresh_hz"] == 239.76,
            "trials.csv (90), summary.csv (5 places x 9 rates), session.json with the rates")

print("4. the eye tracker: a patch with the eye off the + is shown again at the end")
person2 = Person(seed=2)
person2.current = None
person2.open(1920, 1080, 0.5)
mon = ScriptedMonitor(lambda i: (4.0, 0.0) if i in (3, 10) else (0.0, 0.0))
o2 = run_block2(Block2Settings(eye="L", fixation_monitoring="eye tracker", setup=False), person2,
                rng=random.Random(3), fixation_monitor=mon)
broken = [r for r in o2.rows if r["note"] == "FIXATION_BREAK"]
repeats = [r for r in o2.rows if r["repeat"]]
require(len(broken) == 2 and len(repeats) == 2 and len(o2.rows) == 92
        and all(not r["counts"] for r in broken) and o2.counts["fixation_breaks"] == 2,
        "2 patches with the eye 4 deg off: not counted, shown once more at the end")
require(all(r["gaze_max_deg"] != "" for r in o2.rows), "gaze recorded for every patch")

print("5. the screen must hold the blind spot")
small = Person()
small.current = None
small.open(1920, 1080, 0.5)
try:
    run_block2(Block2Settings(eye="R", screen_width_mm=302.0, screen_height_mm=0.0,
                              viewing_distance_mm=560.0, setup=False), small)
    require(False, "too small a screen is refused")
except SessionError as e:
    require("blind spot" in str(e), f"too small a screen is refused: {e}")

print("6. the setup first: Block 1's calibration and blind-spot search")


class Searcher(Person):
    """Sees every static flash except inside its own blind spot (16.0, -3.0) +/- 1 deg;
    in Block 2 answers right everywhere (a cooperative try-out)."""
    def __init__(self):
        super().__init__(refresh=239.76, seed=4)
        self.current = None
        self.last_train = None

    def present_train(self, polarities, frames_per_reversal, total_frames, fixation=None,
                      response_keys=()):
        self.last_train = polarities[0].centre_px
        return super().present_train(polarities, frames_per_reversal, total_frames,
                                     fixation, response_keys)

    def wait_for_key(self, keys, timeout_ms, fixation=None, clear_events=True):
        if "space" in keys and "left" not in keys and self.last_train is not None:
            x, y = self.to_deg(*self.last_train)
            self.last_train = None
            inside = abs(x - 16.0) <= 1.0 and abs(y + 3.0) <= 1.0
            return (None, None) if inside else ("space", 400.0)
        if "left" in keys:
            return "left", 500.0
        return None, None


import math  # noqa: E402
sr = Searcher()
sr.open(1920, 1080, 0.5)
d = 570.0
ppm = 1920 / 542.3


def to_deg(px, py):
    return (math.degrees(math.atan((px - 960) / ppm / d)),
            math.degrees(math.atan((540 - py) / (1080 / min(305.0, 542.3 * 1080 / 1920)) / d)))


sr.to_deg = to_deg
o6 = run_block2(Block2Settings(eye="R"), sr, rng=random.Random(5))
bs = o6.settings.places()[BLIND_SPOT]
require(o6.setup.get("blind_spot_used") == "found by the search"
        and abs(bs[0] - 16.0) <= 1.0 and abs(bs[1] + 3.0) <= 1.0
        and all(r["x_deg"] == bs[0] for r in o6.rows if r["place"] == BLIND_SPOT)
        and len(o6.rows) == 90,
        f"the blind spot is searched first and the Block 2 patch goes there: {bs} "
        "(the average 15.5, -1.5 is seen by this person)")
p7 = Person(seed=9)
p7.current = None
p7.open(1920, 1080, 0.5)
o7 = run_block2(Block2Settings(eye="R"), p7, rng=random.Random(5))
require(o7.setup.get("blind_spot_used") == "found by the search"
        and o7.settings.places()[BLIND_SPOT] == (15.5, -1.5),
        "nothing seen at the average position -> that is the blind spot")



class CalTracker(ScriptedMonitor):
    """A calibratable tracker, always on target, 570 mm away, head centred."""
    calibration_supported = True

    def __init__(self):
        super().__init__(lambda i: (0.0, 0.0))
        self.calls, self.look = [], (0.0, 0.0)

    def target_shown(self, x, y):
        self.look = (x, y)

    def current(self):
        self._add(__import__("time").perf_counter(), self.look)
        return self.look

    def patch_shown(self, t0, t1):
        self.look = (0.0, 0.0)
        super().patch_shown(t0, t1)

    def eye_distance_mm(self, window_s=0.5):
        return 570.0

    def display_area(self):
        return dict(width_mm=542.3, height_mm=305.0)

    def head_position(self):
        return dict(left=(-32.0, 170.0, 570.0), right=(32.0, 170.0, 570.0), distance_mm=570.0,
                    left_box=(0.5, 0.5, 0.5), right_box=(0.5, 0.5, 0.5))

    def cal_enter(self, eye=None):
        self.calls.append(("enter", eye))

    def cal_point(self, x, y):
        return True

    def cal_compute(self):
        self.calls.append("compute")
        return dict(ok=True, status="success", points=5)

    def cal_leave(self):
        pass


ct = CalTracker()
p8 = Person(seed=3)
p8.current = None
p8.open(1920, 1080, 0.5)
o8 = run_block2(Block2Settings(eye="R", fixation_monitoring="eye tracker"), p8,
                rng=random.Random(2), fixation_monitor=ct)
val = (o8.setup.get("gaze_check") or {}).get("validation") or {}
require(("enter", "R") in ct.calls and "compute" in ct.calls and val.get("passed")
        and len(o8.rows) >= 90 and "Tracker accuracy" in results_text(o8),
        f"with the eye tracker: the tested eye is calibrated and its accuracy measured "
        f"first ({val.get('mean_accuracy_deg')} deg), then Block 2 runs")

print("\nALL OK" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)
