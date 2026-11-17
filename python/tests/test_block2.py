"""Block 2 quick test (plan of 17 Nov): Block 1's map gives the places and contrasts;
static first, then 20 and 38 Hz; 2 s patches, 1.2 s to answer, a 500-800 ms random gap;
catch trials; true positives, false presses and right/wrong per spot."""

from __future__ import annotations

import collections
import csv
import json
import math
import pathlib
import random
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from specvis.procedure.block2 import (BLIND_SPOT, CATCH, NOT_SEEN,  # noqa: E402
                                      STATIC, THRESHOLD, Block2Settings,
                                      load_block1, plan_phase, plan_trials, rate_table,
                                      results_text, run_block2, write_outputs)
from specvis.procedure.fixation import ScriptedMonitor  # noqa: E402
from specvis.procedure.session import SessionError  # noqa: E402
from specvis.stimulus.backend import NullBackend  # noqa: E402

ok = True
REFRESH = 239.76


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok    " if condition else "  FAIL  ") + what)
    ok = ok and bool(condition)


def fake_block1(folder: pathlib.Path, eye: str = "L") -> pathlib.Path:
    """A finished 30-spot Block 1 run: the outer column not seen, the rest seen at
    10 % or 60 %; two grid spots dropped as blind spot."""
    folder.mkdir(parents=True, exist_ok=True)
    s = -1 if eye == "L" else 1
    dropped = [[15 * s, -3], [15 * s, 3]]
    rows = []
    i = 0
    for y in (-9, -3, 3, 9):
        for x in (-21, -15, -9, -3, 3, 9, 15, 21):
            if [x, y] in dropped:
                continue
            blind = abs(x) == 21
            c = 0.6 if abs(x) == 15 else 0.1
            rows.append(dict(point=i, x_deg=float(x), y_deg=float(y),
                             category="blind" if blind else ("weak" if c > 0.2 else "seen"),
                             valid=str(not blind), threshold_contrast="" if blind else c,
                             threshold_db="" if blind else round(-20 * math.log10(c), 2)))
            i += 1
    with open(folder / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (folder / "session.json").write_text(json.dumps(dict(
        blind_spot_grid_points_dropped=dropped,
        settings=dict(eye=eye, screen_width_mm=542.3, screen_height_mm=305.0,
                      viewing_distance_mm=570.0, background_cd_m2=10.0,
                      diameter_degrees=2.0, spatial_frequency_cpd=4.0))))
    return folder


class Person(NullBackend):
    """Sees every patch except at ``blind_xy`` (pixel centres), says the right
    direction, and presses on the first ``false_presses`` catch trials."""
    def __init__(self, false_presses=1):
        super().__init__(refresh_hz=REFRESH)
        self.current = None
        self.shown, self.answer_timeouts = [], []
        self.false_left = false_presses
        self.blind_xy = set()

    def present_train(self, polarities, frames_per_reversal, total_frames, fixation=None,
                      response_keys=()):
        self.shown.append((len(polarities), frames_per_reversal, total_frames))
        b = polarities[0].brightness
        horizontal = np.abs(np.diff(b, axis=0)).mean() > np.abs(np.diff(b, axis=1)).mean()
        self.current = (polarities[0].centre_px, "left" if horizontal else "right")
        return super().present_train(polarities, frames_per_reversal, total_frames,
                                     fixation, response_keys)

    def wait_for_key(self, keys, timeout_ms, fixation=None, clear_events=True):
        if "left" not in keys:                      # a gap (only the + is up)
            self.current = None
            return None, None
        self.answer_timeouts.append(timeout_ms)
        if self.current is None:                    # a catch trial
            if self.false_left > 0:
                self.false_left -= 1
                return "left", 700.0
            return None, None
        centre, key = self.current
        self.current = None
        if any(abs(centre[0] - bx) < 1 and abs(centre[1] - by) < 1
               for bx, by in self.blind_xy):
            return None, None
        return key, 400.0


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
        return dict(left=(-32.0, 170.0, 570.0), right=(32.0, 170.0, 570.0),
                    distance_mm=570.0, left_box=(0.5, 0.5, 0.5), right_box=(0.5, 0.5, 0.5))

    def cal_enter(self, eye=None):
        self.calls.append(("enter", eye))

    def cal_point(self, x, y):
        return True

    def cal_compute(self):
        self.calls.append("compute")
        return dict(ok=True, status="success", points=5)

    def cal_leave(self):
        pass


with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    b1 = fake_block1(tmp / "P_L_run")

    print("1. Block 1's map gives the places and contrasts")
    places, meta = load_block1(b1)
    kinds = collections.Counter(p.kind for p in places)
    require(len(places) == 32 and kinds[THRESHOLD] == 22 and kinds[NOT_SEEN] == 8
            and kinds[BLIND_SPOT] == 2,
            f"30 tested spots (22 at threshold, 8 not seen) + 2 blind-spot spots: {dict(kinds)}")
    require(all(p.contrast in (0.1, 0.6) for p in places if p.kind == THRESHOLD)
            and all(p.contrast == 0.0 for p in places if p.kind != THRESHOLD),
            "own threshold where seen; 0 (= highest) where not seen and at the blind spot")
    s = Block2Settings(eye="R", block1_folder=str(b1))
    require(len(Block2Settings(block1_folder=str(b1), blind_spot=False).places()) == 30,
            "--no-blind-spot leaves the 30 grid spots")
    try:
        load_block1(tmp / "nothing")
        require(False, "a missing run is refused")
    except SessionError:
        require(True, "a missing Block 1 run is refused")

    print("2. the plan: static first, then 20 and 38 Hz; each direction twice")
    st = plan_phase(s, 0, random.Random(1))
    fl = plan_phase(s, 1, random.Random(1))
    patches = [t for t in st if t.place]
    combos = collections.Counter((t.place.name, t.orientation) for t in patches)
    require(len(patches) == 128 and set(combos.values()) == {2}
            and {t.requested_hz for t in st} == {STATIC},
            "static: 32 places x 2 directions x 2 = 128 patches")
    require(sum(1 for t in st if t.place is None) == 13
            and sum(1 for t in fl if t.place is None) == 26,
            "catch trials 10 % of the patches: 13 static, 26 flicker")
    require(collections.Counter(t.requested_hz for t in fl if t.place) == {20.0: 128, 38.0: 128},
            "flicker: 128 at 20 Hz + 128 at 38 Hz, mixed")
    allt = plan_trials(s, random.Random(1))
    first = next(i for i, t in enumerate(allt) if t.phase == 1)
    require(all(t.phase == 0 for t in allt[:first]) and all(t.phase == 1 for t in allt[first:]),
            "every static trial comes before any flicker trial")
    r = {x["requested_hz"]: x for x in rate_table(s.rates, REFRESH)}
    require(r[STATIC]["frames_per_reversal"] == 0 and r[20.0]["shown_hz"] == 19.98
            and r[38.0]["shown_hz"] == 39.96,
            "239.76 Hz: static, 20 -> 19.98 Hz, 38 -> 39.96 Hz")

    print("3. a run")
    person = Person(false_presses=2)
    person.open(1920, 1080, 0.5)
    o = run_block2(s, person, rng=random.Random(7))
    require(o.settings.eye == "L", "the eye is taken from the Block 1 run (L), not the default")
    bs = [p for p in o.settings.places() if p.kind == BLIND_SPOT]
    require(all(p.x_deg == -15 for p in bs), "blind spot on the left eye's side (-15 deg)")
    require(not o.aborted and len(o.rows) == 128 + 13 + 256 + 26 and o.counts["catch"] == 39,
            f"423 trials: 384 patches + 39 catch ({len(o.rows)})")
    frames = round(2.0 * REFRESH)
    require(all(n == 1 and t == frames for n, _, t in person.shown[:128])
            and all(n == 2 and t == frames for n, _, t in person.shown[128:]),
            "static: one polarity for 2 s; flicker: two polarities (pattern reversal), 2 s")
    require(1200.0 in person.answer_timeouts and 2000.0 in person.answer_timeouts
            and all(t in (1200.0, 2000.0) for t in person.answer_timeouts),
            "answer window 1.2 s after the patch (catch: the 2 s, then 1.2 s)")
    gaps = [row["gap_ms"] for row in o.rows]
    require(min(gaps) >= 500 and max(gaps) <= 800 and len(set(gaps)) > 50,
            f"random gap 500-800 ms ({min(gaps)}-{max(gaps)})")
    pats = [row for row in o.rows if row["place"] != CATCH]
    require(all(row["outcome"] == "right" for row in pats),
            "this person sees everything here and always says the right direction")
    fps = [row for row in o.rows if row["place"] == CATCH and row["answer"]]
    require(len(fps) == 2 and all(row["outcome"] == "false press" for row in fps),
            "2 presses on catch trials -> 2 false presses")
    c_thr = {row["contrast"] for row in pats if row["kind"] == THRESHOLD}
    c_max = {row["contrast"] for row in pats if row["kind"] != THRESHOLD}
    require(c_thr == {0.1, 0.6} and len(c_max) == 1 and min(c_max) > 0.9,
            f"contrast: Block 1 thresholds {sorted(c_thr)}, else the highest {c_max}")

    print("4. scoring: true positives, misses, false presses, per spot")
    p2 = Person(false_presses=0)
    p2.open(1920, 1080, 0.5)
    d, ppm_x = 570.0, 1920 / 542.3
    ppm_y = 1080 / min(305.0, 542.3 * 1080 / 1920)
    p2.blind_xy = {(960 + d * math.tan(math.radians(-15)) * ppm_x,
                    540 - d * math.tan(math.radians(y)) * ppm_y) for y in (-3, 3)}
    o2 = run_block2(Block2Settings(block1_folder=str(b1)), p2, rng=random.Random(3))
    ov = {(x["condition"], x["places"]): x for x in o2.overall()}
    require(ov[("all", BLIND_SPOT)]["true_positives"] == 0
            and ov[("all", BLIND_SPOT)]["misses"] == 24
            and ov[("all", THRESHOLD)]["tp_pct"] == 100,
            "blind spot: 0 of 24 seen; threshold spots: all seen")
    require(ov[("all", "all")]["catch_trials"] == 39 and ov[("all", "all")]["fp_pct"] == 0
            and ov[("all", "all")]["catch_pct_of_trials"] == 9,
            "39 catch trials (9 % of all trials), no false presses")
    summ = o2.summary()
    require(len(summ) == 32 * 3 and all(x["shown"] == 4 for x in summ),
            "summary: 32 spots x 3 conditions, 4 patches each (2 H + 2 V)")
    one_bs = next(x for x in summ if x["kind"] == BLIND_SPOT and x["condition"] == "static")
    require(one_bs["seen"] == 0 and one_bs["not_seen"] == 4 and one_bs["seen_pct"] == 0,
            "per spot: shown, seen, right, wrong, not seen, %")
    txt = results_text(o2)
    require("static" in txt and "20 Hz" in txt and "38 Hz" in txt and "false presses" in txt,
            "results screen: per condition, seen / right / wrong / false presses")
    folder = write_outputs(o2, tmp / "out")
    require(all((folder / n).exists() for n in ("trials.csv", "summary.csv", "overall.csv",
                                                 "session.json")),
            "trials.csv, summary.csv, overall.csv, session.json written")

    print("5. the eye tracker: the setup calibrates; no perimetry, no blind-spot search")
    ct = CalTracker()
    p3 = Person(false_presses=0)
    p3.open(1920, 1080, 0.5)
    o3 = run_block2(Block2Settings(block1_folder=str(b1), fixation_monitoring="eye tracker"),
                    p3, rng=random.Random(2), fixation_monitor=ct)
    val = (o3.setup.get("gaze_check") or {}).get("validation") or {}
    require(("enter", "L") in ct.calls and val.get("passed")
            and not (o3.setup.get("blind_spot_search") or {}).get("found_xy")
            and len([r for r in o3.rows if r["place"] != CATCH]) == 384
            and all(r["gaze_max_deg"] != "" for r in o3.rows if r["place"] != CATCH),
            "the tested eye is calibrated first; then the 384 patches with gaze recorded")

    print("6. without a map (try-outs): 4 quadrants + the average blind spot")
    p4 = Person(false_presses=0)
    p4.open(1920, 1080, 0.5)
    o4 = run_block2(Block2Settings(eye="R", setup=False), p4, rng=random.Random(1))
    names = {r["place"] for r in o4.rows} - {CATCH}
    require(len(names) == 5 and BLIND_SPOT in names, "4 quadrants + blind spot")
    try:
        Block2Settings(eye="both").validate()
        require(False, "both eyes refused")
    except SessionError:
        require(True, "both eyes refused")

print("\nALL OK" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)
