"""Eye-tracker calibration at the start (Tobii's 5-point, from the app) and the accuracy
check with the tested eye; through a scripted tracker and through the bridge's protocol."""

from __future__ import annotations

import json
import pathlib
import random
import socket
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from specvis.procedure.fixation import (FIX_TRACKER, FixationMonitor,  # noqa: E402
                                       ScriptedMonitor, TobiiMonitor)
from specvis.procedure.perimetry import (FULL_THRESHOLD, H24_2_CENTRAL,  # noqa: E402
                                         TRACKER_CAL_POINTS, PerimetrySettings, run_perimetry,
                                         write_outputs)
from sim_observer import Observer, W, H, healthy  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok    " if condition else "  FAIL  ") + what)
    ok = ok and bool(condition)


class Tracker(FixationMonitor):
    """A calibratable tracker whose eye looks at each dot, off by ``error`` degrees."""
    kind = FIX_TRACKER
    watches_gaze = True
    calibration_supported = True

    def __init__(self, error=0.3, cal_ok=True, distance=520.0, area=(302.0, 302.0 * 1964 / 3024),
                 corners=None, eyes=None):
        super().__init__()
        self.error, self.cal_ok, self.distance, self.area = error, cal_ok, distance, area
        self.corners, self.eyes = corners, eyes
        self.look = (0.0, 0.0)
        self.calls = []

    def target_shown(self, x, y):
        self.look = (x, y)

    def _read(self):
        return (self.look[0] + self.error, self.look[1])

    def eye_distance_mm(self, window_s=0.5):
        return self.distance

    def display_area(self):
        return dict(width_mm=self.area[0], height_mm=self.area[1], **(self.corners or {}))

    def head_position(self):
        if self.eyes:
            return dict(left=self.eyes[0], right=self.eyes[1], distance_mm=self.distance)
        return dict(left=(-32.0, 170.0, self.distance), right=(32.0, 170.0, self.distance),
                    distance_mm=self.distance)

    def cal_enter(self):
        self.calls.append("enter")

    def cal_point(self, x, y):
        self.calls.append(("point", round(x, 2), round(y, 2)))
        return True

    def cal_compute(self):
        self.calls.append("compute")
        return dict(ok=self.cal_ok, status="success" if self.cal_ok else "failure", points=5)

    def cal_leave(self):
        self.calls.append("leave")

    def patch_shown(self, t0, t1):
        self.look = (0.0, 0.0)          # during the test the eye stays on the +


def session(tracker, **over):
    s = PerimetrySettings(participant_id="CAL", strategy=FULL_THRESHOLD, extent=H24_2_CENTRAL,
                          screen_width_mm=302.0, viewing_distance_mm=334.0,
                          screen_height_mm=302.0 * 1964 / 3024,
                          fixation_monitoring=FIX_TRACKER, **over)
    obs = Observer(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
    obs.open(W, H, 0.5)
    tracker.look = (0.0, 0.0)
    return run_perimetry(s, obs, rng=random.Random(3), show_instructions=True,
                         fixation_monitor=tracker)


print("1. a good calibration and accuracy")
t = Tracker(error=0.3)
o = session(t)
pts = [c for c in t.calls if isinstance(c, tuple)]
require(t.calls[0] == "enter" and t.calls[-2:] == ["compute", "leave"]
        and sorted((p[1], p[2]) for p in pts) == sorted(tuple(map(float, q))
                                                       for q in TRACKER_CAL_POINTS),
        "Tobii 5-point calibration: centre and 4 points in from the corners, then compute")
require([(p[1], p[2]) for p in pts] == [tuple(q) for q in o.gaze_check["calibration"]["order"]],
        f"random order, recorded: {o.gaze_check['calibration']['order']}")
v = o.gaze_check.get("validation", {})
require(v.get("passed") and len(v.get("points", [])) == 5
        and abs(v["centre_accuracy_deg"] - 0.3) < 0.05 and abs(v["mean_accuracy_deg"] - 0.3) < 0.05
        and v["data_loss_pct"] == 0 and v["precision_rms_deg"] is not None,
        f"validation (Titta-style): accuracy {v.get('mean_accuracy_deg')} deg, centre "
        f"{v.get('centre_accuracy_deg')}, precision {v.get('precision_rms_deg')}, "
        f"loss {v.get('data_loss_pct')} %")
require(o.gaze_check.get("passed") and o.valid and len(o.points) == 42,
        "passes, and the test then runs (42 spots)")
with tempfile.TemporaryDirectory() as tmp:
    meta = json.loads((write_outputs(o, pathlib.Path(tmp) / "C") / "session.json").read_text())
    require("calibration" in meta["gaze_check"] and "validation" in meta["gaze_check"],
            "calibration and accuracy saved in session.json")

print("2. poor accuracy is caught")
t = Tracker(error=2.0)
o = session(t)
v = o.gaze_check.get("validation", {})
require(v and not v["passed"] and not o.gaze_check.get("passed"),
        f"2 deg off: not passed (centre {v.get('centre_error_deg')} deg); the operator was asked")

print("3. a failed calibration is reported")
t = Tracker(cal_ok=False)
o = session(t)
require(o.gaze_check["calibration"]["ok"] is False, "calibration failure recorded")

print("4. calibration can be switched off; accuracy is still measured")
t = Tracker(error=0.3)
o = session(t, tracker_calibrate=False)
require(not any(isinstance(c, tuple) for c in t.calls) and o.gaze_check["validation"]["passed"],
        "no calibration commands; accuracy check only")

print("5. the eye's distance, measured by the tracker")
t = Tracker(distance=342.0)          # the test geometry is set for 334 mm
o = session(t)
d = o.gaze_check.get("eye_distance_during_test_mm", {})
require(o.gaze_check.get("eye_distance_mm") == 342.0
        and "eye_distance_accepted_off_by_mm" not in o.gaze_check,
        "8 mm off the set distance: within 2 cm, no message")
require(all(r["eye_distance_mm"] == 342.0 for r in o.trial_rows)
        and d.get("mean") == 342.0 and d.get("trials") == len(o.trial_rows),
        f"every trial records the distance; summary {d}")
t = Tracker(distance=414.0)
o = session(t)
require(o.gaze_check.get("eye_distance_accepted_off_by_mm") == 80.0,
        "8 cm too far: the operator is asked to move the chin rest (went on: recorded)")
o = session(Tracker(distance=414.0), check_eye_distance=False)
require("eye_distance_accepted_off_by_mm" not in o.gaze_check,
        "the distance check can be switched off")

obs = Observer(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
run_perimetry(PerimetrySettings(participant_id="POS", strategy=FULL_THRESHOLD,
                                extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER),
              obs, rng=random.Random(3), show_instructions=True,
              fixation_monitor=Tracker(distance=342.0))
text, marks = obs.statuses[0]
require("Distance 34.2 cm" in text and "good" in text
        and sorted(m.label for m in marks if m.label) == ["L", "R"],
        "live positioning screen: both eyes drawn, distance shown, 'good' inside +/-1 cm")

print("5b. the tracker's own screen size is checked")
o = session(Tracker())
require("tracker_display_area" in o.gaze_check
        and "tracker_display_area_mismatch_mm" not in o.gaze_check,
        "same size as the settings: no warning; the tracker's size is saved")
o = session(Tracker(area=(597.0, 335.8)))
require(o.gaze_check.get("tracker_display_area_mismatch_mm") == [295.0, 139.7],
        "tracker still set for another screen (597 x 335.8 mm): warned, the difference saved")

print("5c. lab fixes (15 Nov): eye marks follow the tilted tracker; short gaps ignored")
import math  # noqa: E402
# The lab's tracker: tilted 20 deg, picture bottom 10.9 mm above it (tracker-display output).
lab = dict(top_left=[-265.4, 297.6, 97.7], top_right=[276.9, 297.6, 97.7],
           bottom_left=[-265.4, 10.9, -6.7])
t20 = math.radians(20.0)
centre = [5.75, 10.9 + 152.5 * math.cos(t20), -6.7 + 152.5 * math.sin(t20)]
normal = [0.0, -math.sin(t20), math.cos(t20)]          # out of the screen, towards the person
eye = lambda dx: [centre[0] + dx, centre[1] + 510 * normal[1],  # noqa: E731
                  centre[2] + 510 * normal[2]]
obs = Observer(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
run_perimetry(PerimetrySettings(participant_id="POS2", strategy=FULL_THRESHOLD,
                                extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER),
              obs, rng=random.Random(3), show_instructions=True,
              fixation_monitor=Tracker(distance=334.0, area=(542.3, 305.0), corners=lab,
                                       eyes=(eye(-32.0), eye(32.0))))
text, marks = obs.statuses[0]
ring = next(m for m in marks if not m.label)
eyes_m = [m for m in marks if m.label]
inside = all(math.hypot(m.centre_px[0] - ring.centre_px[0],
                        m.centre_px[1] - ring.centre_px[1]) < ring.radius_px for m in eyes_m)
require(len(eyes_m) == 2 and inside and abs(eyes_m[0].centre_px[1] - ring.centre_px[1]) < 1,
        "eyes level with the screen centre (tracker tilted 20 deg) are drawn inside the circle")


class Gappy(Tracker):
    """On the +, but every 4th sample lost (as at 50 cm: ~19-25 % missing)."""
    n = 0

    def _read(self):
        Gappy.n += 1
        return None if Gappy.n % 4 == 0 else (0.0, 0.0)


class ColourSpy(Observer):
    """Records each colour the + is given."""
    @property
    def fixation_colour(self):
        return getattr(self, "_fc", "white")

    @fixation_colour.setter
    def fixation_colour(self, c):
        self._fc = c
        self.colours = getattr(self, "colours", []) + [c]


def gappy_session(tracker):
    s = PerimetrySettings(participant_id="GAP", strategy=FULL_THRESHOLD, extent=H24_2_CENTRAL,
                          screen_width_mm=302.0, viewing_distance_mm=334.0,
                          screen_height_mm=302.0 * 1964 / 3024, fixation_monitoring=FIX_TRACKER)
    spy = ColourSpy(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
    spy.open(W, H, 0.5)
    out = run_perimetry(s, spy, rng=random.Random(3), show_instructions=False,
                        fixation_monitor=tracker)
    return out, getattr(spy, "colours", [])


o, colours = gappy_session(Gappy())
require(o.counts.get("fixation_reminders", 0) == 0 and "red" not in colours,
        "single lost samples (16 ms) keep the + white and do not restart the hold")


class Blinky(Tracker):
    """On the + for 20 samples, then lost for 10 (160 ms), again and again."""
    n = 0

    def _read(self):
        Blinky.n += 1
        return None if Blinky.n % 30 >= 20 else (0.0, 0.0)


o, colours = gappy_session(Blinky())
require("red" in colours, "gaze lost for longer than 100 ms still counts as off (the + turns red)")

sys.path.insert(0, str(HERE / "tools"))
import tracker_display  # noqa: E402
area = dict(top_left=[-265.4, 297.6, 97.7], top_right=[276.9, 297.6, 97.7],
            bottom_left=[-265.4, 10.9, -6.7])
g = tracker_display.geometry(area)
back = tracker_display.corners(g["width_mm"], g["height_mm"], g["gap_mm"], g["offset_mm"],
                               g["depth_mm"], g["tilt_deg"])
require(all(abs(a - b) < 0.05 for k in area for a, b in zip(area[k], back[k]))
        and abs(g["tilt_deg"] - 20.0) < 0.1 and abs(g["width_mm"] - 542.3) < 0.05,
        f"tracker-display geometry: corners -> gap {g['gap_mm']:.1f} mm, tilt "
        f"{g['tilt_deg']:.1f} deg -> the same corners")

print("6. feedback when the eye moves during a patch")
s = PerimetrySettings(participant_id="FB", strategy=FULL_THRESHOLD, extent=H24_2_CENTRAL,
                      screen_width_mm=302.0, viewing_distance_mm=334.0,
                      fixation_monitoring=FIX_TRACKER)
obs = Observer(healthy, random.Random(4), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
o = run_perimetry(s, obs, rng=random.Random(4), show_instructions=False,
                  fixation_monitor=ScriptedMonitor(lambda i: (6.0, 0.0) if i % 9 == 4 else (0, 0)))
msgs = getattr(obs, "messages", [])
require(o.counts["fixation_breaks"] > 0 and len(msgs) == o.counts["fixation_breaks"]
        == o.counts.get("fixation_feedback") and all("+" in m for m in msgs),
        f"every eye movement during a patch -> 'Keep looking at the +' "
        f"({len(msgs)} of {o.counts['fixation_breaks']} breaks)")
require(o.valid or not o.checks["fixation"], "the session still completes")
obs2 = Observer(healthy, random.Random(4), width_mm=302.0, dist_mm=334.0)
obs2.open(W, H, 0.5)
run_perimetry(PerimetrySettings(**{**s.__dict__, "fixation_feedback": False}), obs2,
              rng=random.Random(4), show_instructions=False,
              fixation_monitor=ScriptedMonitor(lambda i: (6.0, 0.0) if i % 9 == 4 else (0, 0)))
require(not getattr(obs2, "messages", []), "feedback can be switched off")

print("7. the bridge protocol (fake bridge)")
bridge = subprocess.Popen([sys.executable, str(HERE / "tools" / "tobii_bridge.py"), "--fake",
                           "--port", "5575", "--fake-distance-mm", "548"],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
try:
    time.sleep(1.0)
    m = TobiiMonitor("R")
    m._to_deg = lambda px, py: (0.0, 0.0)
    m._size = (1920, 1080)
    m._begin_bridge(port=5575)
    m.cal_enter()
    good = all(m.cal_point(x, y) for x, y in TRACKER_CAL_POINTS)
    res = m.cal_compute()
    m.cal_leave()
    require(good and res["ok"] and res["points"] == 5,
            f"enter, 5 points, compute, leave through the bridge: {res}")
    time.sleep(0.6)
    dist = m.eye_distance_mm()
    require(dist is not None and abs(dist - 548.0) < 0.1,
            f"the bridge sends the eye's distance: {dist} mm")
    da = m.display_area()
    require(da and abs(da["width_mm"] - 542.3) < 0.1 and abs(da["height_mm"] - 305.1) < 0.1,
            f"the bridge reports the tracker's screen: {da['width_mm']} x {da['height_mm']} mm")
    head = m.head_position()
    require(head and head["left"][0] < 0 < head["right"][0] and abs(head["left"][2] - 548) < 0.1,
            f"and both eyes' positions (mm, Tobii user coordinates): {head}")
    m.end()
finally:
    bridge.terminate()
    bridge.wait(5)

sys.exit(0 if ok else 1)
