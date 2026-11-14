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

from specvis.procedure.fixation import FIX_TRACKER, FixationMonitor, TobiiMonitor  # noqa: E402
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

    def __init__(self, error=0.3, cal_ok=True):
        super().__init__()
        self.error, self.cal_ok = error, cal_ok
        self.look = (0.0, 0.0)
        self.calls = []

    def target_shown(self, x, y):
        self.look = (x, y)

    def _read(self):
        return (self.look[0] + self.error, self.look[1])

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
        and [(p[1], p[2]) for p in pts] == [tuple(map(float, q)) for q in TRACKER_CAL_POINTS],
        "Tobii 5-point calibration: centre and 4 points in from the corners, then compute")
v = o.gaze_check.get("validation", {})
require(v.get("passed") and len(v.get("points", [])) == 5 and abs(v["centre_error_deg"] - 0.3) < 0.05,
        f"accuracy measured at 5 dots: centre {v.get('centre_error_deg')} deg, "
        f"mean {v.get('mean_error_deg')}")
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

print("5. the bridge protocol (fake bridge)")
bridge = subprocess.Popen([sys.executable, str(HERE / "tools" / "tobii_bridge.py"), "--fake",
                           "--port", "5575"], stdout=subprocess.DEVNULL,
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
    m.end()
finally:
    bridge.terminate()
    bridge.wait(5)

sys.exit(0 if ok else 1)
