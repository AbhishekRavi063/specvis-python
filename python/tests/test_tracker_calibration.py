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
from sim_observer import Observer, W, H, healthy, right_hemianopia  # noqa: E402

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
                 corners=None, eyes=None, boxes=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
        super().__init__()
        self.error, self.cal_ok, self.distance, self.area = error, cal_ok, distance, area
        self.corners, self.eyes, self.boxes = corners, eyes, boxes
        self.cal_eye = "not asked"
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
        boxes = dict(left_box=self.boxes[0], right_box=self.boxes[1]) if self.boxes else {}
        if self.eyes:
            return dict(left=self.eyes[0], right=self.eyes[1], distance_mm=self.distance,
                        **boxes)
        return dict(left=(-32.0, 170.0, self.distance), right=(32.0, 170.0, self.distance),
                    distance_mm=self.distance, **boxes)

    def cal_enter(self, eye=None):
        self.calls.append("enter")
        self.cal_eye = eye

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
require("red" not in colours and o.counts.get("blinks_between", 0) > 0,
        "gaze lost for longer than 100 ms (a blink) restarts the hold but is not 'looking "
        f"away': the + stays white ({o.counts.get('blinks_between')} blinks)")


class Settle(Tracker):
    """On the + for 60 samples, then lost for 15 (240 ms, a blink), again and again."""
    n = 0

    def _read(self):
        Settle.n += 1
        return None if Settle.n % 75 >= 60 else (0.0, 0.0)


def polls_before_patches(settle_ms):
    Settle.n = 0
    t = Settle()
    s = PerimetrySettings(participant_id="ST", strategy=FULL_THRESHOLD, extent=H24_2_CENTRAL,
                          screen_width_mm=302.0, viewing_distance_mm=334.0,
                          screen_height_mm=302.0 * 1964 / 3024, fixation_monitoring=FIX_TRACKER,
                          post_blink_delay_ms=settle_ms)
    spy = ColourSpy(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
    spy.open(W, H, 0.5)
    out = run_perimetry(s, spy, rng=random.Random(3), show_instructions=False, fixation_monitor=t)
    return Settle.n / max(1, out.counts["presentations"])


a, b = polls_before_patches(0.0), polls_before_patches(300.0)
require(b > a, f"after a blink the next patch waits longer with a 300 ms settle delay "
               f"({a:.1f} -> {b:.1f} gaze polls per patch)")
from specvis.procedure.fixation import FixationMonitor as _FM  # noqa: E402
m = _FM()
for t_, g_ in ((0.000, (0, 0)), (0.004, (0, 0)), (0.008, None), (0.012, None), (0.150, (0, 0)),
               (0.154, None), (0.158, (0, 0)), (0.200, (0, 0))):
    m._add(t_, g_)
require(abs(m.longest_gap_ms(0.0, 0.2) - 146.0) < 0.5 and m.longest_gap_ms(0.16, 0.2) == 0.0,
        f"longest gap in a patch: 146 ms (a blink) vs single lost samples "
        f"({m.longest_gap_ms(0.0, 0.2):.0f} ms)")


print("5d. gentler fixation check (5-deg preset), blinks, positioning, true distance")
from specvis.procedure.fixation import distance_to_screen, judge_patch, screen_plane  # noqa: E402


class Flicky(Tracker):
    """On the +, but every 20th sample 5 deg off (noise / a quick flick)."""
    n = 0

    def _read(self):
        Flicky.n += 1
        return (5.0, 0.0) if Flicky.n % 20 == 0 else (0.0, 0.0)


def flick_session(**over):
    s = PerimetrySettings(participant_id="FL", strategy=FULL_THRESHOLD, extent=H24_2_CENTRAL,
                          screen_width_mm=302.0, viewing_distance_mm=334.0,
                          screen_height_mm=302.0 * 1964 / 3024, fixation_monitoring=FIX_TRACKER,
                          **over)
    spy = ColourSpy(healthy, random.Random(5), width_mm=302.0, dist_mm=334.0)
    spy.open(W, H, 0.5)
    run_perimetry(s, spy, rng=random.Random(5), show_instructions=False,
                  fixation_monitor=Flicky())
    return getattr(spy, "colours", [])


require("red" in flick_session(), "strict (defaults): a single 16-ms flick turns the + red")
require("red" not in flick_session(fixation_off_grace_ms=150.0, fixation_break_max_out_frac=0.5),
        "gentle (150 ms grace): short flicks are ignored, the + stays white")
on, off = (0.5, 0.0), (4.0, 0.0)
require(judge_patch([on] * 9 + [off], 3.0)[0] is False
        and judge_patch([on] * 9 + [off], 3.0, max_out_frac=0.25)[0] is True
        and judge_patch([on] * 6 + [off] * 4, 3.0, max_out_frac=0.25)[0] is False,
        "patch judged: 1 sample in 10 off is a break only in strict mode; 4 in 10 always")


class Blinker(Tracker):
    def patch_shown(self, t0, t1):
        self.look = (0.0, 0.0)


blinky = ScriptedMonitor(lambda i: None if i % 7 == 3 else (0.0, 0.0))
obs = Observer(healthy, random.Random(6), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
o = run_perimetry(PerimetrySettings(participant_id="BL", strategy=FULL_THRESHOLD,
                                    extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                    viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER),
                  obs, rng=random.Random(6), show_instructions=False, fixation_monitor=blinky)
require(o.counts.get("blinks", 0) > 0 and o.counts["fixation_breaks"] == 0
        and not getattr(obs, "messages", []),
        f"blinks during a patch ({o.counts.get('blinks')}): repeated, not counted as looking "
        "away, no 'Keep looking' message")

obs = Observer(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
high = (eye(-32.0)[0], eye(-32.0)[1] + 100 * math.cos(t20), eye(-32.0)[2] + 100 * math.sin(t20))
high_r = (eye(32.0)[0], eye(32.0)[1] + 100 * math.cos(t20), eye(32.0)[2] + 100 * math.sin(t20))
o = run_perimetry(PerimetrySettings(participant_id="POS3", strategy=FULL_THRESHOLD,
                                    extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                    viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER),
                  obs, rng=random.Random(3), show_instructions=True,
                  fixation_monitor=Tracker(distance=334.0, area=(542.3, 305.0), corners=lab,
                                           eyes=(high, high_r)))
text, marks = obs.statuses[0]
require("too high" in text and "Position good" not in text
        and all(m.colour == "orange" for m in marks if m.label)
        and o.gaze_check.get("head_position_accepted_off") is True,
        "eyes 10 cm above the screen centre: orange, 'lower the chin rest', and going on anyway "
        "is recorded")
plane = screen_plane(lab)
require(abs(distance_to_screen(eye(0.0), plane) - 510.0) < 0.5
        and abs(distance_to_screen(eye(0.0), None) - eye(0.0)[2]) < 1e-9
        and eye(0.0)[2] - 510.0 > 10,
        f"eye distance square on to the tilted screen: 510 mm (the tracker's z reads "
        f"{eye(0.0)[2]:.0f})")


class AlwaysBlinky(ScriptedMonitor):
    """Every patch has a 150-ms gap in the tracked eye."""
    def longest_gap_ms(self, t0, t1):
        return 150.0


obs = Observer(right_hemianopia, random.Random(8), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
o = run_perimetry(PerimetrySettings(participant_id="BG", strategy=FULL_THRESHOLD,
                                    extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                    viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER,
                                    blink_min_ms=100.0),
                  obs, rng=random.Random(8), show_instructions=False,
                  fixation_monitor=AlwaysBlinky(lambda i: (0.0, 0.0)))
notes = [r.get("note", "") for r in o.trial_rows]
require(o.counts.get("blinks", 0) > 0 and any(str(n).startswith("BLINK") for n in notes)
        and all(p.zest.finished() for p in o.points) and o.counts["fixation_breaks"] == 0,
        f"a not-seen patch with a blink-long gap is repeated ({o.counts.get('blinks')} times, "
        "'BLINK' in trials.csv), at most 3 times per point, so blind points still finish")
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

print("6b. lab report 16 Nov: watching the eye after the patch, head box, lids, clock, pauses")
from specvis.procedure.fixation import ClockAlign, LiveFixation  # noqa: E402

live = LiveFixation(3.0, 150.0, 150.0)
seq = [(0.000, (0, 0)), (0.016, (12, 0)), (0.100, (12, 0)), (0.200, (12, 0)),
       (0.216, None), (0.300, None), (0.400, None), (0.416, (0.5, 0))]
states = [live.update(t, g) for t, g in seq]
require(states == ["on", "on", "on", "off", "off", "off", "unknown", "on"],
        f"live +: a look away turns it red after the grace time, a long gap is 'unknown', "
        f"never red; back on the + -> white: {states}")


class AfterPatchLook(ScriptedMonitor):
    """On the + during every patch; then the eye goes 12 deg away for the next 8 looks
    (while the person answers), and comes back."""
    def __init__(self):
        super().__init__(lambda i: (0.0, 0.0))
        self.away = 0

    def patch_shown(self, t0, t1):
        super().patch_shown(t0, t1)
        self.away = 8

    def current(self):
        g = (12.0, 0.0) if self.away > 0 else (0.0, 0.0)
        self.away = max(0, self.away - 1)
        self._add(time.perf_counter(), g)
        return g


class FrameObserver(Observer):
    """A screen with a frame loop: on_frame is called 12 times per wait, and the colour of
    the + is noted each time."""
    def wait_for_key_watching(self, keys, timeout_ms, fixation=None, clear_events=True,
                              on_frame=None):
        for _ in range(12):
            if on_frame is not None:
                on_frame()
                self.colours.append(self.fixation_colour)
            time.sleep(0.0005)
        return self.wait_for_key(keys, timeout_ms, fixation, clear_events)


obs = FrameObserver(healthy, random.Random(5), width_mm=302.0, dist_mm=334.0)
obs.colours = []
obs.fixation_colour = "white"          # a screen that can colour the +
obs.open(W, H, 0.5)
o = run_perimetry(PerimetrySettings(participant_id="AFTER", strategy=FULL_THRESHOLD,
                                    extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                    viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER),
                  obs, rng=random.Random(5), show_instructions=False,
                  fixation_monitor=AfterPatchLook())
after = [r for r in o.trial_rows if r.get("look_away_after_ms") not in ("", None)]
require("red" in obs.colours and obs.colours[-1] == "white",
        "the + turns red while the person answers with the eye off it, and white again "
        "when it is back")
require(o.counts.get("look_aways_after_patch", 0) > 0 and o.counts["fixation_breaks"] == 0
        and all(p.zest.finished() for p in o.points),
        f"look-aways after the patch are counted ({o.counts.get('look_aways_after_patch')}) "
        "but the answers still count (no fixation breaks)")
require(after and max(float(r["gaze_max_after_deg"]) for r in after) > 11.9
        and max(float(r["look_away_after_ms"]) for r in after) > 0,
        "trials.csv: gaze_max_after_deg and look_away_after_ms for each patch")

obs = Observer(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
o = run_perimetry(PerimetrySettings(participant_id="NOBOX", strategy=FULL_THRESHOLD,
                                    extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                    viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER),
                  obs, rng=random.Random(3), show_instructions=True,
                  fixation_monitor=Tracker(distance=334.0, boxes=None))
text, marks = obs.statuses[0]
require("Position good" not in text and "does not say where the head is" in text
        and all(m.colour == "orange" for m in marks if m.label)
        and o.gaze_check.get("head_position_accepted_off") is True,
        "no head-box data from the tracker -> NOT 'good' (the edge check could not run), "
        "with a hint to restart the bridge")
obs = Observer(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
o = run_perimetry(PerimetrySettings(participant_id="EDGE", strategy=FULL_THRESHOLD,
                                    extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                    viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER),
                  obs, rng=random.Random(3), show_instructions=True,
                  fixation_monitor=Tracker(distance=334.0,
                                           boxes=((0.95, 0.5, 0.5), (0.9, 0.5, 0.5))))
text, _ = obs.statuses[0]
require("edge of the tracker's view" in text and "Position good" not in text,
        "eyes at the edge of the tracker's view -> not good, 'centre the head'")
obs = Observer(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
t = Tracker(distance=334.0)
o = run_perimetry(PerimetrySettings(participant_id="OK", strategy=FULL_THRESHOLD,
                                    extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                    viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER),
                  obs, rng=random.Random(3), show_instructions=True, fixation_monitor=t)
require("Position good" in obs.statuses[0][0] and t.cal_eye is None,
        "box data in the middle -> 'Position good'; calibration of both eyes by default")
obs = Observer(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
t = Tracker(distance=334.0)
o = run_perimetry(PerimetrySettings(participant_id="MONO", strategy=FULL_THRESHOLD,
                                    extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                    viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER,
                                    tracker_monocular_calibration=True, eye="R"),
                  obs, rng=random.Random(3), show_instructions=True, fixation_monitor=t)
require(t.cal_eye == "R", "tracker_monocular_calibration: only the tested eye is calibrated")


class LidMonitor(AlwaysBlinky):
    """Every patch has a 150-ms gap; the lid during it is ``lid`` mm (open = 10 mm)."""
    def __init__(self, lid):
        super().__init__(lambda i: (0.0, 0.0))
        self.lid = lid

    def openness_between(self, t0, t1):
        return [10.0, self.lid, 10.0]

    def typical_openness(self):
        return 10.0


for lid, expect in ((9.5, "TRACKING_LOST"), (1.0, "BLINK")):
    obs = Observer(right_hemianopia, random.Random(8), width_mm=302.0, dist_mm=334.0)
    obs.open(W, H, 0.5)
    o = run_perimetry(PerimetrySettings(participant_id="LID", strategy=FULL_THRESHOLD,
                                        extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                        viewing_distance_mm=334.0,
                                        fixation_monitoring=FIX_TRACKER, blink_min_ms=100.0),
                      obs, rng=random.Random(8), show_instructions=False,
                      fixation_monitor=LidMonitor(lid))
    notes = [str(r.get("note", "")) for r in o.trial_rows]
    other = "BLINK" if expect == "TRACKING_LOST" else "TRACKING_LOST"
    require(any(n.startswith(expect) for n in notes)
            and not any(n.startswith(other) for n in notes)
            and o.counts["fixation_breaks"] == 0
            and all(p.zest.finished() for p in o.points)
            and any(r.get("eye_openness_min_mm") == lid for r in o.trial_rows),
            f"a gap with the lid at {lid} mm (open 10) -> {expect}, repeated, never a "
            "fixation break; eye_openness_min_mm logged")

clk = ClockAlign()
jitter = [0.003, 0.010, 0.002, 0.030, 0.002, 0.005]
stamps = [clk.stamp(1000.0 + 0.004 * i, 50.0 + 0.004 * i + j) for i, j in enumerate(jitter)]
require(all(abs(s_ - (50.0 + 0.004 * i + 0.002)) < 1e-6 for i, s_ in enumerate(stamps[4:], 4))
        and abs(stamps[3] - (50.0 + 0.012 + 0.002)) < 1e-6,
        "tracker time stamps: a sample delayed 30 ms on the way is still put at the moment "
        "it was taken (smallest delay seen)")
clk = ClockAlign()
require([clk.stamp(0.0, 5.0), clk.stamp(0.0, 5.1)] == [5.0, 5.1],
        "a tracker time that does not move on -> the arrival time is used")


class Troubled(ScriptedMonitor):
    def __init__(self):
        super().__init__(lambda i: (0.0, 0.0))
        self.trouble = 2

    def problem(self):
        if self.trouble:
            self.trouble -= 1
            return "the eye tracker lost its connection"
        return None


obs = Observer(healthy, random.Random(2), width_mm=302.0, dist_mm=334.0)
obs.open(W, H, 0.5)
o = run_perimetry(PerimetrySettings(participant_id="LOST", strategy=FULL_THRESHOLD,
                                    extent=H24_2_CENTRAL, screen_width_mm=302.0,
                                    viewing_distance_mm=334.0, fixation_monitoring=FIX_TRACKER),
                  obs, rng=random.Random(2), show_instructions=False, fixation_monitor=Troubled())
paused = [sc for sc in obs.screens if sc.startswith("PAUSED: the eye tracker lost")]
require(len(paused) == 2 and o.counts.get("tracker_pauses") == 2
        and all(p.zest.finished() for p in o.points),
        "tracker connection lost -> the test pauses with a message until it is back")

print("6c. lab report 16-17 Nov: one-eye calibration answers, settle times, early warning, map words")
import importlib.util  # noqa: E402
from specvis.procedure.fixation import tobii_calibration_ok  # noqa: E402
from specvis.procedure.perimetry import H24_2_9, FIXATION_BREAK  # noqa: E402
from specvis.procedure import maps  # noqa: E402

_spec = importlib.util.spec_from_file_location("tobii_bridge", HERE / "tools" / "tobii_bridge.py")
tobii_bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tobii_bridge)
SDK = tobii_bridge.FakeSDK
for name, check in (("bridge", tobii_bridge.calibration_ok), ("app", tobii_calibration_ok)):
    require(check(SDK, SDK.CALIBRATION_STATUS_SUCCESS, None)
            and check(SDK, SDK.CALIBRATION_STATUS_SUCCESS_RIGHT_EYE, SDK.SELECTED_EYE_RIGHT)
            and check(SDK, SDK.CALIBRATION_STATUS_SUCCESS_LEFT_EYE, SDK.SELECTED_EYE_LEFT)
            and not check(SDK, SDK.CALIBRATION_STATUS_SUCCESS_LEFT_EYE, SDK.SELECTED_EYE_RIGHT)
            and not check(SDK, SDK.CALIBRATION_STATUS_SUCCESS_RIGHT_EYE, SDK.SELECTED_EYE_LEFT)
            and not check(SDK, SDK.CALIBRATION_STATUS_SUCCESS_RIGHT_EYE, None)
            and not check(SDK, SDK.CALIBRATION_STATUS_FAILURE, SDK.SELECTED_EYE_RIGHT),
            f"{name}: one-eye calibration 'success, right eye' is a success for the right eye "
            "only; 'success, left eye' for a right-eye calibration is not")


class PollCounter(ScriptedMonitor):
    """Gaze from a script; counts the polls since the last key screen and before each patch."""
    def __init__(self, gaze_for_patch=lambda i: (0.0, 0.0)):
        super().__init__(gaze_for_patch)
        self.polls = 0
        self.since_screen = None
        self.after_screen = []

    def current(self):
        self.polls += 1
        if self.since_screen is not None:
            self.since_screen += 1
        return super().current()

    def patch_shown(self, t0, t1):
        if self.since_screen is not None:
            self.after_screen.append(self.since_screen)
            self.since_screen = None
        super().patch_shown(t0, t1)


class ScreenSpy(Observer):
    """Tells the monitor when a screen closed with a key."""
    def show_screen(self, text, marks=(), keys=("space",)):
        out = super().show_screen(text, marks, keys)
        self.monitor.since_screen = 0
        return out


def resume_polls(resume_ms):
    mon = PollCounter()
    spy = ScreenSpy(healthy, random.Random(3), width_mm=302.0, dist_mm=334.0)
    spy.monitor = mon
    spy.open(W, H, 0.5)
    run_perimetry(PerimetrySettings(participant_id="RES", strategy=FULL_THRESHOLD, extent=H24_2_9,
                                    screen_width_mm=302.0, viewing_distance_mm=334.0,
                                    screen_height_mm=302.0 * 1964 / 3024,
                                    fixation_monitoring=FIX_TRACKER, resume_delay_ms=resume_ms),
                  spy, rng=random.Random(3), show_instructions=True, fixation_monitor=mon)
    return mon.after_screen


a, b = resume_polls(0.0), resume_polls(1200.0)
require(a and b and b[0] - a[0] >= 1200 / 16.0 - 1,
        f"after a key screen the first patch waits resume_delay_ms more with the eye on the + "
        f"({a[0]} -> {b[0]} gaze polls of 16 ms)")


def settle_polls(settle_ms):
    mon = PollCounter(lambda i: (6.0, 0.0) if i % 9 == 4 else (0.0, 0.0))
    obs = Observer(healthy, random.Random(4), width_mm=302.0, dist_mm=334.0)
    obs.open(W, H, 0.5)
    o = run_perimetry(PerimetrySettings(participant_id="LAS", strategy=FULL_THRESHOLD,
                                        extent=H24_2_9, screen_width_mm=302.0,
                                        viewing_distance_mm=334.0,
                                        fixation_monitoring=FIX_TRACKER,
                                        look_away_settle_ms=settle_ms),
                      obs, rng=random.Random(4), show_instructions=False, fixation_monitor=mon)
    return mon.polls, o


(pa, oa), (pb, ob) = settle_polls(0.0), settle_polls(600.0)
breaks = ob.counts["fixation_breaks"]
require(breaks > 0 and pb - pa >= breaks * (600 / 16.0 - 1)
        and all(p.zest.finished() for p in ob.points),
        f"after a look away the next patch waits look_away_settle_ms with the eye back "
        f"({breaks} breaks: {pa} -> {pb} gaze polls)")
Flicky.n = 0
flick_obs = ColourSpy(healthy, random.Random(5), width_mm=302.0, dist_mm=334.0)
flick_obs.open(W, H, 0.5)
fo = run_perimetry(PerimetrySettings(participant_id="FL2", strategy=FULL_THRESHOLD,
                                     extent=H24_2_9, screen_width_mm=302.0,
                                     viewing_distance_mm=334.0,
                                     screen_height_mm=302.0 * 1964 / 3024,
                                     fixation_monitoring=FIX_TRACKER, look_away_settle_ms=600.0,
                                     resume_delay_ms=1200.0),
                   flick_obs, rng=random.Random(5), show_instructions=False,
                   fixation_monitor=Flicky())
require(all(p.zest.finished() for p in fo.points),
        "strict mode with a 16-ms flick every 20 samples still finishes (the settle is asked "
        "once per patch, so flicks cannot stall it)")


class EarlyEnd(Observer):
    """A screen with a frame loop that honours on_frame() returning True: the wait ends."""
    ended = 0

    def wait_for_key_watching(self, keys, timeout_ms, fixation=None, clear_events=True,
                              on_frame=None):
        end = time.perf_counter() + 0.08
        while on_frame is not None and time.perf_counter() < end:
            if on_frame() is True:
                self.ended += 1
                return None, None
            time.sleep(0.002)
        return self.wait_for_key(keys, timeout_ms, fixation, clear_events)


def early_run(gaze):
    obs_ = EarlyEnd(healthy, random.Random(6), width_mm=302.0, dist_mm=334.0)
    obs_.open(W, H, 0.5)
    out = run_perimetry(PerimetrySettings(participant_id="EARLY", strategy=FULL_THRESHOLD,
                                          extent=H24_2_9, screen_width_mm=302.0,
                                          viewing_distance_mm=334.0,
                                          fixation_monitoring=FIX_TRACKER),
                        obs_, rng=random.Random(6), show_instructions=False,
                        fixation_monitor=ScriptedMonitor(gaze))
    return obs_, out


early_obs, eo = early_run(lambda i: (6.0, 0.0) if i % 7 == 3 else (0.0, 0.0))
fb = [r for r in eo.trial_rows if str(r.get("note", "")).startswith(FIXATION_BREAK)]
# A patch answered while it was still up has no answer window to end, so not every
# break ends one; every early end must be a break.
require(0 < early_obs.ended <= eo.counts["fixation_breaks"] == len(fb)
        == len(getattr(early_obs, "messages", []))
        and all(p.zest.finished() for p in eo.points),
        f"an eye movement during the patch ends the answer window at once "
        f"({early_obs.ended} early ends of {len(fb)} breaks); every break is still logged "
        "FIXATION_BREAK, warned and repeated")
calm_obs, co = early_run(lambda i: (0.0, 0.0))
require(calm_obs.ended == 0 and co.counts["fixation_breaks"] == 0,
        "the eye on the + -> the answer window is never cut short")

no_table = maps._names({"prior_table": "not used (pilot mode)", "settings": {"weak_contrast": 0.2}})
with_table = maps._names({"prior_table": "normal table, 12 sessions", "settings": {}})
require(no_table["seen"] == "seen at 20% contrast or less"
        and no_table["weak"] == "needed more than 20% contrast"
        and maps._names({"prior_table": "none"})["seen"] == no_table["seen"]
        and with_table["seen"] == "seen (normal)" and with_table["weak"].startswith("weak")
        and no_table["blind"] == with_table["blind"] == "not seen at maximum contrast",
        "map legend: without a normal table it says the contrast cut-off, with one "
        "'normal / worse than normal'; never 'blind'")

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
    require(head["left_box"] == (0.55, 0.5, 0.5) and head["right_box"] == (0.45, 0.5, 0.5),
            "and where each eye is in the tracker's view (user position guide)")
    m.cal_enter("R")
    res = m.cal_compute()
    m.cal_leave()
    require(m.cal_mode == "right eye only" and res.get("mode") == "right eye only",
            f"monocular calibration of the tested eye through the bridge: {res}")
    m.cal_enter("R")
    good = all(m.cal_point(x, y) for x, y in TRACKER_CAL_POINTS)
    res = m.cal_compute()
    require(good and res["ok"] and res["status"] == "calibration_status_success_right_eye",
            f"one-eye calibration: 'success, right eye' answers are accepted (lab 16 Nov): {res}")
    m._cmd("fake_cal_status", status="calibration_status_success_left_eye")
    require(not m.cal_point(0.5, 0.5) and not m.cal_compute()["ok"],
            "a 'success, left eye' answer to a right-eye calibration is not a success")
    m._cmd("fake_cal_status", status=None)
    m.cal_leave()
    m.cal_enter("L")
    res = m.cal_compute()
    m.cal_leave()
    require(res["ok"] and res["status"] == "calibration_status_success_left_eye",
            f"and the left eye's: {res}")
    m._cmd("fake_lid", mm=3.5)
    time.sleep(0.3)
    lids = m.openness_between(time.perf_counter() - 0.1, time.perf_counter())
    require(lids and lids[-1] == 3.5, f"the bridge sends the lid opening: {lids[-3:]}")
    require(m.problem() is None, "no problem while gaze arrives")
    m._cmd("fake_event", event="connection_lost")
    time.sleep(0.2)
    require(m.problem() and "lost" in m.problem()
            and any(e["event"] == "connection_lost" for e in m.events),
            f"a tracker notification arrives and is kept: {m.events}")
    m._cmd("fake_event", event="connection_restored")
    time.sleep(0.2)
    require(m.problem() is None, "connection restored -> no problem")
    require(m.clock.offset_s is not None, "samples are timed by the tracker's own clock")
    m.end()
finally:
    bridge.terminate()
    bridge.wait(5)

print("\nALL OK" if ok else "\nFAILURES")
sys.exit(0 if ok else 1)
