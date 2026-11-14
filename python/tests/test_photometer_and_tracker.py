"""The ColorCAL readings -> luminance table and dB range, measured scales in the app, and
gaze from the Tobii bridge (Python 3.10 SDK -> UDP -> the app)."""

from __future__ import annotations

import collections
import json
import math
import pathlib
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.app import store  # noqa: E402
from specvis.core import photometry  # noqa: E402
from specvis.core.luminance import LuminanceCalibration  # noqa: E402
from specvis.procedure.fixation import FIX_TRACKER, TobiiMonitor  # noqa: E402
from specvis.procedure.perimetry import (FULL_THRESHOLD, H24_2, categorise,  # noqa: E402
                                         contrast_range_db, PerimetrySettings)
import sim_observer  # noqa: E402
from sim_observer import healthy, run  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok    " if condition else "  FAIL  ") + what)
    ok = ok and bool(condition)


def screen(level: float) -> float:           # an assumed screen: 0.3 to 300 cd/m2, gamma 2.2
    return 0.3 + 299.7 * (level / 255.0) ** 2.2


print("1. readings -> table and dB range")
coarse = photometry.coarse_levels()
first = photometry.background_level(coarse, [screen(v) for v in coarse], 10.0)
levels = sorted(set(coarse) | set(range(first - 8, first + 9)))
bg = photometry.background_level(levels, [screen(v) for v in levels], 10.0)
require(abs(screen(bg) - 10.0) == min(abs(screen(v) - 10.0) for v in range(256)),
        f"background 10 cd/m2 -> grey {bg}, the nearest level once the fine readings are in")
cds = [screen(v) for v in levels]
table = photometry.table_from_readings(levels, cds)
require(len(table) == 101 and abs(table[0] - 0.3) < 1e-9 and abs(table[-1] - 300.0) < 1e-9,
        "101-entry table from black to white")
worst = max(abs(table[b] - screen(b / 100 * 255)) / screen(b / 100 * 255) for b in range(101))
require(worst < 0.10, f"table within 10 % of the screen everywhere (worst {100 * worst:.1f} %)")
s = photometry.summarise(levels, cds, bg)
step = max(screen(bg + 1) - screen(bg), screen(bg) - screen(bg - 1)) / screen(bg)
require(abs(s["grey_step_contrast"] - step) < 1e-12,
        f"one grey step = {s['grey_step_db']:.1f} dB, from the readings around the background")
require(abs(s["strongest_contrast"] - (screen(bg) - 0.3) / screen(bg)) < 1e-12,
        f"strongest patch {s['strongest_db']:.2f} dB (limited by black)")
require(abs(s["middle_db_no_dither"] - (s["strongest_db"] + s["grey_step_db"]) / 2) < 1e-9,
        f"middle {s['middle_db_no_dither']:.1f} dB")
noisy = photometry.table_from_readings([0, 5, 10, 255], [0.31, 0.30, 0.35, 300.0])
require(all(b >= a for a, b in zip(noisy, noisy[1:])), "a dark reading that dips is held, "
        "never decreasing")
try:
    photometry.table_from_readings([10, 255], [1.0, 300.0])
    require(False, "readings must include black (0) and white (255)")
except ValueError:
    require(True, "readings must include black (0) and white (255)")
u = photometry.uniformity({"centre": 10.0, "top-left": 9.5, "right": 10.2})
require(abs(u["max_difference_pct"] - 5.0) < 1e-9, "uniformity: largest difference 5 %")

print("2. a measured scale in the app")
with tempfile.TemporaryDirectory() as tmp:
    scales = store.ScaleStore(pathlib.Path(tmp) / "scales.json")
    cal0 = LuminanceCalibration.from_table(table, "x")
    six = [float(cal0.luminance_at(b)) for b in (0, 20, 40, 60, 80, 100)]
    scales.put(store.LuminanceScale("S_1", "Lab", six, table=table))
    back = store.ScaleStore(pathlib.Path(tmp) / "scales.json").get("S_1")
    cal = back.calibration()
    require(back.table == table and cal.calibrated and "ColorCAL" in cal.source,
            "saved and loaded; the measured table is used, not the 6-point fit")
    import run_test_process  # noqa: E402
    settings, c = run_test_process.settings_and_calibration(
        {"settings": dict(PerimetrySettings().__dict__, zest=PerimetrySettings().zest.__dict__),
         "scale": json.loads(json.dumps(back.__dict__))})
    require(list(c.table) == table, "the test process uses the measured table too")
    r = contrast_range_db(PerimetrySettings(dithering=False), cal)
    require(abs(r["faintest_db"] - r["grey_step_db"]) < 1e-9 and r["calibrated"],
            f"app's range on the measured screen: {r['strongest_db']:.1f} to "
            f"{r['faintest_db']:.1f} dB, middle {r['middle_db']:.1f}")

print("3. gaze from the Tobii bridge")


def sender(port: int, stop: threading.Event, x=0.5, y=0.5, eye_valid=1):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while not stop.is_set():
        s.sendto(json.dumps({"t": 0, "l": [x, y, eye_valid], "r": [x, y, eye_valid],
                             "model": "FAKE", "serial": "1"}).encode(), ("127.0.0.1", port))
        time.sleep(1 / 200)
    s.close()


class Screen:
    size_px = (3024, 1964)


stop = threading.Event()
threading.Thread(target=sender, args=(5599, stop), daemon=True).start()
m = TobiiMonitor("R")
m._to_deg = lambda px, py: ((px - 1512) / 50.0, (982 - py) / 50.0)
m._size = Screen.size_px
m._begin_bridge(port=5599)
time.sleep(0.1)
g = m.current()
require(g is not None and abs(g[0]) < 1e-9 and abs(g[1]) < 1e-9 and "via bridge" in m.describe(),
        f"centre of the screen -> (0, 0) deg; {m.describe()}")
require(len(m.samples) > 5, f"samples keep arriving ({len(m.samples)})")
m.end()
stop.set()
time.sleep(0.05)
m2 = TobiiMonitor("R")
m2._to_deg, m2._size = (lambda px, py: (0.0, 0.0)), Screen.size_px
try:
    m2._begin_bridge(port=5598, wait_s=0.5)
    require(False, "no bridge running -> a clear error")
except RuntimeError as e:
    require("tracker-bridge" in str(e), "no bridge running -> a clear error telling what to start")

print("4. a whole staircase session with the eye tracker (fake bridge)")


class TimedObserver(sim_observer.Observer):
    """Each patch takes real time, as on a screen, so gaze samples arrive during it."""

    def present_train(self, *a, **k):
        time.sleep(0.04)
        return super().present_train(*a, **k)


sim_observer.Observer = TimedObserver
stop = threading.Event()
threading.Thread(target=sender, args=(5555, stop), daemon=True).start()
time.sleep(0.05)
o, _ = run(healthy, 31, settings_overrides=dict(strategy=FULL_THRESHOLD, extent=H24_2,
                                                screen_width_mm=302.0, viewing_distance_mm=240.0,
                                                fixation_monitoring=FIX_TRACKER,
                                                stimulus_ms=40.0))
stop.set()
cats = collections.Counter(categorise(o, p) for p in o.points)
require(o.valid and cats["seen"] == 52, f"valid, all 52 seen {dict(cats)}")
require(o.counts["gaze_checked"] > 0 and o.counts["fixation_breaks"] == 0
        and o.counts["blind_spot"] == 0,
        f"every patch checked by gaze ({o.counts['gaze_checked']}), no breaks, no blind-spot "
        "checks needed")
require("bridge" in o.fixation_monitor, f"recorded: {o.fixation_monitor}")

sys.exit(0 if ok else 1)
