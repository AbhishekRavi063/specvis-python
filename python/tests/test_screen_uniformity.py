"""The screen-evenness tool (tools/screen_uniformity.py): where the places and rings go,
and a whole run with a pretend meter and keys (ENTER, R, ESC) -> the CSV / JSON files."""

from __future__ import annotations

import csv
import json
import pathlib
import sys
import tempfile

PY = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PY), str(PY / "tools")]

from specvis.core import photometry  # noqa: E402
import screen_uniformity  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok    " if condition else "  FAIL  ") + what)
    ok = ok and bool(condition)


print("1. places: the same fractions of any screen")
fx, fy = photometry.place_fractions()
require(abs(fx - (0.5 - 30 / 542.3)) < 1e-12 and abs(fy - (0.5 - 30 / 305)) < 1e-12,
        f"default fractions {fx:.4f} x {fy:.4f} (30 mm in on the lab's 542.3 x 305 mm)")
for w, h in ((1920, 1080), (2560, 1440), (1280, 1024)):
    x, y = photometry.place_pixels("top-left", w, h, fx, fy)
    c = photometry.place_pixels("centre", w, h, fx, fy)
    b = photometry.place_pixels("bottom", w, h, fx, fy)
    r = photometry.place_pixels("right", w, h, fx, fy)
    require(abs(x + fx * w) < 1e-9 and abs(y - fy * h) < 1e-9 and c == (0.0, 0.0)
            and b == (0.0, -fy * h) and r == (fx * w, 0.0),
            f"{w}x{h}: top-left at ({x:.1f}, {y:.1f}) px from the centre, bottom and right too")
require(abs(fx * 1920 - 853.8) < 0.1 and abs(fy * 1080 - 433.8) < 0.1,
        "1920x1080: 30 mm in from each edge of the BenQ = 106 / 106 px")
gx, gy = photometry.place_fractions(inset_mm=40, width_mm=598.0, height_mm=336.4)
require(abs(gx - (0.5 - 40 / 598)) < 1e-12 and abs(gy - (0.5 - 40 / 336.4)) < 1e-12,
        "--inset-mm 40 on a 598 x 336.4 mm screen")
require(len(photometry.PLACES) == 9 and {"top", "bottom", "left", "right"} <= set(photometry.PLACES),
        "9 places, top and bottom included")

print("2. the ring stays inside the screen at every place")
for w_mm, h_mm in ((542.3, 305.0), (598.0, 336.4), (527.0, 296.4), (300.0, 200.0)):
    r = photometry.ring_radius_mm(60.0, w_mm, h_mm, photometry.PLACES, fx, fy)
    inside = all(abs(px) * w_mm + r <= w_mm / 2 and abs(py) * h_mm + r <= h_mm / 2
                 for px, py in (photometry.place_offset(p, fx, fy) for p in photometry.PLACES))
    require(inside and r <= 30.0, f"{w_mm} x {h_mm} mm: ring {2 * r:.0f} mm across, inside")

print("3. fitted map and colours")
pts = [(0, 0), (0, 120), (0, -120), (-240, 0), (240, 0), (-240, 120), (240, 120),
       (-240, -120), (240, -120)]
true = [-(x / 240) ** 2 * 8 - (y / 120) ** 2 * 6 for x, y in pts]
coef = photometry.fit_screen_map(pts, true)
require(abs(float(photometry.screen_map_at(coef, 240, 120)) - (-14)) < 1e-6,
        "a quadratic screen is fitted exactly")
require(photometry.fit_screen_map(pts[:5], true[:5]) is None, "fewer than 6 places: no fit")
require([photometry.evenness_colour(v) for v in (-4, -5, -9, -10, -12)]
        == ["green", "orange", "orange", "red", "red"], "green > -5, orange -5..-10, red <= -10")
require(photometry.unsteady([10.0, 10.1, 10.05]) == 0.0
        and photometry.unsteady([10.0, 10.5, 10.1]) > 2.0, "spread above 2 % is flagged")


class FakeUI:
    """Keys come from a list; the meter reads what the screen shows at the current place."""

    size = (1920.0, 1080.0)

    def __init__(self, keys):
        self.keys, self.place_xy, self.targets = list(keys), (0, 0), []

    def set_grey(self, level):
        self.level = level

    def set_ring_mm(self, mm):
        self.ring_mm = mm

    def show_target(self, message, x, y):
        self.place_xy = (x, y)
        self.targets.append(message.split("\n")[0])

    def wait_key(self):
        return self.keys.pop(0)

    def show_grey(self):
        pass

    def keep_alive(self, until=None, worker=None):
        if worker is not None:
            worker.join()


class FakeMeter:
    serial = "TEST"

    def __init__(self, ui):
        self.ui, self.calls = ui, 0

    def read(self):            # 25 cd/m2 in the middle, 12 % dimmer at the edges
        self.calls += 1
        x, y = self.ui.place_xy
        return 25.0 * (1 - 0.12 * ((x / 960) ** 2 + (y / 540) ** 2) / 1.0)


def logger(lines):
    return lambda *a, **k: lines.append(" ".join(str(x) for x in a))


print("4. a whole run: ENTER at every place, R once (into a folder not made yet)")
with tempfile.TemporaryDirectory() as tmp0:
    tmp = pathlib.Path(tmp0) / "output" / "uniformity"     # lab PC 24 Nov: the figure
    screen_uniformity.OUT_DIR = tmp                         # crashed when it was missing
    args = screen_uniformity.parse_args(["uniformity", "--simulate", "--label", "HP 527sf",
                                         "--note", "Brightness 100", "--settle", "0",
                                         "--width-mm", "598", "--figure",
                                         "--grid", "24-2 within 9"])
    keys = ["enter", "enter", "repeat", "enter"] + ["enter"] * 7
    ui = FakeUI(keys)
    meter = FakeMeter(ui)
    lines = []
    rc = screen_uniformity.run(args, meter, ui, dict(refresh_hz=100.0), log=logger(lines))
    csvs = list(pathlib.Path(tmp).glob("*.csv"))
    require(rc == 0 and len(csvs) == 1 and "HP_527sf_grey87" in csvs[0].name,
            f"one CSV: {csvs[0].name if csvs else None}")
    rows = list(csv.DictReader(csvs[0].open()))
    require(list(rows[0]) == ["place", "x_frac", "y_frac", "cd_m2", "min", "max", "sd",
                              "pct_vs_centre"], "CSV columns as asked")
    require([r["place"] for r in rows] == list(photometry.PLACES)
            and float(rows[0]["pct_vs_centre"]) == 0.0, "all 9 places, centre at 0 %")
    require(meter.calls == 10 * 3 and any("again" in t.lower() for t in ui.targets),
            "R measured the previous place (TOP) again: 10 places x 3 readings")
    info = json.loads(next(pathlib.Path(tmp).glob("*_summary.json")).read_text())
    require(info["label"] == "HP 527sf" and info["note"] == "Brightness 100"
            and info["meter_serial"] == "TEST" and info["screen"]["refresh_hz"] == 100.0
            and info["screen"]["width_mm"] == 598.0 and info["grey"] == 87
            and abs(info["largest_difference_pct"] - 12 * (fx ** 2 * 4 + fy ** 2 * 4)) < 0.01,
            f"summary: label, note, screen, meter, grey, largest {info['largest_difference_pct']} %")
    require(any("within +/-10 %" in s or "MORE than" in s for s in lines), "verdict printed")
    require(list(pathlib.Path(tmp).glob("*.png")) and list(pathlib.Path(tmp).glob("*.svg"))
            and "grid spots" in info.get("grid_estimate", ""),
            f"figure with the grid: {info.get('grid_estimate')}")

print("5. ESC stops cleanly and keeps what was measured")
with tempfile.TemporaryDirectory() as tmp:
    screen_uniformity.OUT_DIR = pathlib.Path(tmp)
    args = screen_uniformity.parse_args(["--simulate", "--settle", "0", "--label", "BenQ"])
    ui = FakeUI(["enter", "enter", "enter", "escape"])
    lines = []
    rc = screen_uniformity.run(args, FakeMeter(ui), ui, {}, log=logger(lines))
    files = sorted(p.name for p in pathlib.Path(tmp).iterdir())
    require(rc == 1 and len(files) == 2 and all("_PARTIAL" in f for f in files),
            f"_PARTIAL files: {files}")
    rows = list(csv.DictReader(next(pathlib.Path(tmp).glob("*.csv")).open()))
    info = json.loads(next(pathlib.Path(tmp).glob("*_summary.json")).read_text())
    require([r["place"] for r in rows] == ["centre", "top", "bottom"] and info["stopped_early"],
            "the 3 places measured before ESC are kept")

print("6. bad places are refused")
try:
    screen_uniformity.parse_args(["--places", "top,bottom"])
    require(False, "no centre -> refused")
except SystemExit:
    require(True, "no centre -> refused")
try:
    screen_uniformity.parse_args(["--places", "centre,middle"])
    require(False, "unknown place -> refused")
except SystemExit:
    require(True, "unknown place -> refused")

sys.exit(0 if ok else 1)
