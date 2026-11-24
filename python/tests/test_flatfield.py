"""Flat-field (screen evenness) correction: the surface through the 9 places, the static
dithered background, patches keeping their contrast, the map store and import, and
nothing drawn differently when no map is chosen."""

from __future__ import annotations

import csv
import math
import pathlib
import random
import sys
import tempfile

import numpy as np

PY = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PY), str(PY / "tools"), str(PY / "tests")]

from specvis.core import flatfield, photometry  # noqa: E402
from specvis.core.gabor import GaborParameters, render_brightness, snap_background  # noqa: E402
from specvis.core.luminance import TABLE_SIZE, LuminanceCalibration  # noqa: E402
from specvis.procedure.perimetry import (H24_2_9, PerimetrySettings, grid_points,  # noqa: E402
                                         results_text, _meta)
from specvis.stimulus.backend import NullBackend, Texture  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok    " if condition else "  FAIL  ") + what)
    ok = ok and bool(condition)


# The lab's first map: median of 4 BenQ maps, grey 87, 24 Nov 2026.
LAB = {"centre": 0.0, "top": -12.28, "bottom": -9.73, "left": -9.62, "right": -6.24,
       "top-left": -9.89, "top-right": -10.75, "bottom-left": -10.82, "bottom-right": -4.54}
FX, FY = photometry.place_fractions()


def lab_map(**kw) -> flatfield.FlatFieldMap:
    places = []
    for p, pct in LAB.items():
        x, y = photometry.place_offset(p, FX, FY)
        places.append(dict(place=p, x_frac=x, y_frac=y, pct=pct))
    return flatfield.FlatFieldMap(id="FF_test", name="BenQ median 24 Nov", places=places,
                                  monitor="BenQ_XL2546K", **kw)


def fake_scale(k: float = 106.2 / 255 ** 1.9) -> LuminanceCalibration:
    """L = k * grey^1.9 (grey 0-255), as a 101-entry table over brightness 0-100 %."""
    grey = np.arange(TABLE_SIZE) / 100.0 * 255.0
    return LuminanceCalibration.from_table(k * grey ** 1.9, "fake gamma-1.9 scale")


print("1. the surface")
m = lab_map()
s = m.surface()
pts = [(p["x_frac"], p["y_frac"]) for p in m.places]
err = max(abs(float(s(x, y)) - LAB[p["place"]]) for (x, y), p in zip(pts, m.places))
require(s.kind == "exact 3x3 biquadratic" and err < 1e-6,
        f"exact biquadratic through all 9 places (largest miss {err:.1e} %)")
require(abs(float(s(0, 0))) < 1e-9, "0 at the centre")
require(abs(float(s(0.5, 0.5)) - LAB["top-right"]) < 1e-9
        and abs(float(s(0, -0.5)) - LAB["bottom"]) < 1e-9
        and abs(float(s(-0.5, 0.2)) - float(s(-FX, 0.2))) < 1e-9,
        "held constant beyond the outer places (no overshoot at the screen edge)")
six = flatfield.Surface([x for x, _ in pts[:7]], [y for _, y in pts[:7]],
                        [LAB[p["place"]] for p in m.places[:7]])
require(six.kind == "6-term fit" and abs(float(six(0, 0))) < 1e-9,
        "with 7 places: the 6-term fit, through the centre")
try:
    flatfield.Surface([0, 0.1, 0.2], [0, 0.1, 0.2], [0, -1, -2])
    require(False, "fewer than 6 places refused")
except ValueError:
    require(True, "fewer than 6 places refused")
require(abs(float(flatfield.gain(-10.0, 0.8)) - 1 / 0.92) < 1e-12
        and float(flatfield.gain(-90.0, 1.0)) == 2.0,
        "gain 1 / (1 + s m / 100), limited to x2")

print("2. the background")
cal = fake_scale()
bg_cd, bg = snap_background(cal, 10.0)
W, H = 1920, 1080
ff = flatfield.FlatField(m, cal, bg, W, H)
levels = ff.background_levels()
target = ff.target_levels()
for name, (r, c) in (("centre", (H // 2 - 20, W // 2 - 20)), ("top-left corner", (30, 30)),
                     ("bottom-right corner", (H - 70, W - 70))):
    blk = (slice(r, r + 40), slice(c, c + 40))
    d = float(levels[blk].mean() - target[blk].mean())
    require(abs(d) < 0.1, f"{name}: mean grey within 0.1 step of the target ({d:+.3f})")
require(np.all(levels == np.round(levels)) and levels.shape == (H, W),
        "whole grey levels, one per pixel, row 0 = top")
require(abs(float(target[H // 2, W // 2]) - bg * 255) < 0.02
        and float(target[5, W // 2]) > float(target[H // 2, W // 2]),
        "the centre stays at the background; the dimmer top is drawn brighter")
lum = cal.luminance_at(levels / 255.0 * 100.0)
plain = cal.luminance_at(bg * 100.0)
top = float(lum[100:140, 940:980].mean() / plain - 1)
require(abs(top - (float(ff.gain[120, 960]) - 1)) < 0.01,
        f"under the top place the screen gives +{100 * top:.1f} % (planned "
        f"+{100 * (float(ff.gain[120, 960]) - 1):.1f} %)")
again = flatfield.FlatField(m, cal, bg, W, H).background_levels()
require(np.array_equal(levels, again) and ff.background_levels() is levels,
        "static: the same dither every time (fixed seed), computed once")
other = flatfield.FlatField(m, cal, bg, W, H, seed=2).background_levels()
require(not np.array_equal(levels, other), "another seed, another dither")
nod = flatfield.FlatField(m, cal, bg, W, H, dither=False).background_levels()
require(np.all(nod == np.round(target)), "without dither: the target rounded")
try:
    flatfield.FlatField(m, LuminanceCalibration.uncalibrated(), 0.5, W, H)
    require(False, "an uncalibrated scale is refused")
except ValueError:
    require(True, "an uncalibrated scale is refused")

print("3. the patches keep their contrast")
ppd = W / 542.3 * 570 * math.tan(math.radians(1))           # px per degree near the centre
d_mm, px_mm = 570.0, W / 542.3
kept, _ = grid_points(PerimetrySettings(extent=H24_2_9, eye="R"))
edge = max(kept, key=lambda p: (abs(p[0]), abs(p[1])))
for name, (xd, yd) in (("centre", (0, 0)), (f"edge spot {edge}", edge)):
    centre = (W / 2 + d_mm * math.tan(math.radians(xd)) * px_mm,
              H / 2 - d_mm * math.tan(math.radians(yd)) * px_mm)
    g = GaborParameters(2.0, 1.0, 45.0, 0.0, 2.0 / 6, bg_cd, 0.4)
    b = render_brightness(g, cal, ppd, ppd)
    corr = ff.correct_patch(b, centre)
    l0 = cal.luminance_at(np.round(b * 255) / 255 * 100)
    l1 = cal.luminance_at(corr * 100)
    mich = lambda a: (a.max() - a.min()) / (a.max() + a.min())  # noqa: E731
    gk = ff.gain_at(centre)
    require(abs(mich(l1) - mich(l0)) < 0.01 and abs(l1.mean() / l0.mean() - gk) < 0.01,
            f"{name}: Michelson {mich(l0):.3f} -> {mich(l1):.3f}, level x{gk:.3f}")
    h, w = b.shape
    c0, r0 = int(round(centre[0] - w / 2)), int(round(centre[1] - h / 2))
    surround = np.isclose(b, bg)
    under = levels[r0:r0 + h, c0:c0 + w] / 255.0
    require(surround.sum() > 0 and np.array_equal(corr[surround], under[surround]),
            f"{name}: the patch's surround is exactly the background image (no square)")

print("4. off by default; recorded when on")
nb = NullBackend()
nb.open(W, H, bg)
tex = Texture(b, (W / 2, H / 2))
require(nb.texture_pixels(tex) is tex.brightness and nb.flat_field_info["name"] == "none",
        "no map: the patch drawn is the very same array (byte-identical), 'none' recorded")
nb.set_flat_field(ff)
require(nb.flat_field_info["name"] == m.name and nb.flat_field_info["strength"] == 0.8
        and nb.flat_field_info["seed"] == 1 and "fake" in nb.flat_field_info["luminance_scale"],
        "with a map: name, strength, seed and scale recorded")
import sim_observer  # noqa: E402
out, obs = sim_observer.run(sim_observer.healthy, 3, extent="24-2 within 9")
require(out.flat_field.get("name") == "none" and _meta(out)["flat_field"]["name"] == "none"
        and "flat-field correction: none" in results_text(out),
        "a session without a map records 'none', and the report says so")
obs2 = sim_observer.Observer(sim_observer.healthy, random.Random(4))
obs2.open(sim_observer.W, sim_observer.H, bg)
obs2.set_flat_field(flatfield.FlatField(m, cal, bg, sim_observer.W, sim_observer.H))
from specvis.procedure.perimetry import run_perimetry  # noqa: E402
out2 = run_perimetry(PerimetrySettings(participant_id="FF", extent="24-2 within 9",
                                       screen_width_mm=sim_observer.WIDTH_MM,
                                       viewing_distance_mm=sim_observer.DIST_MM),
                     obs2, calibration=cal, rng=random.Random(4), show_instructions=False)
require(_meta(out2)["flat_field"]["name"] == m.name
        and "flat-field correction: BenQ median 24 Nov, 80%" in results_text(out2),
        "a session with a map records it, and the report line says 'name, 80%'")

print("5. maps: import from uniformity CSVs, the median, the store")
with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    files = []
    for k, shift in enumerate((0.0, 1.0, -1.0, 30.0)):        # the 4th: one bad run
        f = tmp / f"u{k}.csv"
        with open(f, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["place", "x_frac", "y_frac", "cd_m2", "min", "max", "sd",
                         "pct_vs_centre"])
            for p in m.places:
                pct = 0.0 if p["place"] == "centre" else p["pct"] + shift
                wr.writerow([p["place"], p["x_frac"], p["y_frac"], 25, 25, 25, 0, pct])
        files.append(f)
    places, notes = flatfield.places_from_csvs(files)
    med = {p["place"]: p["pct"] for p in places}
    require(all(abs(med[p] - (LAB[p] + (0.5 if p != "centre" else 0))) < 1e-9 for p in LAB)
            and not notes, "median per place over 4 runs (one bad run does not pull it)")
    import flatfield as tool  # noqa: E402  (tools/flatfield.py)
    store = flatfield.FlatFieldStore(tmp / "maps.json")
    args = tool.parse_args(["flatfield", "--median", *map(str, files), "--name", "Lab map",
                            "--monitor", "BenQ_XL2546K", "--note", "Brightness 20"])
    lines = []
    made = tool.import_map(args, store, log=lambda *a, **k: lines.append(" ".join(map(str, a))))
    back = flatfield.FlatFieldStore(tmp / "maps.json").find("Lab map")
    require(back is not None and back.id == made.id and back.strength == 0.8
            and back.sources == [f.name for f in files] and len(back.places) == 9,
            "saved and found again by name, strength 0.8, sources kept")
    require(any("lift" in s for s in lines), "the import prints the planned lift per place")
    single = tool.parse_args(["--from-csv", str(files[0]), "--name", "one"])
    require(single.from_csv and single.strength == 0.8, "--from-csv takes one file")
    try:
        tool.parse_args(["--median", str(files[0])])
        require(False, "a map without --name is refused")
    except SystemExit:
        require(True, "a map without --name is refused")
    planned = tool.planned_lift(ff, W / 2, 120)
    require(abs(planned - (float(ff.gain[120, 960]) - 1)) < 0.01,
            f"--check's planned lift at the top: +{100 * planned:.1f} %")

sys.exit(0 if ok else 1)
