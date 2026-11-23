#!/usr/bin/env python
"""Screen evenness with the ColorCAL: one grey over the whole screen, read at 9 places.
A red ring with a cross shows where to put the meter. No calibration is made.

    python\\specvis.bat uniformity --label BenQ_XL2546K --note "Brightness 20, DyAc off" --auto-zero
    python/specvis.sh uniformity --simulate --windowed           # try it without a meter

When: after any change to the monitor's settings, and before each site's sessions (IIT
Kanpur and Jodhpur run it the same way, so their monitors can be compared). Time: about
5 min for the zero and warm-up (--auto-zero), then about 1 min per place.

At each place: put the ColorCAL flat on the screen, centred on the cross, touching
LIGHTLY (pressing makes ripples / a white patch and wrong readings), then press ENTER.
R = measure the previous place again; ESC = stop (what was measured is kept, _PARTIAL).

What it can and cannot tell:
  * The meter sits FLAT on the screen, so it measures the panel's own evenness (the
    backlight). It cannot see how a TN panel darkens when looked at from an angle (what a
    participant sees from the chin rest). Keep the eyes level with the screen centre.
  * An even dimming toward the edges lowers the background and the patch together, so
    their CONTRAST (what the test measures) stays the same.

Results: python/output/uniformity/ (kept local, never on GitHub).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics
import sys
import time

PY = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PY), str(PY / "tools")]

from specvis.core import photometry  # noqa: E402
from uniformity_targets import PsychoUI, measure_places  # noqa: E402

OUT_DIR = PY / "output" / "uniformity"
LIMIT_PCT = 10.0            # our suggestion: within +/-10 % of the centre is "even enough"


def parse_args(argv=None):
    from specvis.procedure.perimetry import EXTENTS, PerimetrySettings
    default_w = PerimetrySettings().screen_width_mm
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="serial port, if not found by itself "
                                                 "(lab PC: COM9)")
    ap.add_argument("--screen", type=int, default=0, help="which screen to measure")
    ap.add_argument("--windowed", action="store_true", help="for trying the tool only")
    ap.add_argument("--simulate", action="store_true", help="no meter: a pretend screen")
    ap.add_argument("--auto-zero", action="store_true",
                    help="zero the meter in the dark first (cap ON), then warm up")
    ap.add_argument("--warmup-min", type=float, default=5.0)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--grey", type=int, default=None,
                   help="grey level 0-255 (default 87 = the 10 cd/m2 background of 14 Nov)")
    g.add_argument("--cd", type=float, default=None,
                   help="use a measured luminance scale to find the grey for this cd/m2")
    ap.add_argument("--scale", default=None,
                    help="with --cd: the scale's name or id (default: the newest measured)")
    ap.add_argument("--places", default=",".join(photometry.PLACES),
                    help="places to measure, in order (default all 9)")
    ap.add_argument("--repeats", type=int, default=3, help="readings per place")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds before reading")
    ap.add_argument("--width-mm", type=float, default=default_w,
                    help=f"screen width in mm (default {default_w}, the app's setting)")
    ap.add_argument("--height-mm", type=float, default=None,
                    help="screen height in mm (default: from the width and the pixels)")
    ap.add_argument("--inset-mm", type=float, default=None,
                    help="put the outer places this many mm in from the edges (default: "
                         "the lab's fractions, 30 mm on a 542.3 x 305 mm screen)")
    ap.add_argument("--ring-mm", type=float, default=60.0, help="target ring diameter")
    ap.add_argument("--rings-on-black", action="store_true",
                    help="show the targets on black (measuring is still on the grey)")
    ap.add_argument("--label", default="", help="e.g. BenQ_XL2546K or HP527sf")
    ap.add_argument("--note", default="", help='monitor settings, e.g. "Brightness 20, DyAc off"')
    ap.add_argument("--figure", action="store_true", help="also save a map (PNG and SVG)")
    ap.add_argument("--grid", default=None, choices=EXTENTS,
                    help='with --figure: draw this test grid, e.g. "24-2 within 9"')
    ap.add_argument("--distance-mm", type=float, default=570.0, help="viewing distance (grid)")
    ap.add_argument("--eye", default="R", choices=("R", "L"), help="eye tested (grid)")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["uniformity"]:        # specvis.bat passes every word (%*), its own too
        argv = argv[1:]
    args = ap.parse_args(argv)
    args.places = [p.strip() for p in args.places.split(",") if p.strip()]
    for p in args.places:
        try:
            photometry.place_offset(p, 0.1, 0.1)
        except ValueError as e:
            ap.error(str(e))
    if "centre" not in args.places:
        ap.error("--places must include centre (everything is compared with it)")
    return args


def grey_for(args) -> tuple:
    """(grey level, how it was chosen)."""
    if args.cd is None:
        level = 87 if args.grey is None else args.grey
        if not 0 <= level <= 255:
            raise SystemExit("--grey must be 0 to 255")
        return level, f"grey {level}"
    from specvis.app import store
    scales = [s for s in store.ScaleStore().all() if len(s.table) == 101]
    if args.scale:
        scales = [s for s in scales if args.scale in (s.id, s.name)]
    if not scales:
        raise SystemExit("--cd needs a ColorCAL-measured luminance scale (python/specvis.sh "
                         "calibrate); none found" + (f" named {args.scale!r}" if args.scale
                                                      else "") + ". Use --grey instead.")
    scale = max(scales, key=lambda s: s.created)
    cal = scale.calibration()
    if not cal.can_achieve(args.cd):
        raise SystemExit(f"{args.cd} cd/m2 is outside scale '{scale.name}'")
    level = int(round(float(cal.brightness_for(args.cd)) / 100.0 * 255.0))
    return level, f"grey {level} = {args.cd:g} cd/m2 by scale '{scale.name}' ({scale.created})"


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "screen"


def report(readings, fx, fy, w_mm, h_mm, log=print):
    """The rows (one per place) and the largest difference from the centre."""
    centre = statistics.mean(readings["centre"])
    rows = []
    for place, vals in readings.items():
        x, y = photometry.place_offset(place, fx, fy)
        m = statistics.mean(vals)
        rows.append(dict(place=place, x_frac=round(x, 4), y_frac=round(y, 4),
                         x_mm=x * w_mm, y_mm=y * h_mm, cd_m2=m, min=min(vals), max=max(vals),
                         sd=statistics.stdev(vals) if len(vals) > 1 else 0.0,
                         pct_vs_centre=100.0 * (m - centre) / centre, readings=vals))
    worst = max(abs(r["pct_vs_centre"]) for r in rows)
    return rows, centre, worst


def write_outputs(stem: pathlib.Path, rows, info: dict) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["place", "x_frac", "y_frac", "cd_m2", "min", "max", "sd", "pct_vs_centre"])
        for r in rows:
            w.writerow([r["place"], r["x_frac"], r["y_frac"], round(r["cd_m2"], 4),
                        round(r["min"], 4), round(r["max"], 4), round(r["sd"], 4),
                        round(r["pct_vs_centre"], 2)])
    pathlib.Path(f"{stem}_summary.json").write_text(json.dumps(info, indent=2) + "\n")


def grid_spots_mm(grid: str, distance_mm: float, eye: str):
    """The app's test spots on the screen: x_mm = d * tan(x_deg), from the fixation (the
    screen centre)."""
    import math
    from specvis.procedure.perimetry import PerimetrySettings, grid_points
    kept, _ = grid_points(PerimetrySettings(extent=grid, eye=eye))
    return [(distance_mm * math.tan(math.radians(x)), distance_mm * math.tan(math.radians(y)))
            for x, y in kept]


def draw_figure(stem, rows, w_mm, h_mm, title, spots=None) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    coef = photometry.fit_screen_map([(r["x_mm"], r["y_mm"]) for r in rows],
                                     [r["pct_vs_centre"] for r in rows])
    fig, ax = plt.subplots(figsize=(10, 10 * h_mm / w_mm + 1.2))
    if coef is not None:
        xs = np.linspace(-w_mm / 2, w_mm / 2, 220)
        ys = np.linspace(-h_mm / 2, h_mm / 2, 130)
        gx, gy = np.meshgrid(xs, ys)
        z = photometry.screen_map_at(coef, gx, gy)
        im = ax.imshow(z, extent=(-w_mm / 2, w_mm / 2, -h_mm / 2, h_mm / 2), origin="lower",
                       cmap="viridis", vmin=min(-15.0, float(z.min())),
                       vmax=max(0.0, float(z.max())))
        cs = ax.contour(gx, gy, z, levels=range(-30, 1, 2), colors="white", linewidths=0.6)
        ax.clabel(cs, fmt="%d %%", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.7, label="% vs centre (fitted)")
    for r in rows:
        ax.plot(r["x_mm"], r["y_mm"], "s", color="black", ms=9)
        ax.annotate(f"{r['place']}\n{r['pct_vs_centre']:+.1f} %", (r["x_mm"], r["y_mm"]),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8,
                    color="black", bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
    lines = []
    if spots and coef is not None:
        est = [float(photometry.screen_map_at(coef, x, y)) for x, y in spots]
        for (x, y), e in zip(spots, est):
            shown = round(e)                               # colour by the printed number
            ax.add_patch(plt.Circle((x, y), 6, fill=False, lw=2,
                                    color=photometry.evenness_colour(shown)))
            ax.text(x, y, f"{shown:d}", ha="center", va="center", fontsize=6)
        lines.append(f"grid spots: estimated {round(max(est)):+d} to {round(min(est)):+d} % "
                     f"vs centre ({len(spots)} spots)")
    ax.set_xlim(-w_mm / 2, w_mm / 2)
    ax.set_ylim(-h_mm / 2, h_mm / 2)
    ax.set_aspect("equal")
    ax.set_xlabel("mm from the screen centre")
    ax.set_title(title, fontsize=10)
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return lines


def run(args, meter, ui, screen_info: dict, log=print) -> int:
    """Measure, then print and save. ``ui`` draws the targets (PsychoUI, or a test fake)."""
    level, how = grey_for(args)
    w_px, h_px = ui.size
    w_mm = args.width_mm
    h_mm = args.height_mm or w_mm * h_px / w_px
    fx, fy = photometry.place_fractions(inset_mm=args.inset_mm, width_mm=w_mm, height_mm=h_mm)
    ring_r = photometry.ring_radius_mm(args.ring_mm, w_mm, h_mm, args.places, fx, fy)
    if ring_r < args.ring_mm / 2:
        log(f"(ring made {2 * ring_r:.0f} mm across so it stays inside the screen)")
    if hasattr(ui, "set_ring_mm"):
        ui.set_ring_mm(2 * ring_r)
    log(f"Measuring {how} at {len(args.places)} places; screen {w_mm:.1f} x {h_mm:.1f} mm, "
        f"{w_px:.0f} x {h_px:.0f} px; places {100 * fx:.1f} % / {100 * fy:.1f} % from the "
        "centre", flush=True)

    readings, stopped = measure_places(ui, meter, args.places, fx, fy, level,
                                       repeats=args.repeats, settle_s=args.settle, log=log)

    label = safe_label(args.label)
    stem = OUT_DIR / f"uniformity_{label}_grey{level}_{time.strftime('%Y%m%d_%H%M%S')}"
    if stopped:
        stem = pathlib.Path(f"{stem}_PARTIAL")
    info = dict(label=args.label, note=args.note, grey=level, grey_chosen_by=how,
                screen=dict(width_mm=round(w_mm, 1), height_mm=round(h_mm, 1),
                            width_px=int(w_px), height_px=int(h_px), **screen_info),
                meter_serial=str(getattr(meter, "serial", "")),
                simulated=bool(args.simulate), repeats=args.repeats,
                place_fractions=dict(x=round(fx, 4), y=round(fy, 4)),
                inset_mm=args.inset_mm, places_planned=args.places,
                created=time.strftime("%Y-%m-%d %H:%M:%S"), stopped_early=stopped)

    if "centre" not in readings:
        if readings:
            info["readings"] = readings
            stem.parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(f"{stem}_summary.json").write_text(json.dumps(info, indent=2) + "\n")
            log(f"Stopped before the centre was measured; readings saved in {stem}_summary.json")
        else:
            log("Stopped; nothing measured.")
        return 1

    rows, centre, worst = report(readings, fx, fy, w_mm, h_mm)
    notes = photometry.uniformity_warnings({r["place"]: r["cd_m2"] for r in rows}, readings)
    info.update(centre_cd_m2=round(centre, 4), largest_difference_pct=round(worst, 2),
                within_10_pct=worst <= LIMIT_PCT, checks=notes,
                places={r["place"]: dict(cd_m2=round(r["cd_m2"], 4),
                                         pct_vs_centre=round(r["pct_vs_centre"], 2),
                                         readings=[round(v, 4) for v in r["readings"]])
                        for r in rows})

    log(f"\n=== {args.label or 'screen'}: {how}{'  (SIMULATED)' if args.simulate else ''}"
        f"{'  STOPPED EARLY' if stopped else ''} ===")
    log(f"{'place':13s} {'cd/m2':>8s}  {'vs centre':>9s}")
    for r in rows:
        log(f"{r['place']:13s} {r['cd_m2']:8.3f}  {r['pct_vs_centre']:+8.1f} %")
    log(f"\nLargest difference from the centre: {worst:.1f} %  ->  "
        + (f"within +/-{LIMIT_PCT:g} %" if worst <= LIMIT_PCT
           else f"MORE than +/-{LIMIT_PCT:g} %"))
    log("(An even dimming toward the edges keeps the patches' contrast the same. The meter "
        "flat on the screen cannot see the TN viewing-angle effect from the chin rest.)")
    for n in notes:
        log("CHECK:", n)

    if args.figure and not stopped:
        spots = grid_spots_mm(args.grid, args.distance_mm, args.eye) if args.grid else None
        title = (f"{args.label or 'screen'}  grey {level}  {args.note}\n"
                 + (f"grid {args.grid}, {args.distance_mm:.0f} mm, eye {args.eye}"
                    if args.grid else ""))
        for line in draw_figure(stem, rows, w_mm, h_mm, title, spots):
            log(line)
            info["grid_estimate"] = line
        info["figure"] = f"{stem.name}.png"
    write_outputs(stem, rows, info)
    log(f"Saved {stem}.csv and {stem.name}_summary.json")
    return 1 if stopped else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    from calibrate_photometer import (ColorCalMeter, SimulatedMeter, _keep_screen_awake,
                                      auto_zero)
    grey_for(args)                                   # a bad --cd stops before the meter
    print(f"This opens on screen {args.screen}. If the monitor you want to test is not the "
          "main display, use --screen or make it the main display in Windows.")
    meter = SimulatedMeter() if args.simulate else ColorCalMeter(args.port)
    _keep_screen_awake(True)
    if args.auto_zero or getattr(meter, "needs_zero", False):
        auto_zero(meter, args.warmup_min, simulated=args.simulate)

    from psychopy import visual
    win = visual.Window(size=(1200, 800), screen=args.screen, fullscr=not args.windowed,
                        allowGUI=args.windowed, color=-1.0, colorSpace="rgb", units="pix")
    win.mouseVisible = False
    try:
        rate = win.getActualFrameRate(nIdentical=10, nMaxFrames=120, nWarmUpFrames=10)
    except Exception:                                # noqa: BLE001 -- information only
        rate = None
    ui = PsychoUI(win, args.width_mm, args.ring_mm, on_black=args.rings_on_black)
    try:
        return run(args, meter, ui,
                   dict(screen_index=args.screen,
                        refresh_hz=round(rate, 2) if rate else None))
    finally:
        _keep_screen_awake(False)
        try:
            win.close()
        except Exception:                            # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
