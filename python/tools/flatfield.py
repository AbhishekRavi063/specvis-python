#!/usr/bin/env python
"""Flat-field (screen evenness) maps: import them from uniformity runs, list them, and
check one on the screen with the ColorCAL.

    python\\specvis.bat flatfield --from-csv python\\output\\uniformity\\X.csv --name "BenQ 24 Nov"
    python\\specvis.bat flatfield --median A.csv B.csv C.csv D.csv --name "BenQ median 24 Nov" ^
                                  --monitor BenQ_XL2546K --note "Standard, Brightness 20, DyAc+ off"
    python\\specvis.bat flatfield --list
    python\\specvis.bat flatfield --check "BenQ median 24 Nov" --scale "Lab screen DyAc off Brightness 20" --port COM9

What the correction does: the background is drawn slightly brighter where the screen's
backlight is weaker, and every patch is multiplied by the same local gain, so its
contrast stays exactly as asked. It is OFF unless a map is chosen (app Step 2, or
run_block2.py --flat-field NAME).

A map belongs to ONE monitor and its settings: after any change to Brightness, DyAc+,
Gamma etc. make a new luminance scale and a new map, and check it.

Making a map: several `specvis uniformity` runs (on the tripod, touching lightly, room
dark; placing the meter again changes readings by up to ~20 %, so one run is not
enough), combined with --median, then --check: at each ring the meter is put down ONCE
and reads plain grey, the corrected grey, and plain again; the measured lift is
compared with the planned lift. That is the only reliable check (placement changes
readings more than the correction does).

Maps are kept in python/calibration/flatfield_maps.json (local, never on GitHub).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import time

PY = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PY), str(PY / "tools")]

from specvis.core import flatfield, photometry  # noqa: E402


def parse_args(argv=None):
    from specvis.procedure.perimetry import PerimetrySettings
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-csv", metavar="CSV", help="one uniformity CSV")
    g.add_argument("--median", nargs="+", metavar="CSV",
                   help="several uniformity CSVs: the median per place")
    g.add_argument("--list", action="store_true", help="the maps on this PC")
    g.add_argument("--check", metavar="MAP", help="A/B check of a map with the ColorCAL")
    ap.add_argument("--name", default="", help="the map's name, shown in the app")
    ap.add_argument("--strength", type=float, default=flatfield.DEFAULT_STRENGTH,
                    help="how much of the measured unevenness to correct (default 0.8)")
    ap.add_argument("--seed", type=int, default=flatfield.DEFAULT_SEED,
                    help="the static dither's seed (default 1)")
    ap.add_argument("--monitor", default="", help="e.g. BenQ_XL2546K")
    ap.add_argument("--note", default="", help="the monitor's settings")
    ap.add_argument("--width-mm", type=float, default=PerimetrySettings().screen_width_mm)
    ap.add_argument("--height-mm", type=float, default=305.0)
    # --check
    ap.add_argument("--scale", default="", help="--check: the luminance scale (name or id)")
    ap.add_argument("--cd", type=float, default=10.0, help="--check: the background, cd/m2")
    ap.add_argument("--places", default=",".join(p for p in photometry.PLACES if p != "centre"),
                    help="--check: places to check")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--port", default=None)
    ap.add_argument("--screen", type=int, default=0)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="--check without a meter")
    ap.add_argument("--auto-zero", action="store_true")
    ap.add_argument("--warmup-min", type=float, default=5.0)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["flatfield"]:            # specvis.bat passes every word (%*)
        argv = argv[1:]
    args = ap.parse_args(argv)
    if (args.from_csv or args.median) and not args.name:
        ap.error("give the map a --name")
    if not 0 < args.strength <= 1:
        ap.error("--strength must be above 0 and at most 1")
    return args


def import_map(args, store: flatfield.FlatFieldStore = None, log=print) -> flatfield.FlatFieldMap:
    paths = [pathlib.Path(p) for p in ([args.from_csv] if args.from_csv else args.median)]
    places, notes = flatfield.places_from_csvs(paths)
    screen = {}
    for p in paths:                                   # the screen, from a summary if there
        js = p.with_name(p.stem + "_summary.json")
        if js.exists():
            screen = json.loads(js.read_text()).get("screen", {})
            break
    m = flatfield.FlatFieldMap(id=flatfield.new_map_id(), name=args.name, places=places,
                               monitor=args.monitor, note=args.note, screen=screen,
                               strength=args.strength, seed=args.seed,
                               sources=[p.name for p in paths])
    surface = m.surface()
    store = store or flatfield.FlatFieldStore()
    store.put(m)
    log(f"Saved flat-field map '{m.name}' ({m.id}): {len(places)} places, {surface.kind}, "
        f"strength {m.strength:g}, from {len(paths)} file(s).")
    log(f"{'place':13s} {'% vs centre':>11s}  {'lift at strength':>16s}")
    for p in places:
        lift = 100.0 * (float(flatfield.gain(p["pct"], m.strength)) - 1.0)
        log(f"{p['place']:13s} {p['pct']:+11.2f}  {lift:+15.1f} %")
    for n in notes:
        log("NOTE:", n)
    log("Choose it in the app (Step 2, Flat-field map), or run_block2.py --flat-field "
        f'"{m.name}". Check it first: specvis flatfield --check "{m.name}" --scale ...')
    return m


def list_maps(store=None, log=print) -> None:
    maps = (store or flatfield.FlatFieldStore()).all()
    if not maps:
        log("No flat-field maps yet (specvis flatfield --median ... --name ...).")
    for m in maps:
        worst = min(p["pct"] for p in m.places)
        log(f"{m.created}  {m.id}  '{m.name}'  {m.monitor}  strength {m.strength:g}  "
            f"{len(m.places)} places, darkest {worst:+.1f} %  {m.note}")


def planned_lift(ff: flatfield.FlatField, x_px: float, y_px: float, half: int = 12) -> float:
    """What the corrected screen should give under the meter: the mean luminance of the
    rendered levels in a small square, against the plain background."""
    h, w = ff.background_levels().shape
    r, c = int(round(y_px)), int(round(x_px))
    block = ff.background_levels()[max(0, r - half):min(h, r + half + 1),
                                   max(0, c - half):min(w, c + half + 1)]
    lum = ff.calibration.luminance_at(block / 255.0 * 100.0)
    return float(lum.mean()) / ff.background_cd_m2 - 1.0


def check(args) -> int:
    from calibrate_photometer import (ColorCalMeter, SimulatedMeter, _keep_screen_awake,
                                      _windows_monitors, auto_zero)
    from specvis.app.store import ScaleStore
    from specvis.core.gabor import snap_background
    from uniformity_targets import PsychoUI, Stopped, read_meter

    fmap = flatfield.FlatFieldStore().find(args.check)
    if fmap is None:
        raise SystemExit(f"no flat-field map '{args.check}' (specvis flatfield --list)")
    scales = [s for s in ScaleStore().all() if args.scale in (s.name, s.id)]
    if not scales or len(scales[0].table) != 101:
        raise SystemExit("--check needs --scale: a ColorCAL-measured luminance scale")
    cal = scales[0].calibration()
    _, bg = snap_background(cal, args.cd)
    level = int(round(bg * 255))
    print(f"This opens on screen {args.screen}. If the monitor to check is not the main "
          "display, use --screen or make it the main display in Windows.")
    meter = SimulatedMeter() if args.simulate else ColorCalMeter(args.port)
    _keep_screen_awake(True)
    if args.auto_zero or getattr(meter, "needs_zero", False):
        auto_zero(meter, args.warmup_min, simulated=args.simulate)

    from psychopy import visual
    import numpy as np
    win = visual.Window(size=(1200, 800), screen=args.screen, fullscr=not args.windowed,
                        allowGUI=args.windowed, color=bg * 2 - 1, colorSpace="rgb",
                        units="pix")
    win.mouseVisible = False
    rows, stopped = [], False
    try:
        ui = PsychoUI(win, args.width_mm, 60.0)
        w_phys, h_phys = int(win.size[0]), int(win.size[1])
        ff = flatfield.FlatField(fmap, cal, bg, w_phys, h_phys)
        corrected = visual.ImageStim(win, image=np.flipud(ff.background_levels() / 255 * 2 - 1),
                                     units="pix", size=ui.size, interpolate=False)
        fx, fy = photometry.place_fractions()
        h_mm = args.width_mm * ui.size[1] / ui.size[0]
        ui.set_ring_mm(2 * photometry.ring_radius_mm(60.0, args.width_mm, h_mm,
                                                     photometry.PLACES, fx, fy))
        ui.set_grey(level)
        win.backgroundFit = None             # PsychoPy must not rescale the image
        places = [p.strip() for p in args.places.split(",") if p.strip()]

        def phase(show_corrected: bool):
            win.backgroundImage = corrected if show_corrected else None
            win.color = bg * 2 - 1
            for _ in range(2):
                if show_corrected:
                    corrected.draw()
                win.flip()
            ui.keep_alive(until=time.time() + args.settle)
            if args.simulate:
                meter.set_level(level)
            vals = read_meter(ui, meter, args.repeats)
            if args.simulate and show_corrected:
                vals = [v * (1 + plan) for v in vals]
            return statistics.mean(vals)

        print(f"\nA/B check of '{fmap.name}' at grey {level} ({args.cd:g} cd/m2 by "
              f"'{scales[0].name}'), strength {ff.strength:g}:")
        for i, place in enumerate(places, 1):
            x, y = photometry.place_pixels(place, ui.size[0], ui.size[1], fx, fy)
            win.backgroundImage = None
            ui.show_target(f"Check {i} of {len(places)}: {place.upper()}\n\nPut the ColorCAL "
                           "on the cross, touching lightly.\nIt stays there for 3 readings "
                           "(plain, corrected, plain).\n\nENTER = go.   (ESC stops)", x, y)
            if ui.wait_key() == "escape":
                stopped = True
                break
            sx = (x + ui.size[0] / 2) * ui.win.size[0] / ui.size[0]
            sy = (ui.size[1] / 2 - y) * ui.win.size[1] / ui.size[1]
            plan = planned_lift(ff, sx, sy)
            print(f"  {place}: measuring... don't move the meter", flush=True)
            a1 = phase(False)
            b = phase(True)
            a2 = phase(False)
            lift = b / ((a1 + a2) / 2) - 1
            drift = abs(a2 - a1) / ((a1 + a2) / 2)
            rows.append(dict(place=place, plain_1=round(a1, 4), corrected=round(b, 4),
                             plain_2=round(a2, 4), measured_lift_pct=round(100 * lift, 2),
                             planned_lift_pct=round(100 * plan, 2),
                             difference_pct=round(100 * (lift - plan), 2),
                             plain_drift_pct=round(100 * drift, 2)))
            r = rows[-1]
            print(f"  {place:13s} measured {r['measured_lift_pct']:+6.1f} %   planned "
                  f"{r['planned_lift_pct']:+6.1f} %   (plain drift {r['plain_drift_pct']:.1f} %)"
                  + ("   UNSTEADY: repeat" if drift > 0.02 else ""), flush=True)
    except Stopped:
        stopped = True
    finally:
        try:
            win.close()
        except Exception:                                # noqa: BLE001
            pass
        if args.auto_zero:
            _windows_monitors(True)
            time.sleep(1.0)
        _keep_screen_awake(False)
        if args.auto_zero:
            time.sleep(0.5)
            _windows_monitors(True)
    if not rows:
        return 1
    worst = max(abs(r["difference_pct"]) for r in rows)
    print(f"\nLargest difference measured vs planned: {worst:.1f} %  ->  "
          + ("the correction does what it should" if worst <= 2.0 else
             "MORE than 2 %: repeat the places marked, or measure the map again"))
    out = PY / "output" / "uniformity"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / (f"flatfield_check_{fmap.id}_{time.strftime('%Y%m%d_%H%M%S')}"
                  + ("_PARTIAL" if stopped else ""))
    with open(f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Saved {stem}.csv")
    return 1 if stopped else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        list_maps()
        return 0
    if args.check:
        return check(args)
    import_map(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
