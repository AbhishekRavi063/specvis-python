#!/usr/bin/env python
"""Measure the stimulus screen with the ColorCAL (Cambridge Research Systems) and save a
luminance scale the app uses (Step 2 -> Existing luminance scale).

    python/specvis.sh calibrate --name "Lab 24in, bg 10"            # the real thing
    python/specvis.sh calibrate --simulate --windowed --no-wait     # try without a meter

Before measuring:
  * screen: fixed brightness setting; dynamic contrast, eco / auto-brightness and
    overdrive OFF; warmed up 30 minutes; room dark, as in the test;
  * ColorCAL on its tripod (Manfrotto 495 head), against the centre of the screen,
    plugged in by USB.

What it does:
  1. greys 0, 5, ... 60, then 75, 90 ... 255, full screen, each read --repeats times
     (the screen curve; the dark end bends most);
  2. every grey level within 8 of the background (the faintest step lives there);
  3. optionally (--uniformity) the background at 7 places on the screen, moving the
     meter when asked;
  4. saves the readings (python/calibration/photometer/*.csv and *.json) and a luminance
     scale, and prints the dB range and its middle (the staircase seeds' start).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from specvis.app import store  # noqa: E402
from specvis.core import photometry  # noqa: E402
from specvis.core.luminance import MEASUREMENT_POINTS, LuminanceCalibration  # noqa: E402

OUT_DIR = HERE / "calibration" / "photometer"
PLACES = ["centre", "top-left", "top-right", "bottom-left", "bottom-right", "left", "right"]


class SimulatedMeter:
    """An assumed 300 cd/m^2, gamma-2.2 screen with 0.3 cd/m^2 black and 0.5 % noise:
    for trying the tool without a ColorCAL. Never use its numbers."""

    def __init__(self):
        self.level = 0
        self.serial = "SIMULATED"

    def set_level(self, level: int) -> None:
        self.level = level

    def read(self) -> float:
        true = 0.3 + (300.0 - 0.3) * (self.level / 255.0) ** 2.2
        return true * (1.0 + random.gauss(0.0, 0.005))


def _candidate_ports():
    """Serial ports a ColorCAL may be on (it shows up as a USB modem / COM port)."""
    import glob
    if sys.platform == "darwin":
        return sorted(glob.glob("/dev/cu.usbmodem*")) or ["/dev/cu.usbmodem0001"]
    if sys.platform.startswith("linux"):
        return sorted(glob.glob("/dev/ttyACM*")) or ["/dev/ttyACM0"]
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()] or [3]
    except ImportError:
        return [3]


class ColorCalMeter:
    """The ColorCAL through PsychoPy's official CRS plugin (psychopy-crs)."""

    def __init__(self, port=None):
        try:
            from psychopy_crs.colorcal import ColorCAL
        except ImportError as e:
            raise SystemExit("The PsychoPy CRS plugin is missing. Install it with:\n"
                             "  Mac:     python/.venv/bin/python -m pip install psychopy-crs\n"
                             "  Windows: python\\.venv\\Scripts\\python -m pip install "
                             "psychopy-crs") from e
        ports = [port] if port else _candidate_ports()
        self.cc = None
        for p in ports:
            cc = None
            try:
                cc = ColorCAL(p)
                found = getattr(cc, "OK", False) and cc.getInfo()[0]
            except Exception:                   # noqa: BLE001 -- not a ColorCAL; next port
                found = False
            if found:
                self.cc = cc
                print(f"ColorCAL on port {p}")
                break
            com = getattr(cc, "com", None)
            if com:
                try:
                    com.close()
                except Exception:               # noqa: BLE001
                    pass
        if self.cc is None:
            raise SystemExit("ColorCAL not found. Check the USB cable, or give --port "
                             "(Mac: ls /dev/cu.usbmodem*; Windows: Device Manager -> COM).")
        ok, serial, firmware, _ = self.cc.getInfo()
        self.serial = str(serial)
        print(f"ColorCAL connected: serial {serial}, firmware {firmware}")
        if self.cc.getNeedsCalibrateZero():
            input("The ColorCAL needs a zero calibration. Cover the sensor completely "
                  "(cap on / face down on the desk), then press Enter... ")
            if not self.cc.calibrateZero():
                raise SystemExit("Zero calibration failed: was the sensor fully covered?")
            input("Done. Put it back against the centre of the screen and press Enter... ")

    def read(self) -> float:
        ok, x, y, z = self.cc.measure()
        if not ok:
            raise RuntimeError("the ColorCAL did not return a reading")
        return float(y)            # Y = luminance, cd/m^2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=time.strftime("ColorCAL %Y-%m-%d %H%M"),
                    help="name of the luminance scale shown in the app")
    ap.add_argument("--background-cd", type=float, default=10.0,
                    help="the background you plan to use, cd/m^2 (default 10)")
    ap.add_argument("--port", default=None, help="serial port, if not found by itself")
    ap.add_argument("--screen", type=int, default=0, help="which screen to measure")
    ap.add_argument("--repeats", type=int, default=3, help="readings per grey level")
    ap.add_argument("--settle", type=float, default=1.0,
                    help="seconds to wait after each change before reading")
    ap.add_argument("--uniformity", action="store_true",
                    help="also read the background at 7 places (you move the meter)")
    ap.add_argument("--simulate", action="store_true", help="no meter: a pretend screen")
    ap.add_argument("--windowed", action="store_true", help="for trying the tool only")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for SPACE (trying)")
    ap.add_argument("--no-save-scale", action="store_true",
                    help="save the readings but do not add a scale to the app")
    args = ap.parse_args()

    meter = SimulatedMeter() if args.simulate else ColorCalMeter(args.port)

    from psychopy import core, event, visual
    win = visual.Window(size=(1200, 800), screen=args.screen, fullscr=not args.windowed,
                        allowGUI=args.windowed, color=-1.0, colorSpace="rgb", units="norm")
    win.mouseVisible = False
    text = visual.TextStim(win, "", color=0.6, height=0.06, wrapWidth=1.8)

    def say(msg: str) -> None:
        text.text = msg
        text.draw()
        win.flip()
        if args.no_wait:
            return
        keys = event.waitKeys(keyList=["space", "escape"])
        if "escape" in keys:
            win.close()
            raise SystemExit("stopped")

    def grey(level: int) -> None:
        win.color = level / 255.0 * 2.0 - 1.0
        win.flip()
        win.flip()                     # the colour shows from the next frame
        if isinstance(meter, SimulatedMeter):
            meter.set_level(level)

    def measure(level: int):
        grey(level)
        core.wait(args.settle)
        vals = [meter.read() for _ in range(args.repeats)]
        if "escape" in event.getKeys(["escape"]):
            win.close()
            raise SystemExit("stopped")
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    rows = []
    try:
        say("ColorCAL against the CENTRE of the screen, room dark.\n\n"
            "The screen will turn different greys for a few minutes.\n"
            "Do not move the meter or touch the screen.\n\nSPACE to start, ESC to stop.")
        coarse = photometry.coarse_levels()
        for i, lv in enumerate(coarse):
            m, sd = measure(lv)
            rows.append((lv, m, sd))
            print(f"  grey {lv:3d}: {m:9.4f} cd/m2  (sd {sd:.4f})   [{i + 1}/{len(coarse)}]",
                  flush=True)
        bg_level = photometry.background_level([r[0] for r in rows], [r[1] for r in rows],
                                               args.background_cd)
        print(f"background {args.background_cd:g} cd/m2 is about grey {bg_level}; "
              "reading every level around it", flush=True)
        done = {r[0] for r in rows}
        for lv in range(max(0, bg_level - 8), min(255, bg_level + 8) + 1):
            if lv in done:
                continue
            m, sd = measure(lv)
            rows.append((lv, m, sd))
            print(f"  grey {lv:3d}: {m:9.4f} cd/m2  (sd {sd:.4f})", flush=True)
        # With every level around it measured, the background is now exact.
        bg_level = photometry.background_level([r[0] for r in rows], [r[1] for r in rows],
                                               args.background_cd)

        places = {}
        if args.uniformity:
            for place in PLACES:
                say(f"Move the ColorCAL to the {place.upper()} of the screen\n"
                    "(about 3 cm in from the edges), then press SPACE.")
                m, _ = measure(bg_level)
                places[place] = m
                print(f"  background at {place:12s}: {m:.4f} cd/m2", flush=True)
        say("Finished. You can remove the meter.\n\nSPACE to close.")
    finally:
        win.close()

    levels, cds = [r[0] for r in rows], [r[1] for r in rows]
    table = photometry.table_from_readings(levels, cds)
    summary = photometry.summarise(levels, cds, bg_level)
    if places:
        summary.update(photometry.uniformity(places))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    with open(f"{stem}_readings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grey_level", "brightness_pct", "cd_m2", "sd", "repeats"])
        for lv, m, sd in sorted(rows):
            w.writerow([lv, round(lv / 255 * 100, 3), round(m, 5), round(sd, 5), args.repeats])
    info = dict(name=args.name, meter=("SIMULATED" if args.simulate else "ColorCAL"),
                serial=meter.serial, screen=args.screen, settle_s=args.settle,
                background_target_cd_m2=args.background_cd, summary=summary,
                uniformity=places, table=table, created=time.strftime("%Y-%m-%d %H:%M"))
    pathlib.Path(f"{stem}_summary.json").write_text(json.dumps(info, indent=2) + "\n")

    if not args.no_save_scale and not args.simulate:
        cal = LuminanceCalibration.from_table(table, args.name)
        six = [float(cal.luminance_at(b)) for b in MEASUREMENT_POINTS]
        scale = store.LuminanceScale(store.new_scale_id(), args.name, six, table=table,
                                     notes=f"ColorCAL serial {meter.serial}; {len(rows)} grey "
                                           f"levels; {time.strftime('%Y-%m-%d %H:%M')}")
        store.ScaleStore().put(scale)
        print(f"\nSaved luminance scale '{args.name}' -> choose it in the app, Step 2.")

    s = summary
    print("\n=== RESULT", "(SIMULATED - not a real screen)" if args.simulate else "", "===")
    print(f"black {s['black_cd_m2']:.3f} cd/m2, white {s['white_cd_m2']:.1f} cd/m2")
    print(f"background: grey {s['background_level']} = {s['background_cd_m2']:.3f} cd/m2")
    print(f"strongest patch at this background: {s['strongest_db'] + 0.0:.1f} dB")
    print(f"one grey step at the background:    {s['grey_step_db']:.1f} dB "
          "(the faintest without dithering)")
    print(f"middle of the range (no dithering): {s['middle_db_no_dither']:.1f} dB "
          "= the staircase seeds' start")
    if places:
        print(f"background uniformity: largest difference from the centre "
              f"{s['max_difference_pct']:.1f} %")
    print(f"readings saved: {stem}_readings.csv / _summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
