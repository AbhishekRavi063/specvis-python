#!/usr/bin/env python
"""Measure the stimulus screen with the ColorCAL (Cambridge Research Systems) and save a
luminance scale the app uses (Step 2 -> Existing luminance scale).

    python/specvis.sh calibrate --name "Lab 24in, bg 10"            # the real thing
    python/specvis.sh calibrate --simulate --windowed --no-wait     # try without a meter

Meter fixed at the screen centre (cannot be covered by hand)? Let the tool do it all:

    python\specvis.bat calibrate --name "Lab screen" --port COM9 --auto-zero --no-wait

--auto-zero switches the monitor off (Windows), zero-calibrates the ColorCAL in the dark,
switches the monitor back on, waits --warmup-min minutes, and then measures. --no-wait
starts and finishes without SPACE. Windows is kept from blanking the screen meanwhile.

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
import threading
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

    def zero(self) -> bool:
        return True


CRS_USB_VENDOR = 0x0861        # Cambridge Research Systems (the ColorCAL's USB ID)


def _candidate_ports():
    """Serial ports a ColorCAL may be on (it shows up as a USB modem / COM port).

    The ColorCAL is found by its USB vendor ID. Other ports are never opened: on the lab
    PC, Bluetooth serial ports and Intel AMT (SOL) froze the program when opened.
    """
    import glob
    try:
        from serial.tools import list_ports
        ports = list(list_ports.comports())
        crs = [p.device for p in ports if p.vid == CRS_USB_VENDOR]
        if crs:
            return crs
        usb = [p.device for p in ports
               if p.vid is not None and "bluetooth" not in (p.description or "").lower()]
        if usb or ports:
            return usb
    except ImportError:
        pass
    if sys.platform == "darwin":
        return sorted(glob.glob("/dev/cu.usbmodem*")) or ["/dev/cu.usbmodem0001"]
    if sys.platform.startswith("linux"):
        return sorted(glob.glob("/dev/ttyACM*")) or ["/dev/ttyACM0"]
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()] or [3]
    except ImportError:
        return [3]


def _windows_monitors(on: bool) -> bool:
    """Switch every monitor off (standby) or back on. Windows only; True if it was sent."""
    if sys.platform != "win32":
        return False
    import ctypes
    user32 = ctypes.windll.user32
    hwnd_broadcast, wm_syscommand, sc_monitorpower = 0xFFFF, 0x0112, 0xF170
    # PostMessage, not SendMessage: a window that does not answer would freeze us.
    user32.PostMessageW(hwnd_broadcast, wm_syscommand, sc_monitorpower, -1 if on else 2)
    if on:                                      # a tiny mouse nudge wakes the screen for sure
        user32.mouse_event(0x0001, 0, 1, 0, 0)
        user32.mouse_event(0x0001, 0, -1, 0, 0)
    return True


def _keep_screen_awake(on: bool) -> None:
    """Stop Windows blanking the screen or sleeping while we measure."""
    if sys.platform != "win32":
        return
    import ctypes
    es_continuous, es_system, es_display = 0x80000000, 0x00000001, 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(
        es_continuous | (es_system | es_display if on else 0))


def auto_zero(meter, warmup_min: float, simulated: bool = False) -> None:
    """Zero-calibrate the ColorCAL with the monitor switched off, then let it warm up."""
    print("\nAUTOMATIC ZERO CALIBRATION")
    if simulated:
        print("  (simulated meter: monitor not switched, no waiting)")
        meter.zero()
        return
    if _windows_monitors(False):
        print("  monitor switched off; waiting 10 s in the dark...", flush=True)
    else:
        input("  Switch the monitor OFF with its power button (room dark), then press Enter... ")
    time.sleep(10)
    ok = meter.zero()
    dark = meter.read()
    print(f"  zero calibration: {'OK' if ok else 'FAILED'}; reading in the dark now "
          f"{dark:.4f} cd/m2", flush=True)
    if not _windows_monitors(True):
        input("  Switch the monitor back ON, then press Enter... ")
    _keep_screen_awake(True)
    if not ok:
        raise SystemExit("Zero calibration failed: is any light reaching the sensor?")
    end = time.time() + warmup_min * 60
    while time.time() < end:
        left = (end - time.time()) / 60
        print(f"  monitor warming up: {left:4.1f} min left", flush=True)
        time.sleep(min(60, max(1, end - time.time())))
    print("  warm-up done; measuring now\n", flush=True)


def fast_measure(com, calib_matrix, timeout_s: float = 10.0) -> float:
    """One ColorCAL MES reading over an open serial port, returning Y (cd/m^2).

    The ColorCAL answers 'OK00,<x>,<y>,<z>' and then a '>' prompt; the raw values are
    turned into CIE XYZ with the device's calibration matrix, as psychopy-crs does.
    """
    import re
    import numpy as np
    try:
        com.reset_input_buffer()
    except AttributeError:
        com.flushInput()
    com.write(b"MES\n")
    com.flush()
    com.timeout = 0.05
    buf, text = b"", ""
    end = time.time() + timeout_s
    while time.time() < end:
        buf += com.read(256)
        text = buf.decode("utf-8", "ignore")
        if ("OK00" in text or "ER" in text) and text.rstrip("\r\n ").endswith(">"):
            break
    line = next((s for s in re.split(r"[\r\n>]+", text) if s.startswith("OK00")), None)
    if line is None:
        raise RuntimeError(f"reply not understood: {text!r}")
    raw = np.array([v.strip() for v in line.split(",")[1:4]], dtype=float)
    return float(np.dot(np.asarray(calib_matrix, dtype=float), raw)[1])


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
        self.needs_zero = bool(self.cc.getNeedsCalibrateZero())
        if self.needs_zero:
            print("The ColorCAL asks for a zero calibration: use --auto-zero.")

    def zero(self) -> bool:
        """Zero calibration: only valid when no light reaches the sensor."""
        return bool(self.cc.calibrateZero())

    def read(self) -> float:
        """Luminance (cd/m^2). Sends MES and reads the reply directly: psychopy-crs 0.0.2's
        measure() waits out several 5 s timeouts around the '>' prompt (22 s a reading on
        the lab PC, against 2.6 s for the reply itself)."""
        if getattr(self, "_slow", False):
            return self._driver_read()
        try:
            y = fast_measure(self.cc.com, self.cc.calibMatrix)
        except Exception as e:                          # noqa: BLE001 -- fall back once
            print(f"  fast reading failed ({e}); using the driver's slower measure()")
            self._slow = True
            return self._driver_read()
        if not getattr(self, "_checked", False):        # once: does it agree with the driver?
            self._checked = True
            ref = self._driver_read()
            print(f"  check: fast reading {y:.4f} vs driver {ref:.4f} cd/m2", flush=True)
            if abs(y - ref) > max(0.02, 0.05 * abs(ref)):
                print("  they differ: using the driver's slower measure() from now on")
                self._slow = True
                return ref
        return y

    def _driver_read(self) -> float:
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
    ap.add_argument("--auto-zero", action="store_true",
                    help="switch the monitor off, zero-calibrate the meter, switch it on, "
                         "wait --warmup-min, then measure (meter fixed at the centre)")
    ap.add_argument("--warmup-min", type=float, default=10.0,
                    help="minutes to wait after the monitor comes back on (default 10)")
    args = ap.parse_args()

    meter = SimulatedMeter() if args.simulate else ColorCalMeter(args.port)
    _keep_screen_awake(True)
    if args.auto_zero:
        auto_zero(meter, args.warmup_min, simulated=args.simulate)
    elif getattr(meter, "needs_zero", False):
        raise SystemExit("Run again with --auto-zero (the meter needs a zero calibration).")

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

    def keep_alive(until=None, worker=None) -> None:
        """Redraw the grey window while waiting, so Windows never marks it 'Not responding'
        (it does after ~5 s without a redraw, and may then show other windows in front --
        which is what happened on the lab PC while the meter read dark greys)."""
        while (worker is not None and worker.is_alive()) or (until is not None
                                                             and time.time() < until):
            win.flip()
            if "escape" in event.getKeys(["escape"]):
                win.close()
                raise SystemExit("stopped")

    def measure(level: int):
        grey(level)
        keep_alive(until=time.time() + args.settle)
        out = {}

        def read_all():
            try:
                out["vals"] = [meter.read() for _ in range(args.repeats)]
            except BaseException as e:                 # noqa: BLE001 -- raised below
                out["error"] = e

        worker = threading.Thread(target=read_all, daemon=True)
        worker.start()
        keep_alive(worker=worker)
        if "error" in out:
            raise out["error"]
        vals = out["vals"]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    rows = []
    places = {}
    bg_level = None
    stopped = False
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

        if args.uniformity:
            for place in PLACES:
                say(f"Move the ColorCAL to the {place.upper()} of the screen\n"
                    "(about 3 cm in from the edges), then press SPACE.")
                m, _ = measure(bg_level)
                places[place] = m
                print(f"  background at {place:12s}: {m:.4f} cd/m2", flush=True)
    except SystemExit:
        stopped = True
    _keep_screen_awake(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / time.strftime("%Y%m%d_%H%M%S")

    def write_readings(suffix: str = "") -> None:
        with open(f"{stem}_readings{suffix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["grey_level", "brightness_pct", "cd_m2", "sd", "repeats"])
            for lv, m, sd in sorted(rows):
                w.writerow([lv, round(lv / 255 * 100, 3), round(m, 5), round(sd, 5),
                            args.repeats])

    if stopped:
        # Stopped part-way: keep what was measured (no luminance scale: it needs the whole
        # curve), then close the window.
        if rows:
            write_readings("_PARTIAL")
            print(f"\nStopped. The {len(rows)} readings taken are saved in "
                  f"{stem}_readings_PARTIAL.csv (no luminance scale was made).")
        _close(win)
        return 1

    # Everything is saved BEFORE the window is closed: on the lab PC closing the window
    # once took an hour, with the results still unsaved.
    try:                                # a message only; do not wait for a key
        text.text = "Finished. Saving the results...\nYou can remove the meter."
        text.draw()
        win.flip()
    except Exception:                   # noqa: BLE001 -- the window may already be gone
        pass
    levels, cds = [r[0] for r in rows], [r[1] for r in rows]
    table = photometry.table_from_readings(levels, cds)
    summary = photometry.summarise(levels, cds, bg_level)
    if places:
        summary.update(photometry.uniformity(places))

    write_readings()
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
    dark = [(lv, m) for lv, m, _ in sorted(rows) if lv <= 60]
    negative = [lv for lv, m in dark if m < 0]
    zeros = [lv for lv, m in dark if m == 0]
    warnings = []
    if negative:
        warnings.append(f"NEGATIVE readings at grey {negative}: the meter's zero is off. Run "
                        "again with --auto-zero.")
    if len(zeros) > 2:
        warnings.append(f"readings of exactly 0 at grey {zeros}: below what the meter can "
                        "read, or the monitor crushes its darks (check its Black level / "
                        "Dark boost setting and 'Output dynamic range: Full').")
    info["warnings"] = warnings
    pathlib.Path(f"{stem}_summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print("\n=== RESULT", "(SIMULATED - not a real screen)" if args.simulate else "", "===")
    for w_ in warnings:
        print("WARNING:", w_)
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
    print(f"readings saved: {stem}_readings.csv / _summary.json", flush=True)
    _close(win)
    return 0


def _close(win) -> None:
    """Close the stimulus window. Results are already saved; if closing hangs, the window
    (or the PowerShell / Command Prompt window) can simply be closed."""
    print("closing the measurement window...", flush=True)
    try:
        win.close()
    except Exception:                                   # noqa: BLE001 -- nothing left to lose
        pass


if __name__ == "__main__":
    raise SystemExit(main())
