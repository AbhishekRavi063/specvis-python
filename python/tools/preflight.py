#!/usr/bin/env python
"""Before a real session: is everything connected and set up?

    python/specvis.sh preflight --width-mm 531 --distance-mm 330

Checks, without showing any stimulus:
  1. the screens PsychoPy sees, and whether the 24-2 fits at this width and distance;
  2. the ColorCAL answers (skip with --no-photometer);
  3. a measured luminance scale exists (from python/specvis.sh calibrate);
  4. gaze arrives from the eye-tracker bridge (skip with --no-tracker).
Each line says OK or what to do.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import sys
import time

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from specvis.app import store  # noqa: E402
from specvis.procedure.fixation import BRIDGE_PORT  # noqa: E402
from specvis.procedure.perimetry import H24_2, PerimetrySettings, fit_problem  # noqa: E402

results = []


def report(ok: bool, what: str, fix: str = "") -> None:
    results.append(ok)
    print(("  OK    " if ok else "  TODO  ") + what + ("" if ok or not fix else f"\n        -> {fix}"))


def screens():
    try:
        import pyglet
        display = pyglet.canvas.get_display() if hasattr(pyglet, "canvas") else pyglet.display.get_display()
        return [(s.width, s.height) for s in display.get_screens()]
    except Exception:                                   # noqa: BLE001 -- informational
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width-mm", type=float, required=True,
                    help="the stimulus screen's visible width, measured with a ruler")
    ap.add_argument("--distance-mm", type=float, default=330.0, help="eye to screen")
    ap.add_argument("--screen", type=int, default=0)
    ap.add_argument("--no-photometer", action="store_true")
    ap.add_argument("--no-tracker", action="store_true")
    args = ap.parse_args()

    print("1. screen and grid")
    sc = screens()
    if sc:
        for i, (w, h) in enumerate(sc):
            print(f"        screen {i}: {w} x {h}" + ("   <- stimulus" if i == args.screen else ""))
    if args.screen < len(sc):
        w, h = sc[args.screen]
        s = PerimetrySettings(extent=H24_2, screen_width_mm=args.width_mm,
                              viewing_distance_mm=args.distance_mm)
        problem = fit_problem(s, w, h)
        report(problem is None, f"the 24-2 fits at {args.distance_mm:.0f} mm on a "
               f"{args.width_mm:.0f} mm wide screen", problem or "")
    else:
        report(False, f"screen {args.screen} not found", "check the cable / --screen")

    print("2. photometer")
    if args.no_photometer:
        print("        skipped")
    else:
        try:
            sys.path.insert(0, str(HERE / "tools"))
            from calibrate_photometer import ColorCalMeter
            m = ColorCalMeter(None)
            lum = m.read()
            report(True, f"ColorCAL answers (serial {m.serial}); reading now {lum:.3f} cd/m2")
        except SystemExit as e:
            report(False, "ColorCAL", str(e))
        except Exception as e:                          # noqa: BLE001 -- shown to the user
            report(False, "ColorCAL", f"{type(e).__name__}: {e}")

    print("3. luminance scale")
    measured = [s for s in store.ScaleStore().all() if len(s.table) == 101]
    if measured:
        newest = max(measured, key=lambda s: s.created)
        report(True, f"measured scale '{newest.name}' ({newest.created}); choose it in Step 2")
    else:
        report(False, "no ColorCAL-measured luminance scale yet",
               "python/specvis.sh calibrate --name \"Lab screen\" --uniformity")

    try:
        import tobii_research as tr              # the app's own Python has the SDK
    except ImportError:
        tr = None
    print("4. eye tracker" + (" (Tobii SDK in the app)" if tr else " (bridge)"))
    if args.no_tracker:
        print("        skipped")
    elif tr is not None:
        trackers = tr.find_all_eyetrackers()
        if not trackers:
            report(False, "no Tobii eye tracker found",
                   "check the USB cable, and that Tobii Pro Eye Tracker Manager sees it; "
                   "close Tobii Pro Lab")
        else:
            t = trackers[0]
            got = []
            cb = lambda d: got.append(bool(d.get("right_gaze_point_validity")  # noqa: E731
                                           or d.get("left_gaze_point_validity")))
            t.subscribe_to(tr.EYETRACKER_GAZE_DATA, cb, as_dictionary=True)
            time.sleep(2.0)
            t.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, cb)
            report(len(got) > 0, f"{t.model} ({t.serial_number}): {len(got) / 2:.0f} samples/s, "
                   f"{sum(got)} with a valid eye (look at the screen to see valid ones)",
                   "no gaze data: check the tracker in Eye Tracker Manager")
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("127.0.0.1", BRIDGE_PORT))
            sock.settimeout(0.2)
            n, valid, model = 0, 0, ""
            end = time.monotonic() + 2.0
            while time.monotonic() < end:
                try:
                    msg = json.loads(sock.recv(4096).decode("utf-8"))
                except socket.timeout:
                    continue
                n += 1
                valid += bool((msg.get("r") or [0, 0, 0])[2] or (msg.get("l") or [0, 0, 0])[2])
                model = msg.get("model", model)
            report(n > 0, f"gaze from the bridge: {n / 2:.0f} samples/s, {valid} valid "
                   f"({model or '?'})",
                   "start it in another terminal: python/specvis.sh tracker-bridge")
        except OSError as e:
            report(False, f"port {BRIDGE_PORT}", f"in use? close other ReFreM windows ({e})")
        finally:
            sock.close()

    print()
    print("READY" if all(results) else "NOT READY: see the TODO lines above")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
