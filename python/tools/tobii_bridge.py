#!/usr/bin/env python3.10
"""Forward Tobii Pro gaze to the ReFreM app over UDP on this computer.

The Tobii Pro SDK (``tobii-research``) is published for Python 3.10 only, and the app
runs on Python 3.11, so the SDK runs here, in its own environment, and every gaze sample
is sent to the app on 127.0.0.1 (port 5555). Uses only the standard library and the SDK.

    python/specvis.sh tracker-setup     # once: python/.venv-tobii with tobii-research
    python/specvis.sh tracker-bridge    # before each session, in its own terminal;
                                        # leave it running, stop with Ctrl-C

Before a session the tracker's display area (the stimulus screen) must be set up, and
the participant calibrated, in Tobii Pro Eye Tracker Manager.

Each UDP packet is JSON: {"t": device time (us), "l": [x, y, valid], "r": [x, y, valid],
"model": ..., "serial": ...}, x and y as fractions of the display area (0-1, top-left),
null when invalid.
"""

import argparse
import json
import math
import socket
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--serial", default=None, help="use this tracker if several are found")
    ap.add_argument("--fake", action="store_true",
                    help="no tracker: send a steady gaze on the screen centre (testing)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ("127.0.0.1", args.port)
    count = [0]

    def send(sample: dict) -> None:
        sock.sendto(json.dumps(sample).encode("utf-8"), dest)
        count[0] += 1

    if args.fake:
        print(f"FAKE gaze -> 127.0.0.1:{args.port} (centre of the screen). Ctrl-C to stop.")
        try:
            while True:
                send({"t": int(time.time() * 1e6), "l": [0.5, 0.5, 1], "r": [0.5, 0.5, 1],
                      "model": "FAKE", "serial": "none"})
                time.sleep(1 / 120)
        except KeyboardInterrupt:
            return 0

    try:
        import tobii_research as tr
    except ImportError:
        print("tobii-research is not installed in this Python "
              f"({sys.version.split()[0]}). Run: python/specvis.sh tracker-setup")
        return 1

    found = tr.find_all_eyetrackers()
    if args.serial:
        found = [t for t in found if t.serial_number == args.serial]
    if not found:
        print("No Tobii eye tracker found. Check the USB cable, and that Tobii Pro Eye "
              "Tracker Manager can see it.")
        return 1
    tracker = found[0]
    model, serial = tracker.model, tracker.serial_number
    print(f"Tracker: {model} ({serial}), {tracker.get_gaze_output_frequency():g} Hz")

    def point(data, side):
        valid = int(bool(data.get(f"{side}_gaze_point_validity")))
        x, y = data.get(f"{side}_gaze_point_on_display_area", (float("nan"),) * 2)
        if not valid or math.isnan(x) or math.isnan(y):
            return [None, None, 0]
        return [x, y, 1]

    def on_gaze(data):
        send({"t": data.get("system_time_stamp"), "l": point(data, "left"),
              "r": point(data, "right"), "model": model, "serial": serial})

    tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, on_gaze, as_dictionary=True)
    print(f"Sending gaze to 127.0.0.1:{args.port}. Leave this running; Ctrl-C to stop.")
    try:
        last = 0
        while True:
            time.sleep(2.0)
            n = count[0]
            print(f"  {(n - last) / 2.0:6.1f} samples/s", flush=True)
            last = n
    except KeyboardInterrupt:
        pass
    finally:
        tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, on_gaze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
