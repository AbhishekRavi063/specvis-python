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

The app can also run the Tobii calibration through the bridge: it sends JSON commands to
127.0.0.1:5556 ({"id": n, "cmd": "cal_enter" | "cal_point" (with "x", "y" as display-area
fractions) | "cal_compute" | "cal_leave" | "ping"}) and gets {"id": n, "ok": ..., ...} back.

Each UDP gaze packet is JSON: {"t": device time (us), "l": [x, y, valid], "r": [x, y, valid],
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
    cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)       # commands from the app
    cmd.bind(("127.0.0.1", args.port + 1))

    def send(sample: dict) -> None:
        sock.sendto(json.dumps(sample).encode("utf-8"), dest)
        count[0] += 1

    def serve(handle, wait_s: float) -> None:
        """Answer every command waiting (or arriving within wait_s)."""
        cmd.settimeout(wait_s)
        while True:
            try:
                data, addr = cmd.recvfrom(4096)
            except (socket.timeout, BlockingIOError):
                return
            try:
                msg = json.loads(data.decode("utf-8"))
                reply = handle(msg)
            except Exception as e:                     # noqa: BLE001 -- told to the app
                msg, reply = locals().get("msg", {}), dict(ok=False, error=str(e))
            reply["id"] = msg.get("id") if isinstance(msg, dict) else None
            cmd.sendto(json.dumps(reply).encode("utf-8"), addr)
            print(f"  command {msg.get('cmd') if isinstance(msg, dict) else '?'}: "
                  f"{'ok' if reply.get('ok') else reply}", flush=True)
            cmd.settimeout(0.0)

    if args.fake:
        look = [0.5, 0.5]

        def fake_handle(msg):
            if msg.get("cmd") == "fake_look":          # tests: where the pretend eye looks
                look[:] = [float(msg["x"]), float(msg["y"])]
            return dict(ok=True, status="calibration_status_success", points=5)

        print(f"FAKE gaze -> 127.0.0.1:{args.port}; commands on {args.port + 1}. "
              "Ctrl-C to stop.")
        try:
            while True:
                send({"t": int(time.time() * 1e6), "l": [look[0], look[1], 1],
                      "r": [look[0], look[1], 1], "model": "FAKE", "serial": "none"})
                serve(fake_handle, 1 / 120)
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

    calib = tr.ScreenBasedCalibration(tracker)
    in_cal = [False]

    def handle(msg):
        c = msg.get("cmd")
        if c == "ping":
            return dict(ok=True, model=model, serial=serial)
        if c == "cal_enter":
            if not in_cal[0]:
                calib.enter_calibration_mode()
                in_cal[0] = True
            return dict(ok=True)
        if c == "cal_point":
            x, y = float(msg["x"]), float(msg["y"])
            status = calib.collect_data(x, y)
            if status != tr.CALIBRATION_STATUS_SUCCESS:      # Tobii: try a point once more
                status = calib.collect_data(x, y)
            return dict(ok=status == tr.CALIBRATION_STATUS_SUCCESS, status=str(status))
        if c == "cal_compute":
            res = calib.compute_and_apply()
            return dict(ok=res.status == tr.CALIBRATION_STATUS_SUCCESS, status=str(res.status),
                        points=len(res.calibration_points))
        if c == "cal_leave":
            if in_cal[0]:
                calib.leave_calibration_mode()
                in_cal[0] = False
            return dict(ok=True)
        return dict(ok=False, error=f"unknown command {c!r}")

    tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, on_gaze, as_dictionary=True)
    print(f"Sending gaze to 127.0.0.1:{args.port}; commands on {args.port + 1}. "
          "Leave this running; Ctrl-C to stop.")
    try:
        last, t_last = 0, time.time()
        while True:
            serve(handle, 0.25)
            if time.time() - t_last >= 2.0:
                n = count[0]
                print(f"  {(n - last) / (time.time() - t_last):6.1f} samples/s", flush=True)
                last, t_last = n, time.time()
    except KeyboardInterrupt:
        pass
    finally:
        if in_cal[0]:
            calib.leave_calibration_mode()
        tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, on_gaze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
