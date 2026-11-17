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
fractions) | "cal_compute" | "cal_leave" | "ping" | "display_area" | "set_display_area" (with
"top_left", "top_right", "bottom_left" as [x, y, z] mm)}) and gets {"id": n, "ok": ...} back.
The display area is the screen's position as the tracker knows it (Tobii user coordinates).

Each UDP gaze packet is JSON: {"t": device time (us), "l": [x, y, valid], "r": [x, y, valid],
"lz": mm, "rz": mm, "model": ..., "serial": ...}: x and y as fractions of the display area
(0-1, top-left), null when invalid; lz / rz how far each eye is from the tracker (the z of
the SDK's gaze origin, in mm), null when invalid; lo / ro each eye's full position
([x, y, z] mm in Tobii's User Coordinate System: origin at the tracker's front centre,
x to the person's right, y up, z towards the person), for the head-positioning screen;
lu / ru each eye's place in the tracker's view from Tobii's user position guide ([x, y, z],
0-1, 0.5 = the middle; null when not valid); lop / rop each eye's lid opening (mm, eye
openness; null when not valid); lp / rp pupil diameter (mm). A tracker notification
(connection lost / restored, calibration or display area changed, device warnings, stream
errors) is sent at once as {"t": ..., "event": name, "info": ...}.

"cal_enter" may carry "eye": "left" | "right" for Tobii's monocular calibration of that eye
only; the reply's "mode" says what was done ("both eyes" when the SDK cannot).
"""

import argparse
import json
import math
import socket
import sys
import time


def area_info(a: dict) -> dict:
    """Corners plus width and height (mm) of a display area."""
    tl, tr_, bl = a["top_left"], a["top_right"], a["bottom_left"]
    width = math.dist(tl, tr_)
    height = math.dist(tl, bl)
    return dict(top_left=list(tl), top_right=list(tr_), bottom_left=list(bl),
                width_mm=round(width, 1), height_mm=round(height, 1))


class FakeSDK:
    """The tobii_research names calibration_ok needs, as the SDK's own strings, so the
    --fake bridge answers a one-eye calibration the way the real tracker does."""
    CALIBRATION_STATUS_SUCCESS = "calibration_status_success"
    CALIBRATION_STATUS_SUCCESS_LEFT_EYE = "calibration_status_success_left_eye"
    CALIBRATION_STATUS_SUCCESS_RIGHT_EYE = "calibration_status_success_right_eye"
    CALIBRATION_STATUS_FAILURE = "calibration_status_failure"
    SELECTED_EYE_LEFT, SELECTED_EYE_RIGHT = "selected_eye_left", "selected_eye_right"


def calibration_ok(tr, status, mono) -> bool:
    """Whether a calibration answer is a success. Lab report 16 Nov: a monocular
    calibration (ScreenBasedMonocularCalibration) answers "success, left/right eye", never
    plain "success" -- comparing with plain success only marked every one-eye calibration
    as failed. ``mono`` is the eye being calibrated (tr.SELECTED_EYE_*), None for both;
    a success for the other eye is not a success."""
    if status == tr.CALIBRATION_STATUS_SUCCESS:
        return True
    if mono is not None and mono == getattr(tr, "SELECTED_EYE_LEFT", object()):
        return status == getattr(tr, "CALIBRATION_STATUS_SUCCESS_LEFT_EYE", object())
    if mono is not None and mono == getattr(tr, "SELECTED_EYE_RIGHT", object()):
        return status == getattr(tr, "CALIBRATION_STATUS_SUCCESS_RIGHT_EYE", object())
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--serial", default=None, help="use this tracker if several are found")
    ap.add_argument("--fake", action="store_true",
                    help="no tracker: send a steady gaze on the screen centre (testing)")
    ap.add_argument("--fake-distance-mm", type=float, default=520.0,
                    help="with --fake: how far the pretend eyes are (mm)")
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
        dist = [args.fake_distance_mm]
        lid = [10.0]
        area = dict(top_left=[-265.4, 297.6, 97.7], top_right=[276.9, 297.6, 97.7],
                    bottom_left=[-265.4, 10.9, -6.7])
        mode = ["both eyes"]
        fake_mono = [None]
        forced = [None]                                # tests: a calibration answer to give

        def fake_handle(msg):
            if msg.get("cmd") == "display_area":
                return dict(ok=True, **area_info(area))
            if msg.get("cmd") == "set_display_area":
                for k in ("top_left", "top_right", "bottom_left"):
                    area[k] = [float(c) for c in msg[k]]
                return dict(ok=True, **area_info(area))
            if msg.get("cmd") == "cal_enter":
                mode[0] = f"{msg['eye']} eye only" if msg.get("eye") else "both eyes"
                fake_mono[0] = {"left": FakeSDK.SELECTED_EYE_LEFT,
                                "right": FakeSDK.SELECTED_EYE_RIGHT}.get(msg.get("eye"))
                return dict(ok=True, mode=mode[0])
            if msg.get("cmd") == "fake_cal_status":    # tests: answer this from now on
                forced[0] = msg.get("status")          # (None: the normal answer again)
                return dict(ok=True)
            if msg.get("cmd") in ("cal_point", "cal_compute"):
                # As the tracker: a one-eye calibration answers "success, <that> eye".
                status = forced[0] or {FakeSDK.SELECTED_EYE_LEFT:
                                       FakeSDK.CALIBRATION_STATUS_SUCCESS_LEFT_EYE,
                                       FakeSDK.SELECTED_EYE_RIGHT:
                                       FakeSDK.CALIBRATION_STATUS_SUCCESS_RIGHT_EYE}.get(
                    fake_mono[0], FakeSDK.CALIBRATION_STATUS_SUCCESS)
                return dict(ok=calibration_ok(FakeSDK, status, fake_mono[0]), status=status,
                            points=5, mode=mode[0])
            if msg.get("cmd") == "fake_look":          # tests: where the pretend eye looks
                look[:] = [float(msg["x"]), float(msg["y"])]
            if msg.get("cmd") == "fake_distance":      # tests: the pretend eye moves
                dist[0] = float(msg["mm"])
            if msg.get("cmd") == "fake_lid":           # tests: the pretend lid opening (mm)
                lid[0] = msg["mm"]
            if msg.get("cmd") == "fake_event":         # tests: a tracker notification
                send({"t": int(time.time() * 1e6), "event": msg["event"], "info": {}})
            return dict(ok=True, status="calibration_status_success", points=5, mode=mode[0])

        print(f"FAKE gaze -> 127.0.0.1:{args.port}; commands on {args.port + 1}. "
              "Ctrl-C to stop.")
        try:
            while True:
                send({"t": int(time.time() * 1e6), "l": [look[0], look[1], 1],
                      "r": [look[0], look[1], 1], "lz": dist[0], "rz": dist[0],
                      "lo": [-32.0, 170.0, dist[0]], "ro": [32.0, 170.0, dist[0]],
                      "lu": [0.55, 0.5, 0.5], "ru": [0.45, 0.5, 0.5],
                      "lop": lid[0], "rop": lid[0],
                      "model": "FAKE", "serial": "none"})
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

    def depth(data, side):
        """How far the eye is from the tracker (mm): z of the gaze origin."""
        if not data.get(f"{side}_gaze_origin_validity"):
            return None
        z = data.get(f"{side}_gaze_origin_in_user_coordinate_system", (float("nan"),) * 3)[2]
        return None if math.isnan(z) else round(float(z), 1)

    def origin(data, side):
        """The eye's position (mm, User Coordinate System), for positioning."""
        if not data.get(f"{side}_gaze_origin_validity"):
            return None
        v = data.get(f"{side}_gaze_origin_in_user_coordinate_system")
        if not v or any(math.isnan(c) for c in v):
            return None
        return [round(float(c), 1) for c in v]

    def box(data, side):
        """The eye's place in the track box (0-1 on each axis, 0.5 = the middle)."""
        if not data.get(f"{side}_gaze_origin_validity"):
            return None
        v = data.get(f"{side}_gaze_origin_in_trackbox_coordinate_system")
        if not v or any(math.isnan(c) for c in v):
            return None
        return [round(float(c), 3) for c in v]

    def pupil(data, side):
        if not data.get(f"{side}_pupil_validity"):
            return None
        p = data.get(f"{side}_pupil_diameter", float("nan"))
        return None if math.isnan(p) else round(float(p), 2)

    # The latest user position guide and eye openness, sent with each gaze sample.
    latest = dict(lu=None, ru=None, lop=None, rop=None)

    def on_user_position(data):
        """Tobii's user position guide: 0-1 per axis, 0.5 = the middle of the view (SDK
        2.1 no longer sends track-box coordinates with the gaze)."""
        for short, side in (("l", "left"), ("r", "right")):
            p = data.get(f"{side}_user_position")
            latest[short + "u"] = (None if not data.get(f"{side}_user_position_validity")
                                   or not p or any(math.isnan(c) for c in p)
                                   else [round(float(c), 3) for c in p])

    def on_openness(data):
        for short, side in (("l", "left"), ("r", "right")):
            v = data.get(f"{side}_eye_openness_value")
            latest[short + "op"] = (None if not data.get(f"{side}_eye_validity") or v is None
                                    or math.isnan(v) else round(float(v), 2))

    def on_gaze(data):
        packet = {"t": data.get("system_time_stamp"), "l": point(data, "left"),
                  "r": point(data, "right"), "lz": depth(data, "left"),
                  "rz": depth(data, "right"),
                  "lo": origin(data, "left"), "ro": origin(data, "right"),
                  "lb": box(data, "left"), "rb": box(data, "right"),
                  "lp": pupil(data, "left"), "rp": pupil(data, "right"),
                  "model": model, "serial": serial}
        packet.update(latest)
        if not has_openness:
            del packet["lop"], packet["rop"]
        send(packet)

    def on_event(name):
        def handler(data):
            try:
                info = json.loads(json.dumps(data, default=str))
            except (TypeError, ValueError):
                info = str(data)
            t = data.get("system_time_stamp") if isinstance(data, dict) else None
            send({"t": t, "event": name, "info": info})
            print(f"  tracker event: {name} {info}", flush=True)
        return handler

    subs = []

    def subscribe(stream, callback):
        if stream is None:
            return False
        try:
            tracker.subscribe_to(stream, callback, as_dictionary=True)
        except Exception as e:                          # noqa: BLE001 -- not on this model
            print(f"  (could not subscribe to {stream}: {e})")
            return False
        subs.append((stream, callback))
        return True

    calib = tr.ScreenBasedCalibration(tracker)
    mono = None
    in_cal = [False]

    def handle(msg):
        c = msg.get("cmd")
        if c == "ping":
            return dict(ok=True, model=model, serial=serial)
        if c == "display_area":
            da = tracker.get_display_area()
            return dict(ok=True, **area_info(dict(top_left=da.top_left, top_right=da.top_right,
                                                  bottom_left=da.bottom_left)))
        if c == "set_display_area":
            corners = {k: tuple(float(v) for v in msg[k])
                       for k in ("top_left", "top_right", "bottom_left")}
            tracker.set_display_area(tr.DisplayArea(corners))
            da = tracker.get_display_area()
            return dict(ok=True, **area_info(dict(top_left=da.top_left, top_right=da.top_right,
                                                  bottom_left=da.bottom_left)))
        nonlocal calib, mono

        def cal_ok(status) -> bool:
            return calibration_ok(tr, status, mono)

        if c == "cal_enter":
            if in_cal[0]:
                calib.leave_calibration_mode()
                in_cal[0] = False
            eye = msg.get("eye")
            mono = None
            if eye in ("left", "right") and hasattr(tr, "ScreenBasedMonocularCalibration"):
                calib = tr.ScreenBasedMonocularCalibration(tracker)
                mono = tr.SELECTED_EYE_LEFT if eye == "left" else tr.SELECTED_EYE_RIGHT
            else:
                calib = tr.ScreenBasedCalibration(tracker)
            calib.enter_calibration_mode()
            in_cal[0] = True
            return dict(ok=True, mode=f"{eye} eye only" if mono is not None else "both eyes")
        if c == "cal_point":
            x, y = float(msg["x"]), float(msg["y"])
            a = (x, y) if mono is None else (x, y, mono)
            status = calib.collect_data(*a)
            if not cal_ok(status):                           # Tobii: try a point once more
                status = calib.collect_data(*a)
            return dict(ok=cal_ok(status), status=str(status))
        if c == "cal_compute":
            res = calib.compute_and_apply()
            return dict(ok=cal_ok(res.status), status=str(res.status),
                        points=len(res.calibration_points),
                        mode="both eyes" if mono is None else "one eye only")
        if c == "cal_leave":
            if in_cal[0]:
                calib.leave_calibration_mode()
                in_cal[0] = False
            return dict(ok=True)
        return dict(ok=False, error=f"unknown command {c!r}")

    caps = getattr(tracker, "device_capabilities", ())
    has_openness = (getattr(tr, "CAPABILITY_HAS_EYE_OPENNESS_DATA", None) in caps and
                    subscribe(getattr(tr, "EYETRACKER_EYE_OPENNESS_DATA", None), on_openness))
    has_upg = subscribe(getattr(tr, "EYETRACKER_USER_POSITION_GUIDE", None), on_user_position)
    for name in ("CONNECTION_LOST", "CONNECTION_RESTORED", "CALIBRATION_CHANGED",
                 "DISPLAY_AREA_CHANGED", "GAZE_OUTPUT_FREQUENCY_CHANGED",
                 "EYE_TRACKING_MODE_CHANGED", "DEVICE_FAULTS", "DEVICE_WARNINGS"):
        subscribe(getattr(tr, "EYETRACKER_NOTIFICATION_" + name, None), on_event(name.lower()))
    subscribe(getattr(tr, "EYETRACKER_STREAM_ERRORS", None), on_event("stream_error"))
    subscribe(tr.EYETRACKER_GAZE_DATA, on_gaze)
    print(f"User position guide: {'yes' if has_upg else 'NO'};  eye openness: "
          f"{'yes' if has_openness else 'no'}")
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
        for stream, callback in subs:
            try:
                tracker.unsubscribe_from(stream, callback)
            except Exception:                           # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
