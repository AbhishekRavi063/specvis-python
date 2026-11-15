"""Where the eye is looking, during perimetry: the fixation monitor.

The procedure asks a monitor two things: where the eye is **now** (to start a patch
only once the eye is on the +), and where it was **during** a patch (to throw away a
trial in which the eye moved). Gaze is given in visual-field degrees from the +,
x right and y up, or None when there is no usable sample (a blink, tracking lost).

Three monitors:

* :class:`NoMonitor`: assumes perfect fixation. Used with the blind-spot check, or with
  no check at all.
* :class:`MouseMonitor`: the mouse pointer stands in for the eye. Only for trying the
  procedure without a tracker: keep the pointer on the + and the test runs; move it
  away and trials are thrown away.
* :class:`TobiiMonitor`: a Tobii Pro eye tracker (built for the Pro Fusion) through the
  Tobii Pro SDK. **Not yet tried on the device.** The tracker's display area must be
  set up, and the eye calibrated, in Tobii Pro Eye Tracker Manager before the session.
  The SDK (``tobii-research``) is published for **Python 3.10 only**, and the app runs
  on 3.11; so unless the SDK imports here, gaze comes from ``tools/tobii_bridge.py``,
  run in its own Python 3.10 environment, which forwards every sample over UDP on this
  computer (127.0.0.1, port ``BRIDGE_PORT``).

:class:`ScriptedMonitor` gives tests a gaze path without any hardware.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

Gaze = Optional[Tuple[float, float]]           # field degrees from the +, or None
Sample = Tuple[float, Gaze]                    # (time.perf_counter(), gaze)

FIX_BLIND_SPOT, FIX_NONE, FIX_TRACKER, FIX_MOUSE = "blind spot", "none", "eye tracker", "mouse"
FIXATION_MODES = (FIX_BLIND_SPOT, FIX_NONE, FIX_TRACKER, FIX_MOUSE)
GAZE_MODES = (FIX_TRACKER, FIX_MOUSE)          # the modes that watch the eye
BRIDGE_PORT = 5555                             # tools/tobii_bridge.py -> the app (UDP)
BRIDGE_CMD_PORT = BRIDGE_PORT + 1              # the app -> the bridge: calibration commands


def screen_plane(area: Optional[dict]):
    """(centre, right, up, normal) of the screen from a display area's corners, or None.
    The normal points out of the screen, towards the person."""
    try:
        import numpy as np
        tl, tr_, bl = (np.array(area[k], dtype=float)
                       for k in ("top_left", "top_right", "bottom_left"))
    except (KeyError, TypeError, ValueError):
        return None
    right, up = tr_ - tl, tl - bl
    if not (np.linalg.norm(right) > 0 and np.linalg.norm(up) > 0):
        return None
    right, up = right / np.linalg.norm(right), up / np.linalg.norm(up)
    normal = np.cross(right, up)
    if normal[2] < 0:                       # towards the person (+z in Tobii's UCS)
        normal = -normal
    return (bl + (tr_ - tl) / 2.0 + (tl - bl) / 2.0, right, up, normal)


def distance_to_screen(origin, plane) -> Optional[float]:
    """The eye's square-on distance from the screen plane (mm); with no plane, the
    tracker's z (distance from the tracker, which reads about 4-5 cm more on the lab's
    tilted tracker)."""
    if origin is None:
        return None
    if plane is not None and not any(math.isnan(c) for c in origin[:3]):
        import numpy as np
        return float(abs((np.array(origin[:3], dtype=float) - plane[0]) @ plane[3]))
    z = origin[2]
    return None if math.isnan(z) else float(z)


class FixationMonitor:
    """The interface the procedure uses. Subclasses fill :meth:`_read` or push samples."""

    kind = FIX_NONE
    watches_gaze = False

    def __init__(self) -> None:
        self.samples: List[Sample] = []
        self._to_deg: Optional[Callable[[float, float], Tuple[float, float]]] = None
        self._lock = threading.Lock()

    def begin(self, backend, to_deg: Callable[[float, float], Tuple[float, float]]) -> None:
        """Start watching. ``to_deg`` turns screen pixels (top-left, y down) into field
        degrees from the +."""
        self._to_deg = to_deg

    def end(self) -> None:
        pass

    def _read(self) -> Gaze:
        return (0.0, 0.0)

    def current(self) -> Gaze:
        """Gaze now. Also kept as a sample, so polled monitors build a record too."""
        g = self._read()
        self._add(time.perf_counter(), g)
        return g

    def _add(self, t: float, g: Gaze) -> None:
        with self._lock:
            self.samples.append((t, g))

    def between(self, t0: float, t1: float) -> List[Gaze]:
        with self._lock:
            return [g for t, g in self.samples if t0 <= t <= t1]

    def longest_gap_ms(self, t0: float, t1: float) -> float:
        """The longest stretch between t0 and t1 with no valid gaze (ms), from the last
        valid sample before the gap (or t0) to the next valid one (or t1); 0 when no
        sample was missing. A blink lasts roughly 100-400 ms; lost single samples a few ms."""
        with self._lock:
            pts = [(t, g) for t, g in self.samples if t0 <= t <= t1]
        if not any(g is None for _, g in pts):
            return 0.0
        longest, gap_start, last_valid = 0.0, None, t0
        for t, g in pts:
            if g is None:
                if gap_start is None:
                    gap_start = last_valid
            else:
                if gap_start is not None:
                    longest, gap_start = max(longest, t - gap_start), None
                last_valid = t
        if gap_start is not None:
            longest = max(longest, t1 - gap_start)
        return 1000.0 * longest

    def patch_shown(self, t0: float, t1: float) -> None:
        """Called after every patch with its start and end (time.perf_counter())."""

    # -- calibration (eye trackers that can be calibrated from the app) -----------------
    calibration_supported = False

    def cal_enter(self) -> None:
        raise NotImplementedError

    def cal_point(self, x: float, y: float) -> bool:
        """Collect calibration data while the person looks at (x, y): fractions of the
        display area, from the top-left. True when the tracker accepted the point."""
        raise NotImplementedError

    def cal_compute(self) -> dict:
        """Compute and apply the calibration: {"ok": bool, "status": str, "points": n}."""
        raise NotImplementedError

    def cal_leave(self) -> None:
        raise NotImplementedError

    def eye_distance_mm(self, window_s: float = 0.5) -> Optional[float]:
        """How far the tested eye is from the screen (mm), if the tracker can tell."""
        return None

    def display_area(self) -> Optional[dict]:
        """Where the tracker thinks the screen is: {"width_mm", "height_mm", corners}.
        None when there is no tracker to ask."""
        return None

    def set_screen_plane(self, area: Optional[dict]) -> None:
        """Tell the monitor where the screen is (the tracker's display area: corners in
        mm, User Coordinate System), so the eye's distance is measured to the screen
        plane, square on, rather than to the tracker. Ignored without corners."""
        self._plane = screen_plane(area)

    def head_position(self) -> Optional[dict]:
        """For the positioning screen: {"left": (x, y, z) mm or None, "right": ...,
        "distance_mm"}, in Tobii's User Coordinate System (origin at the tracker's front
        centre, x to the person's right, y up, z towards the person). None when the
        tracker cannot tell."""
        return None

    def target_shown(self, x_deg: float, y_deg: float) -> None:
        """A calibration / validation dot is now at (x_deg, y_deg). Real trackers ignore
        this; simulated observers use it to look there."""

    def describe(self) -> str:
        return "no eye tracking"


class NoMonitor(FixationMonitor):
    """Perfect fixation assumed; records nothing."""

    def current(self) -> Gaze:
        return (0.0, 0.0)

    def between(self, t0: float, t1: float) -> List[Gaze]:
        return [(0.0, 0.0)]


class MouseMonitor(FixationMonitor):
    """The mouse pointer as the eye -- for trying the procedure without a tracker.

    A polled monitor: it has samples only when asked, so the procedure polls it just
    before and just after each patch."""

    kind = FIX_MOUSE
    watches_gaze = True

    def begin(self, backend, to_deg) -> None:
        super().begin(backend, to_deg)
        self._backend = backend
        if backend.pointer_px() is None:
            raise RuntimeError("this display cannot report the mouse position")

    def _read(self) -> Gaze:
        px = self._backend.pointer_px()
        return None if px is None else self._to_deg(*px)

    def describe(self) -> str:
        return "MOUSE stands in for the eye (testing only, not a real fixation check)"


class ScriptedMonitor(FixationMonitor):
    """For tests: gaze from a function of the number of patches shown so far."""

    kind = FIX_TRACKER
    watches_gaze = True

    def __init__(self, gaze_for_patch: Callable[[int], Gaze]):
        super().__init__()
        self.gaze_for_patch = gaze_for_patch
        self.patch = 0

    def current(self) -> Gaze:
        return (0.0, 0.0)                       # always back on the + between patches

    def patch_shown(self, t0: float, t1: float) -> None:
        self._add((t0 + t1) / 2.0, self.gaze_for_patch(self.patch))
        self.patch += 1

    def describe(self) -> str:
        return "scripted gaze (test)"


class TobiiMonitor(FixationMonitor):
    """A Tobii Pro screen-based eye tracker (Pro Fusion) via the Tobii Pro SDK.

    Gaze arrives on the SDK's own thread (120 or 250 samples a second) and is stamped
    with ``time.perf_counter()`` on arrival, so it lines up with the procedure's clock to
    within the SDK's latency (a few ms). Only the tested eye is used: the other eye is
    patched, and its samples are invalid anyway.

    The SDK reports gaze as a fraction of the *display area* set up in Tobii Pro Eye
    Tracker Manager; that area must be the stimulus screen, exactly.
    """

    kind = FIX_TRACKER
    watches_gaze = True

    def __init__(self, eye: str = "R", serial: Optional[str] = None):
        super().__init__()
        self.eye = eye
        self.serial = serial
        self._tracker = None
        self._size = (0, 0)
        self._depths: List[Tuple[float, float]] = []
        self.model = ""

    def begin(self, backend, to_deg) -> None:
        super().begin(backend, to_deg)
        self._size = backend.size_px
        try:
            import tobii_research as tr          # noqa: F401
        except ImportError:
            self._begin_bridge()
            return
        self._tr = tr
        found = tr.find_all_eyetrackers()
        if self.serial:
            found = [t for t in found if t.serial_number == self.serial]
        if not found:
            raise RuntimeError("No Tobii eye tracker found. Check the USB cable and that "
                               "Tobii Pro Eye Tracker Manager can see it.")
        self._tracker = found[0]
        self.model = f"{self._tracker.model} ({self._tracker.serial_number})"
        self._tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self._on_gaze, as_dictionary=True)

    def _on_gaze(self, data: dict) -> None:
        side = "right" if self.eye == "R" else "left"
        other = "left" if side == "right" else "right"
        # Eye distance: the z of the gaze origin (mm from the tracker). The tracker sits on
        # the screen's bottom bezel, so this is the eye's distance from the screen plane.
        for s in (side, other):
            if data.get(f"{s}_gaze_origin_validity"):
                d = distance_to_screen(
                    data.get(f"{s}_gaze_origin_in_user_coordinate_system",
                             (float("nan"),) * 3), getattr(self, "_plane", None))
                if d is not None:
                    with self._lock:
                        self._depths.append((time.perf_counter(), d))
                        if len(self._depths) > 2000:
                            del self._depths[:1000]
                    break
        eyes, boxes = {}, {}
        for s in ("left", "right"):
            o = data.get(f"{s}_gaze_origin_in_user_coordinate_system")
            eyes[s] = (None if not data.get(f"{s}_gaze_origin_validity") or not o
                       or any(math.isnan(c) for c in o) else tuple(float(c) for c in o))
            # Where the eye is in the tracker's track box (0-1 on each axis; 0.5 = the
            # middle), straight from the Tobii SDK.
            b = data.get(f"{s}_gaze_origin_in_trackbox_coordinate_system")
            boxes[s] = (None if eyes[s] is None or not b or any(math.isnan(c) for c in b)
                        else tuple(float(c) for c in b))
        with self._lock:
            self._eyes = (time.perf_counter(), eyes, boxes)
        g: Gaze = None
        if data.get(f"{side}_gaze_point_validity"):
            fx, fy = data[f"{side}_gaze_point_on_display_area"]
            if not (math.isnan(fx) or math.isnan(fy)):
                g = self._to_deg(fx * self._size[0], fy * self._size[1])
        self._add(time.perf_counter(), g)

    def head_position(self) -> Optional[dict]:
        with self._lock:
            stamp, eyes, boxes = getattr(self, "_eyes", (0.0, {}, {}))
        if time.perf_counter() - stamp > 0.2:
            eyes, boxes = {}, {}
        return dict(left=eyes.get("left"), right=eyes.get("right"),
                    left_box=boxes.get("left"), right_box=boxes.get("right"),
                    distance_mm=self.eye_distance_mm(0.3))

    def eye_distance_mm(self, window_s: float = 0.5) -> Optional[float]:
        """Median eye-to-screen distance over the last ``window_s`` seconds (mm), from the
        tracker; the tested eye, or the other one when the tested eye is not seen. Square
        on to the screen once :meth:`set_screen_plane` has been given the display area;
        before that, the distance from the tracker."""
        now = time.perf_counter()
        with self._lock:
            zs = [z for t, z in self._depths if now - t <= window_s]
        if not zs:
            return None
        zs.sort()
        return zs[len(zs) // 2]

    def current(self) -> Gaze:
        """The latest sample, if it is fresh (under 50 ms old)."""
        with self._lock:
            if not self.samples:
                return None
            t, g = self.samples[-1]
        return g if time.perf_counter() - t < 0.05 else None

    # -- the Python 3.10 bridge ----------------------------------------------------------

    calibration_supported = True

    # -- calibration: through the SDK here, or through the bridge -------------------------

    def _cmd(self, cmd: str, timeout_s: float = 15.0, **kw) -> dict:
        """Send a command to the bridge and wait for its answer."""
        self._cmd_id = getattr(self, "_cmd_id", 0) + 1
        msg = dict(id=self._cmd_id, cmd=cmd, **kw)
        sock = self._cmd_sock
        sock.settimeout(0.5)
        sock.sendto(json.dumps(msg).encode("utf-8"), ("127.0.0.1", self._cmd_port))
        end = time.perf_counter() + timeout_s
        while time.perf_counter() < end:
            try:
                reply = json.loads(sock.recv(4096).decode("utf-8"))
            except socket.timeout:
                continue
            if reply.get("id") == self._cmd_id:
                return reply
        raise RuntimeError(f"the eye-tracker bridge did not answer '{cmd}' "
                           "(is python/specvis.sh tracker-bridge still running?)")

    def display_area(self) -> Optional[dict]:
        if getattr(self, "_sock", None) is not None:
            r = self._cmd("display_area", timeout_s=3.0)
            return r if r.get("ok") else None
        da = self._tracker.get_display_area()
        return dict(width_mm=round(float(da.width), 1), height_mm=round(float(da.height), 1),
                    top_left=list(da.top_left), top_right=list(da.top_right),
                    bottom_left=list(da.bottom_left))

    def cal_enter(self) -> None:
        if getattr(self, "_sock", None) is not None:
            self._cmd("cal_enter")
            return
        self._calib = self._tr.ScreenBasedCalibration(self._tracker)
        self._calib.enter_calibration_mode()

    def cal_point(self, x: float, y: float) -> bool:
        if getattr(self, "_sock", None) is not None:
            return bool(self._cmd("cal_point", x=x, y=y).get("ok"))
        tr = self._tr
        status = self._calib.collect_data(x, y)
        if status != tr.CALIBRATION_STATUS_SUCCESS:
            status = self._calib.collect_data(x, y)
        return status == tr.CALIBRATION_STATUS_SUCCESS

    def cal_compute(self) -> dict:
        if getattr(self, "_sock", None) is not None:
            r = self._cmd("cal_compute", timeout_s=30.0)
            return dict(ok=bool(r.get("ok")), status=r.get("status", r.get("error", "")),
                        points=r.get("points", 0))
        res = self._calib.compute_and_apply()
        return dict(ok=res.status == self._tr.CALIBRATION_STATUS_SUCCESS,
                    status=str(res.status), points=len(res.calibration_points))

    def cal_leave(self) -> None:
        if getattr(self, "_sock", None) is not None:
            self._cmd("cal_leave")
            return
        self._calib.leave_calibration_mode()

    def _begin_bridge(self, port: int = BRIDGE_PORT, wait_s: float = 3.0) -> None:
        """Gaze from tools/tobii_bridge.py over UDP. Fails if nothing arrives in time."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind(("127.0.0.1", port))
        except OSError as e:
            self._sock.close()
            raise RuntimeError(f"Port {port} is in use: is another ReFreM test running?") from e
        self._sock.settimeout(0.2)
        first = None
        deadline = time.perf_counter() + wait_s
        while first is None and time.perf_counter() < deadline:
            try:
                first = self._sock.recv(4096)
            except socket.timeout:
                pass
        if first is None:
            self._sock.close()
            raise RuntimeError(
                "No gaze from the eye tracker. Start the bridge first, in its own "
                "terminal:  python/specvis.sh tracker-bridge   (it needs Python 3.10 and "
                "tobii-research: python/specvis.sh tracker-setup)")
        self._handle_packet(first)
        self._cmd_port = port + 1
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _handle_packet(self, data: bytes) -> None:
        try:
            msg = json.loads(data.decode("utf-8"))
        except ValueError:
            return
        if msg.get("model") and not self.model:
            self.model = f"{msg['model']} ({msg.get('serial', '?')}) via bridge"
        data = {}
        for short, long in (("r", "right"), ("l", "left")):
            x, y, valid = msg.get(short) or (None, None, 0)
            data[f"{long}_gaze_point_validity"] = valid
            data[f"{long}_gaze_point_on_display_area"] = (
                float("nan") if x is None else float(x), float("nan") if y is None else float(y))
            z = msg.get(short + "z")
            data[f"{long}_gaze_origin_validity"] = z is not None
            o = msg.get(short + "o")
            data[f"{long}_gaze_origin_in_user_coordinate_system"] = (
                tuple(float(c) for c in o) if o else
                (float("nan"), float("nan"), float("nan") if z is None else float(z)))
            b = msg.get(short + "b")
            data[f"{long}_gaze_origin_in_trackbox_coordinate_system"] = (
                tuple(float(c) for c in b) if b else (float("nan"),) * 3)
        self._on_gaze(data)

    def _listen(self) -> None:
        while self._running:
            try:
                self._handle_packet(self._sock.recv(4096))
            except socket.timeout:
                continue
            except OSError:
                break

    def end(self) -> None:
        if getattr(self, "_sock", None) is not None:
            self._running = False
            self._thread.join(timeout=1.0)
            self._sock.close()
            self._sock = None
            self._cmd_sock.close()
        if self._tracker is not None:
            self._tracker.unsubscribe_from(self._tr.EYETRACKER_GAZE_DATA, self._on_gaze)
            self._tracker = None

    def describe(self) -> str:
        return f"Tobii eye tracker {self.model}".strip()


def monitor_for(mode: str, eye: str) -> FixationMonitor:
    if mode == FIX_TRACKER:
        return TobiiMonitor(eye)
    if mode == FIX_MOUSE:
        return MouseMonitor()
    return NoMonitor()


def judge_patch(gaze: Sequence[Gaze], tolerance_deg: float, min_valid: float = 0.5,
                max_out_frac: float = 0.0) -> Tuple[bool, float, float]:
    """Was the eye on the + during a patch? Returns (ok, largest offset in degrees,
    fraction of valid samples). Too few valid samples (a blink) is not ok either: the
    patch may have been missed with the eye shut. With ``max_out_frac`` > 0, up to that
    share of the valid samples may be off the + (tracker noise, a brief flick) and the
    patch still counts; 0 = every sample must be on it."""
    if not gaze:
        return False, float("nan"), 0.0
    valid = [g for g in gaze if g is not None]
    frac = len(valid) / len(gaze)
    offsets = [math.hypot(*g) for g in valid]
    worst = max(offsets, default=float("nan"))
    out = sum(o > tolerance_deg for o in offsets) / len(offsets) if offsets else 1.0
    ok = frac >= min_valid and bool(offsets) and out <= max_out_frac
    return ok, worst, frac
