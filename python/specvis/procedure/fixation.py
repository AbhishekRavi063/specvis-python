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
Sample = Tuple[float, Gaze]                    # (time.monotonic(), gaze)

FIX_BLIND_SPOT, FIX_NONE, FIX_TRACKER, FIX_MOUSE = "blind spot", "none", "eye tracker", "mouse"
FIXATION_MODES = (FIX_BLIND_SPOT, FIX_NONE, FIX_TRACKER, FIX_MOUSE)
GAZE_MODES = (FIX_TRACKER, FIX_MOUSE)          # the modes that watch the eye
BRIDGE_PORT = 5555                             # tools/tobii_bridge.py -> the app (UDP)


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
        self._add(time.monotonic(), g)
        return g

    def _add(self, t: float, g: Gaze) -> None:
        with self._lock:
            self.samples.append((t, g))

    def between(self, t0: float, t1: float) -> List[Gaze]:
        with self._lock:
            return [g for t, g in self.samples if t0 <= t <= t1]

    def patch_shown(self, t0: float, t1: float) -> None:
        """Called after every patch with its start and end (time.monotonic())."""

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
    with ``time.monotonic()`` on arrival, so it lines up with the procedure's clock to
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
        g: Gaze = None
        if data.get(f"{side}_gaze_point_validity"):
            fx, fy = data[f"{side}_gaze_point_on_display_area"]
            if not (math.isnan(fx) or math.isnan(fy)):
                g = self._to_deg(fx * self._size[0], fy * self._size[1])
        self._add(time.monotonic(), g)

    def current(self) -> Gaze:
        """The latest sample, if it is fresh (under 50 ms old)."""
        with self._lock:
            if not self.samples:
                return None
            t, g = self.samples[-1]
        return g if time.monotonic() - t < 0.05 else None

    # -- the Python 3.10 bridge ----------------------------------------------------------

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
        deadline = time.monotonic() + wait_s
        while first is None and time.monotonic() < deadline:
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
        side = "r" if self.eye == "R" else "l"
        x, y, valid = msg.get(side) or (None, None, 0)
        x = float("nan") if x is None else float(x)
        y = float("nan") if y is None else float(y)
        self._on_gaze({f"{'right' if side == 'r' else 'left'}_gaze_point_validity": valid,
                       f"{'right' if side == 'r' else 'left'}_gaze_point_on_display_area":
                           (x, y)})

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


def judge_patch(gaze: Sequence[Gaze], tolerance_deg: float, min_valid: float = 0.5
                ) -> Tuple[bool, float, float]:
    """Was the eye on the + during a patch? Returns (ok, largest offset in degrees,
    fraction of valid samples). Too few valid samples (a blink) is not ok either: the
    patch may have been missed with the eye shut."""
    if not gaze:
        return False, float("nan"), 0.0
    valid = [g for g in gaze if g is not None]
    frac = len(valid) / len(gaze)
    worst = max((math.hypot(*g) for g in valid), default=float("nan"))
    ok = frac >= min_valid and not math.isnan(worst) and worst <= tolerance_deg
    return ok, worst, frac
