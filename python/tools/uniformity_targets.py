"""Shared by screen_uniformity.py and calibrate_photometer.py --uniformity: a red target
ring with a cross shows where the ColorCAL goes; the operator places it and presses
ENTER; the ring and text disappear (plain grey only), and the meter reads.

Keys at a target: ENTER / keypad ENTER / SPACE = measure here; R = measure the previous
place again; ESC = stop (the places already measured are kept).

The screen is drawn by ``PsychoUI``; ``measure_places`` only talks to a small interface
(size, show_target, wait_key, show_grey, keep_alive), so the tests drive it with a fake.
"""

from __future__ import annotations

import statistics
import threading
import time

from specvis.core import photometry

KEYS = {"return": "enter", "num_enter": "enter", "enter": "enter", "space": "enter",
        "r": "repeat", "escape": "escape"}


class Stopped(Exception):
    """ESC while measuring."""


class PsychoUI:
    """The target screens on a PsychoPy window (made by the caller)."""

    def __init__(self, win, width_mm: float, ring_mm: float, on_black: bool = False,
                 auto: bool = False):
        from psychopy import visual
        self.win, self.on_black, self.auto = win, on_black, auto
        scale = float(getattr(win, "contentScaleFactor", 1.0) or 1.0)
        # 'pix' units are logical points; win.size is physical pixels on a Retina Mac.
        self.size = (win.size[0] / scale, win.size[1] / scale)
        self.px_per_mm = self.size[0] / width_mm
        r = ring_mm / 2.0 * self.px_per_mm
        red = (1.0, -1.0, -1.0)
        self.ring = visual.Circle(win, radius=r, edges=128, units="pix", fillColor=None,
                                  lineColor=red, lineWidth=6)
        self.cross = [visual.Line(win, start=(-0.4 * r, 0), end=(0.4 * r, 0), units="pix",
                                  lineColor=red, lineWidth=3),
                      visual.Line(win, start=(0, -0.4 * r), end=(0, 0.4 * r), units="pix",
                                  lineColor=red, lineWidth=3)]
        self.text = visual.TextStim(win, "", units="pix", color=0.8,
                                    height=self.size[1] * 0.028, wrapWidth=self.size[0] * 0.55)
        self.grey = -1.0

    def set_ring_mm(self, ring_mm: float) -> None:
        r = ring_mm / 2.0 * self.px_per_mm
        self.ring.radius = r
        self.cross[0].start, self.cross[0].end = (-0.4 * r, 0), (0.4 * r, 0)
        self.cross[1].start, self.cross[1].end = (0, -0.4 * r), (0, 0.4 * r)

    def set_grey(self, level: int) -> None:
        self.grey = level / 255.0 * 2.0 - 1.0

    def show_target(self, message: str, x: float, y: float) -> None:
        w, h = self.size
        for s in [self.ring] + self.cross:
            s.pos = (x, y)
        # the text on the half of the screen away from the ring
        if y > 0:
            self.text.pos = (0, -h * 0.22)
        elif y < 0:
            self.text.pos = (0, h * 0.22)
        elif x != 0:
            self.text.pos = (-w * 0.25 if x > 0 else w * 0.25, 0)
        else:
            self.text.pos = (0, -h * 0.28)
        self.text.text = message
        self.win.color = -1.0 if self.on_black else self.grey
        for _ in range(2):                 # the window colour shows from the next frame
            self.ring.draw()
            for c in self.cross:
                c.draw()
            self.text.draw()
            self.win.flip()

    def wait_key(self) -> str:
        if self.auto:
            return "enter"
        from psychopy import event
        event.clearEvents()
        key = event.waitKeys(keyList=list(KEYS))[0]
        return KEYS[key]

    def show_grey(self) -> None:
        self.win.color = self.grey
        self.win.flip()
        self.win.flip()

    def keep_alive(self, until: float = None, worker=None) -> None:
        """Redraw while waiting so Windows never marks the window 'Not responding'."""
        from psychopy import event
        while (worker is not None and worker.is_alive()) or (until is not None
                                                             and time.time() < until):
            self.win.flip()
            if "escape" in event.getKeys(["escape"]):
                raise Stopped()


def read_meter(ui, meter, repeats: int):
    """`repeats` readings in a thread while the window keeps redrawing."""
    out = {}

    def work():
        try:
            out["vals"] = [float(meter.read()) for _ in range(repeats)]
        except BaseException as e:                 # noqa: BLE001 -- raised below
            out["error"] = e

    th = threading.Thread(target=work, daemon=True)
    th.start()
    ui.keep_alive(worker=th)
    th.join()
    if "error" in out:
        raise out["error"]
    return out["vals"]


def measure_places(ui, meter, places, fx: float, fy: float, grey_level: int,
                   repeats: int = 3, settle_s: float = 1.5, log=print):
    """Each place in turn: target, ENTER, plain grey, settle, read. Returns
    ({place: [readings]}, stopped). R at a target measures the previous place again."""
    ui.set_grey(grey_level)
    if hasattr(meter, "set_level"):              # the simulated meter
        meter.set_level(grey_level)
    results, i, n = {}, 0, len(places)
    try:
        while i < n:
            place = places[i]
            x, y = photometry.place_pixels(place, ui.size[0], ui.size[1], fx, fy)
            again = f"\n(R = measure {places[i - 1].upper()} again)" if i > 0 else ""
            ui.show_target(f"Place {i + 1} of {n}: {place.upper()}\n\n"
                           "Put the ColorCAL flat on the screen, centred on the cross inside "
                           "the ring. Touch lightly.\n\nThen press ENTER.   (ESC stops)" + again,
                           x, y)
            key = ui.wait_key()
            if key == "escape":
                raise Stopped()
            target = i
            if key == "repeat":
                if i == 0:
                    continue
                target = i - 1
                place = places[target]
                x, y = photometry.place_pixels(place, ui.size[0], ui.size[1], fx, fy)
                log(f"  measuring {place.upper()} again: put the meter back there "
                    "(its target is not shown)")
                ui.show_target(f"Again: {place.upper()}\n\nPut the ColorCAL back on the cross, "
                               "then press ENTER.   (ESC stops)", x, y)
                if ui.wait_key() == "escape":
                    raise Stopped()
            ui.show_grey()
            log(f"  {place}: measuring... don't move the meter", flush=True)
            ui.keep_alive(until=time.time() + settle_s)
            vals = read_meter(ui, meter, repeats)
            results[place] = vals
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            spread = 100.0 * (max(vals) - min(vals)) / m if m > 0 else 0.0
            log(f"  {place:13s} {m:8.3f} cd/m2   spread {spread:.1f} %  sd {sd:.3f}   "
                f"(readings {', '.join(f'{v:.3f}' for v in vals)})", flush=True)
            if photometry.unsteady(vals):
                log("  WARNING: readings unsteady: meter moved or pressed too hard? "
                    "(R = repeat this place)", flush=True)
            i = target + 1 if key == "enter" else i
    except Stopped:
        return results, True
    return results, False
