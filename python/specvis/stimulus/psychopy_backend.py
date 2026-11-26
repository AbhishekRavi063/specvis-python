"""The PsychoPy implementation of :class:`~specvis.stimulus.backend.StimulusBackend`.

This is the only module in the package that knows what draws anything. Everything
above the seam -- staircases, Gabor generation, geometry, trial log -- imports none of
it, which is why this file can be replaced without touching the rest.

**Reversal is driven by counting frames.** The loop in :meth:`present_train` decides
which polarity to draw from the frame number alone and never consults a clock. A
wall-clock timer cannot hold the 12.5-20 ms phases the 25-40 Hz condition needs: the
jitter is comparable to the interval being measured.

**Counting convention.** A train of ``N`` frames at ``k`` frames per reversal contains
``N // k`` phase periods. The onset is the first phase, so the number of *inversions*
inside the train is one fewer. ``reversals_presented`` counts **phase periods**, which
is what :meth:`ReversalPlan.expected_reversals` predicts, so the two agree. The SSVEP
fundamental sits at ``refresh / k`` either way.
"""

from __future__ import annotations

import statistics
from typing import List, Optional, Sequence

import numpy as np

from .backend import Fixation, Mark, PresentationReport, StimulusBackend, Texture

#: A frame interval longer than this multiple of one refresh counts as dropped.
DROP_FACTOR = 1.5

#: Frames used to measure the refresh rate before the first trial.
REFRESH_PROBE_FRAMES = 240


def count_dropped(intervals_ms: Sequence[float], refresh_hz: float) -> int:
    """Frames that took longer than 1.5 refreshes.

    Judged against the refresh measured when the window opened, **not** against the
    train's own median. Lab PC, 24 Nov: with the first flat-field version every frame
    took two refreshes (240 Hz ran as 120 Hz), the median doubled with them, and a
    median-based count said 0 dropped. Here such a train counts every frame as dropped.
    The median is used only when no refresh was measured.
    """
    if not intervals_ms:
        return 0
    if refresh_hz == refresh_hz and refresh_hz > 0:        # not NaN
        limit = DROP_FACTOR * 1000.0 / refresh_hz
    else:
        limit = DROP_FACTOR * statistics.median(intervals_ms)
    return sum(1 for i in intervals_ms if i > limit)


class PsychoPyBackend(StimulusBackend):
    """A full-screen PsychoPy window driven frame by frame.

    :param screen: index of the stimulus display. On a two-monitor rig this must be the
                   participant's monitor, not the operator's.
    :param fullscreen: always true for data collection. Windowed mode exists so the
                       geometry can be checked during development, and it gives **no**
                       timing guarantee at all -- the compositor is in the path.
    """

    def __init__(self, screen: int = 0, fullscreen: bool = True,
                 monitor_name: str = "refrem-stimulus"):
        self._screen = screen
        self._fullscreen = fullscreen
        self._monitor_name = monitor_name
        self._win = None
        self._measured_refresh = float("nan")
        self._fixation = None
        self._scale = 1.0
        self.fixation_colour = "white"      # set by the procedure (Specvis step 4)

    # -- lifecycle -------------------------------------------------------------

    def open(self, width_px: int, height_px: int, background_brightness: float) -> None:
        from psychopy import visual

        # PsychoPy's colour axis runs -1..1; ours runs 0..1.
        background = background_brightness * 2.0 - 1.0

        self._win = visual.Window(
            size=(width_px, height_px),
            screen=self._screen,
            fullscr=self._fullscreen,
            allowGUI=False,
            color=background,
            colorSpace="rgb",
            units="pix",
            waitBlanking=True,          # the whole point: flip() blocks until vsync
            useFBO=True,
            checkTiming=False,          # measured explicitly below instead
        )
        # On a Retina display PsychoPy's "pix" units are logical points, while
        # win.size reports physical pixels: 2 physical per point on a MacBook.
        # Everything above the seam works in physical pixels; _to_units converts.
        self._scale = float(getattr(self._win, "contentScaleFactor", 1.0) or 1.0)
        # Hide the mouse pointer. On a MacBook a hand brushing the trackpad while
        # pressing SPACE brings the pointer back, and near the + it looks like a white
        # "L" -- something on screen that is not a stimulus. Hidden again every trial.
        self._win.mouseVisible = False
        self._measured_refresh = self._measure_refresh()

    def set_flat_field(self, flat_field) -> None:
        """Draw the background as the flat-field image (static: computed once, the same
        on every frame), or as one flat grey again with None.

        **The image is drawn ONCE, not every frame.** Lab PC, 24 Nov (Intel UHD 630,
        1920 x 1080 at 240 Hz): drawing a full-screen image every frame takes two
        refreshes (8.3 ms; 6.1 ms even as 8-bit, 5.0 ms even without PsychoPy's float
        buffer), so 240 Hz ran as 120 Hz. With a map chosen the window is therefore never
        cleared: the background stays in PsychoPy's frame buffer, and before new content
        is drawn only the rectangles the last frames drew on (a patch, the +) are painted
        back from the same image. A train then costs what it costs without the map
        (measured 4.17 ms a frame). Whole screens of text are rare and untimed: those
        repaint the full image once (one slow frame) and are then shown unchanged.
        Needs ``useFBO=True`` (set in :meth:`open`): the frame buffer keeps its contents
        from one flip to the next, which a plain back buffer does not promise.
        """
        from psychopy import visual

        win = self._require_window()
        self._flat_field = flat_field
        win.backgroundImage = None
        self._background_stim = None
        self._crops = {}
        self._dirty = []
        if flat_field is None:
            return                        # the next flip clears to plain grey again
        if not getattr(win, "useFBO", False):
            raise RuntimeError("the flat-field background needs useFBO=True")
        if tuple(flat_field.size) != tuple(int(v) for v in win.size):
            raise ValueError(f"flat-field made for {flat_field.size}, window is "
                             f"{tuple(win.size)}")
        levels = flat_field.background_levels()
        # PsychoPy draws an array's first row at the BOTTOM: flip so row 0 is the top.
        image = np.flipud(levels / 255.0 * 2.0 - 1.0)
        self._background_stim = visual.ImageStim(
            win, image=image, units="pix", pos=(0, 0),
            size=(win.size[0] / self._scale, win.size[1] / self._scale),
            interpolate=False, colorSpace="rgb")
        self._dirty = "all"
        self._restore()
        win.flip(clearBuffer=False)
        # Re-measure with a train-like frame (a patch-sized piece repainted each frame).
        # The drop count is judged against this refresh, so it must be the real one.
        before = self._measured_refresh
        probe = self._background_crop((int(win.size[0]) // 2 - 100,
                                       int(win.size[1]) // 2 - 100, 200, 200))
        self._measured_refresh = self._measure_refresh(probe.draw)
        if before == before and abs(self._measured_refresh / before - 1.0) > 0.02:
            print(f"WARNING: refresh {before:.2f} Hz before the flat-field background, "
                  f"{self._measured_refresh:.2f} Hz with it. The display cannot keep up.",
                  flush=True)

    def _flat_field_on(self) -> bool:
        return getattr(self, "_background_stim", None) is not None

    def _background_crop(self, rect):
        """The background image over one rectangle (column, row, width, height in
        physical pixels from the top-left), as an ImageStim placed exactly there."""
        from psychopy import visual

        win = self._require_window()
        w_scr, h_scr = int(win.size[0]), int(win.size[1])
        c0, r0, w, h = rect
        c1, r1 = min(c0 + w, w_scr), min(r0 + h, h_scr)
        c0, r0 = max(c0, 0), max(r0, 0)
        key = (c0, r0, c1, r1)
        stim = self._crops.get(key)
        if stim is None:
            if len(self._crops) > 200:
                self._crops.clear()
            levels = self._flat_field.background_levels()[r0:r1, c0:c1]
            image = np.flipud(levels / 255.0 * 2.0 - 1.0)
            stim = visual.ImageStim(win, image=image, units="pix",
                                    pos=self._to_units(((c0 + c1) / 2.0, (r0 + r1) / 2.0)),
                                    size=((c1 - c0) / self._scale, (r1 - r0) / self._scale),
                                    interpolate=False, colorSpace="rgb")
            self._crops[key] = stim
        return stim

    def _mark_dirty(self, rect) -> None:
        """Flat-field mode: something was drawn over ``rect`` (or "all")."""
        if not self._flat_field_on() or self._dirty == "all":
            return
        if rect == "all":
            self._dirty = "all"
        elif rect not in self._dirty:
            self._dirty.append(rect)

    def _restore(self) -> None:
        """Flat-field mode: paint the background back where the last frames drew."""
        if not self._flat_field_on():
            return
        if self._dirty == "all":
            self._background_stim.draw()
        else:
            for rect in self._dirty:
                if rect[2] > 0 and rect[3] > 0:
                    self._background_crop(rect).draw()
        self._dirty = []

    def _flip(self):
        """flip(); with the flat-field background the frame buffer is kept (not cleared)."""
        return self._require_window().flip(clearBuffer=not self._flat_field_on())

    @staticmethod
    def _patch_rect(texture: Texture):
        """Where :meth:`_to_stim` puts a patch, in physical pixels (same rounding)."""
        h, w = np.asarray(texture.brightness).shape[:2]
        cx, cy = texture.centre_px
        return (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(w), int(h))

    def _fixation_rect(self, centre_px, size_px):
        margin = 4.0 * self._scale                      # the 2-point line width, and more
        half = size_px / 2.0 + margin
        c0, r0 = int(np.floor(centre_px[0] - half)), int(np.floor(centre_px[1] - half))
        side = int(np.ceil(2.0 * half)) + 2
        return (c0, r0, side, side)

    def _draw_background(self) -> None:
        stim = getattr(self, "_background_stim", None)
        if stim is not None:
            stim.draw()

    def close(self) -> None:
        if self._win is not None:
            self._win.close()
            self._win = None

    @property
    def size_px(self) -> tuple[int, int]:
        win = self._require_window()
        return (int(win.size[0]), int(win.size[1]))

    def _to_units(self, centre_px: tuple[float, float]) -> tuple[float, float]:
        """Physical pixels from the top-left, y down -> PsychoPy pix: centred, y up,
        in logical points."""
        win = self._require_window()
        cx, cy = centre_px
        return ((cx - win.size[0] / 2.0) / self._scale,
                (win.size[1] / 2.0 - cy) / self._scale)

    def _require_window(self):
        if self._win is None:
            raise RuntimeError("the stimulus window is not open")
        return self._win

    # -- refresh ---------------------------------------------------------------

    def _measure_refresh(self, draw=None) -> float:
        """Measure the refresh rate rather than trusting what the OS reports.

        An adaptive-refresh panel advertises a nominal rate it does not hold. The
        development MacBook's ProMotion display reports 120 Hz and measures 118.9, and
        that difference turns every "exact" reversal rate into an inexact one.

        The median frame interval is used, not the mean: a handful of long frames at
        window startup would drag a mean badly and say nothing about the steady state.
        """
        self._require_window()
        intervals: List[float] = []
        last = None
        for _ in range(REFRESH_PROBE_FRAMES):
            if draw is not None:
                draw()
            stamp = self._flip()
            if last is not None:
                intervals.append(stamp - last)
            last = stamp

        usable = [i for i in intervals if i > 0]
        if not usable:
            return float("nan")
        return 1.0 / statistics.median(usable)

    @property
    def refresh_hz(self) -> float:
        return self._measured_refresh

    # -- drawing ---------------------------------------------------------------

    def _to_stim(self, texture: Texture):
        from psychopy import visual

        win = self._require_window()
        # Our arrays run top row first (as the Java Specvis and the generator assume);
        # PsychoPy draws an array's first row at the BOTTOM (OpenGL's convention). Without
        # this flip every patch is shown upside down: a 45 deg tilt appears as 135 deg.
        # texture_pixels: the patch as generated, or corrected for the screen's evenness.
        image = np.flipud(np.asarray(self.texture_pixels(texture), dtype=float)) * 2.0 - 1.0

        # PsychoPy places by centre with the origin at screen centre and y upward;
        # our geometry measures from the top-left with y downward. Converted in
        # _to_units, in one place, so no caller has to remember the flip. The size is
        # in points too, so each texel lands on exactly one physical pixel.
        # Snap the patch's corner to a whole physical pixel. With interpolate=False an
        # edge between pixels makes PsychoPy duplicate or skip about 1 column in 8, which
        # breaks up the stripes and the dithering. The shift is under half a pixel.
        h_px, w_px = image.shape
        cx, cy = texture.centre_px
        snapped = (round(cx - w_px / 2.0) + w_px / 2.0, round(cy - h_px / 2.0) + h_px / 2.0)
        pos = self._to_units(snapped)

        return visual.ImageStim(win, image=image, pos=pos, units="pix",
                                size=(image.shape[1] / self._scale,
                                      image.shape[0] / self._scale),
                                interpolate=False)

    def draw_fixation(self, centre_px: tuple[float, float], size_px: float) -> None:
        """A cross, not a dot. A vertex gives the eye something to hold."""
        from psychopy import visual

        win = self._require_window()
        pos = self._to_units(centre_px)
        half = size_px / 2.0 / self._scale

        if self._fixation is None:
            self._fixation = [
                visual.Line(win, units="pix", lineColor="white", lineWidth=2),
                visual.Line(win, units="pix", lineColor="white", lineWidth=2),
            ]
            self._fixation_state = None
        colour = getattr(self, "fixation_colour", "white")
        # Set the lines only when the + moves or changes colour. Lab PC, 24 Nov: setting
        # them on every frame (each set rebuilds the line) made trains with the + run
        # at ~5 ms a frame instead of 4.17 at 240 Hz. Same pixels either way.
        state = (pos, half, str(colour))
        if getattr(self, "_fixation_state", None) != state:
            for line in self._fixation:
                line.lineColor = colour
            self._fixation[0].start = (pos[0] - half, pos[1])
            self._fixation[0].end = (pos[0] + half, pos[1])
            self._fixation[1].start = (pos[0], pos[1] - half)
            self._fixation[1].end = (pos[0], pos[1] + half)
            self._fixation_state = state
        for line in self._fixation:
            line.draw()
        self._mark_dirty(self._fixation_rect(centre_px, size_px))

    # -- presentation ----------------------------------------------------------

    def _draw_fixation(self, fixation: Optional[Fixation]) -> None:
        if fixation is not None:
            self.draw_fixation(fixation[0], fixation[1])

    def present_train(self, polarities: Sequence[Texture], frames_per_reversal: int,
                      total_frames: int, fixation: Optional[Fixation] = None,
                      response_keys: Sequence[str] = ()) -> PresentationReport:
        from psychopy import core, event

        if not polarities:
            raise ValueError("a train needs at least one polarity")
        if total_frames < 1:
            raise ValueError("a train needs at least one frame")

        win = self._require_window()
        if win.mouseVisible:
            win.mouseVisible = False
        stims = [self._to_stim(t) for t in polarities]
        flat = self._flat_field_on()

        # Upload both textures before the train starts. Building an ImageStim inside
        # the loop would drop frames at the one moment that must not drop any.
        # They are drawn into the back buffer and then cleared, so nothing shows.
        for stim in stims:
            stim.draw()
        if not flat:
            win.clearBuffer()
            self._draw_fixation(fixation)
            win.flip()
        else:
            # Flat-field: paint the background back over the uploads and over whatever
            # the last frames drew (also uploads those pieces), then the +. During the
            # train only the patch is drawn; the + stays in the kept frame buffer.
            self._restore()
            patch_rects = list(dict.fromkeys(self._patch_rect(t) for t in polarities))
            for rect in patch_rects:
                self._background_crop(rect).draw()
            self._draw_fixation(fixation)
            fix_rect = self._fixation_rect(*fixation) if fixation is not None else None
            # Only if the + overlaps the patch must it be redrawn over it each frame
            # (its piece of background first, so its soft edges do not build up).
            overlap = fix_rect is not None and any(
                r[0] < fix_rect[0] + fix_rect[2] and fix_rect[0] < r[0] + r[2] and
                r[1] < fix_rect[1] + fix_rect[3] and fix_rect[1] < r[1] + r[3]
                for r in patch_rects)
            fix_crop = self._background_crop(fix_rect) if overlap else None
            win.flip(clearBuffer=False)

        stamps: List[float] = []
        onset = None
        count = len(stims)
        keys: List[tuple[str, float]] = []
        clock = core.Clock()
        if response_keys:
            event.clearEvents()

        for frame in range(total_frames):
            if frames_per_reversal > 0 and count > 1:
                index = (frame // frames_per_reversal) % count
            else:
                index = 0
            if not flat:
                stims[index].draw()
                self._draw_fixation(fixation)
            else:
                if fix_crop is not None:
                    fix_crop.draw()
                stims[index].draw()
                if fix_crop is not None:
                    self._draw_fixation(fixation)
            if frame == 0:
                win.callOnFlip(clock.reset)
            stamp = win.flip(clearBuffer=not flat)
            if onset is None:
                onset = stamp
            stamps.append(stamp)
            if response_keys:
                keys += [(k, t * 1000.0) for k, t in
                         event.getKeys(keyList=list(response_keys), timeStamped=clock)]
        if flat:
            for rect in patch_rects:
                self._mark_dirty(rect)

        intervals = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
        dropped = count_dropped(intervals, self._measured_refresh)

        phases = (total_frames // frames_per_reversal) if frames_per_reversal > 0 else 0

        return PresentationReport(
            frames_presented=total_frames,
            reversals_presented=phases,
            measured_refresh_hz=self._measured_refresh,
            dropped_frames=dropped,
            frame_intervals_ms=intervals,
            onset_timestamp=onset or 0.0,
            keys=keys,
        )

    # -- input -----------------------------------------------------------------

    def wait_for_key(self, keys: Sequence[str], timeout_ms: Optional[float],
                     fixation: Optional[Fixation] = None, clear_events: bool = True
                     ) -> tuple[Optional[str], Optional[float]]:
        return self.wait_for_key_watching(keys, timeout_ms, fixation, clear_events)

    def wait_for_key_watching(self, keys: Sequence[str], timeout_ms: Optional[float],
                              fixation: Optional[Fixation] = None, clear_events: bool = True,
                              on_frame=None) -> tuple[Optional[str], Optional[float]]:
        from psychopy import core, event

        win = self._require_window()
        if clear_events:
            event.clearEvents()
        clock = core.Clock()
        limit = None if timeout_ms is None else timeout_ms / 1000.0
        flat = self._flat_field_on()
        shown = None                        # flat-field: the + colour now on screen

        while True:
            pressed = event.getKeys(keyList=list(keys), timeStamped=clock)
            if pressed:
                key, when = pressed[0]
                return key, when * 1000.0
            if limit is not None and clock.getTime() >= limit:
                return None, None
            if on_frame is not None and on_frame() is True:
                return None, None           # the caller ended the wait (eye left the +)
            if not flat:
                self._draw_fixation(fixation)
                win.flip()
                continue
            # Flat-field: repaint only when something changed (first frame, or the +
            # colour); otherwise flip the kept frame unchanged.
            colour = getattr(self, "fixation_colour", "white")
            if shown is None or colour != shown:
                self._restore()
                self._draw_fixation(fixation)
                shown = colour
            win.flip(clearBuffer=False)

    def pointer_px(self) -> Optional[tuple[float, float]]:
        """The mouse, in physical pixels from the top-left (PsychoPy gives points from
        the centre, y up)."""
        from psychopy import event

        win = self._require_window()
        if getattr(self, "_mouse", None) is None:
            self._mouse = event.Mouse(win=win, visible=False)
        x, y = self._mouse.getPos()
        return (x * self._scale + win.size[0] / 2.0, win.size[1] / 2.0 - y * self._scale)

    def show_status(self, text: str, marks: Sequence[Mark], ms: float,
                    keys: Sequence[str] = ("space",)) -> Optional[str]:
        """Text at the top and marks (circles) for about ``ms`` ms; the key pressed or None."""
        from psychopy import event, visual

        win = self._require_window()
        w, h = win.size[0] / self._scale, win.size[1] / self._scale
        line_height = h * 0.024
        # Lab PC, 15 Nov: build the text and circles once and only update them
        # on later calls (the positioning screen calls this every 100 ms; building new
        # stimuli each time made it slow to react), and bring the window to the front so
        # SPACE / ESC reach it.
        cache = getattr(self, "_status_cache", None)
        if cache is None or cache["win"] is not win:
            cache = self._status_cache = dict(win=win, circles=[], labels=[], text=visual.TextStim(
                win, text="", units="pix", height=line_height, color="white",
                wrapWidth=w * 0.85, alignText="center", pos=(0, h * 0.5 - line_height),
                anchorVert="top"))
            try:
                win.winHandle.activate()
            except Exception:                           # noqa: BLE001 -- focus is best effort
                pass
        if cache["text"].text != text:
            cache["text"].text = text
        draw = [cache["text"]]
        while len(cache["circles"]) < len(marks):
            cache["circles"].append(visual.Circle(win, radius=10, units="pix", fillColor=None,
                                                  lineWidth=3))
            cache["labels"].append(visual.TextStim(win, text="", units="pix",
                                                   height=line_height * 0.8))
        for mark, circle, label in zip(marks, cache["circles"], cache["labels"]):
            pos = self._to_units(mark.centre_px)
            circle.radius = mark.radius_px / self._scale
            circle.pos = pos
            circle.lineColor = mark.colour
            draw.append(circle)
            if mark.label:
                if label.text != mark.label:
                    label.text = mark.label
                label.pos = pos
                label.color = mark.colour
                draw.append(label)
        refresh = self._measured_refresh if self._measured_refresh == self._measured_refresh \
            else 60.0
        flat = self._flat_field_on()
        for n in range(max(1, int(round(ms / 1000.0 * refresh)))):
            if not flat:
                for d in draw:
                    d.draw()
                win.flip()
            else:
                # Flat-field: the full background once (text and circles anywhere), then
                # the same frame is shown unchanged.
                if n == 0:
                    self._dirty = "all"
                    self._restore()
                    for d in draw:
                        d.draw()
                    self._dirty = "all"
                win.flip(clearBuffer=False)
            pressed = event.getKeys(keyList=list(keys))
            if pressed:
                return pressed[0]
        return None

    def show_message(self, text: str, ms: float, fixation: Optional[Fixation] = None) -> None:
        """A message above the + for ``ms`` ms (counted in frames), no key needed."""
        from psychopy import visual

        win = self._require_window()
        h = win.size[1] / self._scale
        message = visual.TextStim(win, text=text, units="pix", height=h * 0.03,
                                  color="white", pos=(0, h * 0.12), wrapWidth=h)
        refresh = self._measured_refresh if self._measured_refresh == self._measured_refresh \
            else 60.0
        flat = self._flat_field_on()
        for n in range(max(1, int(round(ms / 1000.0 * refresh)))):
            if not flat:
                message.draw()
                self._draw_fixation(fixation)
                win.flip()
                continue
            if n == 0:                      # flat-field: draw once, then show unchanged
                self._dirty = "all"
                self._restore()
                message.draw()
                self._draw_fixation(fixation)
                self._dirty = "all"
            win.flip(clearBuffer=False)

    def show_screen(self, text: str, marks: Sequence[Mark] = (),
                    keys: Sequence[str] = ("space",)) -> str:
        """Text, plus labelled circles placed in screen pixels (top-left origin)."""
        from psychopy import event, visual

        win = self._require_window()
        w, h = win.size[0] / self._scale, win.size[1] / self._scale   # points
        line_height = h * 0.024
        has_marks = bool(marks)
        message = visual.TextStim(
            win, text=text, units="pix", height=line_height * (0.75 if has_marks else 1.0), color="white",
            wrapWidth=w * 0.8, alignText="center",
            pos=(0, h * 0.5 - line_height * 0.6) if has_marks else (0, 0),
            anchorVert="top" if has_marks else "center")
        drawables = [message]
        for mark in marks:
            pos = self._to_units(mark.centre_px)
            radius = mark.radius_px / self._scale
            drawables.append(visual.Circle(win, radius=radius, pos=pos, units="pix",
                                           lineColor=mark.colour, fillColor=None,
                                           lineWidth=3))
            drawables.append(visual.TextStim(win, text=mark.label, units="pix",
                                             height=line_height * 0.85, color=mark.colour,
                                             pos=(pos[0], pos[1] - radius
                                                  - line_height * 1.6),
                                             wrapWidth=w * 0.3))
        if self._flat_field_on():
            self._dirty = "all"
            self._restore()
        for d in drawables:
            d.draw()
        if has_marks:
            self.draw_fixation((win.size[0] / 2.0, win.size[1] / 2.0),
                               line_height * self._scale)
        self._mark_dirty("all")
        self._flip()
        pressed = event.waitKeys(keyList=list(keys))
        return pressed[0]
