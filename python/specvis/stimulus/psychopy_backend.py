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

#: A frame interval longer than this multiple of the median counts as dropped.
DROP_FACTOR = 1.5

#: Frames used to measure the refresh rate before the first trial.
REFRESH_PROBE_FRAMES = 240


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
        self._measured_refresh = self._measure_refresh()

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

    def _measure_refresh(self) -> float:
        """Measure the refresh rate rather than trusting what the OS reports.

        An adaptive-refresh panel advertises a nominal rate it does not hold. The
        development MacBook's ProMotion display reports 120 Hz and measures 118.9, and
        that difference turns every "exact" reversal rate into an inexact one.

        The median frame interval is used, not the mean: a handful of long frames at
        window startup would drag a mean badly and say nothing about the steady state.
        """
        win = self._require_window()
        intervals: List[float] = []
        last = None
        for _ in range(REFRESH_PROBE_FRAMES):
            stamp = win.flip()
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
        image = np.asarray(texture.brightness, dtype=float) * 2.0 - 1.0

        # PsychoPy places by centre with the origin at screen centre and y upward;
        # our geometry measures from the top-left with y downward. Converted in
        # _to_units, in one place, so no caller has to remember the flip. The size is
        # in points too, so each texel lands on exactly one physical pixel.
        pos = self._to_units(texture.centre_px)

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
        colour = getattr(self, "fixation_colour", "white")
        for line in self._fixation:
            line.lineColor = colour
        self._fixation[0].start = (pos[0] - half, pos[1])
        self._fixation[0].end = (pos[0] + half, pos[1])
        self._fixation[1].start = (pos[0], pos[1] - half)
        self._fixation[1].end = (pos[0], pos[1] + half)
        for line in self._fixation:
            line.draw()

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
        stims = [self._to_stim(t) for t in polarities]

        # Upload both textures before the train starts. Building an ImageStim inside
        # the loop would drop frames at the one moment that must not drop any.
        # They are drawn into the back buffer and then cleared, so nothing shows.
        for stim in stims:
            stim.draw()
        win.clearBuffer()
        self._draw_fixation(fixation)
        win.flip()

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
            stims[index].draw()
            self._draw_fixation(fixation)
            if frame == 0:
                win.callOnFlip(clock.reset)
            stamp = win.flip()
            if onset is None:
                onset = stamp
            stamps.append(stamp)
            if response_keys:
                keys += [(k, t * 1000.0) for k, t in
                         event.getKeys(keyList=list(response_keys), timeStamped=clock)]

        intervals = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
        dropped = 0
        if intervals:
            median = statistics.median(intervals)
            dropped = sum(1 for i in intervals if i > median * DROP_FACTOR)

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
        from psychopy import core, event

        win = self._require_window()
        if clear_events:
            event.clearEvents()
        clock = core.Clock()
        limit = None if timeout_ms is None else timeout_ms / 1000.0

        while True:
            pressed = event.getKeys(keyList=list(keys), timeStamped=clock)
            if pressed:
                key, when = pressed[0]
                return key, when * 1000.0
            if limit is not None and clock.getTime() >= limit:
                return None, None
            self._draw_fixation(fixation)
            win.flip()

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
        for d in drawables:
            d.draw()
        if has_marks:
            self.draw_fixation((win.size[0] / 2.0, win.size[1] / 2.0),
                               line_height * self._scale)
        win.flip()
        pressed = event.waitKeys(keyList=list(keys))
        return pressed[0]
