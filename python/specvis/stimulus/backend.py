"""What the rest of the program needs from a display, and nothing more.

Every module above this line -- the staircases, the Gabor generator, the geometry, the
trial log -- is pure Python and numpy and has no idea what draws anything. This is the
one seam where a presentation framework enters.

The seam exists because the 25-40 Hz reversal requirement is the part of this study
most likely to force a change of toolkit, and discovering that after the UI is written
would be expensive. A backend implements eight methods; swapping one out touches nothing
else.

**Frame counting, not timers.** :meth:`StimulusBackend.present_train` is specified in
*frames*, never milliseconds, and must flip on vsync. A wall-clock timer cannot hold
12.5 ms phases: the jitter is comparable to the interval being measured, and the SSVEP
spectrum smears.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: A fixation mark to keep on screen: (centre_px, size_px).
Fixation = Tuple[Tuple[float, float], float]


@dataclass(frozen=True)
class Mark:
    """A labelled circle on an operator or results screen, e.g. one tested location."""

    centre_px: Tuple[float, float]
    radius_px: float
    label: str
    colour: str = "white"


@dataclass
class PresentationReport:
    """What the display actually did, as opposed to what it was asked to do.

    Every field here is **software self-report**. None of it is photodiode-validated,
    and until it is, none of it may be presented as verified timing.
    """

    frames_presented: int
    reversals_presented: int
    measured_refresh_hz: float
    dropped_frames: int
    frame_intervals_ms: List[float] = field(default_factory=list)
    onset_timestamp: float = 0.0
    #: Keys pressed during the train, as (key, ms from the first frame).
    keys: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def dropped_any(self) -> bool:
        return self.dropped_frames > 0

    def actual_reversals_per_second(self, requested_frames_per_reversal: int) -> float:
        if requested_frames_per_reversal <= 0 or self.measured_refresh_hz <= 0:
            return 0.0
        return self.measured_refresh_hz / requested_frames_per_reversal

    def describe(self) -> str:
        return (
            f"{self.frames_presented} frames, {self.reversals_presented} reversals, "
            f"measured {self.measured_refresh_hz:.2f} Hz, "
            f"{self.dropped_frames} dropped"
            + ("  [TIMING IS SOFTWARE SELF-REPORT, NOT PHOTODIODE-VALIDATED]"
               if self.frames_presented else "")
        )


@dataclass(frozen=True)
class Texture:
    """A patch ready to draw: 0-1 brightness, and where it goes.

    The generator produces these; the backend uploads them. Keeping the pixels as a
    plain array means the same patch can be checked numerically in a test that never
    opens a window.
    """

    brightness: np.ndarray          # 2-D, values in [0, 1]
    centre_px: tuple[float, float]

    def __post_init__(self) -> None:
        if self.brightness.ndim != 2:
            raise ValueError("a texture is a 2-D brightness array")
        if self.brightness.min() < 0.0 or self.brightness.max() > 1.0:
            raise ValueError("brightness must be within [0, 1]")


class StimulusBackend(abc.ABC):
    """A full-screen display that can present patches on vsync."""

    @property
    @abc.abstractmethod
    def refresh_hz(self) -> float:
        """Measured refresh rate.

        Measured, not the rate the operating system claims: an adaptive-refresh panel
        reports a nominal rate it does not hold. The development MacBook's ProMotion
        display measures 118.9 Hz, not the 120 it advertises, and that difference makes
        every "exact" rate inexact.
        """

    @abc.abstractmethod
    def open(self, width_px: int, height_px: int, background_brightness: float) -> None:
        """Open the stimulus window and wait for the display to settle."""

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def size_px(self) -> Tuple[int, int]:
        """The window's real size in pixels, which full screen may set differently
        from the size requested in :meth:`open`."""

    @abc.abstractmethod
    def draw_fixation(self, centre_px: tuple[float, float], size_px: float) -> None:
        """Draw the fixation mark. A cross, not a dot -- it gives the eye a vertex."""

    @abc.abstractmethod
    def present_train(self, polarities: Sequence[Texture], frames_per_reversal: int,
                      total_frames: int, fixation: Optional[Fixation] = None,
                      response_keys: Sequence[str] = ()) -> PresentationReport:
        """Present a reversal train, flipping between polarities on vsync.

        :param polarities: two textures of identical mean luminance. A train of one
                           texture is a static patch.
        :param frames_per_reversal: flip every this many frames. Must be an integer so
                                    the two phases are symmetric.
        :param total_frames: how long the train runs, in frames.
        :param fixation: a fixation mark drawn on every frame of the train.
        :param response_keys: keys collected during the train, timed from its first
                              frame, so a press while the stimulus is up is not lost.

        Implementations must count frames, never consult a clock to decide when to
        flip, and must report dropped frames honestly rather than silently catching up.
        """

    @abc.abstractmethod
    def wait_for_key(self, keys: Sequence[str], timeout_ms: Optional[float],
                     fixation: Optional[Fixation] = None, clear_events: bool = True
                     ) -> tuple[Optional[str], Optional[float]]:
        """Wait for one of ``keys``. Returns ``(key, latency_ms)``, or ``(None, None)``.

        Latency is measured from the last flip, so it is comparable across backends.
        It still includes toolkit and keyboard latency and is not a physiological
        reaction time. ``fixation`` stays on screen while waiting; ``clear_events``
        false keeps presses made just before the call.
        """

    @abc.abstractmethod
    def show_screen(self, text: str, marks: Sequence[Mark] = (),
                    keys: Sequence[str] = ("space",)) -> str:
        """Show text (and optional marks) until one of ``keys`` is pressed; return it."""


class NullBackend(StimulusBackend):
    """A backend that draws nothing, for testing everything above the seam.

    Reports a perfect display: the requested frames, no drops, the nominal refresh. A
    test that passes against this proves the logic, never the timing.
    """

    def __init__(self, refresh_hz: float = 120.0,
                 scripted_keys: Optional[List[tuple[str, float]]] = None):
        self._refresh = refresh_hz
        self._scripted = list(scripted_keys or [])
        self.opened = False
        self.trains: List[tuple[int, int]] = []
        self.screens: List[str] = []
        self._size = (0, 0)

    @property
    def refresh_hz(self) -> float:
        return self._refresh

    def open(self, width_px: int, height_px: int, background_brightness: float) -> None:
        self.opened = True
        self._size = (int(width_px), int(height_px))

    def close(self) -> None:
        self.opened = False

    @property
    def size_px(self) -> Tuple[int, int]:
        return self._size

    def draw_fixation(self, centre_px: tuple[float, float], size_px: float) -> None:
        pass

    def show_screen(self, text: str, marks: Sequence[Mark] = (),
                    keys: Sequence[str] = ("space",)) -> str:
        self.screens.append(text)
        return keys[0]

    def present_train(self, polarities: Sequence[Texture], frames_per_reversal: int,
                      total_frames: int, fixation: Optional[Fixation] = None,
                      response_keys: Sequence[str] = ()) -> PresentationReport:
        if not polarities:
            raise ValueError("a train needs at least one polarity")
        self.trains.append((frames_per_reversal, total_frames))
        reversals = (total_frames // frames_per_reversal) if frames_per_reversal else 0
        interval = 1000.0 / self._refresh
        return PresentationReport(
            frames_presented=total_frames,
            reversals_presented=reversals,
            measured_refresh_hz=self._refresh,
            dropped_frames=0,
            frame_intervals_ms=[interval] * total_frames,
        )

    def wait_for_key(self, keys: Sequence[str], timeout_ms: Optional[float],
                     fixation: Optional[Fixation] = None, clear_events: bool = True
                     ) -> tuple[Optional[str], Optional[float]]:
        if not self._scripted:
            return (None, None)
        key, latency = self._scripted.pop(0)
        return (key, latency) if key in keys else (None, None)
