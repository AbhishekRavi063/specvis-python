"""One threshold session: four locations, a reversing Gabor, a staircase per location.

This is the loop the Java's procedure screens drive, rebuilt on the ported pieces --
the staircases, the Gabor generator, the reversal plan, the geometry and the logs.
It talks to the display only through :class:`~specvis.stimulus.backend.StimulusBackend`,
so a whole session can be run against a simulated observer with no window open.

A trial is::

    fixation  ->  reversing Gabor (frames)  ->  response window  ->  inter-trial gap

A press counts from the first frame of the Gabor to the end of the response window,
so a press while the stimulus is still up is not lost. Locations are interleaved at
random, never the same one twice running when another is pending, so the participant
cannot predict where the next patch will appear.

**Honest limits.** Timing is software self-report until a photodiode has checked it.
Contrast is nominal unless the luminance scale came from photometer readings; the
uncalibrated default is labelled as such in every file this writes.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.gabor import (GaborParameters, measure_contrast, minimum_renderable_contrast,
                          render_brightness)
from ..core.geometry import Location, PhysicalScreen, quadrant_locations
from ..core.ladder import brightness_vector
from ..core.luminance import LuminanceCalibration
from ..core.reversal import ReversalPlan
from ..data.gabor_log import GaborPresentationLog
from ..data.trial_log import CATCH, MAIN, PRACTICE, TrialLog
from ..stimulus.backend import Mark, StimulusBackend, Texture
from ..threshold.base import LocationThreshold, Result, Termination
from ..threshold.bisection import BisectionLocationThreshold
from ..threshold.staircase42 import (ContrastLocationThreshold, FourTwoStaircase,
                                     contrast_to_db)

FOUR_TWO = "4-2 staircase"
BISECTION = "Specvis bisection"
METHODS = (FOUR_TWO, BISECTION)

SEEN_KEY = "space"
ABORT_KEY = "escape"


class SessionError(Exception):
    """A setting the display cannot honour. Raised before the first trial."""


class Aborted(Exception):
    pass


@dataclass
class SessionSettings:
    participant_id: str = "demo"
    eye: str = "both"
    method: str = FOUR_TWO

    # Screen, measured. Resolution comes from the window itself.
    screen_width_mm: float = 302.0
    viewing_distance_mm: float = 570.0

    # Where and what
    eccentricity_degrees: float = 5.0
    diameter_degrees: float = 2.0
    spatial_frequency_cpd: float = 4.0
    orientation_degrees: float = 90.0
    mean_luminance: float = 50.0          # on the calibration's own scale
    max_contrast: float = 0.80
    min_contrast: float = 0.08

    # Timing
    reversal_hz: float = 6.0              # reversals per second; 0 = static patch
    train_ms: float = 1000.0
    fixation_ms: float = 500.0
    response_window_ms: float = 1200.0    # after the train ends
    iti_ms: Tuple[float, float] = (300.0, 800.0)

    # Session structure
    practice_trials: int = 2
    catch_fraction: float = 0.10
    break_every_trials: int = 40

    def validate(self) -> None:
        if self.method not in METHODS:
            raise SessionError(f"unknown method {self.method!r}; choose one of {METHODS}")
        if not 0 < self.min_contrast < self.max_contrast <= 1:
            raise SessionError("contrast range must satisfy 0 < min < max <= 1")
        if self.train_ms <= 0:
            raise SessionError("the stimulus duration must be positive")
        if not 0 <= self.catch_fraction < 0.5:
            raise SessionError("catch fraction must be in [0, 0.5)")


@dataclass
class LocationOutcome:
    location: Location
    result: Result
    threshold_contrast: float      # NaN unless valid
    threshold_db: float            # NaN unless valid

    @property
    def status(self) -> str:
        if self.result.valid:
            return "valid"
        return {
            Termination.CEILING_NOT_SEEN: "ceiling",
            Termination.FLOOR_ALWAYS_SEEN: "floor",
            Termination.BUDGET_EXHAUSTED: "budget",
            Termination.NOT_FINISHED: "not finished",
        }.get(self.result.termination, self.result.termination.value)


@dataclass
class SessionOutcome:
    settings: SessionSettings
    locations: List[LocationOutcome]
    trials_run: int
    blank_trials: int
    false_alarms: int
    dropped_frames: int
    measured_refresh_hz: float
    plan: ReversalPlan
    screen: PhysicalScreen
    calibration_source: str
    calibrated: bool
    aborted: bool
    started: str
    duration_s: float
    trial_log: TrialLog = field(repr=False, default_factory=TrialLog)
    gabor_log: GaborPresentationLog = field(repr=False, default_factory=GaborPresentationLog)

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarms / self.blank_trials if self.blank_trials else float("nan")


# ------------------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------------------

def _make_threshold(settings: SessionSettings, floor: float) -> LocationThreshold:
    if settings.method == FOUR_TWO:
        start = math.sqrt(settings.min_contrast * settings.max_contrast)
        return ContrastLocationThreshold(
            FourTwoStaircase(start, settings.max_contrast, settings.min_contrast),
            FOUR_TWO, renderable_floor=floor)
    return BisectionLocationThreshold(brightness_vector(10, 100, 9))


def _level_to_contrast(settings: SessionSettings, level: float) -> float:
    """Bisection works on the 0-100 brightness axis; for a Gabor that axis scales the
    maximum contrast (Specvis-Luminance-and-Threshold-Methods.docx, section 3.7)."""
    if settings.method == BISECTION:
        return level / 100.0 * settings.max_contrast
    return level


# ------------------------------------------------------------------------------
# The session
# ------------------------------------------------------------------------------

def run_session(settings: SessionSettings, backend: StimulusBackend,
                calibration: Optional[LuminanceCalibration] = None,
                locations: Optional[Sequence[Location]] = None,
                rng: Optional[random.Random] = None,
                show_instructions: bool = True,
                on_trial: Optional[Callable[[dict], None]] = None) -> SessionOutcome:
    """Run one session on an already-open backend and return what it produced."""
    settings.validate()
    calibration = calibration or LuminanceCalibration.uncalibrated()
    rng = rng or random.Random()
    locations = list(locations or quadrant_locations(settings.eccentricity_degrees))

    # -- geometry ---------------------------------------------------------------
    width_px, height_px = backend.size_px
    if width_px <= 0 or height_px <= 0:
        raise SessionError("the display reported no size; was it opened?")
    screen = PhysicalScreen(width_px, height_px, settings.screen_width_mm,
                            settings.screen_width_mm * height_px / width_px,
                            settings.viewing_distance_mm)
    geometry = screen.flat_geometry()
    for loc in locations:
        if not geometry.fits(loc.degrees_x, loc.degrees_y, settings.diameter_degrees):
            raise SessionError(
                f"location {loc.index} at ({loc.degrees_x:+.1f}, {loc.degrees_y:+.1f}) deg "
                f"does not fit on this screen ({screen.describe()}). Check the screen "
                "width and viewing distance.")
    ppd_x, ppd_y = geometry.px_per_degree_x, geometry.px_per_degree_y
    fixation = (geometry.fixation_px, max(12.0, 0.5 * ppd_x))

    # -- timing -------------------------------------------------------------------
    refresh = backend.refresh_hz
    if not refresh or math.isnan(refresh):
        raise SessionError("the display's refresh rate could not be measured")
    plan = ReversalPlan.for_rate(settings.reversal_hz, refresh)
    total_frames = max(1, round(settings.train_ms / 1000.0 * refresh))
    frames_per_reversal = 0 if plan.static else plan.frames_per_reversal

    # -- stimulus -------------------------------------------------------------------
    sigma = settings.diameter_degrees / 6.0
    base = GaborParameters(settings.diameter_degrees, settings.spatial_frequency_cpd,
                           settings.orientation_degrees, 0.0, sigma,
                           settings.mean_luminance, settings.max_contrast)
    background = float(calibration.brightness_for(settings.mean_luminance)) / 100.0
    floor = minimum_renderable_contrast(calibration, settings.mean_luminance)
    blank_shape = render_brightness(base, calibration, ppd_x, ppd_y).shape
    blank = np.full(blank_shape, background)

    def polarities(loc: Location, contrast: float) -> List[Texture]:
        centre = geometry.to_px(loc.degrees_x, loc.degrees_y)
        p = base.with_contrast(contrast)
        first = Texture(render_brightness(p, calibration, ppd_x, ppd_y), centre)
        if plan.static:
            return [first]
        return [first, Texture(render_brightness(p.reversed_phase(), calibration,
                                                 ppd_x, ppd_y), centre)]

    # -- state ------------------------------------------------------------------------
    thresholds = {loc.index: _make_threshold(settings, floor) for loc in locations}
    trial_log, gabor_log = TrialLog(), GaborPresentationLog()
    counts = dict(trials=0, blank=0, false_alarms=0, dropped=0)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.monotonic()
    window_ms = int(round(settings.train_ms + settings.response_window_ms))
    display_ms = int(round(total_frames * 1000.0 / refresh))
    aborted = False

    def wait(ms: float) -> None:
        key, _ = backend.wait_for_key([ABORT_KEY], ms, fixation=fixation)
        if key == ABORT_KEY:
            raise Aborted

    def one_trial(block: str, loc: Optional[Location], contrast: float) -> Tuple[bool, float, int]:
        """Present once. Returns (seen, reaction_ms, extra_presses)."""
        wait(settings.fixation_ms)
        textures = (polarities(loc, contrast) if loc is not None
                    else [Texture(blank, geometry.fixation_px)])
        report = backend.present_train(textures, frames_per_reversal, total_frames,
                                       fixation=fixation,
                                       response_keys=[SEEN_KEY, ABORT_KEY])
        counts["dropped"] += report.dropped_frames
        presses = [t for k, t in report.keys if k == SEEN_KEY]
        if any(k == ABORT_KEY for k, _ in report.keys):
            raise Aborted
        if not presses:
            key, latency = backend.wait_for_key([SEEN_KEY, ABORT_KEY],
                                                settings.response_window_ms,
                                                fixation=fixation, clear_events=False)
            if key == ABORT_KEY:
                raise Aborted
            if key == SEEN_KEY:
                presses.append(settings.train_ms + latency)
        # A press ends the trial; the random gap that follows keeps the pace from
        # becoming predictable.
        wait(rng.uniform(*settings.iti_ms))
        seen = bool(presses)
        return seen, (presses[0] if seen else float("nan")), max(0, len(presses) - 1)

    def log(block: str, loc: Optional[Location], contrast: float, seen: bool,
            rt: float, extra: int, level: float) -> None:
        x = loc.degrees_x if loc else 0.0
        y = loc.degrees_y if loc else 0.0
        index = loc.index if loc else -1
        brightness_value = int(round(level)) if settings.method == BISECTION else \
            int(round(100.0 * contrast / settings.max_contrast))
        trial_log.record(block, int((time.monotonic() - t0) * 1000), index, x, y,
                         brightness_value, -1, settings.mean_luminance,
                         contrast if loc else float("nan"), seen, rt, extra,
                         display_ms, window_ms)
        if loc is not None:
            p = base.with_contrast(contrast)
            gabor_log.record(index, x, y, brightness_value, contrast,
                             measure_contrast(p, calibration, ppd_x, ppd_y),
                             settings.mean_luminance, settings.spatial_frequency_cpd,
                             settings.orientation_degrees, 0.0, sigma,
                             settings.diameter_degrees, calibration.calibrated)
        if on_trial is not None:
            on_trial(dict(block=block, location=index, contrast=contrast, seen=seen,
                          reaction_ms=rt))

    try:
        if show_instructions:
            key = backend.show_screen(
                "Keep your eyes on the central +.\n\n"
                "A striped patch will flicker near the +.\n"
                f"Press SPACE as soon as you see it.\n"
                "If you see nothing, do nothing.\n\n"
                "Press SPACE to start.   (ESC stops at any time)",
                keys=[SEEN_KEY, ABORT_KEY])
            if key == ABORT_KEY:
                raise Aborted

        # Practice: clearly visible, logged, never touches a threshold.
        for _ in range(settings.practice_trials):
            loc = rng.choice(locations)
            seen, rt, extra = one_trial(PRACTICE, loc, settings.max_contrast)
            log(PRACTICE, loc, settings.max_contrast, seen, rt, extra,
                100.0 if settings.method == BISECTION else settings.max_contrast)

        last_index = None
        while True:
            pending = [loc for loc in locations if not thresholds[loc.index].finished()]
            if not pending:
                break

            if rng.random() < settings.catch_fraction:
                seen, rt, extra = one_trial(CATCH, None, 0.0)
                counts["blank"] += 1
                counts["false_alarms"] += int(seen)
                log(CATCH, None, 0.0, seen, rt, extra, 0.0)
            else:
                choices = [l for l in pending if l.index != last_index] or pending
                loc = rng.choice(choices)
                threshold = thresholds[loc.index]
                level = threshold.next_level()
                contrast = _level_to_contrast(settings, level)
                seen, rt, extra = one_trial(MAIN, loc, contrast)
                threshold.record(seen)
                log(MAIN, loc, contrast, seen, rt, extra, level)
                last_index = loc.index

            counts["trials"] += 1
            if (settings.break_every_trials and counts["trials"] % settings.break_every_trials == 0
                    and any(not t.finished() for t in thresholds.values())):
                trial_log.record_break()
                done = sum(t.finished() for t in thresholds.values())
                key = backend.show_screen(
                    f"Short break.  {done} of {len(locations)} locations finished.\n\n"
                    "Blink and rest your eyes.\n\nPress SPACE to continue.",
                    keys=[SEEN_KEY, ABORT_KEY])
                if key == ABORT_KEY:
                    raise Aborted
    except Aborted:
        aborted = True

    outcomes = []
    for loc in locations:
        result = thresholds[loc.index].result()
        if result.valid:
            c = _level_to_contrast(settings, result.value)
            outcomes.append(LocationOutcome(loc, result, c, contrast_to_db(c)))
        else:
            outcomes.append(LocationOutcome(loc, result, float("nan"), float("nan")))

    return SessionOutcome(
        settings=settings, locations=outcomes, trials_run=counts["trials"],
        blank_trials=counts["blank"], false_alarms=counts["false_alarms"],
        dropped_frames=counts["dropped"], measured_refresh_hz=refresh, plan=plan,
        screen=screen, calibration_source=calibration.source,
        calibrated=calibration.calibrated, aborted=aborted, started=started,
        duration_s=time.monotonic() - t0, trial_log=trial_log, gabor_log=gabor_log)


# ------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------

def _num(value: float, digits: int = 4) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) \
        else f"{value:.{digits}f}"


def summary_rows(outcome: SessionOutcome) -> List[dict]:
    rows = []
    for o in outcome.locations:
        rows.append(dict(
            participant=outcome.settings.participant_id, eye=outcome.settings.eye,
            method=outcome.settings.method, location=o.location.index,
            quadrant=o.location.quadrant, x_deg=o.location.degrees_x,
            y_deg=o.location.degrees_y, status=o.status, valid=o.result.valid,
            threshold_contrast=_num(o.threshold_contrast),
            threshold_db=_num(o.threshold_db, 2), trials=o.result.trials,
            termination=o.result.termination.value, note=o.result.note))
    return rows


def write_outputs(outcome: SessionOutcome, directory: pathlib.Path) -> pathlib.Path:
    """Write the session folder: trials.csv, gabor_presentations.csv, summary.csv,
    session.json. Returns the folder."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outcome.trial_log.write_to(directory)
    outcome.gabor_log.write_to(directory)

    rows = summary_rows(outcome)
    with (directory / "summary.csv").open("w", newline="") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join('"%s"' % str(v).replace('"', "'") if "," in str(v) else str(v)
                             for v in r.values()) + "\n")

    meta = dict(
        started=outcome.started, duration_s=round(outcome.duration_s, 1),
        aborted=outcome.aborted, trials_run=outcome.trials_run,
        blank_catch_trials=outcome.blank_trials, false_alarms=outcome.false_alarms,
        false_alarm_rate=None if math.isnan(outcome.false_alarm_rate)
        else round(outcome.false_alarm_rate, 4),
        measured_refresh_hz=round(outcome.measured_refresh_hz, 3),
        reversal_requested_hz=outcome.plan.requested_hz,
        reversal_actual_hz=round(outcome.plan.actual_reversals_per_second, 4),
        frames_per_reversal=outcome.plan.frames_per_reversal,
        dropped_frames=outcome.dropped_frames,
        timing_note="software self-report; not photodiode-validated",
        screen=outcome.screen.describe(),
        luminance_scale=outcome.calibration_source,
        luminance_calibrated=outcome.calibrated,
        settings=dataclasses.asdict(outcome.settings),
    )
    (directory / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
    return directory


def results_marks(outcome: SessionOutcome, backend_size: Tuple[int, int]) -> List[Mark]:
    """The results map: one circle per location, coloured by status."""
    width_px, height_px = backend_size
    geometry = outcome.screen.flat_geometry()
    colours = {"valid": "lightgreen", "ceiling": "tomato", "floor": "orange"}
    radius = outcome.settings.diameter_degrees / 2.0 * geometry.px_per_degree_x
    marks = []
    for o in outcome.locations:
        if o.result.valid:
            label = f"{o.threshold_contrast:.3f}  ({o.threshold_db:.1f} dB)\n{o.result.trials} trials"
        else:
            label = f"{o.status.upper()}\n{o.result.trials} trials"
        marks.append(Mark(geometry.to_px(o.location.degrees_x, o.location.degrees_y),
                          radius, label, colours.get(o.status, "lightgrey")))
    return marks


def results_text(outcome: SessionOutcome) -> str:
    fa = ("no blank trials" if not outcome.blank_trials else
          f"{outcome.false_alarms}/{outcome.blank_trials} blank trials pressed")
    lines = [
        ("SESSION STOPPED EARLY" if outcome.aborted else "Session complete"),
        f"{outcome.settings.method}  |  {outcome.trials_run} trials  |  "
        f"false alarms: {fa}",
        f"Reversal {outcome.plan.actual_reversals_per_second:.2f}/s at "
        f"{outcome.measured_refresh_hz:.1f} Hz  |  dropped frames: {outcome.dropped_frames}",
        ("" if outcome.calibrated else
         "Contrast is NOMINAL: screen not photometer-calibrated.  ")
        + "Timing is software self-report.",
        "",
        "Press SPACE to finish.",
    ]
    return "\n".join(lines)
