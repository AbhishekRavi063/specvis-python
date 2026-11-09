"""Monocular perimetry: a threshold at every point of a grid, to map the visual field.

This is Phase 1 of ReFreM: map each eye, so the EEG Gabors can be placed in truly
blind and truly seeing parts of the field. It is the method Pragathi proposed -- a
SITA-style Bayesian search -- built without Zeiss's normative database:

1. **Seed points first.** One point per quadrant is finished before the rest start.
2. **Starting guesses** come from our own table of normal thresholds (built from
   pilot sessions with :mod:`specvis.procedure.prior_table`) and from finished
   neighbours -- **in the same quadrant only**. Stroke field defects have sharp
   borders at the vertical and horizontal midlines; borrowing across them would
   blur exactly the border the map is for.
3. **ZEST** at every point (:mod:`specvis.threshold.zest`), with the quick blind rule.
4. **Pilot mode** switches the starting guesses off, so the first 3-5 participants
   give clean, independent thresholds for the table.

Catch trials, as in ``untitled.py``: blank presentations (false alarms), strong
presentations at points already seen well (misses), and presentations at the
blind spot (fixation losses -- only meaningful with the other eye covered).

**Coordinates.** Points are in visual-field degrees, **x right and y up**, relative to
fixation, as on a clinical printout. They are placed on the flat screen with the
tangent (``d * tan``) model, so a point labelled 21 degrees really is 21 degrees.
The patch itself is drawn at the central pixels-per-degree, as PsychoPy's
``degFlatPos`` units do.

**Honest limits.** Contrast is nominal unless the luminance scale came from a
photometer; timing is software self-report. Both are written into every session.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.gabor import (GaborParameters, measure_contrast, minimum_renderable_contrast,
                          render_brightness)
from ..core.geometry import PhysicalScreen
from ..core.luminance import LuminanceCalibration
from ..stimulus.backend import Mark, StimulusBackend, Texture
from ..threshold.base import Termination
from ..threshold.zest import ZEST, ZestLocationThreshold, ZestSettings
from .prior_table import PriorTable
from .session import Aborted, SessionError

SEEN_KEY = "space"
ABORT_KEY = "escape"

#: Aflah's 72-point grid (untitled.py): 4 deg apart across, 6 deg up/down, no point on
#: a midline.
GRID_X = (1, 5, 9, 13, 17, 21)
GRID_Y = (1, 7, 13)
FULL, CENTRAL = "full", "central"
EXTENTS = (FULL, CENTRAL)

# Point and trial kinds, as written to trials.csv
ZEST_TRIAL, PROBE_TRIAL = "zest", "zest_probe_max"
BLANK, HIGH, BLIND_SPOT, PRACTICE = "blank_catch", "high_catch", "blind_spot", "practice"

# Map categories
SEEN, WEAK, BLIND, NOT_FINISHED = "seen", "weak", "blind", "not_finished"


@dataclass
class PerimetrySettings:
    participant_id: str = "pilot"
    eye: str = "R"                        # the eye tested; the other is covered
    extent: str = FULL                    # FULL = +/-21 x +/-13, CENTRAL = +/-13 x +/-13
    pilot_mode: bool = True               # no starting guesses: clean data for the table

    # Screen, measured
    screen_width_mm: float = 302.0
    viewing_distance_mm: float = 350.0

    # Stimulus: the ReFreM SSVEP Gabor, shown static
    diameter_degrees: float = 2.0
    spatial_frequency_cpd: float = 4.0
    orientation_degrees: float = 45.0
    mean_luminance: float = 50.0          # on the calibration's own scale

    # Timing (untitled.py)
    stimulus_ms: float = 200.0
    fixation_ms: float = 500.0
    response_window_ms: float = 1200.0    # after the stimulus ends
    iti_ms: Tuple[float, float] = (300.0, 800.0)
    practice_trials: int = 3
    break_every_trials: int = 60

    # Catch trials and validity (untitled.py)
    p_blank: float = 0.10
    p_high: float = 0.05
    p_blind_spot: float = 0.05
    high_contrast: float = 0.90
    high_catch_max_threshold: float = 0.30
    blind_spot_contrast: float = 0.80
    blind_spot_temporal_deg: float = 15.5  # Rohrschneider 2004: 15.5 deg temporal,
    blind_spot_below_deg: float = 1.5      # 1.5 deg below fixation
    max_false_alarm_rate: float = 0.15
    max_miss_rate: float = 0.10
    max_fixation_loss_rate: float = 0.20

    # Speed-ups
    seed_points: Tuple[Tuple[int, int], ...] = ((-9, 7), (9, 7), (-9, -7), (9, -7))
    neighbour_radius_deg: float = 8.0
    prior_sd_log: float = 0.5             # deliberately broad
    prior_flat_weight: float = 0.2
    #: Without a normal table, a threshold above this counts as "weak" on the map.
    #: Provisional: replace with the table's normal range once it exists.
    weak_contrast: float = 0.20

    # Fixation and keys (Specvis step 4)
    fixation_colour: str = "white"
    fixation_monitoring: str = "blind spot"   # or "none": no blind-spot catch trials
    answer_key: str = "space"
    pause_key: str = "p"
    cancel_key: str = "escape"
    background_brightness_pct: float = 50.0   # with a luminance scale, sets the mean

    zest: ZestSettings = field(default_factory=ZestSettings)

    def validate(self) -> None:
        if len({self.answer_key, self.pause_key, self.cancel_key}) != 3:
            raise SessionError("the answer, pause and cancel keys must be different")
        if self.eye not in ("L", "R"):
            raise SessionError("perimetry is monocular: eye must be 'L' or 'R'")
        if self.extent not in EXTENTS:
            raise SessionError(f"extent must be one of {EXTENTS}")
        if self.p_blank + self.p_high + self.p_blind_spot >= 0.5:
            raise SessionError("catch trials must be under half of all trials")
        if self.stimulus_ms <= 0 or self.viewing_distance_mm <= 0 or self.screen_width_mm <= 0:
            raise SessionError("stimulus time, distance and screen width must be positive")

    @property
    def temporal_sign(self) -> int:
        """The blind spot is on the temporal side: right for the right eye."""
        return 1 if self.eye == "R" else -1


@dataclass
class PerimetryPoint:
    index: int
    x_deg: float                       # right +
    y_deg: float                       # up +
    zest: ZestLocationThreshold
    started: bool = False
    prior_source: str = ""

    @property
    def quadrant(self) -> Tuple[bool, bool]:
        return (self.x_deg > 0, self.y_deg > 0)

    @property
    def quadrant_name(self) -> str:
        return ("upper" if self.y_deg > 0 else "lower") + "-" + \
            ("right" if self.x_deg > 0 else "left")

    def neighbour_value_log(self) -> Optional[float]:
        """What a finished point tells its neighbours, on the log10 contrast axis."""
        if not self.zest.finished():
            return None
        r = self.zest.result()
        if r.valid:
            return math.log10(r.value)
        if r.termination is Termination.CEILING_NOT_SEEN:
            return float(self.zest.grid[-1])
        if r.termination is Termination.FLOOR_ALWAYS_SEEN:
            return float(self.zest.grid[0])
        return self.zest.estimate_log


def grid_points(settings: PerimetrySettings) -> Tuple[List[Tuple[int, int]],
                                                       List[Tuple[int, int]]]:
    """The test points, and the grid points dropped because they fall in the blind spot.

    Returned as (x, y) in field degrees. The dropped points lie in the physiological
    blind spot (temporal 12-18 deg, 3 below to 1 above), where every eye is blind;
    they are checked by the blind-spot catch trials instead.
    """
    xs = [s * x for x in GRID_X for s in (-1, 1)]
    ys = [s * y for y in GRID_Y for s in (-1, 1)]
    if settings.extent == CENTRAL:
        xs = [x for x in xs if abs(x) <= 13]
    pts = sorted(((x, y) for x in xs for y in ys), key=lambda p: (math.hypot(*p), p))
    kept, dropped = [], []
    for x, y in pts:
        temporal = x * settings.temporal_sign
        (dropped if 12 <= temporal <= 18 and -3 <= y <= 1 else kept).append((x, y))
    return kept, dropped


@dataclass
class PerimetryOutcome:
    settings: PerimetrySettings
    points: List[PerimetryPoint]
    dropped_points: List[Tuple[int, int]]
    trial_rows: List[dict]
    counts: Dict[str, int]
    screen: PhysicalScreen
    measured_refresh_hz: float
    calibration_source: str
    calibrated: bool
    prior_table_source: str
    aborted: bool
    started: str
    duration_s: float
    point_px: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    patch_radius_px: float = 0.0

    @property
    def false_alarm_rate(self) -> float:
        n = self.counts["blank"]
        return self.counts["false_alarms"] / n if n else float("nan")

    @property
    def miss_rate(self) -> float:
        n = self.counts["high"]
        return self.counts["misses"] / n if n else float("nan")

    @property
    def fixation_loss_rate(self) -> float:
        n = self.counts["blind_spot"]
        return self.counts["blind_spot_seen"] / n if n else float("nan")

    @property
    def checks(self) -> Dict[str, bool]:
        """Each reliability check, True when passed. A check with no trials fails."""
        s = self.settings

        def under(rate: float, limit: float) -> bool:
            return not math.isnan(rate) and rate <= limit

        return {
            "false_alarms": under(self.false_alarm_rate, s.max_false_alarm_rate),
            "misses": under(self.miss_rate, s.max_miss_rate),
            # Fixation monitoring switched off: not checked, so not failed. session.json
            # records that it was off.
            "fixation": (s.fixation_monitoring == "none"
                         or under(self.fixation_loss_rate, s.max_fixation_loss_rate)),
            "completed": not self.aborted,
        }

    @property
    def valid(self) -> bool:
        return all(self.checks.values())


def blind_spot_position(settings: PerimetrySettings) -> Tuple[float, float]:
    """Where the blind-spot catch trials go, in field degrees (x right, y up)."""
    return (settings.temporal_sign * settings.blind_spot_temporal_deg,
            -settings.blind_spot_below_deg)


def field_half_degrees(settings: PerimetrySettings, width_px: int, height_px: int
                       ) -> Tuple[float, float]:
    """Half the field the screen covers, horizontally and vertically, in degrees."""
    height_mm = settings.screen_width_mm * height_px / width_px
    d = settings.viewing_distance_mm
    return (math.degrees(math.atan(settings.screen_width_mm / 2.0 / d)),
            math.degrees(math.atan(height_mm / 2.0 / d)))


def fit_problem(settings: PerimetrySettings, width_px: int, height_px: int) -> Optional[str]:
    """Why the grid does not fit on this screen, or None when it does.

    Uses the same tangent placement as the test, so a grid that passes here is drawn
    wholly on screen. Only the physical width, distance and aspect ratio matter.
    """
    if width_px <= 0 or height_px <= 0 or settings.screen_width_mm <= 0 \
            or settings.viewing_distance_mm <= 0:
        return "the screen width, viewing distance and resolution must all be set"
    d = settings.viewing_distance_mm
    half_w_mm = settings.screen_width_mm / 2.0
    half_h_mm = settings.screen_width_mm * height_px / width_px / 2.0
    radius_mm = d * math.radians(settings.diameter_degrees / 2.0)   # central scale
    kept, _ = grid_points(settings)
    hx, hy = field_half_degrees(settings, width_px, height_px)
    for x, y in kept + [blind_spot_position(settings)]:
        mx = abs(d * math.tan(math.radians(x)))
        my = abs(d * math.tan(math.radians(y)))
        if mx + radius_mm > half_w_mm or my + radius_mm > half_h_mm:
            return (f"the point ({x:+g}, {y:+g}) deg does not fit on this screen "
                    f"({settings.screen_width_mm:.0f} mm wide at {d:.0f} mm covers "
                    f"+/-{hx:.1f} x +/-{hy:.1f} deg). Check the screen width and viewing "
                    "distance, sit closer, or use the central grid.")
    return None


def settings_to_dict(settings: PerimetrySettings) -> dict:
    """Plain JSON-ready settings, for presets, settings files and the test process."""
    return dataclasses.asdict(settings)


def settings_from_dict(values: dict) -> PerimetrySettings:
    """The inverse of :func:`settings_to_dict`. Unknown keys are ignored and missing
    ones keep their defaults, so settings files from older versions still load."""
    known = {f.name for f in dataclasses.fields(PerimetrySettings)}
    kwargs = {k: v for k, v in values.items() if k in known}
    if isinstance(kwargs.get("zest"), dict):
        zest_known = {f.name for f in dataclasses.fields(ZestSettings)}
        kwargs["zest"] = ZestSettings(**{k: v for k, v in kwargs["zest"].items()
                                         if k in zest_known})
    if "iti_ms" in kwargs:
        kwargs["iti_ms"] = tuple(kwargs["iti_ms"])
    if "seed_points" in kwargs:
        kwargs["seed_points"] = tuple(tuple(p) for p in kwargs["seed_points"])
    return PerimetrySettings(**kwargs)


# ------------------------------------------------------------------------------
# The session
# ------------------------------------------------------------------------------

def run_perimetry(settings: PerimetrySettings, backend: StimulusBackend,
                  calibration: Optional[LuminanceCalibration] = None,
                  prior_table: Optional[PriorTable] = None,
                  rng: Optional[random.Random] = None,
                  show_instructions: bool = True,
                  on_trial: Optional[Callable[[dict], None]] = None) -> PerimetryOutcome:
    """Run one eye on an already-open backend and return what it produced."""
    settings.validate()
    seen_key, abort_key, pause_key = settings.answer_key, settings.cancel_key, settings.pause_key
    calibration = calibration or LuminanceCalibration.uncalibrated()
    if hasattr(backend, "fixation_colour"):
        backend.fixation_colour = settings.fixation_colour
    rng = rng or random.Random()
    use_table = prior_table if (prior_table is not None and not settings.pilot_mode) else None

    # -- geometry: tangent placement, central scale for the patch -----------------
    width_px, height_px = backend.size_px
    if width_px <= 0 or height_px <= 0:
        raise SessionError("the display reported no size; was it opened?")
    screen = PhysicalScreen(width_px, height_px, settings.screen_width_mm,
                            settings.screen_width_mm * height_px / width_px,
                            settings.viewing_distance_mm)
    ppd_x, ppd_y = screen.local_px_per_degree_x(0.0), screen.local_px_per_degree_y(0.0)
    centre = (width_px / 2.0, height_px / 2.0)
    d = settings.viewing_distance_mm

    def to_px(x_deg: float, y_deg: float) -> Tuple[float, float]:
        return (centre[0] + d * math.tan(math.radians(x_deg)) * screen.px_per_mm_x,
                centre[1] - d * math.tan(math.radians(y_deg)) * screen.px_per_mm_y)

    radius_px = settings.diameter_degrees / 2.0 * max(ppd_x, ppd_y)
    kept, dropped = grid_points(settings)
    blind_spot_xy = blind_spot_position(settings)
    problem = fit_problem(settings, width_px, height_px)
    if problem:
        raise SessionError(problem)

    # -- timing ----------------------------------------------------------------------
    refresh = backend.refresh_hz
    if not refresh or math.isnan(refresh):
        raise SessionError("the display's refresh rate could not be measured")
    stim_frames = max(1, round(settings.stimulus_ms / 1000.0 * refresh))
    display_ms = stim_frames * 1000.0 / refresh

    # -- stimulus ------------------------------------------------------------------------
    sigma = settings.diameter_degrees / 6.0
    base = GaborParameters(settings.diameter_degrees, settings.spatial_frequency_cpd,
                           settings.orientation_degrees, 0.0, sigma,
                           settings.mean_luminance, 1.0)
    floor = minimum_renderable_contrast(calibration, settings.mean_luminance)
    background = float(calibration.brightness_for(settings.mean_luminance)) / 100.0
    blank = np.full(render_brightness(base, calibration, ppd_x, ppd_y).shape, background)
    fixation = (centre, max(12.0, 0.5 * ppd_x))

    def texture(x: float, y: float, contrast: float) -> Texture:
        return Texture(render_brightness(base.with_contrast(contrast), calibration,
                                         ppd_x, ppd_y), to_px(x, y))

    # -- state -------------------------------------------------------------------------
    points = [PerimetryPoint(i, float(x), float(y),
                             ZestLocationThreshold(settings.zest, renderable_floor=floor))
              for i, (x, y) in enumerate(kept)]
    seeds = [p for p in points if (p.x_deg, p.y_deg) in settings.seed_points]
    counts = dict(trials=0, blank=0, false_alarms=0, high=0, misses=0, blind_spot=0,
                  blind_spot_seen=0, dropped_frames=0)
    rows: List[dict] = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.monotonic()
    aborted = False

    def wait(ms: float) -> None:
        key, _ = backend.wait_for_key([abort_key, pause_key], ms, fixation=fixation)
        if key == abort_key:
            raise Aborted
        if key == pause_key:
            key = backend.show_screen(
                f"PAUSED.\n\nPress {seen_key.upper()} to continue   ({abort_key.upper()} stops).",
                keys=[seen_key, abort_key])
            if key == abort_key:
                raise Aborted

    def present(tex: Texture) -> Tuple[bool, float]:
        """One presentation. Returns (seen, reaction ms from stimulus onset)."""
        wait(settings.fixation_ms)
        report = backend.present_train([tex], 0, stim_frames, fixation=fixation,
                                       response_keys=[seen_key, abort_key])
        counts["dropped_frames"] += report.dropped_frames
        if any(k == abort_key for k, _ in report.keys):
            raise Aborted
        presses = [t for k, t in report.keys if k == seen_key]
        if not presses:
            key, latency = backend.wait_for_key([seen_key, abort_key],
                                                settings.response_window_ms,
                                                fixation=fixation, clear_events=False)
            if key == abort_key:
                raise Aborted
            if key == seen_key:
                presses.append(display_ms + latency)
        wait(rng.uniform(*settings.iti_ms))
        return bool(presses), (presses[0] if presses else float("nan"))

    def log(kind: str, point: Optional[PerimetryPoint], x: Optional[float],
            y: Optional[float], contrast: float, seen: bool, rt: float, note: str = "") -> None:
        rendered = (measure_contrast(base.with_contrast(contrast), calibration, ppd_x, ppd_y)
                    if contrast > 0 else 0.0)
        row = dict(trial=len(rows) + 1, ms_from_start=int((time.monotonic() - t0) * 1000),
                   kind=kind, point=point.index if point else "",
                   x_deg="" if x is None else x, y_deg="" if y is None else y,
                   requested_contrast=round(contrast, 5), rendered_contrast=round(rendered, 5),
                   seen=int(seen), reaction_ms="" if math.isnan(rt) else round(rt, 1),
                   estimate_contrast=(round(10 ** point.zest.estimate_log, 5) if point else ""),
                   posterior_sd_log10=(round(point.zest.posterior_sd, 4) if point else ""),
                   note=note)
        rows.append(row)
        if on_trial is not None:
            on_trial(row)

    def start_point(p: PerimetryPoint) -> None:
        """Give a point its starting guess, at its first presentation."""
        p.started = True
        if settings.pilot_mode:
            p.prior_source = "none (pilot mode)"
            return
        guesses, sources = [], []
        if use_table is not None:
            entry = use_table.lookup(settings.eye, p.x_deg, p.y_deg)
            if entry is not None:
                guesses.append(entry)
                sources.append("table")
        near = [q.neighbour_value_log() for q in points
                if q is not p and q.quadrant == p.quadrant
                and math.hypot(q.x_deg - p.x_deg, q.y_deg - p.y_deg) <= settings.neighbour_radius_deg]
        near = [v for v in near if v is not None]
        if near:
            guesses.append(float(np.median(near)))
            sources.append(f"{len(near)} neighbours")
        if guesses:
            p.zest.set_prior(float(np.mean(guesses)), settings.prior_sd_log,
                             settings.prior_flat_weight)
            p.prior_source = " + ".join(sources)
        else:
            p.prior_source = "none"

    try:
        if show_instructions:
            key = backend.show_screen(
                f"Cover your {'LEFT' if settings.eye == 'R' else 'RIGHT'} eye.\n\n"
                "Keep looking at the central +  all the time.\n"
                "Striped patches will flash in different places.\n\n"
                f"Press {seen_key.upper()} as soon as you see one, even if it is faint.\n"
                "If you see nothing, do nothing.\n"
                "Blink just after you press, not while waiting.\n\n"
                f"Press {seen_key.upper()} to start.   ({pause_key.upper()} pauses, "
                f"{abort_key.upper()} stops at any time)",
                keys=[seen_key, abort_key])
            if key == abort_key:
                raise Aborted

        for _ in range(settings.practice_trials):
            p = rng.choice(seeds or points)
            seen, rt = present(texture(p.x_deg, p.y_deg, settings.high_contrast))
            log(PRACTICE, None, p.x_deg, p.y_deg, settings.high_contrast, seen, rt)

        last = None
        while True:
            pending = [p for p in points if not p.zest.finished()]
            if not pending:
                break
            pending_seeds = [p for p in seeds if not p.zest.finished()]
            if pending_seeds:
                pending = pending_seeds

            r = rng.random()
            high_candidates = [p for p in points if p.zest.finished() and p.zest.result().valid
                               and p.zest.result().value <= settings.high_catch_max_threshold]
            if r < settings.p_blank:
                seen, rt = present(Texture(blank, centre))
                counts["blank"] += 1
                counts["false_alarms"] += int(seen)
                log(BLANK, None, None, None, 0.0, seen, rt, "FALSE_ALARM" if seen else "")
            elif r < settings.p_blank + settings.p_high and high_candidates:
                p = rng.choice(high_candidates)
                seen, rt = present(texture(p.x_deg, p.y_deg, settings.high_contrast))
                counts["high"] += 1
                counts["misses"] += int(not seen)
                log(HIGH, None, p.x_deg, p.y_deg, settings.high_contrast, seen, rt,
                    "" if seen else "MISS")
            elif (settings.fixation_monitoring != "none"
                  and r < settings.p_blank + settings.p_high + settings.p_blind_spot):
                bx, by = blind_spot_xy
                seen, rt = present(texture(bx, by, settings.blind_spot_contrast))
                counts["blind_spot"] += 1
                counts["blind_spot_seen"] += int(seen)
                log(BLIND_SPOT, None, bx, by, settings.blind_spot_contrast, seen, rt,
                    "FIXATION_LOSS" if seen else "")
            else:
                choices = [p for p in pending if p is not last] or pending
                p = rng.choice(choices)
                if not p.started:
                    start_point(p)
                kind = PROBE_TRIAL if p.zest.probing else ZEST_TRIAL
                contrast = p.zest.next_level()
                seen, rt = present(texture(p.x_deg, p.y_deg, contrast))
                p.zest.record(seen)
                note = p.zest.result().termination.value if p.zest.finished() else ""
                log(kind, p, p.x_deg, p.y_deg, contrast, seen, rt, note)
                last = p

            counts["trials"] += 1
            if (settings.break_every_trials and counts["trials"] % settings.break_every_trials == 0
                    and any(not p.zest.finished() for p in points)):
                done = sum(p.zest.finished() for p in points)
                key = backend.show_screen(
                    f"Short break.  {done} of {len(points)} points finished.\n\n"
                    "Rest your eye and blink.\n"
                    "Keep the other eye covered.\n\n"
                    f"Press {seen_key.upper()} to continue.",
                    keys=[seen_key, abort_key])
                if key == abort_key:
                    raise Aborted
    except Aborted:
        aborted = True

    return PerimetryOutcome(
        settings=settings, points=points, dropped_points=dropped, trial_rows=rows,
        counts=counts, screen=screen, measured_refresh_hz=refresh,
        calibration_source=calibration.source, calibrated=calibration.calibrated,
        prior_table_source=(use_table.source if use_table is not None else
                            ("not used (pilot mode)" if settings.pilot_mode else "none")),
        aborted=aborted, started=started, duration_s=time.monotonic() - t0,
        point_px={p.index: to_px(p.x_deg, p.y_deg) for p in points},
        patch_radius_px=radius_px)


# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------

def categorise(outcome: PerimetryOutcome, p: PerimetryPoint,
               table: Optional[PriorTable] = None) -> str:
    """seen / weak / blind / not_finished, for the map.

    With a normal table, 'weak' means worse than the table's mean + 2 SD at that point.
    Without one it is the provisional fixed cut-off ``weak_contrast``.
    """
    if not p.zest.finished():
        return NOT_FINISHED
    r = p.zest.result()
    if r.termination is Termination.CEILING_NOT_SEEN:
        return BLIND
    if r.termination is Termination.FLOOR_ALWAYS_SEEN:
        return SEEN
    value_log = math.log10(r.value) if r.valid else p.zest.estimate_log
    if table is not None:
        entry = table.normal_range(outcome.settings.eye, p.x_deg, p.y_deg)
        if entry is not None:
            mean, sd = entry
            return WEAK if value_log > mean + 2.0 * sd else SEEN
    return WEAK if 10 ** value_log > outcome.settings.weak_contrast else SEEN


def summary_rows(outcome: PerimetryOutcome, table: Optional[PriorTable] = None) -> List[dict]:
    s = outcome.settings
    common = dict(participant=s.participant_id, eye=s.eye, pilot_mode=s.pilot_mode,
                  extent=s.extent, session_valid=outcome.valid, method=ZEST)
    rows = []
    for p in outcome.points:
        r = p.zest.result()
        rows.append(dict(
            common, point=p.index, x_deg=p.x_deg, y_deg=p.y_deg,
            quadrant=p.quadrant_name,
            eccentricity_deg=round(math.hypot(p.x_deg, p.y_deg), 2),
            category=categorise(outcome, p, table),
            valid=r.valid,
            threshold_contrast=round(r.value, 5) if r.valid else "",
            threshold_log10=round(math.log10(r.value), 4) if r.valid else "",
            threshold_db=round(-20 * math.log10(r.value), 2) if r.valid else "",
            estimate_contrast=round(10 ** p.zest.estimate_log, 5) if p.zest.trials else "",
            posterior_sd_log10=round(p.zest.posterior_sd, 4) if p.zest.trials else "",
            trials=r.trials, termination=r.termination.value, note=r.note,
            starting_guess=p.prior_source,
            starting_guess_contrast=(round(10 ** p.zest.prior_centre_log, 5)
                                     if p.zest.prior_centre_log is not None else "")))
    return rows


#: Why a check failed, and what to do about it -- shown when a session is not valid.
FAILURE_REASONS = {
    "false_alarms": "Too many presses when nothing was shown (guessing). "
                    "Press only when you really see stripes.",
    "misses": "Strong, easy patches were missed (tired or not attending). "
              "Take a break before repeating.",
    "fixation": "Blind-spot patches were seen. Check the other eye is FULLY covered "
                "(opaque patch), the right eye was chosen, and keep looking at the +.",
    "completed": "The test was stopped before the end.",
}

#: Keys offered on the results screen by the runner.
OTHER_EYE_KEY, REPEAT_KEY, FINISH_KEY = "o", "r", "space"


def results_text(outcome: PerimetryOutcome, offer_keys: bool = False) -> str:
    c, s = outcome.checks, outcome.settings

    def rate(x: float) -> str:
        return "no trials" if math.isnan(x) else f"{100 * x:.0f}%"

    tick = {True: "ok", False: "FAILED"}
    lines = [
        ("STOPPED EARLY" if outcome.aborted else "Finished") +
        f"  |  {s.participant_id}  {s.eye} eye  |  {outcome.counts['trials']} trials, "
        f"{outcome.duration_s / 60:.1f} min",
        f"False presses {rate(outcome.false_alarm_rate)} [{tick[c['false_alarms']]}]   "
        f"Misses {rate(outcome.miss_rate)} [{tick[c['misses']]}]   "
        f"Blind-spot seen {rate(outcome.fixation_loss_rate)} [{tick[c['fixation']]}]",
    ]
    if outcome.valid:
        lines.append("SESSION VALID")
    else:
        lines.append("SESSION NOT VALID - repeat this eye")
        lines += [FAILURE_REASONS[k] for k, passed in c.items() if not passed]
    lines.append("Green = seen   yellow = weak   red = blind   (number = faintest contrast seen)")
    other = "LEFT" if s.eye == "R" else "RIGHT"
    lines.append(f"SPACE = finish     O = test the {other} eye now     R = repeat this eye"
                 if offer_keys else "Press SPACE to finish.")
    return "\n".join(lines)


_MAP_COLOURS = {SEEN: "#1D9E75", WEAK: "#EF9F27", BLIND: "#D85A30", NOT_FINISHED: "#888780"}


def _layout(outcome: PerimetryOutcome) -> Tuple[List[float], List[float]]:
    """Columns (x, left to right) and rows (y, top to bottom) of the grid.

    Maps are drawn as an even grid of these, not at true degrees: the rows at +1 and
    -1 deg are only 2 deg apart and would overlap. Labels give the true degrees.
    """
    xs = sorted({p.x_deg for p in outcome.points} | {float(x) for x, _ in outcome.dropped_points})
    ys = sorted({p.y_deg for p in outcome.points} | {float(y) for _, y in outcome.dropped_points},
                reverse=True)
    return xs, ys


def _label(outcome: PerimetryOutcome, p: PerimetryPoint, cat: str) -> str:
    r = p.zest.result()
    if r.valid:
        return f"{100 * r.value:.0f}%" if r.value >= 0.095 else f"{100 * r.value:.1f}%"
    return {BLIND: "blind", NOT_FINISHED: "-"}.get(cat, "faint")


def results_marks(outcome: PerimetryOutcome, table: Optional[PriorTable] = None) -> List[Mark]:
    """The map on the stimulus screen: one circle per point, on an even grid below the
    text, left/right and up/down as in the field."""
    xs, ys = _layout(outcome)
    w, h = outcome.screen.resolution_x, outcome.screen.resolution_y
    # Symmetric about the screen centre, so the fixation cross the backend draws there
    # sits exactly between the middle columns and rows -- where fixation really was.
    left, right, top, bottom = 0.06 * w, 0.94 * w, 0.22 * h, 0.78 * h
    step_x = (right - left) / max(1, len(xs) - 1)
    step_y = (bottom - top) / max(1, len(ys) - 1)
    radius = 0.22 * min(step_x, step_y)
    marks = []
    for p in outcome.points:
        cat = categorise(outcome, p, table)
        centre = (left + xs.index(p.x_deg) * step_x, top + ys.index(p.y_deg) * step_y)
        marks.append(Mark(centre, radius, _label(outcome, p, cat), _MAP_COLOURS[cat]))
    for x, y in outcome.dropped_points:
        centre = (left + xs.index(float(x)) * step_x, top + ys.index(float(y)) * step_y)
        marks.append(Mark(centre, radius, "blind spot", "lightgrey"))
    return marks


def draw_map(outcome: PerimetryOutcome, path: pathlib.Path,
             table: Optional[PriorTable] = None) -> pathlib.Path:
    """The map as a picture: field degrees, x right, y up, as on a clinical printout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    s = outcome.settings
    xs, ys = _layout(outcome)
    col = {x: i for i, x in enumerate(xs)}
    row = {y: len(ys) - 1 - j for j, y in enumerate(ys)}       # top row highest
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(xs), 1.6 + 0.85 * len(ys)), dpi=120)
    for p in outcome.points:
        cat = categorise(outcome, p, table)
        cx, cy = col[p.x_deg], row[p.y_deg]
        ax.add_patch(Rectangle((cx - 0.45, cy - 0.45), 0.9, 0.9,
                               facecolor=_MAP_COLOURS[cat], edgecolor="white"))
        ax.text(cx, cy + 0.1, _label(outcome, p, cat), ha="center", va="center",
                fontsize=8, color="white")
        ax.text(cx, cy - 0.25, f"n={p.zest.trials}", ha="center", va="center",
                fontsize=6, color="white")
    for x, y in outcome.dropped_points:
        cx, cy = col[float(x)], row[float(y)]
        ax.add_patch(Rectangle((cx - 0.45, cy - 0.45), 0.9, 0.9, facecolor="#D3D1C7",
                               edgecolor="white"))
        ax.text(cx, cy, "blind\nspot", ha="center", va="center", fontsize=6, color="#444441")
    mid_x = (len(xs) - 1) / 2.0        # the grid has no zero column or row
    mid_y = (len(ys) - 1) / 2.0
    ax.axvline(mid_x, color="#444441", lw=0.8)
    ax.axhline(mid_y, color="#444441", lw=0.8)
    ax.plot(mid_x, mid_y, marker="+", color="black", ms=12, mew=2)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_ylim(-0.6, len(ys) - 0.4)
    ax.set_aspect("equal")
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in reversed(ys)])
    ax.tick_params(length=0, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("degrees from fixation (left  -  right)")
    ax.set_ylabel("degrees (down  -  up)")
    status = "VALID" if outcome.valid else "NOT VALID"
    ax.set_title(f"{s.participant_id}  -  {s.eye} eye  -  {status}"
                 f"{'  -  pilot mode' if s.pilot_mode else ''}"
                 f"{'' if outcome.calibrated else '  -  contrast nominal (uncalibrated)'}",
                 fontsize=10)
    ax.legend(handles=[Patch(color=_MAP_COLOURS[k], label=k.replace("_", " "))
                       for k in (SEEN, WEAK, BLIND, NOT_FINISHED)],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_outputs(outcome: PerimetryOutcome, directory: pathlib.Path,
                  table: Optional[PriorTable] = None) -> pathlib.Path:
    """trials.csv, summary.csv, session.json and map.png. Returns the folder."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def write_csv(name: str, rows: Sequence[dict]) -> None:
        if not rows:
            return
        with (directory / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("trials.csv", outcome.trial_rows)
    write_csv("summary.csv", summary_rows(outcome, table))
    s = outcome.settings
    meta = dict(
        kind="perimetry", started=outcome.started, duration_s=round(outcome.duration_s, 1),
        aborted=outcome.aborted, session_valid=outcome.valid, checks=outcome.checks,
        false_alarm_rate=outcome.false_alarm_rate, miss_rate=outcome.miss_rate,
        fixation_loss_rate=outcome.fixation_loss_rate, counts=outcome.counts,
        measured_refresh_hz=outcome.measured_refresh_hz,
        screen=dict(resolution=[outcome.screen.resolution_x, outcome.screen.resolution_y],
                    width_mm=outcome.screen.width_mm, height_mm=outcome.screen.height_mm,
                    viewing_distance_mm=outcome.screen.viewing_distance_mm,
                    field_degrees=[outcome.screen.field_degrees_x,
                                   outcome.screen.field_degrees_y]),
        calibration=dict(source=outcome.calibration_source, calibrated=outcome.calibrated),
        prior_table=outcome.prior_table_source,
        blind_spot_grid_points_dropped=outcome.dropped_points,
        honest_limits=[
            "Timing is software self-report, not photodiode-validated.",
            "Contrast is nominal unless the luminance scale was photometer-measured.",
        ],
        settings=dataclasses.asdict(s),
    )
    (directory / "session.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    draw_map(outcome, directory / "map.png", table)
    return directory
