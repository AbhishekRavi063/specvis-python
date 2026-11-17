"""Block 2, quick test (17 Nov 2026): which way do the stripes go, at which flicker rate?

A pattern-reversal Gabor flickers for a few seconds at one place in the field; then the
person says whether its stripes were horizontal (LEFT arrow) or vertical (RIGHT arrow).
Every combination is shown once, in a random order:

* places: the 4 quadrants at (+/-5, +/-5) deg (the proposal's EEG sites) and the
  natural blind spot of the tested eye (15.5 deg temporal, 1.5 deg below; Rohrschneider
  2004), which is blind only with the other eye covered -- the stand-in for a blind area
  in a healthy person (meeting with Elliott, 30 Oct);
* orientations: horizontal and vertical (a third one later);
* reversal rates: 20, 21, 25, 27, 33, 38, 50, 71 and 100 Hz (asked for, 17 Nov).

Before the patches, Block 1's own setup runs (``setup``): with the eye tracker, head
position, calibration of the tested eye and the accuracy check; then the blind-spot
search, so the blind-spot patch goes where this person does not see, not at the average.

A forced choice: in the blind spot the answers should be right about half the time
(chance); in the seeing quadrants nearly always. That is the check of the hypothesis.

**Rates are whole frames.** A patch can only change on a screen refresh, so the rate
shown is refresh / k for a whole number k of frames per reversal (see
:mod:`specvis.core.reversal`). Most asked-for rates are not exact on any common
screen; the nearest exact one is shown, and both are written to every row and shown
to the operator before the start, so a rate is never reported as one it was not.

The patch is the study's Gabor (2 deg, 4 c/deg) at the highest contrast the screen
gives at the background (the proposal's 5 s of reversal, then 400 ms). With an eye
tracker, gaze is recorded during every patch; a patch with the eye off the + is shown
again once at the end, and the row says so.
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
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.gabor import GaborParameters, render_brightness
from ..core.geometry import PhysicalScreen
from ..core.luminance import LuminanceCalibration
from ..core.reversal import ReversalPlan
from ..stimulus.backend import StimulusBackend, Texture
from .fixation import FIX_NONE, FixationMonitor, judge_patch, monitor_for
from .perimetry import (FULL_THRESHOLD, H24_2_9, PerimetrySettings, background_for,
                        default_calibration, run_perimetry)
from .session import Aborted, SessionError

HORIZONTAL, VERTICAL = "horizontal", "vertical"
#: Gabor orientation in degrees (specvis.core.gabor): 0 = stripes across (horizontal).
ORIENTATION_DEG = {HORIZONTAL: 0.0, VERTICAL: 90.0}
#: Block 1's eye-tracker settings, as in preset 4 (our suggestions, to tune), for the setup.
PRESET4_TRACKER = dict(eye_distance_tolerance_mm=20.0, fixation_tolerance_deg=3.0,
                       fixation_hold_ms=150.0, fixation_wait_ms=8000.0,
                       fixation_dropout_ms=150.0, fixation_off_grace_ms=150.0,
                       fixation_break_max_out_frac=0.25, blink_min_ms=100.0,
                       post_blink_delay_ms=300.0, tracker_monocular_calibration=True)
#: The answer keys: LEFT = horizontal, RIGHT = vertical (Elliott: left/right for
#: orientation; a joystick later).
ANSWER_KEYS = {"left": HORIZONTAL, "right": VERTICAL}
BLIND_SPOT = "blind spot"
QUADRANTS = {"upper right": (5.0, 5.0), "upper left": (-5.0, 5.0),
             "lower left": (-5.0, -5.0), "lower right": (5.0, -5.0)}


@dataclass
class Block2Settings:
    participant_id: str = "pilot"
    eye: str = "R"                        # the tested eye; the other one is covered
    # The lab screen and seating (preset 4).
    screen_width_mm: float = 542.3
    screen_height_mm: float = 305.0
    viewing_distance_mm: float = 570.0
    background_cd_m2: float = 10.0
    # The patch (the proposal: 2 deg, 4 c/deg, pattern reversal).
    diameter_degrees: float = 2.0
    spatial_frequency_cpd: float = 4.0
    #: 0 = the highest contrast the screen gives at the background ("full contrast").
    contrast: float = 0.0
    quadrants: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(QUADRANTS))
    blind_spot: bool = True
    blind_spot_temporal_deg: float = 15.5
    blind_spot_below_deg: float = 1.5
    #: Block 1's setup first (tracker calibration + accuracy, blind-spot search).
    setup: bool = True
    #: Where the blind-spot patch goes; None = the average (set by the setup's search).
    blind_spot_xy: Optional[Tuple[float, float]] = None
    orientations: Tuple[str, ...] = (HORIZONTAL, VERTICAL)
    frequencies_hz: Tuple[float, ...] = (20.0, 21.0, 25.0, 27.0, 33.0, 38.0, 50.0, 71.0, 100.0)
    train_ms: float = 5000.0              # the proposal: 5 s of reversal ...
    gap_ms: float = 400.0                 # ... then 400 ms
    break_every: int = 30
    #: After SPACE (start, breaks) only the + shows for this long (lab report 16 Nov).
    resume_delay_ms: float = 1200.0
    # Eye tracker (our suggestion, to tune): a patch counts when at least half its
    # samples are valid and at most 10 % of them are more than 2 deg off the + -- the
    # blind spot is only ~5.5 deg wide, so the eye must stay closer than in Block 1.
    fixation_monitoring: str = FIX_NONE
    fixation_tolerance_deg: float = 2.0
    fixation_max_out_frac: float = 0.10
    repeat_broken: bool = True
    cancel_key: str = "escape"

    def validate(self) -> None:
        if self.eye not in ("L", "R"):
            raise SessionError("Block 2 tests one eye (L or R): the blind spot is only "
                               "blind with the other eye covered")
        if not self.frequencies_hz or any(f <= 0 for f in self.frequencies_hz):
            raise SessionError("every reversal rate must be above 0 Hz")
        if any(o not in ORIENTATION_DEG for o in self.orientations):
            raise SessionError(f"orientations must be among {list(ORIENTATION_DEG)}")
        if self.train_ms <= 0 or self.viewing_distance_mm <= 0 or self.screen_width_mm <= 0:
            raise SessionError("train length, distance and screen width must be above 0")

    @property
    def temporal_sign(self) -> int:
        return 1 if self.eye == "R" else -1

    def places(self) -> Dict[str, Tuple[float, float]]:
        out = dict(self.quadrants)
        if self.blind_spot:
            out[BLIND_SPOT] = (tuple(self.blind_spot_xy) if self.blind_spot_xy else
                               (self.temporal_sign * self.blind_spot_temporal_deg,
                                -self.blind_spot_below_deg))
        return out

    def perimetry_settings(self) -> PerimetrySettings:
        """The same screen as a perimetry settings object, for the shared helpers."""
        return PerimetrySettings(participant_id=self.participant_id, eye=self.eye,
                                 screen_width_mm=self.screen_width_mm,
                                 screen_height_mm=self.screen_height_mm,
                                 viewing_distance_mm=self.viewing_distance_mm,
                                 background_cd_m2=self.background_cd_m2,
                                 diameter_degrees=self.diameter_degrees,
                                 spatial_frequency_cpd=self.spatial_frequency_cpd,
                                 blind_spot_temporal_deg=self.blind_spot_temporal_deg,
                                 blind_spot_below_deg=self.blind_spot_below_deg)

    def setup_settings(self) -> PerimetrySettings:
        """Block 1 as preset 4, setup only: positioning, calibration, accuracy check,
        blind-spot search."""
        return dataclasses.replace(
            self.perimetry_settings(), strategy=FULL_THRESHOLD, extent=H24_2_9,
            pilot_mode=True, dithering=False, fixation_monitoring=self.fixation_monitoring,
            setup_only=True, blind_spot_search=self.blind_spot,
            **(PRESET4_TRACKER if self.fixation_monitoring == "eye tracker" else {}))


@dataclass(frozen=True)
class Block2Trial:
    place: str
    x_deg: float
    y_deg: float
    orientation: str
    requested_hz: float


def plan_trials(settings: Block2Settings, rng: random.Random) -> List[Block2Trial]:
    """Every place x orientation x rate once, in a random order."""
    trials = [Block2Trial(name, x, y, o, float(f))
              for name, (x, y) in settings.places().items()
              for o in settings.orientations for f in settings.frequencies_hz]
    rng.shuffle(trials)
    return trials


def rate_table(frequencies: Sequence[float], refresh_hz: float) -> List[dict]:
    """What each asked-for rate becomes on this screen."""
    rows = []
    for f in frequencies:
        plan = ReversalPlan.for_rate(f, refresh_hz)
        shown = refresh_hz / plan.frames_per_reversal
        rows.append(dict(requested_hz=f, frames_per_reversal=plan.frames_per_reversal,
                         shown_hz=round(shown, 2), off_by_hz=round(shown - f, 2),
                         exact=abs(shown - f) < 0.05))
    return rows


@dataclass
class Block2Outcome:
    settings: Block2Settings
    rows: List[dict]
    rates: List[dict]
    refresh_hz: float
    contrast: float
    background_cd_m2: float
    calibration_source: str
    calibrated: bool
    started: str
    duration_s: float
    aborted: bool
    counts: Dict[str, int]
    fixation_monitor: str
    setup: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> List[dict]:
        """Per place and rate: answers that count, how many right."""
        out = []
        for place in self.settings.places():
            for r in self.rates:
                rows = [x for x in self.rows if x["place"] == place
                        and x["requested_hz"] == r["requested_hz"] and x["counts"]]
                right = sum(1 for x in rows if x["correct"])
                out.append(dict(place=place, requested_hz=r["requested_hz"],
                                shown_hz=r["shown_hz"], answers=len(rows), correct=right,
                                percent_correct=(round(100.0 * right / len(rows))
                                                 if rows else "")))
        return out


def run_block2(settings: Block2Settings, backend: StimulusBackend,
               calibration: Optional[LuminanceCalibration] = None,
               rng: Optional[random.Random] = None, show_instructions: bool = True,
               fixation_monitor: Optional[FixationMonitor] = None) -> Block2Outcome:
    settings.validate()
    rng = rng or random.Random()
    ps = settings.perimetry_settings()
    calibration = calibration or default_calibration(ps)
    abort_key = settings.cancel_key
    monitor = fixation_monitor or monitor_for(settings.fixation_monitoring, settings.eye)

    # -- Block 1's setup: tracker calibration + accuracy check, blind-spot search --------
    setup: Dict[str, object] = {}
    if settings.setup and show_instructions:
        first = run_perimetry(settings.setup_settings(), backend, calibration=calibration,
                              rng=rng, show_instructions=True, fixation_monitor=monitor)
        found = (first.blind_spot_search or {}).get("found_xy")
        setup = dict(gaze_check=first.gaze_check, blind_spot_search=first.blind_spot_search,
                     blind_spot_used=("found by the search" if found
                                      else "the average position"))
        if first.aborted:
            return Block2Outcome(settings=settings, rows=[], rates=[],
                                 refresh_hz=backend.refresh_hz, contrast=0.0,
                                 background_cd_m2=0.0, calibration_source=calibration.source,
                                 calibrated=calibration.calibrated,
                                 started=time.strftime("%Y-%m-%dT%H:%M:%S"), duration_s=0.0,
                                 aborted=True, counts=dict(trials=0, repeated=0,
                                                           fixation_breaks=0,
                                                           dropped_frames=0),
                                 fixation_monitor=settings.fixation_monitoring, setup=setup)
        if found:
            settings = dataclasses.replace(settings, blind_spot_xy=tuple(found))

    # -- geometry: tangent placement, as in Block 1 -----------------------------------
    width_px, height_px = backend.size_px
    if width_px <= 0 or height_px <= 0:
        raise SessionError("the display reported no size; was it opened?")
    height_mm = min(settings.screen_height_mm or 1e9,
                    settings.screen_width_mm * height_px / width_px)
    screen = PhysicalScreen(width_px, height_px, settings.screen_width_mm, height_mm,
                            settings.viewing_distance_mm)
    ppd_x, ppd_y = screen.local_px_per_degree_x(0.0), screen.local_px_per_degree_y(0.0)
    centre = (width_px / 2.0, height_px / 2.0)
    d = settings.viewing_distance_mm
    radius_mm = d * math.radians(settings.diameter_degrees / 2.0)
    for name, (x, y) in settings.places().items():
        if (abs(d * math.tan(math.radians(x))) + radius_mm > settings.screen_width_mm / 2
                or abs(d * math.tan(math.radians(y))) + radius_mm > height_mm / 2):
            raise SessionError(f"the {name} patch at ({x:+g}, {y:+g}) deg does not fit "
                               f"on this screen at {d:.0f} mm")

    def to_px(x: float, y: float) -> Tuple[float, float]:
        return (centre[0] + d * math.tan(math.radians(x)) * screen.px_per_mm_x,
                centre[1] - d * math.tan(math.radians(y)) * screen.px_per_mm_y)

    def to_deg(px: float, py: float) -> Tuple[float, float]:
        return (math.degrees(math.atan((px - centre[0]) / screen.px_per_mm_x / d)),
                math.degrees(math.atan((centre[1] - py) / screen.px_per_mm_y / d)))

    fixation = (centre, max(12.0, 0.5 * ppd_x))

    # -- timing and patch ------------------------------------------------------------------
    refresh = backend.refresh_hz
    if not refresh or math.isnan(refresh):
        raise SessionError("the display's refresh rate could not be measured")
    rates = rate_table(settings.frequencies_hz, refresh)
    frames_for = {r["requested_hz"]: r["frames_per_reversal"] for r in rates}
    total_frames = max(1, round(settings.train_ms / 1000.0 * refresh))
    mean_luminance, _ = background_for(ps, calibration)
    top = min(1.0, calibration.max_achievable_contrast(mean_luminance))
    contrast = min(settings.contrast, top) if settings.contrast > 0 else top
    sigma = settings.diameter_degrees / 6.0
    textures: Dict[Tuple[str, str], List[Texture]] = {}
    for name, (x, y) in settings.places().items():
        for o in settings.orientations:
            p = GaborParameters(settings.diameter_degrees, settings.spatial_frequency_cpd,
                                ORIENTATION_DEG[o], 0.0, sigma, mean_luminance, contrast)
            textures[(name, o)] = [
                Texture(render_brightness(p, calibration, ppd_x, ppd_y), to_px(x, y)),
                Texture(render_brightness(p.reversed_phase(), calibration, ppd_x, ppd_y),
                        to_px(x, y))]

    watching = getattr(monitor, "watches_gaze", False)

    rows: List[dict] = []
    counts = dict(trials=0, repeated=0, fixation_breaks=0, dropped_frames=0)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.perf_counter()
    aborted = False

    def wait(ms: float) -> None:
        key, _ = backend.wait_for_key([abort_key], ms, fixation=fixation)
        if key == abort_key:
            raise Aborted

    def screen_key(text: str, keys=("space",)) -> str:
        key = backend.show_screen(text, keys=list(keys) + [abort_key])
        if key == abort_key:
            raise Aborted
        wait(settings.resume_delay_ms)
        return key

    def one(trial: Block2Trial, repeat: bool) -> dict:
        wait(settings.gap_ms)
        k = frames_for[trial.requested_hz]
        t_on = time.perf_counter()
        report = backend.present_train(textures[(trial.place, trial.orientation)], k,
                                       total_frames, fixation=fixation,
                                       response_keys=[abort_key])
        t_off = time.perf_counter()
        if any(key == abort_key for key, _ in report.keys):
            raise Aborted
        monitor.patch_shown(t_on, t_off)
        counts["trials"] += 1
        counts["dropped_frames"] += report.dropped_frames
        key, rt = backend.wait_for_key(list(ANSWER_KEYS) + [abort_key], None,
                                       fixation=fixation)
        if key == abort_key:
            raise Aborted
        answer = ANSWER_KEYS.get(key or "", "")
        row = dict(trial=len(rows) + 1, ms_from_start=int((time.perf_counter() - t0) * 1000),
                   place=trial.place, x_deg=trial.x_deg, y_deg=trial.y_deg,
                   orientation=trial.orientation, answer=answer,
                   correct=int(answer == trial.orientation),
                   requested_hz=trial.requested_hz, frames_per_reversal=k,
                   shown_hz=round(refresh / k, 3),
                   measured_refresh_hz=round(report.measured_refresh_hz or refresh, 3),
                   train_frames=report.frames_presented,
                   dropped_frames=report.dropped_frames, contrast=round(contrast, 4),
                   reaction_ms="" if rt is None else round(rt), repeat=int(repeat),
                   gaze_max_deg="", gaze_valid_pct="", counts=1, note="")
        if watching:
            ok, worst, frac = judge_patch(monitor.between(t_on, t_off),
                                          settings.fixation_tolerance_deg,
                                          max_out_frac=settings.fixation_max_out_frac)
            row.update(gaze_max_deg="" if math.isnan(worst) else round(worst, 2),
                       gaze_valid_pct=round(100 * frac))
            if not ok:
                counts["fixation_breaks"] += 1
                row.update(counts=0, note="FIXATION_BREAK")
        rows.append(row)
        return row

    monitor_started = False
    try:
        if watching:
            try:
                monitor.begin(backend, to_deg)
            except RuntimeError as e:
                raise SessionError(f"Eye tracker: {e}")
            monitor_started = True
        if show_instructions:
            lines = "\n".join(
                f"  {r['requested_hz']:g} Hz -> {r['shown_hz']:g} Hz"
                + ("" if r["exact"] else f"  ({r['off_by_hz']:+g})") for r in rates)
            screen_key(
                f"For the operator. Screen: {refresh:.2f} Hz. A patch can only change on a "
                f"screen refresh, so these reversal rates are shown:\n\n{lines}\n\n"
                f"Blind-spot patch at ({settings.places().get(BLIND_SPOT, ('-', '-'))[0]}, "
                f"{settings.places().get(BLIND_SPOT, ('-', '-'))[1]}) deg"
                f" ({setup.get('blind_spot_used', 'the average position')}).\n"
                f"Contrast {contrast:.0%}. Cover the {'left' if settings.eye == 'R' else 'right'}"
                f" eye.\n\nSPACE = go on")
            screen_key(
                "Keep looking at the  +  in the middle all the time.\n\n"
                "A striped patch will flicker for a few seconds somewhere near it.\n"
                "When it has gone, say which way its stripes went:\n\n"
                "LEFT arrow  =  horizontal stripes  (like  =  )\n"
                "RIGHT arrow  =  vertical stripes  (like  ||  )\n\n"
                "If you did not see it, guess.\n\nPress SPACE to start.")
        trials = plan_trials(settings, rng)
        broken: List[Block2Trial] = []
        for i, trial in enumerate(trials):
            if i and settings.break_every and i % settings.break_every == 0:
                screen_key(f"Short break.  {i} of {len(trials)} done.\n\n"
                           "Press SPACE to go on.")
            if not one(trial, False)["counts"]:
                broken.append(trial)
        if settings.repeat_broken and broken:
            rng.shuffle(broken)
            for trial in broken:
                counts["repeated"] += 1
                one(trial, True)
    except Aborted:
        aborted = True
    finally:
        if monitor_started:
            monitor.end()

    return Block2Outcome(settings=settings, rows=rows, rates=rates, refresh_hz=refresh,
                         contrast=contrast, background_cd_m2=mean_luminance,
                         calibration_source=calibration.source,
                         calibrated=calibration.calibrated, started=started,
                         duration_s=time.perf_counter() - t0, aborted=aborted, counts=counts,
                         fixation_monitor=(monitor.describe() if watching
                                           else settings.fixation_monitoring), setup=setup)


def write_outputs(outcome: Block2Outcome, directory: pathlib.Path) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    if outcome.rows:
        with open(directory / "trials.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(outcome.rows[0]))
            w.writeheader()
            w.writerows(outcome.rows)
    summary = outcome.summary()
    with open(directory / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    meta = dict(block="2 (quick test, 17 Nov 2026)", started=outcome.started,
                duration_s=round(outcome.duration_s, 1), aborted=outcome.aborted,
                settings=dataclasses.asdict(outcome.settings), places=outcome.settings.places(),
                measured_refresh_hz=round(outcome.refresh_hz, 3), rates=outcome.rates,
                contrast=round(outcome.contrast, 4),
                background_cd_m2=round(outcome.background_cd_m2, 3),
                calibration=dict(source=outcome.calibration_source,
                                 calibrated=outcome.calibrated),
                counts=outcome.counts, fixation_monitor=outcome.fixation_monitor,
                setup=outcome.setup,
                honest_limits=["Timing is software self-report, not photodiode-validated.",
                               "Rates are whole frames per reversal: shown_hz, not "
                               "requested_hz, is what was on the screen."])
    (directory / "session.json").write_text(json.dumps(meta, indent=2))
    return directory


def results_text(outcome: Block2Outcome) -> str:
    """Per place: percent right over all rates (chance = 50 %)."""
    lines = ["Block 2" + ("  (STOPPED EARLY)" if outcome.aborted else ""), ""]
    for place in outcome.settings.places():
        rows = [r for r in outcome.rows if r["place"] == place and r["counts"]]
        right = sum(r["correct"] for r in rows)
        pct = f"{100 * right / len(rows):.0f}%" if rows else "-"
        lines.append(f"  {place:12s} {right}/{len(rows)} right  ({pct}; chance 50%)")
    val = (outcome.setup.get("gaze_check") or {}).get("validation") or {}
    if val.get("mean_accuracy_deg") is not None:
        lines += ["", f"Tracker accuracy: {val['mean_accuracy_deg']} deg (centre "
                  f"{val.get('centre_accuracy_deg')} deg)"]
    if outcome.setup.get("blind_spot_used"):
        lines.append(f"Blind spot: {outcome.setup['blind_spot_used']} "
                     f"{outcome.settings.places().get(BLIND_SPOT)}")
    lines += ["", f"Eye off the +: {outcome.counts['fixation_breaks']}   "
              f"dropped frames: {outcome.counts['dropped_frames']}"]
    return "\n".join(lines)
