"""Block 2, quick test (plan of 17 Nov 2026, for the Elliott meeting).

No new perimetry: Block 1's saved map (``summary.csv`` + ``session.json`` of one run) gives
the places and the contrast of every patch:

* every tested spot of the map (the 30 spots of "24-2 within 9") at **its own threshold
  contrast** from Block 1 (the "%" in the map's boxes);
* the spots Block 1 marked "not seen" (red), and the natural blind spot (the grid spots
  Block 1 dropped as blind spot, hatched), at **the highest contrast** the screen gives.

Two stripe directions, horizontal (LEFT arrow) and vertical (RIGHT arrow), each shown
twice per spot: 4 patches per spot per condition. Conditions, one phase after another:
**static** first (the patch does not reverse); when it is done, **20 and 38 Hz** pattern
reversal, mixed in a random order.

Every trial: a random gap of 500-800 ms (only the +), the patch for 2 s, then 1.2 s more to
answer (an answer while the patch is up counts too). **Every trial lasts the same**: an
early answer does not shorten it, and only the first answer counts, so a second press can
never land in the next trial (lab report 17 Nov). A press less than 250 ms after onset is
an anticipation (it is ignored and the row says ANTICIPATION). No press = not seen.
**Catch trials** (nothing shown, same timing) are mixed in; a press on one is a false press.

``forced_choice`` (off by default; for blindsight, as in Sanchez-Lopez et al. and Elliott):
the person must guess on every trial; a trial with no answer in time is shown again once at
the end. With ``awareness_rating`` a 1-3 rating follows each trial (1 = saw nothing, 2 =
something, 3 = clearly); "seen" is then a rating of 2 or 3, and the direction answers on
unseen trials are tested against chance (50 %).

Results per spot and condition: shown, seen (pressed), right and wrong direction; overall:
true-positive (seen) rate, false-press rate on catch trials.

**Rates are whole frames**: refresh / k (see :mod:`specvis.core.reversal`); the asked-for
and the shown rate are both written to every row. The rate shown depends on the refresh
PsychoPy measures at the start, not the one Windows names: at 239.76 Hz 38 -> 39.96 Hz,
at 240.5 Hz (the lab BenQ, 17 Nov) 20 -> 20.04 and 38 -> 40.09 Hz. Set the screen to its
highest refresh first.

Time (lab, 17 Nov): about 38 min per eye with the setup and the breaks.

Thresholds were measured with Block 1's 45-degree patch; the patches here are horizontal
and vertical (usually a little easier to see than oblique ones).
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

from ..core.flatfield import NO_FLAT_FIELD, describe as describe_flat_field
from ..core.gabor import GaborParameters, render_brightness
from ..core.geometry import PhysicalScreen
from ..core.luminance import LuminanceCalibration
from ..core.reversal import ReversalPlan
from ..stimulus.backend import StimulusBackend, Texture
from .fixation import FIX_NONE, FixationMonitor, judge_patch, monitor_for
from .perimetry import (FULL_THRESHOLD, H24_2_9, PerimetrySettings, background_for,
                        default_calibration, flat_field_info, run_perimetry)
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
#: The answer keys: LEFT = horizontal, RIGHT = vertical.
ANSWER_KEYS = {"left": HORIZONTAL, "right": VERTICAL}
#: The awareness rating (``awareness_rating``): top-row or keypad digits.
RATING_KEYS = {"1": 1, "2": 2, "3": 3, "num_1": 1, "num_2": 2, "num_3": 3}
STATIC = 0.0                     # a "rate" of 0 Hz = a static patch
CATCH = "catch"
BLIND_SPOT = "blind spot"
#: Kinds of place.
THRESHOLD, NOT_SEEN = "at threshold", "not seen in Block 1"
QUADRANTS = {"upper right": (5.0, 5.0), "upper left": (-5.0, 5.0),
             "lower left": (-5.0, -5.0), "lower right": (5.0, -5.0)}


@dataclass(frozen=True)
class Place:
    name: str
    x_deg: float
    y_deg: float
    kind: str                    # THRESHOLD, NOT_SEEN or BLIND_SPOT
    #: Block 1's threshold contrast; 0 = the highest the screen gives.
    contrast: float = 0.0
    block1_db: str = ""


def load_block1(folder) -> Tuple[List[Place], dict]:
    """Places and contrasts from one Block 1 run (its summary.csv and session.json)."""
    folder = pathlib.Path(folder)
    try:
        meta = json.loads((folder / "session.json").read_text())
        with open(folder / "summary.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, ValueError) as e:
        raise SessionError(f"cannot read the Block 1 run in {folder}: {e}")
    places: List[Place] = []
    for r in rows:
        x, y = float(r["x_deg"]), float(r["y_deg"])
        name = f"({x:+g}, {y:+g})"
        valid = str(r.get("valid")) == "True" and r.get("threshold_contrast")
        if valid and r.get("category") in ("seen", "weak"):
            places.append(Place(name, x, y, THRESHOLD, float(r["threshold_contrast"]),
                                r.get("threshold_db", "")))
        else:
            places.append(Place(name, x, y, NOT_SEEN, 0.0, r.get("category", "")))
    for x, y in meta.get("blind_spot_grid_points_dropped") or []:
        places.append(Place(f"blind spot ({x:+g}, {y:+g})", float(x), float(y), BLIND_SPOT))
    if not places:
        raise SessionError(f"the Block 1 run in {folder} has no spots")
    return places, meta


@dataclass
class Block2Settings:
    participant_id: str = "pilot"
    eye: str = "R"                        # the tested eye; the other one is covered
    # The lab screen and seating (preset 4); taken from Block 1's run when one is given.
    screen_width_mm: float = 542.3
    screen_height_mm: float = 305.0
    viewing_distance_mm: float = 570.0
    background_cd_m2: float = 10.0
    # The patch (the proposal: 2 deg, 4 c/deg).
    diameter_degrees: float = 2.0
    spatial_frequency_cpd: float = 4.0
    #: Block 1's run folder (the map); None = the 4 quadrants + the average blind spot
    #: at the highest contrast (try-outs only).
    block1_folder: Optional[str] = None
    #: Where the blind-spot patch goes without a Block 1 run; None = the average.
    blind_spot: bool = True
    blind_spot_temporal_deg: float = 15.5
    blind_spot_below_deg: float = 1.5
    blind_spot_xy: Optional[Tuple[float, float]] = None
    #: Block 1's setup first (eye-tracker calibration and accuracy check; no perimetry).
    setup: bool = True
    orientations: Tuple[str, ...] = (HORIZONTAL, VERTICAL)
    repeats: int = 2                      # each direction twice per spot per condition
    #: Phases, one after another: static first, then 20 and 38 Hz mixed (17 Nov plan).
    phases: Tuple[Tuple[float, ...], ...] = ((STATIC,), (20.0, 38.0))
    train_ms: float = 2000.0              # the patch is up for 2 s
    response_ms: float = 1200.0           # then 1.2 s more to answer
    iti_min_ms: float = 500.0             # random gap before each patch
    iti_max_ms: float = 800.0
    #: Catch trials (nothing shown) as a share of the patches in each phase: our
    #: suggestion, 10 %.
    catch_fraction: float = 0.10
    #: A press sooner than this after onset is an anticipation, not an answer (our
    #: suggestion, from the lab report of 17 Nov).
    anticipation_ms: float = 250.0
    #: Off = "press only if seen" (detection). On = a guess on every trial (needed to show
    #: blindsight); trials with no answer in time are shown again once at the end.
    forced_choice: bool = False
    #: After each trial: 1 = saw nothing, 2 = saw something, 3 = saw it clearly.
    awareness_rating: bool = False
    break_every: int = 40
    #: After SPACE (start, breaks) only the + shows for this long (lab report 16 Nov).
    resume_delay_ms: float = 1200.0
    # Eye tracker: a patch counts when at least half its samples are valid and at most
    # 25 % of them are more than 3 deg off the + -- Block 1's preset 4 rule (our
    # suggestion, to tune). 2 deg / 10 % threw out 93 % of patches on 17 Nov, with the
    # tracker's own centre error at 1.5 deg.
    fixation_monitoring: str = FIX_NONE
    fixation_tolerance_deg: float = 3.0
    fixation_max_out_frac: float = 0.25
    repeat_broken: bool = False
    cancel_key: str = "escape"

    def validate(self) -> None:
        if self.eye not in ("L", "R"):
            raise SessionError("Block 2 tests one eye (L or R): the blind spot is only "
                               "blind with the other eye covered")
        rates = [f for ph in self.phases for f in ph]
        if not rates or any(f < 0 for f in rates):
            raise SessionError("every rate must be 0 (static) or above")
        if any(o not in ORIENTATION_DEG for o in self.orientations):
            raise SessionError(f"orientations must be among {list(ORIENTATION_DEG)}")
        if self.train_ms <= 0 or self.viewing_distance_mm <= 0 or self.screen_width_mm <= 0:
            raise SessionError("patch length, distance and screen width must be above 0")
        if not 0 <= self.iti_min_ms <= self.iti_max_ms:
            raise SessionError("the gap must be 0 or more, min no larger than max")

    @property
    def temporal_sign(self) -> int:
        return 1 if self.eye == "R" else -1

    @property
    def rates(self) -> List[float]:
        out: List[float] = []
        for ph in self.phases:
            out += [f for f in ph if f not in out]
        return out

    def places(self) -> List[Place]:
        if self.block1_folder:
            return [p for p in load_block1(self.block1_folder)[0]
                    if self.blind_spot or p.kind != BLIND_SPOT]
        out = [Place(n, x, y, NOT_SEEN) for n, (x, y) in QUADRANTS.items()]
        if self.blind_spot:
            x, y = (tuple(self.blind_spot_xy) if self.blind_spot_xy else
                    (self.temporal_sign * self.blind_spot_temporal_deg,
                     -self.blind_spot_below_deg))
            out.append(Place(BLIND_SPOT, x, y, BLIND_SPOT))
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
        """Block 1 as preset 4, setup only: positioning, calibration, accuracy check
        (the blind-spot search only without a Block 1 map)."""
        return dataclasses.replace(
            self.perimetry_settings(), strategy=FULL_THRESHOLD, extent=H24_2_9,
            pilot_mode=True, dithering=False, fixation_monitoring=self.fixation_monitoring,
            setup_only=True, blind_spot_search=self.blind_spot and not self.block1_folder,
            **(PRESET4_TRACKER if self.fixation_monitoring == "eye tracker" else {}))


def settings_from_block1(settings: Block2Settings) -> Block2Settings:
    """Take the eye, screen, seating, background and patch from the Block 1 run, so the
    thresholds mean the same thing here."""
    if not settings.block1_folder:
        return settings
    _, meta = load_block1(settings.block1_folder)
    b1 = meta.get("settings") or {}
    keep = {k: b1[k] for k in ("eye", "screen_width_mm", "screen_height_mm",
                               "viewing_distance_mm", "background_cd_m2",
                               "diameter_degrees", "spatial_frequency_cpd")
            if b1.get(k) not in (None, "")}
    return dataclasses.replace(settings, **keep)


@dataclass(frozen=True)
class Block2Trial:
    place: Optional[Place]                # None = a catch trial
    orientation: str                      # "" for a catch trial
    requested_hz: float
    phase: int


def plan_phase(settings: Block2Settings, phase: int, rng: random.Random) -> List[Block2Trial]:
    """One phase: every place x direction x rate, ``repeats`` times, plus catch trials,
    in a random order."""
    rates = settings.phases[phase]
    trials = [Block2Trial(p, o, float(f), phase)
              for p in settings.places() for o in settings.orientations for f in rates
              for _ in range(settings.repeats)]
    n_catch = round(settings.catch_fraction * len(trials))
    trials += [Block2Trial(None, "", float(rates[i % len(rates)]), phase)
               for i in range(n_catch)]
    rng.shuffle(trials)
    return trials


def plan_trials(settings: Block2Settings, rng: random.Random) -> List[Block2Trial]:
    out: List[Block2Trial] = []
    for i in range(len(settings.phases)):
        out += plan_phase(settings, i, rng)
    return out


def rate_label(hz: float) -> str:
    return "static" if hz == STATIC else f"{hz:g} Hz"


def rate_table(frequencies: Sequence[float], refresh_hz: float) -> List[dict]:
    """What each asked-for rate becomes on this screen (static stays static)."""
    rows = []
    for f in frequencies:
        if f == STATIC:
            rows.append(dict(requested_hz=f, frames_per_reversal=0, shown_hz=0.0,
                             off_by_hz=0.0, exact=True))
            continue
        plan = ReversalPlan.for_rate(f, refresh_hz)
        shown = refresh_hz / plan.frames_per_reversal
        rows.append(dict(requested_hz=f, frames_per_reversal=plan.frames_per_reversal,
                         shown_hz=round(shown, 2), off_by_hz=round(shown - f, 2),
                         exact=abs(shown - f) < 0.05))
    return rows


def _pct(a: int, b: int):
    return round(100.0 * a / b) if b else ""


def binomial_p(right: int, n: int, chance: float = 0.5):
    """One-sided: the chance of ``right`` or more of ``n`` by guessing."""
    if not n:
        return ""
    return sum(math.comb(n, k) * chance ** k * (1 - chance) ** (n - k)
               for k in range(right, n + 1))


@dataclass
class Block2Outcome:
    settings: Block2Settings
    rows: List[dict]
    rates: List[dict]
    refresh_hz: float
    background_cd_m2: float
    calibration_source: str
    calibrated: bool
    started: str
    duration_s: float
    aborted: bool
    counts: Dict[str, int]
    fixation_monitor: str
    setup: Dict[str, object] = field(default_factory=dict)
    #: The flat-field (screen evenness) correction used, or {"name": "none"}.
    flat_field: Dict[str, object] = field(default_factory=dict)

    def _patch_rows(self) -> List[dict]:
        return [r for r in self.rows if r["place"] != CATCH and r["counts"]]

    def summary(self) -> List[dict]:
        """Per spot and condition: shown, seen (pressed), right, wrong, not seen."""
        out = []
        rows = self._patch_rows()
        for p in self.settings.places():
            for rate in self.rates:
                rs = [r for r in rows if r["place"] == p.name
                      and r["requested_hz"] == rate["requested_hz"]]
                pressed = [r for r in rs if r["seen"]]
                unseen = [r for r in rs if not r["seen"] and r["answer"]]
                right = sum(r["correct"] for r in pressed)
                right_unseen = sum(r["correct"] for r in unseen)
                out.append(dict(place=p.name, x_deg=p.x_deg, y_deg=p.y_deg, kind=p.kind,
                                block1_db=p.block1_db,
                                contrast=rs[0]["contrast"] if rs else "",
                                condition=rate_label(rate["requested_hz"]),
                                shown_hz=rate["shown_hz"], shown=len(rs),
                                seen=len(pressed), right=right,
                                wrong=len(pressed) - right, not_seen=len(rs) - len(pressed),
                                seen_pct=_pct(len(pressed), len(rs)),
                                right_pct_of_shown=_pct(right, len(rs)),
                                right_pct_of_seen=_pct(right, len(pressed)),
                                unseen_answered=len(unseen), right_unseen=right_unseen,
                                right_pct_of_unseen=_pct(right_unseen, len(unseen))))
        return out

    def overall(self) -> List[dict]:
        """Per condition: true positives (patch seen), false presses on catch trials."""
        out = []
        for rate in self.rates + [None]:
            def pick(rs):
                return [r for r in rs if rate is None
                        or r["requested_hz"] == rate["requested_hz"]]
            patches = pick(self._patch_rows())
            catches = pick([r for r in self.rows if r["place"] == CATCH])
            fp = sum(1 for r in catches if r["seen"])
            all_trials = len(patches) + len(catches)
            for kind in (None, THRESHOLD, NOT_SEEN, BLIND_SPOT):
                ps = [r for r in patches if kind is None or r["kind"] == kind]
                if kind is not None and not ps:
                    continue
                k_tp = sum(1 for r in ps if r["seen"])
                k_right = sum(r["correct"] for r in ps if r["seen"])
                un = [r for r in ps if not r["seen"] and r["answer"]]
                un_right = sum(r["correct"] for r in un)
                p = binomial_p(k_right, k_tp)
                p_un = binomial_p(un_right, len(un))
                out.append(dict(
                    condition="all" if rate is None else rate_label(rate["requested_hz"]),
                    places="all" if kind is None else kind,
                    patches=len(ps), true_positives=k_tp, tp_pct=_pct(k_tp, len(ps)),
                    misses=len(ps) - k_tp, right=k_right, wrong=k_tp - k_right,
                    right_pct_of_seen=_pct(k_right, k_tp),
                    p_seen_vs_50=p if p == "" else float(f"{p:.3g}"),
                    unseen_answered=len(un), right_unseen=un_right,
                    right_pct_of_unseen=_pct(un_right, len(un)),
                    p_unseen_vs_50=p_un if p_un == "" else float(f"{p_un:.3g}"),
                    catch_trials=len(catches) if kind is None else "",
                    catch_pct_of_trials=_pct(len(catches), all_trials) if kind is None else "",
                    false_positives=fp if kind is None else "",
                    fp_pct=_pct(fp, len(catches)) if kind is None else ""))
        return out


def run_block2(settings: Block2Settings, backend: StimulusBackend,
               calibration: Optional[LuminanceCalibration] = None,
               rng: Optional[random.Random] = None, show_instructions: bool = True,
               fixation_monitor: Optional[FixationMonitor] = None) -> Block2Outcome:
    settings = settings_from_block1(settings)
    settings.validate()
    rng = rng or random.Random()
    ps = settings.perimetry_settings()
    calibration = calibration or default_calibration(ps)
    abort_key = settings.cancel_key
    monitor = fixation_monitor or monitor_for(settings.fixation_monitoring, settings.eye)
    places = settings.places()
    zero = dict(trials=0, catch=0, repeated=0, fixation_breaks=0, dropped_frames=0)

    # -- Block 1's setup: tracker calibration + accuracy check (no perimetry) ------------
    setup: Dict[str, object] = {}
    if settings.setup and show_instructions and (
            settings.fixation_monitoring != FIX_NONE or not settings.block1_folder):
        if settings.block1_folder:
            key = backend.show_screen(
                "Block 2: the stripes test.\n\nFirst a short eye-tracker check "
                "(look at the targets), then the stripes.\nThis is NOT the flash test "
                "again.\n\nPress SPACE to go on.", keys=["space", settings.cancel_key])
            if key == settings.cancel_key:
                return Block2Outcome(settings=settings, rows=[], rates=[],
                                     refresh_hz=backend.refresh_hz, background_cd_m2=0.0,
                                     calibration_source=calibration.source,
                                     calibrated=calibration.calibrated,
                                     started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     duration_s=0.0, aborted=True, counts=dict(zero),
                                     fixation_monitor=settings.fixation_monitoring)
        first = run_perimetry(settings.setup_settings(), backend, calibration=calibration,
                              rng=rng, show_instructions=True, fixation_monitor=monitor)
        found = (first.blind_spot_search or {}).get("found_xy")
        setup = dict(gaze_check=first.gaze_check, blind_spot_search=first.blind_spot_search)
        if first.aborted:
            return Block2Outcome(settings=settings, rows=[], rates=[],
                                 refresh_hz=backend.refresh_hz, background_cd_m2=0.0,
                                 calibration_source=calibration.source,
                                 calibrated=calibration.calibrated,
                                 started=time.strftime("%Y-%m-%dT%H:%M:%S"), duration_s=0.0,
                                 aborted=True, counts=dict(zero),
                                 fixation_monitor=settings.fixation_monitoring, setup=setup)
        if found and not settings.block1_folder:
            settings = dataclasses.replace(settings, blind_spot_xy=tuple(found))
            places = settings.places()
    setup["places_from"] = (f"Block 1 run {pathlib.Path(settings.block1_folder).name}"
                            if settings.block1_folder else
                            "no Block 1 run: 4 quadrants + blind spot, highest contrast")

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
    for pl in places:
        name, x, y = pl.name, pl.x_deg, pl.y_deg
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

    # -- timing and patches ----------------------------------------------------------------
    refresh = backend.refresh_hz
    if not refresh or math.isnan(refresh):
        raise SessionError("the display's refresh rate could not be measured")
    rates = rate_table(settings.rates, refresh)
    frames_for = {r["requested_hz"]: r["frames_per_reversal"] for r in rates}
    total_frames = max(1, round(settings.train_ms / 1000.0 * refresh))
    mean_luminance, _ = background_for(ps, calibration)
    top = min(1.0, calibration.max_achievable_contrast(mean_luminance))
    sigma = settings.diameter_degrees / 6.0
    contrast_of = {pl.name: (min(pl.contrast, top) if pl.contrast > 0 else top)
                   for pl in places}
    textures: Dict[Tuple[str, str], List[Texture]] = {}
    for pl in places:
        for o in settings.orientations:
            p = GaborParameters(settings.diameter_degrees, settings.spatial_frequency_cpd,
                                ORIENTATION_DEG[o], 0.0, sigma, mean_luminance,
                                contrast_of[pl.name])
            textures[(pl.name, o)] = [
                Texture(render_brightness(p, calibration, ppd_x, ppd_y),
                        to_px(pl.x_deg, pl.y_deg)),
                Texture(render_brightness(p.reversed_phase(), calibration, ppd_x, ppd_y),
                        to_px(pl.x_deg, pl.y_deg))]

    watching = getattr(monitor, "watches_gaze", False)
    rows: List[dict] = []
    counts = dict(zero)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.perf_counter()
    aborted = False
    answer_keys = list(ANSWER_KEYS)

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
        gap = rng.uniform(settings.iti_min_ms, settings.iti_max_ms)
        wait(gap)                               # (clears any key pressed before)
        static = trial.requested_hz == STATIC
        k = frames_for[trial.requested_hz]
        catch = trial.place is None
        if catch:
            # Nothing shown: the + alone for as long as a patch would be up.
            t_on = time.perf_counter()
            presses = []
            left = settings.train_ms
            while left > 1:
                key, rt = backend.wait_for_key(answer_keys + [abort_key], left,
                                               fixation=fixation)
                if key == abort_key:
                    raise Aborted
                if key is None:
                    break
                presses.append((key, settings.train_ms - left + rt))
                left = settings.train_ms - presses[-1][1]
            t_off = time.perf_counter()
            frames, dropped, measured = 0, 0, refresh
        else:
            tex = textures[(trial.place.name, trial.orientation)]
            t_on = time.perf_counter()
            report = backend.present_train(tex[:1] if static else tex,
                                           total_frames if static else k, total_frames,
                                           fixation=fixation,
                                           response_keys=answer_keys + [abort_key])
            t_off = time.perf_counter()
            if any(key == abort_key for key, _ in report.keys):
                raise Aborted
            presses = [(key, rt) for key, rt in report.keys if key in ANSWER_KEYS]
            frames, dropped = report.frames_presented, report.dropped_frames
            measured = report.measured_refresh_hz or refresh
            monitor.patch_shown(t_on, t_off)
        anticipated = any(rt < settings.anticipation_ms for _, rt in presses)
        on_time = [(key, rt) for key, rt in presses if rt >= settings.anticipation_ms]
        if on_time:
            key, rt = on_time[0]
            wait(settings.response_ms)          # the window runs out; later keys ignored
        else:
            key, rt = backend.wait_for_key(answer_keys + [abort_key], settings.response_ms,
                                           fixation=fixation)
            if key == abort_key:
                raise Aborted
            if key is not None:
                wait(max(0.0, settings.response_ms - rt))
                rt += settings.train_ms
        rating = ""
        if settings.awareness_rating:
            got = backend.show_screen(
                "Did you see a patch?\n\n1 = nothing     2 = something     3 = clearly",
                keys=list(RATING_KEYS) + [abort_key])
            if got == abort_key:
                raise Aborted
            rating = RATING_KEYS[got]
        counts["catch" if catch else "trials"] += 1
        counts["dropped_frames"] += dropped
        answer = ANSWER_KEYS.get(key or "", "")
        seen = (rating >= 2) if settings.awareness_rating else bool(answer)
        pl = trial.place
        notes = (["ANTICIPATION"] if anticipated else []) + (
            ["NO_ANSWER"] if settings.forced_choice and not answer and not catch else [])
        row = dict(trial=len(rows) + 1, ms_from_start=int((time.perf_counter() - t0) * 1000),
                   phase=trial.phase + 1, condition=rate_label(trial.requested_hz),
                   place=CATCH if catch else pl.name, kind=CATCH if catch else pl.kind,
                   x_deg="" if catch else pl.x_deg, y_deg="" if catch else pl.y_deg,
                   orientation=trial.orientation, answer=answer,
                   seen=int(seen), awareness=rating,
                   outcome=(("false press" if seen else "correct rejection") if catch else
                            ("miss" if not answer else
                             "right" if answer == trial.orientation else "wrong")),
                   correct=int(bool(answer) and answer == trial.orientation),
                   requested_hz=trial.requested_hz, frames_per_reversal=k,
                   shown_hz=0.0 if static or not k else round(refresh / k, 3),
                   measured_refresh_hz=round(measured, 3), train_frames=frames,
                   dropped_frames=dropped,
                   contrast="" if catch else round(contrast_of[pl.name], 4),
                   block1_db="" if catch else pl.block1_db, gap_ms=round(gap),
                   reaction_ms_from_onset="" if rt is None else round(rt),
                   repeat=int(repeat), gaze_max_deg="", gaze_valid_pct="",
                   counts=0 if "NO_ANSWER" in notes else 1, note=" ".join(notes))
        if watching and not catch:
            ok, worst, frac = judge_patch(monitor.between(t_on, t_off),
                                          settings.fixation_tolerance_deg,
                                          max_out_frac=settings.fixation_max_out_frac)
            row.update(gaze_max_deg="" if math.isnan(worst) else round(worst, 2),
                       gaze_valid_pct=round(100 * frac))
            if not ok:
                counts["fixation_breaks"] += 1
                row.update(counts=0, note=" ".join(["FIXATION_BREAK"] + notes))
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
        for phase in range(len(settings.phases)):
            trials = plan_phase(settings, phase, rng)
            if show_instructions:
                lines = "\n".join(
                    f"  {rate_label(r['requested_hz'])} -> "
                    + ("static" if r["requested_hz"] == STATIC else f"{r['shown_hz']:g} Hz")
                    + ("" if r["exact"] else f"  ({r['off_by_hz']:+g})")
                    for r in rates if r["requested_hz"] in settings.phases[phase])
                n_catch = sum(1 for t in trials if t.place is None)
                screen_key(
                    f"For the operator. Part {phase + 1} of {len(settings.phases)}: "
                    f"{len(trials)} trials ({n_catch} catch). Screen {refresh:.2f} Hz:\n\n"
                    f"{lines}\n\n{setup['places_from']}.\n"
                    f"Cover the {'left' if settings.eye == 'R' else 'right'} eye.\n\n"
                    "SPACE = go on")
                screen_key(
                    "Keep looking at the  +  in the middle all the time.\n\n"
                    "A striped patch will "
                    + ("appear" if settings.phases[phase] == (STATIC,) else "flicker")
                    + " for 2 seconds somewhere around it.\n"
                    + ("Say which way its stripes went, EVERY time:\n\n"
                       if settings.forced_choice else
                       "If you see it, say which way its stripes went:\n\n")
                    + "LEFT arrow  =  horizontal stripes  (like  =  )\n"
                    "RIGHT arrow  =  vertical stripes  (like  ||  )\n\n"
                    "Answer ONCE. You may answer while the patch is still on.\n"
                    + ("If you saw nothing, guess (quickly).\n" if settings.forced_choice
                       else "If you see nothing, do not press.\n")
                    + ("After each one, say how well you saw it: 1 = nothing, "
                       "2 = something, 3 = clearly.\n" if settings.awareness_rating else "")
                    + "\nPress SPACE to start.")
            broken: List[Block2Trial] = []
            for i, trial in enumerate(trials):
                if i and settings.break_every and i % settings.break_every == 0:
                    screen_key(f"Short break.  {i} of {len(trials)} done.\n\n"
                               "Press SPACE to go on.")
                row = one(trial, False)
                if not row["counts"] and (settings.repeat_broken
                                          or "NO_ANSWER" in row["note"]):
                    broken.append(trial)
            if broken:
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
                         background_cd_m2=mean_luminance,
                         calibration_source=calibration.source,
                         calibrated=calibration.calibrated, started=started,
                         duration_s=time.perf_counter() - t0, aborted=aborted, counts=counts,
                         fixation_monitor=(monitor.describe() if watching
                                           else settings.fixation_monitoring), setup=setup,
                         flat_field=flat_field_info(backend))


def _write_csv(path: pathlib.Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_outputs(outcome: Block2Outcome, directory: pathlib.Path) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory / "trials.csv", outcome.rows)
    _write_csv(directory / "summary.csv", outcome.summary())
    _write_csv(directory / "overall.csv", outcome.overall())
    meta = dict(block="2 (quick test, plan of 17 Nov 2026)", started=outcome.started,
                duration_s=round(outcome.duration_s, 1), aborted=outcome.aborted,
                settings=dataclasses.asdict(outcome.settings),
                places=[dataclasses.asdict(p) for p in outcome.settings.places()],
                measured_refresh_hz=round(outcome.refresh_hz, 3), rates=outcome.rates,
                background_cd_m2=round(outcome.background_cd_m2, 3),
                calibration=dict(source=outcome.calibration_source,
                                 calibrated=outcome.calibrated),
                flat_field=outcome.flat_field or dict(NO_FLAT_FIELD),
                counts=outcome.counts, fixation_monitor=outcome.fixation_monitor,
                setup=outcome.setup, overall=outcome.overall(),
                honest_limits=["Timing is software self-report, not photodiode-validated.",
                               "Rates are whole frames per reversal: shown_hz, not "
                               "requested_hz, is what was on the screen.",
                               "Thresholds come from Block 1's 45-degree patch; these "
                               "patches are horizontal and vertical."])
    (directory / "session.json").write_text(json.dumps(meta, indent=2, default=str))
    return directory


def results_text(outcome: Block2Outcome) -> str:
    """Per condition: seen (true positives), right direction, false presses."""
    lines = ["Block 2" + ("  (STOPPED EARLY)" if outcome.aborted else ""), ""]
    for r in outcome.overall():
        if r["places"] == "all":
            lines.append(f"{r['condition']}:  seen {r['true_positives']}/{r['patches']}"
                         f" ({r['tp_pct']}%)  right {r['right']}  wrong {r['wrong']}  |"
                         f"  catch {r['catch_trials']} ({r['catch_pct_of_trials']}% of "
                         f"trials), false presses {r['false_positives']} ({r['fp_pct']}%)")
        else:
            lines.append(f"    {r['places']:22s} seen {r['true_positives']}/{r['patches']}"
                         f" ({r['tp_pct']}%)  right {r['right']}  wrong {r['wrong']}")
    val = (outcome.setup.get("gaze_check") or {}).get("validation") or {}
    if val.get("mean_accuracy_deg") is not None:
        lines += ["", f"Tracker accuracy: {val['mean_accuracy_deg']} deg (centre "
                  f"{val.get('centre_accuracy_deg')} deg)"]
    if (outcome.setup.get("gaze_check") or {}).get("operator_choice"):
        lines.append("WARNING: the eye-tracker accuracy check FAILED; the run went on.")
    lines += ["", f"Eye off the +: {outcome.counts['fixation_breaks']}   "
              f"dropped frames: {outcome.counts['dropped_frames']}",
              describe_flat_field(outcome.flat_field)]
    return "\n".join(lines)
