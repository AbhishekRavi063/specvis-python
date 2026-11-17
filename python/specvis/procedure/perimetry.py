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
4. **Pilot mode** switches the starting guesses off, so the 15-20 table
   people (tested with ZEST (full)) give clean, independent thresholds for the table.
5. **SITA Fast** (``strategy``, the default) puts SITA Fast's published rules around
   ZEST (see :mod:`specvis.threshold.zest`): level-dependent stopping, early stop
   after one "seen" at normal points, a two-model starting guess, a second search at
   points far from their guess, quicker blind confirmation, pacing that follows the
   participant's reaction times, presses too fast to be real counted as false, and a
   final re-estimate with the measured false-answer rates. The four seed points keep
   the strict search, as SITA's primary points do.

**Background** (Padmini, after SITA): 10 cd/m^2, the Humphrey's. Without a
photometer scale the display is *assumed* (gamma 2.2, 300 cd/m^2 peak) and every
result says so. On an 8-bit screen so dark a background cannot draw faint patches
(one grey step is ~4% contrast), so patches are **dithered** (noisy-bit method).

Catch trials, as in ``untitled.py``: blank presentations (false alarms), strong
presentations at points already seen well (misses), and presentations at the
blind spot (fixation losses -- only meaningful with the other eye covered).

**Coordinates.** Points are in visual-field degrees, **x right and y up**, relative to
fixation, as on a clinical printout. They are placed on the flat screen with the
tangent (``d * tan``) model, so a point labelled 21 degrees really is 21 degrees.
The patch itself is drawn at the central pixels-per-degree, as PsychoPy's
``degFlatPos`` units do.

**Fixation** (``fixation_monitoring``): the blind-spot check (Heijl-Krakau), none, or
gaze -- an eye tracker (Tobii Pro Fusion, :mod:`specvis.procedure.fixation`), or the
mouse standing in for one in testing. With gaze, a patch is shown only once the eye has
held the + for ``fixation_hold_ms``; a trial in which the eye left the + during the
patch is thrown away and the point is tested again later. The blind-spot trials are
then dropped, and their share goes to the other two checks.

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
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.gabor import (GaborParameters, dither_8bit, measure_contrast,
                          minimum_renderable_contrast, nominal_contrast, render_brightness,
                          snap_background)
from ..core.geometry import PhysicalScreen
from ..core.luminance import LuminanceCalibration
from ..stimulus.backend import Mark, StimulusBackend, Texture
from ..threshold.base import Termination
from ..threshold.full_threshold import FullThresholdLocation, FullThresholdSettings
from ..threshold.zest import ZEST, ZestLocationThreshold, ZestSettings
from .fixation import (FIX_BLIND_SPOT, FIX_NONE, FIXATION_MODES, GAZE_MODES,
                       FixationMonitor, LiveFixation, judge_patch, monitor_for)
from .prior_table import PriorTable
from .session import Aborted, SessionError

SEEN_KEY = "space"
ABORT_KEY = "escape"

#: Tobii's standard 5-point calibration (Tobii Pro SDK calibration example): the centre
#: and 4 points 10% in from the corners, as fractions of the display area (top-left 0,0).
TRACKER_CAL_POINTS = ((0.5, 0.5), (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9))
#: Validation as in Titta (Niehorster et al. 2020): a diamond between the calibration
#: points, plus the centre (where the + is). The centre comes first in reports.
TRACKER_VAL_POINTS = ((0.5, 0.5), (0.5, 0.25), (0.75, 0.5), (0.5, 0.75), (0.25, 0.5))

#: Aflah's 72-point grid (untitled.py): 4 deg apart across, 6 deg up/down, no point on
#: a midline.
GRID_X = (1, 5, 9, 13, 17, 21)
GRID_Y = (1, 7, 13)
FULL, CENTRAL, H24_2 = "full", "central", "24-2"
#: The 24-2 without its top and bottom rows (+/-21 deg) and its 2 nasal 27-deg points:
#: what fits a 24.5" screen at 50-53 cm, the Tobii Pro Fusion's range (50-80 cm).
H24_2_CENTRAL = "24-2 central"
#: The 24-2 positions within +/-15 deg in every direction (3, 9, 15 on both axes): 6 deg
#: apart everywhere and equal on all sides (Padmini, 14 Nov). 36 points; the 2 blind-spot
#: points left out -> 34 tested. Fits the 24.5" BenQ at 50-52.8 cm (the Tobii: 50-80 cm).
H24_2_15 = "24-2 within 15"
#: The 24-2 positions within +/-9 deg up/down (rows at 3 and 9 deg), out to 21 deg across
#: (16 Nov, for the meeting with Elliott): 32 points; the 2 blind-spot points left out
#: -> 30 tested. Fits the 24.5" BenQ at 57 cm with room to spare (up to ~67 cm), where the
#: Tobii Pro Fusion tracks well; the +/-15 deg rows do not fit beyond 53.2 cm.
H24_2_9 = "24-2 within 9"
#: A 5-deg grid (15 Nov, our own; not a clinical grid): 5 deg apart everywhere, 2.5 deg
#: off both midlines (like the 24-2's 3 deg), out to 22.5 deg across and 12.5 deg up/down.
#: 60 positions; those whose patch overlaps the blind spot are left out (4) -> 56 tested.
#: Made for the Tobii Pro Fusion at 55-62 cm on the 24.5" BenQ (it does not track
#: properly at 50-52 cm), with the + in the centre.
GRID_5 = "5 deg grid"
GRID_5_X = (2.5, 7.5, 12.5, 17.5, 22.5)
GRID_5_Y = (2.5, 7.5, 12.5)
GRID_5_SEEDS = ((-7.5, 7.5), (7.5, 7.5), (-7.5, -7.5), (7.5, -7.5))
#: The physiological blind spot's usual size, for leaving out grid points: about 5.5 deg
#: wide and 7.5 deg high (textbook average; its centre is set by blind_spot_temporal_deg /
#: blind_spot_below_deg). A point is left out when its patch centre is within this box
#: widened by 0.5 deg.
BLIND_SPOT_HALF_W, BLIND_SPOT_HALF_H, BLIND_SPOT_MARGIN = 2.75, 3.75, 0.5
EXTENTS = (H24_2_15, H24_2_9, H24_2, H24_2_CENTRAL, GRID_5, FULL, CENTRAL)
#: The grids offered in the app (15 Nov: only the study grid, so nobody picks another by
#: mistake). The others stay in the code so older sessions and settings still load.
APP_EXTENTS = (H24_2_CENTRAL, H24_2_9, GRID_5)
#: What the grid choices are called in the app, so they can be compared at a glance.
EXTENT_NAMES = {
    H24_2_15: "24-2 within +/-15 deg: 34 spots, 6 deg apart, equal on all sides (50-52.8 cm)",
    H24_2_9: "24-2: 30 spots, 6 deg apart, rows at 3 and 9 deg only (57 cm, eye tracker)",
    H24_2: "24-2: 52 spots, 6 deg apart (needs ~33-37 cm; too close for the eye tracker)",
    H24_2_CENTRAL: "24-2: 42 points, 6 deg apart, without the top/bottom rows (51 cm)",
    GRID_5: "5 deg grid: 56 points, 5 deg apart everywhere (up to 62 cm, eye tracker)",
    FULL: "Aflah grid: 68 spots, 4 x 6 deg apart (50-60 cm)",
    CENTRAL: "Aflah grid, central: 46 spots (+/-13 deg)",
}
#: The Humphrey 24-2 (the clinical standard; also used on flat screens, e.g. Melbourne
#: Rapid Fields): points 6 deg apart, 3 deg off both midlines, out to 21 deg, plus two
#: nasal points at 27 deg. 54 points; the 2 at 15 deg temporal, 3 deg above and below,
#: lie in the blind spot and are left out, as in the Humphrey's analysis -> 52 tested.
H24_2_ROWS = {3: (3, 9, 15, 21), 9: (3, 9, 15, 21), 15: (3, 9, 15), 21: (3, 9)}
H24_2_15_ROWS = {3: (3, 9, 15), 9: (3, 9, 15), 15: (3, 9, 15)}
H24_2_9_ROWS = {3: (3, 9, 15, 21), 9: (3, 9, 15, 21)}
H24_2_NASAL = 27
H24_2_SEEDS = ((-9, 9), (9, 9), (-9, -9), (9, -9))   # the Humphrey's primary points

# Point and trial kinds, as written to trials.csv
ZEST_TRIAL, PROBE_TRIAL = "zest", "zest_probe_max"
STAIRCASE_TRIAL = "staircase"             # a Full Threshold (4-2) trial
BLANK, HIGH, BLIND_SPOT, PRACTICE = "blank_catch", "high_catch", "blind_spot", "practice"
BLIND_SPOT_SEARCH, SEARCH_CHECK = "blind_spot_search", "blind_spot_search_check"
FIXATION_BREAK = "FIXATION_BREAK"

# Strategies
SITA_FAST, ZEST_FULL, FULL_THRESHOLD = "sita_fast", "zest", "full_threshold"
STRATEGIES = (SITA_FAST, ZEST_FULL, FULL_THRESHOLD)
STRATEGY_NAMES = {SITA_FAST: "SITA Fast (ZEST engine)", ZEST_FULL: "ZEST (full)",
                  FULL_THRESHOLD: "Full Threshold 4-2 (Humphrey, as in OPI)"}

# Map categories
SEEN, WEAK, BLIND, NOT_FINISHED = "seen", "weak", "blind", "not_finished"


@dataclass
class PerimetrySettings:
    participant_id: str = "pilot"
    eye: str = "R"                        # the eye tested; the other is covered
    #: The lab setup (BenQ 24.5", Tobii Pro Fusion 50-80 cm): the 24-2 without its top/bottom
    #: rows at 52 cm. H24_2 = the full 24-2 (52 points, ~33-37 cm); FULL = Aflah's +/-21 x +/-13
    #: (68 points, 50-60 cm); CENTRAL = +/-13 x +/-13
    extent: str = H24_2_CENTRAL
    pilot_mode: bool = True               # no starting guesses: clean data for the table
    strategy: str = FULL_THRESHOLD        # the team's choice (2026-11-13): no normal table

    # Screen, measured
    screen_width_mm: float = 542.3        # the lab BenQ 24.5" 16:9: 542.3 x 305.0 mm, the size
                                          # a 24.5" 16:9 panel has and the eye tracker uses
                                          # (the monitor reports a rounded 544 x 303)
    viewing_distance_mm: float = 510.0    # in the eye tracker's 50-80 cm; the study grid
                                          # fits up to 52.8 cm on the BenQ
    #: The screen's height as measured or reported (BenQ: 305.0 mm). Used, when smaller than
    #: width x pixel ratio, for the "does it fit" check (the safe side); spots are placed
    #: with square pixels, as the panel's are. 0 = from the width and the pixels.
    screen_height_mm: float = 305.0

    # Stimulus: the ReFreM SSVEP Gabor, shown static
    diameter_degrees: float = 2.0
    spatial_frequency_cpd: float = 4.0
    orientation_degrees: float = 45.0
    mean_luminance: float = 50.0          # on the calibration's own scale; used only
                                          # when background_cd_m2 is 0
    # Background (Padmini, after SITA/Humphrey)
    background_cd_m2: float = 10.0        # 0 = use mean_luminance as given
    assumed_max_cd_m2: float = 300.0      # display model when there is no photometer scale
    assumed_gamma: float = 2.2
    dithering: bool = True                # noisy-bit: faint patches on a dark background

    # Timing (untitled.py)
    stimulus_ms: float = 200.0
    fixation_ms: float = 0.0              # extra wait before each patch (the + is always on)
    response_window_ms: float = 1200.0    # after the stimulus ends (the most it can be)
    iti_ms: Tuple[float, float] = (300.0, 600.0)
    # SITA-style pacing
    adaptive_window: bool = True          # the window follows the participant's reaction times
    min_response_window_ms: float = 500.0
    miss_extra_pause_ms: float = 300.0    # SITA Fast's extra pause after a "not seen"
    early_press_ms: float = 180.0         # a press sooner than this after onset is a false press
    post_process: bool = True             # re-estimate with the measured false-answer rates
    practice_trials: int = 3
    break_every_trials: int = 60

    # Catch trials and validity (untitled.py)
    p_blank: float = 0.05                 # fewer: early presses also count as false presses
    p_high: float = 0.05
    p_blind_spot: float = 0.05
    high_contrast: float = 0.90
    high_catch_max_threshold: float = 0.30
    blind_spot_contrast: float = 0.80
    blind_spot_temporal_deg: float = 15.5  # Rohrschneider 2004: 15.5 deg temporal,
    blind_spot_below_deg: float = 1.5      # 1.5 deg below fixation
    #: Find this person's blind spot before the test (as the Humphrey does), starting at
    #: the average position above, instead of using the average for everyone (Padmini).
    blind_spot_search: bool = True
    blind_spot_search_step_deg: float = 1.5   # spacing of the positions tried
    blind_spot_search_repeats: int = 2        # a position must be missed this many times
    blind_spot_search_attempts: int = 3       # searches before the average is used
    max_false_alarm_rate: float = 0.15
    max_miss_rate: float = 0.10
    max_fixation_loss_rate: float = 0.20
    #: A check fails only with at least this many bad answers as well as a rate over its
    #: limit. With ~10 catch trials of each kind (SITA Fast keeps them few), one or two
    #: chance lapses would otherwise fail a good session about 1 time in 6.
    min_failures: int = 3

    # Speed-ups
    seed_points: Tuple[Tuple[int, int], ...] = ((-9, 7), (9, 7), (-9, -7), (9, -7))
    #: Neighbours for starting guesses / the growth pattern: same quadrant, this close.
    #: 8.5 takes in the 24-2's diagonal neighbours (8.49 deg); on the old grid it picks
    #: exactly the same neighbours as 8.0 did.
    neighbour_radius_deg: float = 8.5
    prior_sd_log: float = 0.5             # deliberately broad
    prior_flat_weight: float = 0.2
    # SITA Fast's starting guess: narrower (a normal table is trusted more), with SITA's
    # "abnormal" model in place of most of the flat part. Tuned by simulation.
    fast_prior_sd_log: float = 0.25
    fast_prior_flat_weight: float = 0.05
    fast_prior_sd_better_log: float = 0.12   # nobody is much better than normal
    prior_abnormal_weight: float = 0.2    # Turpin et al. 2003: abnormal:normal 1:4
    #: SITA's 12 dB rule: reopen a point this far from its guess. 12 dB is about a third
    #: of the Humphrey's normal range; our contrast range is ~2.5 log units, so 0.8.
    retest_log: float = 0.8
    #: Without a normal table, a threshold above this counts as "weak" on the map.
    #: Provisional: replace with the table's normal range once it exists.
    weak_contrast: float = 0.20

    # Fixation and keys (Specvis step 4)
    fixation_colour: str = "white"
    fixation_monitoring: str = FIX_BLIND_SPOT  # "none", "eye tracker" or "mouse" (testing)
    # With gaze (eye tracker / mouse)
    fixation_tolerance_deg: float = 2.5   # the eye must stay within this of the +
    fixation_hold_ms: float = 200.0       # held this long before a patch is shown
    fixation_wait_ms: float = 5000.0      # then a "look at the +" reminder
    #: Forgiveness (our suggestions, to tune on the lab PC; patients find it hard to hold
    #: the + exactly). A tracker gap up to dropout_ms keeps the last on/off state (the lab
    #: PC fix, 15 Nov); gaze off the + for no longer than off_grace_ms (noise, a flick of
    #: the eye) does not count as off; and a patch is a fixation break only when more than
    #: break_max_out_frac of its valid gaze samples are off the + (0 = any one sample).
    fixation_dropout_ms: float = 100.0
    fixation_off_grace_ms: float = 0.0
    fixation_break_max_out_frac: float = 0.0
    #: Blinks (our suggestions). The eye not seen is never "looking away": no red +, no
    #: fixation break. A gap in the tracked eye of at least blink_min_ms during a patch
    #: that was not seen = the patch may have fallen in a blink: it is repeated (0 = only
    #: when the eye was missing for most of the patch). After the eye comes back from a
    #: gap longer than fixation_dropout_ms, the next patch waits post_blink_delay_ms more
    #: (on top of the hold), so the lid and the eye settle first.
    blink_min_ms: float = 0.0
    post_blink_delay_ms: float = 0.0
    #: Lab report 16 Nov (our choice): no patch straight after SPACE or after looking
    #: back. After any screen closed with a key (start, break, pause, reminder) only the +
    #: shows, and the eye must be on it for resume_delay_ms before the next patch. After
    #: the eye was off the + (red +, or a trial thrown away for an eye movement) it must
    #: be back on it for look_away_settle_ms. Both come on top of fixation_hold_ms; without
    #: an eye tracker resume_delay_ms is a plain wait.
    resume_delay_ms: float = 1200.0
    look_away_settle_ms: float = 600.0
    #: Feedback to the person (our addition; Sanchez-Lopez et al. gave constant feedback
    #: on fixation): the + turns red while the eye is off it, and after a trial thrown
    #: away for an eye movement "Keep looking at the +" shows for a moment.
    fixation_feedback: bool = True
    fixation_off_colour: str = "red"
    #: Eye tracker: calibrate each person at the start, Tobii-style (a dot that shrinks
    #: and bursts at the centre and near the 4 corners), then measure the accuracy at 5
    #: dots with the tested eye. The accuracy limits are our suggestion (not from a
    #: published source): well under the 2.5 deg fixation window.
    tracker_calibrate: bool = True
    tracker_validate: bool = True
    tracker_max_centre_error_deg: float = 1.0
    tracker_max_mean_error_deg: float = 1.5
    tracker_max_data_loss_pct: float = 20.0     # our suggestion
    #: Tobii's monocular calibration of the tested eye only (lab report 16 Nov, B4; Tobii
    #: Pro SDK ScreenBasedMonocularCalibration). Off by default until the lab compares the
    #: validation accuracy with it on and off.
    tracker_monocular_calibration: bool = False
    #: Eye openness (Tobii Pro Fusion; lab report B3). A gap in the gaze during which the
    #: lid fell below this share of the person's own median opening is a blink; a gap with
    #: the lid open throughout is lost tracking (not the person's fault, logged apart).
    #: 0.5 is our suggestion, to tune from the pilot's eye_openness_min_mm column. Without
    #: valid openness the earlier rule (any long gap = a blink) is used.
    blink_openness_frac: float = 0.5
    #: The tracker's own idea of the screen size (its display area) must match the
    #: settings, or every gaze position is off; checked at the start (tolerance in mm).
    display_area_tolerance_mm: float = 3.0
    #: How high the eyes sit above the tracker when they are level with the screen centre
    #: (half the screen height + the bezel): the middle of the positioning screen.
    tracker_eye_height_mm: float = 170.0
    #: The tracker measures how far the eye is (Tobii gaze origin). At the start the test
    #: asks to move the chin rest when it is further than this from the set viewing
    #: distance (our suggestion); every trial records the distance measured.
    check_eye_distance: bool = True
    eye_distance_tolerance_mm: float = 10.0
    answer_key: str = "space"
    pause_key: str = "p"
    cancel_key: str = "escape"
    background_brightness_pct: float = 50.0   # with a luminance scale, sets the mean

    zest: ZestSettings = field(default_factory=ZestSettings)

    # Full Threshold (strategy FULL_THRESHOLD): the Humphrey 4-2 staircase as in OPI.
    # No normal table: seed points start at ft_start_contrast, the rest from finished
    # neighbours (the growth pattern). The contrast range is the ZEST one above.
    #: 0 = the middle of the measurable range in dB (strongest the screen can show at
    #: this background, to the faintest: one grey step, or the search floor when
    #: dithered). No normal values are needed; the start does not change the results
    #: (simulation), only the time. A positive value is used as given.
    ft_start_contrast: float = 0.0
    ft_coarse_step_db: float = 8.0        # contrast dB; = Humphrey 4 dB of light
    ft_fine_step_db: float = 4.0          # = Humphrey 2 dB
    ft_retest_db: float = 8.0             # second staircase if this far from the start
    ft_max_presentations: int = 30        # safety net per point
    #: NOT part of the Humphrey/OPI method (our suggestion, off by default): a point
    #: whose finished neighbours are all blind starts at the maximum contrast.
    ft_blind_neighbour_start: bool = False

    def full_threshold_settings(self) -> FullThresholdSettings:
        return FullThresholdSettings(self.zest.min_contrast, self.zest.max_contrast,
                                     self.ft_coarse_step_db, self.ft_fine_step_db,
                                     self.ft_retest_db, self.ft_max_presentations)

    def validate(self) -> None:
        if len({self.answer_key, self.pause_key, self.cancel_key}) != 3:
            raise SessionError("the answer, pause and cancel keys must be different")
        if self.eye not in ("L", "R"):
            raise SessionError("perimetry is monocular: eye must be 'L' or 'R'")
        if self.extent not in EXTENTS:
            raise SessionError(f"extent must be one of {EXTENTS}")
        if self.fixation_monitoring not in FIXATION_MODES:
            raise SessionError(f"fixation monitoring must be one of {FIXATION_MODES}")
        if self.strategy not in STRATEGIES:
            raise SessionError(f"strategy must be one of {STRATEGIES}")
        if self.strategy == FULL_THRESHOLD:
            try:
                self.full_threshold_settings().validate()
            except ValueError as e:
                raise SessionError(f"Full Threshold: {e}") from None
            if self.ft_start_contrast and not (self.zest.min_contrast <= self.ft_start_contrast
                                               <= self.zest.max_contrast):
                raise SessionError("Full Threshold: the start contrast must be inside "
                                   "the contrast range")
        if self.p_blank + self.p_high + self.p_blind_spot >= 0.5:
            raise SessionError("catch trials must be under half of all trials")
        if self.stimulus_ms <= 0 or self.viewing_distance_mm <= 0 or self.screen_width_mm <= 0:
            raise SessionError("stimulus time, distance and screen width must be positive")
        if (self.blind_spot_search_repeats < 1 or self.blind_spot_search_attempts < 1
                or self.blind_spot_search_step_deg <= 0):
            raise SessionError("the blind-spot search needs at least 1 repeat, 1 attempt "
                               "and a positive step")

    @property
    def watches_gaze(self) -> bool:
        return self.fixation_monitoring in GAZE_MODES

    def catch_rates(self) -> Tuple[float, float, float]:
        """(blank, strong, blind spot) shares. With gaze there are no blind-spot trials;
        their share goes half to each of the other two checks."""
        if self.fixation_monitoring == FIX_BLIND_SPOT:
            return self.p_blank, self.p_high, self.p_blind_spot
        if self.watches_gaze:
            return self.p_blank + self.p_blind_spot / 2, self.p_high + self.p_blind_spot / 2, 0.0
        return self.p_blank, self.p_high, 0.0

    @property
    def temporal_sign(self) -> int:
        """The blind spot is on the temporal side: right for the right eye."""
        return 1 if self.eye == "R" else -1


@dataclass
class PerimetryPoint:
    index: int
    x_deg: float                       # right +
    y_deg: float                       # up +
    #: The point's threshold search: ZEST, or a Full Threshold staircase (same questions).
    zest: ZestLocationThreshold
    started: bool = False
    prior_source: str = ""
    expected_log: Optional[float] = None   # the starting guess, for SITA's retest rule
    seed: bool = False

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


def seeds_for(settings: PerimetrySettings) -> Tuple[Tuple[float, float], ...]:
    """One seed (primary) point per quadrant: the settings' own when they are on the
    grid, else the grid's standard ones (the Humphrey's (+/-9, +/-9) on the 24-2)."""
    kept, _ = grid_points(settings)
    on_grid = {(float(x), float(y)) for x, y in kept}
    own = tuple((float(x), float(y)) for x, y in settings.seed_points)
    if all(p in on_grid for p in own):
        return own
    if settings.extent in (H24_2, H24_2_CENTRAL, H24_2_15, H24_2_9):
        return tuple((float(x), float(y)) for x, y in H24_2_SEEDS)
    if settings.extent == GRID_5:
        return GRID_5_SEEDS
    return own


def contrast_range_db(settings: PerimetrySettings, calibration: LuminanceCalibration
                      ) -> Dict[str, float]:
    """The measurable range in contrast dB (larger = fainter) at this background.

    strongest: the largest contrast the screen can show around the background without
    clipping (and not above the search maximum). faintest: one 8-bit grey step at the
    background; with dithering, the search floor (``zest.min_contrast``), which must
    be checked with the photometer. middle: halfway, in dB -- the Full Threshold seeds'
    start when ``ft_start_contrast`` is 0.
    """
    mean, _ = background_for(settings, calibration)
    strong = min(settings.zest.max_contrast, calibration.max_achievable_contrast(mean))
    step = minimum_renderable_contrast(calibration, mean)
    faint = settings.zest.min_contrast if settings.dithering else max(step, settings.zest.min_contrast)
    strong = max(strong, faint)
    s_db, f_db = -20 * math.log10(strong), -20 * math.log10(faint)
    return dict(strongest_db=round(s_db, 2), faintest_db=round(f_db, 2),
                middle_db=round((s_db + f_db) / 2, 2), grey_step_db=round(-20 * math.log10(step), 2),
                strongest_contrast=strong, faintest_contrast=faint,
                background_cd_m2=round(mean, 3), calibrated=calibration.calibrated)


def grid_points(settings: PerimetrySettings) -> Tuple[List[Tuple[int, int]],
                                                       List[Tuple[int, int]]]:
    """The test points, and the grid points dropped because they fall in the blind spot.

    Returned as (x, y) in field degrees. The dropped points lie in the physiological
    blind spot (temporal 12-18 deg, 3 below to 1 above), where every eye is blind;
    they are checked by the blind-spot catch trials instead.
    """
    if settings.extent == GRID_5:
        bx, by = settings.temporal_sign * settings.blind_spot_temporal_deg, -settings.blind_spot_below_deg
        pts = sorted(((sx * x, sy * y) for x in GRID_5_X for y in GRID_5_Y
                      for sx in (-1, 1) for sy in (-1, 1)), key=lambda p: (math.hypot(*p), p))
        kept, dropped = [], []
        for x, y in pts:
            in_spot = (abs(x - bx) <= BLIND_SPOT_HALF_W + BLIND_SPOT_MARGIN
                       and abs(y - by) <= BLIND_SPOT_HALF_H + BLIND_SPOT_MARGIN)
            (dropped if in_spot else kept).append((x, y))
        return kept, dropped
    if settings.extent in (H24_2, H24_2_CENTRAL, H24_2_15, H24_2_9):
        central = settings.extent != H24_2
        rows = {H24_2_15: H24_2_15_ROWS, H24_2_9: H24_2_9_ROWS}.get(settings.extent, H24_2_ROWS)
        pts = [(sx * x, sy * y) for y, row in rows.items() for x in row
               for sx in (-1, 1) for sy in (-1, 1) if not (central and y == 21)]
        if not central:
            nasal = -settings.temporal_sign * H24_2_NASAL
            pts += [(nasal, 3), (nasal, -3)]
        pts.sort(key=lambda p: (math.hypot(*p), p))
        kept, dropped = [], []
        for x, y in pts:
            (dropped if x * settings.temporal_sign == 15 and abs(y) == 3 else kept).append((x, y))
        return kept, dropped
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
    background_cd_m2: float = float("nan")
    post_processing: Dict[str, float] = field(default_factory=dict)
    fixation_monitor: str = ""
    gaze_check: Dict[str, object] = field(default_factory=dict)
    gaze_samples: List[Tuple[float, Optional[float], Optional[float]]] = field(default_factory=list)
    #: Where the blind-spot catch trials went (found by the search, or the average).
    blind_spot_xy: Optional[Tuple[float, float]] = None
    blind_spot_search: Dict[str, object] = field(default_factory=dict)
    #: The measurable contrast range at this background and the seeds' start (dB).
    contrast_range: Dict[str, object] = field(default_factory=dict)

    @property
    def early_press_rate(self) -> float:
        """Presses too soon after onset to be a response to the patch, per patch shown."""
        n = self.counts.get("presentations", 0)
        return self.counts.get("early_presses", 0) / n if n else float("nan")

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
        """Blind-spot patches seen, or with gaze, patches thrown away for an eye movement."""
        if self.settings.watches_gaze:
            n = self.counts.get("gaze_checked", 0)
            return self.counts.get("fixation_breaks", 0) / n if n else float("nan")
        n = self.counts["blind_spot"]
        return self.counts["blind_spot_seen"] / n if n else float("nan")

    @property
    def fixation_failures(self) -> int:
        return self.counts.get("fixation_breaks" if self.settings.watches_gaze
                               else "blind_spot_seen", 0)

    @property
    def checks(self) -> Dict[str, bool]:
        """Each reliability check, True when passed. A check with no trials fails."""
        s = self.settings

        def under(rate: float, limit: float, bad: int) -> bool:
            return not math.isnan(rate) and (rate <= limit or bad < s.min_failures)

        c = self.counts
        return {
            "false_alarms": under(self.false_alarm_rate, s.max_false_alarm_rate,
                                  c["false_alarms"]),
            "misses": under(self.miss_rate, s.max_miss_rate, c["misses"]),
            # Fixation monitoring switched off: not checked, so not failed. session.json
            # records that it was off.
            "fixation": (s.fixation_monitoring == FIX_NONE
                         or under(self.fixation_loss_rate, s.max_fixation_loss_rate,
                                  self.fixation_failures)),
            "completed": not self.aborted,
        }

    @property
    def valid(self) -> bool:
        return all(self.checks.values())


def blind_spot_position(settings: PerimetrySettings) -> Tuple[float, float]:
    """The average blind spot (where the search starts), in field degrees (x right,
    y up)."""
    return (settings.temporal_sign * settings.blind_spot_temporal_deg,
            -settings.blind_spot_below_deg)


def blind_spot_candidates(settings: PerimetrySettings) -> List[Tuple[float, float]]:
    """The positions the blind-spot search tries, nearest the average first.

    A grid ``blind_spot_search_step_deg`` apart, up to two steps each way from the
    average (a 5 x 5 square without its corners), kept where blind spots lie: 11-20 deg
    temporal, at most 1 deg above the + (the blind spot is centred slightly below it).
    With the defaults: 18 positions, 12.5 to 18.5 deg temporal and 0 to 4.5 deg below.
    """
    x0, y0 = blind_spot_position(settings)
    step, sign = settings.blind_spot_search_step_deg, settings.temporal_sign
    found = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if abs(i) == abs(j) == 2:          # a 5 x 5 square without its corners
                continue
            x, y = x0 + sign * i * step, y0 + j * step
            if 11.0 <= sign * x <= 20.0 and y <= 1.0:
                found.append((round(x, 3), round(y, 3)))
    return sorted(found, key=lambda p: (round(math.hypot(p[0] - x0, p[1] - y0), 6),
                                        abs(p[1] - y0), p))


def background_for(settings: PerimetrySettings, calibration: LuminanceCalibration
                   ) -> Tuple[float, float]:
    """(mean luminance, brightness 0-1) of the background, on a real 8-bit grey level."""
    target = settings.background_cd_m2 if settings.background_cd_m2 > 0 else \
        settings.mean_luminance
    return snap_background(calibration, target)


def default_calibration(settings: PerimetrySettings) -> LuminanceCalibration:
    """Without a photometer scale: an assumed display when the background is asked for
    in cd/m^2, else the old linear 0-100 scale."""
    if settings.background_cd_m2 > 0:
        return LuminanceCalibration.assumed_display(settings.assumed_max_cd_m2,
                                                    settings.assumed_gamma)
    return LuminanceCalibration.uncalibrated()


def zest_settings_for(settings: PerimetrySettings, seed: bool) -> ZestSettings:
    """Seeds keep the strict search (SITA's primary points); the rest run SITA Fast."""
    if settings.strategy != SITA_FAST or seed:
        return settings.zest
    return dataclasses.replace(settings.zest, fast_stop=True)


def fit_height_mm(settings: PerimetrySettings, width_px: int, height_px: int) -> float:
    """The screen height used to check that the grid fits: the measured / reported height
    when it is the smaller (the safe side), else width x the pixels' aspect ratio."""
    from_pixels = settings.screen_width_mm * height_px / width_px
    if settings.screen_height_mm > 0:
        return min(from_pixels, settings.screen_height_mm)
    return from_pixels


def field_half_degrees(settings: PerimetrySettings, width_px: int, height_px: int
                       ) -> Tuple[float, float]:
    """Half the field the screen covers, horizontally and vertically, in degrees."""
    height_mm = fit_height_mm(settings, width_px, height_px)
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
    half_h_mm = fit_height_mm(settings, width_px, height_px) / 2.0
    radius_mm = d * math.radians(settings.diameter_degrees / 2.0)   # central scale
    kept, _ = grid_points(settings)
    hx, hy = field_half_degrees(settings, width_px, height_px)
    extra = [blind_spot_position(settings)]
    if settings.blind_spot_search and settings.fixation_monitoring == FIX_BLIND_SPOT:
        extra = blind_spot_candidates(settings)
    for x, y in kept + extra:
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
                  on_trial: Optional[Callable[[dict], None]] = None,
                  fixation_monitor: Optional[FixationMonitor] = None) -> PerimetryOutcome:
    """Run one eye on an already-open backend and return what it produced.

    ``fixation_monitor`` overrides the one ``settings.fixation_monitoring`` names (tests
    pass a scripted one)."""
    settings.validate()
    seen_key, abort_key, pause_key = settings.answer_key, settings.cancel_key, settings.pause_key
    calibration = calibration or default_calibration(settings)
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
    mean_luminance, background = background_for(settings, calibration)
    base = GaborParameters(settings.diameter_degrees, settings.spatial_frequency_cpd,
                           settings.orientation_degrees, 0.0, sigma, mean_luminance, 1.0)
    # Dithered patches are shown correctly on average below one grey step; the floor is
    # then the search range, not the 8-bit step.
    floor = (0.0 if settings.dithering
             else minimum_renderable_contrast(calibration, mean_luminance))
    blank = np.full(render_brightness(base, calibration, ppd_x, ppd_y).shape, background)
    fixation = (centre, max(12.0, 0.5 * ppd_x))
    # Full Threshold: never ask for more contrast than the screen can show at this
    # background (a clipped patch would be logged at a contrast it never had); the seeds
    # start at the middle of the range unless a start is given.
    db_range = contrast_range_db(settings, calibration)
    ft_settings = dataclasses.replace(settings.full_threshold_settings(),
                                      max_contrast=db_range["strongest_contrast"])
    ft_start = (settings.ft_start_contrast if settings.ft_start_contrast
                else 10 ** (-db_range["middle_db"] / 20))
    ft_start = min(ft_settings.max_contrast, max(ft_settings.min_contrast, ft_start))
    db_range["ft_start_db"] = round(-20 * math.log10(ft_start), 2)
    db_range["ft_start_rule"] = ("given" if settings.ft_start_contrast
                                 else "middle of the measurable range")
    noise = np.random.default_rng(rng.randrange(2 ** 32))

    def texture(x: float, y: float, contrast: float) -> Texture:
        image = render_brightness(base.with_contrast(contrast), calibration, ppd_x, ppd_y)
        if settings.dithering:
            image = dither_8bit(image, noise)
        return Texture(image, to_px(x, y))

    # -- fixation monitor ---------------------------------------------------------------
    def to_deg(px: float, py: float) -> Tuple[float, float]:
        return (math.degrees(math.atan((px - centre[0]) / screen.px_per_mm_x / d)),
                math.degrees(math.atan((centre[1] - py) / screen.px_per_mm_y / d)))

    monitor = fixation_monitor or monitor_for(settings.fixation_monitoring, settings.eye)
    watching = settings.watches_gaze
    p_blank, p_high, p_blind_spot = settings.catch_rates()
    gaze_check: Dict[str, object] = {}

    # -- state -------------------------------------------------------------------------
    seed_set = {(float(x), float(y)) for x, y in seeds_for(settings)}
    points = []
    for i, (x, y) in enumerate(kept):
        is_seed = (float(x), float(y)) in seed_set
        if settings.strategy == FULL_THRESHOLD:
            finder = FullThresholdLocation(ft_settings, ft_start, renderable_floor=floor)
        else:
            finder = ZestLocationThreshold(zest_settings_for(settings, is_seed),
                                           renderable_floor=floor)
        points.append(PerimetryPoint(i, float(x), float(y), finder, seed=is_seed))
    seeds = [p for p in points if p.seed]
    counts = dict(trials=0, blank=0, false_alarms=0, high=0, misses=0, blind_spot=0,
                  blind_spot_seen=0, dropped_frames=0, presentations=0, early_presses=0,
                  retests=0, gaze_checked=0, fixation_breaks=0)
    gaze_info: Dict[str, object] = {}      # about the last patch, for its trials.csv row
    reaction_times: List[float] = []
    rows: List[dict] = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    aborted = False

    def wait_key(keys, ms, clear_events=True, on_frame=None):
        """wait_for_key, with ``on_frame`` called every refresh where the screen can."""
        watching_wait = getattr(backend, "wait_for_key_watching", None)
        if on_frame is not None and watching_wait is not None:
            return watching_wait(keys, ms, fixation=fixation, clear_events=clear_events,
                                 on_frame=on_frame)
        return backend.wait_for_key(keys, ms, fixation=fixation, clear_events=clear_events)

    # Settle time (ms) the next patch must wait, with the eye on the +, before it comes:
    # set after key screens (resume_delay_ms) and after the eye was off the + (look_away_settle_ms).
    pending_settle = [0.0]

    def need_settle(ms: float) -> None:
        pending_settle[0] = max(pending_settle[0], float(ms))

    def wait(ms: float, on_frame=None) -> None:
        key, _ = wait_key([abort_key, pause_key], ms, on_frame=on_frame)
        if key == abort_key:
            raise Aborted
        if key == pause_key:
            key = backend.show_screen(
                f"PAUSED.\n\nPress {seen_key.upper()} to continue   ({abort_key.upper()} stops).",
                keys=[seen_key, abort_key])
            if key == abort_key:
                raise Aborted
            need_settle(settings.resume_delay_ms)

    def response_window() -> float:
        """SITA's timing: after a few answers the window follows this participant's
        reaction times (mean + 2.5 SD), never above the set window."""
        if not settings.adaptive_window or len(reaction_times) < 6:
            return settings.response_window_ms
        recent = reaction_times[-30:]
        limit = float(np.mean(recent) + 2.5 * np.std(recent)) - display_ms
        return float(np.clip(limit, settings.min_response_window_ms,
                             settings.response_window_ms))

    POLL_MS = 16.0
    BOX_EDGE = 0.15                 # our choice: the eyes kept off the outer 15 % of the box

    def fixation_colour(colour: str) -> None:
        """The + in red while the eye is off it (feedback on); back to normal on it."""
        if settings.fixation_feedback and hasattr(backend, "fixation_colour"):
            backend.fixation_colour = colour

    def wait_for_fixation() -> None:
        """Hold the patch back until the eye has been on the + for ``fixation_hold_ms``.
        Time is counted in the waits asked for, so a test without a screen runs at
        full speed; on a real screen each wait is at least that long."""
        held = waited = 0.0
        # Lab PC, 15 Nov: a time buffer. The tracker drops single samples very
        # often at 50 cm (19 % missing, >1000 gaps under 0.1 s in a minute), and each one
        # turned the + red and restarted the hold. Now a gap of up to DROPOUT_MS keeps the
        # last valid state; only gaze seen outside the tolerance, or lost for longer, is "off".
        # A trial that starts inside a gap is "not known yet", not "off": no red + until
        # the gap is longer than DROPOUT_MS (but no hold time is counted either).
        # A gap longer than that (a blink, or the eye out of the tracker's view) is not
        # "looking away" either: the + stays as it is, the hold starts again, and when the
        # eye is back the patch waits post_blink_delay_ms more.
        DROPOUT_MS = settings.fixation_dropout_ms
        last_on, missing, off_for = None, 0.0, 0.0
        settle, pending_settle[0] = pending_settle[0], 0.0
        came_back = False
        while True:
            if pending_settle[0] > 0:                 # e.g. P pressed during a poll below
                settle, pending_settle[0] = max(settle, pending_settle[0]), 0.0
            trouble = monitor.problem()
            if trouble:
                # Lab report B6: pause rather than test on without a working tracker.
                counts["tracker_pauses"] = counts.get("tracker_pauses", 0) + 1
                key = backend.show_screen(
                    f"PAUSED: {trouble}.\n\nCheck the eye tracker (and the bridge window), "
                    f"then press {seen_key.upper()} to go on   ({abort_key.upper()} stops).",
                    keys=[seen_key, abort_key])
                if key == abort_key:
                    raise Aborted
                held = waited = 0.0
                settle = max(settle, settings.resume_delay_ms)
                continue
            g = monitor.current()
            if g is not None:
                if missing > DROPOUT_MS:              # back from a blink
                    settle = max(settle, settings.post_blink_delay_ms)
                    counts["blinks_between"] = counts.get("blinks_between", 0) + 1
                missing = 0.0
                if math.hypot(*g) <= settings.fixation_tolerance_deg:
                    if last_on is False and not came_back:   # back from looking away
                        came_back = True             # once per patch: flicks cannot stall it
                        settle = max(settle, settings.look_away_settle_ms)
                    last_on, off_for = True, 0.0
                else:
                    off_for += POLL_MS
                    if off_for > settings.fixation_off_grace_ms:
                        last_on = False
                on = bool(last_on)
            else:
                missing += POLL_MS
                if missing > DROPOUT_MS:
                    last_on, off_for = None, 0.0      # not known: a blink, not looking away
                on = bool(last_on) and missing <= DROPOUT_MS
            unknown = last_on is None and off_for <= settings.fixation_off_grace_ms
            fixation_colour(settings.fixation_colour if on or unknown
                            else settings.fixation_off_colour)
            if on and settle > 0:
                settle -= POLL_MS                     # the eye is back: let it settle
                on = False
            if on:
                held += POLL_MS
                if held >= settings.fixation_hold_ms:
                    return
            else:
                held = 0.0
            waited += POLL_MS
            if waited >= settings.fixation_wait_ms:
                key = backend.show_screen(
                    f"Please look at the  +  in the middle.\n\nPress {seen_key.upper()} "
                    f"to go on   ({abort_key.upper()} stops).", keys=[seen_key, abort_key])
                if key == abort_key:
                    raise Aborted
                counts["fixation_reminders"] = counts.get("fixation_reminders", 0) + 1
                waited = held = 0.0
                settle = max(settle, settings.resume_delay_ms)
            wait(POLL_MS)

    def frames(ms: float) -> int:
        return max(1, int(round(ms / 1000.0 * refresh)))

    def abc_target() -> np.ndarray:
        """The calibration / validation target of Thaler et al. (2013, Vision Res 76:31):
        a bull's-eye with a cross ('ABC'), which gave the steadiest fixation; Titta's
        default. Outer disc 0.6 deg, cross and inner dot 0.2 deg, on the background."""
        r_out = max(4.0, 0.3 * ppd_x)
        half_bar = max(1.0, 0.1 * ppd_x)
        r_in = max(1.5, 0.1 * ppd_x)
        r = int(math.ceil(r_out)) + 1
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
        dist = np.hypot(xx, yy)
        img = np.full(dist.shape, background, dtype=float)
        img[dist <= r_out] = 1.0
        img[(np.abs(xx) <= half_bar) | (np.abs(yy) <= half_bar)] = background
        img[dist > r_out] = background
        img[dist <= r_in] = 1.0
        return img

    TARGET = None

    def show_target(centre: Tuple[float, float], ms: float) -> None:
        nonlocal TARGET
        if TARGET is None:
            TARGET = abc_target()
        backend.present_train([Texture(TARGET, centre)], 0, frames(ms), fixation=None)

    def hold_target_while(centre: Tuple[float, float], job) -> object:
        """Keep the target on screen while ``job`` runs (the tracker collects data)."""
        out = {}

        def run():
            try:
                out["value"] = job()
            except BaseException as e:                 # noqa: BLE001 -- raised below
                out["error"] = e
        th = threading.Thread(target=run, daemon=True)
        th.start()
        while th.is_alive():
            show_target(centre, 50.0)
        if "error" in out:
            raise out["error"]
        return out.get("value")

    def calibrate_tracker() -> dict:
        """Tobii's standard 5-point calibration (Tobii Pro SDK example): the centre and 4
        points 10% in from the corners, in random order (as Titta); at each, the target is
        shown and, after 0.7 s ('wait a little for user to focus'), the data collected."""
        w, h = width_px, height_px
        pts = list(TRACKER_CAL_POINTS)
        rng.shuffle(pts)
        if settings.tracker_monocular_calibration:
            monitor.cal_enter(settings.eye)        # the tested eye only (lab report B4)
        else:
            monitor.cal_enter()
        accepted = []
        try:
            for fx, fy in pts:
                centre = (fx * w, fy * h)
                monitor.target_shown(*to_deg(*centre))
                show_target(centre, 700.0)
                accepted.append(bool(hold_target_while(centre,
                                                       lambda: monitor.cal_point(fx, fy))))
            result = monitor.cal_compute()
        finally:
            monitor.cal_leave()
            monitor.target_shown(0.0, 0.0)
        result["points"] = result.get("points", 0)
        result["points_accepted"] = accepted
        result["order"] = pts
        return result

    def validate_tracker() -> dict:
        """Titta's validation: points between the calibration points (a diamond, plus the
        centre: the + matters most here), random order; 0.7 s to settle, then 500 ms of
        gaze. Accuracy = mean offset (deg); precision = RMS sample-to-sample (deg);
        data loss = share of samples without gaze."""
        w, h = width_px, height_px
        pts = list(TRACKER_VAL_POINTS)
        rng.shuffle(pts)
        results = {}
        for fx, fy in pts:
            centre = (fx * w, fy * h)
            xd, yd = to_deg(*centre)
            monitor.target_shown(xd, yd)
            show_target(centre, 700.0)
            t0 = time.perf_counter()
            for _ in range(10):
                show_target(centre, 50.0)
                monitor.current()
            got = monitor.between(t0, time.perf_counter())
            valid = [g for g in got if g is not None]
            offs = [math.hypot(g[0] - xd, g[1] - yd) for g in valid]
            s2s = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(valid, valid[1:])]
            results[(fx, fy)] = dict(
                x_deg=round(xd, 2), y_deg=round(yd, 2),
                accuracy_deg=round(float(np.mean(offs)), 2) if offs else None,
                precision_rms_deg=(round(float(math.sqrt(np.mean(np.square(s2s)))), 3)
                                   if s2s else None),
                data_loss_pct=round(100.0 * (1 - len(valid) / len(got)), 1) if got else 100.0)
        monitor.target_shown(0.0, 0.0)
        points = [results[p] for p in TRACKER_VAL_POINTS]          # fixed order to report
        centre = points[0]
        accs = [p["accuracy_deg"] for p in points if p["accuracy_deg"] is not None]
        mean_acc = float(np.mean(accs)) if accs else float("nan")
        loss = float(np.mean([p["data_loss_pct"] for p in points]))
        passed = (len(accs) == len(points) and centre["accuracy_deg"] is not None
                  and centre["accuracy_deg"] <= settings.tracker_max_centre_error_deg
                  and mean_acc <= settings.tracker_max_mean_error_deg
                  and loss <= settings.tracker_max_data_loss_pct)
        precs = [p["precision_rms_deg"] for p in points if p["precision_rms_deg"] is not None]
        return dict(points=points, centre_accuracy_deg=centre["accuracy_deg"],
                    mean_accuracy_deg=round(mean_acc, 2),
                    precision_rms_deg=round(float(np.mean(precs)), 3) if precs else None,
                    data_loss_pct=round(loss, 1), passed=passed,
                    # kept under the earlier names too
                    centre_error_deg=centre["accuracy_deg"], mean_error_deg=round(mean_acc, 2))

    def validation_marks(val: dict) -> List[Mark]:
        marks = []
        for p in val["points"]:
            ok = p["accuracy_deg"] is not None and p["accuracy_deg"] <= \
                settings.tracker_max_mean_error_deg
            label = "no data" if p["accuracy_deg"] is None else f"{p['accuracy_deg']:.2f} deg"
            marks.append(Mark(to_px(p["x_deg"], p["y_deg"]), max(10.0, 0.4 * ppd_x), label,
                              "lightgreen" if ok else "orange"))
        return marks

    def calibrate_and_validate() -> None:
        other = "LEFT" if settings.eye == "R" else "RIGHT"
        attempt = 0
        do_cal = settings.tracker_calibrate and monitor.calibration_supported
        while True:
            attempt += 1
            if do_cal:
                key = backend.show_screen(
                    "Eye tracker calibration.\n\nKeep BOTH eyes open, chin on the chin rest.\n"
                    "A target will appear in 5 places. Look at its centre each time "
                    "until it moves.\n\n"
                    f"Press {seen_key.upper()} to start   ({abort_key.upper()} stops).",
                    keys=[seen_key, abort_key])
                if key == abort_key:
                    raise Aborted
                cal = calibrate_tracker()
                gaze_check["calibration"] = dict(cal, attempt=attempt)
                if not cal.get("ok"):
                    key = backend.show_screen(
                        f"Calibration did not work ({cal.get('status')}).\n\nCheck the chin "
                        "rest and that "
                        + ("the test eye is" if "only" in str(cal.get("mode", "")) else "both eyes are")
                        + " seen by the tracker.\n\n"
                        f"{seen_key.upper()} = go on anyway   R = calibrate again   "
                        f"{abort_key.upper()} = stop", keys=[seen_key, "r", abort_key])
                    if key == abort_key:
                        raise Aborted
                    if key == "r":
                        continue
            if not settings.tracker_validate:
                gaze_check["passed"] = True
                return
            while True:
                key = backend.show_screen(
                    f"Now cover your {other} eye.\n\nThe target will appear 5 more times. "
                    "Look at its centre each time.\n\n"
                    f"Press {seen_key.upper()} to start   ({abort_key.upper()} stops).",
                    keys=[seen_key, abort_key])
                if key == abort_key:
                    raise Aborted
                val = validate_tracker()
                gaze_check["validation"] = dict(val, attempt=attempt)
                gaze_check["passed"] = bool(val["passed"])
                gaze_check["offset_deg"] = val["centre_accuracy_deg"]
                verdict = "GOOD" if val["passed"] else "NOT GOOD ENOUGH"
                acc = (f"{val['mean_accuracy_deg']:.2f}" if val["mean_accuracy_deg"] ==
                       val["mean_accuracy_deg"] else "-")
                key = backend.show_screen(
                    f"Eye tracker accuracy: {verdict}\n"
                    f"accuracy {acc} deg (centre {val['centre_accuracy_deg']} deg)   "
                    f"precision {val['precision_rms_deg']} deg   "
                    f"data loss {val['data_loss_pct']:.0f} %\n"
                    f"limits: centre {settings.tracker_max_centre_error_deg:g}, average "
                    f"{settings.tracker_max_mean_error_deg:g} deg, loss "
                    f"{settings.tracker_max_data_loss_pct:g} %\n\n"
                    f"{seen_key.upper()} = start the test   V = check again   "
                    f"R = calibrate again   {abort_key.upper()} = stop",
                    marks=validation_marks(val), keys=[seen_key, "v", "r", abort_key])
                if key == abort_key:
                    raise Aborted
                if key == "v":
                    continue
                break
            if key == "r" and do_cal:
                continue
            return

    def distance_now():
        """The eye's distance from the screen now (mm, rounded), or '' when not measured."""
        if not watching:
            return ""
        d = monitor.eye_distance_mm(0.3)
        return "" if d is None else round(d, 1)

    def check_display_area() -> None:
        """Does the eye tracker know the screen at the size the test uses? (On the lab PC
        it was still set for an older 27" monitor.)"""
        try:
            da = monitor.display_area()
        except RuntimeError:
            da = None
        if not da:
            return
        gaze_check["tracker_display_area"] = da
        if hasattr(monitor, "set_screen_plane"):
            # From now on the eye's distance is measured square on to the screen, not to
            # the (tilted) tracker, which read ~4-5 cm more on the lab PC.
            monitor.set_screen_plane(da)
        dw = da["width_mm"] - settings.screen_width_mm
        dh = da["height_mm"] - settings.screen_height_mm if settings.screen_height_mm > 0 else 0
        tol = settings.display_area_tolerance_mm
        if abs(dw) <= tol and abs(dh) <= tol:
            return
        gaze_check["tracker_display_area_mismatch_mm"] = [round(dw, 1), round(dh, 1)]
        key = backend.show_screen(
            f"The eye tracker has the screen as {da['width_mm']:.1f} x {da['height_mm']:.1f} mm, "
            f"but these settings say {settings.screen_width_mm:.1f} x "
            f"{settings.screen_height_mm:.1f} mm.\nEvery gaze position will be off.\n\n"
            "Set the tracker's screen with:  python\\specvis.bat tracker-display\n\n"
            f"{seen_key.upper()} = go on anyway   {abort_key.upper()} = stop",
            keys=[seen_key, abort_key])
        if key == abort_key:
            raise Aborted

    def position_head() -> None:
        """Live head positioning (as Tobii Eye Tracker Manager and Titta show it): the eyes
        drawn where the tracker sees them, against a target circle, with the distance
        live; green when the distance is right. The operator presses SPACE when it is."""
        set_mm, tol = settings.viewing_distance_mm, settings.eye_distance_tolerance_mm
        cx, cy = width_px / 2.0, height_px / 2.0
        scale = 3.0                               # screen px per mm of head movement
        ring = max(60.0, 0.12 * height_px)
        eye_y0 = settings.tracker_eye_height_mm   # eye height above the tracker, centred
        # Lab PC, 15 Nov: Tobii's y axis follows the tracker's tilt (20 deg
        # here), so "170 mm above the tracker" is not "level with the screen centre". When
        # the tracker's display area is known, place each eye by its offset from the screen
        # centre measured along the screen's own right / up directions.
        plane = None
        da = gaze_check.get("tracker_display_area") or {}
        try:
            tl, tr_, bl = (np.array(da[k], dtype=float)
                           for k in ("top_left", "top_right", "bottom_left"))
            right_v, up_v = tr_ - tl, tl - bl
            right_v, up_v = right_v / np.linalg.norm(right_v), up_v / np.linalg.norm(up_v)
            centre = bl + (tr_ - tl) / 2.0 + (tl - bl) / 2.0
            plane = (centre, right_v, up_v)
        except (KeyError, TypeError, ValueError):
            plane = None

        def eye_px(o):
            if plane is None:
                return (cx + o[0] * scale, cy - (o[1] - eye_y0) * scale)
            v = np.array(o[:3], dtype=float) - plane[0]
            return (cx + float(v @ plane[1]) * scale, cy - float(v @ plane[2]) * scale)

        last_d = None
        last_good = False
        while True:
            head = monitor.head_position() or {}
            d = head.get("distance_mm")
            marks = [Mark((cx, cy), ring, "", "grey")]
            dist_ok = d is not None and abs(d - set_mm) <= tol
            # Lab report 15 Nov: green must also mean "both eyes inside the circle", and
            # (our addition) inside the tracker's track box, not at its edge.
            placed = []
            for side in ("left", "right"):
                o = head.get(side)
                if o is not None:
                    placed.append((side, eye_px(o), o))
            inside = bool(placed) and all(
                math.hypot(px - cx, py - cy) <= ring for _, (px, py), _ in placed)
            # Lab report 16 Nov, B2: where each eye is in the tracker's view (Tobii's user
            # position guide). No data is not "fine": the check could not run.
            boxes = [head.get(f"{side}_box") for side, _, _ in placed]
            box_known = bool(placed) and all(b is not None for b in boxes)
            in_box = box_known and all(BOX_EDGE <= c <= 1 - BOX_EDGE
                                       for b in boxes for c in b[:2])
            good = dist_ok and inside and in_box
            for side, pos, o in placed:
                r = max(8.0, 0.22 * ring * set_mm / max(o[2], 1.0))
                marks.append(Mark(pos, r, "L" if side == "left" else "R",
                                  "lightgreen" if good else "orange"))
            if d is None:
                line = "Eyes not found: sit with the chin on the chin rest, look at the screen."
            else:
                line = (f"Distance {d / 10:.1f} cm  (set {set_mm / 10:.1f} cm)  -  " +
                        ("good" if dist_ok else
                         f"move {'closer' if d > set_mm else 'back'} by "
                         f"{abs(d - set_mm) / 10:.1f} cm"))
                if placed and not inside:
                    mx = sum(p[1][0] for p in placed) / len(placed) - cx
                    my = sum(p[1][1] for p in placed) / len(placed) - cy
                    if abs(my) >= abs(mx):
                        line += ("\nEyes too high: lower the chin rest" if my < 0 else
                                 "\nEyes too low: raise the chin rest")
                    else:
                        line += ("\nEyes too far right: move the head to the left" if mx > 0
                                 else "\nEyes too far left: move the head to the right")
                elif placed and not box_known:
                    line += ("\nThe tracker does not say where the head is in its view: "
                             "restart the tracker bridge (python\\specvis.bat tracker-bridge)")
                elif not in_box:
                    line += "\nEyes near the edge of the tracker's view: centre the head"
                if good:
                    line += "\nPosition good"
            key = backend.show_status(
                "Position the head: both eyes inside the circle.\n" + line +
                f"\n\n{seen_key.upper()} = continue   {abort_key.upper()} = stop",
                marks, 100.0, keys=[seen_key, abort_key])
            last_d = d if d is not None else last_d
            last_good = good
            if key == abort_key:
                raise Aborted
            if key == seen_key:
                gaze_check["eye_distance_mm"] = None if last_d is None else round(last_d, 1)
                if last_d is not None and abs(last_d - set_mm) > tol:
                    gaze_check["eye_distance_accepted_off_by_mm"] = round(last_d - set_mm, 1)
                if not last_good:                  # the operator went on anyway
                    gaze_check["head_position_accepted_off"] = True
                return

    def check_eye_distance() -> None:
        """The eye distance, for trackers that report it but no head position."""
        d = None
        for _ in range(int(1000 / POLL_MS)):
            monitor.current()
            wait(POLL_MS)
            d = monitor.eye_distance_mm(0.5)
        gaze_check["eye_distance_mm"] = None if d is None else round(d, 1)
        if d is not None and abs(d - settings.viewing_distance_mm) > \
                settings.eye_distance_tolerance_mm:
            gaze_check["eye_distance_accepted_off_by_mm"] = round(
                d - settings.viewing_distance_mm, 1)

    def check_gaze_at_start() -> None:
        """Before the test: is the tracker's gaze on the + when the person looks at it?
        A large offset means the calibration is off."""
        while True:
            key = backend.show_screen(
                "Eye tracker check.\n\nLook at the  +  in the middle and keep looking "
                f"at it.\n\nPress {seen_key.upper()} when ready   ({abort_key.upper()} stops).",
                keys=[seen_key, abort_key])
            if key == abort_key:
                raise Aborted
            got = []
            for _ in range(int(1500 / POLL_MS)):
                got.append(monitor.current())
                wait(POLL_MS)
            valid = [g for g in got if g is not None]
            frac = len(valid) / len(got)
            offset = (float(np.median([math.hypot(*g) for g in valid])) if valid
                      else float("nan"))
            gaze_check.update(offset_deg=round(offset, 2) if valid else None,
                              valid_fraction=round(frac, 2))
            if frac >= 0.5 and offset <= settings.fixation_tolerance_deg / 2.0:
                gaze_check["passed"] = True
                return
            gaze_check["passed"] = False
            reason = ("the eye was not tracked" if frac < 0.5 else
                      f"gaze is {offset:.1f} deg away from the +")
            key = backend.show_screen(
                f"Eye tracker check FAILED: {reason}.\n\nRecalibrate in Tobii Pro Eye "
                "Tracker Manager (and check the chin rest and the eye patch).\n\n"
                f"Press {seen_key.upper()} to check again   ({abort_key.upper()} stops).",
                keys=[seen_key, abort_key])
            if key == abort_key:
                raise Aborted

    blink_repeats: Dict[Tuple[float, float], int] = {}
    MAX_BLINK_REPEATS = 3          # then a not-seen answer stands (a blind point never ends)

    def present(tex: Texture, where: Optional[Tuple[float, float]] = None
                ) -> Tuple[Optional[bool], float]:
        """One presentation. Returns (seen, reaction ms from stimulus onset); seen is
        None when the trial was thrown away because the eye left the +.

        A press sooner than ``early_press_ms`` after onset cannot be an answer to this
        patch (SITA counts it as a false positive): it is recorded, not taken as seen.
        """
        if settings.fixation_ms > 0:
            wait(settings.fixation_ms)
        if watching:
            wait_for_fixation()
        elif pending_settle[0] > 0:
            # No tracker: a plain + only lead-in. Taken before the wait, so a pause (P)
            # during it asks for a fresh lead-in instead of being wiped out.
            lead_in, pending_settle[0] = pending_settle[0], 0.0
            wait(lead_in)
        counts["presentations"] += 1
        t_on = time.perf_counter()
        monitor.current()
        report = backend.present_train([tex], 0, stim_frames, fixation=fixation,
                                       response_keys=[seen_key, abort_key])
        monitor.current()
        t_off = time.perf_counter()
        monitor.patch_shown(t_on, t_off)
        counts["dropped_frames"] += report.dropped_frames
        if any(k == abort_key for k, _ in report.keys):
            raise Aborted
        presses = [t for k, t in report.keys if k == seen_key]
        early = [t for t in presses if t < settings.early_press_ms]
        presses = [t for t in presses if t >= settings.early_press_ms]
        # Lab report 16 Nov, B1: the + also turns red while the person answers and between
        # patches when the eye leaves it (feedback only: the answer still counts, the eye
        # was on the + while the patch was shown, as with the Humphrey).
        live = LiveFixation(settings.fixation_tolerance_deg, settings.fixation_off_grace_ms,
                            settings.fixation_dropout_ms)

        def watch_frame() -> None:
            state = live.update(time.perf_counter(), monitor.current())
            off = state == LiveFixation.OFF
            if off:
                need_settle(settings.look_away_settle_ms)
            fixation_colour(settings.fixation_off_colour if off else settings.fixation_colour)

        # Lab report 16 Nov: the patch is judged as soon as its gaze samples are in (a few
        # tens of ms after it ends), not after the answer window. A clear eye movement
        # during the patch ends the window at once, so "Keep looking at the +" comes
        # straight away; the trial is thrown away and repeated later in any case.
        JUDGE_AFTER_S = 0.05
        early_break = [False, False]                 # [judged yet, broken]

        def answer_frame() -> Optional[bool]:
            watch_frame()
            if not early_break[0] and time.perf_counter() >= t_off + JUDGE_AFTER_S:
                early_break[0] = True
                ok, _, f = judge_patch(monitor.between(t_on, t_off),
                                       settings.fixation_tolerance_deg,
                                       max_out_frac=settings.fixation_break_max_out_frac)
                early_break[1] = (not ok) and f >= 0.5
            return True if early_break[1] else None

        on_frame = watch_frame if watching else None
        if not presses:
            key, latency = wait_key([seen_key, abort_key], response_window(),
                                    clear_events=False,
                                    on_frame=answer_frame if watching else None)
            if key == abort_key:
                raise Aborted
            if key == seen_key:
                presses.append(display_ms + latency)
        if early:
            counts["early_presses"] += 1
        seen = bool(presses)

        gaze_info.clear()
        fixated = True
        if watching:
            fixated, worst, frac = judge_patch(monitor.between(t_on, t_off),
                                               settings.fixation_tolerance_deg,
                                               max_out_frac=settings.fixation_break_max_out_frac)
            if early_break[1] and fixated:
                # The answer window was cut short for this break: a late sample must not
                # turn it back into a valid "not seen" the person had no time to answer.
                fixated, frac = False, max(frac, 0.5)
            counts["gaze_checked"] += 1
            gaze_info.update(gaze_max_deg="" if math.isnan(worst) else round(worst, 2),
                             gaze_valid_pct=round(100 * frac))
            # A blink: the patch is repeated, but the person is not told off for it.
            blink = not fixated and frac < 0.5
            gap_blink = (not blink and not seen and settings.blink_min_ms > 0
                         and where is not None
                         and blink_repeats.get(where, 0) < MAX_BLINK_REPEATS
                         and monitor.longest_gap_ms(t_on, t_off) >= settings.blink_min_ms)
            if gap_blink:
                blink, fixated = True, False          # the patch may have fallen in a blink
                blink_repeats[where] = blink_repeats.get(where, 0) + 1
            # Lab report B3: eye openness tells a real blink (lid closed) from lost
            # tracking with the lid open. Both repeat the patch; they are logged apart.
            lids = monitor.openness_between(t_on - 0.05, t_off + 0.05)
            valid_lids = [v for v in lids if v is not None]
            if valid_lids:
                gaze_info["eye_openness_min_mm"] = round(min(valid_lids), 2)
            tracking_lost = False
            if blink:
                typical = monitor.typical_openness()
                if valid_lids and typical and len(valid_lids) == len(lids) and \
                        min(valid_lids) >= settings.blink_openness_frac * typical:
                    tracking_lost = True              # the lid stayed open
            # The eye not seen for most of the patch is a blink, not a look away.
            counts["fixation_breaks"] += int(not fixated and frac >= 0.5 and not gap_blink)
            if tracking_lost:
                gaze_info["tracking_lost"] = 1
                counts["tracking_lost"] = counts.get("tracking_lost", 0) + 1
            elif blink:
                gaze_info["blink"] = 1
                counts["blinks"] = counts.get("blinks", 0) + 1
            if blink and settings.post_blink_delay_ms > 0:
                wait(settings.post_blink_delay_ms, on_frame)
            if not fixated and not blink and settings.fixation_feedback:
                # Right away, so the person knows; the next patch waits for the eye anyway.
                fixation_colour(settings.fixation_off_colour)
                backend.show_message("Keep looking at the  +", 1000.0, fixation=fixation)
                counts["fixation_feedback"] = counts.get("fixation_feedback", 0) + 1
            if not fixated and not blink:
                need_settle(settings.look_away_settle_ms)

        gap = rng.uniform(*settings.iti_ms) + (0.0 if seen else settings.miss_extra_pause_ms)
        wait(gap, on_frame)
        if watching:
            # B1 (b): where the eye went after the patch (answer time and the pause), for
            # analysis; whether it should void the answer is for the team (C3).
            after_max, after_ms = monitor.look_away(t_off, time.perf_counter(),
                                                    settings.fixation_tolerance_deg)
            gaze_info.update(gaze_max_after_deg="" if math.isnan(after_max)
                             else round(after_max, 2),
                             look_away_after_ms=round(after_ms))
            if after_ms > max(settings.fixation_off_grace_ms, 0.0):
                counts["look_aways_after_patch"] = counts.get("look_aways_after_patch", 0) + 1
        return (seen if fixated else None), (presses[0] if presses else float("nan"))

    def log(kind: str, point: Optional[PerimetryPoint], x: Optional[float],
            y: Optional[float], contrast: float, seen: bool, rt: float, note: str = "") -> None:
        measure = nominal_contrast if settings.dithering else measure_contrast
        rendered = (measure(base.with_contrast(contrast), calibration, ppd_x, ppd_y)
                    if contrast > 0 else 0.0)
        row = dict(trial=len(rows) + 1, ms_from_start=int((time.perf_counter() - t0) * 1000),
                   kind=kind, point=point.index if point else "",
                   x_deg="" if x is None else x, y_deg="" if y is None else y,
                   requested_contrast=round(contrast, 5), rendered_contrast=round(rendered, 5),
                   seen=int(seen), reaction_ms="" if math.isnan(rt) else round(rt, 1),
                   estimate_contrast=(round(10 ** point.zest.estimate_log, 5) if point else ""),
                   posterior_sd_log10=_sd(point),
                   gaze_max_deg=gaze_info.get("gaze_max_deg", ""),
                   eye_distance_mm=distance_now(),
                   gaze_valid_pct=gaze_info.get("gaze_valid_pct", ""),
                   gaze_max_after_deg=gaze_info.get("gaze_max_after_deg", ""),
                   look_away_after_ms=gaze_info.get("look_away_after_ms", ""),
                   eye_openness_min_mm=gaze_info.get("eye_openness_min_mm", ""),
                   note=note)
        rows.append(row)
        if on_trial is not None:
            on_trial(row)

    def neighbours(p: PerimetryPoint) -> List[PerimetryPoint]:
        """Finished points near p, in its own quadrant."""
        return [q for q in points
                if q is not p and q.quadrant == p.quadrant and q.zest.finished()
                and math.hypot(q.x_deg - p.x_deg, q.y_deg - p.y_deg) <= settings.neighbour_radius_deg]

    def start_full_threshold(p: PerimetryPoint) -> None:
        """The growth pattern: seeds start at ft_start (the middle of the range unless
        given); other points at the
        median of their finished neighbours' thresholds (blind neighbours are not
        used unless ft_blind_neighbour_start is on). No table."""
        if p.seed:
            p.prior_source = "fixed start (seed point)"
            return
        near = neighbours(p)
        measured = [q.neighbour_value_log() for q in near
                    if q.zest.result().termination is not Termination.CEILING_NOT_SEEN]
        measured = [v for v in measured if v is not None]
        blind = len(near) - len(measured)
        if measured:
            p.zest.set_start(10 ** float(np.median(measured)))
            p.expected_log = p.zest.prior_centre_log
            p.prior_source = f"growth pattern: {len(measured)} neighbours"
        elif blind and settings.ft_blind_neighbour_start:
            p.zest.set_start(ft_settings.max_contrast)
            p.expected_log = p.zest.prior_centre_log
            p.prior_source = f"maximum: {blind} blind neighbours (our rule)"
        else:
            p.prior_source = "fixed start (no measured neighbour)"

    def start_point(p: PerimetryPoint) -> None:
        """Give a point its starting guess, at its first presentation."""
        p.started = True
        if settings.strategy == FULL_THRESHOLD:
            start_full_threshold(p)
            return
        if settings.pilot_mode:
            p.prior_source = "none (pilot mode)"
            return
        guesses, sources = [], []
        entry = None
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
            p.expected_log = float(np.mean(guesses))
            if settings.strategy == SITA_FAST:
                p.zest.set_prior(p.expected_log, settings.fast_prior_sd_log,
                                 settings.fast_prior_flat_weight,
                                 settings.prior_abnormal_weight,
                                 sd_better_log=settings.fast_prior_sd_better_log,
                                 normal_log=entry)
            else:
                p.zest.set_prior(p.expected_log, settings.prior_sd_log,
                                 settings.prior_flat_weight)
            p.prior_source = " + ".join(sources)
        else:
            p.prior_source = "none"

    def needs_retest(p: PerimetryPoint) -> bool:
        """SITA's second staircase: a result more than ``retest_log`` from its guess."""
        if settings.strategy != SITA_FAST or p.expected_log is None or p.zest.retested:
            return False
        r = p.zest.result()
        return r.valid and abs(math.log10(r.value) - p.expected_log) > settings.retest_log

    search: Dict[str, object] = {}

    def search_blind_spot() -> None:
        """Find this person's blind spot, as the Humphrey does before its test.

        The average position first: missed ``blind_spot_search_repeats`` times, it is
        the blind spot (most people). Seen, all 17 nearby positions are tried (each until
        seen, or missed that many times) and the missed one nearest the middle of all the
        missed ones is used, so the patch sits well inside the blind spot, not on its
        edge. After every other search patch a strong patch at a test point keeps the
        person pressing, alternately on the blind spot's side and the other side: if the
        blind spot's side misses them all while the other side sees, that half is
        probably blind (hemianopia) and the blind-spot checks cannot show fixation.
        Not found: search again or use the average position (the operator chooses).
        """
        nonlocal blind_spot_xy
        temporal = [p for p in points if p.x_deg * settings.temporal_sign > 0]
        nasal = [p for p in points if p.x_deg * settings.temporal_sign < 0]
        candidates = blind_spot_candidates(settings)
        attempts = 0
        while True:
            attempts += 1
            shown = {"temporal": 0, "nasal": 0}
            seen_n = {"temporal": 0, "nasal": 0}
            flashes = 0
            side = "temporal"

            def check(which: str) -> None:
                pool = temporal if which == "temporal" else nasal
                if not pool:
                    return
                q = rng.choice(pool)
                seen, rt = present(texture(q.x_deg, q.y_deg, settings.high_contrast))
                shown[which] += 1
                seen_n[which] += int(bool(seen))
                log(SEARCH_CHECK, None, q.x_deg, q.y_deg, settings.high_contrast,
                    bool(seen), rt, f"{which} side")

            def missed_every_time(x: float, y: float) -> bool:
                nonlocal flashes, side
                for _ in range(settings.blind_spot_search_repeats):
                    seen, rt = present(texture(x, y, settings.blind_spot_contrast))
                    flashes += 1
                    log(BLIND_SPOT_SEARCH, None, x, y, settings.blind_spot_contrast,
                        bool(seen), rt, f"attempt {attempts}")
                    if flashes % 2 == 1:          # a strong patch after every other one
                        check(side)
                        side = "nasal" if side == "temporal" else "temporal"
                    if seen:
                        return False
                return True

            found = None
            if missed_every_time(*candidates[0]):         # the average: most people
                found = candidates[0]
            else:
                missed = [c for c in candidates[1:] if missed_every_time(*c)]
                if missed:                                # the middle of the missed ones
                    mx = sum(x for x, _ in missed) / len(missed)
                    my = sum(y for _, y in missed) / len(missed)
                    found = min(missed, key=lambda c: (math.hypot(c[0] - mx, c[1] - my), c))
                    search["positions_missed"] = [list(c) for c in missed]
            while temporal and shown["temporal"] < 2:
                check("temporal")
            while nasal and shown["nasal"] < 1:
                check("nasal")
            search.update(attempts=attempts, search_patches=flashes,
                          strong_patches={k: f"{seen_n[k]}/{shown[k]} seen" for k in shown})
            blind_side = (shown["temporal"] >= 2 and seen_n["temporal"] == 0
                          and seen_n["nasal"] >= 1)
            search["blind_spot_side_not_seen"] = blind_side
            if found is not None:
                blind_spot_xy = found
                search.update(status="found", found_xy=list(found))
                break
            if attempts >= settings.blind_spot_search_attempts:
                key = seen_key
            else:
                key = backend.show_screen(
                    "Blind spot NOT found: every position tried was seen.\n\n"
                    "Check that the other eye is FULLY covered, the chin rest, and that "
                    "the participant keeps looking at the  +.\n\n"
                    f"R = search again     {seen_key.upper()} = go on with the average "
                    f"position   ({abort_key.upper()} stops)",
                    keys=[seen_key, "r", abort_key])
            if key == abort_key:
                raise Aborted
            if key != "r":
                search.update(status="not found: used the average position",
                              found_xy=None)
                break
        if search.get("blind_spot_side_not_seen"):
            key = backend.show_screen(
                "Note for the operator: strong patches on the blind spot's side were all "
                "missed, so that half of the field may be blind.\n\n"
                "Then the blind-spot checks cannot show whether the eye stays on the +. "
                "Use the eye tracker for this eye if possible.\n\n"
                f"Press {seen_key.upper()} to go on   ({abort_key.upper()} stops).",
                keys=[seen_key, abort_key])
            if key == abort_key:
                raise Aborted

    monitor_started = False
    try:
        if watching:
            try:
                monitor.begin(backend, to_deg)
            except RuntimeError as e:
                raise SessionError(f"Fixation monitor ({settings.fixation_monitoring}): {e}")
            monitor_started = True
        if watching and show_instructions:
            check_display_area()
            if settings.check_eye_distance:
                if monitor.head_position() is not None:
                    position_head()
                else:
                    check_eye_distance()
            if monitor.calibration_supported and (settings.tracker_calibrate
                                                  or settings.tracker_validate):
                calibrate_and_validate()
            else:
                check_gaze_at_start()
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
            need_settle(settings.resume_delay_ms)

        for _ in range(settings.practice_trials):
            p = rng.choice(seeds or points)
            seen, rt = present(texture(p.x_deg, p.y_deg, settings.high_contrast))
            log(PRACTICE, None, p.x_deg, p.y_deg, settings.high_contrast, bool(seen), rt,
                FIXATION_BREAK if seen is None else "")

        if settings.fixation_monitoring == FIX_BLIND_SPOT and p_blind_spot > 0:
            if settings.blind_spot_search:
                search_blind_spot()
            else:
                search.update(status="off: used the average position")

        last = None
        while True:
            pending = [p for p in points if not p.zest.finished()]
            if not pending:
                break
            pending_seeds = [p for p in seeds if not p.zest.finished()]
            if pending_seeds:
                pending = pending_seeds
            elif settings.strategy == FULL_THRESHOLD:
                # Growth pattern: grow outwards from finished points -- start a point
                # only once a neighbour is finished (points already started go on).
                ready = [p for p in pending if p.started or neighbours(p)]
                pending = ready or pending

            r = rng.random()
            high_candidates = [p for p in points if p.zest.finished() and p.zest.result().valid
                               and p.zest.result().value <= settings.high_catch_max_threshold]
            if r < p_blank:
                seen, rt = present(Texture(blank, centre))
                if seen is not None:
                    counts["blank"] += 1
                    counts["false_alarms"] += int(seen)
                log(BLANK, None, None, None, 0.0, bool(seen), rt,
                    FIXATION_BREAK if seen is None else ("FALSE_ALARM" if seen else ""))
            elif r < p_blank + p_high and high_candidates:
                p = rng.choice(high_candidates)
                seen, rt = present(texture(p.x_deg, p.y_deg, settings.high_contrast))
                if seen is not None:
                    counts["high"] += 1
                    counts["misses"] += int(not seen)
                log(HIGH, None, p.x_deg, p.y_deg, settings.high_contrast, bool(seen), rt,
                    FIXATION_BREAK if seen is None else ("" if seen else "MISS"))
            elif p_blind_spot > 0 and r < p_blank + p_high + p_blind_spot:
                bx, by = blind_spot_xy
                seen, rt = present(texture(bx, by, settings.blind_spot_contrast))
                if seen is not None:
                    counts["blind_spot"] += 1
                    counts["blind_spot_seen"] += int(seen)
                log(BLIND_SPOT, None, bx, by, settings.blind_spot_contrast, bool(seen), rt,
                    FIXATION_BREAK if seen is None else ("FIXATION_LOSS" if seen else ""))
            else:
                choices = [p for p in pending if p is not last] or pending
                p = rng.choice(choices)
                last = p
                if not p.started:
                    start_point(p)
                kind = (PROBE_TRIAL if p.zest.probing else
                        STAIRCASE_TRIAL if settings.strategy == FULL_THRESHOLD else ZEST_TRIAL)
                contrast = p.zest.next_level()
                seen, rt = present(texture(p.x_deg, p.y_deg, contrast), (p.x_deg, p.y_deg))
                if seen is None:
                    # The eye left the + (or blinked): the answer says nothing about this point.
                    log(kind, p, p.x_deg, p.y_deg, contrast, False, rt,
                        ("BLINK" if gaze_info.get("blink") else
                         "TRACKING_LOST" if gaze_info.get("tracking_lost") else FIXATION_BREAK)
                        + " (tested again later)")
                else:
                    if seen and not math.isnan(rt):
                        reaction_times.append(rt)
                    p.zest.record(seen)
                    note = p.zest.result().termination.value if p.zest.finished() else ""
                    if p.zest.finished() and needs_retest(p):
                        p.zest.reopen()
                        counts["retests"] += 1
                        note = "RETEST (far from expected)"
                    log(kind, p, p.x_deg, p.y_deg, contrast, seen, rt, note)

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
                need_settle(settings.resume_delay_ms)
    except Aborted:
        aborted = True
    finally:
        if monitor_started:
            monitor.end()
            # Lab report B6: tracker notifications, and how the tracker's clock was put on
            # ours (for lining gaze up with patches, and later with the EEG).
            if getattr(monitor, "events", None):
                gaze_check["tracker_events"] = list(monitor.events)
            clock = getattr(monitor, "clock", None)
            if clock is not None and clock.offset_s is not None:
                gaze_check["tracker_clock"] = dict(
                    rule="sample time = tracker system time stamp + the smallest "
                         "(arrival - tracker time) over the last 1000 samples",
                    offset_s=round(clock.offset_s, 6))

    dists = [r["eye_distance_mm"] for r in rows if r.get("eye_distance_mm") not in ("", None)]
    if dists:
        gaze_check["eye_distance_during_test_mm"] = dict(
            mean=round(float(np.mean(dists)), 1), min=round(float(min(dists)), 1),
            max=round(float(max(dists)), 1), trials=len(dists),
            set_mm=settings.viewing_distance_mm)

    # SITA's post-processing: every estimate again, with the false-answer rates this
    # participant actually showed (blank presses + too-early presses; strong misses).
    post: Dict[str, float] = {}
    if settings.post_process and settings.strategy == SITA_FAST:
        fp_opportunities = counts["blank"] + counts["presentations"]
        fp = ((counts["false_alarms"] + counts["early_presses"]) / fp_opportunities
              if fp_opportunities else settings.zest.guess)
        fn = counts["misses"] / counts["high"] if counts["high"] else settings.zest.lapse
        guess = max(settings.zest.guess, min(fp, 0.3))
        lapse = max(settings.zest.lapse, min(fn, 0.3))
        for p in points:
            if p.zest.trials:
                p.zest.recompute(guess, lapse)
        post = dict(guess=round(guess, 4), lapse=round(lapse, 4))

    return PerimetryOutcome(
        settings=settings, points=points, dropped_points=dropped, trial_rows=rows,
        counts=counts, screen=screen, measured_refresh_hz=refresh,
        calibration_source=calibration.source, calibrated=calibration.calibrated,
        prior_table_source=(use_table.source if use_table is not None else
                            ("not used (pilot mode)" if settings.pilot_mode else "none")),
        aborted=aborted, started=started, duration_s=time.perf_counter() - t0,
        point_px={p.index: to_px(p.x_deg, p.y_deg) for p in points},
        patch_radius_px=radius_px, background_cd_m2=mean_luminance, post_processing=post,
        fixation_monitor=(monitor.describe() if watching else settings.fixation_monitoring),
        gaze_check=gaze_check,
        blind_spot_xy=(blind_spot_xy if settings.fixation_monitoring == FIX_BLIND_SPOT
                       else None),
        blind_spot_search=search,
        contrast_range=db_range,
        gaze_samples=([(round(t - t0, 4), None if g is None else round(g[0], 3),
                        None if g is None else round(g[1], 3)) for t, g in monitor.samples]
                      if watching else []))


# ------------------------------------------------------------------------------
# Results
# ------------------------------------------------------------------------------

def _sd(point: Optional[PerimetryPoint]):
    """ZEST's posterior SD for trials.csv / summary.csv; blank for a staircase."""
    if point is None or math.isnan(point.zest.posterior_sd):
        return ""
    return round(point.zest.posterior_sd, 4)


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
                  extent=s.extent, session_valid=outcome.valid,
                  method=STRATEGY_NAMES.get(s.strategy, ZEST))
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
            posterior_sd_log10=_sd(p) if p.zest.trials else "",
            trials=r.trials, termination=r.termination.value, note=r.note,
            seed_point=p.seed, retested=p.zest.retested,
            early_stop=p.zest.early_stop,
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
    "fixation": "The eye left the + too often (or blind-spot patches were seen). Check "
                "the other eye is FULLY covered, the chin rest, and the eye tracker "
                "calibration; remind the participant to keep looking at the +.",
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
        + (f"Eye left the + {rate(outcome.fixation_loss_rate)}" if s.watches_gaze else
         f"Blind-spot seen {rate(outcome.fixation_loss_rate)}") + f" [{tick[c['fixation']]}]",
    ]
    bs = outcome.blind_spot_search
    if bs.get("status") == "found":
        x, y = bs["found_xy"]
        lines.append(f"Blind spot found at {abs(x):g} deg {'right' if x > 0 else 'left'}, "
                     f"{abs(y):g} deg {'below' if y < 0 else 'above'} the +")
    elif bs.get("status"):
        lines.append("Blind spot NOT found: checks used the average position")
    if bs.get("blind_spot_side_not_seen"):
        lines.append("Blind spot's side looked blind: the blind-spot checks cannot show "
                     "fixation for this eye")
    if outcome.valid:
        lines.append("SESSION VALID")
    else:
        lines.append("SESSION NOT VALID - repeat this eye")
        lines += [FAILURE_REASONS[k] for k, passed in c.items() if not passed]
    lines.append("Green = seen   yellow = weak   red = blind   (number = threshold: the contrast "
                 "seen about 6 times in 10)")
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


def _meta(outcome: PerimetryOutcome) -> dict:
    """What session.json holds about the run (also what the maps' headings use)."""
    s = outcome.settings
    return dict(
        kind="perimetry", started=outcome.started, duration_s=round(outcome.duration_s, 1),
        aborted=outcome.aborted, session_valid=outcome.valid, checks=outcome.checks,
        false_alarm_rate=outcome.false_alarm_rate, miss_rate=outcome.miss_rate,
        fixation_loss_rate=outcome.fixation_loss_rate, counts=outcome.counts,
        fixation_monitor=outcome.fixation_monitor, gaze_check=outcome.gaze_check,
        measured_refresh_hz=outcome.measured_refresh_hz,
        screen=dict(resolution=[outcome.screen.resolution_x, outcome.screen.resolution_y],
                    width_mm=outcome.screen.width_mm, height_mm=outcome.screen.height_mm,
                    viewing_distance_mm=outcome.screen.viewing_distance_mm,
                    field_degrees=[outcome.screen.field_degrees_x,
                                   outcome.screen.field_degrees_y]),
        calibration=dict(source=outcome.calibration_source, calibrated=outcome.calibrated),
        strategy=STRATEGY_NAMES.get(s.strategy, s.strategy),
        background_cd_m2=round(outcome.background_cd_m2, 3),
        dithering=s.dithering,
        early_press_rate=outcome.early_press_rate,
        post_processing=outcome.post_processing,
        prior_table=outcome.prior_table_source,
        blind_spot_grid_points_dropped=outcome.dropped_points,
        blind_spot=dict(used_xy=(list(outcome.blind_spot_xy) if outcome.blind_spot_xy
                                 else None),
                        average_xy=list(blind_spot_position(s)),
                        search=outcome.blind_spot_search),
        grid=s.extent,
        contrast_range=outcome.contrast_range,
        honest_limits=[
            "Timing is software self-report, not photodiode-validated.",
            "Contrast is nominal unless the luminance scale was photometer-measured.",
        ],
        settings=dataclasses.asdict(s),
    )


def draw_map(outcome: PerimetryOutcome, path: pathlib.Path,
             table: Optional[PriorTable] = None) -> pathlib.Path:
    """The grid map as a picture (see ``maps.draw_grid_map``)."""
    from specvis.procedure import maps
    rows = summary_rows(outcome, table)
    meta = json.loads(json.dumps(_meta(outcome), default=str))
    return maps.draw_grid_map(rows, path, meta=meta)


def write_outputs(outcome: PerimetryOutcome, directory: pathlib.Path,
                  table: Optional[PriorTable] = None) -> pathlib.Path:
    """trials.csv, summary.csv, session.json, map.png (grid) and contour_map.png, and
    gaze.csv when gaze was watched. Returns the folder."""
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
    if outcome.gaze_samples:
        write_csv("gaze.csv", [dict(seconds=t, x_deg="" if x is None else x,
                                    y_deg="" if y is None else y)
                               for t, x, y in outcome.gaze_samples])
    write_csv("summary.csv", summary_rows(outcome, table))
    (directory / "session.json").write_text(
        json.dumps(_meta(outcome), indent=2, default=str) + "\n")
    from specvis.procedure import maps
    maps.save_session_maps(directory)          # map.png and contour_map.png
    return directory
