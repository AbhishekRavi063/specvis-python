"""The 4-2 dB staircase, and the dB-axis machinery it sits on.

Ported from ``org.refrem.staircase.AbstractDbStaircase`` and ``FourTwoStaircase``.

Unlike the bisection in :mod:`specvis.threshold.bisection`, this method searches
**continuous contrast**, not a fixed ladder of rungs. It has a stated estimator (the
mean of the contrasts at which reversals occurred), a stated stopping rule, and -- the
reason it exists -- it detects both ends of the range instead of only one.

Named for its step sizes: 4 dB until the first reversal, 2 dB after.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import List, Optional

from .base import LocationThreshold, Result, Termination

#: Contrast is expressed as attenuation from a reference of 1.0, so dB is positive
#: and larger means fainter.
REFERENCE_CONTRAST = 1.0
MIN_REPRESENTABLE_CONTRAST = 1e-6
BOUNDARY_EPSILON_DB = 1e-9


def contrast_to_db(michelson: float) -> float:
    if michelson <= MIN_REPRESENTABLE_CONTRAST:
        raise ValueError(f"contrast must be > {MIN_REPRESENTABLE_CONTRAST}, got {michelson}")
    return -20.0 * math.log10(michelson / REFERENCE_CONTRAST)


def db_to_contrast(db: float) -> float:
    return REFERENCE_CONTRAST * (10.0 ** (-db / 20.0))


class Estimator(enum.Enum):
    """How the final number is taken from the run."""

    REVERSAL_MEAN = "mean of the contrasts at which reversals occurred"
    LAST_SEEN = "the faintest contrast reported seen"


@dataclass
class Step:
    db: float
    seen: bool


class DbStaircase:
    """A staircase on a dB contrast axis. Subclasses choose the step schedule.

    Direction convention: a "seen" response moves **up** in dB, i.e. to a fainter
    stimulus. ``min_db`` is therefore the *highest* contrast and ``max_db`` the lowest,
    which reads backwards until you remember the axis is attenuation.
    """

    def __init__(self, start_contrast: float, max_contrast: float, min_contrast: float,
                 required_reversals: int, max_trials: int, min_trials: int,
                 estimator: Estimator = Estimator.REVERSAL_MEAN):
        if not min_contrast < max_contrast:
            raise ValueError("min contrast must be below max contrast")
        if not min_contrast <= start_contrast <= max_contrast:
            raise ValueError(
                f"start contrast {start_contrast} outside range "
                f"[{min_contrast}, {max_contrast}]"
            )

        self.min_db = contrast_to_db(max_contrast)   # highest contrast
        self.max_db = contrast_to_db(min_contrast)   # lowest contrast
        self._current_db = contrast_to_db(start_contrast)

        self._required_reversals = required_reversals
        self._max_trials = max_trials
        self._min_trials = min_trials
        self._estimator = estimator

        self._trials = 0
        self._reversals = 0
        self._previous_seen: Optional[bool] = None
        self._direction = 0
        self._last_seen_db: Optional[float] = None
        self._finished = False
        self._termination = Termination.NOT_FINISHED
        self._note = ""
        self.history: List[Step] = []
        self._reversal_dbs: List[float] = []

    # -- subclass hook ---------------------------------------------------------

    def step_db(self, reversals_so_far: int) -> float:
        raise NotImplementedError

    # -- driving ---------------------------------------------------------------

    @property
    def current_contrast(self) -> float:
        return db_to_contrast(self._current_db)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def trials(self) -> int:
        return self._trials

    @property
    def reversals(self) -> int:
        return self._reversals

    def record(self, seen: bool) -> None:
        if self._finished:
            raise RuntimeError("staircase already finished")

        self._trials += 1
        self.history.append(Step(self._current_db, seen))
        if seen:
            self._last_seen_db = self._current_db

        new_direction = +1 if seen else -1
        if self._previous_seen is not None and new_direction != self._direction:
            self._reversals += 1
            self._reversal_dbs.append(self._current_db)
        self._direction = new_direction
        self._previous_seen = seen

        # The reversal criterion alone is not enough. A lapse on the first trial can
        # create a reversal on the second, and a two-trial "threshold" resting on one
        # miss and one hit is not a measurement. min_trials is the floor.
        if self._reversals >= self._required_reversals and self._trials >= self._min_trials:
            self._stop(Termination.CONVERGED, "")
            return
        if self._trials >= self._max_trials:
            self._stop(Termination.BUDGET_EXHAUSTED,
                       "trial budget exhausted before reversal criterion")
            return

        next_db = self._current_db + new_direction * self.step_db(self._reversals)

        # Boundary handling. The boundary is *presented* rather than skipped, and the
        # run ends as a failure only when the answer AT the boundary shows the
        # threshold cannot be bracketed inside the range. Ending early instead made
        # about 4 dB of the operator's range unusable and recorded a note claiming the
        # maximum had been shown when it had not.
        at_highest_contrast = self._current_db <= self.min_db + BOUNDARY_EPSILON_DB
        at_lowest_contrast = self._current_db >= self.max_db - BOUNDARY_EPSILON_DB

        if next_db < self.min_db:
            if at_highest_contrast and not seen:
                self._stop(Termination.CEILING_NOT_SEEN,
                           "not seen at the maximum available contrast, which was "
                           "presented; no threshold within range")
                return
            next_db = self.min_db

        if next_db > self.max_db:
            if at_lowest_contrast and seen:
                self._stop(Termination.FLOOR_ALWAYS_SEEN,
                           "still seen at the minimum presented contrast, which was "
                           "presented; threshold below the measurable floor")
                return
            next_db = self.max_db

        self._current_db = next_db

    def _stop(self, termination: Termination, note: str) -> None:
        self._finished = True
        self._termination = termination
        self._note = note

    # -- reporting -------------------------------------------------------------

    def _has_unseen(self) -> bool:
        return any(not s.seen for s in self.history)

    def _mean_reversal_db(self) -> float:
        if not self._reversal_dbs:
            return math.nan
        return sum(self._reversal_dbs) / len(self._reversal_dbs)

    def outcome(self) -> Result:
        if not self._finished:
            return Result.unusable(Termination.NOT_FINISHED, self._trials, "still running")

        bracketed = self._termination is Termination.CONVERGED
        have_seen = self._last_seen_db is not None

        # An estimate is only meaningful if the run actually bracketed the transition:
        # at least one seen and one unseen response, and the reversal criterion met.
        valid = bracketed and have_seen and self._has_unseen()

        reason = self._note
        if not valid and not reason:
            if not have_seen:
                reason = "stimulus never seen; no threshold estimate"
            elif not self._has_unseen():
                reason = "stimulus always seen; no threshold estimate"

        if not have_seen:
            return Result.unusable(self._termination, self._trials, reason)

        reversal_mean = self._mean_reversal_db()
        db = (reversal_mean
              if self._estimator is Estimator.REVERSAL_MEAN and not math.isnan(reversal_mean)
              else self._last_seen_db)

        if not valid:
            return Result.unusable(self._termination, self._trials, reason)
        return Result(True, db_to_contrast(db), self._termination, self._trials, reason)


class FourTwoStaircase(DbStaircase):
    """4 dB steps until the first reversal, 2 dB after it."""

    DEFAULT_COARSE_STEP_DB = 4.0
    DEFAULT_FINE_STEP_DB = 2.0
    DEFAULT_REQUIRED_REVERSALS = 2
    DEFAULT_MAX_TRIALS = 20
    DEFAULT_MIN_TRIALS = 4

    def __init__(self, start_contrast: float, max_contrast: float, min_contrast: float,
                 coarse_step_db: float = DEFAULT_COARSE_STEP_DB,
                 fine_step_db: float = DEFAULT_FINE_STEP_DB,
                 required_reversals: int = DEFAULT_REQUIRED_REVERSALS,
                 max_trials: int = DEFAULT_MAX_TRIALS,
                 min_trials: int = DEFAULT_MIN_TRIALS,
                 estimator: Estimator = Estimator.REVERSAL_MEAN):
        super().__init__(start_contrast, max_contrast, min_contrast,
                         required_reversals, max_trials, min_trials, estimator)
        self._coarse = coarse_step_db
        self._fine = fine_step_db

    def step_db(self, reversals_so_far: int) -> float:
        return self._coarse if reversals_so_far == 0 else self._fine


class ContrastLocationThreshold(LocationThreshold):
    """A contrast staircase wearing the common interface.

    Validity is stricter than the staircase's own: the run must have converged **and**
    never have been clamped to the display's renderable floor. A contrast below the
    floor renders as a uniform grey field, which would be logged as a stimulus that
    was never actually shown.
    """

    def __init__(self, staircase: DbStaircase, name: str,
                 renderable_floor: float = 0.0):
        self._s = staircase
        self._name = name
        self._floor = renderable_floor
        self._clamped_to_floor = 0

    def next_level(self) -> float:
        contrast = self._s.current_contrast
        if self._floor > 0 and contrast < self._floor:
            self._clamped_to_floor += 1
            return self._floor
        return contrast

    def record(self, seen: bool) -> None:
        self._s.record(seen)

    def finished(self) -> bool:
        return self._s.finished

    def result(self) -> Result:
        outcome = self._s.outcome()
        if outcome.valid and self._clamped_to_floor:
            return Result.unusable(
                outcome.termination, outcome.trials,
                f"{self._clamped_to_floor} presentation(s) were clamped to the display's "
                f"renderable floor of {self._floor:.5f}; the requested contrast was never shown",
            )
        return outcome

    @property
    def trials(self) -> int:
        return self._s.trials

    @property
    def method_name(self) -> str:
        return self._name
