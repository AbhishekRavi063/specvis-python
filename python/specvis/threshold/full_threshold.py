"""Full Threshold: the Humphrey 4-2 staircase for one visual-field location.

The rule is the Open Perimetry Interface's ``FT`` (Turpin, Artes & McKendrick 2012,
*J Vision* 12(11); OPI ``R/full_threshold.r``), which reproduces the first Humphrey
Field Analyzer (Bebie, Fankhauser & Spahr 1976; Johnson, Chauhan & Shapiro 1993):

1. Start at the given level. Seen -> fainter; not seen -> stronger.
2. Steps: ``coarse_step_db`` until the first reversal, ``fine_step_db`` after it.
3. Stop after **2 reversals**; the threshold is the **last level seen**.
4. Stop early: **not seen twice at the maximum** -> blind (CEILING_NOT_SEEN);
   **seen twice at the minimum** -> better than the range (FLOOR_ALWAYS_SEEN).
5. **Second staircase**: if the first stopped by reversals and its result is more
   than ``retest_db`` from the start, run a second staircase starting at that result;
   its result is the final one.

**Units.** Here dB is *contrast* dB, ``-20 log10(contrast)`` (larger = fainter), so one
Humphrey dB of light (``-10 log10``) is about **two** of ours. The defaults are the
Humphrey's sizes in our units: steps 8 then 4 (Humphrey 4 then 2), second staircase
when more than 8 from the start (Humphrey 4).

No table of normal values is needed: the start comes from the growth pattern (the
perimetry procedure supplies it from finished neighbours, or a fixed start).

The class answers the same questions the perimetry procedure asks of
:class:`specvis.threshold.zest.ZestLocationThreshold`, so the procedure can use either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import LocationThreshold, Result, Termination

FULL_THRESHOLD = "Full Threshold (4-2)"


def to_db(contrast: float) -> float:
    return -20.0 * math.log10(contrast)


def from_db(db: float) -> float:
    return 10.0 ** (-db / 20.0)


@dataclass(frozen=True)
class FullThresholdSettings:
    min_contrast: float = 0.001
    max_contrast: float = 1.0
    coarse_step_db: float = 8.0         # Humphrey 4 dB of light
    fine_step_db: float = 4.0           # Humphrey 2 dB of light
    retest_db: float = 8.0              # Humphrey 4 dB of light
    max_presentations: int = 30         # safety net only; OPI has none

    def validate(self) -> None:
        if not 0 < self.min_contrast < self.max_contrast <= 1.0:
            raise ValueError("need 0 < min contrast < max contrast <= 1")
        if self.coarse_step_db <= 0 or self.fine_step_db <= 0 or self.retest_db < 0:
            raise ValueError("steps must be positive and the retest distance >= 0")
        if self.max_presentations < 4:
            raise ValueError("allow at least 4 presentations per point")


class FullThresholdLocation(LocationThreshold):
    """One location, one or two 4-2 staircases."""

    def __init__(self, settings: Optional[FullThresholdSettings] = None,
                 start_contrast: float = 0.05, renderable_floor: float = 0.0):
        s = settings or FullThresholdSettings()
        s.validate()
        self.s = s
        lo = max(s.min_contrast, renderable_floor) if renderable_floor > 0 else s.min_contrast
        self.lo_db = to_db(s.max_contrast)      # strongest patch (smallest dB)
        self.hi_db = to_db(lo)                  # faintest patch the range allows
        # What ZEST exposes as its grid ends, for the procedure's neighbour rule.
        self.grid = np.array([math.log10(lo), math.log10(s.max_contrast)])
        self._trials = 0
        self._stop: Optional[Termination] = None
        self.retested = False
        self.history: List[tuple] = []          # (contrast dB, seen, staircase 1 or 2)
        self.set_start(start_contrast)

    # -- the starting level (growth pattern) -------------------------------------------

    def set_start(self, contrast: float) -> None:
        """Where the first staircase starts. Only before the first presentation."""
        if self._trials:
            raise RuntimeError("the start can only be set before the first presentation")
        self.start_db = self._clip(to_db(min(max(contrast, 1e-6), 1.0)))
        self.prior_centre_log = -self.start_db / 20.0
        self._begin(self.start_db)

    def _begin(self, start_db: float) -> None:
        self._level_db = start_db
        self._reversals = 0
        self._previous: Optional[bool] = None
        self._last_seen_db: Optional[float] = None
        self._seen_at_min = 0     # seen at the faintest level
        self._missed_at_max = 0   # not seen at the strongest level

    def _clip(self, db: float) -> float:
        return min(self.hi_db, max(self.lo_db, db))

    # -- what the procedure asks ----------------------------------------------------------

    def next_level(self) -> float:
        return from_db(self._level_db)

    def record(self, seen: bool) -> None:
        if self.finished():
            raise RuntimeError("location already finished")
        self._trials += 1
        level = self._level_db
        self.history.append((round(level, 3), bool(seen), 2 if self.retested else 1))
        if self._previous is not None and seen != self._previous:
            self._reversals += 1
        self._previous = seen
        if seen:
            self._last_seen_db = level
            if level >= self.hi_db - 1e-9:
                self._seen_at_min += 1
        elif level <= self.lo_db + 1e-9:
            self._missed_at_max += 1

        if self._missed_at_max >= 2:
            self._stop = Termination.CEILING_NOT_SEEN
        elif self._seen_at_min >= 2:
            self._stop = Termination.FLOOR_ALWAYS_SEEN
        elif self._reversals >= 2:
            self._stop = Termination.CONVERGED
        elif self._trials >= self.s.max_presentations:
            self._stop = Termination.BUDGET_EXHAUSTED
        if self._stop is not None:
            self._maybe_second_staircase()
            return
        step = self.s.coarse_step_db if self._reversals == 0 else self.s.fine_step_db
        self._level_db = self._clip(level + (step if seen else -step))

    def _maybe_second_staircase(self) -> None:
        """OPI: a result more than ``retest_db`` from the start gets a second staircase,
        starting at that result."""
        if (self.retested or self._stop is not Termination.CONVERGED
                or self._last_seen_db is None
                or abs(self._last_seen_db - self.start_db) <= self.s.retest_db):
            return
        self.retested = True
        self._stop = None
        self.first_result_db = self._last_seen_db
        self._begin(self._last_seen_db)

    def finished(self) -> bool:
        return self._stop is not None

    def result(self) -> Result:
        if self._stop is None:
            return Result.unusable(Termination.NOT_FINISHED, self._trials, "still running")
        if self._stop is Termination.CEILING_NOT_SEEN:
            return Result.unusable(self._stop, self._trials,
                                   "not seen twice at the maximum contrast")
        if self._stop is Termination.FLOOR_ALWAYS_SEEN:
            return Result.unusable(self._stop, self._trials,
                                   "seen twice at the minimum contrast: better than the range")
        if self._stop is Termination.BUDGET_EXHAUSTED or self._last_seen_db is None:
            return Result.unusable(self._stop, self._trials,
                                   "presentation limit reached before 2 reversals")
        note = "second staircase" if self.retested else ""
        return Result(True, from_db(self._last_seen_db), self._stop, self._trials, note)

    @property
    def trials(self) -> int:
        return self._trials

    @property
    def method_name(self) -> str:
        return FULL_THRESHOLD

    # -- what the procedure reads from ZEST, given staircase meanings ---------------------

    @property
    def estimate_log(self) -> float:
        """The best value so far: the last level seen, else the current level."""
        db = self._last_seen_db if self._last_seen_db is not None else self._level_db
        return -db / 20.0

    posterior_sd = float("nan")       # a staircase has no posterior
    probing = False
    early_stop = False
    blind_by_quick_rule = False

    def recompute(self, guess: float, lapse: float) -> None:
        """SITA's post-processing does not apply to a staircase."""

    def reopen(self) -> None:
        raise RuntimeError("Full Threshold runs its own second staircase")
