"""ZEST: a Bayesian threshold search for one visual-field location.

ZEST (King-Smith et al. 1994, *Vision Research* 34:885) keeps a probability
distribution over where the threshold might be. Each presentation is made at the
mean of that distribution; each answer multiplies the distribution by how likely
that answer was for every candidate threshold; the search stops once the
distribution is narrow. This is the core that SITA builds on. What SITA adds -- a
prior from a large age-normal database, and information borrowed from neighbouring
points -- enters here only through :meth:`ZestLocationThreshold.set_prior`, so the
perimetry procedure can supply it from our own pilot table or from finished
neighbours.

Everything is on a **log10 Michelson contrast** axis, as in ``untitled.py``. The
psychometric function is the same Weibull QUEST+ uses there::

    p(seen | x, t) = guess + (1 - guess - lapse) * (1 - exp(-10 ** (slope * (x - t))))

**Quick blind rule.** A location missed at a clearly visible contrast that has never
been seen is shown at the maximum contrast. Missed ``quick_blind_misses`` times there
-> reported blind (CEILING_NOT_SEEN). Seen there -> the normal search resumes. A blind
point otherwise costs up to ``max_trials`` presentations for an answer known after a
few.

**Edges are failures, not thresholds.** An estimate at the top of the searched range
is reported CEILING_NOT_SEEN and one at the bottom FLOOR_ALWAYS_SEEN, both invalid --
unlike Specvis's bisection, which reports the floor as a valid threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import LocationThreshold, Result, Termination

ZEST = "ZEST"


@dataclass(frozen=True)
class ZestSettings:
    """Search range, observer model and stopping rule. Defaults match untitled.py."""

    min_contrast: float = 0.001
    max_contrast: float = 1.0
    step_log: float = 0.05            # grid spacing of candidate thresholds, log10
    slope: float = 3.5
    guess: float = 0.02               # false-alarm rate assumed by the model
    lapse: float = 0.02               # accidental misses (blinks, lapses)
    min_trials: int = 4
    max_trials: int = 20
    sd_stop: float = 0.15             # posterior SD at which a location is done, log10
    start_contrast: float = 0.5       # first presentation when there is no prior
    quick_blind_trigger: float = 0.5
    quick_blind_misses: int = 3
    boundary_margin: float = 0.10     # this close to a range end = edge, not threshold

    def __post_init__(self) -> None:
        if not 0 < self.min_contrast < self.max_contrast <= 1:
            raise ValueError("contrast range must satisfy 0 < min < max <= 1")
        if self.min_trials < 1 or self.max_trials < self.min_trials:
            raise ValueError("need 1 <= min_trials <= max_trials")

    @property
    def log_grid(self) -> np.ndarray:
        lo, hi = math.log10(self.min_contrast), math.log10(self.max_contrast)
        n = int(round((hi - lo) / self.step_log)) + 1
        return np.linspace(lo, hi, n)


def p_seen(log_contrast: float, log_thresholds: np.ndarray, s: ZestSettings) -> np.ndarray:
    """Probability of 'seen' at one contrast, for every candidate threshold."""
    z = np.clip(s.slope * (log_contrast - log_thresholds), -50.0, 50.0)
    return s.guess + (1.0 - s.guess - s.lapse) * (1.0 - np.exp(-np.power(10.0, z)))


def gaussian_prior(grid: np.ndarray, centre_log: float, sd_log: float,
                   flat_weight: float) -> np.ndarray:
    """A broad bump at ``centre_log``, part of it kept flat so a wrong guess can be
    overturned by the data."""
    bump = np.exp(-0.5 * ((grid - centre_log) / max(sd_log, 1e-6)) ** 2)
    bump /= bump.sum()
    flat = np.full(grid.size, 1.0 / grid.size)
    prior = (1.0 - flat_weight) * bump + flat_weight * flat
    return prior / prior.sum()


class ZestLocationThreshold(LocationThreshold):
    """One location, searched by ZEST, with the quick blind rule."""

    def __init__(self, settings: Optional[ZestSettings] = None,
                 renderable_floor: float = 0.0):
        s = settings or ZestSettings()
        if renderable_floor > s.min_contrast:
            # Never ask the display for a contrast it cannot draw.
            s = ZestSettings(**{**s.__dict__, "min_contrast": renderable_floor})
        self.s = s
        self.grid = s.log_grid
        self.posterior = np.full(self.grid.size, 1.0 / self.grid.size)
        self.prior_centre_log: Optional[float] = None
        self._trials = 0
        self._last_log: Optional[float] = None
        self.ever_seen = False
        self.ever_missed = False
        self._floor_seen = 0
        self.probing = False
        self._probed = False
        self._top_misses = 0
        self._stop: Optional[Termination] = None

    # -- prior -------------------------------------------------------------------

    def set_prior(self, centre_log: float, sd_log: float = 0.5,
                  flat_weight: float = 0.2) -> None:
        """Start from a guess instead of from nothing. Only before the first trial."""
        if self._trials:
            raise RuntimeError("the prior can only be set before the first presentation")
        self.posterior = gaussian_prior(self.grid, centre_log, sd_log, flat_weight)
        self.prior_centre_log = float(centre_log)

    # -- estimate ----------------------------------------------------------------

    @property
    def estimate_log(self) -> float:
        return float(np.sum(self.posterior * self.grid))

    @property
    def posterior_sd(self) -> float:
        m = self.estimate_log
        return float(math.sqrt(np.sum(self.posterior * (self.grid - m) ** 2)))

    # -- LocationThreshold -------------------------------------------------------

    def next_level(self) -> float:
        if self.probing:
            log_c = math.log10(self.s.max_contrast)
        elif self._trials == 0 and self.prior_centre_log is None:
            log_c = math.log10(self.s.start_contrast)
        else:
            log_c = self.estimate_log
        log_c = float(np.clip(log_c, self.grid[0], self.grid[-1]))
        self._last_log = log_c
        return 10.0 ** log_c

    def record(self, seen: bool) -> None:
        if self._last_log is None:
            raise RuntimeError("record() before next_level()")
        log_c = self._last_log
        likelihood = p_seen(log_c, self.grid, self.s)
        self.posterior = self.posterior * (likelihood if seen else 1.0 - likelihood)
        self.posterior /= self.posterior.sum()
        self._trials += 1
        self.ever_seen = self.ever_seen or seen
        self.ever_missed = self.ever_missed or not seen
        if seen and log_c <= self.grid[0] + 1e-9:
            self._floor_seen += 1
        self._last_log = None

        if self.probing:
            if seen:
                self.probing = False            # not blind: normal search resumes
            else:
                self._top_misses += 1
                if self._top_misses >= self.s.quick_blind_misses:
                    self._stop = Termination.CEILING_NOT_SEEN
                    return
        elif (not seen and not self.ever_seen and not self._probed
              and 10.0 ** log_c >= self.s.quick_blind_trigger):
            self.probing = self._probed = True

        if not self.ever_missed and self._floor_seen >= 2:
            # Seen even at the faintest contrast the display can draw: no threshold
            # was bracketed. Specvis reports this as a valid threshold; we do not.
            self._stop = Termination.FLOOR_ALWAYS_SEEN
        elif self._trials >= self.s.max_trials:
            self._stop = Termination.BUDGET_EXHAUSTED
        elif (not self.probing and self.ever_missed and self._trials >= self.s.min_trials
              and self.posterior_sd < self.s.sd_stop):
            # Only a search with at least one miss has bracketed a threshold.
            self._stop = Termination.CONVERGED

    def finished(self) -> bool:
        return self._stop is not None

    @property
    def blind_by_quick_rule(self) -> bool:
        return self._stop is Termination.CEILING_NOT_SEEN and self._top_misses >= \
            self.s.quick_blind_misses

    def result(self) -> Result:
        n = self._trials
        if self._stop is None:
            return Result.unusable(Termination.NOT_FINISHED, n,
                                   f"stopped early; estimate {10 ** self.estimate_log:.4f}")
        if self.blind_by_quick_rule:
            return Result.unusable(Termination.CEILING_NOT_SEEN, n,
                                   f"missed {self._top_misses}x at "
                                   f"{self.s.max_contrast:.2f} contrast")
        if self._stop is Termination.FLOOR_ALWAYS_SEEN or not self.ever_missed:
            return Result.unusable(Termination.FLOOR_ALWAYS_SEEN, n,
                                   f"never missed, down to {10 ** self.grid[0]:.4f} contrast")
        est, sd = self.estimate_log, self.posterior_sd
        if est >= self.grid[-1] - self.s.boundary_margin:
            return Result.unusable(Termination.CEILING_NOT_SEEN, n,
                                   "estimate at the top of the range")
        if est <= self.grid[0] + self.s.boundary_margin:
            return Result.unusable(Termination.FLOOR_ALWAYS_SEEN, n,
                                   "estimate at the bottom of the range")
        note = f"posterior SD {sd:.3f} log10"
        if self._stop is Termination.BUDGET_EXHAUSTED:
            note += f"; stopped at {self.s.max_trials} trials, not converged"
        return Result(True, 10.0 ** est, self._stop, n, note)

    @property
    def trials(self) -> int:
        return self._trials

    @property
    def method_name(self) -> str:
        return ZEST
