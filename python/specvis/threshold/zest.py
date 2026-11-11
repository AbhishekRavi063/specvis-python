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

**SITA Fast mode** (``fast_stop``; Bengtsson & Heijl 1998, Bengtsson et al. 1997,
Turpin et al. 2003). ZEST stays the engine -- Turpin et al. found it more accurate than
a SITA-like staircase when the starting guess is wrong, which is the edge of a blind
area -- and SITA Fast's rules are put around it:

* the stopping precision depends on the level, like SITA's error-related factor
  (ERF): looser where sensitivity is high (normal), stricter where it is low;
* a point may stop **after a single "seen"** when the estimate is already precise and
  agrees with the starting guess -- in practice only at normal points;
* the starting guess is SITA's two-model prior: mostly "normal" around the guess, part
  "abnormal" (anywhere worse), and a little flat;
* a point whose result is far from its guess can be **reopened** for a second, strict
  search (SITA's second staircase), and one never seen is confirmed blind after fewer
  misses at the maximum;
* after the test, :meth:`ZestLocationThreshold.recompute` re-estimates every point with
  the false-answer rates actually measured (SITA's post-processing).

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

    # SITA Fast (off by default; the perimetry procedure switches it on)
    fast_stop: bool = False
    fast_sd_base: float = 0.13        # SD limit = base + slope * sensitivity (-log10 c),
    fast_sd_slope: float = 0.06       # clipped to [base, fast_sd_max]: the ERF idea
    fast_sd_max: float = 0.25
    fast_min_trials: int = 2
    fast_near_log: float = 0.25       # early stop without a miss only this close to normal
    fast_abnormal_log: float = 0.3    # worse than normal by this much = an abnormal point:
    fast_min_trials_abnormal: int = 4  # it gets a full bracket (seen and missed), as in SITA
    fast_blind_misses: int = 2        # misses at the maximum that confirm "blind"
    retest_min_trials: int = 3        # a reopened point: at least this many more, and
    retest_sd_stop: float = 0.15      # this strict, before it stops again

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
                   flat_weight: float, abnormal_weight: float = 0.0,
                   abnormal_offset_log: float = 0.3,
                   sd_better_log: Optional[float] = None) -> np.ndarray:
    """A broad bump at ``centre_log``, part of it kept flat so a wrong guess can be
    overturned by the data.

    ``abnormal_weight`` adds SITA's second model: probability spread evenly over every
    threshold at least ``abnormal_offset_log`` worse than the guess (Turpin et al. 2003
    mixed abnormal and normal 1:4, i.e. 0.2).

    ``sd_better_log`` narrows the "normal" bump on the better side: nobody sees much
    better than normal, so -- as in SITA's normal model -- one "seen" at the expected
    value already pins a normal point down.
    """
    sd = np.where(grid < centre_log, sd_better_log or sd_log, sd_log)
    bump = np.exp(-0.5 * ((grid - centre_log) / np.maximum(sd, 1e-6)) ** 2)
    bump /= bump.sum()
    flat = np.full(grid.size, 1.0 / grid.size)
    if sd_better_log:
        # The flat part, too, stops a little better than normal (4 SD of the better side).
        possible = grid >= centre_log - 4.0 * sd_better_log
        flat = possible / possible.sum()
    worse = grid >= centre_log + abnormal_offset_log
    abnormal = worse / worse.sum() if worse.any() else flat
    prior = ((1.0 - flat_weight - abnormal_weight) * bump + flat_weight * flat
             + abnormal_weight * abnormal)
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
        self._prior = self.posterior.copy()
        self.history: list = []           # (log contrast, seen) of every presentation
        self.prior_centre_log: Optional[float] = None
        self.normal_log: Optional[float] = None   # the normal table's value, when known
        self._trials = 0
        self._last_log: Optional[float] = None
        self.ever_seen = False
        self._seen_in_search = False      # seen since this search (or its retest) began
        self.ever_missed = False
        self._floor_seen = 0
        self.probing = False
        self._probed = False
        self._top_misses = 0
        self._stop: Optional[Termination] = None
        self.early_stop = False           # stopped by SITA Fast without any miss
        self.retesting = False
        self.retested = False
        self._retest_start = 0

    # -- prior -------------------------------------------------------------------

    def set_prior(self, centre_log: float, sd_log: float = 0.5,
                  flat_weight: float = 0.2, abnormal_weight: float = 0.0,
                  sd_better_log: Optional[float] = None,
                  normal_log: Optional[float] = None) -> None:
        """Start from a guess instead of from nothing. Only before the first trial."""
        if self._trials:
            raise RuntimeError("the prior can only be set before the first presentation")
        self.posterior = gaussian_prior(self.grid, centre_log, sd_log, flat_weight,
                                        abnormal_weight, sd_better_log=sd_better_log)
        self._prior = self.posterior.copy()
        self.prior_centre_log = float(centre_log)
        self.normal_log = None if normal_log is None else float(normal_log)

    # -- SITA Fast ---------------------------------------------------------------

    @property
    def fast_sd_limit(self) -> float:
        """The ERF-like stopping precision at the current estimate."""
        s = self.s
        return float(np.clip(s.fast_sd_base + s.fast_sd_slope * -self.estimate_log,
                             s.fast_sd_base, s.fast_sd_max))

    @property
    def _blind_misses(self) -> int:
        return self.s.fast_blind_misses if self.s.fast_stop else self.s.quick_blind_misses

    def reopen(self) -> None:
        """A second, strict search at a point whose result was far from its guess
        (SITA's second staircase). The evidence so far is kept."""
        if self._stop is None or self.retested:
            return
        self._stop = None
        self.early_stop = False
        # A fresh blind check: a blind point that one chance press made look "weak" is
        # missed throughout the retest, and is then probed at the maximum again.
        self._seen_in_search = False
        self._probed = False
        self._top_misses = 0
        self.retesting = self.retested = True
        self._retest_start = self._trials

    def recompute(self, guess: float, lapse: float) -> None:
        """Re-estimate from the prior and every answer, with the false-answer rates
        measured in this session (SITA's post-processing). The stopping decision is
        not changed, only the estimate."""
        guess = float(np.clip(guess, 0.0, 0.3))
        lapse = float(np.clip(lapse, 0.0, 0.3))
        s = ZestSettings(**{**self.s.__dict__, "guess": guess, "lapse": lapse})
        post = self._prior.copy()
        for log_c, seen in self.history:
            lik = p_seen(log_c, self.grid, s)
            post = post * (lik if seen else 1.0 - lik)
            post /= post.sum()
        self.posterior = post

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
        self.history.append((log_c, bool(seen)))
        self.ever_seen = self.ever_seen or seen
        self._seen_in_search = self._seen_in_search or seen
        self.ever_missed = self.ever_missed or not seen
        if seen and log_c <= self.grid[0] + 1e-9:
            self._floor_seen += 1
        self._last_log = None

        if self.probing:
            if seen:
                self.probing = False            # not blind: normal search resumes
            else:
                self._top_misses += 1
                if self._top_misses >= self._blind_misses:
                    self._stop = Termination.CEILING_NOT_SEEN
                    return
        elif (not seen and not self._seen_in_search and not self._probed
              and 10.0 ** log_c >= self.s.quick_blind_trigger):
            self.probing = self._probed = True

        if not self.ever_missed and self._floor_seen >= 2:
            # Seen even at the faintest contrast the display can draw: no threshold
            # was bracketed. Specvis reports this as a valid threshold; we do not.
            self._stop = Termination.FLOOR_ALWAYS_SEEN
        elif self._trials >= self.s.max_trials:
            self._stop = Termination.BUDGET_EXHAUSTED
        elif self.probing:
            pass
        elif self.retesting:
            if (self._trials - self._retest_start >= self.s.retest_min_trials
                    and self.ever_missed and self.posterior_sd < self.s.retest_sd_stop):
                self._stop = Termination.CONVERGED
        elif self.s.fast_stop:
            # SITA Fast stops early only where the point looks normal; anything worse than
            # the normal table (or with no table) gets a full bracket first.
            est = self.estimate_log
            normal = (self.normal_log is not None
                      and est <= self.normal_log + self.s.fast_abnormal_log)
            near = normal and abs(est - self.normal_log) <= self.s.fast_near_log
            need = self.s.fast_min_trials if normal else self.s.fast_min_trials_abnormal
            # Near-blind: one "seen" among misses may be a chance press, so a point that
            # would be reported as needing 50% contrast or more must be seen twice.
            seen_count = sum(1 for _, seen_ in self.history if seen_)
            confirmed = (est < math.log10(self.s.quick_blind_trigger) or seen_count >= 2)
            if (self.ever_seen and self._trials >= need and confirmed
                    and self.posterior_sd < self.fast_sd_limit
                    and (self.ever_missed or near)):
                self._stop = Termination.CONVERGED
                self.early_stop = not self.ever_missed
        elif (self.ever_missed and self._trials >= self.s.min_trials
              and self.posterior_sd < self.s.sd_stop):
            # Only a search with at least one miss has bracketed a threshold.
            self._stop = Termination.CONVERGED

    def finished(self) -> bool:
        return self._stop is not None

    @property
    def blind_by_quick_rule(self) -> bool:
        return self._stop is Termination.CEILING_NOT_SEEN and self._top_misses >= \
            self._blind_misses

    def result(self) -> Result:
        n = self._trials
        if self._stop is None:
            return Result.unusable(Termination.NOT_FINISHED, n,
                                   f"stopped early; estimate {10 ** self.estimate_log:.4f}")
        if self.blind_by_quick_rule:
            return Result.unusable(Termination.CEILING_NOT_SEEN, n,
                                   f"missed {self._top_misses}x at "
                                   f"{self.s.max_contrast:.2f} contrast")
        if self._stop is Termination.FLOOR_ALWAYS_SEEN or not (self.ever_missed
                                                              or self.early_stop):
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
        if self.early_stop:
            note += "; SITA Fast early stop (seen near the expected value)"
        if self.retested:
            note += "; retested (far from expected)"
        if self._stop is Termination.BUDGET_EXHAUSTED:
            note += f"; stopped at {self.s.max_trials} trials, not converged"
        return Result(True, 10.0 ** est, self._stop, n, note)

    @property
    def trials(self) -> int:
        return self._trials

    @property
    def method_name(self) -> str:
        return ZEST
