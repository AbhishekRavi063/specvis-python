"""Pattern-reversal timing: what a display can actually present, and what it cannot.

The proposal's high-beta/gamma condition needs reversal at **25-40 Hz**, a phase flip
every 12.5-20 ms. That is the binding constraint on the entire software stack, and it
is why reversal here is driven by **counting vsync frames** rather than by any
wall-clock timer.

Two rates, and confusing them misplaces the EEG response by an octave::

    reversals per second = refresh / frames_per_reversal
    cycles per second    = refresh / (2 * frames_per_reversal)

One full cycle is **two** reversals. The SSVEP fundamental appears at the **reversal**
rate, so that is the number to report.

Whether a rate is presentable is a property of **the target monitor**, never of
displays in general:

===========  =========================================
refresh      exactly presentable reversal rates
===========  =========================================
120 Hz       60, **40**, 30, 24, 20
100 Hz       **50**, 33.3, **25**, 20
76 Hz        **38** (k=2)
118.9 Hz     none of 25, 38, 40 or 50
===========  =========================================

No common refresh gives both 25 and 40 exactly; that needs a refresh divisible by 200.
Ask what the laboratory monitor runs at before calling any rate impossible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

#: How close a realisable rate must be to count as an exact divisor of the refresh.
#: Only meaningful against a *nominal* refresh; a measured one is never round.
DIVISOR_TOLERANCE_HZ = 1e-6


@dataclass(frozen=True)
class ReversalPlan:
    """How to present one reversal rate on one display, or why it cannot be.

    ``frames_per_reversal`` is an integer, so a full cycle is ``2 * k`` frames and the
    two phases are automatically symmetric. Asymmetric phases would put energy at the
    cycle rate as well as the reversal rate and muddy the spectrum.
    """

    requested_hz: float
    refresh_hz: float
    frames_per_reversal: int
    static: bool = False

    # -- construction ----------------------------------------------------------

    @classmethod
    def static_patch(cls, refresh_hz: float) -> "ReversalPlan":
        """No reversal at all -- the 200 ms static patch of the prototype brief."""
        return cls(0.0, refresh_hz, 0, static=True)

    @classmethod
    def for_rate(cls, requested_hz: float, refresh_hz: float) -> "ReversalPlan":
        """Nearest presentable plan for this rate on this display.

        Always returns a plan. Ask :meth:`is_exact` and :meth:`usable` before using it;
        an inexact plan is not refused here because the operator may legitimately
        accept it, but it must never be silently presented as the requested rate.
        """
        if requested_hz <= 0:
            return cls.static_patch(refresh_hz)
        if refresh_hz <= 0:
            raise ValueError("refresh rate must be positive")

        k = max(1, round(refresh_hz / requested_hz))
        return cls(float(requested_hz), float(refresh_hz), int(k))

    # -- what it actually produces ---------------------------------------------

    @property
    def actual_reversals_per_second(self) -> float:
        if self.static or self.frames_per_reversal == 0:
            return 0.0
        return self.refresh_hz / self.frames_per_reversal

    @property
    def actual_cycles_per_second(self) -> float:
        return self.actual_reversals_per_second / 2.0

    @property
    def error_hz(self) -> float:
        return abs(self.actual_reversals_per_second - self.requested_hz)

    def is_exact(self) -> bool:
        """Whether the rate divides the refresh exactly.

        This is the **procurement** question -- "can this panel do 40 Hz?" -- and it
        is only meaningful against a *nominal* refresh. A measured refresh is never a
        round number: this display measures 119.966 Hz, so every rate fails this test
        against it while being perfectly usable. Use :meth:`error_is_resolvable` to
        decide whether a measured discrepancy actually matters.
        """
        if self.static:
            return True
        return self.error_hz <= DIVISOR_TOLERANCE_HZ

    def frequency_resolution_hz(self, duration_ms: float) -> float:
        """FFT bin width for a train of this length: ``1000 / duration_ms``.

        A 5 s train resolves 0.2 Hz. Nothing finer than this can be distinguished in
        the spectrum however carefully it was presented.
        """
        if duration_ms <= 0:
            raise ValueError("duration must be positive")
        return 1000.0 / duration_ms

    def error_is_resolvable(self, duration_ms: float) -> bool:
        """Whether the rate error is large enough to show up in the spectrum.

        The honest test for a measured display. Presenting 39.989 instead of 40.000
        over a 5 s train is an error of 0.011 Hz against a bin width of 0.2 Hz -- it
        cannot be resolved, so calling it a problem would be noise. An error of 1 Hz
        over the same train lands in a different bin and must be reported.
        """
        if self.static:
            return False
        return self.error_hz > self.frequency_resolution_hz(duration_ms) / 2.0

    def usable(self) -> bool:
        """At least two frames per cycle, so the two phases are distinguishable."""
        return self.static or self.frames_per_reversal >= 1

    def refusal_reason(self, duration_ms: float = 5000.0) -> Optional[str]:
        """Why this plan should not be presented as the requested rate, if it should not.

        Judged against the train length, because that sets what the spectrum can
        resolve. The default is the proposal's 5 s train.
        """
        if not self.usable():
            return (f"{self.requested_hz} reversals/s needs fewer than one frame per "
                    f"phase at {self.refresh_hz} Hz")
        if self.error_is_resolvable(duration_ms):
            return (f"{self.requested_hz} reversals/s is not presentable at "
                    f"{self.refresh_hz:.3f} Hz; nearest is "
                    f"{self.actual_reversals_per_second:.4f} "
                    f"({self.frames_per_reversal} frames per reversal, error "
                    f"{self.error_hz:.4f} Hz, which exceeds the "
                    f"{self.frequency_resolution_hz(duration_ms):.3f} Hz resolution of a "
                    f"{duration_ms:.0f} ms train)")
        return None

    def expected_reversals(self, duration_ms: float) -> int:
        """Reversals in a train of this length, for cross-checking what was presented.

        ``floor(duration * refresh / k)`` on the unrounded frame count, as the Java
        computes it: 4999 ms at 60 Hz and 12 frames per reversal is 24, not 25. The
        behavioural procedure invalidates a trial only when the count presented is
        more than one away from this, so the floor and the frame counter may differ
        by one without consequence.
        """
        if self.static or self.frames_per_reversal == 0 or self.refresh_hz <= 0:
            return 0
        frames = duration_ms / 1000.0 * self.refresh_hz
        return int(math.floor(frames / self.frames_per_reversal))

    def describe(self, duration_ms: float = 5000.0) -> str:
        if self.static:
            return f"static patch, no reversal (display {self.refresh_hz:.2f} Hz)"
        note = ""
        if self.error_is_resolvable(duration_ms):
            note = (f"  [REQUESTED {self.requested_hz}, error {self.error_hz:.4f} Hz "
                    f"IS RESOLVABLE over {duration_ms:.0f} ms]")
        elif self.error_hz > DIVISOR_TOLERANCE_HZ:
            note = (f"  [requested {self.requested_hz}, error {self.error_hz:.4f} Hz, "
                    f"below the {self.frequency_resolution_hz(duration_ms):.3f} Hz "
                    f"resolution of a {duration_ms:.0f} ms train]")
        return (
            f"{self.actual_reversals_per_second:.4f} reversals/s "
            f"({self.actual_cycles_per_second:.4f} cycles/s), "
            f"{self.frames_per_reversal} frame(s) per reversal at "
            f"{self.refresh_hz:.2f} Hz" + note
        )


def presentable_rates(refresh_hz: float, low_hz: float, high_hz: float) -> list[float]:
    """Every exactly presentable reversal rate in a band, brightest ordering first.

    Use this to answer "can this monitor do the 25-40 Hz condition?" before procuring
    one, rather than discovering the answer during a recording session.
    """
    rates = []
    k = 1
    while refresh_hz / k >= low_hz:
        rate = refresh_hz / k
        if low_hz <= rate <= high_hz:
            rates.append(rate)
        k += 1
        if k > 10000:
            break
    return rates
