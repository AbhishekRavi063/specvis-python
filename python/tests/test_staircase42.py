"""The 4-2 staircase: the trace, the stopping rule, and both boundaries."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.threshold.base import Termination  # noqa: E402
from specvis.threshold.staircase42 import (  # noqa: E402
    ContrastLocationThreshold, FourTwoStaircase, contrast_to_db, db_to_contrast,
)

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


def main() -> int:
    print("=== 1. Step sizes are 4 dB then 2 dB ===")
    s = FourTwoStaircase(0.2530, 0.80, 0.08)
    first = s.current_contrast
    s.record(True)
    second = s.current_contrast
    require(abs(contrast_to_db(second) - contrast_to_db(first) - 4.0) < 1e-9,
            f"first step is 4 dB ({first:.4f} -> {second:.4f})")
    s.record(False)                       # a reversal
    third = s.current_contrast
    require(abs(contrast_to_db(third) - contrast_to_db(second) + 2.0) < 1e-9,
            f"after the first reversal the step is 2 dB ({second:.4f} -> {third:.4f})")

    print("=== 2. A converging run stops on reversals, not before min_trials ===")
    s = FourTwoStaircase(0.2530, 0.80, 0.08)
    trace = []
    for seen in (True, False, True, True):
        trace.append(round(s.current_contrast, 4))
        s.record(seen)
    require(trace == [0.2530, 0.1596, 0.2010, 0.1596],
            f"trace matches the documented worked example: {trace}")
    require(s.finished and s.trials == 4, "stops at 4 presentations, the min_trials floor")
    result = s.outcome()
    require(result.valid and result.termination is Termination.CONVERGED,
            f"and reports a valid converged threshold of {result.value:.4f}")

    print("=== 3. A lapse on trial one cannot produce a 2-trial threshold ===")
    s = FourTwoStaircase(0.2530, 0.80, 0.08)
    s.record(False)
    s.record(True)
    require(not s.finished,
            "two reversals inside two trials do not stop the run (min_trials)")

    print("=== 4. Both boundaries are detected, and the edge is presented ===")
    s = FourTwoStaircase(0.2530, 0.80, 0.08)
    while not s.finished and s.trials < 30:
        s.record(True)
    r = s.outcome()
    require(r.termination is Termination.FLOOR_ALWAYS_SEEN and not r.valid,
            "always seen -> floor_always_seen, reported invalid not as a threshold")
    require(any(abs(step.db - contrast_to_db(0.08)) < 1e-9 for step in s.history),
            "the minimum contrast was actually presented before the run ended")

    s = FourTwoStaircase(0.2530, 0.80, 0.08)
    while not s.finished and s.trials < 30:
        s.record(False)
    r = s.outcome()
    require(r.termination is Termination.CEILING_NOT_SEEN and not r.valid,
            "never seen -> ceiling_not_seen, reported invalid")
    require(any(abs(step.db - contrast_to_db(0.80)) < 1e-9 for step in s.history),
            "the maximum contrast was actually presented before the run ended")

    print("=== 5. A contrast below the display floor invalidates the estimate ===")
    # A requested contrast the display cannot render shows a uniform grey field. If
    # that is logged as a stimulus, a "threshold" is recorded for a patch never shown.
    t = ContrastLocationThreshold(
        FourTwoStaircase(0.1000, 0.80, 0.0100), "4-2", renderable_floor=0.05)
    presented = []
    for seen in (True, True, False, True):      # converges, but dips below the floor
        presented.append(round(t.next_level(), 4))
        t.record(seen)
    require(t.finished(), f"the run converged (presented {presented})")
    require(any(abs(c - 0.05) < 1e-9 for c in presented),
            "at least one presentation was clamped up to the floor")
    require(not t.result().valid,
            "clamping makes the run invalid even though the staircase converged")
    require("renderable floor" in t.result().note, "and the note says why")

    # A run that fails for its own reason keeps its own explanation.
    t2 = ContrastLocationThreshold(
        FourTwoStaircase(0.2530, 0.80, 0.08), "4-2", renderable_floor=0.0)
    while not t2.finished() and t2.trials < 30:
        t2.next_level()
        t2.record(True)
    require("below the measurable floor" in t2.result().note,
            "a floor_always_seen run reports the staircase's reason, not the clamp's")

    print("=== 6. dB axis round-trips ===")
    for c in (0.0147, 0.08, 0.253, 0.80):
        require(abs(db_to_contrast(contrast_to_db(c)) - c) < 1e-12, f"{c} round-trips")

    if not ok:
        print("\nFAILED")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
