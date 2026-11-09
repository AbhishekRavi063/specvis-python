"""The seam between the science and the display.

Everything above this seam is pure Python and numpy. These checks run against
NullBackend, so they prove the *logic* of presentation and never the timing -- timing
is photodiode work, and nothing here should be read as validating it.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.core.gabor import GaborParameters, render_brightness  # noqa: E402
from specvis.core.luminance import LuminanceCalibration  # noqa: E402
from specvis.core.reversal import ReversalPlan  # noqa: E402
from specvis.stimulus.backend import NullBackend, Texture  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


def main() -> int:
    cal = LuminanceCalibration.from_measurements(
        [0.5, 20.0, 60.0, 110.0, 165.0, 214.67], "synthetic")
    p = GaborParameters(2.0, 4.0, 90.0, 0.0, 0.33, 57.84, 0.25)
    a = Texture(render_brightness(p, cal, 37, 37), (400.0, 300.0))
    b = Texture(render_brightness(p.reversed_phase(), cal, 37, 37), (400.0, 300.0))

    print("=== 1. A texture is a valid brightness field ===")
    require(a.brightness.min() >= 0 and a.brightness.max() <= 1,
            "brightness stays within [0, 1]")
    try:
        Texture(np.array([2.0, 0.0]), (0.0, 0.0))
        require(False, "a 1-D or out-of-range texture is rejected")
    except ValueError:
        require(True, "a 1-D or out-of-range texture is rejected")

    print("=== 2. The two polarities differ in pattern, not in mean ===")
    require(abs(a.brightness.mean() - b.brightness.mean()) < 1e-12,
            "identical mean brightness, so the reversal is not a luminance flicker")
    require(float(np.mean(np.abs(a.brightness - b.brightness) > 1e-9)) > 0.9,
            "but the pattern really does invert")

    print("=== 3. A 5 s train at 40 rev/s, as the proposal specifies ===")
    plan = ReversalPlan.for_rate(40.0, 120.0)
    backend = NullBackend(120.0)
    backend.open(1512, 982, 0.5)
    frames = round(5000 / 1000 * backend.refresh_hz)
    report = backend.present_train([a, b], plan.frames_per_reversal, frames)

    require(report.frames_presented == 600, "5 s at 120 Hz is 600 frames")
    require(report.reversals_presented == 200, "which is 200 reversals at 3 frames each")
    require(report.reversals_presented == plan.expected_reversals(5000),
            "and the plan predicted exactly that, independently")
    require(abs(report.actual_reversals_per_second(plan.frames_per_reversal) - 40.0) < 1e-9,
            "the report recomputes 40 rev/s from the measured refresh")

    print("=== 4. Timing claims are labelled as self-report ===")
    require("NOT PHOTODIODE-VALIDATED" in report.describe(),
            "every presentation report says its timing is unvalidated")

    print("=== 5. The seam is honest about what it did ===")
    require(backend.trains == [(3, 600)],
            "the backend records what it was actually asked to present")
    require(not report.dropped_any, "NullBackend reports a perfect display -- by fiat")

    print("=== 6. Scripted responses drive the logic without a display ===")
    scripted = NullBackend(120.0, scripted_keys=[("space", 342.0), ("escape", 10.0)])
    key, latency = scripted.wait_for_key(["space"], 1200.0)
    require(key == "space" and latency == 342.0, "a scripted press comes back")
    key, _ = scripted.wait_for_key(["space"], 1200.0)
    require(key is None, "a key not in the accepted set is not reported")
    key, _ = scripted.wait_for_key(["space"], 1200.0)
    require(key is None, "and an exhausted script reads as no response")

    if not ok:
        print("\nFAILED")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
