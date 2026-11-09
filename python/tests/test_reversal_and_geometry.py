"""Reversal timing and stimulus geometry.

The 25-40 Hz condition is the binding constraint on the whole stack, so the two rate
formulas and the per-monitor feasibility table are pinned here. Getting the two rates
the wrong way round misplaces the SSVEP response by an octave.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.core.geometry import (  # noqa: E402
    PhysicalScreen, ScreenGeometry, degrees_subtended, quadrant_locations,
)
from specvis.core.reversal import ReversalPlan, presentable_rates  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


def main() -> int:
    print("=== 1. Reversals and cycles are not the same number ===")
    p = ReversalPlan.for_rate(40.0, 120.0)
    require(p.frames_per_reversal == 3, "40 rev/s at 120 Hz is 3 frames per reversal")
    require(abs(p.actual_reversals_per_second - 40.0) < 1e-9,
            "reversals/s = refresh / frames_per_reversal")
    require(abs(p.actual_cycles_per_second - 20.0) < 1e-9,
            "cycles/s is half that -- one cycle is two reversals")
    require(p.is_exact(), "and it is exact, so the phases are symmetric")

    print("=== 2. Feasibility is a property of the monitor, not of displays ===")
    require([round(r, 3) for r in presentable_rates(120, 25, 40)] == [40.0, 30.0],
            "120 Hz gives 40 and 30 exactly in the 25-40 band")
    require([round(r, 3) for r in presentable_rates(100, 25, 40)] == [33.333, 25.0],
            "100 Hz gives 33.3 and 25")
    require(round(presentable_rates(76, 25, 40)[0], 3) == 38.0,
            "76 Hz gives 38 exactly")
    require(not ReversalPlan.for_rate(25.0, 120.0).is_exact(),
            "25 rev/s is NOT exact at 120 Hz -- it lands on 24")
    # No common refresh gives both 25 and 40: that needs a refresh divisible by 200.
    require(not (set(presentable_rates(120, 25, 40)) >= {25.0, 40.0}),
            "no single refresh in the table gives both 25 and 40 exactly")

    print("=== 3. Exactness is judged against what the spectrum can resolve ===")
    # A *measured* refresh is never round -- this display measures 119.966 Hz -- so
    # testing against exact division would flag every rate. The question that matters
    # is whether the error lands in a different FFT bin.
    bad = ReversalPlan.for_rate(25.0, 120.0)
    require(bad.refusal_reason() is not None,
            f"25 rev/s at 120 Hz is refused: {bad.refusal_reason()[:50]}...")
    require(abs(bad.actual_reversals_per_second - 24.0) < 1e-9,
            "and reports what it would actually present (24.0)")

    measured = ReversalPlan.for_rate(40.0, 119.966)
    require(not measured.is_exact(),
            "40 rev/s does not divide a measured 119.966 Hz exactly")
    require(abs(measured.frequency_resolution_hz(5000) - 0.2) < 1e-9,
            "a 5 s train resolves 0.2 Hz")
    require(not measured.error_is_resolvable(5000),
            f"but the {measured.error_hz:.4f} Hz error is below that, so it is usable")
    require(measured.refusal_reason(5000) is None,
            "and it is therefore not refused")

    require(ReversalPlan.for_rate(25.0, 119.966).error_is_resolvable(5000),
            "while 25 rev/s on the same display is off by 1.0 Hz, which IS resolvable")

    # A shorter train resolves less, so it tolerates more error.
    require(not ReversalPlan.for_rate(25.0, 119.966).error_is_resolvable(200),
            "over a 200 ms train the same 1.0 Hz error falls inside one bin")

    print("=== 4. A 5 s train presents the expected number of reversals ===")
    # Proposal section 6a: 5 s of reversal, then 400 ms ISI.
    require(ReversalPlan.for_rate(40.0, 120.0).expected_reversals(5000) == 200,
            "40 rev/s for 5 s is 200 reversals")
    require(ReversalPlan.for_rate(6.0, 120.0).expected_reversals(5000) == 30,
            "6 rev/s for 5 s is 30 reversals")
    require(ReversalPlan.static_patch(120.0).expected_reversals(200) == 0,
            "a static patch presents none")

    print("=== 5. The proposal's four locations ===")
    locations = quadrant_locations(5.0)
    require(len(locations) == 4, "four quadrants, as in Fig. 1")
    require([l.quadrant for l in locations] ==
            ["upper-left", "upper-right", "lower-right", "lower-left"],
            "ordered so the index is stable across sessions")
    require(all(abs(abs(l.degrees_x) - 5.0) < 1e-9 and abs(abs(l.degrees_y) - 5.0) < 1e-9
                for l in locations),
            "all at +/-5 degrees horizontal and vertical")

    print("=== 6. Degrees and pixels ===")
    screen = ScreenGeometry(1512, 982, 64.01, 41.11)
    require(abs(screen.px_per_degree_x - 1512 / 64.01) < 1e-9, "pixels per degree x")
    cx, cy = screen.fixation_px
    require(abs(cx - 756.0) < 1e-9 and abs(cy - 491.0) < 1e-9,
            "fixation is centred when no offset is set")
    x, y = screen.to_px(-5.0, -5.0)
    require(x < cx and y < cy,
            "upper-left in the visual field is up and left on screen")
    require(screen.fits(5.0, 5.0, 2.0), "a 2 degree patch at 5 degrees fits")
    # The screen subtends 64.01 degrees, so the edge is at +/-32.
    require(screen.fits(31.0, 0.0, 2.0), "31 degrees is still inside a 64 degree field")
    require(not screen.fits(33.0, 0.0, 2.0), "and 33 degrees is off the screen")

    require(abs(degrees_subtended(300.0, 600.0) - 28.072) < 0.001,
            "a 300 mm screen at 600 mm subtends 28.07 degrees, not 30")

    print("=== 7. The spherical correction, which the flat model omits ===")
    # Specvis's own model uses one px/degree everywhere. The screen is flat and the
    # visual field is not, so the error grows with eccentricity.
    screen = PhysicalScreen(1512, 982, 302.0, 196.0, 600.0)
    require(abs(screen.sphericity_factor(0.0) - 1.0) < 1e-12,
            "no correction at the centre")
    for eccentricity, factor in ((5.0, 1.00765), (10.0, 1.03109), (20.0, 1.13247)):
        require(abs(screen.sphericity_factor(eccentricity) - factor) < 1e-5,
                f"{eccentricity:.0f} deg -> {factor:.5f} "
                f"({100 * (factor - 1):+.2f}% vs centre)")
    require(abs(screen.local_px_per_degree_x(0.0) - 52.429) < 0.001,
            "centre scale from measured millimetres, not a stated field")
    require(screen.local_px_per_degree_x(20.0) > screen.local_px_per_degree_x(5.0),
            "the local scale grows with eccentricity")
    require(abs(screen.field_degrees_x - degrees_subtended(302.0, 600.0)) < 1e-9,
            "the field it subtends is derived, not typed in")
    require(abs(screen.flat_geometry().px_per_degree_x
                - 1512 / screen.field_degrees_x) < 1e-9,
            "and the flat approximation can be derived from it for comparison")

    if not ok:
        print("\nFAILED")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
