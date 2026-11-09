"""Invariants of the ported core that must not drift from the Java."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.core.gabor import (  # noqa: E402
    GaborParameters, luminance_field, measure_contrast, minimum_renderable_contrast,
    patch_size_px,
)
from specvis.core.ladder import (  # noqa: E402
    ContrastLadder, brightness_vector, contrast_db, db_to_contrast, perimetric_db,
)
from specvis.core.luminance import LuminanceCalibration  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


def main() -> int:
    print("=== 1. The ladder reproduces upstream's truncation ===")
    v = brightness_vector(10, 100, 9)
    require(v == [10, 13, 17, 23, 31, 42, 56, 74, 100],
            f"default 9-rung ladder matches the Java: {v}")
    require(brightness_vector(10, 100, 9, rounding="nearest")
            == [10, 13, 18, 24, 32, 42, 56, 75, 100],
            "rounding='nearest' gives the ladder the settings imply")

    ladder = ContrastLadder.from_vector(v, 0.8)
    require(abs(ladder.step_range_db() - 0.36) < 0.01,
            f"truncation spreads the steps over 0.36 dB ({ladder.step_range_db():.2f})")
    require(abs(ladder.irregularity_db() - 0.22) < 0.01,
            f"which the Java's metadata reports as 0.22 dB irregularity, the worst "
            f"deviation from the mean step ({ladder.irregularity_db():.2f})")
    require(brightness_vector(1, 40, 3, spread_log=False) == [1, 21, 40],
            "a linear ladder is rounded, not truncated: linspace rounds before the cast")
    require(abs(ladder.min_contrast - 0.08) < 1e-12
            and abs(ladder.max_michelson - 0.80) < 1e-12,
            "ladder spans 0.08 to 0.80 Michelson")

    print("=== 2. Contrast dB round-trips ===")
    for c in (0.08, 0.1596, 0.253, 0.8):
        require(abs(db_to_contrast(contrast_db(c)) - c) < 1e-12,
                f"{c} -> {contrast_db(c):.2f} dB -> back")
    require(abs(contrast_db(0.5) - 6.02) < 0.01, "half contrast is ~6 dB")
    require(abs(contrast_db(0.1) - 20.0) < 1e-9, "one tenth is exactly 20 dB")

    print("=== 2b. Perimetric dB is a different quantity from contrast dB ===")
    # Functions.decibelsValue: 10*log10(max/stimulus). This is the number in
    # session_data.txt and on the results-map colourbar.
    require(abs(perimetric_db(214.67, 3.26103) - 18.18) < 1e-9,
            "the pilot's dimmest rung is 18.18 dB, matching the Java colourbar")
    require(abs(perimetric_db(214.67, 214.67)) < 1e-9, "the top rung is 0 dB")
    require(abs(perimetric_db(214.67, 3.26103) - contrast_db(0.08)) > 3.0,
            "and it is NOT the same number as contrast dB for the same rung "
            f"({perimetric_db(214.67, 3.26103)} vs {contrast_db(0.08):.2f})")
    # 10*log10 of a luminance ratio halves every 3 dB, not every 6.
    require(abs(perimetric_db(100.0, 50.0) - 3.01) < 0.01,
            "halving luminance is ~3 dB here, where halving contrast is ~6 dB there")

    print("=== 3. Luminance calibration inverts itself ===")
    cal = LuminanceCalibration.from_measurements(
        [0.5, 20.0, 60.0, 110.0, 165.0, 214.67], "synthetic test scale")
    require(cal.calibrated, "a fitted scale reports itself calibrated")
    require(not LuminanceCalibration.uncalibrated().calibrated,
            "an uncalibrated scale says so, so output can be labelled")
    for b in (0.0, 17.0, 42.0, 73.5, 100.0):
        back = cal.brightness_for(cal.luminance_at(b))
        require(abs(back - b) < 1e-6, f"brightness {b} round-trips ({back:.6f})")
    require(cal.brightness_for(-999.0) == 0.0 and cal.brightness_for(1e9) == 100.0,
            "targets outside the range clamp to the endpoints")

    print("=== 4. The Gabor keeps its mean and its geometry ===")
    p = GaborParameters(2.0, 4.0, 90.0, 0.0, 0.33, 57.84, 0.25)
    w, h = patch_size_px(p, 37.0, 37.0)
    require(w % 2 == 1 and h % 2 == 1,
            f"patch is odd-sized so the centre pixel exists ({w}x{h})")

    field = luminance_field(p, 37.0, 37.0)
    deviation = abs(field.mean() / p.mean_luminance - 1.0)
    require(deviation < 1e-12,
            f"DC correction holds the mean at nominal (deviation {deviation:.2e})")

    # A patch whose mean differs from the background is a luminance increment and can
    # be detected without resolving the grating, so this matters at low frequency.
    low = GaborParameters(2.0, 1.0, 90.0, 90.0, 0.33, 57.84, 0.25)
    low_dev = abs(luminance_field(low, 37.0, 37.0).mean() / low.mean_luminance - 1.0)
    require(low_dev < 1e-12,
            f"and still holds at 1 c/deg cosine phase, the worst case ({low_dev:.2e})")

    print("=== 5. Pattern reversal swaps polarity, not mean luminance ===")
    reversed_field = luminance_field(p.reversed_phase(), 37.0, 37.0)
    require(abs(field.mean() - reversed_field.mean()) < 1e-12,
            "both polarities have identical mean luminance")
    differing = float(np.mean(np.abs(field - reversed_field) > 1e-9))
    require(differing > 0.9,
            f"but {differing * 100:.0f}% of pixels differ, so the pattern really reverses")

    print("=== 6. The display floor is reported, not ignored ===")
    floor = minimum_renderable_contrast(cal, 57.84)
    require(0 < floor < 0.05, f"floor at this mean luminance is {floor:.5f} Michelson")
    rendered = measure_contrast(p, cal, 37.0, 37.0)
    require(0 < rendered <= p.michelson_contrast,
            f"rendered contrast {rendered:.4f} does not exceed the requested "
            f"{p.michelson_contrast:.4f}")

    if not ok:
        print("\nFAILED")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
