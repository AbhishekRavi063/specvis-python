"""Dropped-frame counting must catch a train where EVERY frame is slow.

Lab PC, 24 Nov 2026: with the first flat-field version each frame took two refreshes
(240 Hz shown as 120 Hz). The old count compared each frame with the train's own median,
which doubled too, so it said 0 dropped. Runs without PsychoPy (no window is opened)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.stimulus.backend import Texture  # noqa: E402
from specvis.stimulus.psychopy_backend import PsychoPyBackend, count_dropped  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok    " if condition else "  FAIL  ") + what)
    ok = ok and bool(condition)


REFRESH = 240.0
one = 1000.0 / REFRESH
good = [one] * 239
require(count_dropped(good, REFRESH) == 0, "a train at 240 Hz: 0 dropped")
good_one_late = good[:100] + [2 * one] + good[101:]
require(count_dropped(good_one_late, REFRESH) == 1, "one frame of two refreshes: 1 dropped")
# The fake slow display: every frame takes two refreshes (the 24 Nov flat-field bug).
slow = [2 * one] * 239
require(count_dropped(slow, REFRESH) == 239,
        f"every frame two refreshes: all {len(slow)} counted as dropped (was 0)")
mixed = [2 * one, 3 * one] * 119
require(count_dropped(mixed, REFRESH) == len(mixed), "frames of 2-3 refreshes: all dropped")
require(count_dropped(good_one_late, float("nan")) == 1,
        "no measured refresh: falls back to the train's own median")
require(count_dropped([], REFRESH) == 0, "an empty train: 0")

# The piece of flat-field background painted back after a train must be exactly where
# the patch was drawn (same rounding as _to_stim and FlatField._window).
for centre, shape in [((1260.0, 340.0), (30, 30)), ((960.5, 540.25), (31, 47)),
                      ((10.0, 1075.0), (40, 40))]:
    rect = PsychoPyBackend._patch_rect(Texture(np.zeros(shape), centre))
    h, w = shape
    corner = (round(centre[0] - w / 2.0), round(centre[1] - h / 2.0))
    require(rect == (corner[0], corner[1], w, h),
            f"patch {w}x{h} at {centre}: repaint rectangle {rect}")

sys.exit(0 if ok else 1)
