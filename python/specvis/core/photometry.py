"""Photometer readings -> the screen's luminance table and its contrast range.

The ColorCAL tool (``tools/calibrate_photometer.py``) shows full-screen greys, reads the
luminance of each (cd/m^2), and hands the readings here. Nothing in this module talks to
hardware, so it can be tested.

Contrast is Michelson contrast around the background, as everywhere in the app, and
contrast dB is ``-20 log10(contrast)`` (larger = fainter).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from .luminance import TABLE_SIZE

LEVELS = 255            # 8-bit: grey levels 0..255; brightness % = level / 255 * 100


def coarse_levels():
    """The screen curve's grey levels: every 5 up to 60 (the dark end bends most, and the
    patch's dark stripes live there), then every 15 to white."""
    return sorted(set(range(0, 61, 5)) | set(range(60, 256, 15)) | {255})


def clean_readings(levels: Sequence[int], cd_m2: Sequence[float]):
    """Sorted by level, one value per level (the mean of repeats), never decreasing.

    A real screen's luminance rises with the grey level; a reading that falls is meter
    noise at the dark end, so it is held at the previous value.
    """
    by_level: Dict[int, List[float]] = {}
    for lv, cd in zip(levels, cd_m2):
        by_level.setdefault(int(lv), []).append(float(cd))
    lv = np.array(sorted(by_level), dtype=float)
    cd = np.array([float(np.mean(by_level[int(v)])) for v in lv])
    return lv, np.maximum.accumulate(cd)


def table_from_readings(levels: Sequence[int], cd_m2: Sequence[float]) -> List[float]:
    """The app's 101-entry table (cd/m^2 at brightness 0..100 %), interpolated linearly
    between the measured grey levels. Needs readings at both 0 and 255."""
    lv, cd = clean_readings(levels, cd_m2)
    if lv[0] != 0 or lv[-1] != LEVELS:
        raise ValueError("readings at grey level 0 and 255 are both needed")
    brightness = lv / LEVELS * 100.0
    return [float(v) for v in np.interp(np.arange(TABLE_SIZE, dtype=float), brightness, cd)]


def background_level(levels: Sequence[int], cd_m2: Sequence[float], target_cd_m2: float) -> int:
    """The 8-bit grey level whose (interpolated) luminance is nearest the target."""
    lv, cd = clean_readings(levels, cd_m2)
    all_levels = np.arange(LEVELS + 1, dtype=float)
    fine = np.interp(all_levels, lv, cd)
    return int(all_levels[int(np.argmin(np.abs(fine - target_cd_m2)))])


def summarise(levels: Sequence[int], cd_m2: Sequence[float], bg_level: int) -> Dict[str, float]:
    """Black, white, the background, and the contrast range in dB at that background.

    strongest: the largest Michelson contrast a patch can have around the background
    without clipping (limited by black or white). faintest: one grey step either side of
    the background, from the readings at bg_level - 1 and + 1 (measure those). middle:
    halfway between them in dB -- where the staircase seeds start without dithering.
    """
    lv, cd = clean_readings(levels, cd_m2)
    lum = dict(zip(lv.astype(int), cd))
    bg = lum.get(bg_level)
    if bg is None or bg <= 0:
        raise ValueError(f"no positive reading at the background level {bg_level}")
    black, white = float(cd[0]), float(cd[-1])
    strongest = max(0.0, min(white - bg, bg - black) / bg)
    steps = [abs(lum[n] - bg) for n in (bg_level - 1, bg_level + 1) if n in lum]
    step = max(steps) / bg if steps else float("nan")
    out = dict(black_cd_m2=black, white_cd_m2=white, background_level=bg_level,
               background_cd_m2=float(bg), strongest_contrast=strongest,
               grey_step_contrast=step)
    out["strongest_db"] = -20 * math.log10(strongest) if strongest > 0 else float("inf")
    out["grey_step_db"] = -20 * math.log10(step) if step and step > 0 else float("nan")
    out["middle_db_no_dither"] = (out["strongest_db"] + out["grey_step_db"]) / 2
    return out


def uniformity(readings: Dict[str, float]) -> Dict[str, float]:
    """The background read at several places: the largest difference from the centre."""
    centre = readings.get("centre")
    if not centre:
        raise ValueError("a 'centre' reading is needed")
    worst = max(abs(v - centre) / centre for v in readings.values())
    return dict(centre_cd_m2=centre, max_difference_pct=100.0 * worst)
