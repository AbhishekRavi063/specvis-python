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


#: Lab report 23 Nov (our suggestions): the readings at one place must agree within
#: this many % (more = the meter moved), and a corner may differ from the other corners
#: by at most this many % points (more = placed differently, or pressing on the screen).
SPREAD_LIMIT_PCT, ODD_CORNER_LIMIT_PCT = 2.0, 5.0


def uniformity_warnings(readings: Dict[str, float],
                        repeats: Dict[str, Sequence[float]] = None) -> List[str]:
    """Readings that look like a placement problem rather than the screen: repeats that
    disagree (the meter moved) and a corner unlike the other corners (further in from
    the edge, or the screen pressed: a white patch under the meter)."""
    notes = []
    for place, vals in (repeats or {}).items():
        vals = [float(v) for v in vals]
        if len(vals) > 1 and np.mean(vals) > 0:
            spread = 100.0 * (max(vals) - min(vals)) / float(np.mean(vals))
            if spread > SPREAD_LIMIT_PCT:
                notes.append(f"{place}: its {len(vals)} readings differ by {spread:.1f} % "
                             f"(over {SPREAD_LIMIT_PCT:g} %): the meter probably moved. "
                             "Repeat it, on the tripod, touching the screen lightly.")
    centre = readings.get("centre")
    corners = [p for p in readings if p.count("-") == 1 and p.split("-")[0] in ("top", "bottom")]
    if centre and len(corners) >= 3:
        pct = {p: 100.0 * (readings[p] - centre) / centre for p in corners}
        for p in corners:
            others = float(np.median([v for q, v in pct.items() if q != p]))
            if abs(pct[p] - others) > ODD_CORNER_LIMIT_PCT:
                notes.append(f"{p}: {pct[p]:+.1f} % against {others:+.1f} % at the other "
                             "corners: probably placed further in, or pressing on the "
                             "screen (a white patch). Repeat it.")
    return notes


# -- where the meter goes: the same relative places on any screen ------------------------

#: The nine places, in the order they are measured. Top and bottom are included: they
#: show the TN viewing-angle shading and the backlight's top/bottom difference.
PLACES = ("centre", "top", "bottom", "left", "right",
          "top-left", "top-right", "bottom-left", "bottom-right")

#: Lab, 23 Nov 2026: 30 mm in from the edges of the BenQ's 542.3 x 305 mm screen. As
#: fractions of the width / height from the centre, so other screens can be compared.
DEFAULT_FRACTIONS = (0.5 - 30.0 / 542.3, 0.5 - 30.0 / 305.0)


def place_fractions(fractions=DEFAULT_FRACTIONS, inset_mm: float = None,
                    width_mm: float = None, height_mm: float = None):
    """(fx, fy): how far the outer places sit from the centre, as fractions of the screen
    width and height. With inset_mm and the screen size: that many mm in from the edges."""
    if inset_mm is None:
        return float(fractions[0]), float(fractions[1])
    if not width_mm or not height_mm:
        raise ValueError("--inset-mm needs the screen's width and height in mm")
    fx, fy = 0.5 - inset_mm / width_mm, 0.5 - inset_mm / height_mm
    if fx <= 0 or fy <= 0:
        raise ValueError(f"an inset of {inset_mm} mm leaves no room on this screen")
    return fx, fy


def place_offset(place: str, fx: float, fy: float):
    """The place as (x, y) fractions of the screen from its centre; y up."""
    if place not in PLACES:
        raise ValueError(f"unknown place {place!r}; choose from {', '.join(PLACES)}")
    x = -fx if "left" in place else fx if "right" in place else 0.0
    y = fy if place.startswith("top") else -fy if place.startswith("bottom") else 0.0
    return x, y


def place_pixels(place: str, width_px: float, height_px: float, fx: float, fy: float):
    """The place in pixels from the screen centre (y up), as PsychoPy 'pix' units."""
    x, y = place_offset(place, fx, fy)
    return x * width_px, y * height_px


def ring_radius_mm(ring_mm: float, width_mm: float, height_mm: float,
                   places: Sequence[str], fx: float, fy: float, margin_mm: float = 2.0) -> float:
    """The target ring's radius: ring_mm / 2, made smaller if needed so the ring stays
    inside the screen (margin_mm from the edges) at every place."""
    room = []
    for p in places:
        x, y = place_offset(p, fx, fy)
        room.append(min(width_mm * (0.5 - abs(x)), height_mm * (0.5 - abs(y))) - margin_mm)
    return max(1.0, min([ring_mm / 2.0] + room))


def unsteady(vals: Sequence[float], limit_pct: float = None) -> float:
    """The readings' spread (max - min) in % of their mean if above the limit, else 0."""
    limit_pct = SPREAD_LIMIT_PCT if limit_pct is None else limit_pct
    vals = [float(v) for v in vals]
    if len(vals) < 2 or np.mean(vals) <= 0:
        return 0.0
    spread = 100.0 * (max(vals) - min(vals)) / float(np.mean(vals))
    return spread if spread > limit_pct else 0.0


def fit_screen_map(points_mm: Sequence[Sequence[float]], pct: Sequence[float]):
    """Least-squares p0 + p1 x^2 + p2 y^2 + p3 x + p4 y + p5 xy (x, y in mm from the
    centre) through the measured % differences, with p0 = 0: the centre is the reference
    (0 % by definition), so the map passes through it. Returns the 6 coefficients, or None
    with fewer than 6 places."""
    if len(pct) < 6:
        return None
    x, y = np.array([p[0] for p in points_mm], float), np.array([p[1] for p in points_mm], float)
    a = np.column_stack([x * x, y * y, x, y, x * y])
    coef, *_ = np.linalg.lstsq(a, np.asarray(pct, float), rcond=None)
    return [0.0] + [float(c) for c in coef]


def screen_map_at(coef, x_mm, y_mm):
    x, y = np.asarray(x_mm, float), np.asarray(y_mm, float)
    return coef[0] + coef[1] * x * x + coef[2] * y * y + coef[3] * x + coef[4] * y + coef[5] * x * y


def evenness_colour(pct_rounded: float) -> str:
    """green above -5 %, orange -5 to -10 %, red at -10 % or less (lab request)."""
    return "green" if pct_rounded > -5 else "orange" if pct_rounded > -10 else "red"
