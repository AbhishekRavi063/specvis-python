"""Where a stimulus goes on the screen, in degrees of visual angle.

Ported from the geometry in ``Functions.pxForOneDg`` and ``createListOfStimuli``, with
the proposal's own four-quadrant layout added.

Degrees of visual angle depend on viewing distance, which Specvis handles indirectly:
the operator states the visual field the screen subtends, and pixels-per-degree follows
from the resolution. That keeps the arithmetic simple but means **the stated field must
be re-derived whenever the chair moves**. :func:`degrees_subtended` does that from a
measured distance and screen size, so the number can be checked rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


def px_per_degree(resolution_px: int, field_degrees: float) -> float:
    """Pixels per degree, from the field the screen subtends."""
    if field_degrees <= 0:
        raise ValueError("the visual field must be positive")
    return resolution_px / field_degrees


def degrees_subtended(screen_size_mm: float, viewing_distance_mm: float) -> float:
    """The visual angle a screen dimension subtends at a given distance.

    Use this to check the field figure typed into the settings. A 30 cm wide screen at
    60 cm subtends 28.07 degrees, not 30.
    """
    if viewing_distance_mm <= 0:
        raise ValueError("viewing distance must be positive")
    return 2.0 * math.degrees(math.atan(screen_size_mm / 2.0 / viewing_distance_mm))


@dataclass(frozen=True)
class ScreenGeometry:
    """Resolution, the field it subtends, and where fixation sits on it."""

    width_px: int
    height_px: int
    field_degrees_x: float
    field_degrees_y: float
    fixation_offset_degrees_x: float = 0.0
    fixation_offset_degrees_y: float = 0.0

    @property
    def px_per_degree_x(self) -> float:
        return px_per_degree(self.width_px, self.field_degrees_x)

    @property
    def px_per_degree_y(self) -> float:
        return px_per_degree(self.height_px, self.field_degrees_y)

    @property
    def fixation_px(self) -> Tuple[float, float]:
        """Fixation in pixels, measured from the top-left corner."""
        return (
            self.width_px / 2.0 + self.fixation_offset_degrees_x * self.px_per_degree_x,
            self.height_px / 2.0 + self.fixation_offset_degrees_y * self.px_per_degree_y,
        )

    def to_px(self, degrees_x: float, degrees_y: float) -> Tuple[float, float]:
        """Screen pixels for a position in degrees relative to fixation.

        Positive y is **downward**, matching the screen's own coordinate system rather
        than the visual-field convention. The conversion to field coordinates happens
        once, at output, so there is only one place to get the sign wrong.
        """
        fx, fy = self.fixation_px
        return (fx + degrees_x * self.px_per_degree_x,
                fy + degrees_y * self.px_per_degree_y)

    def fits(self, degrees_x: float, degrees_y: float, diameter_degrees: float) -> bool:
        """Whether a patch of this size at this position is wholly on screen."""
        x, y = self.to_px(degrees_x, degrees_y)
        rx = diameter_degrees / 2.0 * self.px_per_degree_x
        ry = diameter_degrees / 2.0 * self.px_per_degree_y
        return (x - rx >= 0 and x + rx <= self.width_px
                and y - ry >= 0 and y + ry <= self.height_px)


@dataclass(frozen=True)
class PhysicalScreen:
    """Screen geometry from measured millimetres, with the sphericity correction.

    Ported from ``org.refrem.calibration.ScreenGeometry``.

    :class:`ScreenGeometry` above uses Specvis's own flat model -- one pixels-per-degree
    for the whole screen, derived from the field the operator states. That is an
    approximation, and it gets worse with eccentricity, because the screen is flat
    while the visual field is spherical.

    The local scale at eccentricity ``e`` is::

        px_per_degree(e) = px_per_mm * viewing_distance_mm * (pi / 180) / cos^2(e)

    How much it matters:

    ====  ========  ==================
    ecc.  factor    error if ignored
    ====  ========  ==================
    5     1.00765   +0.77%
    10    1.03109   +3.11%
    20    1.13247   +13.25%
    30    1.33333   +33.33%
    ====  ========  ==================

    At the proposal's +/-5 degree locations this is below measurement noise. For any
    wider-field mapping it is not: a patch sized with the centre scale at 20 degrees is
    13% wrong.
    """

    resolution_x: int
    resolution_y: int
    width_mm: float
    height_mm: float
    viewing_distance_mm: float

    def __post_init__(self) -> None:
        if self.width_mm <= 0 or self.height_mm <= 0:
            raise ValueError("screen dimensions must be positive")
        if self.viewing_distance_mm <= 0:
            raise ValueError("viewing distance must be positive")

    @property
    def px_per_mm_x(self) -> float:
        return self.resolution_x / self.width_mm

    @property
    def px_per_mm_y(self) -> float:
        return self.resolution_y / self.height_mm

    def local_px_per_degree_x(self, eccentricity_degrees: float = 0.0) -> float:
        """Horizontal scale at this eccentricity -- the right factor for sizing a patch."""
        c = math.cos(math.radians(eccentricity_degrees))
        return self.px_per_mm_x * self.viewing_distance_mm * (math.pi / 180.0) / (c * c)

    def local_px_per_degree_y(self, eccentricity_degrees: float = 0.0) -> float:
        c = math.cos(math.radians(eccentricity_degrees))
        return self.px_per_mm_y * self.viewing_distance_mm * (math.pi / 180.0) / (c * c)

    def sphericity_factor(self, eccentricity_degrees: float) -> float:
        """How much larger the local scale is than the centre scale: ``1 / cos^2(e)``."""
        c = math.cos(math.radians(eccentricity_degrees))
        return 1.0 / (c * c)

    @property
    def field_degrees_x(self) -> float:
        return degrees_subtended(self.width_mm, self.viewing_distance_mm)

    @property
    def field_degrees_y(self) -> float:
        return degrees_subtended(self.height_mm, self.viewing_distance_mm)

    def flat_geometry(self) -> "ScreenGeometry":
        """The flat approximation Specvis itself uses, derived from this screen.

        Provided so the two models can be compared on the same physical setup rather
        than argued about.
        """
        return ScreenGeometry(self.resolution_x, self.resolution_y,
                              self.field_degrees_x, self.field_degrees_y)

    def describe(self) -> str:
        return (f"{self.resolution_x}x{self.resolution_y} px, "
                f"{self.width_mm:.0f}x{self.height_mm:.0f} mm at "
                f"{self.viewing_distance_mm:.0f} mm -> "
                f"{self.field_degrees_x:.2f}x{self.field_degrees_y:.2f} degrees, "
                f"{self.local_px_per_degree_x():.2f} px/deg at centre")


@dataclass(frozen=True)
class Location:
    """One tested point in the visual field."""

    index: int
    degrees_x: float
    degrees_y: float

    @property
    def quadrant(self) -> str:
        """Visual-field quadrant, with positive y meaning *below* fixation on screen."""
        vertical = "upper" if self.degrees_y < 0 else "lower"
        horizontal = "left" if self.degrees_x < 0 else "right"
        return f"{vertical}-{horizontal}"


def quadrant_locations(eccentricity_degrees: float = 5.0) -> List[Location]:
    """The proposal's four locations: one per quadrant at +/- eccentricity.

    Section 6a, Fig. 1 -- four quadrants at +/-5 degrees horizontal and vertical from
    central fixation. Ordered upper-left, upper-right, lower-right, lower-left, so the
    index is stable across sessions and analysis can rely on it.
    """
    e = abs(eccentricity_degrees)
    return [
        Location(0, -e, -e),
        Location(1, +e, -e),
        Location(2, +e, +e),
        Location(3, -e, +e),
    ]


def grid_locations(spacing_degrees_x: float, spacing_degrees_y: float,
                   columns: int, rows: int) -> List[Location]:
    """A regular grid centred on fixation, as Specvis's own perimetry uses."""
    if columns < 1 or rows < 1:
        raise ValueError("a grid needs at least one column and one row")
    locations = []
    index = 0
    for row in range(rows):
        for column in range(columns):
            x = (column - (columns - 1) / 2.0) * spacing_degrees_x
            y = (row - (rows - 1) / 2.0) * spacing_degrees_y
            locations.append(Location(index, x, y))
            index += 1
    return locations
