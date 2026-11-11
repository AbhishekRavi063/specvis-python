"""Mapping between the 0-100 brightness setting and luminance in cd/m^2.

Ported from ``org.refrem.calibration.LuminanceCalibration``.

Specvis stores photometer readings at brightness 0/20/40/60/80/100 and fits a
degree-2 polynomial through them to produce a 101-entry lookup table. This reproduces
that forward fit and the inverse mapping, which is a linear interpolation over the
table rather than an analytic inversion -- the fit is not guaranteed monotonic, and
interpolating stays well behaved where inverting would not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

TABLE_SIZE = 101
FIT_DEGREE = 2
MEASUREMENT_POINTS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)


@dataclass(frozen=True)
class LuminanceCalibration:
    """A brightness/luminance relationship, in both directions.

    ``calibrated`` is false for :meth:`uncalibrated`, which assumes a linear
    relationship on an arbitrary scale. Contrast computed through an uncalibrated
    instance is **not physically meaningful**; the flag exists so every consumer can
    label its output honestly rather than quietly presenting nominal numbers as
    measured ones.
    """

    table: np.ndarray
    calibrated: bool
    source: str

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_measurements(cls, measured: Sequence[float], source: str) -> "LuminanceCalibration":
        """Fit from six photometer readings at brightness 0/20/40/60/80/100."""
        values = np.asarray(measured, dtype=float)
        if values.shape != (6,):
            raise ValueError("expected 6 measurements (0/20/40/60/80/100)")

        coefficients = np.polyfit(np.array(MEASUREMENT_POINTS), values, FIT_DEGREE)
        brightness = np.arange(TABLE_SIZE, dtype=float)
        table = np.clip(np.polyval(coefficients, brightness), 0.0, None)
        return cls(table, True, source)

    @classmethod
    def from_table(cls, table: Sequence[float], source: str,
                   calibrated: bool = True) -> "LuminanceCalibration":
        """Use a 101-entry table directly, e.g. one Specvis already fitted."""
        values = np.asarray(table, dtype=float)
        if values.shape != (TABLE_SIZE,):
            raise ValueError(f"expected {TABLE_SIZE} entries, got {values.shape[0]}")
        return cls(values, calibrated, source)

    @classmethod
    def uncalibrated(cls) -> "LuminanceCalibration":
        """Linear brightness, for development before a photometer is available."""
        return cls(
            np.arange(TABLE_SIZE, dtype=float),
            False,
            "UNCALIBRATED - linear brightness assumed, values are not cd/m^2",
        )

    @classmethod
    def assumed_display(cls, max_cd_m2: float = 300.0, gamma: float = 2.2
                        ) -> "LuminanceCalibration":
        """A typical display, **assumed, not measured**: luminance = peak x (b/100)^gamma.

        Lets a background be asked for in cd/m^2 (e.g. the Humphrey's 10 cd/m^2) before
        a photometer is available, and makes nominal contrasts roughly right instead of
        about half the real value, as the linear model gives on a gamma-2.2 screen. Still
        ``calibrated=False``: every result says the luminance was assumed.
        """
        if max_cd_m2 <= 0 or gamma <= 0:
            raise ValueError("peak luminance and gamma must be positive")
        b = np.arange(TABLE_SIZE, dtype=float) / 100.0
        return cls(max_cd_m2 * b ** gamma, False,
                   f"ASSUMED display (gamma {gamma:g}, peak {max_cd_m2:g} cd/m^2) - "
                   "not measured with a photometer")

    # -- forward ---------------------------------------------------------------

    def luminance_at(self, brightness):
        """cd/m^2 at a brightness in [0, 100], interpolated between table entries."""
        b = np.clip(np.asarray(brightness, dtype=float), 0.0, 100.0)
        return np.interp(b, np.arange(TABLE_SIZE, dtype=float), self.table)

    # -- inverse ---------------------------------------------------------------

    def brightness_for(self, target_luminance):
        """Brightness in [0, 100] producing the given luminance.

        Targets outside the achievable range clamp to the nearest endpoint, so callers
        that care about clipping must check :meth:`can_achieve` first.

        Scans the table segment by segment and takes the **first** bracketing segment,
        exactly as the Java does, rather than assuming the table is sorted: a degree-2
        fit through real photometer readings is not guaranteed monotonic, and a
        bisection search over a non-monotonic table would silently pick a different
        root. Vectorised over the 100 segments, so a whole patch costs 100 array
        operations rather than one Python loop per pixel.
        """
        target = np.asarray(target_luminance, dtype=float)
        flat = target.ravel()
        out = np.full(flat.shape, 100.0)
        assigned = np.zeros(flat.shape, dtype=bool)

        low, high = self.table[0], self.table[-1]

        below = flat <= low
        out[below] = 0.0
        assigned |= below

        above = (flat >= high) & ~assigned
        out[above] = 100.0
        assigned |= above

        for i in range(TABLE_SIZE - 1):
            if assigned.all():
                break
            a, b = self.table[i], self.table[i + 1]
            lo, hi = (a, b) if a <= b else (b, a)
            hit = (flat >= lo) & (flat <= hi) & ~assigned
            if not hit.any():
                continue
            out[hit] = float(i) if a == b else i + (flat[hit] - a) / (b - a)
            assigned |= hit

        if target.ndim == 0:
            return float(out[0])
        return out.reshape(target.shape)

    def can_achieve(self, luminance: float) -> bool:
        """Within the table's endpoints, as the Java checks it.

        Endpoints, not the table's extremes: a fit through a saturating panel can peak
        below brightness 100, and the Java then treats the overshoot as unachievable.
        """
        return float(self.table[0]) <= luminance <= float(self.table[-1])

    def max_achievable_contrast(self, mean_luminance: float) -> float:
        """The largest Michelson contrast that fits around this mean without clipping."""
        if mean_luminance <= 0:
            return 0.0
        amplitude = min(self.max_luminance - mean_luminance,
                        mean_luminance - self.min_luminance)
        return max(0.0, amplitude / mean_luminance)

    @property
    def min_luminance(self) -> float:
        return float(self.table[0])

    @property
    def max_luminance(self) -> float:
        return float(self.table[-1])
