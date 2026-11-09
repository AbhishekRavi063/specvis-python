"""One row per Gabor presentation: what was asked for, and what was rendered.

Ported from ``org.specvis.logic.GaborPresentationLog``; writes the same
``gabor_presentations.csv``. ``tests/test_java_parity.py`` checks the file against the
Java's.

``requested_michelson`` is the carrier contrast the staircase asked for.
``rendered_michelson`` is measured from the rendered patch through the luminance
calibration. They differ mainly because a Gabor's extrema sit where the Gaussian
envelope is below 1, and near the display floor quantisation adds to that. Neither is
a photometric measurement of the screen.
"""

from __future__ import annotations

import math
import pathlib
import time
from dataclasses import dataclass
from typing import List, Optional

from .javafmt import fmt, format_f

HEADER = (
    "# requested_michelson is the CARRIER contrast the staircase asked for.\n"
    "# rendered_michelson is measured from the rendered image and converted through the "
    "luminance calibration.\n"
    "# They differ mainly because a Gabor's extrema sit where the Gaussian envelope is "
    "below 1; near the display floor,\n"
    "# quantisation adds to the difference. Neither is a photometric measurement of the "
    "screen.\n"
    "# luminance_scale_fitted says only whether a fitted Specvis luminance scale was "
    "available. Specvis does not\n"
    "# record whether that scale came from a photometer, so TRUE is not a calibration "
    "claim: treat every\n"
    "# contrast and luminance value here as nominal unless your own records say "
    "otherwise.\n"
)

COLUMNS = (
    "presentation", "host_nanos", "stimulus_index", "x_deg", "y_deg", "brightness_value",
    "requested_michelson", "rendered_michelson", "rendered_minus_requested",
    "requested_contrast_db", "rendered_contrast_db", "mean_luminance_cdm2",
    "spatial_frequency_cpd", "orientation_deg", "phase_deg", "sigma_deg", "diameter_deg",
    "luminance_scale_fitted",
)


def _contrast_db(michelson: float) -> float:
    """``GaborContrastLadder.contrastToDb``: NaN, not an error, for a zero contrast."""
    return math.nan if michelson <= 0 else -20.0 * math.log10(michelson)


@dataclass(frozen=True)
class Entry:
    host_nanos: int
    stimulus_index: int
    x_deg: float
    y_deg: float
    brightness_value: int
    requested_michelson: float
    rendered_michelson: float
    mean_luminance: float
    spatial_frequency_cpd: float
    orientation_degrees: float
    phase_degrees: float
    sigma_degrees: float
    diameter_degrees: float
    scale_fitted: bool

    def row(self, presentation: int) -> str:
        return ",".join((
            str(presentation), str(self.host_nanos), str(self.stimulus_index),
            fmt(self.x_deg), fmt(self.y_deg), str(self.brightness_value),
            fmt(self.requested_michelson), fmt(self.rendered_michelson),
            fmt(self.rendered_michelson - self.requested_michelson),
            fmt(_contrast_db(self.requested_michelson)),
            fmt(_contrast_db(self.rendered_michelson)),
            fmt(self.mean_luminance), fmt(self.spatial_frequency_cpd),
            fmt(self.orientation_degrees), fmt(self.phase_degrees), fmt(self.sigma_degrees),
            fmt(self.diameter_degrees), "TRUE" if self.scale_fitted else "FALSE",
        ))


class GaborPresentationLog:
    """The Gabor log for one session."""

    def __init__(self) -> None:
        self._entries: List[Entry] = []

    def clear(self) -> None:
        self._entries.clear()

    def record(self, stimulus_index: int, x_deg: float, y_deg: float, brightness_value: int,
               requested_michelson: float, rendered_michelson: float, mean_luminance: float,
               spatial_frequency_cpd: float, orientation_degrees: float, phase_degrees: float,
               sigma_degrees: float, diameter_degrees: float, scale_fitted: bool,
               host_nanos: Optional[int] = None) -> None:
        """``host_nanos`` defaults to a monotonic clock, as ``System.nanoTime`` is."""
        self._entries.append(Entry(
            time.monotonic_ns() if host_nanos is None else host_nanos,
            stimulus_index, x_deg, y_deg, int(brightness_value), requested_michelson,
            rendered_michelson, mean_luminance, spatial_frequency_cpd, orientation_degrees,
            phase_degrees, sigma_degrees, diameter_degrees, scale_fitted,
        ))

    def __len__(self) -> int:
        return len(self._entries)

    def text(self) -> str:
        """The whole of ``gabor_presentations.csv``."""
        lines = [",".join(COLUMNS)] + [e.row(i + 1) for i, e in enumerate(self._entries)]
        return HEADER + "\n".join(lines) + "\n"

    def write_to(self, directory: pathlib.Path) -> None:
        """Write ``gabor_presentations.csv``. Does nothing when empty."""
        if not self._entries:
            return
        path = pathlib.Path(directory) / "gabor_presentations.csv"
        with path.open("w", newline="") as handle:
            handle.write(self.text())

    def summarise(self) -> str:
        if not self._entries:
            return "Gabor presentations: none recorded\n"
        diffs = [e.rendered_michelson - e.requested_michelson for e in self._entries]
        worst = max([0.0] + [abs(d) for d in diffs])
        return (
            f"Gabor presentations: {len(self._entries)} recorded in gabor_presentations.csv\n"
            f"  lowest carrier contrast requested: "
            f"{format_f(min(e.requested_michelson for e in self._entries), 5)}\n"
            f"  lowest Michelson contrast rendered: "
            f"{format_f(min(e.rendered_michelson for e in self._entries), 5)}\n"
            f"  mean rendered minus requested: {format_f(sum(diffs) / len(diffs), 5)}, "
            f"worst {format_f(worst, 5)}\n"
            "  (difference is dominated by Gaussian envelope geometry, not error)\n"
        )
