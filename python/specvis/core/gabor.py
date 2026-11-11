"""Gabor patch generation.

Ported from ``org.refrem.stimulus.GaborPatch``. Vectorised with numpy, but the
arithmetic is the same arithmetic in the same order, and
``tests/test_java_parity.py`` compares every rendered 8-bit pixel and the measured
contrast against the Java's own output.

A Gabor is a sinusoidal carrier under a Gaussian envelope::

    L(x, y) = L_mean * (1 + C * carrier(x, y) * envelope(x, y))

Two details in the original matter more than they look:

**Odd pixel dimensions.** With an even width the centre falls between pixels and the
carrier peak is never sampled, so a cosine-phase patch renders at measurably less than
its nominal contrast. That looks like a calibration error and is not one.

**DC correction.** A sine carrier is odd about the centre so its Gaussian-weighted sum
cancels, but a cosine carrier is even and does not. The residual scales as
``exp(-2*pi^2*f^2*sigma^2)`` -- 0.03% of mean at 4 c/deg, but +1.46% at 1 c/deg. A
patch whose mean differs from the background is partly a luminance increment and can
be detected without resolving the grating at all, so the DC is removed rather than
tolerated. Removing it changes peak modulation by at most ``|mean|``, so the achieved
Michelson contrast is no longer exactly the requested value -- which is why
:func:`measure_contrast` exists and why both numbers are logged per presentation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .luminance import LuminanceCalibration


@dataclass(frozen=True)
class GaborParameters:
    """Everything that defines a patch, independent of where it is shown."""

    diameter_degrees: float
    spatial_frequency_cpd: float
    orientation_degrees: float
    phase_degrees: float
    sigma_degrees: float
    mean_luminance: float
    michelson_contrast: float
    hard_aperture: bool = False

    def __post_init__(self) -> None:
        if self.diameter_degrees <= 0:
            raise ValueError("diameter must be positive")
        if self.sigma_degrees <= 0:
            raise ValueError("sigma must be positive")
        if not 0 <= self.michelson_contrast <= 1:
            raise ValueError("Michelson contrast must be in [0, 1]")
        if self.mean_luminance <= 0:
            raise ValueError("mean luminance must be positive")

    def with_contrast(self, contrast: float) -> "GaborParameters":
        """The same patch at a different contrast, for a staircase."""
        return GaborParameters(
            self.diameter_degrees, self.spatial_frequency_cpd, self.orientation_degrees,
            self.phase_degrees, self.sigma_degrees, self.mean_luminance,
            float(contrast), self.hard_aperture,
        )

    def reversed_phase(self) -> "GaborParameters":
        """The opposite polarity, for pattern reversal.

        Both polarities have identical mean luminance -- that is the point of a
        reversal, and it is what makes the SSVEP response specific to the pattern
        rather than to a luminance flicker.
        """
        return GaborParameters(
            self.diameter_degrees, self.spatial_frequency_cpd, self.orientation_degrees,
            (self.phase_degrees + 180.0) % 360.0, self.sigma_degrees,
            self.mean_luminance, self.michelson_contrast, self.hard_aperture,
        )


def _to_odd(n: int) -> int:
    """Nearest odd size at least 1, so the patch has a true centre pixel."""
    n = max(1, n)
    return n if n % 2 == 1 else n + 1


def patch_size_px(p: GaborParameters, px_per_degree_x: float,
                  px_per_degree_y: float) -> tuple[int, int]:
    """Rendered size in pixels: ceil of the requested extent, forced odd."""
    w = _to_odd(math.ceil(p.diameter_degrees * px_per_degree_x))
    h = _to_odd(math.ceil(p.diameter_degrees * px_per_degree_y))
    if w < 1 or h < 1:
        raise ValueError(f"patch is smaller than one pixel: {w}x{h}")
    return w, h


def luminance_field(p: GaborParameters, px_per_degree_x: float,
                    px_per_degree_y: float) -> np.ndarray:
    """The patch as luminance in cd/m^2, before any display calibration.

    Two passes, as in the original: build the modulation and its mean, then subtract
    the mean and scale.
    """
    width, height = patch_size_px(p, px_per_degree_x, px_per_degree_y)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    theta = math.radians(p.orientation_degrees)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    phi = math.radians(p.phase_degrees)
    two_pi_f = 2.0 * math.pi * p.spatial_frequency_cpd
    two_sigma_sq = 2.0 * p.sigma_degrees * p.sigma_degrees
    aperture_radius = p.diameter_degrees / 2.0

    x_deg = (np.arange(width, dtype=float) - cx) / px_per_degree_x
    y_deg = (np.arange(height, dtype=float) - cy) / px_per_degree_y
    xx, yy = np.meshgrid(x_deg, y_deg)

    r_sq = xx * xx + yy * yy
    envelope = np.exp(-r_sq / two_sigma_sq)
    if p.hard_aperture:
        envelope = np.where(np.sqrt(r_sq) > aperture_radius, 0.0, envelope)

    carrier = np.sin(two_pi_f * (xx * sin_t + yy * cos_t) + phi)

    modulation = carrier * envelope
    modulation = modulation - modulation.mean()          # pass 2: DC removed

    return p.mean_luminance * (1.0 + p.michelson_contrast * modulation)


def render_brightness(p: GaborParameters, calibration: LuminanceCalibration,
                      px_per_degree_x: float, px_per_degree_y: float) -> np.ndarray:
    """The patch as 0-1 display brightness, clipped, ready for a texture."""
    field = luminance_field(p, px_per_degree_x, px_per_degree_y)
    brightness = calibration.brightness_for(field) / 100.0
    return np.clip(brightness, 0.0, 1.0)


def dither_8bit(brightness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noisy-bit dithering (Allard & Faubert 2008, Behav Res Methods 40:735).

    Each pixel is rounded up or down to a neighbouring 8-bit level at random, with the
    probability that keeps its *average* exact. Contrasts finer than one grey step --
    which a dark background needs -- are then shown correctly on average, the noise
    being far too fine to see. Pixels already on an 8-bit level (the background, and the
    patch's own edge) are left exactly as they are, so no noisy square appears.
    """
    levels = np.asarray(brightness, dtype=float) * 255.0
    exact = np.isclose(levels, np.round(levels), atol=0.01)
    noisy = np.floor(levels + rng.random(levels.shape))
    out = np.where(exact, np.round(levels), noisy)
    return np.clip(out, 0.0, 255.0) / 255.0


def snap_background(calibration: LuminanceCalibration, target_luminance: float
                    ) -> Tuple[float, float]:
    """(mean luminance, brightness 0-1) of the 8-bit grey level nearest the target.

    The patch's mean must equal the background exactly, or a dithered patch would show
    a faint square; so the background is put on a real 8-bit level first.
    """
    level = round(float(calibration.brightness_for(target_luminance)) / 100.0 * 255.0)
    level = min(255, max(0, level))
    brightness = level / 255.0
    return float(calibration.luminance_at(brightness * 100.0)), brightness


def nominal_contrast(p: GaborParameters, calibration: LuminanceCalibration,
                     px_per_degree_x: float, px_per_degree_y: float) -> float:
    """Michelson contrast of the patch as drawn *before* 8-bit rounding: what a
    dithered patch shows on average."""
    brightness = render_brightness(p, calibration, px_per_degree_x, px_per_degree_y)
    luminance = calibration.luminance_at(brightness * 100.0)
    l_min, l_max = float(luminance.min()), float(luminance.max())
    if l_min + l_max <= 0:
        return 0.0
    return (l_max - l_min) / (l_max + l_min)


def measure_contrast(p: GaborParameters, calibration: LuminanceCalibration,
                     px_per_degree_x: float, px_per_degree_y: float) -> float:
    """Michelson contrast of what the display will actually show.

    Round-trips through the calibration and the 8-bit quantisation, so this is the
    number that describes the stimulus -- not the one that was requested. Log both.
    """
    brightness = render_brightness(p, calibration, px_per_degree_x, px_per_degree_y)
    quantised = np.round(brightness * 255.0) / 255.0
    luminance = calibration.luminance_at(quantised * 100.0)
    l_min, l_max = float(luminance.min()), float(luminance.max())
    if l_min + l_max <= 0:
        return 0.0
    return (l_max - l_min) / (l_max + l_min)


def minimum_renderable_contrast(calibration: LuminanceCalibration,
                                mean_luminance: float) -> float:
    """The faintest patch an 8-bit display can show at this mean luminance.

    One code step either side of the mean. Asking a staircase for anything below this
    produces a uniform grey field that is logged as a stimulus, which is how a
    "threshold" gets recorded for a patch that was never shown.
    """
    mean_brightness = calibration.brightness_for(mean_luminance)
    step = 100.0 / 255.0
    up = calibration.luminance_at(min(100.0, mean_brightness + step))
    down = calibration.luminance_at(max(0.0, mean_brightness - step))
    amplitude = max(abs(float(up) - mean_luminance), abs(mean_luminance - float(down)))
    return amplitude / mean_luminance if mean_luminance > 0 else 1.0
