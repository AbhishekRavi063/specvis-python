"""The ladder of stimulus levels a threshold search walks.

Ported from ``createBrightnessVector`` / ``Functions.logspace`` and
``org.specvis.logic.GaborContrastLadder``.

.. note::
   **Upstream truncates.** ``Functions.logspace`` calls ``round(d, 0)`` and discards
   the result, and the caller then does ``(int) d``. Every rung is therefore biased
   dimmer, and the intended uniform 2.50 dB spacing of the default ladder becomes
   2.28-2.64 dB: a 0.36 dB spread of step sizes where the settings imply none (the
   Java's own "step irregularity" figure for it is 0.22 dB, the worst deviation from
   the mean step). Linear ladders are unaffected: ``linspace`` does round.

   :func:`brightness_vector` reproduces that faithfully by default, because a port
   that silently fixes it would not reproduce Java thresholds. Pass
   ``rounding="nearest"`` for the ladder the settings actually describe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import List, Literal, Sequence

Rounding = Literal["specvis", "nearest"]


def round_half_up(value: float, places: int) -> float:
    """``Functions.round``: ``new BigDecimal(value).setScale(places, HALF_UP)``.

    Not Python's :func:`round`, which rounds exact ties to even: 10.625 is 10.62 there
    and 10.63 in the Java. ``Decimal(value)`` is the exact binary value, as
    ``new BigDecimal(double)`` is, so the two agree on every input.
    """
    return float(Decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def brightness_vector(start: int, stop: int, length: int, spread_log: bool = True,
                      rounding: Rounding = "specvis") -> List[int]:
    """The 0-100 brightness levels, dimmest first.

    :param rounding: ``"specvis"`` reproduces the procedures exactly: a logarithmic
                     ladder is **truncated** (the ``logspace`` bug), a linear one is
                     rounded half-up by ``linspace`` and then truncated, which leaves it
                     rounded. ``"nearest"`` gives the evenly spaced ladder the UI implies.
    """
    if length < 2:
        raise ValueError("a ladder needs at least two rungs")
    if start <= 0 or stop <= 0:
        raise ValueError("brightness must be positive for a logarithmic ladder")

    n = length - 1
    if spread_log:
        exact = [start * math.pow(stop / start, i / n) for i in range(length)]
    else:
        # linspace's own arithmetic: start + i*step for all but the last, then stop itself.
        step = (stop - start) / n
        exact = [start + i * step for i in range(n)] + [float(stop)]

    if rounding == "nearest":
        return [int(round(v)) for v in exact]
    if spread_log:
        return [int(v) for v in exact]
    return [int(round_half_up(v, 0)) for v in exact]


def contrast_db(michelson: float) -> float:
    """Contrast attenuation in dB: ``-20*log10(contrast)``. Higher means fainter.

    .. warning::
       This is **not** clinical perimetric dB. Perimetry uses ``10*log10`` of a
       *luminance* ratio; this is ``20*log10`` of a *contrast* ratio. The two share a
       unit name and nothing else, and a number from one is not comparable with a
       number from the other.
    """
    if michelson <= 0:
        raise ValueError("contrast must be positive")
    return -20.0 * math.log10(michelson)


def db_to_contrast(db: float) -> float:
    """Inverse of :func:`contrast_db`."""
    return 10.0 ** (-db / 20.0)


@dataclass(frozen=True)
class ContrastLadder:
    """A brightness vector reinterpreted as Michelson contrasts for a Gabor.

    Specvis's brightness axis is repurposed: each rung's fraction of maximum
    brightness scales the maximum carrier contrast.
    """

    brightness: List[int]
    max_contrast: float
    max_brightness: int

    @classmethod
    def from_vector(cls, brightness: Sequence[int], max_contrast: float,
                    max_brightness: int = 100) -> "ContrastLadder":
        if not 0 < max_contrast <= 1:
            raise ValueError("max Michelson contrast must be in (0, 1]")
        return cls(list(brightness), float(max_contrast), int(max_brightness))

    @property
    def contrasts(self) -> List[float]:
        return [b / self.max_brightness * self.max_contrast for b in self.brightness]

    @property
    def min_contrast(self) -> float:
        return min(self.contrasts)

    @property
    def max_michelson(self) -> float:
        return max(self.contrasts)

    def steps_db(self) -> List[float]:
        """Spacing between adjacent rungs, in contrast dB, dimmest first.

        Signed as the Java signs it, so a ladder that is not ascending shows up as a
        negative step rather than being hidden by an absolute value.
        """
        c = self.contrasts
        return [contrast_db(c[i]) - contrast_db(c[i + 1]) for i in range(len(c) - 1)]

    def mean_step_db(self) -> float:
        steps = self.steps_db()
        return sum(steps) / len(steps) if steps else math.nan

    def irregularity_db(self) -> float:
        """``stepIrregularityDb``: the largest deviation of any step from the mean step.

        This is the number the Java writes into the session metadata -- 0.22 dB for
        the default 9-rung ladder. :meth:`step_range_db` is the different number
        (0.36 dB there) quoted in the analysis documents.
        """
        steps = self.steps_db()
        mean = self.mean_step_db()
        return max((abs(s - mean) for s in steps), default=0.0)

    def step_range_db(self) -> float:
        """Largest minus smallest step: 2.28-2.64 dB, so 0.36 dB, for the default ladder."""
        steps = self.steps_db()
        return max(steps) - min(steps) if steps else 0.0

    def describe(self) -> str:
        c = self.contrasts
        lines = [
            f"rungs: {len(c)}, from {min(c):.5f} to {max(c):.5f} Michelson",
            f"mean step: {self.mean_step_db():.2f} dB contrast",
            f"step irregularity: {self.irregularity_db():.2f} dB",
            "contrast dB = -20*log10(Michelson); NOT clinical perimetric dB",
        ]
        for i, (b, m) in enumerate(zip(self.brightness, c)):
            lines.append(f"  {i:2d}: {b:3d} -> {m:.5f}, {contrast_db(m):.2f} dB")
        return "\n".join(lines)


def perimetric_db(max_luminance: float, stimulus_luminance: float,
                  places: int = 2) -> float:
    """Specvis's own dB: ``10*log10(maxLuminance / stimulusLuminance)``.

    Ported from ``Functions.decibelsValue``. This is the number written to
    ``session_data.txt`` and drawn on the results-map colourbar, so it is needed for
    output parity with the Java.

    .. warning::
       **This is a different quantity from :func:`contrast_db`**, despite both being
       called dB. This one is ``10*log10`` of a *luminance* ratio; that one is
       ``20*log10`` of a *contrast* ratio. A number from one is not comparable with a
       number from the other.

    .. note::
       The Java signature takes a ``backgroundLuminance`` argument **and never uses
       it**. It is omitted here rather than accepted and ignored. The consequence is
       worth knowing: the value is a ratio against *the ladder's own top rung*, not a
       contrast against the background, so changing the maximum brightness setting
       changes every dB in the file with no change in the participant. It is not
       portable between settings, displays or sites.
    """
    if stimulus_luminance <= 0:
        raise ValueError("stimulus luminance must be positive")
    if max_luminance <= 0:
        raise ValueError("maximum luminance must be positive")
    return round_half_up(10.0 * math.log10(max_luminance / stimulus_luminance), places)
