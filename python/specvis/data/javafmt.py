"""Java's ``String.format`` number formatting, so output files match byte for byte.

Python's ``%g`` is not Java's ``%g``, and the difference is visible in every output
file the Java writes:

=========  ===========  ==========
value      Java %.6g    Python %.6g
=========  ===========  ==========
0.8        0.800000     0.8
214.668    214.668      214.668
1e-05      1.00000e-05  1e-05
=========  ===========  ==========

Java also rounds differently. Its ``Formatter`` (JDK 8) takes the shortest decimal
digits that identify the double -- the ``Double.toString`` digits, which ``repr`` also
produces -- and rounds those **half-up**. Python rounds the exact binary value half to
even. So 0.125 at two places is 0.13 in Java and 0.12 in Python, and 2.675 (binary
2.67499...) is 2.68 in Java and 2.67 in Python.

Both functions here start from ``Decimal(repr(value))`` and round half-up, which is
the Java behaviour. ``tests/test_java_parity.py`` checks them against the Java itself.

One known limit: JDK 8's ``Double.toString`` is occasionally not the shortest
round-trip form (JDK-4511638, fixed in JDK 19) -- 2e23 is ``1.9999999999999998E23``
there -- and Java formats from those digits. That is not emulated. It affects values
like 2e23 and the smallest subnormal, nothing an experiment produces.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Context, Decimal

#: Enough digits for ``%.5f`` of the largest double (309 integer digits), which Java
#: prints in full. The default context's 28 digits would raise instead.
_WIDE = Context(prec=400, rounding=ROUND_HALF_UP)


def _special(value: float) -> str | None:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return None


def _sign(value: float) -> str:
    return "-" if math.copysign(1.0, value) < 0 else ""


def format_f(value: float, places: int) -> str:
    """Java ``%.<places>f``."""
    special = _special(value)
    if special is not None:
        return special
    d = Decimal(repr(abs(value))).quantize(Decimal(1).scaleb(-places), context=_WIDE)
    return _sign(value) + format(d, "f")


def format_g(value: float, precision: int = 6) -> str:
    """Java ``%.<precision>g``: significant digits, trailing zeros kept.

    Decimal notation when the rounded value's exponent is in ``[-4, precision)``,
    scientific (``d.ddddde-05``, at least two exponent digits) otherwise.
    """
    special = _special(value)
    if special is not None:
        return special
    precision = max(1, precision)
    sign = _sign(value)

    if value == 0.0:
        return sign + "0." + "0" * (precision - 1) if precision > 1 else sign + "0"

    d = Decimal(repr(abs(value)))
    rounded = d.quantize(Decimal(1).scaleb(d.adjusted() - precision + 1), context=_WIDE)
    exponent = rounded.adjusted()          # rounding 9.999995 up moves the exponent

    if -4 <= exponent < precision:
        decimals = precision - 1 - exponent
        return sign + format(rounded.quantize(Decimal(1).scaleb(-decimals), context=_WIDE), "f")

    mantissa = rounded.scaleb(-exponent, context=_WIDE).quantize(
        Decimal(1).scaleb(-(precision - 1)), context=_WIDE)
    exp_sign = "-" if exponent < 0 else "+"
    return f"{sign}{format(mantissa, 'f')}e{exp_sign}{abs(exponent):02d}"


def fmt(value: float) -> str:
    """The output files' cell format: ``%.6g``, or ``NA`` for a missing number."""
    return "NA" if math.isnan(value) else format_g(value, 6)
