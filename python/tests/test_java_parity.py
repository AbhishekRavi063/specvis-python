"""The Python port must produce what the Java produces, on the Java's own output.

``java_parity_cases.txt.gz`` was generated once from the original Java Specvis (in the
ReFreM workspace, not in this repository), by driving the real Java classes over
thousands of inputs and recording their outputs. No Java is needed to run this test.
This test recomputes every case with the Python and requires agreement, module by
module: the ladder, bisection, 4-2 staircase, luminance calibration, Gabor rendering,
reversal timing, geometry, perimetric dB, Java number formatting, and the
``trials.csv`` and ``gabor_presentations.csv`` files byte for byte.

The other tests check the Python against expectations written by hand. This one is
the only check against the Java itself, so it is the one that can catch a port that
is internally consistent and still wrong. It found six such divergences when it was
written.

If this test fails, the port is wrong -- not the fixture. The fixture is frozen: the Java
version is retired, so it is never regenerated.
"""

from __future__ import annotations

import base64
import gzip
import math
import pathlib
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from specvis.core.gabor import (  # noqa: E402
    GaborParameters, measure_contrast, minimum_renderable_contrast, render_brightness,
)
from specvis.core.geometry import PhysicalScreen  # noqa: E402
from specvis.core.ladder import ContrastLadder, brightness_vector, perimetric_db  # noqa: E402
from specvis.core.luminance import LuminanceCalibration  # noqa: E402
from specvis.core.reversal import ReversalPlan  # noqa: E402
from specvis.data.gabor_log import GaborPresentationLog  # noqa: E402
from specvis.data.javafmt import format_f, format_g  # noqa: E402
from specvis.data.trial_log import TrialLog  # noqa: E402
from specvis.threshold.bisection import (  # noqa: E402
    is_complete, max_answers, next_brightness_index, remaining_levels, surviving_index,
)
from specvis.threshold.staircase42 import (  # noqa: E402
    ContrastLocationThreshold, FourTwoStaircase,
)

FIXTURE = pathlib.Path(__file__).with_name("java_parity_cases.txt.gz")
DURATIONS_MS = (200, 1000, 1500, 2000, 4999, 5000, 5001, 10000)
# The 4-2 staircase's own outcome names, mapped to the LocationThreshold vocabulary.
JAVA_TERMINATION = {"REVERSALS_REACHED": "CONVERGED", "MAX_TRIALS_REACHED": "BUDGET_EXHAUSTED"}

results = defaultdict(lambda: [0, 0, []])      # check -> [cases, failures, examples]


def check(name: str, ok: bool, detail: str = "") -> None:
    r = results[name]
    r[0] += 1
    if not ok:
        r[1] += 1
        if len(r[2]) < 3:
            r[2].append(detail)


def floats(text: str):
    return [float(v) for v in text.split(",")] if text else []


def ints(text: str):
    return [int(v) for v in text.split(",")] if text else []


def same(a: float, b: float, rel: float = 1e-9, abs_: float = 1e-12) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def text(b64: str) -> str:
    return base64.b64decode(b64).decode("utf-8")


def first_difference(a: str, b: str) -> str:
    for i, (x, y) in enumerate(zip(a.splitlines(), b.splitlines())):
        if x != y:
            return f"line {i + 1}: py={x!r} java={y!r}"
    return f"lengths differ: py {len(a)} chars, java {len(b)}"


def ladder_case(f):
    start, end, n, log = int(f[1]), int(f[2]), int(f[3]), f[4] == "true"
    got, want = brightness_vector(start, end, n, spread_log=log), ints(f[5])
    check(f"ladder, {'logarithmic' if log else 'linear'}", got == want,
          f"{start}..{end} x{n}: py={got} java={want}")


def contrast_ladder_case(f):
    lad = ContrastLadder.from_vector(ints(f[1]), float(f[2]))
    check("contrast ladder: lowest and highest", same(lad.min_contrast, float(f[3]))
          and same(lad.max_michelson, float(f[4])))
    steps, want = lad.steps_db(), floats(f[5])
    check("contrast ladder: steps", len(steps) == len(want)
          and all(same(a, b, 1e-9, 1e-9) for a, b in zip(steps, want)))
    check("contrast ladder: mean step", same(lad.mean_step_db(), float(f[6]), 1e-9, 1e-9))
    check("contrast ladder: irregularity", same(lad.irregularity_db(), float(f[7]), 1e-9, 1e-9),
          f"py={lad.irregularity_db():.4f} java={float(f[7]):.4f}")


def bisect_case(f):
    length, answers = int(f[1]), ints(f[2])
    done = is_complete(answers, length)
    got = (-99 if done else next_brightness_index(answers, length),
           surviving_index(answers, length), "true" if done else "false",
           "".join("1" if r else "0" for r in remaining_levels(answers, length)),
           max_answers(length))
    want = (int(f[3]), int(f[4]), f[5], f[6], int(f[7]))
    check("bisection: every reachable state", got == want,
          f"len={length} answers={answers}: py={got} java={want}")


def calibration_case(f):
    m = floats(f[1])
    cal = LuminanceCalibration.from_measurements(m, "parity")
    table = [float(cal.luminance_at(b)) for b in range(101)]
    check("calibration: fitted table",
          all(same(a, b, 1e-9) for a, b in zip(table, floats(f[2]))))
    check("calibration: brightness_for",
          all(abs(cal.brightness_for(t) - b) < 1e-9 for t, b in zip(floats(f[3]), floats(f[4]))))
    bad = [t for t, a in zip(floats(f[3]), ints(f[5])) if int(cal.can_achieve(t)) != a]
    check("calibration: can_achieve", not bad, f"targets {bad[:2]}")
    check("calibration: luminance_at",
          all(same(float(cal.luminance_at(x)), y, 1e-9) for x, y in zip(floats(f[6]), floats(f[7]))))
    mean = float(f[8])
    check("calibration: minimum renderable contrast",
          same(minimum_renderable_contrast(cal, mean), float(f[9]), 1e-9))
    check("calibration: max achievable contrast",
          same(cal.max_achievable_contrast(mean), float(f[10]), 1e-9))


def gabor_case(f):
    cal = (LuminanceCalibration.uncalibrated() if f[1] == "U"
           else LuminanceCalibration.from_measurements(floats(f[1]), "parity"))
    diam, sigma, sf, orient, phase, c, mean = map(float, f[2:9])
    ppdx, ppdy, w, h = float(f[10]), float(f[11]), int(f[12]), int(f[13])
    p = GaborParameters(diam, sf, orient, phase, sigma, mean, c, f[9] == "true")
    pixels = np.round(render_brightness(p, cal, ppdx, ppdy) * 255.0).astype(int)
    want = np.array(ints(f[14])).reshape(h, w)
    # One 8-bit code on an occasional pixel is a rounding tie between numpy and
    # JavaFX's colour conversion; anything more is a real difference.
    off = int((pixels != want).sum()) if pixels.shape == want.shape else -1
    check("gabor: rendered pixels (within 1 code, at most 0.1% of pixels)",
          off >= 0 and int(np.abs(pixels - want).max()) <= 1 and off <= max(1, want.size // 1000),
          f"{off} of {want.size} pixels differ")
    check("gabor: measured contrast", same(measure_contrast(p, cal, ppdx, ppdy), float(f[15]), 1e-6))


def staircase_case(f):
    s = FourTwoStaircase(float(f[1]), float(f[2]), float(f[3]))
    responses, want = [r == "1" for r in f[4].split(",")], floats(f[5])
    got = []
    for r in responses:
        if s.finished:
            break
        got.append(s.current_contrast)
        s.record(r)
    check("4-2 staircase: level sequence", s.finished and len(got) == len(want)
          and all(same(a, b) for a, b in zip(got, want)))
    out, valid = s.outcome(), f[6] == "true"
    check("4-2 staircase: termination", out.termination.name == JAVA_TERMINATION.get(f[8], f[8]))
    check("4-2 staircase: validity", out.valid == valid)
    if valid:
        check("4-2 staircase: threshold", same(out.value, float(f[7])))


def location_threshold_case(f):
    minb, maxb = int(f[1]), int(f[2])
    mich, floor, start = float(f[3]), float(f[4]), float(f[6])
    # The start-contrast rule is not ported yet; the Python is handed the Java's choice.
    t = ContrastLocationThreshold(FourTwoStaircase(start, maxb / 100.0 * mich, minb / 100.0 * mich),
                                  "4-2", renderable_floor=floor)
    responses, want = [r == "1" for r in f[7].split(",")], floats(f[8])
    got = []
    for r in responses:
        if t.finished():
            break
        got.append(t.next_level() / mich * 100.0)
        t.record(r)
    check("4-2 via LocationThreshold: brightness sequence",
          len(got) == len(want) and all(same(a, b) for a, b in zip(got, want)))
    res, valid = t.result(), f[9] == "true"
    check("4-2 via LocationThreshold: termination", res.termination.name == f[11])
    check("4-2 via LocationThreshold: validity", res.valid == valid)
    if valid:
        check("4-2 via LocationThreshold: threshold", same(res.value / mich * 100.0, float(f[10])))


def reversal_case(f):
    rate, refresh = float(f[1]), float(f[2])
    p = ReversalPlan.for_rate(rate, refresh)
    check("reversal: frames per reversal", p.frames_per_reversal == int(f[3]))
    check("reversal: actual rates", same(p.actual_reversals_per_second, float(f[4]))
          and same(p.actual_cycles_per_second, float(f[5])))
    check("reversal: is_exact", p.is_exact() == (f[6] == "true"))
    got = [p.expected_reversals(d) for d in DURATIONS_MS]
    check("reversal: expected reversals", got == ints(f[7]),
          f"{rate}/s at {refresh} Hz: py={got} java={ints(f[7])}")


def geometry_case(f):
    s = PhysicalScreen(int(f[1]), int(f[2]), float(f[3]), float(f[4]), float(f[5]))
    ecc = float(f[6])
    check("geometry: local px/deg", same(s.local_px_per_degree_x(ecc), float(f[7]), 1e-12)
          and same(s.local_px_per_degree_y(ecc), float(f[8]), 1e-12))


def db_case(f):
    got, want = perimetric_db(float(f[1]), float(f[2])), float(f[3])
    check("perimetric dB, rounded", got == want, f"py={got} java={want}")


#: Doubles whose JDK 8 ``Double.toString`` digits are not the shortest round-trip digits
#: (JDK-4511638, fixed in JDK 19). Java formats from those digits and Python from
#: ``repr``, so these two differ by design and are not emulated: one is the smallest
#: subnormal, the other is 2e23, and no experiment value comes near either. Any *other*
#: formatting difference fails the test.
JDK8_DIGIT_ANOMALIES = {"4.9E-324", "1.9999999999999998E23"}


def format_case(f):
    v = float(f[1])
    got = (format_g(v, 6), format_f(v, 1), format_f(v, 0), format_f(v, 5))
    if f[1] in JDK8_DIGIT_ANOMALIES:
        check("Java number formatting: known JDK 8 digit anomalies, still present",
              got != tuple(f[2:6]), f"{f[1]} now matches; remove it from JDK8_DIGIT_ANOMALIES")
        return
    check("Java number formatting", got == tuple(f[2:6]), f"{f[1]}: py={got} java={tuple(f[2:6])}")


def trial_log_case(f):
    log = TrialLog()
    for row in filter(None, f[1].split("/")):
        b, ms, idx, x, y, bv, bi, lum, mich, seen, rt, extra, disp, win = row.split(";")
        log.record(b, int(ms), int(idx), float(x), float(y), int(bv), int(bi), float(lum),
                   float(mich), seen == "true", float(rt), int(extra), int(disp), int(win))
    with tempfile.TemporaryDirectory() as d:
        log.write_to(pathlib.Path(d))
        got = (pathlib.Path(d) / "trials.csv").read_bytes().decode("utf-8")
    want = text(f[2])
    check("trials.csv, byte for byte", got == want, first_difference(got, want))
    # Breaks are not in the file, so take the count from the Java's own summary.
    breaks = int(text(f[3]).split("rests taken: ")[1].split("\n")[0])
    for _ in range(breaks):
        log.record_break()
    got, want = log.summarise(), text(f[3])
    check("trial log summary text", got == want, first_difference(got, want))


def gabor_log_case(f):
    java_file = text(f[2])
    nanos = [int(line.split(",")[1]) for line in java_file.splitlines()
             if line and line[0].isdigit()]
    log = GaborPresentationLog()
    for row, ns in zip(filter(None, f[1].split("/")), nanos):
        idx, x, y, bv, req, ren, mean, orient, phase, fitted = row.split(";")
        log.record(int(idx), float(x), float(y), int(bv), float(req), float(ren), float(mean),
                   4.0, float(orient), float(phase), 0.333, 2.0, fitted == "true", host_nanos=ns)
    with tempfile.TemporaryDirectory() as d:
        log.write_to(pathlib.Path(d))
        got = (pathlib.Path(d) / "gabor_presentations.csv").read_bytes().decode("utf-8")
    check("gabor_presentations.csv, byte for byte", got == java_file,
          first_difference(got, java_file))
    check("gabor log summary text", log.summarise() == text(f[3]),
          first_difference(log.summarise(), text(f[3])))


def empty_logs_case(f):
    check("empty log summaries", TrialLog().summarise() == text(f[1])
          and GaborPresentationLog().summarise() == text(f[2]))


HANDLERS = {
    "LADDER": ladder_case, "CLADDER": contrast_ladder_case, "BISECT": bisect_case,
    "CAL": calibration_case, "GABOR": gabor_case, "STAIR": staircase_case,
    "CMLT": location_threshold_case, "REV": reversal_case, "GEOM": geometry_case,
    "DB": db_case, "FMT": format_case, "TRIALLOG": trial_log_case,
    "GABORLOG": gabor_log_case, "TRIALLOG_EMPTY": empty_logs_case,
}


def main() -> int:
    if not FIXTURE.exists():
        print(f"FAIL  {FIXTURE.name} is missing (it is part of the repository)")
        return 1
    seen_kinds = set()
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("|")
            seen_kinds.add(fields[0])
            HANDLERS[fields[0]](fields)

    missing = set(HANDLERS) - seen_kinds
    if missing:
        print(f"FAIL  the fixture has no cases for {sorted(missing)}")
        return 1

    width = max(len(name) for name in results)
    failed = 0
    for name in sorted(results):
        cases, failures, examples = results[name]
        print(f"  {'ok  ' if not failures else 'FAIL'} {name:{width}}  {cases - failures}/{cases}")
        for e in examples:
            print(f"         e.g. {e}")
        failed += failures
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
