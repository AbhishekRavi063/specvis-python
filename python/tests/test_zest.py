"""ZEST for one location: it finds thresholds, stops, and never hides an edge."""

from __future__ import annotations

import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.threshold.base import Termination  # noqa: E402
from specvis.threshold.zest import ZestLocationThreshold, ZestSettings  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


def observer(true_log, rng, lapse=0.03, guess=0.03):
    """Sees with a Weibull of slope 3.5 on log contrast; None = blind."""
    def seen(contrast: float) -> bool:
        if true_log is None:
            return rng.random() < guess
        p = guess + (1 - guess - lapse) * (1 - math.exp(-10 ** (3.5 * (math.log10(contrast) - true_log))))
        return rng.random() < p
    return seen


def run(z: ZestLocationThreshold, see) -> ZestLocationThreshold:
    while not z.finished():
        z.record(see(z.next_level()))
    return z


rng = random.Random(7)

print("1. finds the threshold at every level")
for true_c in (0.02, 0.05, 0.15, 0.40):
    t = math.log10(true_c)
    zs = [run(ZestLocationThreshold(), observer(t, rng)) for _ in range(200)]
    err = statistics.median(abs(z.estimate_log - t) for z in zs)
    valid = sum(z.result().valid for z in zs)
    # The stopping rule is a posterior SD of 0.15 log10, so that is the precision asked for.
    require(err < 0.15, f"true {true_c:.2f}: median error {err:.3f} log10 (< 0.15)")
    require(valid >= 190, f"true {true_c:.2f}: {valid}/200 valid")

print("2. stays within the trial limits")
s = ZestSettings()
trials = [run(ZestLocationThreshold(), observer(math.log10(0.05), rng)).trials
          for _ in range(200)]
require(min(trials) >= s.min_trials and max(trials) <= s.max_trials,
        f"trials {min(trials)}-{max(trials)} within {s.min_trials}-{s.max_trials}")

print("3. quick blind rule")
zs = [run(ZestLocationThreshold(), observer(None, rng)) for _ in range(200)]
blind = [z for z in zs if z.result().termination is Termination.CEILING_NOT_SEEN]
require(len(blind) >= 180, f"never-seen point reported blind in {len(blind)}/200")
require(all(not z.result().valid and z.result().value == -1.0 for z in blind),
        "blind is invalid, value -1 (Specvis's 'no threshold')")
require(statistics.mean(z.trials for z in zs) < 6.0,
        f"and quickly: {statistics.mean(z.trials for z in zs):.1f} trials on average")
z = ZestLocationThreshold()
z.next_level(); z.record(False)                       # missed at the 50% start
require(z.probing and abs(z.next_level() - 1.0) < 1e-9, "a miss at 50% triggers a 100% check")
z.record(True)
require(not z.probing and not z.finished(), "seen at 100%: the normal search resumes")

print("4. an observer who sees everything is a FLOOR failure, not a threshold")
z = run(ZestLocationThreshold(), lambda c: True)
require(z.result().termination is Termination.FLOOR_ALWAYS_SEEN and not z.result().valid,
        "always seen -> FLOOR_ALWAYS_SEEN, invalid (no floor blindness)")

print("5. a good starting guess saves trials")
t = math.log10(0.05)
flat = statistics.mean(run(ZestLocationThreshold(), observer(t, rng)).trials for _ in range(200))
guided = []
for _ in range(200):
    z = ZestLocationThreshold()
    z.set_prior(t)
    guided.append(run(z, observer(t, rng)).trials)
require(statistics.mean(guided) < flat - 1.0,
        f"{flat:.1f} trials without a guess, {statistics.mean(guided):.1f} with one")
try:
    z = ZestLocationThreshold()
    z.next_level(); z.record(True)
    z.set_prior(t)
    require(False, "a prior cannot be set after the first trial")
except RuntimeError:
    require(True, "a prior cannot be set after the first trial")

print("6. never asks for a contrast the display cannot draw")
z = ZestLocationThreshold(renderable_floor=0.0078)
levels = []
while not z.finished():
    c = z.next_level()
    levels.append(c)
    z.record(True)
require(min(levels) >= 0.0078 - 1e-12, f"lowest level {min(levels):.4f} >= renderable floor")

sys.exit(0 if ok else 1)
