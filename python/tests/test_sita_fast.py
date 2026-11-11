"""SITA Fast on the ZEST engine, the 10 cd/m^2 background with dithering, and the
hill-of-vision normal table -- against simulated observers, no window open."""

from __future__ import annotations

import math
import pathlib
import random
import statistics
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from specvis.core.gabor import (GaborParameters, dither_8bit, nominal_contrast,  # noqa: E402
                                render_brightness, snap_background)
from specvis.core.luminance import LuminanceCalibration  # noqa: E402
from specvis.procedure.perimetry import (BLIND, WEAK, ZEST_FULL, PerimetrySettings,  # noqa: E402
                                         background_for, categorise, default_calibration,
                                         write_outputs)
from specvis.procedure.prior_table import build_prior_table  # noqa: E402
from specvis.threshold.zest import ZestLocationThreshold, ZestSettings  # noqa: E402
from sim_observer import healthy, right_hemianopia, run  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


print("1. background 10 cd/m2 on an assumed display, dithered patches")
s = PerimetrySettings()
cal = default_calibration(s)
mean, b = background_for(s, cal)
require(not cal.calibrated and "ASSUMED" in cal.source, "no photometer: the display is assumed")
require(abs(mean - 10.0) < 0.6 and abs(b * 255 - round(b * 255)) < 1e-9,
        f"background {mean:.2f} cd/m2 on a real grey level (RGB {round(b * 255)})")
p = GaborParameters(2.0, 4.0, 45.0, 0.0, 2.0 / 6.0, mean, 0.01)
raw = render_brightness(p, cal, 60.0, 60.0)
rng = np.random.default_rng(3)
dithered = np.mean([dither_8bit(raw, rng) for _ in range(400)], axis=0)
require(float(np.abs(dithered - raw).max()) < 0.003,
        "dithering is right on average at every pixel")
one = dither_8bit(raw, rng)
require(np.allclose(one * 255, np.round(one * 255)), "every dithered pixel is a real 8-bit level")
require(abs(one[0, 0] - b) < 1e-9 and abs(one[-1, -1] - b) < 1e-9,
        "the patch's corners equal the background exactly (no visible square)")
require(abs(nominal_contrast(p, cal, 60.0, 60.0) - 0.01) < 0.002,
        "a 1% patch is drawn at 1% on average, though one grey step here is ~4%")

print("2. the ZEST engine's SITA Fast rules")
fast = ZestSettings(fast_stop=True)
z = ZestLocationThreshold(fast)
z.set_prior(math.log10(0.03), 0.25, 0.05, 0.2, sd_better_log=0.12,
            normal_log=math.log10(0.03))
for _ in range(3):
    z.next_level()
    z.record(True)
    if z.finished():
        break
require(z.finished() and z.result().valid and z.early_stop,
        f"seen at the normal level: stops without needing a miss ({z.trials} patches)")
z = ZestLocationThreshold(fast)
for _ in range(10):
    z.next_level()
    z.record(False)
    if z.finished():
        break
require(z.blind_by_quick_rule and z.trials <= 6,
        f"never seen: blind after {z.trials} patches (2 misses at maximum)")
z = ZestLocationThreshold(fast)
z.set_prior(math.log10(0.03), 0.25, 0.05, 0.2)
before = z.fast_sd_limit
z.posterior = np.zeros_like(z.posterior)
z.posterior[-5] = 1.0
require(z.fast_sd_limit < before, "the stopping limit is stricter at low sensitivity (ERF idea)")
z = ZestLocationThreshold(ZestSettings())
z.set_prior(math.log10(0.05))
for seen in (True, False, True, False, True, True):
    z.next_level()
    z.record(seen)
est = z.estimate_log
z.recompute(0.2, 0.2)
require(abs(z.estimate_log - est) > 1e-4 and len(z.history) == 6,
        "post-processing re-estimates from every answer with the measured error rates")

print("3. SITA Fast against full ZEST, healthy eyes, with a hill-smoothed normal table")
folder = pathlib.Path(tempfile.mkdtemp())
for i in range(4):
    out, _ = run(healthy, 300 + i, pid=f"N{i}", settings_overrides=dict(strategy=ZEST_FULL))
    write_outputs(out, folder / f"N{i}")
table, messages = build_prior_table(sorted(folder.glob("*/summary.csv")))
require(any("hill of vision" in m for m in messages) and "hill" in table.source,
        "the table is smoothed as a hill of vision")
e = table.entries[(9.0, 7.0)]
require(abs(e.mean_log10 - e.hill_log10) < 1e-12,
        "each point's table value is the published hill model's value there")
require(any("dB/cycle" in m for m in messages),
        "the fitted slopes are reported in the papers' units (dB per cycle)")
require(abs(e.hill_log10 - e.raw_mean_log10) < 0.15,
        f"the hill stays close to the plain average ({-20 * e.hill_log10:.1f} vs "
        f"{-20 * e.raw_mean_log10:.1f} dB)")
fast_out, _ = run(healthy, 9, pilot_mode=False, table=table)
full_out, _ = run(healthy, 9, pilot_mode=False, table=table,
                  settings_overrides=dict(strategy=ZEST_FULL))
err = [abs(p.zest.estimate_log - healthy(p.x_deg, p.y_deg)) for p in fast_out.points
       if p.zest.result().valid]
early = sum(p.zest.early_stop for p in fast_out.points)
require(fast_out.valid and all(p.zest.finished() for p in fast_out.points), "valid, every point finished")
require(fast_out.counts["trials"] < 0.8 * full_out.counts["trials"],
        f"fewer patches: {fast_out.counts['trials']} vs {full_out.counts['trials']} with full ZEST")
require(early > 10, f"{early} normal points stopped without needing a miss")
require(not any(p.zest.early_stop for p in fast_out.points if p.seed),
        "the four seed points keep the strict search")
require(statistics.median(err) < 0.12, f"median error {statistics.median(err):.3f} log10")
require(fast_out.post_processing.get("guess", 0) > 0, f"post-processing ran {fast_out.post_processing}")

print("4. hemianopia: the blind half still found, a surprising point retested")


def hemi_with_odd_point(x, y):
    if abs(x + 5.0) < 0.3 and abs(y - 7.0) < 0.3:
        return math.log10(0.55)                   # far worse than the table says, but
                                                  # clear of the top of the range
    return right_hemianopia(x, y)


out, _ = run(hemi_with_odd_point, 11, pilot_mode=False, table=table)
right = [p for p in out.points if p.x_deg > 0]
cats = [categorise(out, p, table) for p in right]
require(cats.count(BLIND) >= len(right) - 4 and set(cats) <= {BLIND, WEAK},
        f"right side: {cats.count(BLIND)} blind, {cats.count(WEAK)} weak (a chance press), "
        "none called normal")
odd = next(p for p in out.points if (p.x_deg, p.y_deg) == (-5.0, 7.0))
require(odd.zest.retested and out.counts["retests"] >= 1,
        f"the point far from its guess was searched again ({out.counts['retests']} retests)")

print("5. presses too fast to be real are false presses, not answers")
out, obs = run(healthy, 12, early_blank_presses=True)
require(out.counts["early_presses"] > 0 and out.counts["false_alarms"] == 0,
        f"{out.counts['early_presses']} early presses counted, none taken as 'seen'")

print("6. a good session is not failed by one or two chance lapses")
ok_runs = sum(run(healthy, 400 + i, pilot_mode=False, table=table)[0].valid for i in range(8))
require(ok_runs == 8, f"{ok_runs}/8 healthy sessions valid")

sys.exit(0 if ok else 1)
