"""Full Threshold (4-2): the Humphrey staircase as the Open Perimetry Interface writes
it (OPI ``R/full_threshold.r``), and its use in the perimetry procedure."""

from __future__ import annotations

import collections
import json
import math
import pathlib
import random
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.procedure.perimetry import (EXTENT_NAMES, EXTENTS, FULL, FULL_THRESHOLD,  # noqa: E402
                                         H24_2, H24_2_CENTRAL, SITA_FAST,
                                         PerimetrySettings, categorise, contrast_range_db,
                                         default_calibration, fit_problem, grid_points,
                                         seeds_for, settings_from_dict, settings_to_dict,
                                         write_outputs)
from specvis.procedure.session import SessionError  # noqa: E402
from specvis.threshold.base import Termination  # noqa: E402
from specvis.threshold.full_threshold import (FullThresholdLocation,  # noqa: E402
                                              FullThresholdSettings, from_db, to_db)
from sim_observer import healthy, right_hemianopia, run  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok    " if condition else "  FAIL  ") + what)
    ok = ok and bool(condition)


# OPI's own numbers: steps 4 then 2, retest beyond 4, range 0-40 dB, start 25 dB.
OPI = FullThresholdSettings(min_contrast=from_db(40), max_contrast=1.0, coarse_step_db=4,
                            fine_step_db=2, retest_db=4, max_presentations=1000)


def drive(loc: FullThresholdLocation, seen_fn):
    levels = []
    while not loc.finished():
        db = round(to_db(loc.next_level()), 6)
        levels.append(db)
        loc.record(seen_fn(db))
    return levels


def opi_ft(est, seen_fn, lo=0.0, hi=40.0):
    """A line-by-line reference of OPI's FT() (R/full_threshold.r)."""
    def staircase(start):
        cur, revs, n_min, n_max, last, prev, n = start, 0, 0, 0, None, None, 0
        while revs < 2 and n_min < 2 and n_max < 2:
            resp = seen_fn(cur)
            n += 1
            if prev is not None and resp != prev:
                revs += 1
            prev = resp
            if resp:
                last = cur
            if not resp and cur == lo:
                n_min += 1      # not seen at the brightest
            if resp and cur == hi:
                n_max += 1      # seen at the faintest
            delta = (4 if revs == 0 else 2) * (1 if resp else -1)
            cur = min(hi, max(lo, cur + delta))
        if n_max == 2:
            return "Max", hi, n
        if n_min == 2:
            return "Min", lo, n
        return "Reversals", last, n
    reason, final, n = staircase(est)
    if reason == "Reversals" and abs(est - final) > 4:
        reason, final, n2 = staircase(final)
        n += n2
    return reason, final, n


print("1. the OPI rule, worked examples (Humphrey numbers)")
loc = FullThresholdLocation(OPI, from_db(25))
lv = drive(loc, lambda db: db <= 27.5)               # true value about 27
r = loc.result()
require(lv == [25, 29, 27] and r.valid and round(to_db(r.value), 6) == 27
        and not loc.retested, f"near the start: {lv} -> 27, one staircase")
loc = FullThresholdLocation(OPI, from_db(25))
lv = drive(loc, lambda db: db <= 31.5)               # true value about 31
r = loc.result()
require(lv == [25, 29, 33, 31, 31, 35, 33, 31] and round(to_db(r.value), 6) == 31
        and loc.retested and r.trials == 8,
        f"far from the start (6 > 4): second staircase from 31: {lv}")

print("2. the edges")
loc = FullThresholdLocation(OPI, 1.0)
lv = drive(loc, lambda db: False)
require(lv == [0, 0] and loc.result().termination is Termination.CEILING_NOT_SEEN
        and not loc.result().valid, "start at the maximum, missed twice -> blind in 2")
loc = FullThresholdLocation(OPI, from_db(25))
lv = drive(loc, lambda db: False)
require(lv[-2:] == [0, 0] and loc.result().termination is Termination.CEILING_NOT_SEEN,
        f"never seen from 25 dB -> blind after {len(lv)} presentations")
loc = FullThresholdLocation(OPI, from_db(25))
lv = drive(loc, lambda db: True)
require(lv[-2:] == [40, 40] and loc.result().termination is Termination.FLOOR_ALWAYS_SEEN,
        "always seen -> better than the range, at the faintest twice")
require(all(0 <= x <= 40 for x in lv), "levels stay inside the range")

print("3. same answers as OPI's FT for 3000 random observers")
rng = random.Random(1)
mismatch = 0
for i in range(3000):
    t, slope, lapse = rng.uniform(-5, 45), rng.uniform(0.3, 3), rng.uniform(0, 0.1)
    start = rng.choice([25, 30, 15, 0, 40, rng.uniform(0, 40)])
    seed = rng.random()

    def observer(seed=seed, t=t, slope=slope, lapse=lapse):
        r = random.Random(seed)

        def seen(db):
            p = lapse + (1 - 2 * lapse) / (1 + math.exp((db - t) * slope))
            return r.random() < p
        return seen
    reason, final, n = opi_ft(start, observer())
    loc = FullThresholdLocation(OPI, from_db(start))
    drive(loc, observer())
    res = loc.result()
    mine = {Termination.CEILING_NOT_SEEN: ("Min", 0.0), Termination.FLOOR_ALWAYS_SEEN:
            ("Max", 40.0)}.get(res.termination,
                               ("Reversals", to_db(res.value) if res.valid else None))
    if mine[0] != reason or abs(mine[1] - final) > 1e-6 or res.trials != n:
        mismatch += 1
require(mismatch == 0, f"stop reason, final value and presentations all match ({mismatch} differ)")

print("4. our units: Humphrey-size steps by default")
d = FullThresholdSettings()
require((d.coarse_step_db, d.fine_step_db, d.retest_db) == (8.0, 4.0, 8.0),
        "steps 8 then 4 contrast dB (= Humphrey 4 then 2), second staircase beyond 8")
loc = FullThresholdLocation(start_contrast=0.05)
loc.record(True)
require(abs(to_db(loc.next_level()) - (to_db(0.05) + 8)) < 1e-9, "seen -> 8 dB fainter")
loc = FullThresholdLocation(start_contrast=0.05, renderable_floor=0.01)
for _ in range(10):
    if loc.finished():
        break
    loc.record(True)
require(loc.next_level() >= 0.01 - 1e-12, "never below the renderable floor")

print("5. in the perimetry procedure")
FT = dict(strategy=FULL_THRESHOLD)
require(PerimetrySettings().strategy == FULL_THRESHOLD,
        "Full Threshold is the default (the team's choice)")
for bad, why in ((dict(ft_coarse_step_db=0), "zero step"),
                 (dict(ft_start_contrast=5.0), "start outside the range")):
    try:
        PerimetrySettings(**FT, **bad).validate()
        require(False, f"rejects {why}")
    except SessionError:
        require(True, f"rejects {why}")
s = PerimetrySettings(**FT, ft_blind_neighbour_start=True)
require(settings_from_dict(settings_to_dict(s)) == s, "settings save and load")

out, _ = run(healthy, 11, settings_overrides=FT)
cats = collections.Counter(categorise(out, p) for p in out.points)
require(out.valid and cats["seen"] == len(out.points), f"healthy eye: all seen {dict(cats)}")
require(all(p.zest.result().valid for p in out.points), "healthy eye: every point measured")
err = [to_db(p.zest.result().value) - to_db(10 ** healthy(p.x_deg, p.y_deg))
       for p in out.points]
require(abs(sum(err) / len(err)) < 3, f"healthy eye: mean error {sum(err) / len(err):+.1f} dB")
seeds = [p for p in out.points if p.seed]
require(len(seeds) == 4 and all(p.prior_source.startswith("fixed start") for p in seeds),
        "the 4 seed points start at the fixed start")
first = {}
for row in out.trial_rows:
    if row["kind"] == "zest" and row["point"] not in first:
        first[row["point"]] = row["trial"]
last_seed = max(max(r["trial"] for r in out.trial_rows if r["point"] == p.index)
                for p in seeds)
require(all(first[p.index] > last_seed for p in out.points if not p.seed),
        "seed points finish before any other point starts")
require(all(p.prior_source.startswith("growth pattern") for p in out.points if not p.seed),
        "every other point starts from finished neighbours (growth pattern)")
require(not out.post_processing, "no SITA post-processing")

flashes = {}
for rule in (False, True):
    o, _ = run(right_hemianopia, 12, settings_overrides=dict(FT, ft_blind_neighbour_start=rule))
    wrong = sum((right_hemianopia(p.x_deg, p.y_deg) is None) != (categorise(o, p) == "blind")
                for p in o.points)
    require(o.valid and wrong <= 2,
            f"hemianopia, blind-neighbour rule {'on' if rule else 'off'}: blind half "
            f"found ({wrong} points wrong)")
    flashes[rule] = len(o.trial_rows)
    if rule:
        require(any("our rule" in p.prior_source for p in o.points),
                "the rule is used and marked as ours in the output")
    else:
        require(not any("our rule" in p.prior_source for p in o.points),
                "off by default: never used")
require(flashes[True] < 0.8 * flashes[False],
        f"the rule saves flashes in hemianopia: {flashes[False]} -> {flashes[True]}")

with tempfile.TemporaryDirectory() as tmp:
    folder = write_outputs(out, pathlib.Path(tmp) / "FT")
    meta = json.loads((folder / "session.json").read_text())
    summary = (folder / "summary.csv").read_text()
    require("Full Threshold" in json.dumps(meta), "session.json names the method")
    require("Full Threshold" in summary and "nan" not in summary,
            "summary.csv names the method; no 'nan' for the missing ZEST SD")

print("6. the Humphrey 24-2 grid (the default)")
require(PerimetrySettings().extent == H24_2_CENTRAL and PerimetrySettings().screen_width_mm == 544
        and PerimetrySettings().viewing_distance_mm == 520,
        "defaults: the lab setup (BenQ 544 mm, 52 cm, 24-2 without top/bottom rows)")
for eye, sign in (("R", 1), ("L", -1)):
    kept, dropped = grid_points(PerimetrySettings(eye=eye, extent=H24_2))
    require(len(kept) == 52 and sorted(dropped) == sorted([(sign * 15, -3), (sign * 15, 3)]),
            f"{eye} eye: 54 points, the 2 blind-spot points (15 deg temporal, +/-3) left out")
    require(sorted(p for p in kept if abs(p[0]) == 27) == [(-sign * 27, -3), (-sign * 27, 3)],
            f"{eye} eye: the 2 extra points at 27 deg are on the nasal side")
    require({abs(v) for p in kept for v in p} <= {3, 9, 15, 21, 27},
            f"{eye} eye: points at 3, 9, 15, 21 deg (and 27 nasal) only")
require(set(seeds_for(PerimetrySettings(extent=H24_2))) == {(-9.0, 9.0), (9.0, 9.0), (-9.0, -9.0),
                                                              (9.0, -9.0)},
        "seeds at the Humphrey's primary points (+/-9, +/-9)")
require(set(seeds_for(PerimetrySettings(extent=FULL))) == {(-9.0, 7.0), (9.0, 7.0), (-9.0, -7.0),
                                                           (9.0, -7.0)},
        "the old grid keeps its own seeds")
require(fit_problem(PerimetrySettings(extent=H24_2, screen_width_mm=531, viewing_distance_mm=330),
                   1920, 1080)
        is None, "fits a 24-inch 16:9 screen at 33 cm")
require(fit_problem(PerimetrySettings(extent=H24_2, screen_width_mm=531, viewing_distance_mm=400),
                   1920, 1080)
        is not None, "refused on a 24-inch screen at 40 cm (too far for 21 deg up/down)")

print("7. the seeds start at the middle of the measurable dB range")
s = PerimetrySettings(dithering=False)
r = contrast_range_db(s, default_calibration(s))
require(abs(r["middle_db"] - (r["strongest_db"] + r["faintest_db"]) / 2) < 0.02
        and r["faintest_db"] == r["grey_step_db"],
        f"no dithering: {r['strongest_db'] + 0.0:.1f} to {r['faintest_db']:.1f} dB "
        f"(one grey step), middle {r['middle_db']:.1f}")
r = contrast_range_db(PerimetrySettings(), default_calibration(PerimetrySettings()))
require(r["faintest_db"] == 60.0, "dithered: the faintest end is the search floor (60 dB)")
LAB = dict(strategy=FULL_THRESHOLD, extent=H24_2, screen_width_mm=302.0, viewing_distance_mm=240.0)
for f, name, n_blind in ((healthy, "healthy", 0), (right_hemianopia, "hemianopia", 24)):
    o, _ = run(f, 21, settings_overrides=LAB)
    cats = collections.Counter(categorise(o, p) for p in o.points)
    require(o.valid and len(o.points) == 52 and abs(cats["blind"] - n_blind) <= 1,
            f"24-2, {name}: valid, 52 points, {cats['blind']} blind (expected {n_blind})")
    require(o.contrast_range["ft_start_db"] == o.contrast_range["middle_db"]
            and o.contrast_range["ft_start_rule"] == "middle of the measurable range",
            f"24-2, {name}: seeds started at {o.contrast_range['ft_start_db']} dB, the middle")
    first = [r["requested_contrast"] for r in o.trial_rows if r["kind"] == "zest"][:1]
    require(first and abs(-20 * math.log10(first[0]) - o.contrast_range["middle_db"]) < 0.1,
            f"24-2, {name}: the first test flash is at the middle")
o, _ = run(healthy, 22, settings_overrides=dict(LAB, ft_start_contrast=0.1))
require(o.contrast_range["ft_start_rule"] == "given" and abs(o.contrast_range["ft_start_db"] - 20) < 0.01,
        "a given start is used as given (0.1 = 20 dB)")
with tempfile.TemporaryDirectory() as tmp:
    meta = json.loads((write_outputs(o, pathlib.Path(tmp) / "R") / "session.json").read_text())
    require(meta["grid"] == H24_2 and "middle_db" in meta["contrast_range"],
            "session.json records the grid and the dB range")

print("8. the 24-2 without its top/bottom rows (for the eye tracker's 50-80 cm)")
for eye, sign in (("R", 1), ("L", -1)):
    kept, dropped = grid_points(PerimetrySettings(eye=eye, extent=H24_2_CENTRAL))
    full, _ = grid_points(PerimetrySettings(eye=eye, extent=H24_2))
    require(len(kept) == 42 and set(kept) <= set(full)
            and max(abs(y) for _, y in kept) == 15 and max(abs(x) for x, _ in kept) == 21,
            f"{eye} eye: 42 spots, all 24-2 positions, within +/-21 x +/-15 deg")
    require(sorted(dropped) == sorted([(sign * 15, -3), (sign * 15, 3)]),
            f"{eye} eye: the same 2 blind-spot points left out")
require(set(seeds_for(PerimetrySettings(extent=H24_2_CENTRAL))) == {(-9.0, 9.0), (9.0, 9.0),
                                                                     (-9.0, -9.0), (9.0, -9.0)},
        "seeds at (+/-9, +/-9)")
W245 = 542.0          # a 24.5" 16:9 screen
require(fit_problem(PerimetrySettings(extent=H24_2_CENTRAL, screen_width_mm=W245,
                                      viewing_distance_mm=520), 1920, 1080) is None
        and fit_problem(PerimetrySettings(extent=H24_2_CENTRAL, screen_width_mm=W245,
                                          viewing_distance_mm=560), 1920, 1080) is not None,
        "fits a 24.5-inch screen at 52 cm, not at 56 cm")
require(fit_problem(PerimetrySettings(extent=FULL, screen_width_mm=W245,
                                      viewing_distance_mm=600), 1920, 1080) is None,
        "the Aflah grid fits the same screen at 60 cm")
require(all(e in EXTENT_NAMES for e in EXTENTS) and "42 spots" in EXTENT_NAMES[H24_2_CENTRAL]
        and "68 spots" in EXTENT_NAMES[FULL], "the app names each grid with its number of spots")
o, _ = run(right_hemianopia, 23, settings_overrides=dict(LAB, extent=H24_2_CENTRAL))
cats = collections.Counter(categorise(o, p) for p in o.points)
require(o.valid and len(o.points) == 42 and abs(cats["blind"] - 19) <= 1,
        f"hemianopia on the 42-spot grid: {cats['blind']} blind of 19 in the right half")

sys.exit(0 if ok else 1)
