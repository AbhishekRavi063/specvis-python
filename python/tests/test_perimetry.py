"""Monocular perimetry, run against simulated observers with no window open.

The observer looks only at what the backend is asked to draw: the rendered contrast of
the patch and where it is on the screen, converted back to field degrees. It never
sees the thresholds or the procedure's state.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import random
import statistics
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.procedure.perimetry import (BLIND, CENTRAL, FULL, SEEN, ZEST_FULL,  # noqa: E402
                                         PerimetrySettings, categorise, grid_points, results_marks,
                                         run_perimetry, summary_rows, write_outputs)
from specvis.procedure.prior_table import PriorTable, build_prior_table  # noqa: E402
from specvis.procedure.session import SessionError  # noqa: E402
from specvis.stimulus.backend import NullBackend  # noqa: E402
from sim_observer import (DIST_MM, H, W, WIDTH_MM, Observer, healthy,  # noqa: E402
                          right_hemianopia, run)

ZEST_TABLE = dict(strategy=ZEST_FULL)    # how the table people are tested
ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


print("1. the grid")
for eye in ("R", "L"):
    kept, dropped = grid_points(PerimetrySettings(eye=eye))
    temporal = 1 if eye == "R" else -1
    require(len(kept) == 68 and len(dropped) == 4,
            f"{eye} eye: 68 test points, 4 blind-spot points dropped")
    require(all(x * temporal > 0 for x, _ in dropped), f"{eye} eye: dropped points are temporal")
kept, dropped = grid_points(PerimetrySettings(extent=CENTRAL))
require(len(kept) == 46 and max(abs(x) for x, _ in kept) == 13,   # 8 x 6 minus 2
        f"central grid: {len(kept)} points out to 13 deg")

print("2. a healthy eye, pilot mode")
out, obs = run(healthy, 1)
cats = [categorise(out, p) for p in out.points]
errors = [abs(p.zest.estimate_log - healthy(p.x_deg, p.y_deg)) for p in out.points
          if p.zest.result().valid]
require(out.valid, f"session valid (checks {out.checks})")
require(all(p.zest.finished() for p in out.points), "every point finished")
require(cats.count(BLIND) == 0, "no point called blind")
require(statistics.median(errors) < 0.15, f"median threshold error {statistics.median(errors):.3f} log10")
require(all(p.prior_source == "none (pilot mode)" for p in out.points),
        "pilot mode: no point got a starting guess")
first_done = [r["point"] for r in out.trial_rows if r["note"] in ("converged", "floor_always_seen",
                                                                   "ceiling_not_seen", "budget_exhausted")][:4]
seeds = {p.index for p in out.points if (p.x_deg, p.y_deg) in out.settings.seed_points}
require(set(first_done) == seeds, "the four seed points finish first")
xs_shown = [x for x, y, c in obs.shown]
require(max(abs(x) for x in xs_shown) > 20.5, "points really reach 21 deg (tangent placement)")
pilot_trials = out.counts["trials"]
print(f"       ({pilot_trials} trials, false alarms {out.false_alarm_rate:.0%}, "
      f"misses {out.miss_rate:.0%}, blind spot seen {out.fixation_loss_rate:.0%})")

print("3. right hemianopia, guesses on: blind side found, no guess crosses the midline")
out, obs = run(right_hemianopia, 2, pilot_mode=False)
right = [p for p in out.points if p.x_deg > 0]
left = [p for p in out.points if p.x_deg < 0]
blind_right = sum(categorise(out, p) == BLIND for p in right)
seen_left = sum(categorise(out, p) == SEEN for p in left)
require(blind_right >= 0.9 * len(right), f"{blind_right}/{len(right)} right-side points blind")
require(seen_left >= 0.9 * len(left), f"{seen_left}/{len(left)} left-side points seen")
left_guesses = [p.zest.prior_centre_log for p in left if p.zest.prior_centre_log is not None]
require(left_guesses and max(10 ** g for g in left_guesses) < 0.5,
        f"left-side starting guesses stay low (max {max(10 ** g for g in left_guesses):.2f}): "
        "blind right-side points never leaked across the midline")
require(any("neighbours" in p.prior_source for p in out.points), "neighbour guesses were used")

print("4. the checks catch an unreliable session")
out, _ = run(healthy, 3, false_alarm=0.4)
require(not out.checks["false_alarms"] and not out.valid,
        f"40% false presses -> not valid ({out.false_alarm_rate:.0%})")
out, _ = run(healthy, 4, sees_blind_spot=True)
require(not out.checks["fixation"] and not out.valid,
        f"blind spot seen -> fixation check fails ({out.fixation_loss_rate:.0%})")

print("5. escape stops it, and what was done is kept")
out, _ = run(healthy, 5, abort_after=40)
require(out.aborted and not out.valid, "aborted, not valid")
require(len(out.trial_rows) >= 40, f"{len(out.trial_rows)} trials kept")

print("6. output files")
out, _ = run(right_hemianopia, 6)
with tempfile.TemporaryDirectory() as tmp:
    folder = write_outputs(out, pathlib.Path(tmp) / "s")
    names = sorted(p.name for p in folder.iterdir())
    require(names == ["contour_map.png", "map.png", "session.json", "summary.csv", "trials.csv"], f"files {names}")
    require((folder / "map.png").stat().st_size > 10_000, "map.png has content")
    rows = list(csv.DictReader((folder / "summary.csv").open()))
    require(len(rows) == len(out.points), "summary.csv: one row per point")
    meta = json.loads((folder / "session.json").read_text())
    require(meta["calibration"]["calibrated"] is False and "checks" in meta,
            "session.json records calibration and checks")
require(len(results_marks(out)) == len(out.points) + len(out.dropped_points),
        "on-screen map: one mark per point, plus the blind-spot points")

print("7. the normal table: build, mirror the eyes, use it")
with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    paths = []
    for i, (pid, eye) in enumerate([("A", "R"), ("A", "L"), ("B", "R"), ("C", "L")]):
        o, _ = run(healthy, 10 + i, eye=eye, pid=pid, settings_overrides=ZEST_TABLE)
        paths.append(write_outputs(o, tmp / f"{pid}_{eye}") / "summary.csv")
    bad, _ = run(healthy, 20, false_alarm=0.4, pid="D", settings_overrides=ZEST_TABLE)
    paths.append(write_outputs(bad, tmp / "D_R") / "summary.csv")
    fast, _ = run(healthy, 21, pid="E")                      # SITA Fast, pilot mode on
    paths.append(write_outputs(fast, tmp / "E_R") / "summary.csv")
    table, messages = build_prior_table(paths)
    require(any("not valid" in m for m in messages), "an invalid session is skipped")
    require(any("E_R" in m and "not ZEST (full)" in m for m in messages),
            "a SITA Fast session is not used for the table (the plan: ZEST (full) only)")
    people = max(e.n_participants for e in table.entries.values())
    require(people == 3, f"3 people, not 4 eyes (got {people})")
    saved = PriorTable.load(table.save(tmp / "prior_table.csv"))
    header = [l for l in (tmp / "prior_table.csv").read_text().splitlines()
              if not l.startswith("#")][0]
    require("normal_db" in header and "mean_log10" not in header,
            "the table file is written in dB")
    require(all(abs(saved.entries[k].mean_log10 - e.mean_log10) < 1e-4
                for k, e in table.entries.items()),
            "reading the dB file back gives the same normal values")
    old = tmp / "old_log10_table.csv"
    old.write_text("x_temporal_deg,y_deg,n_participants,n_sessions,mean_log10,sd_log10,"
                   "mean_contrast,raw_mean_log10,hill_log10\n9.0,7.0,3,3,-1.2,0.1,0.063,-1.21,-1.2\n")
    require(abs(PriorTable.load(old).lookup("R", 9, 7) + 1.2) < 1e-12,
            "older tables saved in log10 still load")
    r_guess, l_guess = saved.lookup("R", 9, 7), saved.lookup("L", -9, 7)
    require(r_guess is not None and r_guess == l_guess,
            "the right eye's (9,7) and the left eye's (-9,7) are the same temporal point")
    require(saved.normal_range("R", 9, 7) is not None, "a normal range where 3 people contributed")
    guided = [run(healthy, 30 + s, pilot_mode=False, table=saved)[0].counts["trials"]
              for s in range(3)]
    unguided = [run(healthy, 30 + s)[0].counts["trials"] for s in range(3)]
    require(statistics.mean(guided) < statistics.mean(unguided),
            f"with the table: {statistics.mean(guided):.0f} trials vs "
            f"{statistics.mean(unguided):.0f} in pilot mode")

print("8. a grid that does not fit is refused before the first trial")
try:
    obs = Observer(healthy, random.Random(0))
    obs.open(W, H, 0.5)
    run_perimetry(PerimetrySettings(screen_width_mm=302.0, viewing_distance_mm=570.0),
                  obs, show_instructions=False)
    require(False, "the full grid on a 30 cm screen at 57 cm is refused")
except SessionError as e:
    require("does not fit" in str(e), "the full grid on a 30 cm screen at 57 cm is refused")

sys.exit(0 if ok else 1)
