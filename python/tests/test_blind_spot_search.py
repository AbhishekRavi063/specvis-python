"""The blind-spot search (Humphrey-style, Padmini): each person's own blind spot is
found before the test, instead of one average position for everyone."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.procedure.perimetry import (BLIND_SPOT, BLIND_SPOT_SEARCH,  # noqa: E402
                                         SEARCH_CHECK, PerimetrySettings,
                                         blind_spot_candidates, results_text, write_outputs)
from specvis.procedure.maps import _blind_spot  # noqa: E402
from sim_observer import healthy, right_hemianopia, run  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok    " if condition else "  FAIL  ") + what)
    ok = ok and bool(condition)


def search_rows(out):
    return [r for r in out.trial_rows if r["kind"] == BLIND_SPOT_SEARCH]


print("1. the positions tried")
for eye, sign in (("R", 1), ("L", -1)):
    c = blind_spot_candidates(PerimetrySettings(eye=eye))
    require(c[0] == (sign * 15.5, -1.5), f"{eye} eye: starts at the average {c[0]}")
    require(len(c) == 18 and all(11 <= sign * x <= 20 and y <= 1 for x, y in c),
            f"{eye} eye: 18 positions, all temporal 11-20 deg and at most 1 deg above")

print("2. a typical blind spot: found at once")
flashes = []
for seed in range(6):
    out, _ = run(healthy, seed)
    bs = out.blind_spot_search
    require(bs["status"] == "found" and tuple(bs["found_xy"]) == (15.5, -1.5),
            f"seed {seed}: found at the average ({bs.get('found_xy')})")
    flashes.append(len(search_rows(out)) +
                   sum(r["kind"] == SEARCH_CHECK for r in out.trial_rows))
    require(out.checks["fixation"], f"seed {seed}: fixation check passes")
require(max(flashes) <= 6, f"search took {min(flashes)}-{max(flashes)} flashes in all")

print("3. a blind spot further out: the search finds it, the average would not")
moved = (18.5, -3.5)                       # 3 deg further out, 2 deg lower
for seed in range(4):
    out, _ = run(healthy, 10 + seed, blind_spot=moved)
    bs = out.blind_spot_search
    x, y = bs.get("found_xy") or (None, None)
    # well inside: the 2-deg patch (1 deg radius) wholly in the 5 x 5 deg blind spot
    inside = x is not None and abs(x - moved[0]) <= 1.5 and abs(y - moved[1]) <= 1.5
    require(bs["status"] == "found" and inside, f"seed {seed}: found at {x}, {y}")
    checks = [r for r in out.trial_rows if r["kind"] == BLIND_SPOT]
    require(all((r["x_deg"], r["y_deg"]) == (x, y) for r in checks),
            "every blind-spot check went to the found position")
    require(out.checks["fixation"], f"fixation check passes ({out.fixation_loss_rate:.0%})")
flashes = len(search_rows(out)) + sum(r["kind"] == SEARCH_CHECK for r in out.trial_rows)
require(flashes <= 40, f"a moved blind spot took {flashes} flashes")
out, _ = run(healthy, 10, blind_spot=moved, settings_overrides=dict(blind_spot_search=False))
require(out.blind_spot_search["status"].startswith("off") and not out.checks["fixation"],
        f"search off: the average position is seen, fixation 'fails' "
        f"({out.fixation_loss_rate:.0%})")


print("4. left eye, mirrored")
out, _ = run(healthy, 20, eye="L", blind_spot=moved)
x, y = out.blind_spot_search["found_xy"]
require(x < 0 and abs(-x - moved[0]) <= 1.5, f"left eye: found at {x}, {y}")

print("5. no blind spot found (eye wandering): average used, fixation still fails")
out, obs = run(healthy, 30, sees_blind_spot=True)
bs = out.blind_spot_search
require(bs["status"].startswith("not found"), "not found")
require(out.blind_spot_xy == (15.5, -1.5), "checks use the average position")
require(not out.checks["fixation"] and not out.valid, "fixation check fails, not valid")
require(any("NOT found" in t for t in obs.screens), "the operator was told")
require(not out.blind_spot_search["blind_spot_side_not_seen"], "no hemianopia warning")
require("NOT found" in results_text(out), "results screen says so")

print("6. hemianopia on the blind spot's side: the operator is warned")
out, obs = run(right_hemianopia, 40)
bs = out.blind_spot_search
require(bs["blind_spot_side_not_seen"], f"blind spot's side not seen ({bs['strong_patches']})")
require(any("may be blind" in t for t in obs.screens), "the operator was warned")
out, obs = run(right_hemianopia, 41, eye="L")
require(not out.blind_spot_search["blind_spot_side_not_seen"],
        "left eye (blind spot on the seeing side): no warning")

print("7. eye tracker: no search")
out, _ = run(healthy, 50, settings_overrides=dict(fixation_monitoring="none"))
require(not out.blind_spot_search and not search_rows(out), "no search without blind-spot checks")

print("8. saved and drawn")
out, _ = run(healthy, 60, blind_spot=moved)
with tempfile.TemporaryDirectory() as d:
    folder = write_outputs(out, pathlib.Path(d) / "s")
    meta = json.loads((folder / "session.json").read_text())
    used = meta["blind_spot"]["used_xy"]
    require(used == out.blind_spot_search["found_xy"], f"session.json: used {used}")
    require(meta["blind_spot"]["average_xy"] == [15.5, -1.5], "session.json: average kept")
    require(_blind_spot(meta)[0] == tuple(used), "the map draws the found position")
    require((folder / "map.png").exists() and (folder / "contour_map.png").exists(), "maps drawn")
    kinds = (folder / "trials.csv").read_text()
    require(BLIND_SPOT_SEARCH in kinds and SEARCH_CHECK in kinds, "search flashes in trials.csv")

sys.exit(0 if ok else 1)
