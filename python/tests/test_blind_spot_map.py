"""Each person's own blind spot (23 Nov): mapped once (search, 8 edges, an oval, its
centre checked), saved per participant and eye, re-checked in later sessions; grid
points left out from it; and the probe block (1-or-2, then 'saw anything?') at it."""

from __future__ import annotations

import json
import math
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.app.store import BlindSpotStore  # noqa: E402
from specvis.procedure import maps  # noqa: E402
from specvis.procedure.perimetry import (BLIND_SPOT_MAP, BLIND_SPOT_SEARCH,  # noqa: E402
                                         BLIND_SPOT_VERIFY, SEARCH_CHECK, PerimetrySettings,
                                         SessionError, binomial_p_at_least,
                                         fit_blind_spot_oval, grid_points, in_oval,
                                         probe_summary, results_text, settings_from_dict,
                                         settings_to_dict, split_by_oval, write_outputs)
from sim_observer import healthy, right_hemianopia, run  # noqa: E402

ok = True
MAP_KINDS = (BLIND_SPOT_MAP, BLIND_SPOT_VERIFY, BLIND_SPOT_SEARCH, SEARCH_CHECK)
TRUE_HALF = (2.75, 3.75)                  # the textbook 5.5 x 7.5 deg blind spot


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok    " if condition else "  FAIL  ") + what)
    ok = ok and bool(condition)


def mapped_run(seed, centre=(15.5, -1.5), eye="R", threshold=healthy, extra=None, **kw):
    overrides = dict(blind_spot_map=True, fixation_monitoring="none")
    overrides.update(extra or {})
    return run(threshold, seed, eye=eye, settings_overrides=overrides, blind_spot=centre,
               blind_spot_half=TRUE_HALF, blind_spot_oval=True, **kw)


def map_flashes(out) -> int:
    return sum(r["kind"] in MAP_KINDS for r in out.trial_rows)


print("1. the oval through 8 edge points")
cx, cy, a, b = 16.0, -2.0, 2.5, 3.5
edges = [(cx + a * math.cos(math.radians(t)), cy + b * math.sin(math.radians(t)))
         for t in range(0, 360, 45)]
fit = fit_blind_spot_oval(edges)
require(fit is not None and all(abs(u - v) < 1e-3 for u, v in zip(fit, (cx, cy, a, b))),
        f"recovers the oval exactly: {fit}")
require(fit_blind_spot_oval(edges[:3]) is None, "fewer than 4 points: no oval")
require(fit_blind_spot_oval([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]) is None,
        "points on a line: no oval")
require(all(isinstance(v, float) and not hasattr(v, "dtype") for v in fit),
        "plain floats (they go into JSON)")
require(in_oval(cx, cy, fit) and not in_oval(cx + a + 0.1, cy, fit)
        and in_oval(cx + a + 0.1, cy, fit, margin=0.5), "inside / outside / with margin")

print("2. leaving out grid points: the oval, not the average box")
s = PerimetrySettings(eye="R", extent="5 deg grid")
kept, dropped = grid_points(s)
require(len(dropped) == 4, f"average box: 4 left out {dropped}")
everything = kept + dropped
k2, d2 = split_by_oval(everything, (15.5, -1.5, 2.8, 3.8))
require(sorted(d2) == [(12.5, -2.5), (17.5, -2.5)],
        f"an average-sized oval at the average: the 2 inside it {d2} (the box's corners, "
        "(12.5, 2.5) and (17.5, 2.5), are outside an oval and are tested)")
k3, d3 = split_by_oval(everything, (19.0, -4.0, 2.8, 3.8))
require((17.5, -2.5) in d3 and (12.5, 2.5) not in d3 and (12.5, -2.5) not in d3,
        f"a blind spot further out: its own points {d3}")
require(len(k3) + len(d3) == 60, "no point lost or doubled")

print("3. chance")
require(abs(binomial_p_at_least(15, 20) - 0.0207) < 1e-3, "15 of 20 right: p 0.021")
require(abs(binomial_p_at_least(10, 20) - 0.588) < 1e-3, "10 of 20 right: p 0.59")
require(math.isnan(binomial_p_at_least(0, 0)), "no trials: no p")

print("4. mapping finds each person's own blind spot")
for seed, centre in ((1, (15.5, -1.5)), (2, (18.0, -3.0)), (3, (13.5, 0.0)), (4, (19.0, -4.0))):
    out, _ = mapped_run(seed, centre)
    m = out.blind_spot_map
    got = m.get("centre_xy") or [float("nan")] * 2
    err = math.hypot(got[0] - centre[0], got[1] - centre[1])
    size_err = max(abs(m.get("half_width_deg", 99) - TRUE_HALF[0]),
                   abs(m.get("half_height_deg", 99) - TRUE_HALF[1]))
    require(m["status"] == "mapped" and err <= 0.5 and size_err <= 0.5,
            f"blind spot at {centre}: mapped at {got} (off {err:.2f} deg), "
            f"{2 * m.get('half_width_deg', 0):.1f} x {2 * m.get('half_height_deg', 0):.1f} deg, "
            f"{map_flashes(out)} flashes")
    require(map_flashes(out) <= 170, "at most ~170 flashes (about 5 min)")
    oval = (got[0], got[1], m["half_width_deg"], m["half_height_deg"])
    require(all(in_oval(x, y, oval, 0.5) for x, y in out.dropped_points)
            and not any(in_oval(p.x_deg, p.y_deg, oval, 0.5) for p in out.points),
            f"left out exactly the grid points in the oval: {out.dropped_points}")
    require(tuple(out.blind_spot_xy) == tuple(got), "the checks use the oval's centre")
    json.dumps(m)                                   # saved as JSON: must not raise
out, _ = mapped_run(5, (17.0, -2.5), eye="L")          # the observer's x is temporal
got = out.blind_spot_map.get("centre_xy")
require(got and abs(got[0] + 17.0) <= 0.5 and abs(got[1] + 2.5) <= 0.5,
        f"left eye: on the left at {got}")
out, _ = mapped_run(6, (19.0, -4.0))
require(len(out.dropped_points) and (13, -1) not in out.dropped_points,
        f"moved blind spot: the average's points are tested again {out.dropped_points}")

print("5. a saved map: only its centre is checked")
first, _ = mapped_run(7, (18.0, -3.0))
saved = dict(first.blind_spot_map)
again, _ = mapped_run(8, (18.0, -3.0), saved_blind_spot=saved)
m = again.blind_spot_map
require(m["status"] == "reused" and m["centre_xy"] == saved["centre_xy"],
        f"reused: {m['status']} at {m.get('centre_xy')}")
require(map_flashes(again) == 3, f"3 checks only ({map_flashes(again)} flashes)")
require(sorted(again.dropped_points) == sorted(first.dropped_points), "same points left out")
average, _ = mapped_run(9, (15.5, -1.5))
moved, _ = mapped_run(9, (19.0, -4.5), saved_blind_spot=dict(average.blind_spot_map))
m = moved.blind_spot_map
require(m["status"] == "mapped" and "saved_map_failed_check" in m
        and abs(m.get("centre_xy", [0])[0] - 19.0) <= 0.5,
        f"saved centre seen (head moved?): mapped again at {m.get('centre_xy')}")
other_eye, _ = mapped_run(10, (15.5, -1.5), eye="L", saved_blind_spot=saved)
require(other_eye.blind_spot_map["status"] == "mapped", "a right-eye map is not used for the left")
bigger = dict(saved, patch_diameter_deg=3.0)
out, _ = mapped_run(11, (18.0, -3.0), saved_blind_spot=bigger)
require(out.blind_spot_map["status"] == "mapped", "a map made with another patch size is not used")

print("6. hemianopia on the blind spot's side: not mapped, the average stays, no wasted minutes")
out, obs = mapped_run(12, threshold=right_hemianopia)
m = out.blind_spot_map
require(m["status"].startswith("not mapped") and "hemianopia" in m["reason"],
        f"{m['status']}: {m.get('reason')}")
require(map_flashes(out) <= 80, f"stopped early ({map_flashes(out)} flashes)")
require(sorted(out.dropped_points) == sorted(grid_points(out.settings)[1]),
        "the average box's points are left out")
require("NOT mapped" in results_text(out), "the results screen says so")
out, _ = mapped_run(13, (15.5, -1.5), eye="L", threshold=right_hemianopia)
require(out.blind_spot_map["status"] == "mapped", "the other eye (blind spot on the seeing side) maps")

print("7. the probe block")
probe = dict(blind_spot_probe=True)
out, obs = mapped_run(14, (16.0, -2.0), extra=probe)
rows = out.probe_rows
n_expected = 4 * (20 + 5)
require(len(rows) == n_expected, f"{len(rows)} trials (4 contrasts x (20 + 5))")
summ = probe_summary(rows)
spot = [r for r in summ if r["location"] == "blind_spot"]
ctrl = [r for r in summ if r["location"] == "control"]
pooled = sum(r["correct"] for r in spot)
require(abs(pooled / 80 - 0.5) < 0.15, f"no light gets in: {pooled}/80 right at the blind spot")
require(all(r["pct_correct"] >= 80 for r in ctrl), "the mirror point is seen: the task works")
require(all(r["x_deg"] == out.blind_spot_map["centre_xy"][0] for r in rows
            if r["location"] == "blind_spot"), "at the mapped centre")
require(all(r["x_deg"] == -out.blind_spot_map["centre_xy"][0] for r in rows
            if r["location"] == "control"), "control at the mirror point")
require({r["target_interval"] for r in rows} == {1, 2}, "the patch is in either flash")
require(out.probe_status.startswith("done WITHOUT"), "without a tracker: said so")
# Bright patches leak light -- only above the map's 0.8, or the map itself would see it.
scatter = (lambda c: 0.9 if c > 0.9 else 0.0)
out, _ = mapped_run(15, (16.0, -2.0), extra=probe, scatter=scatter)
top = [r for r in probe_summary(out.probe_rows) if r["location"] == "blind_spot"]
require(top[0]["db_below_max"] == 0 and top[0]["p_above_chance"] < 0.01
        and top[-1]["p_above_chance"] > 0.01,
        f"scattered light shows at the strongest only ({top[0]['correct']}/20 vs "
        f"{top[-1]['correct']}/20)")
try:
    PerimetrySettings(blind_spot_probe=True).validate()
    require(False, "a probe without the map is refused")
except SessionError:
    require(True, "a probe without the map is refused")
out, _ = mapped_run(16, extra=probe, threshold=right_hemianopia)
require(not out.probe_rows and out.probe_status.startswith("skipped"),
        f"no map: no probe ({out.probe_status})")

print("8. saved files")
out, _ = mapped_run(17, (18.0, -3.0), extra=probe)
with tempfile.TemporaryDirectory() as d:
    folder = write_outputs(out, pathlib.Path(d) / "s")
    meta = json.loads((folder / "session.json").read_text())
    require(meta["blind_spot"]["map"]["status"] == "mapped", "session.json: the map")
    require(len(meta["blind_spot_probe"]["summary"]) == 8, "session.json: the probe summary")
    require((folder / "blind_spot_probe.csv").exists()
            and (folder / "blind_spot_probe_summary.csv").exists(), "probe CSVs written")
    require(maps._blind_spot_oval(meta) is not None and (folder / "contour_map.png").exists(),
            "the oval is drawn on the map")

    store = BlindSpotStore(pathlib.Path(d) / "blind_spots.json")
    require(store.latest("P1", "R") is None, "nothing saved yet")
    store.add("P1", "R", out.blind_spot_map, "P1_R_1")
    store.add("P1", "R", {"status": "not mapped: used the average", "reason": "x"}, "P1_R_2")
    store.add("P1", "R", {}, "P1_R_3")
    latest = store.latest("P1", "R")
    require(latest and latest["centre_xy"] == out.blind_spot_map["centre_xy"]
            and latest["session_folder"] == "P1_R_1",
            "the latest usable map comes back (a failed one is kept but not reused)")
    require(len(store.history("P1", "R")) == 2, "every result kept; empty ones not saved")
    require(store.latest("P1", "L") is None and store.latest("P2", "R") is None,
            "per participant and per eye")

print("9. settings files keep the new settings")
s = PerimetrySettings(blind_spot_map=True, blind_spot_probe=True,
                      blind_spot_probe_db_below_max=(0.0, 3.0))
back = settings_from_dict(json.loads(json.dumps(settings_to_dict(s))))
require(back.blind_spot_map and back.blind_spot_probe
        and back.blind_spot_probe_db_below_max == (0.0, 3.0), "round trip")
presets = pathlib.Path(__file__).resolve().parents[1] / "presets"
for f in sorted(presets.glob("*.json")):
    st = json.loads(f.read_text())["settings"]
    settings_from_dict(st).validate()
    require(st.get("blind_spot_map") is True, f"{f.name}: map on")

sys.exit(0 if ok else 1)
