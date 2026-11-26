#!/usr/bin/env python
"""Measure what this display can actually present, and check it against the protocol.

Run this on the laboratory monitor before any recording session:

    python/.venv/bin/python python/tools/measure_display.py --fullscreen

Windowed mode is for checking that the code runs. It gives **no timing guarantee** --
the compositor is in the path -- so never quote a windowed number.

Nothing this prints is photodiode-validated. It is the software's own account of what
it did, which is necessary but not sufficient: a display can report a flip it has not
yet shown. Treat these numbers as an upper bound on how good the timing is.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from specvis.core.gabor import GaborParameters, render_brightness  # noqa: E402
from specvis.core.luminance import LuminanceCalibration  # noqa: E402
from specvis.core.reversal import ReversalPlan, presentable_rates  # noqa: E402
from specvis.stimulus.backend import Texture  # noqa: E402
from specvis.stimulus.psychopy_backend import PsychoPyBackend  # noqa: E402

#: The proposal's three bands (section 6a): theta, alpha, high beta/gamma.
PROTOCOL_BANDS = {"theta": (4, 8), "alpha": (8, 12), "high beta/gamma": (25, 40)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullscreen", action="store_true",
                        help="required for any number worth quoting")
    parser.add_argument("--screen", type=int, default=0)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--train-ms", type=float, default=1000.0)
    parser.add_argument("--flat-field", default="",
                        help="draw with this flat-field map (name or id) to check that the "
                             "correction drops no frames; needs --scale")
    parser.add_argument("--scale", default="", help="luminance scale name (with --flat-field)")
    args = parser.parse_args()

    flat_map, calibration = None, LuminanceCalibration.uncalibrated()
    background = 0.5
    if args.flat_field:
        from specvis.app.store import ScaleStore
        from specvis.core.flatfield import FlatFieldStore
        from specvis.core.gabor import snap_background
        flat_map = FlatFieldStore().find(args.flat_field)
        scales = [s for s in ScaleStore().all() if s.name == args.scale or s.id == args.scale]
        if flat_map is None or not scales:
            print("--flat-field needs a known map and --scale (a measured luminance scale)")
            return 1
        calibration = scales[0].calibration()
        _, background = snap_background(calibration, 10.0)

    backend = PsychoPyBackend(screen=args.screen, fullscreen=args.fullscreen)
    backend.open(args.width, args.height, background_brightness=background)
    if flat_map is not None:
        from specvis.core.flatfield import FlatField
        backend.set_flat_field(FlatField(flat_map, calibration, background, *backend.size_px))
    print(f"flat-field correction: {flat_map.describe() if flat_map else 'none'}")

    try:
        refresh = backend.refresh_hz
        print()
        print(f"measured refresh: {refresh:.3f} Hz")
        if not args.fullscreen:
            print("  WINDOWED -- not a timing measurement, do not quote this number")

        print()
        print("exactly presentable reversal rates, per protocol band:")
        for name, (low, high) in PROTOCOL_BANDS.items():
            rates = presentable_rates(refresh, low, high)
            shown = ", ".join(f"{r:.3f}" for r in rates) if rates else "NONE"
            print(f"  {name:16s} {low:2d}-{high:2d} Hz : {shown}")

        mean = float(calibration.luminance_at(background * 100.0))
        patch = GaborParameters(2.0, 4.0, 90.0, 0.0, 0.33, mean if flat_map else 50.0, 0.8)
        centre = (backend.size_px[0] / 2.0 + 300.0, backend.size_px[1] / 2.0 - 200.0) \
            if flat_map else (args.width / 2.0, args.height / 2.0)
        polarities = [
            Texture(render_brightness(patch, calibration, 30, 30), centre),
            Texture(render_brightness(patch.reversed_phase(), calibration, 30, 30), centre),
        ]

        print()
        print(f"presenting {args.train_ms:.0f} ms trains:")
        bin_hz = 1000.0 / args.train_ms
        print(f"  (a {args.train_ms:.0f} ms train resolves {bin_hz:.3f} Hz; "
              f"errors below half that cannot be seen in the spectrum)")
        header = (f"  {'requested':>10} {'actual':>10} {'error':>8} {'k':>3} "
                  f"{'dropped':>8} {'median':>7} {'p95 ms':>8} {'train ms':>9}")
        print(header)
        for requested in (6.0, 10.0, 30.0, 40.0):
            plan = ReversalPlan.for_rate(requested, refresh)
            frames = max(1, round(args.train_ms / 1000.0 * refresh))
            report = backend.present_train(polarities, plan.frames_per_reversal, frames)

            intervals = sorted(report.frame_intervals_ms)
            p95 = intervals[int(len(intervals) * 0.95)] if intervals else float("nan")
            median = statistics.median(intervals) if intervals else float("nan")
            # How long the train really took (first to last flip), against how long
            # it should: a train where EVERY frame is slow (24 Nov: 240 Hz shown as
            # 120 Hz) has no outlier frames, but its length gives it away.
            took = sum(report.frame_intervals_ms)
            should = (frames - 1) * 1000.0 / refresh
            # Judged against this train's own spectral resolution, not against exact
            # division of a measured refresh, which nothing ever satisfies.
            flag = "  NOT PRESENTABLE" if plan.error_is_resolvable(args.train_ms) else ""
            if took > should * 1.02 or median > 1.1 * 1000.0 / refresh:
                flag += f"  TOO SLOW (should take {should:.0f} ms)"
            print(f"  {requested:10.2f} {plan.actual_reversals_per_second:10.3f} "
                  f"{plan.error_hz:8.4f} {plan.frames_per_reversal:3d} "
                  f"{report.dropped_frames:8d} {median:7.3f} {p95:8.3f} {took:9.0f}{flag}")

        print()
        print("TIMING IS SOFTWARE SELF-REPORT. Not photodiode-validated.")
    finally:
        backend.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
