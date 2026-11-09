#!/usr/bin/env python
"""Build our own normal table from pilot perimetry sessions.

    python/specvis.sh build-table                 # every session in python/output/perimetry
    python/specvis.sh build-table DIR_OR_FILE ... # only these

Reads each session's ``summary.csv``, skips sessions that failed their checks, and
writes ``python/output/perimetry/prior_table.csv``. Perimetry runs with pilot mode off
then use it as the starting guess.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from specvis.procedure.prior_table import build_prior_table  # noqa: E402

OUTPUT = HERE / "output" / "perimetry"
TABLE = OUTPUT / "prior_table.csv"


def main(argv) -> int:
    targets = [pathlib.Path(a) for a in argv] or [OUTPUT]
    summaries = []
    for t in targets:
        summaries += [t] if t.is_file() else sorted(t.glob("*/summary.csv"))
    if not summaries:
        print(f"No perimetry sessions found in {', '.join(map(str, targets))}.")
        return 1
    table, messages = build_prior_table(summaries)
    for m in messages:
        print("  " + m)
    if not table.entries:
        print("No valid thresholds to build a table from.")
        return 1
    table.save(TABLE)
    people = max(e.n_participants for e in table.entries.values())
    print(f"\n{table.source}: {len(table.entries)} points, up to {people} people per point.")
    if people < 10:
        print("Fewer than 10 people: usable for testing the method, not yet a reliable "
              "normal table.")
    print(f"Saved {TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
