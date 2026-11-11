#!/usr/bin/env python
"""Redraw map.png (grid) and contour_map.png for saved sessions, from their summary.csv
and session.json. Nothing else in the folder is touched.

    python/specvis.sh redraw-maps                 every session in output/perimetry/
    python/specvis.sh redraw-maps <folder> ...    only these session folders
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from specvis.procedure.maps import save_session_maps  # noqa: E402


def main(argv) -> int:
    folders = [pathlib.Path(a) for a in argv] or sorted(
        d for d in (HERE / "output" / "perimetry").glob("*")
        if d.is_dir() and not d.name.startswith("_") and (d / "summary.csv").exists())
    if not folders:
        print("No saved sessions found.")
        return 0
    for d in folders:
        for png in save_session_maps(d):
            print(f"redrew {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
