#!/usr/bin/env python
"""Specvis - ReFreM: the Specvis-style app running the ReFreM perimetry test.

    python/specvis.sh app            (Mac / Linux)     -- Specvis-style window
    python/specvis.sh app-simple                       -- the earlier one-window app
    python\\specvis.bat app           (Windows)
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

if __name__ == "__main__":
    if "--simple" in sys.argv:
        from specvis.app.main_window import main
    else:
        from specvis.app.specvis_ui import main
    raise SystemExit(main())
