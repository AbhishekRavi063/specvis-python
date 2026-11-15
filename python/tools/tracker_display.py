#!/usr/bin/env python
"""Show or set where the eye tracker thinks the screen is (its "display area").

Eye Tracker Manager on the lab PC has no Display Setup for the Fusion, so this does it
through the bridge (python/specvis.sh tracker-bridge must be running).

    python/specvis.sh tracker-display                       # show the current setting
    python/specvis.sh tracker-display --set                 # size 542.3 x 305, keep the rest
    python/specvis.sh tracker-display --set --gap-mm 12 --tilt-deg 20

The geometry, in Tobii's user coordinates (mm, origin at the tracker's front centre, x to
the person's right, y up, z towards the person):
  * width, height  -- the picture area (BenQ 24.5" 16:9: 542.3 x 305.0 mm);
  * gap            -- how far the picture's bottom edge is above the tracker's centre;
  * offset         -- how far the picture's centre is to the right of the tracker's centre;
  * depth          -- how far the bottom edge is in front of (+) / behind (-) the tracker;
  * tilt           -- the angle between the screen and the tracker's front: the screen's
                      top leans this many degrees towards the person in the tracker's frame
                      (a tracker tilted up at the eyes gives a positive tilt).
Anything not given is kept from the current setting. Measure without a ruler: a phone
level app for angles; a bank card (85.6 x 54 mm) for short gaps.
Before setting, the current one is saved in python/calibration/tracker/.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import socket
import sys
import time

HERE = pathlib.Path(__file__).resolve().parents[1]
BACKUP_DIR = HERE / "calibration" / "tracker"
CMD_PORT = 5556


def ask(cmd: str, port: int = CMD_PORT, timeout_s: float = 5.0, **kw) -> dict:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout_s)
    msg = dict(id=1, cmd=cmd, **kw)
    s.sendto(json.dumps(msg).encode("utf-8"), ("127.0.0.1", port))
    try:
        reply = json.loads(s.recv(65536).decode("utf-8"))
    except socket.timeout:
        raise SystemExit("No answer from the eye-tracker bridge. Start it first:  "
                         "python/specvis.sh tracker-bridge")
    finally:
        s.close()
    if not reply.get("ok"):
        raise SystemExit(f"The bridge refused '{cmd}': {reply}")
    return reply


def geometry(area: dict) -> dict:
    """width, height, gap, offset, depth, tilt from the three corners."""
    tl, tr, bl = (area[k] for k in ("top_left", "top_right", "bottom_left"))
    width = math.dist(tl, tr)
    height = math.dist(tl, bl)
    up = [(a - b) / height for a, b in zip(tl, bl)]
    bc = [bl[i] + (tr[i] - tl[i]) / 2 for i in range(3)]          # bottom-centre
    return dict(width_mm=width, height_mm=height, offset_mm=bc[0], gap_mm=bc[1],
                depth_mm=bc[2], tilt_deg=math.degrees(math.atan2(up[2], up[1])))


def corners(width_mm, height_mm, gap_mm, offset_mm, depth_mm, tilt_deg) -> dict:
    t = math.radians(tilt_deg)
    up = (0.0, math.cos(t), math.sin(t))
    bl = (offset_mm - width_mm / 2, gap_mm, depth_mm)
    tl = tuple(bl[i] + height_mm * up[i] for i in range(3))
    tr = (tl[0] + width_mm, tl[1], tl[2])
    r = lambda v: [round(c, 2) for c in v]                         # noqa: E731
    return dict(top_left=r(tl), top_right=r(tr), bottom_left=r(bl))


def describe(area: dict) -> str:
    g = geometry(area)
    return (f"  size   {g['width_mm']:.1f} x {g['height_mm']:.1f} mm\n"
            f"  gap    {g['gap_mm']:.1f} mm (bottom edge above the tracker)\n"
            f"  offset {g['offset_mm']:.1f} mm (picture centre right of the tracker)\n"
            f"  depth  {g['depth_mm']:.1f} mm (bottom edge in front of the tracker)\n"
            f"  tilt   {g['tilt_deg']:.1f} deg\n"
            f"  corners: top_left {area['top_left']}, top_right {area['top_right']}, "
            f"bottom_left {area['bottom_left']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="store_true", help="change the tracker's setting")
    ap.add_argument("--width-mm", type=float, default=542.3)
    ap.add_argument("--height-mm", type=float, default=305.0)
    ap.add_argument("--gap-mm", type=float, default=None)
    ap.add_argument("--offset-mm", type=float, default=None)
    ap.add_argument("--depth-mm", type=float, default=None)
    ap.add_argument("--tilt-deg", type=float, default=None)
    ap.add_argument("--port", type=int, default=CMD_PORT)
    args = ap.parse_args()

    now = ask("display_area", args.port)
    print("The eye tracker's screen now:")
    print(describe(now))
    if not args.set:
        return 0

    g = geometry(now)
    new = corners(args.width_mm, args.height_mm,
                  g["gap_mm"] if args.gap_mm is None else args.gap_mm,
                  g["offset_mm"] if args.offset_mm is None else args.offset_mm,
                  g["depth_mm"] if args.depth_mm is None else args.depth_mm,
                  g["tilt_deg"] if args.tilt_deg is None else args.tilt_deg)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / time.strftime("display_area_before_%Y%m%d_%H%M%S.json")
    backup.write_text(json.dumps({k: now[k] for k in ("top_left", "top_right", "bottom_left",
                                                       "width_mm", "height_mm")}, indent=2))
    done = ask("set_display_area", args.port, **new)
    print(f"\nSet (the old setting is saved in {backup}):")
    print(describe(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
