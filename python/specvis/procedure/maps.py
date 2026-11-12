"""The two result maps, drawn from a session's saved rows (summary.csv) -- so any run,
old or new, can be redrawn the same way.

* **Grid map** (``map.png``): one box per tested point, on an even grid, left/right and
  up/down as in the field. Box colour = seen / weak / blind / not finished. Big number =
  sensitivity in contrast dB (higher = better); small line = the faintest contrast seen
  there, as a percentage, and the number of patches shown.
* **Contour map** (``contour_map.png``): Specvis's map. The per-point dB values are
  interpolated (bicubic) and cut into iso-bands ``isofactor`` dB wide; "Color" runs blue
  (low) to red (high), "Monochromatic" black to white.

Values are **contrast dB** = ``-20 log10`` (threshold Michelson contrast); higher means a
fainter patch was seen, i.e. better vision. A blind point is drawn at 0 dB; "seen even at
the faintest" at the top of the range; unfinished points are left out. This is not
clinical perimetric dB.
"""

from __future__ import annotations

import colorsys
import csv
import json
import math
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

COLOR, MONO = "Color", "Monochromatic"

# Light fills with dark text read better than white text on strong colours.
_FILL = {"seen": "#CFEBDD", "weak": "#FBE2B8", "blind": "#F4C3B3", "not_finished": "#E6E5E0"}
_EDGE = {"seen": "#1D9E75", "weak": "#D9891A", "blind": "#C4482A", "not_finished": "#888780"}
_TEXT = {"seen": "#0B4D39", "weak": "#6B4100", "blind": "#7A2410", "not_finished": "#444441"}
_NAMES = {"seen": "seen (normal)", "weak": "weak (worse than normal)",
          "blind": "blind (not seen at 100%)", "not_finished": "not finished"}
INK = "#202020"


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11})
    return plt


# --------------------------------------------------------------------------- values

def point_values(summary_rows: Sequence[dict], floor_db: float = 42.0
                 ) -> List[Tuple[float, float, float, str]]:
    """(x, y, dB, category) for every point with an answer."""
    out = []
    for r in summary_rows:
        x, y = float(r["x_deg"]), float(r["y_deg"])
        cat = r.get("category", "")
        if r.get("threshold_contrast"):
            db = -20.0 * math.log10(float(r["threshold_contrast"]))
        elif cat == "blind":
            db = 0.0
        elif r.get("termination") == "floor_always_seen":
            db = floor_db
        else:
            continue
        out.append((x, y, db, cat))
    return out


def interpolated(points, interpolation: int = 6, extent: Optional[Tuple[float, float]] = None):
    """A fine mesh of interpolated dB inside the tested area."""
    from scipy.interpolate import griddata

    pts = np.array([(p[0], p[1]) for p in points])
    vals = np.array([p[2] for p in points])
    xmax = extent[0] if extent else float(np.abs(pts[:, 0]).max())
    ymax = extent[1] if extent else float(np.abs(pts[:, 1]).max())
    step = 1.0 / max(1, interpolation)
    gx = np.arange(-xmax, xmax + step / 2, step)
    gy = np.arange(-ymax, ymax + step / 2, step)
    mx, my = np.meshgrid(gx, gy)
    grid = griddata(pts, vals, (mx, my), method="cubic")
    nearest = griddata(pts, vals, (mx, my), method="nearest")
    grid = np.where(np.isnan(grid), nearest, grid)          # fill the hull's edges
    return gx, gy, np.clip(grid, 0.0, None)


def _colours(n: int, scheme: str):
    if scheme == MONO:
        return [(v, v, v) for v in np.linspace(0.0, 1.0, max(n, 1))]
    hues = np.linspace(250.0, 0.0, max(n, 1))          # Specvis: reversed 0..250
    return [colorsys.hsv_to_rgb(h / 360.0, 0.85, 0.95) for h in hues]


def _title(meta: Optional[dict]) -> Tuple[str, str]:
    """Heading and sub-heading from session.json."""
    if not meta:
        return "", ""
    s = meta.get("settings", {})
    eye = {"R": "Right eye", "L": "Left eye"}.get(s.get("eye", ""), s.get("eye", ""))
    if meta.get("aborted"):
        state = "STOPPED EARLY"
    else:
        state = "VALID" if meta.get("session_valid") else "NOT VALID"
    head = f"{s.get('participant_id', '')}   {eye}   {state}"
    date = str(meta.get("started", ""))[:16].replace("T", " ")
    bits = [date, f"{meta.get('counts', {}).get('trials', '?')} trials",
            f"{meta.get('duration_s', 0) / 60:.1f} min"]
    if s.get("pilot_mode"):
        bits.append("pilot mode")
    if not meta.get("calibration", {}).get("calibrated", False):
        bits.append("contrast nominal (screen not calibrated)")
    return head, "   |   ".join(b for b in bits if b)


def _heading(ax, head: str, sub: str, colour: str) -> None:
    """Bold heading with a grey line under it, left-aligned above the plot."""
    if head:
        ax.annotate(head, (0, 1), xycoords="axes fraction", xytext=(0, 30 if sub else 10),
                    textcoords="offset points", ha="left", va="bottom", fontsize=15,
                    fontweight="bold", color=colour)
    if sub:
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 10),
                    textcoords="offset points", ha="left", va="bottom", fontsize=10,
                    color="#5F5E5A")


def _blind_spot(meta: Optional[dict]) -> Tuple[Optional[Tuple[float, float]],
                                               List[Tuple[float, float]]]:
    if not meta:
        return None, []
    s = meta.get("settings", {})
    side = 1.0 if s.get("eye", "R") == "R" else -1.0
    spot = (side * float(s.get("blind_spot_temporal_deg", 15.5)),
            -float(s.get("blind_spot_below_deg", 1.5)))
    used = (meta.get("blind_spot") or {}).get("used_xy")
    if used:                              # this person's own, found by the search
        spot = (float(used[0]), float(used[1]))
    dropped = [(float(x), float(y)) for x, y in meta.get("blind_spot_grid_points_dropped", [])]
    return spot, dropped


# --------------------------------------------------------------------------- grid map

def draw_grid_map(summary_rows: Sequence[dict], path: pathlib.Path, *,
                  meta: Optional[dict] = None,
                  dropped_points: Optional[Sequence[Tuple[float, float]]] = None
                  ) -> pathlib.Path:
    """One box per point: the faintest contrast seen, its dB, and the patches used."""
    plt = _matplotlib()
    from matplotlib.patches import FancyBboxPatch, Patch

    if dropped_points is None:
        dropped_points = _blind_spot(meta)[1]
    rows = [r for r in summary_rows if r.get("x_deg") not in (None, "")]
    xs = sorted({float(r["x_deg"]) for r in rows} | {float(x) for x, _ in dropped_points})
    ys = sorted({float(r["y_deg"]) for r in rows} | {float(y) for _, y in dropped_points},
                reverse=True)
    if not xs or not ys:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No points in this session", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return pathlib.Path(path)

    col = {x: i for i, x in enumerate(xs)}
    row = {y: len(ys) - 1 - j for j, y in enumerate(ys)}       # top row highest
    fig, ax = plt.subplots(figsize=(1.0 * len(xs) + 1.2, 1.0 * len(ys) + 2.4))

    def box(cx, cy, fill, edge, hatch=None):
        ax.add_patch(FancyBboxPatch((cx - 0.44, cy - 0.44), 0.88, 0.88,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=fill, edgecolor=edge, linewidth=1.2,
                                    hatch=hatch))

    for r in rows:
        cat = r.get("category", "not_finished") or "not_finished"
        cx, cy = col[float(r["x_deg"])], row[float(r["y_deg"])]
        box(cx, cy, _FILL.get(cat, "#FFFFFF"), _EDGE.get(cat, INK))
        ink = _TEXT.get(cat, INK)
        n = r.get("trials", "")
        if r.get("threshold_contrast"):
            c = float(r["threshold_contrast"])
            pct = f"{100 * c:.0f}%" if c >= 0.095 else f"{100 * c:.1f}%"
            big = f"{-20 * math.log10(c):.0f} dB"
            small = f"{pct}  n{n}"
        elif cat == "blind":
            big, small = "blind", f"n{n}"
        elif r.get("termination") == "floor_always_seen":
            big, small = "max", f"faintest seen  n{n}"
        else:
            big, small = "-", f"n{n}"
        ax.text(cx, cy + 0.11, big, ha="center", va="center", fontsize=12,
                fontweight="bold", color=ink)
        ax.text(cx, cy - 0.22, small, ha="center", va="center", fontsize=8, color=ink)
    for x, y in dropped_points:
        cx, cy = col[float(x)], row[float(y)]
        box(cx, cy, "#FFFFFF", "#B4B2A9", hatch="///")
        ax.text(cx, cy, "blind\nspot", ha="center", va="center", fontsize=8,
                color="#5F5E5A", bbox=dict(facecolor="white", edgecolor="none", pad=1))

    mid_x, mid_y = (len(xs) - 1) / 2.0, (len(ys) - 1) / 2.0   # no zero column or row
    ax.plot([mid_x, mid_x], [-0.5, len(ys) - 0.5], color=INK, lw=0.8, zorder=0)
    ax.plot([-0.5, len(xs) - 0.5], [mid_y, mid_y], color=INK, lw=0.8, zorder=0)
    ax.plot(mid_x, mid_y, marker="+", color=INK, ms=16, mew=2.5, zorder=5)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    ax.set_ylim(-0.6, len(ys) - 0.4)
    ax.set_aspect("equal")
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in reversed(ys)])
    ax.tick_params(length=0, labelsize=10, colors="#5F5E5A")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("degrees from fixation   (left  -  right)", color="#5F5E5A")
    ax.set_ylabel("degrees   (down  -  up)", color="#5F5E5A")

    head, sub = _title(meta)
    valid = bool(meta and meta.get("session_valid") and not meta.get("aborted"))
    _heading(ax, head, sub, "#0F6E56" if valid else "#A32D2D")
    cats = [c for c in ("seen", "weak", "blind", "not_finished")
            if any((r.get("category") or "not_finished") == c for r in rows)]
    handles = [Patch(facecolor=_FILL[c], edgecolor=_EDGE[c], label=_NAMES[c]) for c in cats]
    if dropped_points:
        handles.append(Patch(facecolor="white", edgecolor="#B4B2A9", hatch="///",
                             label="blind spot (not tested)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=min(5, len(handles)), frameon=False, fontsize=10)
    fig.text(0.5, 0.01, "Big number = sensitivity in contrast dB (higher = better).   "
             "Small line = the faintest contrast seen, and patches shown.",
             ha="center", fontsize=9, color="#5F5E5A")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    return pathlib.Path(path)


# --------------------------------------------------------------------------- contour map

def draw_contour_map(summary_rows: Sequence[dict], path: pathlib.Path, *,
                     title: str = "", isofactor_db: float = 3.0, interpolation: int = 6,
                     colorbar_density: int = 7, scheme: str = COLOR,
                     show_axes: bool = True, show_fixation: bool = True,
                     blind_spot_xy: Optional[Tuple[float, float]] = None,
                     dropped_points: Sequence[Tuple[float, float]] = (),
                     width_px: int = 1500, show_values: bool = True,
                     meta: Optional[dict] = None) -> pathlib.Path:
    """Specvis's interpolated iso-band map. ``meta`` (session.json) fills the title and
    the blind spot when they are not given."""
    plt = _matplotlib()
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if meta is not None:
        spot, dropped = _blind_spot(meta)
        blind_spot_xy = blind_spot_xy if blind_spot_xy is not None else spot
        dropped_points = dropped_points or dropped
    dpi = 150
    points = point_values(summary_rows)
    fig, ax = plt.subplots(figsize=(width_px / dpi, width_px / dpi * 0.62))
    if len(points) < 4:
        ax.text(0.5, 0.5, "Not enough finished points for a map", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
        return pathlib.Path(path)

    xmax = max(abs(p[0]) for p in points) + 2.0
    ymax = max(abs(p[1]) for p in points) + 2.0
    gx, gy, grid = interpolated(points, interpolation, (xmax, ymax))
    top = max(float(np.nanmax(grid)), isofactor_db)
    levels = np.arange(0.0, top + isofactor_db, isofactor_db)
    cmap = ListedColormap(_colours(len(levels) - 1, scheme))
    norm = BoundaryNorm(levels, cmap.N)
    mesh = ax.contourf(gx, gy, grid, levels=levels, cmap=cmap, norm=norm)
    ax.contour(gx, gy, grid, levels=levels, colors="white", linewidths=0.4, alpha=0.6)

    if dropped_points:                    # one hatched box around the skipped points
        dx = [float(x) for x, _ in dropped_points]
        dy = [float(y) for _, y in dropped_points]
        ax.add_patch(plt.Rectangle((min(dx) - 1.6, min(dy) - 1.6), max(dx) - min(dx) + 3.2,
                                   max(dy) - min(dy) + 3.2, facecolor="#FFFFFF",
                                   edgecolor="#5F5E5A", hatch="///", linewidth=0.8,
                                   zorder=3))
    if show_values:
        for x, y, db, cat in points:
            ax.text(x, y, "blind" if cat == "blind" else f"{db:.0f}", ha="center",
                    va="center", fontsize=9, color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85,
                              edgecolor="none"))
    else:
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=10, c=INK, zorder=4)
    if show_fixation:
        ax.plot(0, 0, marker="+", color=INK, markersize=16, mew=2.5, zorder=7)
        if blind_spot_xy is not None:
            ax.add_patch(plt.Circle(blind_spot_xy, 1.0, facecolor="white",
                                    edgecolor="#C4482A", linewidth=2, zorder=7))
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_aspect("equal")
    if show_axes:
        ax.axhline(0, color=INK, lw=0.7, zorder=2)
        ax.axvline(0, color=INK, lw=0.7, zorder=2)
        ax.set_xlabel("degrees   (left  -  right)", color="#5F5E5A")
        ax.set_ylabel("degrees   (down  -  up)", color="#5F5E5A")
        ax.tick_params(colors="#5F5E5A")
    else:
        ax.set_axis_off()
    ticks = np.linspace(levels[0], levels[-1], max(2, colorbar_density))
    bar = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.04, pad=0.02,
                       ticks=ticks)
    bar.set_label("contrast dB   (higher = better)", color="#5F5E5A")
    bar.ax.set_yticklabels([f"{t:.0f}" for t in ticks])
    bar.outline.set_visible(False)

    head, sub = _title(meta) if meta else (title, "")
    valid = bool(meta and meta.get("session_valid") and not meta.get("aborted"))
    _heading(ax, head, sub, INK if not meta else ("#0F6E56" if valid else "#A32D2D"))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    return pathlib.Path(path)


def export_map_data(summary_rows: Sequence[dict], path: pathlib.Path,
                    interpolation: int = 6) -> pathlib.Path:
    """The per-point dB values and the interpolated mesh, as CSV (Specvis: Export data)."""
    points = point_values(summary_rows)
    path = pathlib.Path(path)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# points: x_deg, y_deg, contrast_db, category"])
        w.writerow(["x_deg", "y_deg", "contrast_db", "category"])
        for p in points:
            w.writerow([p[0], p[1], round(p[2], 3), p[3]])
        if len(points) >= 4:
            gx, gy, grid = interpolated(points, interpolation)
            w.writerow([])
            w.writerow(["# interpolated map: rows = y (top to bottom), columns = x"])
            w.writerow(["y\\x"] + [round(x, 3) for x in gx])
            for j in range(len(gy) - 1, -1, -1):
                w.writerow([round(gy[j], 3)] + [round(v, 3) for v in grid[j]])
    return path


# --------------------------------------------------------------------------- one session

def read_session(folder: pathlib.Path) -> Tuple[List[Dict[str, str]], dict]:
    """(summary rows, session.json) of a saved session folder."""
    folder = pathlib.Path(folder)
    rows: List[Dict[str, str]] = []
    if (folder / "summary.csv").exists():
        with (folder / "summary.csv").open() as f:
            rows = list(csv.DictReader(f))
    try:
        meta = json.loads((folder / "session.json").read_text())
    except (OSError, ValueError):
        meta = {}
    return rows, meta


def save_session_maps(folder: pathlib.Path) -> List[pathlib.Path]:
    """(Re)draw map.png and contour_map.png in a session folder from its saved files."""
    folder = pathlib.Path(folder)
    rows, meta = read_session(folder)
    return [draw_grid_map(rows, folder / "map.png", meta=meta),
            draw_contour_map(rows, folder / "contour_map.png", meta=meta)]
