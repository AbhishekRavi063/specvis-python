"""The results map drawn the way Specvis draws it: an interpolated contour map in dB.

Specvis (``org.contour2dplot.Contour2DMap``) interpolates the per-location dB values
bicubically, cuts the result into iso-bands ``isofactor`` dB wide, and colours each band
-- "Color" runs hue 250 (blue) to 0 (red), "Monochromatic" black to white. The same
controls are offered here: isofactor, interpolation, colour-bar density, map colours,
axes, and the fixation point and blind-spot overlay.

Values are **contrast dB** (``-20 log10`` of the threshold Michelson contrast; higher
means a fainter patch was seen, i.e. better vision). A point reported blind is drawn at
0 dB; "seen even at the faintest" at the top of the range; unfinished points are left
out. This is not clinical perimetric dB.
"""

from __future__ import annotations

import colorsys
import csv
import math
import pathlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

COLOR, MONO = "Color", "Monochromatic"


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


def _colours(n: int, scheme: str):
    if scheme == MONO:
        return [(v, v, v) for v in np.linspace(0.0, 1.0, max(n, 1))]
    hues = np.linspace(250.0, 0.0, max(n, 1))          # Specvis: reversed 0..250
    return [colorsys.hsv_to_rgb(h / 360.0, 1.0, 1.0) for h in hues]


def interpolated(points, interpolation: int = 6, extent: Optional[Tuple[float, float]] = None):
    """A fine mesh of interpolated dB inside the tested area (NaN outside)."""
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


def draw_contour_map(summary_rows: Sequence[dict], path: pathlib.Path, *,
                     title: str = "", isofactor_db: float = 3.0, interpolation: int = 6,
                     colorbar_density: int = 6, scheme: str = COLOR,
                     show_axes: bool = True, show_fixation: bool = True,
                     blind_spot_xy: Optional[Tuple[float, float]] = None,
                     dropped_points: Sequence[Tuple[float, float]] = (),
                     width_px: int = 1100, show_values: bool = False) -> pathlib.Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    points = point_values(summary_rows)
    fig, ax = plt.subplots(figsize=(width_px / 120.0, width_px / 120.0 * 0.62), dpi=120)
    if len(points) < 4:
        ax.text(0.5, 0.5, "Not enough finished points for a map", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(path)
        plt.close(fig)
        return pathlib.Path(path)

    xmax = max(abs(p[0]) for p in points) + 1.0
    ymax = max(abs(p[1]) for p in points) + 1.0
    gx, gy, grid = interpolated(points, interpolation, (xmax, ymax))
    top = max(float(np.nanmax(grid)), isofactor_db)
    levels = np.arange(0.0, top + isofactor_db, isofactor_db)
    colours = _colours(len(levels) - 1, scheme)
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(levels, cmap.N)
    mesh = ax.contourf(gx, gy, grid, levels=levels, cmap=cmap, norm=norm)
    ax.contour(gx, gy, grid, levels=levels, colors="#202020", linewidths=0.3)

    for x, y in dropped_points:
        ax.add_patch(plt.Rectangle((x - 1.5, y - 1.5), 3.0, 3.0, facecolor="#D3D1C7",
                                   edgecolor="#202020", linewidth=0.5, zorder=3))
    ax.scatter([p[0] for p in points], [p[1] for p in points], s=6, c="#202020", zorder=4)
    if show_values:
        for x, y, db, cat in points:
            above = y >= 0
            ax.text(x, y + (0.7 if above else -0.7), "blind" if cat == "blind" else f"{db:.0f}",
                    ha="center", va="bottom" if above else "top", fontsize=7,
                    color="#202020", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.6,
                              edgecolor="none"))
    if show_fixation:
        ax.plot(0, 0, marker="+", color="black", markersize=14, mew=2.5, zorder=5)
        if blind_spot_xy is not None:
            ax.add_patch(plt.Circle(blind_spot_xy, 1.0, facecolor="white",
                                    edgecolor="#d00000", linewidth=1.5, zorder=5))
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_aspect("equal")
    if show_axes:
        ax.axhline(0, color="#202020", lw=0.6)
        ax.axvline(0, color="#202020", lw=0.6)
        ax.set_xlabel("degrees (left - right)")
        ax.set_ylabel("degrees (down - up)")
    else:
        ax.set_axis_off()
    ticks = np.linspace(levels[0], levels[-1], max(2, colorbar_density))
    bar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05,
                       pad=0.16 if show_axes else 0.05, ticks=ticks)
    bar.set_label("contrast dB  (higher = fainter patch seen = better)")
    bar.ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
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
