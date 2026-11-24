"""Flat-field (screen evenness) correction: draw the grey slightly brighter where the
backlight is weaker, so the light reaching the eye is about the same everywhere.

Lab request, 24 Nov 2026. The lab BenQ is about 10 % darker towards its edges (its
backlight). A map of the screen -- % against the centre at the 9 places of the
uniformity tool -- becomes a smooth surface m(x, y); the background is drawn at

    L(x, y) = L_bg / (1 + s * m(x, y) / 100)          s = strength (0.8 by default)

and every patch pixel is multiplied by the same local gain 1 / (1 + s * m / 100), so a
patch's Michelson contrast is unchanged (a common gain cancels) and only the local level
is evened out. Luminance is turned into grey levels with the selected luminance scale
(its table), then rounded with a STATIC dither: one fixed random offset per pixel
(+/- 0.5 grey step, fixed seed), the same on every frame -- it never flickers.

The patch is corrected with the same gain and the same dither offsets as the background
under it, so a patch's surround (equal to the background) comes out exactly equal to the
background image: no visible square.

The map belongs to one monitor and its settings. Nothing here draws; the backend does.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import photometry
from .luminance import LuminanceCalibration

#: The factor 1 + s*m/100 is kept within this range (a bad map can't do much).
FACTOR_LIMITS = (0.5, 1.5)
DEFAULT_STRENGTH = 0.8
DEFAULT_SEED = 1


# -- the surface: 9 places -> every point -------------------------------------------------

class Surface:
    """m(x, y) in % vs the centre, x and y as fractions of the screen from its centre
    (+x right, +y up), as photometry.place_offset gives them.

    With the 9 places (a 3 x 3 layout): the exact biquadratic sum c_ij x^i y^j (i, j =
    0..2) through all 9. Fewer places (at least 6): the 6-term bowl of
    photometry.fit_screen_map, through the centre. Beyond the outermost places the value
    is held constant (x and y clamped to the places' range): the exact surface overshoots
    near the screen edge otherwise.
    """

    def __init__(self, xs: Sequence[float], ys: Sequence[float], pct: Sequence[float]):
        x, y, p = (np.asarray(v, dtype=float) for v in (xs, ys, pct))
        if len(p) < 6:
            raise ValueError("a flat-field map needs at least 6 places (9 for the exact "
                             "surface)")
        self.xlim = (float(x.min()), float(x.max()))
        self.ylim = (float(y.min()), float(y.max()))
        self.kind = "6-term fit"
        self.coef = None
        if len(p) == 9:
            a = np.column_stack([x ** i * y ** j for i in range(3) for j in range(3)])
            if np.linalg.matrix_rank(a) == 9:
                self.coef = np.linalg.solve(a, p)
                self.kind = "exact 3x3 biquadratic"
        if self.coef is None:
            fit = photometry.fit_screen_map(list(zip(x, y)), p)
            if fit is None:
                raise ValueError("cannot fit these places")
            self.bowl = fit

    def __call__(self, x, y):
        x = np.clip(np.asarray(x, dtype=float), *self.xlim)
        y = np.clip(np.asarray(y, dtype=float), *self.ylim)
        if self.coef is None:
            return photometry.screen_map_at(self.bowl, x, y)
        out = np.zeros(np.broadcast(x, y).shape)
        k = 0
        for i in range(3):
            xi = x ** i
            for j in range(3):
                out = out + self.coef[k] * xi * y ** j
                k += 1
        return out


def gain(pct, strength: float):
    """The luminance multiplier that undoes a place pct % dimmer, at this strength."""
    factor = np.clip(1.0 + strength * np.asarray(pct, dtype=float) / 100.0, *FACTOR_LIMITS)
    return 1.0 / factor


# -- the stored map -----------------------------------------------------------------------

@dataclass
class FlatFieldMap:
    id: str
    name: str
    places: List[dict]                      # {place, x_frac, y_frac, pct}
    monitor: str = ""
    note: str = ""                           # monitor settings
    screen: dict = field(default_factory=dict)   # width_mm, height_mm, width_px, height_px
    strength: float = DEFAULT_STRENGTH
    seed: int = DEFAULT_SEED
    sources: List[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))

    def surface(self) -> Surface:
        return Surface([p["x_frac"] for p in self.places], [p["y_frac"] for p in self.places],
                       [p["pct"] for p in self.places])

    def describe(self, strength: Optional[float] = None) -> str:
        s = self.strength if strength is None else strength
        return f"{self.name}, {100 * s:.0f}%"


def new_map_id() -> str:
    import random
    import string
    return "FF_" + time.strftime("%Y%m%d") + "_" + "".join(
        random.choice(string.ascii_letters) for _ in range(4))


def map_from_dict(d: dict) -> FlatFieldMap:
    known = {f.name for f in dataclasses.fields(FlatFieldMap)}
    return FlatFieldMap(**{k: v for k, v in d.items() if k in known})


class FlatFieldStore:
    """python/calibration/flatfield_maps.json (git-ignored: local to each lab PC)."""

    def __init__(self, path: pathlib.Path = None):
        if path is None:
            path = pathlib.Path(__file__).resolve().parents[2] / "calibration" / \
                "flatfield_maps.json"
        self.path = pathlib.Path(path)
        self._items: Dict[str, FlatFieldMap] = {}
        if self.path.exists():
            for d in json.loads(self.path.read_text() or "[]"):
                m = map_from_dict(d)
                self._items[m.id] = m

    def all(self) -> List[FlatFieldMap]:
        return sorted(self._items.values(), key=lambda m: m.created)

    def get(self, mid: str) -> Optional[FlatFieldMap]:
        return self._items.get(mid)

    def find(self, key: str) -> Optional[FlatFieldMap]:
        """By id, or by name (the newest of that name)."""
        if key in self._items:
            return self._items[key]
        named = [m for m in self._items.values() if m.name == key]
        return max(named, key=lambda m: m.created) if named else None

    def put(self, m: FlatFieldMap) -> None:
        m.surface()                                   # an unusable map is refused here
        self._items[m.id] = m
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps([dataclasses.asdict(v) for v in self._items.values()],
                                  indent=2) + "\n")
        tmp.replace(self.path)


def places_from_csvs(paths: Sequence[pathlib.Path]) -> tuple:
    """Places from one or more uniformity CSVs (place, x_frac, y_frac, pct_vs_centre):
    the median per place over the files. Returns (places, notes)."""
    import csv
    import statistics
    per: Dict[str, dict] = {}
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                d = per.setdefault(row["place"], dict(x=float(row["x_frac"]),
                                                      y=float(row["y_frac"]), pct=[]))
                d["pct"].append(float(row["pct_vs_centre"]))
    notes = []
    places = []
    for place in [p for p in photometry.PLACES if p in per] + \
            [p for p in per if p not in photometry.PLACES]:
        d = per[place]
        if len(d["pct"]) != len(paths):
            notes.append(f"{place}: in {len(d['pct'])} of {len(paths)} files")
        places.append(dict(place=place, x_frac=round(d["x"], 4), y_frac=round(d["y"], 4),
                           pct=round(statistics.median(d["pct"]), 3), n=len(d["pct"])))
    if "centre" not in per:
        raise ValueError("the CSVs have no centre")
    return places, notes


# -- the correction for one window ---------------------------------------------------------

def _levels_for(calibration: LuminanceCalibration, luminance: np.ndarray) -> np.ndarray:
    """Continuous 8-bit grey level (0-255) for each luminance, from the scale's table.

    The same as calibration.brightness_for when the table rises steadily (a measured
    ColorCAL table does), but a single interpolation, fast enough for a whole screen."""
    t = np.asarray(calibration.table, dtype=float)
    if np.all(np.diff(t) > 0):
        b = np.interp(luminance, t, np.arange(t.size, dtype=float))
    else:
        b = calibration.brightness_for(luminance)
    return np.asarray(b, dtype=float) / 100.0 * 255.0


class FlatField:
    """The correction for one window: a gain and a dither offset for every physical
    pixel (row 0 = top of the screen), the background image, and patch correction."""

    def __init__(self, fmap: FlatFieldMap, calibration: LuminanceCalibration,
                 background_brightness: float, width_px: int, height_px: int,
                 strength: Optional[float] = None, seed: Optional[int] = None,
                 dither: bool = True):
        if not calibration.calibrated:
            raise ValueError("the flat-field correction needs a measured luminance scale "
                             "(Step 2): it works in cd/m2")
        self.map = fmap
        self.calibration = calibration
        self.strength = fmap.strength if strength is None else float(strength)
        self.seed = fmap.seed if seed is None else int(seed)
        self.dither = dither
        self.size = (int(width_px), int(height_px))
        self.background_brightness = float(background_brightness)
        self.background_cd_m2 = float(calibration.luminance_at(background_brightness * 100.0))
        surface = fmap.surface()
        self.surface_kind = surface.kind
        w, h = self.size
        xs = (np.arange(w) + 0.5) / w - 0.5                 # fraction of the width, +right
        ys = 0.5 - (np.arange(h) + 0.5) / h                 # fraction of the height, +up
        self.pct = surface(xs[None, :], ys[:, None])        # h x w, % vs centre
        self.gain = gain(self.pct, self.strength)
        self.offsets = (np.random.default_rng(self.seed).uniform(-0.5, 0.5, (h, w))
                        if dither else np.zeros((h, w)))
        self._background = None

    def _round(self, levels: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        return np.clip(np.round(levels + offsets), 0, 255)

    def background_levels(self) -> np.ndarray:
        """The corrected background as whole grey levels, h x w, row 0 = top."""
        if self._background is None:
            v = _levels_for(self.calibration, self.background_cd_m2 * self.gain)
            self._background = self._round(v, self.offsets)
        return self._background

    def target_levels(self) -> np.ndarray:
        """Before rounding: the continuous grey level each pixel should show on average."""
        return _levels_for(self.calibration, self.background_cd_m2 * self.gain)

    def _window(self, shape, centre_px):
        """Row/column indices of a patch on the screen, placed exactly as the backend
        places it (its top-left corner snapped to a whole pixel); off-screen parts use
        the nearest edge pixel."""
        h, w = shape
        cx, cy = centre_px
        c0, r0 = int(round(cx - w / 2.0)), int(round(cy - h / 2.0))
        rows = np.clip(np.arange(r0, r0 + h), 0, self.size[1] - 1)
        cols = np.clip(np.arange(c0, c0 + w), 0, self.size[0] - 1)
        return np.ix_(rows, cols)

    def correct_patch(self, brightness: np.ndarray, centre_px) -> np.ndarray:
        """A patch (0-1 brightness, as the generator makes it) corrected at its place:
        every pixel's luminance times the local gain, then to grey levels with the scale
        and the same static dither as the background. Returned as 0-1 brightness."""
        b = np.asarray(brightness, dtype=float)
        idx = self._window(b.shape, centre_px)
        lum = np.asarray(self.calibration.luminance_at(b * 100.0), dtype=float)
        v = _levels_for(self.calibration, lum * self.gain[idx])
        return self._round(v, self.offsets[idx]) / 255.0

    def gain_at(self, centre_px) -> float:
        x, y = centre_px
        r = min(max(int(y), 0), self.size[1] - 1)
        c = min(max(int(x), 0), self.size[0] - 1)
        return float(self.gain[r, c])

    def info(self) -> dict:
        """What a session records."""
        m = self.map
        return dict(name=m.name, id=m.id, monitor=m.monitor, note=m.note,
                    strength=self.strength, seed=self.seed, dither=self.dither,
                    surface=self.surface_kind, places=m.places, map_created=m.created,
                    luminance_scale=self.calibration.source,
                    background_cd_m2=round(self.background_cd_m2, 4),
                    describe=m.describe(self.strength))


NO_FLAT_FIELD = dict(name="none", describe="none")


def describe(info: Optional[dict]) -> str:
    """The report line."""
    return "flat-field correction: " + ((info or NO_FLAT_FIELD).get("describe") or "none")
