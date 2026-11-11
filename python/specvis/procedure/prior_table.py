"""Our own table of normal thresholds, built from pilot perimetry sessions.

SITA starts every point from a large age-normal database that Zeiss does not
publish. This is the substitute Pragathi proposed: test healthy people first, then use
what they saw as the starting guess for everyone after.

**Eyes are aligned before pooling.** A right eye's temporal field is on the right, a
left eye's on the left, and the blind spot follows. Points are stored with x measured
**temporal-positive** (the right eye's own coordinates; a left eye's x is mirrored),
so both eyes of everyone line up, and looked up the same way.

**People are counted, not eyes.** A participant's two eyes are averaged first, so a
table from 5 people has n = 5 at a point, not 10.

**Only ZEST (full) sessions with pilot mode on go in** (the study plan: the table
people are tested with the careful method and no starting guesses, so the table does
not depend on the method it is used to check). Anything else is skipped and named.

**Only valid sessions and valid thresholds go in.** A point reported blind or at the
floor in a healthy person -- usually the edge of the blind spot -- has no threshold to
average and is left out; ``n`` shows how many people contributed at each point.

**Smoothed as a hill of vision, by the published model.** With only a few people each
point's average is noisy, so the table value at every point is read off one smooth
hill fitted to everyone's thresholds. The hill is the published description of
contrast sensitivity across the visual field:

* sensitivity falls **linearly in log units (dB) with eccentricity** -- Pointer & Hess
  (1989), Vision Research 29:1133; Baldwin, Meese & Baker (2012), Journal of Vision
  12(11):23, abstract and p. 6;
* at a rate that differs **by direction**: one slope per cardinal hemi-meridian
  (temporal, nasal, superior, inferior), combined at oblique angles **elliptically** --
  Baldwin et al. (2012), "Interpolation between cardinal meridians" and the Matlab code
  of their Appendix B (p. 17).

So, in log10 threshold contrast (higher = worse; dB = -20 x log10)::

    log10 threshold = a + sqrt((s_h * x)^2 + (s_v * y)^2)

with ``x`` temporal-positive and ``y`` up, in degrees; ``s_h`` is the temporal or the
nasal slope (by the sign of x), ``s_v`` the superior or inferior slope (by the sign of
y). ``a`` is the hill top (fixation). For one spatial frequency, degrees and the
papers' carrier cycles differ only by that frequency, so the slopes are reported both
ways. Only the *form* comes from the papers; ``a`` and the four slopes are fitted to
our own pilot thresholds. Baldwin et al.'s steeper first limb (the "witch's hat" knee,
about 9 cycles = 2.25 deg at 4 c/deg) is not modelled: all our points but the four
innermost lie beyond it.

Each point's table value **is** the hill's height there (``hill_log10``); its plain
average over people is kept beside it (``raw_mean_log10``) for checking. With too few
values to fit a hill, the plain averages are used. The SD used for "worse than
normal" is pooled over all points.

How many people this needs is a judgement, not a rule: a starting guess is usable from
about 10 and good from 15-20; deciding "worse than normal" needs 20-30, age-matched to
the patients. Young lab members are fine for testing the method, not for the final
table.
"""

from __future__ import annotations

import csv
import math
import pathlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

#: The table file is written in contrast dB (-20 x log10 contrast; higher = better), the
#: unit people read. In memory everything stays in log10 contrast, the unit ZEST uses.
COLUMNS = ("x_temporal_deg", "y_deg", "n_participants", "n_sessions", "normal_db",
           "sd_db", "normal_contrast", "plain_average_db", "hill_db")

#: A normal range is only given where at least this many people contributed.
MIN_PEOPLE_FOR_RANGE = 3
#: Floor on the SD used for "worse than normal", so a few very similar people do not
#: make the normal range implausibly narrow.
MIN_SD_LOG = 0.10


def _temporal_x(eye: str, x_deg: float) -> float:
    return x_deg if eye == "R" else -x_deg


@dataclass
class PriorEntry:
    n_participants: int
    n_sessions: int
    mean_log10: float
    sd_log10: float
    raw_mean_log10: float = float("nan")
    hill_log10: float = float("nan")


@dataclass
class PriorTable:
    entries: Dict[Tuple[float, float], PriorEntry] = field(default_factory=dict)
    source: str = ""

    def lookup(self, eye: str, x_deg: float, y_deg: float) -> Optional[float]:
        """Starting guess (log10 contrast) for this eye and point, or None."""
        e = self.entries.get((_temporal_x(eye, x_deg), float(y_deg)))
        return e.mean_log10 if e is not None and not math.isnan(e.mean_log10) else None

    def normal_range(self, eye: str, x_deg: float, y_deg: float
                     ) -> Optional[Tuple[float, float]]:
        """(mean, sd) in log10 contrast, when enough people contributed; else None."""
        e = self.entries.get((_temporal_x(eye, x_deg), float(y_deg)))
        if e is None or e.n_participants < MIN_PEOPLE_FOR_RANGE or math.isnan(e.sd_log10):
            return None
        return e.mean_log10, max(e.sd_log10, MIN_SD_LOG)

    # -- files -------------------------------------------------------------------

    def save(self, path: pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            f.write(f"# ReFreM perimetry normal table. {self.source}\n")
            w = csv.writer(f)
            w.writerow(COLUMNS)
            def db(v):            # log10 contrast -> dB; a spread scales without the sign
                return "" if math.isnan(v) else round(-20.0 * v, 3)
            for (x, y), e in sorted(self.entries.items()):
                w.writerow([x, y, e.n_participants, e.n_sessions, db(e.mean_log10),
                            "" if math.isnan(e.sd_log10) else round(20.0 * e.sd_log10, 3),
                            round(10 ** e.mean_log10, 5), db(e.raw_mean_log10),
                            db(e.hill_log10)])
        return path

    @classmethod
    def load(cls, path: pathlib.Path) -> "PriorTable":
        path = pathlib.Path(path)
        entries = {}
        with path.open() as f:
            lines = [line for line in f if not line.startswith("#")]
        for row in csv.DictReader(lines):
            def num(name, scale=1.0):
                v = row.get(name) or ""
                return float(v) * scale if v else float("nan")
            if "normal_db" in row:                       # current files: dB
                values = (num("normal_db", -0.05), num("sd_db", 0.05),
                          num("plain_average_db", -0.05), num("hill_db", -0.05))
            else:                                        # older files: log10 contrast
                values = (num("mean_log10"), num("sd_log10"), num("raw_mean_log10"),
                          num("hill_log10"))
            entries[(float(row["x_temporal_deg"]), float(row["y_deg"]))] = PriorEntry(
                int(row["n_participants"]), int(row["n_sessions"]), *values)
        return cls(entries, f"{path.name} ({cls._people(entries)} people)")

    @staticmethod
    def _people(entries: Dict[Tuple[float, float], PriorEntry]) -> int:
        return max((e.n_participants for e in entries.values()), default=0)


def _spatial_frequency(summary_path: pathlib.Path) -> Optional[float]:
    """The session's spatial frequency (c/deg), from the session.json beside it."""
    import json
    try:
        meta = json.loads((summary_path.parent / "session.json").read_text())
        return float(meta["settings"]["spatial_frequency_cpd"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


#: The method the table people are tested with (``STRATEGY_NAMES[ZEST_FULL]`` in
#: perimetry.py, written to summary.csv's ``method`` column).
TABLE_METHOD = "ZEST (full)"


def build_prior_table(summary_paths: Iterable[pathlib.Path],
                      include_invalid_sessions: bool = False,
                      any_method: bool = False
                      ) -> Tuple[PriorTable, List[str]]:
    """Pool perimetry ``summary.csv`` files into a table. Returns (table, messages)."""
    # participant -> point -> list of log thresholds (one per eye/session)
    per_person: Dict[str, Dict[Tuple[float, float], List[float]]] = {}
    sessions: Dict[Tuple[float, float], int] = {}
    all_points = set()
    messages: List[str] = []
    frequencies = set()
    used = 0
    for path in summary_paths:
        path = pathlib.Path(path)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            messages.append(f"skipped {path}: empty")
            continue
        if rows[0].get("session_valid") != "True" and not include_invalid_sessions:
            messages.append(f"skipped {path.parent.name}: session not valid")
            continue
        method, pilot = rows[0].get("method", ""), rows[0].get("pilot_mode", "")
        if not any_method and (method != TABLE_METHOD or pilot != "True"):
            messages.append(f"skipped {path.parent.name}: not {TABLE_METHOD} with pilot mode on "
                            f"(was {method or 'unknown method'}, pilot mode "
                            f"{'on' if pilot == 'True' else 'off'})")
            continue
        used += 1
        sf = _spatial_frequency(path)
        if sf:
            frequencies.add(sf)
        person = rows[0]["participant"]
        eye = rows[0]["eye"]
        for row in rows:
            key = (_temporal_x(eye, float(row["x_deg"])), float(row["y_deg"]))
            all_points.add(key)
            if row.get("valid") != "True" or not row.get("threshold_log10"):
                continue
            per_person.setdefault(person, {}).setdefault(key, []).append(
                float(row["threshold_log10"]))
            sessions[key] = sessions.get(key, 0) + 1
        messages.append(f"used {path.parent.name}: {person}, {eye} eye")

    pooled: Dict[Tuple[float, float], List[float]] = {}
    for points in per_person.values():
        for key, values in points.items():
            pooled.setdefault(key, []).append(statistics.fmean(values))  # one per person

    hill = fit_hill(pooled)
    # Pooled SD: every point's spread between people, averaged (by degrees of freedom).
    ss = sum(sum((v - statistics.fmean(vals)) ** 2 for v in vals) for vals in pooled.values())
    dof = sum(len(vals) - 1 for vals in pooled.values())
    pooled_sd = math.sqrt(ss / dof) if dof > 0 else float("nan")

    entries = {}
    for key in sorted(all_points | set(pooled)):
        values = pooled.get(key, [])
        n = len(values)
        raw = statistics.fmean(values) if values else float("nan")
        h = hill(*key) if hill else float("nan")
        if hill:
            mean = h                      # the published model's value at this point
        elif n:
            mean = raw                    # too little data for a hill: plain average
        else:
            continue
        sd = pooled_sd if n > 1 else float("nan")
        entries[key] = PriorEntry(n, sessions.get(key, 0), mean, sd, raw, h)
    table = PriorTable(entries, f"built {time.strftime('%Y-%m-%d')} from {used} sessions, "
                                f"{len(per_person)} people"
                                + (", hill-of-vision smoothed" if hill else ""))
    if hill:
        sf = frequencies.pop() if len(frequencies) == 1 else None
        messages.append("hill of vision fitted: " + hill.describe(sf))
    return table, messages


class Hill:
    """The published hill of vision (Pointer & Hess 1989; Baldwin, Meese & Baker 2012).

    log10 threshold = a + sqrt((s_h * x)^2 + (s_v * y)^2), x temporal-positive, y up,
    in degrees. s_h = temporal or nasal slope, s_v = superior or inferior slope, in
    log10 contrast per degree (x 20 = dB per degree). Equivalently, in dB of
    sensitivity: S = S_top - m(angle) * eccentricity, with m(angle) interpolated
    elliptically between the four hemi-meridian slopes (Baldwin et al. 2012, App. B).
    """

    SLOPES = ("temporal", "nasal", "superior", "inferior")

    def __init__(self, top_log10: float, slopes: Tuple[float, float, float, float]):
        self.a = float(top_log10)
        self.slopes = tuple(float(v) for v in slopes)

    def _sh_sv(self, x: float, y: float) -> Tuple[float, float]:
        t, n, u, d = self.slopes
        return (t if x > 0 else n), (u if y > 0 else d)

    def __call__(self, x: float, y: float) -> float:
        sh, sv = self._sh_sv(x, y)
        return self.a + math.hypot(sh * x, sv * y)

    def describe(self, spatial_frequency_cpd: Optional[float] = None) -> str:
        parts = []
        for name, s in zip(self.SLOPES, self.slopes):
            text = f"{name} {20 * s:.2f} dB/deg"
            if spatial_frequency_cpd:
                text += f" ({20 * s / spatial_frequency_cpd:.2f} dB/cycle)"
            parts.append(text)
        return (f"hill top {-20 * self.a:.1f} dB ({10 ** self.a:.4f} contrast); "
                + ", ".join(parts))


def fit_hill(pooled: Dict[Tuple[float, float], List[float]]) -> Optional[Hill]:
    """Least squares over every person's value at every point (Levenberg-Marquardt,
    numpy only). None when there is too little data: under 12 values, or no points on
    one side of either midline (that side's slope could not be estimated)."""
    import numpy as np
    xs, ys, vs = [], [], []
    for (x, y), values in pooled.items():
        for v in values:
            xs.append(x)
            ys.append(y)
            vs.append(v)
    x, y, v = (np.asarray(a, dtype=float) for a in (xs, ys, vs))
    if v.size < 12 or not ((x > 0).any() and (x < 0).any() and (y > 0).any()
                           and (y < 0).any()):
        return None
    right, up = x > 0, y > 0

    def model(p):
        sh = np.where(right, p[1], p[2])
        sv = np.where(up, p[3], p[4])
        r = np.sqrt((sh * x) ** 2 + (sv * y) ** 2)
        return p[0] + r, sh, sv, r

    def jacobian(p, sh, sv, r):
        r = np.maximum(r, 1e-12)
        dsh, dsv = sh * x * x / r, sv * y * y / r
        return np.column_stack([np.ones_like(x), np.where(right, dsh, 0.0),
                                np.where(right, 0.0, dsh), np.where(up, dsv, 0.0),
                                np.where(up, 0.0, dsv)])

    p = np.array([v.min(), 0.03, 0.03, 0.03, 0.03])
    lam = 1e-3
    pred, sh, sv, r = model(p)
    cost = float(((v - pred) ** 2).sum())
    for _ in range(200):
        j = jacobian(p, sh, sv, r)
        g = j.T @ (v - pred)
        h = j.T @ j
        step = np.linalg.solve(h + lam * np.diag(np.diag(h) + 1e-12), g)
        trial = p + step
        trial[1:] = np.maximum(trial[1:], 0.0)          # slopes cannot be negative
        t_pred, t_sh, t_sv, t_r = model(trial)
        t_cost = float(((v - t_pred) ** 2).sum())
        if t_cost < cost:
            done = cost - t_cost < 1e-12
            p, pred, sh, sv, r, cost = trial, t_pred, t_sh, t_sv, t_r, t_cost
            lam = max(lam / 10, 1e-9)
            if done:
                break
        else:
            lam *= 10
            if lam > 1e9:
                break
    return Hill(p[0], tuple(p[1:]))
