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

**Only valid sessions and valid thresholds go in.** A point reported blind or at the
floor in a healthy person -- usually the edge of the blind spot -- has no threshold to
average and is left out; ``n`` shows how many people contributed at each point.

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

COLUMNS = ("x_temporal_deg", "y_deg", "n_participants", "n_sessions", "mean_log10",
           "sd_log10", "mean_contrast")

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


@dataclass
class PriorTable:
    entries: Dict[Tuple[float, float], PriorEntry] = field(default_factory=dict)
    source: str = ""

    def lookup(self, eye: str, x_deg: float, y_deg: float) -> Optional[float]:
        """Starting guess (log10 contrast) for this eye and point, or None."""
        e = self.entries.get((_temporal_x(eye, x_deg), float(y_deg)))
        return e.mean_log10 if e is not None and e.n_participants >= 1 else None

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
            for (x, y), e in sorted(self.entries.items()):
                w.writerow([x, y, e.n_participants, e.n_sessions, round(e.mean_log10, 4),
                            "" if math.isnan(e.sd_log10) else round(e.sd_log10, 4),
                            round(10 ** e.mean_log10, 5)])
        return path

    @classmethod
    def load(cls, path: pathlib.Path) -> "PriorTable":
        path = pathlib.Path(path)
        entries = {}
        with path.open() as f:
            lines = [line for line in f if not line.startswith("#")]
        for row in csv.DictReader(lines):
            sd = row["sd_log10"]
            entries[(float(row["x_temporal_deg"]), float(row["y_deg"]))] = PriorEntry(
                int(row["n_participants"]), int(row["n_sessions"]),
                float(row["mean_log10"]), float(sd) if sd else float("nan"))
        return cls(entries, f"{path.name} ({cls._people(entries)} people)")

    @staticmethod
    def _people(entries: Dict[Tuple[float, float], PriorEntry]) -> int:
        return max((e.n_participants for e in entries.values()), default=0)


def build_prior_table(summary_paths: Iterable[pathlib.Path],
                      include_invalid_sessions: bool = False
                      ) -> Tuple[PriorTable, List[str]]:
    """Pool perimetry ``summary.csv`` files into a table. Returns (table, messages)."""
    # participant -> point -> list of log thresholds (one per eye/session)
    per_person: Dict[str, Dict[Tuple[float, float], List[float]]] = {}
    sessions: Dict[Tuple[float, float], int] = {}
    messages: List[str] = []
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
        used += 1
        person = rows[0]["participant"]
        eye = rows[0]["eye"]
        for row in rows:
            if row.get("valid") != "True" or not row.get("threshold_log10"):
                continue
            key = (_temporal_x(eye, float(row["x_deg"])), float(row["y_deg"]))
            per_person.setdefault(person, {}).setdefault(key, []).append(
                float(row["threshold_log10"]))
            sessions[key] = sessions.get(key, 0) + 1
        messages.append(f"used {path.parent.name}: {person}, {eye} eye")

    pooled: Dict[Tuple[float, float], List[float]] = {}
    for points in per_person.values():
        for key, values in points.items():
            pooled.setdefault(key, []).append(statistics.fmean(values))  # one per person

    entries = {}
    for key, values in pooled.items():
        sd = statistics.stdev(values) if len(values) > 1 else float("nan")
        entries[key] = PriorEntry(len(values), sessions[key], statistics.fmean(values), sd)
    table = PriorTable(entries, f"built {time.strftime('%Y-%m-%d')} from {used} sessions, "
                                f"{len(per_person)} people")
    return table, messages
