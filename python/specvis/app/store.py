"""Where the app keeps things: participants, presets and finished sessions.

No Qt here, so all of it is testable without a window.

**Participant data never leaves ``python/output/``**, which git ignores: the records
(``participants.json``) and every session folder. Only **presets** live in the
project (``python/presets/``), so the settings everyone uses travel with the code.

Participants are identified by an **ID only** -- no names -- so the files can be
shared inside the team without exposing who was tested.

Nothing here deletes a session. Old sessions, including ones run before the app
existed, are read and listed; participants found only in sessions are listed too.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PYTHON_DIR = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = PYTHON_DIR / "output"
PERIMETRY_DIR = DATA_DIR / "perimetry"
PARTICIPANTS_FILE = DATA_DIR / "participants.json"
PRESETS_DIR = PYTHON_DIR / "presets"
PRIOR_TABLE_FILE = PERIMETRY_DIR / "prior_table.csv"
#: Luminance scales describe a monitor, not a person, so they travel with the code
#: and every computer driving the same lab monitor uses the same calibration.
SCALES_FILE = PYTHON_DIR / "calibration" / "luminance_scales.json"

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")


def valid_id(text: str) -> bool:
    """Letters, digits, '_' and '-': the ID names a folder, so nothing else."""
    return bool(_ID.match(text or ""))


@dataclass
class Participant:
    id: str
    age: str = ""
    sex: str = ""                   # Specvis: gender
    group: str = ""                 # e.g. pilot, healthy control, patient
    acuity_left: str = ""
    acuity_right: str = ""
    notes: str = ""                 # Specvis: additional information
    # Specvis's other patient fields. Optional: the study uses IDs, not names.
    first_name: str = ""
    last_name: str = ""
    birth_date: str = ""            # YYYY-MM-DD
    phone: str = ""
    email: str = ""
    city: str = ""
    postal_code: str = ""
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))
    registered: bool = True         # False: found only in session folders

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("registered")
        return d

    @property
    def name(self) -> str:
        return " ".join(x for x in (self.first_name, self.last_name) if x)


class ParticipantStore:
    def __init__(self, path: pathlib.Path = PARTICIPANTS_FILE):
        self.path = pathlib.Path(path)
        self._items: Dict[str, Participant] = {}
        self.load()

    def load(self) -> None:
        self._items = {}
        if self.path.exists():
            for d in json.loads(self.path.read_text() or "[]"):
                known = {f.name for f in dataclasses.fields(Participant)}
                p = Participant(**{k: v for k, v in d.items() if k in known})
                self._items[p.id] = p

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps([p.to_json() for p in self._items.values()], indent=2) + "\n")
        tmp.replace(self.path)          # never leave a half-written file

    def all(self) -> List[Participant]:
        return sorted(self._items.values(), key=lambda p: p.id.lower())

    def get(self, pid: str) -> Optional[Participant]:
        return self._items.get(pid)

    def add(self, participant: Participant) -> None:
        if not valid_id(participant.id):
            raise ValueError("an ID may use letters, digits, '_' and '-' only (max 40)")
        if participant.id in self._items:
            raise ValueError(f"participant {participant.id} already exists")
        participant.registered = True
        self._items[participant.id] = participant
        self.save()

    def update(self, participant: Participant) -> None:
        if participant.id not in self._items:
            raise KeyError(participant.id)
        participant.registered = True
        self._items[participant.id] = participant
        self.save()


@dataclass
class SessionInfo:
    folder: pathlib.Path
    participant_id: str
    eye: str
    started: str
    valid: bool
    aborted: bool
    pilot_mode: bool
    extent: str
    trials: int
    duration_min: float
    checks: Dict[str, bool]
    rates: Dict[str, float]

    @property
    def map_path(self) -> pathlib.Path:
        return self.folder / "map.png"

    @property
    def summary_path(self) -> pathlib.Path:
        return self.folder / "summary.csv"

    @property
    def label(self) -> str:
        state = "VALID" if self.valid else ("stopped" if self.aborted else "NOT valid")
        return f"{self.started}   {self.eye} eye   {state}   {self.trials} trials"


def read_session(folder: pathlib.Path) -> Optional[SessionInfo]:
    """A perimetry session folder, or None if it is not one (or is unreadable)."""
    try:
        meta = json.loads((folder / "session.json").read_text())
    except (OSError, ValueError):
        return None
    if meta.get("kind") != "perimetry":
        return None
    s = meta.get("settings", {})
    return SessionInfo(
        folder=folder, participant_id=str(s.get("participant_id", "?")).strip(),
        eye=s.get("eye", "?"), started=meta.get("started", ""),
        valid=bool(meta.get("session_valid")), aborted=bool(meta.get("aborted")),
        pilot_mode=bool(s.get("pilot_mode")), extent=s.get("extent", ""),
        trials=int(meta.get("counts", {}).get("trials", 0)),
        duration_min=float(meta.get("duration_s", 0.0)) / 60.0,
        checks=meta.get("checks", {}),
        rates=dict(false_alarms=meta.get("false_alarm_rate"), misses=meta.get("miss_rate"),
                   fixation=meta.get("fixation_loss_rate")))


def list_sessions(perimetry_dir: pathlib.Path = PERIMETRY_DIR,
                  participant_id: Optional[str] = None) -> List[SessionInfo]:
    """Every session, newest first; optionally for one participant."""
    if not perimetry_dir.exists():
        return []
    out = []
    for folder in perimetry_dir.iterdir():
        if folder.is_dir():
            info = read_session(folder)
            if info and (participant_id is None or info.participant_id == participant_id):
                out.append(info)
    return sorted(out, key=lambda i: i.started, reverse=True)


def all_participants(store: ParticipantStore, sessions: List[SessionInfo]) -> List[Participant]:
    """Registered participants plus anyone found only in session folders."""
    people = {p.id: p for p in store.all()}
    for s in sessions:
        if s.participant_id and s.participant_id not in people:
            people[s.participant_id] = Participant(s.participant_id, registered=False,
                                                   notes="found in session folders")
    return sorted(people.values(), key=lambda p: p.id.lower())


# -- presets ---------------------------------------------------------------------

@dataclass
class Preset:
    name: str
    description: str
    settings: dict              # partial: only what the preset fixes
    path: Optional[pathlib.Path] = None


def list_presets(directory: pathlib.Path = PRESETS_DIR) -> List[Preset]:
    if not directory.exists():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            d = json.loads(path.read_text())
            out.append(Preset(d["name"], d.get("description", ""), d.get("settings", {}), path))
        except (OSError, ValueError, KeyError):
            continue
    return out


def save_settings_file(path: pathlib.Path, name: str, description: str,
                       settings: dict) -> pathlib.Path:
    """A settings file in the preset format -- loadable later, like Specvis's .sset."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in settings.items() if k not in ("participant_id", "eye")}
    path.write_text(json.dumps(dict(name=name, description=description, settings=keep),
                               indent=2) + "\n")
    return path


def load_settings_file(path: pathlib.Path) -> Preset:
    d = json.loads(pathlib.Path(path).read_text())
    if "settings" not in d:
        raise ValueError("not a ReFreM settings file")
    return Preset(d.get("name", pathlib.Path(path).stem), d.get("description", ""),
                  d["settings"], pathlib.Path(path))


# -- luminance scales (Specvis step 2) -----------------------------------------------

UNCALIBRATED = "UNCALIBRATED (no photometer)"


@dataclass
class LuminanceScale:
    """Six photometer readings at brightness 0/20/40/60/80/100 %, as Specvis stores them."""

    id: str
    name: str
    measurements: List[float]          # cd/m2 at 0, 20, 40, 60, 80, 100 %
    hue: float = 0.0
    saturation: float = 0.0
    notes: str = ""                    # Specvis: additional information
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))

    def calibration(self):
        from ..core.luminance import LuminanceCalibration
        return LuminanceCalibration.from_measurements(self.measurements, f"scale {self.name}")


def new_scale_id() -> str:
    import random
    import string
    return "S_" + time.strftime("%Y%m%d") + "_" + "".join(
        random.choice(string.ascii_letters) for _ in range(4))


class ScaleStore:
    def __init__(self, path: pathlib.Path = SCALES_FILE):
        self.path = pathlib.Path(path)
        self._items: Dict[str, LuminanceScale] = {}
        if self.path.exists():
            for d in json.loads(self.path.read_text() or "[]"):
                known = {f.name for f in dataclasses.fields(LuminanceScale)}
                s = LuminanceScale(**{k: v for k, v in d.items() if k in known})
                self._items[s.id] = s

    def all(self) -> List[LuminanceScale]:
        return sorted(self._items.values(), key=lambda s: s.name.lower())

    def get(self, sid: str) -> Optional[LuminanceScale]:
        return self._items.get(sid)

    def put(self, scale: LuminanceScale) -> None:
        if len(scale.measurements) != 6:
            raise ValueError("a scale needs six readings: 0, 20, 40, 60, 80, 100 %")
        self._items[scale.id] = scale
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps([dataclasses.asdict(s) for s in self._items.values()],
                                  indent=2) + "\n")
        tmp.replace(self.path)


def fit_quality(measurements: List[float]) -> Dict[str, float]:
    """Specvis's Fit window: chi-squared, p-value and STD of the degree-2 fit."""
    import numpy as np
    from ..core.luminance import MEASUREMENT_POINTS
    x = np.array(MEASUREMENT_POINTS)
    y = np.asarray(measurements, dtype=float)
    fitted = np.polyval(np.polyfit(x, y, 2), x)
    residual = y - fitted
    safe = np.where(fitted > 1e-9, fitted, 1e-9)
    chi2 = float(np.sum(residual ** 2 / safe))
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, df=len(x) - 3))
    except ImportError:                                   # pragma: no cover
        p = float("nan")
    return dict(chi_squared=chi2, p_value=p, std=float(np.std(residual, ddof=0)))
