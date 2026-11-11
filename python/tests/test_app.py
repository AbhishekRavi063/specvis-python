"""The app window, driven offscreen: stores, presets, steps, start, results, history.

The stimulus process is replaced by an in-process run against a simulated observer,
so this checks everything the operator touches without opening PsychoPy.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import random
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from specvis.app import store  # noqa: E402
from specvis.app.main_window import (CompareWindow, HistoryWindow, MainWindow,  # noqa: E402
                                     ResultsWindow)
from specvis.procedure.perimetry import (run_perimetry, settings_from_dict,  # noqa: E402
                                         write_outputs)
from specvis.stimulus.backend import NullBackend  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


class Observer(NullBackend):
    """Right half blind, healthy elsewhere; never sees the blind spot."""

    def __init__(self, w, h, width_mm, dist_mm, rng):
        super().__init__(refresh_hz=120.0)
        self.w, self.h, self.width_mm, self.dist, self.rng = w, h, width_mm, dist_mm, rng

    def present_train(self, polarities, fpr, total, fixation=None, response_keys=()):
        r = super().present_train(polarities, fpr, total, fixation, response_keys)
        b = polarities[0].brightness
        c = float((b.max() - b.min()) / (b.max() + b.min()))
        px, py = polarities[0].centre_px
        x = math.degrees(math.atan((px - self.w / 2) * self.width_mm / self.w / self.dist))
        y = math.degrees(math.atan((self.h / 2 - py) * self.width_mm / self.w / self.dist))
        blind = x > 0 or (13 <= abs(x) <= 18 and -4 <= y <= 1)
        if c < 1e-6 or blind:
            p = 0.02
        else:
            p = 0.02 + 0.96 * (1 - math.exp(-10 ** (3.5 * (math.log10(c) - math.log10(0.04)))))
        if self.rng.random() < p:
            r.keys = [("space", 300.0)]
        return r

    def wait_for_key(self, keys, timeout_ms, fixation=None, clear_events=True):
        return (None, None)


app = QApplication.instance() or QApplication(sys.argv)
for name in ("information", "warning"):
    setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))   # no blocking dialogs

root = pathlib.Path(tempfile.mkdtemp(prefix="refrem_app_test_"))
perimetry_dir = root / "output" / "perimetry"
participants_file = root / "output" / "participants.json"
presets_dir = root / "presets"
shutil.copytree(store.PRESETS_DIR, presets_dir)

# An old session from before the app (participant not registered), which must appear.
old_settings = settings_from_dict(dict(participant_id="OLD01", eye="R"))
obs = Observer(3024, 1964, 302.0, 350.0, random.Random(1))
obs.open(3024, 1964, 0.5)
write_outputs(run_perimetry(old_settings, obs, rng=random.Random(1), show_instructions=False),
              perimetry_dir / "OLD01_R_20261101_100000")
before = sorted(p.name for p in perimetry_dir.rglob("*"))

print("1. stores")
people = store.ParticipantStore(participants_file)
people.add(store.Participant("PILOT02", age="25", group="pilot"))
try:
    people.add(store.Participant("bad name!"))
    require(False, "an ID with spaces or symbols is refused")
except ValueError:
    require(True, "an ID with spaces or symbols is refused")
require(store.ParticipantStore(participants_file).get("PILOT02").age == "25",
        "participants are saved and reloaded")
sessions = store.list_sessions(perimetry_dir)
everyone = [p.id for p in store.all_participants(people, sessions)]
require(everyone == ["OLD01", "PILOT02"], f"old session's participant is listed: {everyone}")
presets = store.list_presets(presets_dir)
require(len(presets) == 3 and presets[0].settings["viewing_distance_mm"] == 350.0,
        f"three shared presets: {[p.name for p in presets]}")
require(presets[1].settings.get("strategy") == "zest" and presets[1].settings["pilot_mode"],
        "the pilot (table) preset uses ZEST (full) with pilot mode on, as the plan says")

print("2. the window")
win = MainWindow(perimetry_dir, participants_file, presets_dir, perimetry_dir / "prior_table.csv")
win.monitor_pixels = lambda: (3024, 1964)          # the MacBook, whatever runs the test
require(win.people_list.count() == 2, "both participants in step 1")
win.people_list.setCurrentRow(1)                    # PILOT02
require(win.current_participant().id == "PILOT02", "PILOT02 selected")

win.preset_box.setCurrentIndex(1)                   # lab monitor pilot: width unknown
win.apply_preset()
require(any("screen width" in p for p in win.problems()),
        "lab preset without a width: start is blocked until it is measured")
require(not win.start_button.isEnabled(), "start button disabled")

win.preset_box.setCurrentIndex(0)                   # MacBook practice
win.apply_preset()
require(win.problems() == [], f"MacBook preset: ready to start ({win.problems()})")
require("fits" in win.fit_label.text(), "step 2 says the grid fits")
win.distance_cm.setValue(57.0)
require(any("does not fit" in p for p in win.problems()),
        "57 cm on the laptop: the grid does not fit, and step 2 says so")
win.distance_cm.setValue(35.0)

win.eye_box.setCurrentIndex(1)
require(win.settings.eye == "L", "eye choice reaches the settings")
win.eye_box.setCurrentIndex(0)
win.sf.setValue(2.0)
require(win.settings.spatial_frequency_cpd == 2.0, "stimulus page reaches the settings")
win.sf.setValue(4.0)
win.extent_box.setCurrentText("central")
require(win.settings.extent == "central", "procedure page reaches the settings")
win.sd_stop.setValue(0.20)
require(abs(win.settings.zest.sd_stop - 0.20) < 1e-9, "the stopping rule reaches ZEST")
for i in range(5):
    win.step_list.setCurrentRow(i)
    require(win.pages.currentIndex() == i, f"step {i + 1} opens")
require("PILOT02" in win.summary.toPlainText() and "central" in win.summary.toPlainText(),
        "step 5 summary shows the participant and the grid")

print("3. settings files")
saved = store.save_settings_file(presets_dir / "mine.json", "mine", "test",
                                 json.loads(json.dumps(win.job()["settings"])))
win.apply_settings_values({"extent": "full"})
win.apply_settings_values(store.load_settings_file(saved).settings)
require(win.settings.extent == "central" and abs(win.settings.zest.sd_stop - 0.20) < 1e-9,
        "save then load settings restores them")
require("participant_id" not in json.loads(saved.read_text())["settings"],
        "a settings file never stores a participant")

print("4. start -> results -> other eye")
shown = []


def fake_launch(mode, job_path, on_done):
    job = json.loads(job_path.read_text())
    s = settings_from_dict(job["settings"])
    backend = Observer(3024, 1964, s.screen_width_mm, s.viewing_distance_mm, random.Random(7))
    backend.open(3024, 1964, 0.5)
    out = run_perimetry(s, backend, rng=random.Random(7), show_instructions=False)
    folder = write_outputs(out, perimetry_dir / f"{s.participant_id}_{s.eye}_2026111{len(shown)}_120000")
    job_path.with_suffix(".result.json").write_text(json.dumps({"folder": str(folder),
                                                                 "valid": out.valid}))
    on_done(0)


win.apply_settings_values({"strategy": "zest", "pilot_mode": True})   # table people
require(win.settings.strategy == "zest", "ZEST (full) chosen for the table people")
win._launch = fake_launch
win.show_results = lambda info: shown.append(info)
win.start_test()
require(len(shown) == 1 and shown[0].participant_id == "PILOT02" and shown[0].eye == "R",
        "a test runs and its results open")
require(shown[0].map_path.exists() and shown[0].valid, "map saved; session valid")
dlg = ResultsWindow(win, shown[0])
require(dlg.table.rowCount() == 46, f"results table has every point ({dlg.table.rowCount()})")
win.eye_box.setCurrentIndex(1)
win.start_test()
require(len(shown) == 2 and shown[1].eye == "L", "the other eye runs next")
require(win.people_list.currentItem().text().startswith("PILOT02") and
        "2 sessions" in win.people_list.currentItem().text(), "step 1 shows 2 sessions")
history = HistoryWindow(win, "PILOT02", perimetry_dir)
require(len(history.sessions) == 2, "history lists both sessions")
CompareWindow(win, history.sessions[0], history.sessions[1])
require(True, "compare window opens with two sessions")

print("5. the normal table, and nothing old was touched")
win.build_table()
require((perimetry_dir / "prior_table.csv").exists(), "normal table built from valid sessions")
after = sorted(p.name for p in perimetry_dir.rglob("*"))
require(set(before) <= set(after), "every file of the old session is still there")
require(json.loads((perimetry_dir / "OLD01_R_20261101_100000" / "session.json").read_text())
        ["settings"]["participant_id"] == "OLD01", "and unchanged")

win.close()
shutil.rmtree(root, ignore_errors=True)
sys.exit(0 if ok else 1)
