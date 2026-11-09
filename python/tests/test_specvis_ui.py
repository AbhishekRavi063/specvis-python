"""The Specvis-style window, driven offscreen, with the test process replaced by an
in-process run against a simulated observer."""

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

import run_test_process  # noqa: E402
from specvis.app import specvis_map, store  # noqa: E402
from specvis.app.specvis_ui import (CompareDialog, PatientResultsDialog,  # noqa: E402
                                    ResultsDialog, SpecvisWindow)
from specvis.procedure.perimetry import run_perimetry, settings_from_dict, write_outputs  # noqa: E402
from specvis.stimulus.backend import NullBackend  # noqa: E402

ok = True


def require(condition: bool, what: str) -> None:
    global ok
    print(("  ok   " if condition else "  FAIL ") + what)
    if not condition:
        ok = False


class Observer(NullBackend):
    def __init__(self, width_mm, dist_mm, rng):
        super().__init__(refresh_hz=120.0)
        self.width_mm, self.dist, self.rng = width_mm, dist_mm, rng
        self.pressed_pause = False

    def present_train(self, polarities, fpr, total, fixation=None, response_keys=()):
        r = super().present_train(polarities, fpr, total, fixation, response_keys)
        b = polarities[0].brightness
        c = float((b.max() - b.min()) / (b.max() + b.min()))
        px, py = polarities[0].centre_px
        w, h = self.size_px
        x = math.degrees(math.atan((px - w / 2) * self.width_mm / w / self.dist))
        y = math.degrees(math.atan((h / 2 - py) * self.width_mm / w / self.dist))
        blind = x > 0 or (13 <= abs(x) <= 18 and -4 <= y <= 1)
        p = 0.02 if c < 1e-6 or blind else \
            0.02 + 0.96 * (1 - math.exp(-10 ** (3.5 * (math.log10(c) - math.log10(0.04)))))
        if self.rng.random() < p:
            r.keys = [("space", 300.0)]
        return r

    def wait_for_key(self, keys, timeout_ms, fixation=None, clear_events=True):
        if "p" in keys and not self.pressed_pause:           # press pause once
            self.pressed_pause = True
            return ("p", 10.0)
        return (None, None)


app = QApplication.instance() or QApplication(sys.argv)
for name in ("information", "warning"):
    setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)

root = pathlib.Path(tempfile.mkdtemp(prefix="refrem_specvis_ui_"))
pdir = root / "output" / "perimetry"
pdir.mkdir(parents=True)
win = SpecvisWindow(pdir, root / "output" / "participants.json", store.PRESETS_DIR,
                    pdir / "prior_table.csv", root / "calibration" / "scales.json")
win.monitor_pixels = lambda: (3024, 1964)

print("1. Step 1 - patient")
require(win.pages.currentIndex() == 0 and win.left_button.text() == "Exit",
        "opens on Step 1 with Exit")
win._go(1)
require(win.pages.currentIndex() == 0, "Next is refused until a patient is chosen")
win.people.add(store.Participant("PILOT02", first_name="", group="pilot", age="25"))
win.patient = win.people.get("PILOT02")
win._show_patient()
require(win.f_patient_id.text() == "PILOT02", "patient shown in the ID field")
win.apply_preset(1)                                     # MacBook practice
require(win.f_settings.text().startswith("MacBook"), "preset name shown in Settings")
require(win.problems() == [], f"ready to start: {win.problems()}")

print("2. Step 2 - screen and luminance scale")
win._go(1)
require(win.left_button.text() == "Back", "Back replaces Exit after Step 1")
require(win.f_res_x.text() == "3024" and "fits" in win.fit_label.text(),
        "screen bounds and 'the grid fits'")
require(abs(float(win.f_field_x.text()) - 46.67) < 0.05, f"tested field {win.f_field_x.text()} deg")
scale = store.LuminanceScale(store.new_scale_id(), "LAB_TEST", [0.5, 10, 40, 90, 160, 250])
win.scales.put(scale)
win.scale_id = scale.id
win._refresh()
require(win.f_scale_name.text() == "LAB_TEST" and "nominal" not in win.f_bg_lum.text(),
        f"measured scale chosen: background {win.f_bg_lum.text()} cd/m2")
q = store.fit_quality(scale.measurements)
require(q["chi_squared"] < 1 and 0 <= q["p_value"] <= 1, f"fit quality {q}")
s2, cal = run_test_process.settings_and_calibration(win.job())
require(cal.calibrated and abs(s2.mean_luminance - float(cal.luminance_at(50))) < 1e-9,
        "the test process gets the scale and sets the background from it")
win.scale_id = None
win._refresh()

print("3. Steps 3 and 4")
win._go(2)
win.f_sf.setValue(2.0)
require(win.settings.spatial_frequency_cpd == 2.0, "stimulus field reaches the settings")
win.f_sf.setValue(4.0)
win.f_iti_c.setValue(400.0)
win.f_iti_r.setValue(300.0)
require(win.settings.iti_ms == (400.0, 700.0), "interval const/rand -> 400-700 ms")
win.f_iti_c.setValue(300.0)
win.f_iti_r.setValue(500.0)
require("dB" in win.f_db_range.text(), f"contrast range shown: {win.f_db_range.text()}")
win._go(3)
win.f_fix_colour.setCurrentText("black")
win.f_key_pause.setCurrentText("p")
require(win.settings.fixation_colour == "black", "fixation colour reaches the settings")
win.f_key_answer.setCurrentText("p")
require(any("different" in p for p in win.problems()), "same key for answer and pause is refused")
win.f_key_answer.setCurrentText("space")
win._apply(dict(pilot_mode=True, zest=dict(sd_stop=0.2)))
require(win.settings.pilot_mode and abs(win.settings.zest.sd_stop - 0.2) < 1e-9,
        "procedure settings applied")

print("4. START -> results -> other eye")
win._go(4)
require(not win.next_button.isVisible() and win.start_button.isEnabled(),
        "START page: no Next, Start enabled")
require("PILOT02" in win.f_summary.toPlainText(), "summary shows the patient")
shown = []


def fake_launch(mode, job_path, on_done):
    job = json.loads(job_path.read_text())
    s, cal = run_test_process.settings_and_calibration(job)
    backend = Observer(s.screen_width_mm, s.viewing_distance_mm, random.Random(len(shown) + 3))
    backend.open(3024, 1964, 0.5)
    out = run_perimetry(s, backend, calibration=cal, rng=random.Random(5), show_instructions=False)
    folder = write_outputs(out, pdir / f"{s.participant_id}_{s.eye}_2026110{len(shown)}_1200")
    job_path.with_suffix(".result.json").write_text(json.dumps({"folder": str(folder),
                                                                 "valid": out.valid}))
    on_done(0)


win._launch = fake_launch
win.show_results = lambda info: shown.append(info)
win.start_test()
require(len(shown) == 1 and shown[0].valid and shown[0].eye == "R",
        "a test runs (pause pressed once on the way) and comes back valid")
require(win.b_show.isEnabled() and win.b_repeat.isEnabled(), "Show results / Repeat enabled")
meta = json.loads((shown[0].folder / "session.json").read_text())
require(meta["settings"]["fixation_colour"] == "black", "settings travel into the session")
win.f_eye.setCurrentIndex(1)
win.start_test()
require(len(shown) == 2 and shown[1].eye == "L", "the other eye runs")

print("5. results, patient results, compare, delete")
d = ResultsDialog(win, shown[0], offer_next=True)
require(not d.map_image.pixmap().isNull(), "contour map drawn")
d.scheme.setCurrentText(specvis_map.MONO)
d.iso.setValue(2.0)
d.redraw()
require(not d.map_image.pixmap().isNull(), "redraw with Monochromatic, isofactor 2")
csv_path = root / "map.csv"
specvis_map.export_map_data(d.rows, csv_path)
require("contrast_db" in csv_path.read_text(), "export data writes the dB values")
pr = PatientResultsDialog(win, "PILOT02", pdir)
require(pr.table.rowCount() == 2, "patient results list both sessions")
CompareDialog(win, pr.sessions[0], pr.sessions[1])
require(True, "compare maps and table open")
pr.table.selectRow(0)
pr.delete()
require(pr.table.rowCount() == 1 and (pdir / "_deleted").exists() and
        any((pdir / "_deleted").iterdir()), "delete moves the session to _deleted (recoverable)")

print("6. remembered next time")
win._save_last()
again = SpecvisWindow(pdir, root / "output" / "participants.json", store.PRESETS_DIR,
                      pdir / "prior_table.csv", root / "calibration" / "scales.json")
require(again.patient and again.patient.id == "PILOT02" and again.settings.eye == "L",
        "patient and settings come back")

win.close()
again.close()
shutil.rmtree(root, ignore_errors=True)
sys.exit(0 if ok else 1)
