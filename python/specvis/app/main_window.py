"""The ReFreM perimetry app: Specvis's step-by-step window, rebuilt for our test.

    Step 1  Participant   who, which eye, which preset
    Step 2  Screen        width, distance, monitor -- and whether the grid fits
    Step 3  Stimulus      the Gabor and the timing, with a preview
    Step 4  Procedure     grid, pilot mode, stopping rule, catch trials, normal table
    Step 5  Start         a summary, then the test runs full screen

After each test the Results window shows the map, the checks and every point, and
offers the other eye or a repeat. The test itself runs in a separate process
(``run_test_process.py``) so PsychoPy and Qt never share an event loop.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import sys
import tempfile
from typing import Callable, List, Optional

import numpy as np
from PyQt6.QtCore import QProcess, Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                             QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                             QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                             QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ..core.gabor import GaborParameters, render_brightness
from ..core.luminance import LuminanceCalibration
from ..procedure.perimetry import (CENTRAL, FAILURE_REASONS, FULL, PerimetrySettings,
                                   blind_spot_position, field_half_degrees, fit_problem,
                                   grid_points, settings_from_dict, settings_to_dict)
from ..procedure.prior_table import PriorTable, build_prior_table
from . import store

PYTHON_DIR = store.PYTHON_DIR
STEPS = ["1  Participant", "2  Screen", "3  Stimulus", "4  Procedure", "5  Start"]
GROUPS = ["pilot", "healthy control", "patient", "other"]
SEXES = ["", "female", "male", "other", "prefer not to say"]


def _spin(lo: float, hi: float, value: float, step: float = 1.0, decimals: int = 1,
          suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(" " + suffix)
    return box


def _ispin(lo: int, hi: int, value: int, suffix: str = "") -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setValue(value)
    if suffix:
        box.setSuffix(" " + suffix)
    return box


def _heading(text: str) -> QLabel:
    label = QLabel(text)
    f = label.font()
    f.setPointSize(f.pointSize() + 4)
    f.setBold(True)
    label.setFont(f)
    return label


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: gray;")
    return label


# ------------------------------------------------------------------------------
# Dialogs
# ------------------------------------------------------------------------------

class ParticipantDialog(QDialog):
    """New or edit. IDs only -- no names -- so records can be shared in the team."""

    def __init__(self, parent=None, participant: Optional[store.Participant] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit participant" if participant else "New participant")
        p = participant or store.Participant("")
        form = QFormLayout(self)
        self.id_edit = QLineEdit(p.id)
        self.id_edit.setPlaceholderText("e.g. PILOT02, HC01, PT01  (no names)")
        self.id_edit.setEnabled(participant is None or not participant.registered)
        self.age = QLineEdit(p.age)
        self.sex = QComboBox()
        self.sex.addItems(SEXES)
        self.sex.setCurrentText(p.sex)
        self.group = QComboBox()
        self.group.addItems(GROUPS)
        self.group.setCurrentText(p.group or "pilot")
        self.acuity_l, self.acuity_r = QLineEdit(p.acuity_left), QLineEdit(p.acuity_right)
        self.notes = QPlainTextEdit(p.notes)
        self.notes.setFixedHeight(70)
        form.addRow("ID *", self.id_edit)
        form.addRow("Age", self.age)
        form.addRow("Sex", self.sex)
        form.addRow("Group", self.group)
        form.addRow("Visual acuity, left eye", self.acuity_l)
        form.addRow("Visual acuity, right eye", self.acuity_r)
        form.addRow("Notes", self.notes)
        form.addRow(_note("Use an ID, never a name. The ID becomes the folder name."))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.created = p.created

    def _accept(self) -> None:
        if not store.valid_id(self.id_edit.text().strip()):
            QMessageBox.warning(self, "ID", "The ID may use letters, digits, '_' and '-' only.")
            return
        self.accept()

    def participant(self) -> store.Participant:
        return store.Participant(
            id=self.id_edit.text().strip(), age=self.age.text().strip(),
            sex=self.sex.currentText(), group=self.group.currentText(),
            acuity_left=self.acuity_l.text().strip(), acuity_right=self.acuity_r.text().strip(),
            notes=self.notes.toPlainText().strip(), created=self.created)


class ResultsWindow(QDialog):
    """One session: the map, the checks, every point. Like Specvis's results window."""

    OTHER_EYE, REPEAT = 10, 11

    def __init__(self, parent, info: store.SessionInfo, offer_next: bool = True):
        super().__init__(parent)
        self.info = info
        self.setWindowTitle(f"Results - {info.participant_id} - {info.eye} eye - {info.started}")
        self.resize(1100, 860)
        layout = QVBoxLayout(self)

        state = "SESSION VALID" if info.valid else (
            "STOPPED EARLY" if info.aborted else "SESSION NOT VALID - repeat this eye")
        head = _heading(f"{info.participant_id}   {info.eye} eye   -   {state}")
        head.setStyleSheet("color: %s;" % ("#0F6E56" if info.valid else "#A32D2D"))
        layout.addWidget(head)

        def rate(key: str) -> str:
            v = info.rates.get(key)
            return "no trials" if v is None or (isinstance(v, float) and math.isnan(v)) \
                else f"{100 * v:.0f}%"

        tick = {True: "ok", False: "FAILED"}
        checks = info.checks
        lines = [f"{info.trials} trials, {info.duration_min:.1f} min   |   "
                 f"pilot mode {'on' if info.pilot_mode else 'off'}   |   grid {info.extent}",
                 f"False presses {rate('false_alarms')} [{tick[checks.get('false_alarms', False)]}]"
                 f"     Missed strong patches {rate('misses')} [{tick[checks.get('misses', False)]}]"
                 f"     Blind-spot seen {rate('fixation')} [{tick[checks.get('fixation', False)]}]"]
        lines += [FAILURE_REASONS[k] for k, ok in checks.items() if not ok and k in FAILURE_REASONS]
        text = QLabel("\n".join(lines))
        text.setWordWrap(True)
        layout.addWidget(text)

        image = QLabel()
        pix = QPixmap(str(info.map_path))
        if not pix.isNull():
            image.setPixmap(pix.scaledToWidth(1060, Qt.TransformationMode.SmoothTransformation))
        else:
            image.setText("(no map.png in this session)")
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(image, 3)

        self.table = QTableWidget()
        cols = ["x_deg", "y_deg", "category", "threshold_contrast", "threshold_db", "trials",
                "termination", "starting_guess"]
        rows = []
        if info.summary_path.exists():
            with info.summary_path.open() as f:
                rows = list(csv.DictReader(f))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                self.table.setItem(i, j, QTableWidgetItem(str(r.get(c, ""))))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 2)

        buttons = QHBoxLayout()
        other = "LEFT" if info.eye == "R" else "RIGHT"
        if offer_next:
            b_other = QPushButton(f"Test the {other} eye now")
            b_other.clicked.connect(lambda: self.done(self.OTHER_EYE))
            b_repeat = QPushButton("Repeat this eye")
            b_repeat.clicked.connect(lambda: self.done(self.REPEAT))
            buttons.addWidget(b_other)
            buttons.addWidget(b_repeat)
        b_folder = QPushButton("Open folder")
        b_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(info.folder))))
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(b_folder)
        buttons.addWidget(b_close)
        layout.addLayout(buttons)


class CompareWindow(QDialog):
    """Two sessions side by side: right vs left eye, or test vs retest."""

    def __init__(self, parent, a: store.SessionInfo, b: store.SessionInfo):
        super().__init__(parent)
        self.setWindowTitle("Compare sessions")
        self.resize(1500, 650)
        layout = QHBoxLayout(self)
        for info in (a, b):
            box = QVBoxLayout()
            box.addWidget(QLabel(f"<b>{info.participant_id}</b> - {info.label}"))
            image = QLabel()
            pix = QPixmap(str(info.map_path))
            if not pix.isNull():
                image.setPixmap(pix.scaledToWidth(720, Qt.TransformationMode.SmoothTransformation))
            box.addWidget(image)
            layout.addLayout(box)


class HistoryWindow(QDialog):
    """Every session of one participant, newest first."""

    def __init__(self, parent, participant_id: str, perimetry_dir: pathlib.Path):
        super().__init__(parent)
        self.setWindowTitle(f"Sessions - {participant_id}")
        self.resize(720, 480)
        self.sessions = store.list_sessions(perimetry_dir, participant_id)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>{participant_id}</b>: {len(self.sessions)} sessions"))
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for s in self.sessions:
            item = QListWidgetItem(s.label)
            item.setForeground(QColor("#0F6E56" if s.valid else "#A32D2D"))
            self.list.addItem(item)
        self.list.itemDoubleClicked.connect(lambda _: self.show_selected())
        layout.addWidget(self.list)
        buttons = QHBoxLayout()
        b_show = QPushButton("Show results")
        b_show.clicked.connect(self.show_selected)
        b_cmp = QPushButton("Compare two")
        b_cmp.clicked.connect(self.compare)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.accept)
        for b in (b_show, b_cmp):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(b_close)
        layout.addLayout(buttons)
        layout.addWidget(_note("Select two sessions (Cmd/Ctrl-click) to compare them."))

    def selected(self) -> List[store.SessionInfo]:
        return [self.sessions[i.row()] for i in self.list.selectedIndexes()]

    def show_selected(self) -> None:
        sel = self.selected()
        if sel:
            ResultsWindow(self, sel[0], offer_next=False).exec()

    def compare(self) -> None:
        sel = self.selected()
        if len(sel) != 2:
            QMessageBox.information(self, "Compare", "Select exactly two sessions.")
            return
        CompareWindow(self, sel[0], sel[1]).exec()


class GridPreview(QWidget):
    """The test points, the dropped blind-spot points, the seeds and the blind-spot
    catch position, in field degrees."""

    def __init__(self, legend: bool = True):
        super().__init__()
        self.settings = PerimetrySettings()
        self.legend = legend
        if legend:
            self.setMinimumSize(520, 280)

    def set_settings(self, settings: PerimetrySettings) -> None:
        self.settings = settings
        self.update()

    def paintEvent(self, event) -> None:
        try:
            self._paint()
        except Exception:                          # noqa: BLE001 -- a paint error must not kill the app
            import traceback
            sys.stderr.write(traceback.format_exc())
            try:
                from .specvis_ui import log_error
                log_error(traceback.format_exc())
            except Exception:                      # noqa: BLE001
                pass

    def _paint(self) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#808080"))
        margin = 22 if self.legend else 4             # room for the two text lines
        plot_h = h - 2 * margin
        scale = min(w / 50.0, plot_h / 32.0)
        cx, cy = w / 2.0, margin + plot_h / 2.0

        def at(x: float, y: float):
            return cx + x * scale, cy - y * scale

        kept, dropped = grid_points(self.settings)
        r = self.settings.diameter_degrees / 2.0 * scale
        for x, y in kept:
            px, py = at(x, y)
            seed = (x, y) in self.settings.seed_points
            p.setPen(QPen(QColor("#FAC775" if seed else "white"), 2))
            p.drawEllipse(int(px - r), int(py - r), int(2 * r), int(2 * r))
        p.setPen(QPen(QColor("#B4B2A9"), 1, Qt.PenStyle.DashLine))
        for x, y in dropped:
            px, py = at(x, y)
            p.drawEllipse(int(px - r), int(py - r), int(2 * r), int(2 * r))
        bx, by = at(*blind_spot_position(self.settings))
        p.setPen(QPen(QColor("#D85A30"), 2))
        p.drawEllipse(int(bx - r), int(by - r), int(2 * r), int(2 * r))
        p.setPen(QPen(QColor("black"), 2))
        p.drawLine(int(cx - 8), int(cy), int(cx + 8), int(cy))
        p.drawLine(int(cx), int(cy - 8), int(cx), int(cy + 8))
        if self.legend:
            p.setPen(QColor("white"))
            p.drawText(8, 15, f"{len(kept)} test points, {self.settings.eye} eye, true positions")
            p.drawText(8, h - 8, "yellow = seed   red = blind-spot check   dashed = skipped")
        p.end()


# ------------------------------------------------------------------------------
# The main window
# ------------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, perimetry_dir: pathlib.Path = store.PERIMETRY_DIR,
                 participants_file: pathlib.Path = store.PARTICIPANTS_FILE,
                 presets_dir: pathlib.Path = store.PRESETS_DIR,
                 prior_table_file: pathlib.Path = store.PRIOR_TABLE_FILE):
        super().__init__()
        self.perimetry_dir = pathlib.Path(perimetry_dir)
        self.prior_table_file = pathlib.Path(prior_table_file)
        self.presets_dir = pathlib.Path(presets_dir)
        self.last_file = self.perimetry_dir.parent / "app_last.json"
        self.people = store.ParticipantStore(participants_file)
        self.settings = PerimetrySettings()
        self.process: Optional[QProcess] = None
        self._job_dir = pathlib.Path(tempfile.mkdtemp(prefix="refrem_job_"))
        self._loading = False

        self.setWindowTitle("ReFreM perimetry")
        self.resize(1080, 760)
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        self.step_list = QListWidget()
        self.step_list.addItems(STEPS)
        self.step_list.setFixedWidth(170)
        self.step_list.currentRowChanged.connect(self._go)
        outer.addWidget(self.step_list)

        right = QVBoxLayout()
        self.pages = QStackedWidget()
        for build in (self._page_participant, self._page_screen, self._page_stimulus,
                      self._page_procedure, self._page_start):
            self.pages.addWidget(build())
        right.addWidget(self.pages, 1)
        nav = QHBoxLayout()
        self.back_button = QPushButton("< Back")
        self.back_button.clicked.connect(lambda: self.step_list.setCurrentRow(self.pages.currentIndex() - 1))
        self.next_button = QPushButton("Next >")
        self.next_button.clicked.connect(lambda: self.step_list.setCurrentRow(self.pages.currentIndex() + 1))
        nav.addStretch(1)
        nav.addWidget(self.back_button)
        nav.addWidget(self.next_button)
        right.addLayout(nav)
        outer.addLayout(right, 1)

        self._load_last()
        self.refresh_participants()
        self.step_list.setCurrentRow(0)
        self._update_all()

    # -- pages -------------------------------------------------------------------

    def _page_participant(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(_heading("Step 1 - Participant"))
        self.people_list = QListWidget()
        self.people_list.currentRowChanged.connect(lambda _: self._update_all())
        self.people_list.itemDoubleClicked.connect(lambda _: self.show_history())
        v.addWidget(self.people_list, 1)
        row = QHBoxLayout()
        for text, slot in (("New", self.new_participant), ("Edit", self.edit_participant),
                           ("Results", self.show_history)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        form = QFormLayout()
        self.eye_box = QComboBox()
        self.eye_box.addItems(["R  - right eye open (cover the LEFT eye)",
                               "L  - left eye open (cover the RIGHT eye)"])
        self.eye_box.currentIndexChanged.connect(self._read_pages)
        form.addRow("Eye tested", self.eye_box)
        preset_row = QHBoxLayout()
        self.preset_box = QComboBox()
        self.presets = store.list_presets(self.presets_dir)
        self.preset_box.addItems([p.name for p in self.presets] or ["(no presets found)"])
        b_apply = QPushButton("Apply preset")
        b_apply.clicked.connect(self.apply_preset)
        preset_row.addWidget(self.preset_box, 1)
        preset_row.addWidget(b_apply)
        form.addRow("Preset", preset_row)
        self.preset_note = _note("")
        form.addRow("", self.preset_note)
        self.preset_box.currentIndexChanged.connect(self._show_preset_note)
        file_row = QHBoxLayout()
        b_load = QPushButton("Load settings file...")
        b_load.clicked.connect(self.load_settings_file)
        b_save = QPushButton("Save settings file...")
        b_save.clicked.connect(self.save_settings_file)
        file_row.addWidget(b_load)
        file_row.addWidget(b_save)
        file_row.addStretch(1)
        form.addRow("Settings", file_row)
        v.addLayout(form)
        v.addWidget(_note("Participant records and results stay in python/output/ on this "
                          "computer (never on GitHub). Presets live in python/presets/ and are "
                          "shared with the code, so everyone uses the same settings."))
        self._show_preset_note()
        return page

    def _page_screen(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(_heading("Step 2 - Screen"))
        form = QFormLayout()
        self.monitor_box = QComboBox()
        for i, s in enumerate(QGuiApplication.screens()):
            g, r = s.geometry(), s.devicePixelRatio()
            self.monitor_box.addItem(f"{i}: {s.name()}  {int(g.width() * r)}x{int(g.height() * r)} px")
        self.monitor_box.currentIndexChanged.connect(self._read_pages)
        form.addRow("Stimulus monitor", self.monitor_box)
        self.width_cm = _spin(0.0, 200.0, 30.2, 0.1, 1, "cm")
        self.distance_cm = _spin(10.0, 200.0, 35.0, 0.5, 1, "cm")
        self.fullscreen = QCheckBox("Full screen (always, for data)")
        self.fullscreen.setChecked(True)
        for w in (self.width_cm, self.distance_cm):
            w.valueChanged.connect(self._read_pages)
        self.fullscreen.toggled.connect(self._read_pages)
        form.addRow("Visible screen width (measured)", self.width_cm)
        form.addRow("Viewing distance, eye to screen", self.distance_cm)
        form.addRow("", self.fullscreen)
        v.addLayout(form)
        self.fit_label = QLabel()
        self.fit_label.setWordWrap(True)
        v.addWidget(self.fit_label)
        b_check = QPushButton("Screen check...  (shows a 10 cm box to measure)")
        b_check.clicked.connect(self.screen_check)
        v.addWidget(b_check)
        v.addWidget(_note(
            "Measure the lit area of the screen with a ruler, not the bezel. MacBook Pro 14\": "
            "30.2 cm. At 57 cm, 1 cm on screen = 1 degree. Use a chin rest and keep the distance "
            "the same for everyone. Turn off auto-brightness, True Tone / Night Shift, eco and "
            "dynamic-contrast modes, and keep the brightness fixed."))
        v.addWidget(_note("Luminance calibration: none yet. Contrast values are nominal until "
                          "the screen is measured with a photometer."))
        v.addStretch(1)
        return page

    def _page_stimulus(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(_heading("Step 3 - Stimulus"))
        h = QHBoxLayout()
        form = QFormLayout()
        s = self.settings
        self.diameter = _spin(0.5, 10.0, s.diameter_degrees, 0.5, 1, "deg")
        self.sf = _spin(0.25, 16.0, s.spatial_frequency_cpd, 0.25, 2, "c/deg")
        self.orientation = _spin(0.0, 180.0, s.orientation_degrees, 15.0, 0, "deg")
        self.stim_ms = _spin(16.0, 2000.0, s.stimulus_ms, 10.0, 0, "ms")
        self.fix_ms = _spin(0.0, 3000.0, s.fixation_ms, 50.0, 0, "ms")
        self.window_ms = _spin(200.0, 5000.0, s.response_window_ms, 100.0, 0, "ms")
        self.iti_lo = _spin(0.0, 3000.0, s.iti_ms[0], 50.0, 0, "ms")
        self.iti_hi = _spin(0.0, 3000.0, s.iti_ms[1], 50.0, 0, "ms")
        for label, w in (("Gabor diameter", self.diameter), ("Spatial frequency", self.sf),
                         ("Orientation", self.orientation), ("Stimulus shown for", self.stim_ms),
                         ("Fixation cross alone", self.fix_ms),
                         ("Response window after stimulus", self.window_ms),
                         ("Gap before next, from", self.iti_lo), ("Gap before next, to", self.iti_hi)):
            w.valueChanged.connect(self._read_pages)
            form.addRow(label, w)
        h.addLayout(form)
        self.gabor_preview = QLabel()
        self.gabor_preview.setFixedSize(220, 220)
        h.addWidget(self.gabor_preview)
        v.addLayout(h)
        v.addWidget(_note("A static Gabor whose mean equals the grey background: lowering the "
                          "contrast makes the stripes fade into the grey; the patch never gets "
                          "brighter or darker. Preview at 80% contrast. The study default is "
                          "2 deg, 4 c/deg, 200 ms -- change only as a team decision."))
        v.addStretch(1)
        return page

    def _page_procedure(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(_heading("Step 4 - Procedure"))
        grid = QGridLayout()
        s = self.settings
        self.extent_box = QComboBox()
        self.extent_box.addItems([FULL, CENTRAL])
        self.pilot = QCheckBox("Pilot mode (first 3-5 healthy people: no starting guesses, "
                               "builds the normal table)")
        self.sd_stop = _spin(0.05, 0.5, s.zest.sd_stop, 0.01, 2, "log10")
        self.min_trials = _ispin(1, 20, s.zest.min_trials)
        self.max_trials = _ispin(2, 60, s.zest.max_trials)
        self.p_blank = _spin(0.0, 20.0, 100 * s.p_blank, 1.0, 0, "%")
        self.p_high = _spin(0.0, 20.0, 100 * s.p_high, 1.0, 0, "%")
        self.p_blind = _spin(0.0, 20.0, 100 * s.p_blind_spot, 1.0, 0, "%")
        self.break_every = _ispin(0, 500, s.break_every_trials, "trials")
        form = QFormLayout()
        form.addRow("Grid", self.extent_box)
        form.addRow("", self.pilot)
        form.addRow("Stop when the estimate's SD is below", self.sd_stop)
        form.addRow("Trials per point, at least", self.min_trials)
        form.addRow("Trials per point, at most", self.max_trials)
        form.addRow("Blank catch trials (false presses)", self.p_blank)
        form.addRow("Strong catch trials (misses)", self.p_high)
        form.addRow("Blind-spot trials (fixation)", self.p_blind)
        form.addRow("Break every", self.break_every)
        for w in (self.sd_stop, self.min_trials, self.max_trials, self.p_blank, self.p_high,
                  self.p_blind, self.break_every):
            w.valueChanged.connect(self._read_pages)
        self.extent_box.currentIndexChanged.connect(self._read_pages)
        self.pilot.toggled.connect(self._read_pages)
        grid.addLayout(form, 0, 0)
        self.grid_preview = GridPreview()
        grid.addWidget(self.grid_preview, 0, 1)
        v.addLayout(grid)
        table_row = QHBoxLayout()
        self.table_label = QLabel()
        self.table_label.setWordWrap(True)
        b_table = QPushButton("Build normal table from valid pilot sessions")
        b_table.clicked.connect(self.build_table)
        table_row.addWidget(self.table_label, 1)
        table_row.addWidget(b_table)
        v.addLayout(table_row)
        v.addStretch(1)
        return page

    def _page_start(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(_heading("Step 5 - Start"))
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.summary.setFont(mono)
        v.addWidget(self.summary, 2)
        self.start_button = QPushButton("START TEST")
        self.start_button.setMinimumHeight(44)
        f = self.start_button.font()
        f.setBold(True)
        self.start_button.setFont(f)
        self.start_button.clicked.connect(self.start_test)
        v.addWidget(self.start_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        v.addWidget(self.log, 1)
        v.addWidget(_note("The test opens full screen. The participant presses SPACE when they "
                          "see the stripes; ESC stops at any time and keeps what was done."))
        return page

    # -- settings <-> widgets --------------------------------------------------------

    def _write_pages(self) -> None:
        """Put self.settings into the widgets."""
        self._loading = True
        s = self.settings
        self.eye_box.setCurrentIndex(0 if s.eye == "R" else 1)
        self.width_cm.setValue(s.screen_width_mm / 10.0)
        self.distance_cm.setValue(s.viewing_distance_mm / 10.0)
        self.diameter.setValue(s.diameter_degrees)
        self.sf.setValue(s.spatial_frequency_cpd)
        self.orientation.setValue(s.orientation_degrees)
        self.stim_ms.setValue(s.stimulus_ms)
        self.fix_ms.setValue(s.fixation_ms)
        self.window_ms.setValue(s.response_window_ms)
        self.iti_lo.setValue(s.iti_ms[0])
        self.iti_hi.setValue(s.iti_ms[1])
        self.extent_box.setCurrentText(s.extent)
        self.pilot.setChecked(s.pilot_mode)
        self.sd_stop.setValue(s.zest.sd_stop)
        self.min_trials.setValue(s.zest.min_trials)
        self.max_trials.setValue(s.zest.max_trials)
        self.p_blank.setValue(100 * s.p_blank)
        self.p_high.setValue(100 * s.p_high)
        self.p_blind.setValue(100 * s.p_blind_spot)
        self.break_every.setValue(s.break_every_trials)
        self._loading = False
        self._update_all()

    def _read_pages(self, *_) -> None:
        """Read the widgets into self.settings."""
        if self._loading:
            return
        d = settings_to_dict(self.settings)
        d.update(
            eye="R" if self.eye_box.currentIndex() == 0 else "L",
            screen_width_mm=self.width_cm.value() * 10.0,
            viewing_distance_mm=self.distance_cm.value() * 10.0,
            diameter_degrees=self.diameter.value(), spatial_frequency_cpd=self.sf.value(),
            orientation_degrees=self.orientation.value(), stimulus_ms=self.stim_ms.value(),
            fixation_ms=self.fix_ms.value(), response_window_ms=self.window_ms.value(),
            iti_ms=(self.iti_lo.value(), max(self.iti_lo.value(), self.iti_hi.value())),
            extent=self.extent_box.currentText(), pilot_mode=self.pilot.isChecked(),
            p_blank=self.p_blank.value() / 100.0, p_high=self.p_high.value() / 100.0,
            p_blind_spot=self.p_blind.value() / 100.0,
            break_every_trials=self.break_every.value())
        d["zest"] = dict(d["zest"], sd_stop=self.sd_stop.value(),
                         min_trials=self.min_trials.value(),
                         max_trials=max(self.min_trials.value(), self.max_trials.value()))
        self.settings = settings_from_dict(d)
        self._update_all()

    # -- derived displays ---------------------------------------------------------------

    def current_participant(self) -> Optional[store.Participant]:
        item = self.people_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def monitor_pixels(self):
        screens = QGuiApplication.screens()
        i = max(0, min(self.monitor_box.currentIndex(), len(screens) - 1))
        g, r = screens[i].geometry(), screens[i].devicePixelRatio()
        return int(g.width() * r), int(g.height() * r)

    def problems(self) -> List[str]:
        out = []
        if self.current_participant() is None:
            out.append("Choose or create a participant (step 1).")
        if self.settings.screen_width_mm <= 0:
            out.append("Enter the measured visible screen width (step 2).")
        else:
            w, h = self.monitor_pixels()
            p = fit_problem(self.settings, w, h)
            if p:
                out.append("The grid does not fit (step 2): " + p)
        try:
            self.settings.validate()
        except Exception as e:                      # noqa: BLE001 -- shown to the user
            out.append(str(e))
        return out

    def _update_all(self) -> None:
        if not hasattr(self, "summary"):
            return
        s = self.settings
        w, h = self.monitor_pixels()
        if s.screen_width_mm > 0:
            hx, hy = field_half_degrees(s, w, h)
            p = fit_problem(s, w, h)
            self.fit_label.setText(
                f"This screen covers +/-{hx:.1f} x +/-{hy:.1f} deg.  " +
                ("The grid fits." if not p else "THE GRID DOES NOT FIT: " + p))
            self.fit_label.setStyleSheet("color: %s;" % ("#0F6E56" if not p else "#A32D2D"))
        else:
            self.fit_label.setText("Enter the measured visible width of this screen.")
            self.fit_label.setStyleSheet("color: #A32D2D;")
        self.grid_preview.set_settings(s)
        self._update_gabor_preview()
        self._update_table_label()
        person = self.current_participant()
        kept, _ = grid_points(s)
        lines = [
            f"Participant      {person.id if person else '(none)'}"
            f"{'  [' + person.group + ']' if person and person.group else ''}",
            f"Eye tested       {s.eye}  (cover the {'LEFT' if s.eye == 'R' else 'RIGHT'} eye)",
            f"Screen           {s.screen_width_mm / 10:.1f} cm wide, {w}x{h} px, "
            f"monitor {self.monitor_box.currentIndex()}, "
            f"{'full screen' if self.fullscreen.isChecked() else 'WINDOWED (not for data)'}",
            f"Distance         {s.viewing_distance_mm / 10:.1f} cm",
            f"Stimulus         Gabor {s.diameter_degrees:g} deg, {s.spatial_frequency_cpd:g} c/deg, "
            f"{s.orientation_degrees:g} deg, {s.stimulus_ms:.0f} ms",
            f"Timing           + {s.fixation_ms:.0f} ms, respond within "
            f"{s.response_window_ms:.0f} ms, gap {s.iti_ms[0]:.0f}-{s.iti_ms[1]:.0f} ms",
            f"Grid             {s.extent}: {len(kept)} points",
            f"Pilot mode       {'ON (no starting guesses)' if s.pilot_mode else 'off'}",
            f"Stopping         SD < {s.zest.sd_stop:.2f} log10, {s.zest.min_trials}-"
            f"{s.zest.max_trials} trials per point",
            f"Catch trials     blank {100 * s.p_blank:.0f}%, strong {100 * s.p_high:.0f}%, "
            f"blind spot {100 * s.p_blind_spot:.0f}%",
            f"Normal table     {self._table_text()}",
        ]
        problems = self.problems()
        if problems:
            lines += ["", "BEFORE YOU CAN START:"] + ["  - " + p for p in problems]
        self.summary.setPlainText("\n".join(lines))
        running = self.process is not None
        self.start_button.setEnabled(not problems and not running)
        self.back_button.setEnabled(self.pages.currentIndex() > 0)
        self.next_button.setEnabled(self.pages.currentIndex() < len(STEPS) - 1)

    def _update_gabor_preview(self) -> None:
        s = self.settings
        ppd = 200.0 / max(s.diameter_degrees, 0.5)
        p = GaborParameters(s.diameter_degrees, s.spatial_frequency_cpd, s.orientation_degrees,
                            0.0, s.diameter_degrees / 6.0, 50.0, 0.8)
        b = render_brightness(p, LuminanceCalibration.uncalibrated(), ppd, ppd)
        img = np.ascontiguousarray((b * 255).astype(np.uint8))
        self._preview_bytes = img                     # keep alive for QImage
        q = QImage(img.data, img.shape[1], img.shape[0], img.shape[1],
                   QImage.Format.Format_Grayscale8)
        self.gabor_preview.setPixmap(QPixmap.fromImage(q).scaled(
            220, 220, Qt.AspectRatioMode.KeepAspectRatio))

    def _table_text(self) -> str:
        if not self.prior_table_file.exists():
            return "none yet (run pilots, then Build normal table)"
        t = PriorTable.load(self.prior_table_file)
        used = "not used (pilot mode)" if self.settings.pilot_mode else "used as starting guesses"
        return f"{t.source} - {used}"

    def _update_table_label(self) -> None:
        self.table_label.setText("Normal table: " + self._table_text())

    def _show_preset_note(self) -> None:
        i = self.preset_box.currentIndex()
        self.preset_note.setText(self.presets[i].description if 0 <= i < len(self.presets) else "")

    def _go(self, index: int) -> None:
        if 0 <= index < self.pages.count():
            self.pages.setCurrentIndex(index)
        self._update_all()

    # -- participants ------------------------------------------------------------------

    def refresh_participants(self, select: Optional[str] = None) -> None:
        sessions = store.list_sessions(self.perimetry_dir)
        counts = {}
        for s in sessions:
            counts[s.participant_id] = counts.get(s.participant_id, 0) + 1
        current = select or (self.current_participant().id if self.current_participant() else
                             getattr(self, "_last_participant", None))
        self.people_list.clear()
        for p in store.all_participants(self.people, sessions):
            n = counts.get(p.id, 0)
            text = f"{p.id}   {p.group or ''}   {n} session{'s' if n != 1 else ''}"
            if not p.registered:
                text += "   (not registered - Edit to add details)"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.people_list.addItem(item)
            if p.id == current:
                self.people_list.setCurrentItem(item)
        self._update_all()

    def new_participant(self) -> None:
        dlg = ParticipantDialog(self)
        if dlg.exec():
            try:
                self.people.add(dlg.participant())
            except ValueError as e:
                QMessageBox.warning(self, "Participant", str(e))
                return
            self.refresh_participants(select=dlg.participant().id)

    def edit_participant(self) -> None:
        p = self.current_participant()
        if p is None:
            return
        dlg = ParticipantDialog(self, p)
        if dlg.exec():
            new = dlg.participant()
            if self.people.get(new.id):
                self.people.update(new)
            else:
                self.people.add(new)                 # registering one found in sessions
            self.refresh_participants(select=new.id)

    def show_history(self) -> None:
        p = self.current_participant()
        if p is not None:
            HistoryWindow(self, p.id, self.perimetry_dir).exec()

    # -- presets and settings files -------------------------------------------------------

    def apply_settings_values(self, values: dict) -> None:
        d = settings_to_dict(self.settings)
        for k, v in values.items():
            if k == "zest" and isinstance(v, dict):
                d["zest"].update(v)
            elif k not in ("participant_id", "eye"):
                d[k] = v
        self.settings = settings_from_dict(d)
        self._write_pages()

    def apply_preset(self) -> None:
        i = self.preset_box.currentIndex()
        if 0 <= i < len(self.presets):
            self.apply_settings_values(self.presets[i].settings)
            self.status.setText(f"Preset applied: {self.presets[i].name}")
            if self.settings.screen_width_mm <= 0:
                QMessageBox.information(
                    self, "Measure the screen",
                    "This preset has no screen width yet. Measure the visible width of the "
                    "monitor with a ruler and enter it in step 2. Then use 'Save settings "
                    "file...' into python/presets/ so everyone shares it.")

    def load_settings_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", str(self.presets_dir),
                                              "ReFreM settings (*.json)")
        if path:
            try:
                self.apply_settings_values(store.load_settings_file(pathlib.Path(path)).settings)
            except (OSError, ValueError) as e:
                QMessageBox.warning(self, "Load settings", str(e))

    def save_settings_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", str(self.presets_dir),
                                              "ReFreM settings (*.json)")
        if path:
            name = pathlib.Path(path).stem.replace("_", " ")
            store.save_settings_file(pathlib.Path(path).with_suffix(".json"), name,
                                     "Saved from the app.", settings_to_dict(self.settings))
            self.presets = store.list_presets(self.presets_dir)
            self.preset_box.clear()
            self.preset_box.addItems([p.name for p in self.presets])

    def build_table(self) -> None:
        summaries = sorted(self.perimetry_dir.glob("*/summary.csv"))
        if not summaries:
            QMessageBox.information(self, "Normal table", "No sessions yet.")
            return
        table, messages = build_prior_table(summaries)
        if not table.entries:
            QMessageBox.information(self, "Normal table",
                                    "No valid sessions to build from:\n\n" + "\n".join(messages))
            return
        table.save(self.prior_table_file)
        people = max(e.n_participants for e in table.entries.values())
        warn = ("\n\nFewer than 10 people: fine for testing the method, not yet a reliable "
                "normal table." if people < 10 else "")
        QMessageBox.information(self, "Normal table",
                                f"{table.source}.\nUp to {people} people per point.{warn}\n\n"
                                + "\n".join(messages))
        self._update_all()

    # -- running ------------------------------------------------------------------

    def job(self) -> dict:
        p = self.current_participant()
        d = settings_to_dict(self.settings)
        d["participant_id"] = p.id if p else "unknown"
        return {"settings": d, "screen": self.monitor_box.currentIndex(),
                "fullscreen": self.fullscreen.isChecked()}

    def _write_job(self) -> pathlib.Path:
        path = self._job_dir / "job.json"
        path.write_text(json.dumps(self.job(), indent=2))
        result = path.with_suffix(".result.json")
        if result.exists():
            result.unlink()
        return path

    def screen_check(self) -> None:
        if self.settings.screen_width_mm <= 0:
            QMessageBox.information(self, "Screen check", "Enter the screen width first.")
            return
        self._launch("check", self._write_job(), lambda code: None)

    def start_test(self) -> None:
        problems = self.problems()
        if problems:
            QMessageBox.warning(self, "Cannot start", "\n".join(problems))
            return
        self._save_last()
        job_path = self._write_job()
        self.status.setText(f"Test running on the stimulus screen: "
                            f"{self.job()['settings']['participant_id']}, {self.settings.eye} eye ...")
        self.log.clear()
        self._launch("test", job_path, lambda code: self._test_finished(job_path))

    def _launch(self, mode: str, job_path: pathlib.Path, on_done: Callable[[int], None]) -> None:
        """Run the stimulus process. Tests replace this with an in-process runner."""
        proc = QProcess(self)
        proc.setWorkingDirectory(str(PYTHON_DIR))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(bytes(proc.readAllStandardOutput()).decode(errors="replace").rstrip()))

        def finished(code, _status):
            self.process = None
            self._update_all()
            on_done(code)

        proc.finished.connect(finished)
        self.process = proc
        self._update_all()
        proc.start(sys.executable, [str(PYTHON_DIR / "run_test_process.py"), mode, str(job_path)])

    def _test_finished(self, job_path: pathlib.Path) -> None:
        result_path = job_path.with_suffix(".result.json")
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError):
            self.status.setText("The test ended without a result (see the log above).")
            return
        if "error" in result:
            self.status.setText("Could not run: " + result["error"])
            QMessageBox.warning(self, "Could not run", result["error"])
            return
        info = store.read_session(pathlib.Path(result["folder"]))
        self.refresh_participants()
        if info is None:
            self.status.setText("Saved, but the session could not be read back.")
            return
        self.status.setText(f"Saved: {info.folder}")
        self.show_results(info)

    def show_results(self, info: store.SessionInfo) -> None:
        choice = ResultsWindow(self, info).exec()
        if choice == ResultsWindow.OTHER_EYE:
            self.eye_box.setCurrentIndex(1 if info.eye == "R" else 0)
            QMessageBox.information(self, "Other eye",
                                    f"Cover the {'RIGHT' if info.eye == 'R' else 'LEFT'} eye, "
                                    "then press OK to start.")
            self.start_test()
        elif choice == ResultsWindow.REPEAT:
            self.start_test()

    # -- remembered between launches --------------------------------------------------

    def _save_last(self) -> None:
        p = self.current_participant()
        try:
            self.last_file.parent.mkdir(parents=True, exist_ok=True)
            self.last_file.write_text(json.dumps(dict(
                participant=p.id if p else None, monitor=self.monitor_box.currentIndex(),
                fullscreen=self.fullscreen.isChecked(),
                settings=settings_to_dict(self.settings)), indent=2))
        except OSError:
            pass

    def _load_last(self) -> None:
        try:
            last = json.loads(self.last_file.read_text())
        except (OSError, ValueError):
            self._write_pages()
            return
        self._last_participant = last.get("participant")
        try:
            self.settings = settings_from_dict(last.get("settings", {}))
        except (TypeError, ValueError):
            pass
        if 0 <= last.get("monitor", 0) < self.monitor_box.count():
            self.monitor_box.setCurrentIndex(last.get("monitor", 0))
        self.fullscreen.setChecked(bool(last.get("fullscreen", True)))
        self._write_pages()

    def closeEvent(self, event) -> None:
        if self.process is not None:
            QMessageBox.information(self, "Test running", "Finish or stop the test first (ESC).")
            event.ignore()
            return
        self._save_last()
        event.accept()


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("ReFreM perimetry")
    window = MainWindow()
    window.show()
    return app.exec()
