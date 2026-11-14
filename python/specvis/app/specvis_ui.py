"""Specvis, rebuilt in Python, running the ReFreM perimetry test.

The window follows Specvis screen for screen -- the same steps, labels, buttons and
dark bottom bar (Exit/Back, Step 1-4, START, Next):

    Step 1  Patient: New / Existing / Edit / Results, ID, Name, Tested eye, Settings
    Step 2  Screen and luminance scale: screen, bounds, size, distance, field,
            stimulus and background scales (New / Existing / Edit / Fit)
    Step 3  Stimulus and background: the Gabor, timing, background, distribution
    Step 4  Fixation and other: fixation point, fixation monitoring, keys, procedure
    START   summary, Start, then Show results / Save results / Repeat / Save settings

Underneath is our method (ZEST, catch trials, blind-spot checks, pilot mode and the
normal table), not Specvis's original bisection.

The test runs in its own process (``run_test_process.py``); see ``main_window.py``.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import math
import pathlib
import shutil
import sys
import tempfile
import time
from typing import Callable, List, Optional

import numpy as np
from PyQt6.QtCore import QDate, QProcess, Qt, QTimer, QUrl
from PyQt6.QtGui import (QColor, QDesktopServices, QFont, QGuiApplication, QImage, QPainter,
                         QPen, QPixmap)
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDateEdit, QDialog,
                             QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
                             QProgressBar, QPushButton, QSizePolicy, QSpinBox, QStackedWidget, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..core.gabor import GaborParameters, minimum_renderable_contrast, render_brightness
from ..core.luminance import MEASUREMENT_POINTS, LuminanceCalibration
from ..procedure.perimetry import (CENTRAL, EXTENT_NAMES, EXTENTS, FAILURE_REASONS, FULL,
                                   FULL_THRESHOLD, H24_2, SITA_FAST,
                                   STRATEGIES,
                                   STRATEGY_NAMES, PerimetrySettings, background_for,
                                   blind_spot_position, default_calibration,
                                   contrast_range_db, field_half_degrees, fit_problem,
                                   grid_points,
                                   settings_from_dict, settings_to_dict)
from ..procedure.fixation import FIX_BLIND_SPOT, FIX_MOUSE, FIX_NONE, FIX_TRACKER
from ..procedure.prior_table import PriorTable, build_prior_table
from . import specvis_map, store
from .main_window import GridPreview

PYTHON_DIR = store.PYTHON_DIR
STEP_NAMES = ["Step 1", "Step 2", "Step 3", "Step 4", "START"]
PROCEDURE = "ReFreM perimetry (SITA Fast with a ZEST engine)"
KEYS = ["space", "return", "p", "escape", "q", "s", "x"]
COLOURS = ["white", "black", "red", "green", "yellow"]

# Specvis's MainStylesheet.css, in Qt terms
STYLE = """
QMainWindow, QDialog { background: #FFFFFF; }
QWidget#page { background: #FFFFFF; }
QLabel { color: #202020; }
QLabel#subtitle { font-size: 16px; color: #5F5E5A; }
QLabel#small { font-size: 13px; color: #5F5E5A; }
QLabel#clock { font-size: 24px; }
QLabel#circle { font-size: 30px; color: #FFFFFF; background: #202020; border-radius: 75px; }
QLabel#display, QWidget#display { background: #F0F0F0; border: 2px solid #202020; }
QLabel#steplabel { font-size: 14px; color: #202020; }
QPushButton { font-size: 17px; padding: 6px 14px; }
QTabBar::tab { font-size: 15px; color: #202020; background: #F0F0F0; padding: 6px 18px;
    border: 1px solid #D3D1C7; }
QTabBar::tab:selected { background: #00AAFF; color: #FFFFFF; }
QCheckBox, QRadioButton { color: #202020; }
QPushButton#body { background: #202020; color: #FFFFFF; border: 0; }
QPushButton#body:hover { background: #545454; }
QPushButton#body:disabled { background: #B4B2A9; color: #F1EFE8; }
QPushButton#nav { background: #00AAFF; color: #FFFFFF; border: 0; font-size: 20px; }
QPushButton#nav:hover { background: #0088CC; }
QPushButton#red { background: #FF5555; color: #FFFFFF; border: 0; font-size: 20px; }
QPushButton#red:hover { background: #CC4444; }
QPushButton#orange { background: #FFAA00; color: #FFFFFF; border: 0; font-size: 20px; }
QPushButton#orange:hover { background: #CC8800; }
QPushButton#green { background: #33CC33; color: #FFFFFF; border: 0; font-size: 20px; }
QPushButton#green:hover { background: #29A329; }
QPushButton#green:disabled { background: #B4B2A9; color: #F1EFE8; }
QFrame#bar { background: #202020; }
QFrame#pill { background: #FFFFFF; border-radius: 25px; }
QLineEdit, QComboBox, QAbstractSpinBox, QDateEdit {
    font-size: 17px; background: #FFFFFF; color: #202020;
    border: 1px solid #B4B2A9; border-radius: 0; padding: 2px 8px; }
QLineEdit[readOnly="true"] { background: #F0F0F0; color: gray; border: 1px solid #D3D1C7; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right;
    width: 30px; border: 0; background: #202020; }
QComboBox::down-arrow { image: url(@DOWN@); width: 14px; height: 11px; }
QComboBox QAbstractItemView { font-size: 16px; selection-background-color: #00AAFF; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { subcontrol-origin: border;
    width: 26px; border: 0; background: #202020; }
QAbstractSpinBox::up-button { subcontrol-position: top right; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: #545454; }
QAbstractSpinBox::up-arrow { image: url(@UP@); width: 10px; height: 8px; }
QAbstractSpinBox::down-arrow { image: url(@DOWN@); width: 10px; height: 8px; }
QAbstractSpinBox:disabled, QComboBox:disabled { background: #F0F0F0; color: gray; }
QPlainTextEdit#textArea { color: #5F5E5A; font-family: Menlo, Consolas, monospace;
    font-size: 12px; border: 1px solid #D3D1C7; }
QCheckBox { font-size: 16px; spacing: 8px; }
QCheckBox::indicator { width: 20px; height: 20px; border: 2px solid #202020;
    background: #FFFFFF; border-radius: 3px; }
QCheckBox::indicator:hover { border-color: #00AAFF; }
QCheckBox::indicator:checked { background: #202020; image: url(@CHECK@); }
QCheckBox::indicator:disabled { border-color: #B4B2A9; background: #F0F0F0; }
QCheckBox::indicator:checked:disabled { background: #B4B2A9; image: url(@CHECK@); }
"""

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"


class StepTick(QWidget):
    """Specvis's progress tick: a dark box, with a white mark once the step is reached."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(18, 18)
        self.checked = False

    def set_checked(self, value: bool) -> None:
        self.checked = bool(value)
        self.update()

    def paintEvent(self, event) -> None:
        try:
            self._paint()
        except Exception:                          # noqa: BLE001 -- a paint error must not kill the app
            import traceback
            log_error(traceback.format_exc())

    def _paint(self) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor("#202020"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, 18, 18, 3, 3)
        if self.checked:
            pen = QPen(QColor("#FFFFFF"), 2.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(4, 9, 8, 13)
            p.drawLine(8, 13, 14, 5)
        p.end()


class FitImage(QLabel):
    """A picture that always fills its box, keeping its shape (rescaled on resize)."""

    def __init__(self, path: Optional[pathlib.Path] = None, empty: str = ""):
        super().__init__()
        self._pix = QPixmap()
        self._empty = empty
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        if path is not None:
            self.set_image(path)

    def set_image(self, path: pathlib.Path) -> None:
        self._pix = QPixmap(str(path))
        self._fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self._pix.isNull():
            self.setText(self._empty)
            return
        r = self.contentsRect()
        self.setPixmap(self._pix.scaled(r.width(), r.height(),
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))


def _fixed(widget, width: int, height: int = 40):
    widget.setFixedSize(width, height)
    return widget


def _lab(text: str, width: int, height: int = 40, size: int = 20) -> QLabel:
    lab = QLabel(text)
    lab.setFixedSize(width, height)
    lab.setStyleSheet(f"font-size: {size}px;")
    return lab


def _row(widgets, spacing: int = 10) -> QHBoxLayout:
    """A centred row of fixed-size widgets, as Specvis's HBox alignment=CENTER."""
    h = QHBoxLayout()
    h.setSpacing(spacing)
    h.addStretch(1)
    for w in widgets:
        h.addWidget(w)
    h.addStretch(1)
    return h


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color: #D3D1C7;")
    return line


def _vseparator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setStyleSheet("color: #D3D1C7;")
    return line


def _box(layout, width: int) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    w.setFixedWidth(width)
    return w


def _body():
    page = QWidget()
    page.setObjectName("page")
    v = QVBoxLayout(page)
    v.setContentsMargins(20, 16, 20, 16)
    v.setSpacing(10)
    return page, v


def _button(text: str, slot: Callable, kind: str = "body") -> QPushButton:
    b = QPushButton(text)
    b.setObjectName(kind)
    b.clicked.connect(slot)
    return b


def _label(text: str, kind: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    if kind:
        lab.setObjectName(kind)
    return lab


def _readonly(text: str = "") -> QLineEdit:
    e = QLineEdit(text)
    e.setReadOnly(True)
    return e


def _spin(lo, hi, value, step=1.0, decimals=1, suffix="") -> QDoubleSpinBox:
    b = QDoubleSpinBox()
    b.setRange(lo, hi)
    b.setDecimals(decimals)
    b.setSingleStep(step)
    b.setValue(value)
    if suffix:
        b.setSuffix(" " + suffix)
    return b


def _ispin(lo, hi, value, suffix="") -> QSpinBox:
    b = QSpinBox()
    b.setRange(lo, hi)
    b.setValue(value)
    if suffix:
        b.setSuffix(" " + suffix)
    return b


def _pair(*widgets) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        h.addWidget(x)
    return w


def _page(title: str):
    page = QWidget()
    page.setObjectName("page")
    outer = QVBoxLayout(page)
    outer.setContentsMargins(60, 24, 60, 12)
    outer.addWidget(_label(title, "title"))
    return page, outer


# ------------------------------------------------------------------------------
# Patient dialogs (Specvis: ViewNewPatient, ViewEditPatient, ViewExistingPatient)
# ------------------------------------------------------------------------------

def new_patient_id() -> str:
    """Specvis-style ID, e.g. P_20261109_kQzA -- or type your own (e.g. PILOT02)."""
    import random
    import string
    return "P_" + time.strftime("%Y%m%d") + "_" + "".join(
        random.choice(string.ascii_letters) for _ in range(4))


class PatientDialog(QDialog):
    def __init__(self, parent, patient: Optional[store.Participant] = None):
        super().__init__(parent)
        editing = patient is not None and patient.registered
        self.setWindowTitle("Edit patient" if editing else "New patient")
        p = patient or store.Participant(new_patient_id())
        form = QFormLayout(self)
        self.f_id = QLineEdit(p.id)
        self.f_id.setEnabled(not editing)
        self.f_first, self.f_last = QLineEdit(p.first_name), QLineEdit(p.last_name)
        self.f_gender = QComboBox()
        self.f_gender.addItems(["", "Female", "Male", "Other"])
        self.f_gender.setCurrentText(p.sex.capitalize() if p.sex else "")
        self.f_birth = QDateEdit()
        self.f_birth.setCalendarPopup(True)
        self.f_birth.setDisplayFormat("yyyy-MM-dd")
        self.f_birth.setDate(QDate.fromString(p.birth_date, "yyyy-MM-dd")
                             if p.birth_date else QDate(1970, 1, 1))
        self.f_has_birth = QCheckBox("known")
        self.f_has_birth.setChecked(bool(p.birth_date))
        self.f_age = _readonly(p.age)
        self.f_birth.dateChanged.connect(self._age)
        self.f_has_birth.toggled.connect(self._age)
        self.f_age_manual = QLineEdit(p.age if not p.birth_date else "")
        self.f_age_manual.setPlaceholderText("or type the age")
        self.f_group = QComboBox()
        self.f_group.addItems(["pilot", "healthy control", "patient", "other"])
        self.f_group.setCurrentText(p.group or "pilot")
        self.f_phone, self.f_email = QLineEdit(p.phone), QLineEdit(p.email)
        self.f_city, self.f_postal = QLineEdit(p.city), QLineEdit(p.postal_code)
        self.f_acl, self.f_acr = QLineEdit(p.acuity_left), QLineEdit(p.acuity_right)
        self.f_notes = QPlainTextEdit(p.notes)
        self.f_notes.setFixedHeight(70)
        form.addRow("ID *", self.f_id)
        form.addRow("First name", self.f_first)
        form.addRow("Last name", self.f_last)
        form.addRow("Gender", self.f_gender)
        form.addRow("Date of birth", _pair(self.f_birth, self.f_has_birth))
        form.addRow("Age", _pair(self.f_age, self.f_age_manual))
        form.addRow("Group", self.f_group)
        form.addRow("Phone", self.f_phone)
        form.addRow("Email", self.f_email)
        form.addRow("City", self.f_city)
        form.addRow("Postal code", self.f_postal)
        form.addRow("Visual acuity L", self.f_acl)
        form.addRow("Visual acuity R", self.f_acr)
        form.addRow("Additional information", self.f_notes)
        form.addRow(_label("Names and contact details are optional. For the study, use "
                           "the ID only (e.g. PILOT02, HC01, PT01).", "note"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.created = p.created
        self._age()

    def _age(self) -> None:
        if not self.f_has_birth.isChecked():
            self.f_age.setText("")
            return
        b = self.f_birth.date().toPyDate()
        today = datetime.date.today()
        self.f_age.setText(str(today.year - b.year - ((today.month, today.day) < (b.month, b.day))))

    def _accept(self) -> None:
        if not store.valid_id(self.f_id.text().strip()):
            QMessageBox.warning(self, "ID", "The ID may use letters, digits, '_' and '-' only.")
            return
        self.accept()

    def patient(self) -> store.Participant:
        birth = self.f_birth.date().toString("yyyy-MM-dd") if self.f_has_birth.isChecked() else ""
        return store.Participant(
            id=self.f_id.text().strip(), first_name=self.f_first.text().strip(),
            last_name=self.f_last.text().strip(), sex=self.f_gender.currentText().lower(),
            birth_date=birth, age=self.f_age.text() or self.f_age_manual.text().strip(),
            group=self.f_group.currentText(), phone=self.f_phone.text().strip(),
            email=self.f_email.text().strip(), city=self.f_city.text().strip(),
            postal_code=self.f_postal.text().strip(), acuity_left=self.f_acl.text().strip(),
            acuity_right=self.f_acr.text().strip(), notes=self.f_notes.toPlainText().strip(),
            created=self.created)


class ExistingPatientDialog(QDialog):
    def __init__(self, parent, people: List[store.Participant], counts: dict):
        super().__init__(parent)
        self.setWindowTitle("Existing patient")
        self.resize(760, 460)
        self.people = people
        v = QVBoxLayout(self)
        self.table = QTableWidget(len(people), 6)
        self.table.setHorizontalHeaderLabels(["ID", "Name", "Age", "Group", "Sessions", "Created"])
        for i, p in enumerate(people):
            for j, val in enumerate([p.id, p.name, p.age, p.group, str(counts.get(p.id, 0)),
                                     p.created if p.registered else "(found in sessions)"]):
                self.table.setItem(i, j, QTableWidgetItem(val))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.doubleClicked.connect(lambda _: self.accept())
        v.addWidget(self.table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def selected(self) -> Optional[store.Participant]:
        rows = self.table.selectionModel().selectedRows()
        return self.people[rows[0].row()] if rows else None


# ------------------------------------------------------------------------------
# Luminance scales (Specvis: ViewNewLumScale / Existing / Edit / Fit)
# ------------------------------------------------------------------------------

class ScaleDialog(QDialog):
    """Six photometer readings, at brightness 0/20/40/60/80/100 %."""

    def __init__(self, parent, scale: Optional[store.LuminanceScale] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit luminance scale" if scale else "New luminance scale")
        s = scale or store.LuminanceScale(store.new_scale_id(), "", [0.0] * 6)
        form = QFormLayout(self)
        self.f_id = _readonly(s.id)
        self.f_name = QLineEdit(s.name)
        self.f_name.setPlaceholderText("e.g. LAB_MONITOR_GRAY")
        self.f_hue = _spin(0, 360, s.hue, 1, 0)
        self.f_sat = _spin(0, 100, s.saturation, 1, 0, "%")
        form.addRow("ID", self.f_id)
        form.addRow("Name", self.f_name)
        form.addRow("Hue", self.f_hue)
        form.addRow("Saturation", self.f_sat)
        form.addRow(_label("Show each brightness full screen (Screen check shows the grey), "
                           "measure with the photometer, type the reading:", "note"))
        self.f_lum = []
        for b, value in zip(MEASUREMENT_POINTS, s.measurements):
            box = _spin(0.0, 5000.0, value, 0.1, 2, "cd/m2")
            self.f_lum.append(box)
            form.addRow(f"Brightness {b:.0f} %", box)
        self.f_notes = QPlainTextEdit(s.notes)
        self.f_notes.setPlaceholderText("photometer, room, date")
        self.f_notes.setFixedHeight(60)
        form.addRow("Additional information", self.f_notes)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.created = s.created

    def _accept(self) -> None:
        vals = [b.value() for b in self.f_lum]
        if not self.f_name.text().strip():
            QMessageBox.warning(self, "Scale", "Give the scale a name.")
            return
        if vals[-1] <= vals[0]:
            QMessageBox.warning(self, "Scale", "The 100 % reading must be brighter than 0 %.")
            return
        self.accept()

    def scale(self) -> store.LuminanceScale:
        return store.LuminanceScale(self.f_id.text(), self.f_name.text().strip(),
                                    [b.value() for b in self.f_lum], self.f_hue.value(),
                                    self.f_sat.value(), self.f_notes.toPlainText().strip(),
                                    self.created)


class ChooseScaleDialog(QDialog):
    def __init__(self, parent, scales: List[store.LuminanceScale]):
        super().__init__(parent)
        self.setWindowTitle("Existing luminance scale")
        self.resize(640, 360)
        self.scales = scales
        v = QVBoxLayout(self)
        self.table = QTableWidget(len(scales) + 1, 3)
        self.table.setHorizontalHeaderLabels(["Name", "0 ... 100 % (cd/m2)", "Created"])
        self.table.setItem(0, 0, QTableWidgetItem(store.UNCALIBRATED))
        for i, s in enumerate(scales, start=1):
            self.table.setItem(i, 0, QTableWidgetItem(s.name))
            self.table.setItem(i, 1, QTableWidgetItem(", ".join(f"{m:g}" for m in s.measurements)))
            self.table.setItem(i, 2, QTableWidgetItem(s.created))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.doubleClicked.connect(lambda _: self.accept())
        v.addWidget(self.table)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return False
        return None if rows[0].row() == 0 else self.scales[rows[0].row() - 1]


class FitDialog(QDialog):
    """Specvis's Fit window: the readings, the fitted curve, chi-squared, p, STD."""

    def __init__(self, parent, scale: store.LuminanceScale, out_png: pathlib.Path):
        super().__init__(parent)
        self.setWindowTitle(f"Fit - {scale.name}")
        q = store.fit_quality(scale.measurements)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cal = scale.calibration()
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=110)
        b = np.arange(101)
        ax.plot(b, cal.luminance_at(b), color="#0088FF", label="fit (degree 2)")
        ax.plot(MEASUREMENT_POINTS, scale.measurements, "o", color="black", label="measured")
        ax.set_xlabel("Brightness (%)")
        ax.set_ylabel("Luminance (cd/m2)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        v = QVBoxLayout(self)
        img = QLabel()
        img.setPixmap(QPixmap(str(out_png)))
        v.addWidget(img)
        form = QFormLayout()
        form.addRow("Chi-squared", _readonly(f"{q['chi_squared']:.4f}"))
        form.addRow("P-value", _readonly(f"{q['p_value']:.4f}"))
        form.addRow("STD", _readonly(f"{q['std']:.4f}"))
        v.addLayout(form)
        v.addWidget(_label("A bad fit (large chi-squared, small p) means a bad measurement: "
                           "measure again.", "note"))
        v.addWidget(_button("Close", self.accept))


# ------------------------------------------------------------------------------
# Results (Specvis: ViewProcedureResults / ViewPatientResultsMap / Compare)
# ------------------------------------------------------------------------------

def _summary_rows(info: store.SessionInfo) -> List[dict]:
    if not info.summary_path.exists():
        return []
    with info.summary_path.open() as f:
        return list(csv.DictReader(f))


def _session_meta(info: store.SessionInfo) -> dict:
    try:
        return json.loads((info.folder / "session.json").read_text())
    except (OSError, ValueError):
        return {}


class ResultsDialog(QDialog):
    """Map, checks, points and info for one session. Specvis's results window."""

    OTHER_EYE, REPEAT = 10, 11

    def __init__(self, parent, info: store.SessionInfo, offer_next: bool = False):
        super().__init__(parent)
        self.info = info
        self.meta = _session_meta(info)
        self.rows = _summary_rows(info)
        self._tmp = pathlib.Path(tempfile.mkdtemp(prefix="refrem_map_"))
        self.setWindowTitle(f"Results - {info.participant_id} - {info.eye} eye - {info.started}")
        self.resize(1250, 880)
        v = QVBoxLayout(self)
        v.addWidget(self._checks_label())

        self.grid_png = self._tmp / "grid.png"
        if self.rows:
            specvis_map.draw_grid_map(self.rows, self.grid_png, meta=self.meta)
        elif info.map_path.exists():
            shutil.copy(info.map_path, self.grid_png)
        self.big_contour = FitImage()
        tabs = QTabWidget()
        tabs.addTab(self._map_tab(), "Maps")
        tabs.addTab(FitImage(self.grid_png, "(no grid map in this session)"), "Grid (large)")
        tabs.addTab(self.big_contour, "Contour (large)")
        tabs.addTab(self._points_tab(), "Points")
        info_text = QPlainTextEdit(json.dumps(self.meta, indent=2))
        info_text.setReadOnly(True)
        tabs.addTab(info_text, "Info")
        v.addWidget(tabs, 1)

        row = QHBoxLayout()
        if offer_next:
            other = "LEFT" if info.eye == "R" else "RIGHT"
            row.addWidget(_button(f"Test the {other} eye now", lambda: self.done(self.OTHER_EYE)))
            row.addWidget(_button("Repeat this eye", lambda: self.done(self.REPEAT)))
        row.addStretch(1)
        row.addWidget(_button("Open folder", lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(info.folder)))))
        row.addWidget(_button("Close", self.reject))
        v.addLayout(row)

    def _checks_label(self) -> QLabel:
        info = self.info

        def rate(key):
            val = info.rates.get(key)
            return "no trials" if val is None or (isinstance(val, float) and math.isnan(val)) \
                else f"{100 * val:.0f}%"

        tick = {True: "ok", False: "FAILED"}
        c = info.checks
        state = "SESSION VALID" if info.valid else ("STOPPED EARLY" if info.aborted else
                                                     "SESSION NOT VALID - repeat this eye")
        lines = [f"<b style='color:{'#0F6E56' if info.valid else '#A32D2D'};font-size:18px'>"
                 f"{info.participant_id} &nbsp; {info.eye} eye &nbsp; - &nbsp; {state}</b>",
                 f"{info.trials} trials, {info.duration_min:.1f} min &nbsp;|&nbsp; pilot mode "
                 f"{'on' if info.pilot_mode else 'off'} &nbsp;|&nbsp; "
                 f"false presses {rate('false_alarms')} [{tick[c.get('false_alarms', False)]}]"
                 f" &nbsp; missed strong {rate('misses')} [{tick[c.get('misses', False)]}]"
                 f" &nbsp; fixation losses {rate('fixation')} [{tick[c.get('fixation', False)]}]"]
        lines += [FAILURE_REASONS[k] for k, ok in c.items() if not ok and k in FAILURE_REASONS]
        lab = QLabel("<br>".join(lines))
        lab.setWordWrap(True)
        return lab

    def _map_tab(self) -> QWidget:
        """The Specvis contour map and the coloured grid (seen / weak / blind, with the
        value in each box) side by side, map controls underneath."""
        w = QWidget()
        v = QVBoxLayout(w)
        maps = QHBoxLayout()
        self.map_image = FitImage()
        self.map_image.setStyleSheet("background:#FFFFFF; border:1px solid #B4B2A9;")
        maps.addWidget(self.map_image, 1)
        self.grid_image = FitImage(self.grid_png, "(no grid map in this session)")
        self.grid_image.setStyleSheet("background:#FFFFFF; border:1px solid #B4B2A9;")
        maps.addWidget(self.grid_image, 1)
        v.addLayout(maps, 1)

        controls = QHBoxLayout()
        self.iso = _spin(0.5, 10.0, 3.0, 0.5, 1, "dB")
        self.interp = _ispin(1, 12, 6)
        self.density = _ispin(2, 20, 7)
        self.scheme = QComboBox()
        self.scheme.addItems([specvis_map.COLOR, specvis_map.MONO])
        self.axes = QCheckBox("Show axes.")
        self.axes.setChecked(True)
        self.fix = QCheckBox("Show fixation point and blindspot positions.")
        self.fix.setChecked(True)
        self.values = QCheckBox("Show values.")
        self.values.setChecked(True)
        for label, widget in (("Isofactor", self.iso), ("Interpolation", self.interp),
                              ("Colorbar density", self.density), ("Map", self.scheme)):
            controls.addWidget(_label(label))
            controls.addWidget(widget)
        v.addLayout(controls)
        row2 = QHBoxLayout()
        for widget in (self.axes, self.fix, self.values):
            row2.addWidget(widget)
        row2.addStretch(1)
        for text, slot in (("Redraw map", self.redraw), ("Export data", self.export),
                           ("Save map image", self.save_image)):
            row2.addWidget(_button(text, slot))
        v.addLayout(row2)
        self.redraw()
        return w

    def _map_png(self) -> pathlib.Path:
        path = self._tmp / "contour.png"
        specvis_map.draw_contour_map(
            self.rows, path, meta=self.meta,
            isofactor_db=self.iso.value(), interpolation=self.interp.value(),
            colorbar_density=self.density.value(), scheme=self.scheme.currentText(),
            show_axes=self.axes.isChecked(), show_fixation=self.fix.isChecked(),
            show_values=self.values.isChecked())
        return path

    def redraw(self) -> None:
        png = self._map_png()
        self.map_image.set_image(png)
        self.big_contour.set_image(png)

    def export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export data",
                                              str(self.info.folder / "map_data.csv"), "CSV (*.csv)")
        if path:
            specvis_map.export_map_data(self.rows, pathlib.Path(path), self.interp.value())

    def save_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save map image",
                                              str(self.info.folder / "contour_map.png"), "PNG (*.png)")
        if path:
            shutil.copy(self._map_png(), path)

    def _points_tab(self) -> QWidget:
        cols = ["x_deg", "y_deg", "quadrant", "category", "threshold_contrast", "threshold_db",
                "trials", "termination", "starting_guess"]
        t = QTableWidget(len(self.rows), len(cols))
        t.setHorizontalHeaderLabels(cols)
        for i, r in enumerate(self.rows):
            for j, c in enumerate(cols):
                t.setItem(i, j, QTableWidgetItem(str(r.get(c, ""))))
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return t


class CompareDialog(QDialog):
    """Specvis's compare maps and compare table, for two sessions."""

    def __init__(self, parent, a: store.SessionInfo, b: store.SessionInfo):
        super().__init__(parent)
        self.setWindowTitle("Compare results")
        self.resize(1500, 760)
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="refrem_cmp_"))
        tabs = QTabWidget()
        maps = QWidget()
        h = QHBoxLayout(maps)
        ra, rb = _summary_rows(a), _summary_rows(b)
        for info, rows, name in ((a, ra, "a.png"), (b, rb, "b.png")):
            png = specvis_map.draw_contour_map(rows, tmp / name, meta=_session_meta(info))
            h.addWidget(FitImage(png), 1)
        tabs.addTab(maps, "Maps")
        da = {(r["x_deg"], r["y_deg"]): r for r in ra}
        db_ = {(r["x_deg"], r["y_deg"]): r for r in rb}
        keys = sorted(set(da) | set(db_), key=lambda k: (-float(k[1]), float(k[0])))
        t = QTableWidget(len(keys), 6)
        t.setHorizontalHeaderLabels(["x_deg", "y_deg", "A threshold dB", "B threshold dB",
                                     "B - A (dB)", "A / B category"])
        for i, k in enumerate(keys):
            ra_, rb_ = da.get(k, {}), db_.get(k, {})
            va, vb = ra_.get("threshold_db", ""), rb_.get("threshold_db", "")
            diff = f"{float(vb) - float(va):+.2f}" if va and vb else ""
            for j, val in enumerate([k[0], k[1], va, vb, diff,
                                     f"{ra_.get('category', '-')} / {rb_.get('category', '-')}"]):
                t.setItem(i, j, QTableWidgetItem(str(val)))
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tabs.addTab(t, "Table")
        v = QVBoxLayout(self)
        v.addWidget(tabs)
        v.addWidget(_button("Close", self.accept))


class PatientResultsDialog(QDialog):
    """Specvis's patient results: every session, with Info, Map, Delete and Compare."""

    def __init__(self, parent, participant_id: str, perimetry_dir: pathlib.Path):
        super().__init__(parent)
        self.setWindowTitle(f"Results - {participant_id}")
        self.resize(920, 520)
        self.participant_id, self.perimetry_dir = participant_id, perimetry_dir
        v = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.doubleClicked.connect(lambda _: self.show_map())
        v.addWidget(self.table)
        v.addWidget(_label("Select two sessions (Cmd/Ctrl-click) to compare them. Delete moves "
                           "a session to output/perimetry/_deleted/ (recoverable).", "note"))
        bar = QFrame()
        bar.setObjectName("bar")
        bar.setFixedHeight(85)
        row = QHBoxLayout(bar)
        row.setContentsMargins(25, 0, 25, 0)
        row.setSpacing(10)
        row.addWidget(_fixed(_button("Delete", self.delete, "red"), 100, 50))
        row.addStretch(1)
        row.addWidget(_fixed(_button("Cancel", self.reject, "orange"), 100, 50))
        row.addWidget(_fixed(_button("Compare", self.compare, "nav"), 120, 50))
        row.addWidget(_fixed(_button("Info", self.show_info, "nav"), 100, 50))
        row.addWidget(_fixed(_button("Map", self.show_map, "nav"), 100, 50))
        v.addWidget(bar)
        self.reload()

    def reload(self) -> None:
        self.sessions = store.list_sessions(self.perimetry_dir, self.participant_id)
        cols = ["Date", "Eye", "Result", "Trials", "Minutes", "Pilot mode", "Grid"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(self.sessions))
        for i, s in enumerate(self.sessions):
            state = "VALID" if s.valid else ("stopped" if s.aborted else "NOT valid")
            for j, val in enumerate([s.started, s.eye, state, str(s.trials),
                                     f"{s.duration_min:.1f}", "on" if s.pilot_mode else "off",
                                     s.extent]):
                self.table.setItem(i, j, QTableWidgetItem(val))
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

    def selected(self) -> List[store.SessionInfo]:
        return [self.sessions[r.row()] for r in self.table.selectionModel().selectedRows()]

    def show_map(self) -> None:
        sel = self.selected()
        if sel:
            ResultsDialog(self, sel[0]).exec()

    def show_info(self) -> None:
        sel = self.selected()
        if sel:
            d = ResultsDialog(self, sel[0])
            d.findChild(QTabWidget).setCurrentIndex(3)
            d.exec()

    def compare(self) -> None:
        sel = self.selected()
        if len(sel) != 2:
            QMessageBox.information(self, "Compare results", "Select exactly two sessions.")
            return
        CompareDialog(self, sel[0], sel[1]).exec()

    def delete(self) -> None:
        sel = self.selected()
        if not sel:
            return
        names = "\n".join(s.folder.name for s in sel)
        if QMessageBox.question(self, "Delete", f"Move these sessions to _deleted?\n\n{names}") \
                != QMessageBox.StandardButton.Yes:
            return
        trash = self.perimetry_dir / "_deleted"
        trash.mkdir(exist_ok=True)
        for s in sel:
            shutil.move(str(s.folder), str(trash / s.folder.name))
        self.reload()


# ------------------------------------------------------------------------------
# Step 4 sub-dialogs (Specvis: fixation monitor settings, procedure settings)
# ------------------------------------------------------------------------------

#: Step 4's "Fix. monitor technique" choices, and the setting each one means.
FIX_CHOICES = {"Blindspot": FIX_BLIND_SPOT, "Eye tracker": FIX_TRACKER,
               "Mouse (test)": FIX_MOUSE, "None": FIX_NONE}


class FixMonitorDialog(QDialog):
    def __init__(self, parent, s: PerimetrySettings):
        super().__init__(parent)
        self.setWindowTitle("Fixation monitor settings - blind spot")
        form = QFormLayout(self)
        self.temporal = _spin(10.0, 20.0, s.blind_spot_temporal_deg, 0.5, 1, "deg")
        self.below = _spin(-5.0, 5.0, s.blind_spot_below_deg, 0.5, 1, "deg")
        self.contrast = _spin(0.1, 1.0, s.blind_spot_contrast, 0.05, 2)
        self.share = _spin(1.0, 20.0, 100 * s.p_blind_spot, 1.0, 0, "% of trials")
        self.search = QCheckBox("Find each person's blind spot at the start")
        self.search.setChecked(s.blind_spot_search)
        self.search.setToolTip("As the Humphrey does: before the test, the patch is flashed "
                               "at the average position and then at nearby ones until one is "
                               "missed twice; the checks then use that position. Off = the "
                               "average position for everyone.")
        self.limit = _spin(0.0, 100.0, 100 * s.max_fixation_loss_rate, 5.0, 0, "%")
        self.tolerance = _spin(1.0, 5.0, s.fixation_tolerance_deg, 0.5, 1, "deg")
        self.hold = _spin(0.0, 1000.0, s.fixation_hold_ms, 50.0, 0, "ms")
        form.addRow(_label("Blind spot (Heijl-Krakau)", "note"))
        form.addRow(self.search)
        form.addRow("Blindspot distance from fix. point X (temporal)", self.temporal)
        form.addRow("Blindspot distance from fix. point Y (below)", self.below)
        form.addRow("Catch stimulus contrast", self.contrast)
        form.addRow("Monitor fixation on", self.share)
        form.addRow(_label("Heijl-Krakau: a strong patch at the blind spot. A press means the "
                           "eye moved. Only meaningful with the other eye covered, and NOT "
                           "when the blind spot falls in a blind half of the field.", "note"))
        form.addRow(_label("Eye tracker / mouse", "note"))
        form.addRow("Eye must stay within", self.tolerance)
        form.addRow("Eye on the + before each patch", self.hold)
        form.addRow(_label("A patch is shown only once the eye is on the +. A trial in which "
                           "the eye left the + is thrown away and tested again later.", "note"))
        form.addRow("Fixation losses allowed, at most", self.limit)
        b = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        b.accepted.connect(self.accept)
        b.rejected.connect(self.reject)
        form.addRow(b)

    def values(self) -> dict:
        return dict(blind_spot_search=self.search.isChecked(),
                    blind_spot_temporal_deg=self.temporal.value(),
                    blind_spot_below_deg=self.below.value(),
                    blind_spot_contrast=self.contrast.value(),
                    p_blind_spot=self.share.value() / 100.0,
                    max_fixation_loss_rate=self.limit.value() / 100.0,
                    fixation_tolerance_deg=self.tolerance.value(),
                    fixation_hold_ms=self.hold.value())


def _range_line(s: PerimetrySettings, cal) -> str:
    """The measurable dB range at this background, and where the staircase seeds start."""
    r = contrast_range_db(s, cal)
    start = (f"{-20 * math.log10(s.ft_start_contrast):.1f} dB (given)" if s.ft_start_contrast
             else f"{r['middle_db']:.1f} dB (middle of the range)")
    return (f"dB range           {r['strongest_db'] + 0.0:.1f} (strongest) to "
            f"{r['faintest_db']:.1f} (faintest{', dithered' if s.dithering else ''}); "
            f"one grey step {r['grey_step_db']:.1f} dB; staircase seeds start {start}"
            + ("" if r["calibrated"] else "  [ASSUMED screen: measure with the photometer]"))


class ProcedureDialog(QDialog):
    def __init__(self, parent, s: PerimetrySettings, table_text: str, build: Callable):
        super().__init__(parent)
        self.setWindowTitle(f"Procedure settings - {PROCEDURE}")
        form = QFormLayout(self)
        self.pilot = QCheckBox("Pilot mode (the 15-20 table people: no starting guesses, "
                               "builds the normal table)")
        self.pilot.setChecked(s.pilot_mode)
        self.sd = _spin(0.05, 0.5, s.zest.sd_stop, 0.01, 2, "log10")
        self.mn = _ispin(1, 20, s.zest.min_trials)
        self.mx = _ispin(2, 60, s.zest.max_trials)
        self.blank = _spin(0.0, 20.0, 100 * s.p_blank, 1.0, 0, "%")
        self.high = _spin(0.0, 20.0, 100 * s.p_high, 1.0, 0, "%")
        self.fa = _spin(0.0, 100.0, 100 * s.max_false_alarm_rate, 1.0, 0, "%")
        self.miss = _spin(0.0, 100.0, 100 * s.max_miss_rate, 1.0, 0, "%")
        self.brk = _ispin(0, 500, s.break_every_trials, "trials")
        self.practice = _ispin(0, 20, s.practice_trials)
        form.addRow(self.pilot)
        form.addRow("Stop when the estimate's SD is below", self.sd)
        form.addRow("Trials per location, at least", self.mn)
        form.addRow("Trials per location, at most", self.mx)
        form.addRow("Blank catch trials (false presses)", self.blank)
        form.addRow("Strong catch trials (misses)", self.high)
        form.addRow("False presses allowed, at most", self.fa)
        form.addRow("Missed strong patches allowed, at most", self.miss)
        form.addRow("Break every", self.brk)
        form.addRow("Practice trials", self.practice)
        # Full Threshold (4-2). Contrast dB: 2 of ours = 1 Humphrey dB.
        self.ft_start = _spin(0.0, 1.0, s.ft_start_contrast, 0.01, 3)
        self.ft_coarse = _spin(1.0, 20.0, s.ft_coarse_step_db, 1.0, 1, "dB")
        self.ft_fine = _spin(0.5, 10.0, s.ft_fine_step_db, 0.5, 1, "dB")
        self.ft_retest = _spin(0.0, 30.0, s.ft_retest_db, 1.0, 1, "dB")
        self.ft_blind = QCheckBox("Start at the maximum next to blind neighbours "
                                  "(our suggestion, NOT the published method)")
        self.ft_blind.setChecked(s.ft_blind_neighbour_start)
        form.addRow(_label("Full Threshold (4-2), in contrast dB "
                           "(2 of ours = 1 Humphrey dB):"))
        form.addRow("Seed points start at contrast (0 = middle of the dB range)",
                    self.ft_start)
        form.addRow("First step", self.ft_coarse)
        form.addRow("Step after the first reversal", self.ft_fine)
        form.addRow("Second staircase if further from the start than", self.ft_retest)
        form.addRow(self.ft_blind)
        form.addRow("Normal table", _label(table_text))
        form.addRow(_button("Build normal table from valid pilot sessions", build))
        b = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        b.accepted.connect(self.accept)
        b.rejected.connect(self.reject)
        form.addRow(b)

    def values(self) -> dict:
        return dict(pilot_mode=self.pilot.isChecked(), p_blank=self.blank.value() / 100.0,
                    p_high=self.high.value() / 100.0,
                    max_false_alarm_rate=self.fa.value() / 100.0,
                    max_miss_rate=self.miss.value() / 100.0,
                    break_every_trials=self.brk.value(), practice_trials=self.practice.value(),
                    ft_start_contrast=self.ft_start.value(),
                    ft_coarse_step_db=self.ft_coarse.value(),
                    ft_fine_step_db=self.ft_fine.value(),
                    ft_retest_db=self.ft_retest.value(),
                    ft_blind_neighbour_start=self.ft_blind.isChecked(),
                    zest=dict(sd_stop=self.sd.value(), min_trials=self.mn.value(),
                              max_trials=max(self.mn.value(), self.mx.value())))


class DistributionDialog(QDialog):
    def __init__(self, parent, s: PerimetrySettings):
        super().__init__(parent)
        self.setWindowTitle("Preview stimuli distribution")
        v = QVBoxLayout(self)
        g = GridPreview()
        g.setMinimumSize(900, 520)
        g.set_settings(s)
        v.addWidget(g)
        v.addWidget(_button("Close", self.accept))


# ------------------------------------------------------------------------------
# The main window
# ------------------------------------------------------------------------------

class SpecvisWindow(QMainWindow):
    def __init__(self, perimetry_dir: pathlib.Path = store.PERIMETRY_DIR,
                 participants_file: pathlib.Path = store.PARTICIPANTS_FILE,
                 presets_dir: pathlib.Path = store.PRESETS_DIR,
                 prior_table_file: pathlib.Path = store.PRIOR_TABLE_FILE,
                 scales_file: pathlib.Path = store.SCALES_FILE):
        super().__init__()
        self.perimetry_dir = pathlib.Path(perimetry_dir)
        self.presets_dir = pathlib.Path(presets_dir)
        self.presets = store.list_presets(self.presets_dir)
        self.prior_table_file = pathlib.Path(prior_table_file)
        self.last_file = self.perimetry_dir.parent / "specvis_last.json"
        self.people = store.ParticipantStore(participants_file)
        self.scales = store.ScaleStore(scales_file)
        self.settings = PerimetrySettings()
        self.patient: Optional[store.Participant] = None
        self.scale_id: Optional[str] = self._newest_measured_scale()
        self.settings_name = "(defaults)"
        self.process: Optional[QProcess] = None
        self.last_session: Optional[store.SessionInfo] = None
        self._job_dir = pathlib.Path(tempfile.mkdtemp(prefix="refrem_job_"))
        self._loading = False
        self._clock_start = 0.0

        self.setWindowTitle("Specvis - ReFreM")
        self.resize(980, 720)
        self.setMinimumSize(940, 690)
        self.setStyleSheet(STYLE.replace("@DOWN@", (ASSETS / "arrow_down.png").as_posix())
                           .replace("@UP@", (ASSETS / "arrow_up.png").as_posix())
                           .replace("@CHECK@", (ASSETS / "check.png").as_posix()))
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.pages = QStackedWidget()
        for build in (self._step1, self._step2, self._step3, self._step4, self._start):
            self.pages.addWidget(build())
        root.addWidget(self.pages, 1)
        root.addWidget(self._bottom_bar())
        self._load_last()
        self._go(0)

    # -- bottom bar (Specvis: Exit/Back, white pill with Step 1-4 / START ticks, Next) ----

    def _bottom_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("bar")
        bar.setFixedHeight(85)
        h = QHBoxLayout(bar)
        h.setContentsMargins(25, 0, 25, 0)
        h.setSpacing(25)
        self.left_button = _button("Exit", self._left, "red")
        self.left_button.setFixedSize(100, 50)
        h.addWidget(self.left_button)
        pill = QFrame()
        pill.setObjectName("pill")
        pill.setFixedHeight(50)
        ph = QHBoxLayout(pill)
        ph.setContentsMargins(10, 2, 10, 2)
        ph.setSpacing(0)
        ph.addStretch(1)
        self.step_ticks = []
        for name in STEP_NAMES:
            cell = QWidget()
            cell.setFixedWidth(100)
            cv = QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(2)
            tick = StepTick()
            cv.addWidget(tick, 0, Qt.AlignmentFlag.AlignHCenter)
            lab = QLabel(name)
            lab.setObjectName("steplabel")
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cv.addWidget(lab)
            self.step_ticks.append(tick)
            ph.addWidget(cell)
        ph.addStretch(1)
        h.addWidget(pill, 1)
        self.next_button = _button("Next", lambda: self._go(self.pages.currentIndex() + 1), "nav")
        self.next_button.setFixedSize(100, 50)
        self.start_button = _button("Start", self.start_test, "green")
        self.start_button.setFixedSize(100, 50)
        right = QStackedWidget()
        right.setFixedSize(100, 50)
        right.addWidget(self.next_button)
        right.addWidget(self.start_button)
        self.right_stack = right
        h.addWidget(right)
        return bar

    def _left(self) -> None:
        if self.pages.currentIndex() == 0:
            self.close()
        else:
            self._go(self.pages.currentIndex() - 1)

    def _go(self, index: int) -> None:
        index = max(0, min(index, self.pages.count() - 1))
        if index > 0 and self.patient is None and self.pages.currentIndex() == 0:
            QMessageBox.information(self, "Patient", "Choose a patient first (New or Existing).")
            index = 0
        self.pages.setCurrentIndex(index)
        self.left_button.setText("Exit" if index == 0 else "Back")
        self.left_button.setObjectName("red" if index == 0 else "nav")
        self.left_button.setStyle(self.left_button.style())
        last = index == self.pages.count() - 1
        self.right_stack.setCurrentWidget(self.start_button if last else self.next_button)
        self.next_button.setVisible(not last)
        for i, tick in enumerate(self.step_ticks):
            tick.set_checked(i <= index)
        self._refresh()

    # -- Step 1: patient (Specvis: ViewStartWindow) -----------------------------------

    def _step1(self) -> QWidget:
        page, v = _body()
        logo = QLabel()
        pix = QPixmap(str(ASSETS / "specvis_logo.jpg"))
        if not pix.isNull():
            logo.setPixmap(pix.scaled(360, 135, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(logo)
        sub = QLabel("ReFreM perimetry")
        sub.setObjectName("subtitle")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(sub)
        v.addWidget(_separator())
        v.addStretch(1)
        buttons = [_fixed(_button(t, s), 100, 40) for t, s in (
            ("New", self.patient_new), ("Existing", self.patient_existing),
            ("Edit", self.patient_edit), ("Results", self.patient_results))]
        v.addLayout(_row([_lab("Patient", 100, 40, 20)] + buttons, spacing=25))
        v.addSpacing(20)
        self.f_patient_id, self.f_patient_name = _fixed(_readonly(), 350, 40), _fixed(_readonly(), 350, 40)
        v.addLayout(_row([_lab("", 100), _lab("ID", 100, 40, 20), self.f_patient_id], spacing=25))
        v.addSpacing(20)
        v.addLayout(_row([_lab("", 100), _lab("Name", 100, 40, 20), self.f_patient_name], spacing=25))
        v.addSpacing(20)
        self.f_eye = _fixed(QComboBox(), 350, 40)
        self.f_eye.addItems(["Right eye  (cover the LEFT)", "Left eye  (cover the RIGHT)"])
        self.f_eye.currentIndexChanged.connect(self._read)
        v.addLayout(_row([_lab("", 100), _lab("Tested eye", 100, 40, 20), self.f_eye], spacing=25))
        v.addSpacing(20)
        v.addWidget(_separator())
        v.addSpacing(10)
        self.f_settings = _fixed(_readonly(), 290, 40)
        load = _fixed(_button("...", self.load_settings), 40, 40)
        load.setToolTip("Load settings (presets are in python/presets/)")
        skip = _fixed(_button("Start", lambda: self._go(4), "green"), 100, 40)
        skip.setToolTip("Skip to START")
        v.addLayout(_row([_lab("Settings", 100, 40, 20), self.f_settings, load, skip], spacing=25))
        v.addStretch(1)
        return page

    def _show_patient(self) -> None:
        p = self.patient
        self.f_patient_id.setText(p.id if p else "")
        self.f_patient_name.setText((p.name or "(ID only)") if p else "")

    def _counts(self) -> dict:
        counts = {}
        for s in store.list_sessions(self.perimetry_dir):
            counts[s.participant_id] = counts.get(s.participant_id, 0) + 1
        return counts

    def patient_new(self) -> None:
        d = PatientDialog(self)
        if d.exec():
            try:
                self.people.add(d.patient())
            except ValueError as e:
                QMessageBox.warning(self, "Patient", str(e))
                return
            self.patient = self.people.get(d.patient().id)
            self._show_patient()
            self._refresh()

    def patient_existing(self) -> None:
        everyone = store.all_participants(self.people, store.list_sessions(self.perimetry_dir))
        d = ExistingPatientDialog(self, everyone, self._counts())
        if d.exec() and d.selected():
            self.patient = d.selected()
            self._show_patient()
            self._refresh()

    def patient_edit(self) -> None:
        if self.patient is None:
            QMessageBox.information(self, "Edit", "Choose a patient first.")
            return
        d = PatientDialog(self, self.patient)
        if d.exec():
            new = d.patient()
            if self.people.get(new.id):
                self.people.update(new)
            else:
                self.people.add(new)
            self.patient = self.people.get(new.id)
            self._show_patient()

    def patient_results(self) -> None:
        if self.patient is None:
            QMessageBox.information(self, "Results", "Choose a patient first.")
            return
        PatientResultsDialog(self, self.patient.id, self.perimetry_dir).exec()

    # -- Step 2: screen and luminance scale (Specvis: ViewScreenAndLumScale) ------------

    def _step2(self) -> QWidget:
        page, v = _body()
        v.addStretch(1)
        self.f_screen = _fixed(QComboBox(), 320, 40)
        self._fill_screens()
        self.f_screen.currentIndexChanged.connect(self._read)
        v.addLayout(_row([_lab("Screen", 250, 40, 20), self.f_screen,
                          _fixed(_button("Refresh", self._fill_screens), 100, 40)]))
        self.f_res_x, self.f_res_y = _fixed(_readonly(), 210, 40), _fixed(_readonly(), 210, 40)
        v.addLayout(_row([_lab("Screen visual bounds (px)", 250, 40, 20), self.f_res_x, self.f_res_y]))
        self.f_width = _fixed(_spin(0.0, 3000.0, 544.0, 1.0, 1), 210, 40)
        self.f_height = _fixed(_readonly(), 210, 40)
        self.f_height.setToolTip("From the width and the screen's aspect ratio")
        v.addLayout(_row([_lab("Screen size (mm)", 250, 40, 20), self.f_width, self.f_height]))
        self.f_distance = _fixed(_spin(100.0, 2000.0, 520.0, 5.0, 0), 210, 40)
        check = _fixed(_button("Screen check", self.screen_check), 210, 40)
        check.setToolTip("Shows a box that must measure exactly 10 cm with a ruler")
        v.addLayout(_row([_lab("Patient distance (mm)", 250, 40, 20), self.f_distance, check]))
        self.f_field_x, self.f_field_y = _fixed(_readonly(), 210, 40), _fixed(_readonly(), 210, 40)
        v.addLayout(_row([_lab("Tested visual field (°)", 250, 40, 20), self.f_field_x, self.f_field_y]))
        self.fit_label = _fixed(QLabel(""), 680, 22)
        self.fit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addLayout(_row([self.fit_label]))
        self.f_fullscreen = QCheckBox()            # always full screen for data
        self.f_fullscreen.setChecked(True)
        self.f_fullscreen.setVisible(False)
        for w in (self.f_width, self.f_distance):
            w.valueChanged.connect(self._read)
        v.addWidget(_separator())
        scale_buttons = lambda: [_fixed(_button(t, s), 100, 40) for t, s in (  # noqa: E731
            ("New", self.scale_new), ("Existing", self.scale_existing),
            ("Edit", self.scale_edit), ("Fit", self.scale_fit))]
        v.addLayout(_row([_lab("Stimulus scale", 250, 40, 20)] + scale_buttons()))
        self.f_scale_name = _fixed(_readonly(), 430, 40)
        v.addLayout(_row([_lab("Scale name", 250, 40, 20), self.f_scale_name]))
        v.addWidget(_separator())
        v.addLayout(_row([_lab("Background scale", 250, 40, 20)] + scale_buttons()))
        self.f_bg_scale = _fixed(_readonly(), 430, 40)
        self.f_bg_scale.setToolTip("The Gabor's mean equals the background: one scale serves both")
        v.addLayout(_row([_lab("Scale name", 250, 40, 20), self.f_bg_scale]))
        v.addStretch(1)
        return page

    def _fill_screens(self) -> None:
        current = self.f_screen.currentIndex() if hasattr(self, "f_screen") else 0
        self.f_screen.blockSignals(True)
        self.f_screen.clear()
        for i, s in enumerate(QGuiApplication.screens()):
            g, r = s.geometry(), s.devicePixelRatio()
            self.f_screen.addItem(f"{i}: {s.name()} ({int(g.width() * r)}x{int(g.height() * r)})")
        self.f_screen.setCurrentIndex(max(0, min(current, self.f_screen.count() - 1)))
        self.f_screen.blockSignals(False)
        self._refresh()

    def monitor_pixels(self):
        screens = QGuiApplication.screens()
        i = max(0, min(self.f_screen.currentIndex(), len(screens) - 1))
        g, r = screens[i].geometry(), screens[i].devicePixelRatio()
        return int(g.width() * r), int(g.height() * r)

    def _newest_measured_scale(self) -> Optional[str]:
        """The newest ColorCAL-measured scale (python/specvis.sh calibrate), if any: used
        when no scale has been chosen, so a calibrated screen is never run as 'assumed'."""
        measured = [s for s in self.scales.all() if len(s.table) == 101]
        return max(measured, key=lambda s: s.created).id if measured else None

    def current_scale(self) -> Optional[store.LuminanceScale]:
        return self.scales.get(self.scale_id) if self.scale_id else None

    def calibration(self) -> LuminanceCalibration:
        s = self.current_scale()
        return s.calibration() if s else default_calibration(self.settings)

    def scale_new(self) -> None:
        d = ScaleDialog(self)
        if d.exec():
            self.scales.put(d.scale())
            self.scale_id = d.scale().id
            self._refresh()

    def scale_existing(self) -> None:
        d = ChooseScaleDialog(self, self.scales.all())
        if d.exec():
            sel = d.selected()
            if sel is not False:
                self.scale_id = sel.id if sel else None
                self._refresh()

    def scale_edit(self) -> None:
        s = self.current_scale()
        if s is None:
            QMessageBox.information(self, "Edit", "Choose or create a measured scale first.")
            return
        d = ScaleDialog(self, s)
        if d.exec():
            self.scales.put(d.scale())
            self._refresh()

    def scale_fit(self) -> None:
        s = self.current_scale()
        if s is None:
            QMessageBox.information(self, "Fit", "No measured scale chosen.")
            return
        FitDialog(self, s, self._job_dir / "fit.png").exec()

    # -- Step 3: stimulus and background (Specvis: ViewStimulusAndBackground) ------------

    def _step3(self) -> QWidget:
        page, v = _body()
        s = self.settings
        H = 34
        left = QVBoxLayout()
        left.setSpacing(4)
        self.f_db_min, self.f_db_max = _fixed(_readonly(), 85, H), _fixed(_readonly(), 85, H)
        self.f_db_range = QLineEdit()             # kept for the summary and tests
        self.f_db_range.setVisible(False)
        left.addLayout(_row([_lab("Decibel range (dB)", 220, H, 18), self.f_db_min, self.f_db_max],
                            spacing=10))
        left.addWidget(_separator())
        self.f_maxc = _fixed(_spin(0.05, 1.0, s.zest.max_contrast, 0.05, 2), 180, H)
        self.f_maxlum = _fixed(_readonly(), 180, H)
        left.addLayout(_row([_lab("Stim. max. contrast", 220, H, 18), self.f_maxc]))
        left.addLayout(_row([_lab("Stim. max. lum. (cd/m2)", 220, H, 18), self.f_maxlum]))
        left.addWidget(_separator())
        self.f_minc = _fixed(_spin(0.0005, 0.5, s.zest.min_contrast, 0.0005, 4), 180, H)
        self.f_minlum = _fixed(_readonly(), 180, H)
        left.addLayout(_row([_lab("Stim. min. contrast", 220, H, 18), self.f_minc]))
        left.addLayout(_row([_lab("Stim. min. lum. (cd/m2)", 220, H, 18), self.f_minlum]))
        left.addWidget(_separator())
        self.f_shape = _fixed(QComboBox(), 180, H)
        self.f_shape.addItems(["Gabor"])
        left.addLayout(_row([_lab("Stimulus shape", 220, H, 18), self.f_shape]))
        self.f_size = _fixed(_spin(0.5, 10.0, s.diameter_degrees, 0.5, 1), 85, H)
        self.f_size_y = _fixed(_readonly(), 85, H)
        left.addLayout(_row([_lab("Stimulus size XY (°)", 220, H, 18), self.f_size, self.f_size_y],
                            spacing=10))
        left.addWidget(_separator())
        self.f_time = _fixed(_spin(16.0, 2000.0, s.stimulus_ms, 10.0, 0), 180, H)
        left.addLayout(_row([_lab("Stim. display time (ms)", 220, H, 18), self.f_time]))
        self.f_iti_c = _fixed(_spin(0.0, 3000.0, s.iti_ms[0], 50.0, 0), 85, H)
        self.f_iti_r = _fixed(_spin(0.0, 3000.0, s.iti_ms[1] - s.iti_ms[0], 50.0, 0), 85, H)
        left.addLayout(_row([_lab("Interval const/rand (ms)", 220, H, 18), self.f_iti_c, self.f_iti_r],
                            spacing=10))
        self.f_fix_ms = _fixed(_spin(0.0, 3000.0, s.fixation_ms, 50.0, 0), 85, H)
        self.f_window = _fixed(_spin(200.0, 5000.0, s.response_window_ms, 100.0, 0), 85, H)
        fw = _lab("Fixation / answer (ms)", 220, H, 18)
        fw.setToolTip("Fixation cross alone before the patch; then the time allowed to answer")
        left.addLayout(_row([fw, self.f_fix_ms, self.f_window], spacing=10))
        left.addWidget(_separator())
        self.f_sf = _fixed(_spin(0.25, 16.0, s.spatial_frequency_cpd, 0.25, 2), 85, H)
        self.f_ori = _fixed(_spin(0.0, 180.0, s.orientation_degrees, 15.0, 0), 85, H)
        left.addLayout(_row([_lab("Gabor c/° · orientation °", 220, H, 17), self.f_sf, self.f_ori],
                            spacing=10))
        self.f_sigma, self.f_phase = _fixed(_readonly(), 85, H), _fixed(_readonly("0"), 85, H)
        left.addLayout(_row([_lab("Gabor sigma ° · phase °", 220, H, 17), self.f_sigma, self.f_phase],
                            spacing=10))

        right = QVBoxLayout()
        right.setSpacing(4)
        self.f_bg = _fixed(_spin(1.0, 200.0, s.background_cd_m2, 1.0, 1), 180, H)
        self.f_bg.setToolTip("Background luminance asked for, cd/m2. 10 = the Humphrey "
                             "perimeter's (SITA). Exact only with a photometer scale (Step 2).")
        self.f_bg_lum = _fixed(_readonly(), 180, H)
        self.f_bg_lum.setToolTip("What the screen gives: the nearest grey level, and whether it "
                                 "is measured or assumed")
        right.addLayout(_row([_lab("Bg. lum. wanted (cd/m2)", 220, H, 17), self.f_bg]))
        right.addLayout(_row([_lab("Bg. lum. shown (cd/m2)", 220, H, 17), self.f_bg_lum]))
        self.f_dither = _fixed(QCheckBox("Dithering (faint patches on a dark background)"), 450, H)
        self.f_dither.setChecked(s.dithering)
        self.f_dither.setToolTip("Noisy-bit dithering: lets an 8-bit screen show contrasts finer "
                                 "than one grey step. Keep ON with a dark background.")
        self.f_dither.toggled.connect(self._read)
        right.addLayout(_row([self.f_dither]))
        right.addWidget(_separator())
        self.f_grid = _fixed(QComboBox(), 450, H)
        for e in EXTENTS:
            self.f_grid.addItem(EXTENT_NAMES[e], e)
        self.f_grid.setToolTip("The test grid. With the eye tracker (Tobii Pro Fusion: 50-80 cm) "
                               "use '24-2 without top/bottom rows' (50-53 cm) or 'Aflah grid' "
                               "(50-60 cm). The full 24-2 needs about 33-37 cm on a 24\" screen.")
        right.addLayout(_row([self.f_grid]))
        self.f_spacing = _fixed(_readonly("6 / 6"), 85, H)
        self.f_spacing.setToolTip("Grid spacing across / up-down, in degrees; no point on a midline")
        right.addLayout(_row([_lab("Dist. between s. XY (°)", 220, H, 18), self.f_spacing],
                             spacing=10))
        right.addWidget(_separator())
        self.f_preview = QLabel()
        self.f_preview.setObjectName("display")
        self.f_preview.setFixedSize(290, 200)
        self.f_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.f_preview, 0, Qt.AlignmentFlag.AlignHCenter)
        right.addWidget(_fixed(_button("Preview stimuli distribution",
                                       lambda: DistributionDialog(self, self.settings).exec()),
                               290, H), 0, Qt.AlignmentFlag.AlignHCenter)
        right.addWidget(_separator())
        self.f_sph = QCheckBox("Correction for sphericity of the field of view.")
        self.f_sph.setChecked(True)
        self.f_sph.setEnabled(False)
        right.addWidget(self.f_sph, 0, Qt.AlignmentFlag.AlignHCenter)
        right.addStretch(1)

        for w in (self.f_size, self.f_sf, self.f_ori, self.f_maxc, self.f_minc, self.f_time,
                  self.f_fix_ms, self.f_window, self.f_iti_c, self.f_iti_r, self.f_bg):
            w.valueChanged.connect(self._read)
        self.f_grid.currentIndexChanged.connect(self._read)
        cols = QHBoxLayout()
        cols.addStretch(1)
        lw, rw = QWidget(), QWidget()
        lw.setLayout(left)
        rw.setLayout(right)
        lw.setFixedWidth(450)
        rw.setFixedWidth(450)
        cols.addWidget(lw)
        cols.addWidget(_vseparator())
        cols.addWidget(rw)
        cols.addStretch(1)
        v.addStretch(1)
        v.addLayout(cols)
        v.addStretch(1)
        return page

    # -- Step 4: fixation and other (Specvis: ViewFixationAndOther) -----------------------

    def _step4(self) -> QWidget:
        page, v = _body()
        top = QHBoxLayout()
        tl, tr = QVBoxLayout(), QVBoxLayout()
        tl.setSpacing(10)
        tr.setSpacing(10)
        self.f_fix_colour = _fixed(QComboBox(), 180, 40)
        self.f_fix_colour.addItems(COLOURS)
        self.f_fix_colour.setToolTip("Colour of the + the participant looks at (keep white)")
        tl.addLayout(_row([_lab("Fixation point color", 210, 40, 18), self.f_fix_colour], spacing=10))
        self.f_fix_lum = _fixed(_readonly("= screen white"), 180, 40)
        tl.addLayout(_row([_lab("Fix. point lum. (cd/m2)", 210, 40, 18), self.f_fix_lum], spacing=10))
        self.f_fix_size = _fixed(_readonly("0.5 / 0.5"), 180, 40)
        tl.addLayout(_row([_lab("Fix. point size XY (°)", 210, 40, 18), self.f_fix_size], spacing=10))
        self.f_fix_mon = _fixed(QComboBox(), 180, 40)
        self.f_fix_mon.addItems(list(FIX_CHOICES))
        self.f_fix_mon.setToolTip("Blindspot: Heijl-Krakau catch trials.  Eye tracker: Tobii Pro "
                                  "(gaze-contingent; trials with eye movements are repeated).  "
                                  "Mouse (test): the mouse stands in for the eye, to try the "
                                  "procedure without a tracker.")
        tr.addLayout(_row([_lab("Fix. monitor technique", 210, 40, 18), self.f_fix_mon], spacing=10))
        tr.addWidget(_fixed(_button("Fixation monitor settings", self.fix_monitor_settings), 400, 40),
                     0, Qt.AlignmentFlag.AlignHCenter)
        self.f_fix_loc = _fixed(_readonly("0 / 0"), 180, 40)
        tr.addLayout(_row([_lab("Fix. point loc. XY (°)", 210, 40, 18), self.f_fix_loc], spacing=10))
        top.addStretch(1)
        top.addWidget(_box(tl, 450))
        top.addWidget(_vseparator())
        top.addWidget(_box(tr, 450))
        top.addStretch(1)
        v.addStretch(1)
        v.addLayout(top)
        v.addWidget(_separator())
        self.grid_pane = GridPreview(legend=False)
        self.grid_pane.setObjectName("display")
        self.grid_pane.setFixedSize(290, 145)
        v.addWidget(self.grid_pane, 0, Qt.AlignmentFlag.AlignHCenter)
        v.addWidget(_fixed(_button("Preview stimuli distribution",
                                   lambda: DistributionDialog(self, self.settings).exec()), 290, 40),
                    0, Qt.AlignmentFlag.AlignHCenter)
        v.addWidget(_separator())
        bottom = QHBoxLayout()
        bl, br = QVBoxLayout(), QVBoxLayout()
        bl.setSpacing(10)
        br.setSpacing(10)
        self.f_key_answer, self.f_key_pause, self.f_key_cancel = (
            _fixed(QComboBox(), 200, 40), _fixed(QComboBox(), 200, 40), _fixed(QComboBox(), 200, 40))
        for box, name in ((self.f_key_answer, "Answer to stimulus"),
                          (self.f_key_pause, "Pause procedure"), (self.f_key_cancel, "Cancel procedure")):
            box.addItems(KEYS)
            bl.addLayout(_row([_lab(name, 190, 40, 18), box], spacing=10))
        self.f_procedure = _fixed(QComboBox(), 200, 40)
        self.f_procedure.addItems(["SITA Fast", "ZEST (full)", "Full Threshold (4-2)"])
        self.f_procedure.setToolTip("SITA Fast: SITA Fast's published rules around a ZEST engine "
                                    "(about 4 min a healthy eye, 8 min with hemianopia, with a "
                                    "normal table). ZEST (full): the slower, stricter search, "
                                    "used with pilot mode to build the normal table. Full "
                                    "Threshold (4-2): the Humphrey staircase as in the Open "
                                    "Perimetry Interface; needs no normal table.")
        self.f_procedure.currentIndexChanged.connect(self._read)
        br.addLayout(_row([_lab("Procedure", 190, 40, 18), self.f_procedure], spacing=10))
        br.addLayout(_row([_lab("", 190, 40), _fixed(_button("Settings", self.procedure_settings),
                                                      200, 40)], spacing=10))
        self.f_pilot = _fixed(QCheckBox("On"), 200, 40)
        self.f_pilot.setToolTip("ON for the 15-20 table people, with ZEST (full) (builds the normal table); "
                                "OFF afterwards (uses the table)")
        self.f_pilot.toggled.connect(self._read)
        br.addLayout(_row([_lab("Pilot mode", 190, 40, 18), self.f_pilot], spacing=10))
        bottom.addStretch(1)
        bottom.addWidget(_box(bl, 450))
        bottom.addWidget(_vseparator())
        bottom.addWidget(_box(br, 450))
        bottom.addStretch(1)
        v.addLayout(bottom)
        self.f_proc_summary = QLabel("")
        self.f_proc_summary.setVisible(False)       # shown in the START summary instead
        v.addStretch(1)
        for box in (self.f_fix_colour, self.f_fix_mon, self.f_key_answer, self.f_key_pause,
                    self.f_key_cancel):
            box.currentIndexChanged.connect(self._read)
        return page

    def fix_monitor_settings(self) -> None:
        d = FixMonitorDialog(self, self.settings)
        if d.exec():
            self._apply(d.values())

    def procedure_settings(self) -> None:
        d = ProcedureDialog(self, self.settings, self._table_text(), self.build_table)
        if d.exec():
            self._apply(d.values())

    # -- START (Specvis: ViewProcedurePreview) ---------------------------------------------

    def _start(self) -> QWidget:
        page, v = _body()
        cols = QHBoxLayout()
        left = QVBoxLayout()
        left.setSpacing(10)
        self.f_summary = QPlainTextEdit()
        self.f_summary.setObjectName("textArea")
        self.f_summary.setReadOnly(True)
        self.f_summary.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        left.addWidget(self.f_summary, 1)
        self.b_show = _fixed(_button("Show results", self.show_last_results), 450, 40)
        self.b_save = _fixed(_button("Save results", self.save_results), 450, 40)
        self.b_repeat = _fixed(_button("Repeat session", self.start_test), 450, 40)
        self.b_save_settings = _fixed(_button("Save settings", self.save_settings), 450, 40)
        for b in (self.b_show, self.b_save, self.b_repeat, self.b_save_settings):
            left.addWidget(b)
        right = QVBoxLayout()
        right.setSpacing(20)
        right.addStretch(1)
        self.clock = QLabel("00:00:00")
        self.clock.setObjectName("clock")
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.clock)
        self.state_circle = QLabel("Start")
        self.state_circle.setObjectName("circle")
        self.state_circle.setFixedSize(150, 150)
        self.state_circle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.state_circle, 0, Qt.AlignmentFlag.AlignHCenter)
        self.busy = QProgressBar()
        self.busy.setFixedWidth(240)
        self.busy.setTextVisible(False)
        self.busy.setRange(0, 1)
        right.addWidget(self.busy, 0, Qt.AlignmentFlag.AlignHCenter)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setObjectName("small")
        self.status.setFixedWidth(430)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.status, 0, Qt.AlignmentFlag.AlignHCenter)
        self.log = QPlainTextEdit()
        self.log.setObjectName("textArea")
        self.log.setReadOnly(True)
        self.log.setFixedSize(430, 90)
        right.addWidget(self.log, 0, Qt.AlignmentFlag.AlignHCenter)
        right.addStretch(1)
        cols.addStretch(1)
        cols.addWidget(_box(left, 450))
        cols.addWidget(_vseparator())
        cols.addWidget(_box(right, 450))
        cols.addStretch(1)
        v.addLayout(cols, 1)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        return page

    def _tick(self) -> None:
        t = int(time.monotonic() - self._clock_start)
        self.clock.setText(f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}")

    # -- settings <-> widgets ----------------------------------------------------------

    def _apply(self, values: dict) -> None:
        d = settings_to_dict(self.settings)
        for k, val in values.items():
            if k == "zest" and isinstance(val, dict):
                d["zest"].update(val)
            elif k not in ("participant_id",):
                d[k] = val
        self.settings = settings_from_dict(d)
        self._write()

    def _write(self) -> None:
        self._loading = True
        s = self.settings
        self.f_eye.setCurrentIndex(0 if s.eye == "R" else 1)
        self.f_width.setValue(s.screen_width_mm)
        self.f_distance.setValue(s.viewing_distance_mm)
        self.f_size.setValue(s.diameter_degrees)
        self.f_sf.setValue(s.spatial_frequency_cpd)
        self.f_ori.setValue(s.orientation_degrees)
        self.f_maxc.setValue(s.zest.max_contrast)
        self.f_minc.setValue(s.zest.min_contrast)
        self.f_time.setValue(s.stimulus_ms)
        self.f_fix_ms.setValue(s.fixation_ms)
        self.f_window.setValue(s.response_window_ms)
        self.f_iti_c.setValue(s.iti_ms[0])
        self.f_iti_r.setValue(s.iti_ms[1] - s.iti_ms[0])
        self.f_bg.setValue(s.background_cd_m2 if s.background_cd_m2 > 0 else 10.0)
        self.f_dither.setChecked(s.dithering)
        self.f_procedure.setCurrentIndex(STRATEGIES.index(s.strategy))
        self.f_grid.setCurrentIndex(max(0, self.f_grid.findData(s.extent)))
        self.f_fix_colour.setCurrentText(s.fixation_colour)
        self.f_fix_mon.setCurrentText(next((k for k, v in FIX_CHOICES.items()
                                            if v == s.fixation_monitoring), "Blindspot"))
        self.f_key_answer.setCurrentText(s.answer_key)
        self.f_key_pause.setCurrentText(s.pause_key)
        self.f_key_cancel.setCurrentText(s.cancel_key)
        self.f_pilot.setChecked(s.pilot_mode)
        self._loading = False
        self._refresh()

    def _read(self, *_) -> None:
        if self._loading:
            return
        d = settings_to_dict(self.settings)
        d.update(
            eye="R" if self.f_eye.currentIndex() == 0 else "L",
            screen_width_mm=self.f_width.value(), viewing_distance_mm=self.f_distance.value(),
            diameter_degrees=self.f_size.value(), spatial_frequency_cpd=self.f_sf.value(),
            orientation_degrees=self.f_ori.value(), stimulus_ms=self.f_time.value(),
            fixation_ms=self.f_fix_ms.value(), response_window_ms=self.f_window.value(),
            iti_ms=(self.f_iti_c.value(), self.f_iti_c.value() + self.f_iti_r.value()),
            background_cd_m2=self.f_bg.value(), dithering=self.f_dither.isChecked(),
            strategy=STRATEGIES[self.f_procedure.currentIndex()],
            extent=self.f_grid.currentData() or H24_2,
            fixation_colour=self.f_fix_colour.currentText(),
            fixation_monitoring=FIX_CHOICES.get(self.f_fix_mon.currentText(), FIX_BLIND_SPOT),
            answer_key=self.f_key_answer.currentText(), pause_key=self.f_key_pause.currentText(),
            cancel_key=self.f_key_cancel.currentText(), pilot_mode=self.f_pilot.isChecked())
        d["zest"] = dict(d["zest"], max_contrast=self.f_maxc.value(),
                         min_contrast=min(self.f_minc.value(), self.f_maxc.value() / 2))
        try:
            self.settings = settings_from_dict(d)
        except ValueError:
            pass
        self._refresh()

    # -- derived displays ------------------------------------------------------------------

    def problems(self) -> List[str]:
        out = []
        if self.patient is None:
            out.append("Choose a patient (Step 1: New or Existing).")
        if self.settings.screen_width_mm <= 0:
            out.append("Enter the measured screen width (Step 2).")
        else:
            w, h = self.monitor_pixels()
            p = fit_problem(self.settings, w, h)
            if p:
                out.append("Step 2: " + p)
        if (self.settings.fixation_monitoring == FIX_TRACKER
                and not 500.0 <= self.settings.viewing_distance_mm <= 800.0):
            out.append("Step 2: the eye tracker (Tobii Pro Fusion) works only 50-80 cm from the "
                       "eye; the viewing distance is "
                       f"{self.settings.viewing_distance_mm / 10:.0f} cm. Use a grid that fits "
                       "at 50 cm or more (24-2 without top/bottom rows, or Aflah grid).")
        try:
            self.settings.validate()
        except Exception as e:                         # noqa: BLE001 -- shown to the user
            out.append(str(e))
        return out

    def _table_text(self) -> str:
        if not self.prior_table_file.exists():
            return "none yet (run pilots, then build it)"
        t = PriorTable.load(self.prior_table_file)
        return f"{t.source} - {'not used (pilot mode)' if self.settings.pilot_mode else 'used'}"

    def _refresh(self) -> None:
        if not hasattr(self, "f_summary") or not hasattr(self, "f_preview"):
            return
        s = self.settings
        w, h = self.monitor_pixels()
        self.f_res_x.setText(str(w))
        self.f_res_y.setText(str(h))
        self.f_height.setText(f"{s.screen_width_mm * h / w:.1f} mm" if w else "")
        if s.screen_width_mm > 0 and w:
            hx, hy = field_half_degrees(s, w, h)
            self.f_field_x.setText(f"{2 * hx:.2f}")
            self.f_field_y.setText(f"{2 * hy:.2f}")
            p = fit_problem(s, w, h)
            self.fit_label.setText("The grid fits on this screen." if not p else "THE GRID DOES NOT FIT: " + p)
            self.fit_label.setStyleSheet("color: %s;" % ("#0F6E56" if not p else "#A32D2D"))
        else:
            self.fit_label.setText("Enter the measured visible width of this screen.")
            self.fit_label.setStyleSheet("color: #A32D2D;")
        scale = self.current_scale()
        self.f_scale_name.setText(scale.name if scale else store.UNCALIBRATED)
        self.f_bg_scale.setText(self.f_scale_name.text() + "  (same as stimulus)")
        cal = self.calibration()
        bg_lum, bg_b = background_for(s, cal)
        self.f_bg_lum.setText(f"{bg_lum:.2f}" + ("" if cal.calibrated else " (assumed)"))
        self.f_bg_lum.setToolTip(f"Grey level RGB {round(bg_b * 255)}. "
                                 + ("Measured with the photometer scale." if cal.calibrated
                                    else "Assumed display (no photometer scale yet): "
                                    + cal.source))
        floor = 0.0 if s.dithering else minimum_renderable_contrast(cal, bg_lum)
        top = min(s.zest.max_contrast, cal.max_achievable_contrast(bg_lum))
        lo_c = max(floor, s.zest.min_contrast)
        self.f_db_min.setText(f"{-20 * math.log10(top) + 0.0:.1f}")
        self.f_db_max.setText(f"{-20 * math.log10(lo_c):.1f}")
        lum = bg_lum
        self.f_maxlum.setText(f"{lum * (1 + top):.2f}" + ("" if cal.calibrated else " (assumed)"))
        self.f_minlum.setText(f"{lum * (1 + lo_c):.3f}" + ("" if cal.calibrated else " (assumed)"))
        self.f_size_y.setText(f"{s.diameter_degrees:g}")
        self.grid_pane.set_settings(s)
        self.f_db_range.setText(f"{-20 * math.log10(top) + 0.0:.1f} to {-20 * math.log10(lo_c):.1f} dB"
                                f"   (contrast {top:.2f} to {lo_c:.4f})")
        self.f_sigma.setText(f"{s.diameter_degrees / 6:.3f}")
        self.f_spacing.setText("6 / 6" if s.extent.startswith("24-2") else "4 / 6")
        self._preview()
        self.f_proc_summary.setText(
            f"{STRATEGY_NAMES[s.strategy]};  pilot mode {'ON' if s.pilot_mode else 'off'};  "
            + (f"steps {s.ft_coarse_step_db:g} then {s.ft_fine_step_db:g} dB, stop after 2 "
               f"reversals, second staircase if > {s.ft_retest_db:g} dB from the start"
               + (", blind-neighbour start ON (not in the published method)"
                  if s.ft_blind_neighbour_start else "")
               if s.strategy == FULL_THRESHOLD else
               (f"stop: SITA Fast (seed points SD < {s.zest.sd_stop:.2f} log10)"
                if s.strategy == SITA_FAST else f"stop at SD < {s.zest.sd_stop:.2f} log10")
               + f", {s.zest.min_trials}-{s.zest.max_trials} trials per location")
            + ";  catch: blank "
            f"{100 * s.p_blank:.0f}%, strong {100 * s.p_high:.0f}%, blind spot "
            f"{100 * s.p_blind_spot:.0f}%;  normal table: {self._table_text()}")
        kept, _ = grid_points(s)
        p = self.patient
        lines = [
            f"Patient            {p.id if p else '(none)'}{('  ' + p.name) if p and p.name else ''}"
            f"{('  [' + p.group + ']') if p and p.group else ''}",
            f"Tested eye         {s.eye}   -> cover the {'LEFT' if s.eye == 'R' else 'RIGHT'} eye",
            f"Settings           {self.settings_name}",
            f"Screen             {w}x{h} px, {s.screen_width_mm:.0f} mm wide",
            f"Patient distance   {s.viewing_distance_mm:.0f} mm",
            f"Luminance scale    {self.f_scale_name.text()}",
            f"Stimulus           Gabor {s.diameter_degrees:g}°, {s.spatial_frequency_cpd:g} c/°, "
            f"{s.orientation_degrees:g}°, {s.stimulus_ms:.0f} ms",
            f"Contrast           {s.zest.min_contrast:g} to {s.zest.max_contrast:g}",
            _range_line(s, cal),
            f"Timing             + {s.fixation_ms:.0f} ms, answer <= {s.response_window_ms:.0f} ms"
            + (", adaptive" if s.adaptive_window else ""),
            f"Interval           {s.iti_ms[0]:.0f} + 0-{s.iti_ms[1] - s.iti_ms[0]:.0f} ms",
            f"Background         {self.f_bg_lum.text()} cd/m2"
            + (", dithered" if s.dithering else ""),
            f"Stimuli            {EXTENT_NAMES.get(s.extent, s.extent)}: {len(kept)} tested",
            f"Fixation           {s.fixation_colour} +, check: {s.fixation_monitoring}",
            f"Keys               {s.answer_key} / {s.pause_key} / {s.cancel_key}",
            f"Procedure          {STRATEGY_NAMES[s.strategy]}, pilot mode "
            f"{'ON' if s.pilot_mode else 'off'}",
            f"Normal table       {self._table_text()}",
        ]
        problems = self.problems()
        if problems:
            lines += ["", "BEFORE YOU CAN START:"] + ["  - " + x for x in problems]
        self.f_summary.setPlainText("\n".join(lines))
        running = self.process is not None
        self.start_button.setEnabled(not problems and not running)
        self.b_repeat.setEnabled(self.last_session is not None and not running and not problems)
        for b in (self.b_show, self.b_save):
            b.setEnabled(self.last_session is not None and not running)

    def _preview(self) -> None:
        s = self.settings
        ppd = 200.0 / max(s.diameter_degrees, 0.5)
        cal = self.calibration()
        mean, _ = background_for(s, cal)
        g = GaborParameters(s.diameter_degrees, s.spatial_frequency_cpd, s.orientation_degrees,
                            0.0, s.diameter_degrees / 6.0, mean,
                            min(0.8, s.zest.max_contrast, cal.max_achievable_contrast(mean)))
        b = render_brightness(g, cal, ppd, ppd)
        img = np.ascontiguousarray((b * 255).astype(np.uint8))
        self._preview_bytes = img
        q = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format.Format_Grayscale8)
        self.f_preview.setPixmap(QPixmap.fromImage(q).scaled(190, 190, Qt.AspectRatioMode.KeepAspectRatio))

    # -- presets and settings files ----------------------------------------------------------

    def apply_preset(self, index: int) -> None:
        if index <= 0:
            return
        p = self.presets[index - 1]
        self._apply(p.settings)
        self.settings_name = p.name
        self.f_settings.setText(p.name)
        self.f_settings.setToolTip(p.description)
        if self.settings.screen_width_mm <= 0:
            QMessageBox.information(
                self, "Measure the screen",
                "This preset has no screen width yet. Measure the visible width of the monitor, "
                "enter it in Step 2, run the Screen check, then Save settings over this preset.")
        self._refresh()

    def load_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", str(self.presets_dir),
                                              "ReFreM settings (*.json)")
        if path:
            try:
                preset = store.load_settings_file(pathlib.Path(path))
            except (OSError, ValueError) as e:
                QMessageBox.warning(self, "Load settings", str(e))
                return
            self._apply(preset.settings)
            self.settings_name = preset.name
            self.f_settings.setText(preset.name)
            self.f_settings.setToolTip(preset.description)
            if self.settings.screen_width_mm <= 0:
                QMessageBox.information(
                    self, "Measure the screen",
                    "These settings have no screen width yet. Measure the visible width of the "
                    "monitor, enter it in Step 2, run the Screen check, then Save settings.")
            self._refresh()

    def save_settings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save settings", str(self.presets_dir),
                                              "ReFreM settings (*.json)")
        if path:
            path = pathlib.Path(path).with_suffix(".json")
            name = path.stem.replace("_", " ")
            store.save_settings_file(path, name, "Saved from Specvis - ReFreM.",
                                     settings_to_dict(self.settings))
            self.presets = store.list_presets(self.presets_dir)
            self.settings_name = name
            self.f_settings.setText(name)

    def build_table(self) -> None:
        summaries = sorted(self.perimetry_dir.glob("*/summary.csv"))
        table, messages = build_prior_table(summaries) if summaries else (None, [])
        if not table or not table.entries:
            QMessageBox.information(self, "Normal table",
                                    "No valid ZEST (full) pilot-mode sessions to build "
                                    "from.\n\n" + "\n".join(messages))
            return
        table.save(self.prior_table_file)
        people = max(e.n_participants for e in table.entries.values())
        warn = ("\n\nFewer than 10 people: fine for testing the method, not yet a reliable "
                "normal table." if people < 10 else "")
        QMessageBox.information(self, "Normal table", f"{table.source}.\nUp to {people} people "
                                f"per location.{warn}\n\n" + "\n".join(messages))
        self._refresh()

    # -- running -----------------------------------------------------------------------

    def job(self) -> dict:
        d = settings_to_dict(self.settings)
        d["participant_id"] = self.patient.id if self.patient else "unknown"
        scale = self.current_scale()
        return {"settings": d, "screen": self.f_screen.currentIndex(),
                "fullscreen": self.f_fullscreen.isChecked(),
                "scale": dataclasses.asdict(scale) if scale else None}

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
        self.status.setText(f"Test running on the stimulus screen: {self.patient.id}, "
                            f"{self.settings.eye} eye ...")
        self.log.clear()
        self._clock_start = time.monotonic()
        self._timer.start(1000)
        self.state_circle.setText("Running")
        self.state_circle.setStyleSheet("background: #00AAFF;")
        self.busy.setRange(0, 0)
        self._launch("test", job_path, lambda code: self._test_finished(job_path))

    def _launch(self, mode: str, job_path: pathlib.Path, on_done: Callable[[int], None]) -> None:
        """Run the stimulus process. Tests replace this with an in-process runner."""
        proc = QProcess(self)
        proc.setWorkingDirectory(str(PYTHON_DIR))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self.log.appendPlainText(
            bytes(proc.readAllStandardOutput()).decode(errors="replace").rstrip()))

        def finished(code, _status):
            self.process = None
            self._refresh()
            on_done(code)

        proc.finished.connect(finished)
        self.process = proc
        self._refresh()
        proc.start(sys.executable, [str(PYTHON_DIR / "run_test_process.py"), mode, str(job_path)])

    def _test_finished(self, job_path: pathlib.Path) -> None:
        self._timer.stop()
        self.busy.setRange(0, 1)
        self.state_circle.setText("Done")
        self.state_circle.setStyleSheet("background: #33CC33;")
        try:
            result = json.loads(job_path.with_suffix(".result.json").read_text())
        except (OSError, ValueError):
            self.status.setText("The test ended without a result (see the log).")
            return
        if "error" in result:
            self.status.setText("Could not run: " + result["error"])
            QMessageBox.warning(self, "Could not run", result["error"])
            return
        info = store.read_session(pathlib.Path(result["folder"]))
        if info is None:
            self.status.setText("Saved, but the session could not be read back.")
            return
        self.last_session = info
        self.status.setText(f"Saved automatically: {info.folder}")
        self._refresh()
        self.show_results(info)

    def show_results(self, info: store.SessionInfo) -> None:
        choice = ResultsDialog(self, info, offer_next=True).exec()
        if choice == ResultsDialog.OTHER_EYE:
            self.f_eye.setCurrentIndex(1 if info.eye == "R" else 0)
            QMessageBox.information(self, "Other eye",
                                    f"Cover the {'RIGHT' if info.eye == 'R' else 'LEFT'} eye, "
                                    "then press OK to start.")
            self.start_test()
        elif choice == ResultsDialog.REPEAT:
            self.start_test()

    def show_last_results(self) -> None:
        if self.last_session:
            self.show_results(self.last_session)

    def save_results(self) -> None:
        """Results are saved automatically; this copies the session somewhere else too."""
        if not self.last_session:
            return
        target = QFileDialog.getExistingDirectory(self, "Save results to (a copy)")
        if target:
            dest = pathlib.Path(target) / self.last_session.folder.name
            shutil.copytree(self.last_session.folder, dest, dirs_exist_ok=True)
            self.status.setText(f"Copied to {dest} (the original stays in python/output/).")

    # -- remembered between launches -----------------------------------------------------

    def _save_last(self) -> None:
        try:
            self.last_file.parent.mkdir(parents=True, exist_ok=True)
            self.last_file.write_text(json.dumps(dict(
                patient=self.patient.id if self.patient else None, scale=self.scale_id,
                screen=self.f_screen.currentIndex(), fullscreen=self.f_fullscreen.isChecked(),
                settings_name=self.settings_name, settings=settings_to_dict(self.settings)),
                indent=2))
        except OSError:
            pass

    def _load_last(self) -> None:
        try:
            last = json.loads(self.last_file.read_text())
        except (OSError, ValueError):
            self._write()
            return
        try:
            self.settings = settings_from_dict(last.get("settings", {}))
        except (TypeError, ValueError):
            pass
        self.scale_id = last.get("scale") if self.scales.get(last.get("scale") or "") else None
        if self.scale_id is None:
            self.scale_id = self._newest_measured_scale()
        self.settings_name = last.get("settings_name", "(last used)")
        self.f_settings.setText(self.settings_name)
        if 0 <= last.get("screen", 0) < self.f_screen.count():
            self.f_screen.setCurrentIndex(last.get("screen", 0))
        self.f_fullscreen.setChecked(bool(last.get("fullscreen", True)))
        pid = last.get("patient")
        if pid:
            everyone = store.all_participants(self.people, store.list_sessions(self.perimetry_dir))
            self.patient = next((p for p in everyone if p.id == pid), None)
        self._show_patient()
        self._write()

    def closeEvent(self, event) -> None:
        if self.process is not None:
            QMessageBox.information(self, "Test running", "Finish or stop the test first.")
            event.ignore()
            return
        self._save_last()
        event.accept()


ERROR_LOG = store.DATA_DIR / "app_errors.log"


def log_error(text: str) -> None:
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG.open("a") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}")
    except OSError:
        pass


def install_error_handler(app) -> None:
    """PyQt6 ends the whole app (macOS "Python quit unexpectedly") on any Python error
    inside the window code unless sys.excepthook is replaced. This keeps the app
    running, writes the error to python/output/app_errors.log and says so."""
    import traceback

    def handler(kind, value, tb):
        text = "".join(traceback.format_exception(kind, value, tb))
        log_error(text)
        sys.__stderr__.write(text)
        try:
            QMessageBox.warning(None, "Something went wrong",
                                f"{kind.__name__}: {value}\n\nThe app is still running; any "
                                f"test data already saved is safe. Details were written to\n"
                                f"{ERROR_LOG}\n\nPlease send that file.")
        except Exception:                          # noqa: BLE001 -- never raise from here
            pass

    sys.excepthook = handler


def use_light_theme(app) -> None:
    """Specvis is white with dark text. Without this, a Mac in Dark Mode turns the
    default text white on the white pages (tab names, tick-box labels vanish)."""
    from PyQt6.QtGui import QPalette

    app.setStyle("Fusion")
    pal = QPalette()
    for role, colour in ((QPalette.ColorRole.Window, "#FFFFFF"),
                         (QPalette.ColorRole.WindowText, "#202020"),
                         (QPalette.ColorRole.Base, "#FFFFFF"),
                         (QPalette.ColorRole.AlternateBase, "#F0F0F0"),
                         (QPalette.ColorRole.Text, "#202020"),
                         (QPalette.ColorRole.Button, "#F0F0F0"),
                         (QPalette.ColorRole.ButtonText, "#202020"),
                         (QPalette.ColorRole.ToolTipBase, "#FFFFFF"),
                         (QPalette.ColorRole.ToolTipText, "#202020"),
                         (QPalette.ColorRole.Highlight, "#00AAFF"),
                         (QPalette.ColorRole.HighlightedText, "#FFFFFF"),
                         (QPalette.ColorRole.PlaceholderText, "#888780")):
        pal.setColor(role, QColor(colour))
    app.setPalette(pal)


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("Specvis - ReFreM")
    use_light_theme(app)
    install_error_handler(app)
    window = SpecvisWindow()
    window.show()
    return app.exec()
