"""A simulated observer for perimetry tests: it sees only what the backend is asked
to draw (the patch's light and where it is), never the procedure's state."""

from __future__ import annotations

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from specvis.core.gabor import GaborParameters, luminance_field  # noqa: E402
from specvis.core.geometry import PhysicalScreen  # noqa: E402
from specvis.core.luminance import LuminanceCalibration  # noqa: E402
from specvis.procedure.perimetry import FULL, PerimetrySettings, run_perimetry  # noqa: E402
from specvis.stimulus.backend import NullBackend  # noqa: E402

W, H = 3024, 1964                      # 14-inch MacBook Pro, full screen
WIDTH_MM, DIST_MM = 302.0, 350.0

# The display the procedure assumes without a photometer (10 cd/m^2 background on a
# gamma-2.2, 300 cd/m^2 screen). The observer turns brightness back into light with it.
DISPLAY = LuminanceCalibration.assumed_display(300.0, 2.2)
_PPD = PhysicalScreen(W, H, 302.0, 302.0 * 1964 / 3024, 350.0
                      ).local_px_per_degree_x(0.0)
_TEMPLATE = luminance_field(GaborParameters(2.0, 4.0, 45.0, 0.0, 2.0 / 6.0, 1.0, 1.0),
                            _PPD, _PPD) - 1.0


def seen_contrast(brightness) -> float:
    """The patch's contrast as an ideal eye reads it: the light, projected onto the
    Gabor's own shape -- so dithering noise averages out, as it does for a real eye."""
    lum = DISPLAY.luminance_at(np.asarray(brightness) * 100.0)
    mean = float(lum[0, 0])
    if mean <= 0 or lum.shape != _TEMPLATE.shape:
        return float((lum.max() - lum.min()) / max(lum.max() + lum.min(), 1e-9))
    return abs(float(((lum / mean - 1.0) * _TEMPLATE).sum() / (_TEMPLATE ** 2).sum()))

def healthy(x: float, y: float) -> float:
    return math.log10(0.02 + 0.004 * math.hypot(x, y))


def right_hemianopia(x: float, y: float):
    return None if x > 0 else healthy(x, y)          # right half blind


class Observer(NullBackend):
    """Sees a patch by a Weibull on its rendered contrast at its field position."""

    def __init__(self, threshold, rng, false_alarm=0.02, lapse=0.02,
                 sees_blind_spot=False, abort_after=None, eye="R", press_ms=350.0,
                 early_blank_presses=False):
        super().__init__(refresh_hz=120.0)
        self.press_ms, self.early_blank_presses = press_ms, early_blank_presses
        self.threshold, self.rng = threshold, rng
        self.false_alarm, self.lapse = false_alarm, lapse
        self.sees_blind_spot, self.abort_after, self.eye = sees_blind_spot, abort_after, eye
        self.shown = []

    def field_deg(self, px, py):
        mm_x, mm_y = (px - W / 2) * WIDTH_MM / W, (H / 2 - py) * WIDTH_MM / W
        return (math.degrees(math.atan(mm_x / DIST_MM)), math.degrees(math.atan(mm_y / DIST_MM)))

    def present_train(self, polarities, frames_per_reversal, total_frames,
                      fixation=None, response_keys=()):
        report = super().present_train(polarities, frames_per_reversal, total_frames,
                                       fixation, response_keys)
        c = seen_contrast(polarities[0].brightness)
        x, y = self.field_deg(*polarities[0].centre_px)
        self.shown.append((round(x, 1), round(y, 1), c))
        if self.abort_after is not None and len(self.shown) > self.abort_after:
            report.keys = [("escape", 10.0)]
            return report
        temporal = x if self.eye == "R" else -x
        in_blind_spot = 13.0 <= temporal <= 18.0 and -4.0 <= y <= 1.0
        if c < 1e-6:
            if self.early_blank_presses:
                report.keys = [("space", 90.0)]      # pressing before anything could be seen
                return report
            p = self.false_alarm
        elif in_blind_spot and not self.sees_blind_spot:
            p = self.false_alarm
        else:
            t = self.threshold(x, y)
            p = self.false_alarm if t is None else self.false_alarm + (
                1 - self.false_alarm - self.lapse) * (1 - math.exp(-10 ** (3.5 * (math.log10(c) - t))))
        if self.rng.random() < p:
            report.keys = [("space", self.press_ms)]
        return report

    def wait_for_key(self, keys, timeout_ms, fixation=None, clear_events=True):
        return (None, None)


def run(threshold, seed, eye="R", settings_overrides=None, **kw):
    settings = PerimetrySettings(participant_id=kw.pop("pid", f"SIM{seed}"), eye=eye,
                                 screen_width_mm=WIDTH_MM, viewing_distance_mm=DIST_MM,
                                 pilot_mode=kw.pop("pilot_mode", True),
                                 extent=kw.pop("extent", FULL),
                                 **(settings_overrides or {}))
    table = kw.pop("table", None)
    obs = Observer(threshold, random.Random(seed), eye=eye, **kw)
    obs.open(W, H, 0.5)
    out = run_perimetry(settings, obs, prior_table=table, rng=random.Random(seed),
                        show_instructions=False)
    return out, obs
