"""One row per presentation, for every stimulus shape.

Ported from ``org.specvis.logic.TrialLog``. The file it writes, ``trials.csv``, is
**the same file** the Java writes -- same comment header, same fifteen columns, same
number formatting -- so a session replayed through the Python can be compared with
the Java's byte for byte. ``tests/test_java_parity.py`` checks that against the Java.

Why it exists: ``session_data.txt`` holds one row per *location*, the final threshold
and nothing about how it was reached. With three or four presentations per location
and no reversal criterion, a single late or missed press moves the result by a whole
rung, and the session file cannot show that it happened. This log can.

``reaction_time_ms`` is measured from the event that showed the stimulus to the key
event. It includes toolkit and keyboard latency and has not been checked against a
photodiode: a coarse behavioural measure, not a reaction time.

Specvis accepts a response until the end of the inter-stimulus interval, not just
while the stimulus is on screen, so a reaction time well past ``display_time_ms`` is a
late press that still counted as "seen".

What the Java keeps elsewhere is kept elsewhere here too: requested and rendered Gabor
contrast are in :mod:`specvis.data.gabor_log`, and per-trial reversal timing belongs
to the behavioural trial log, which is not ported yet.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import List

from .javafmt import fmt, format_f

PRACTICE = "practice"
MAIN = "main"
#: A blank presentation: nothing is shown and no threshold is touched. A press on one
#: of these is a false alarm, and these rows are the only way that rate can be measured.
CATCH = "catch"

HEADER = (
    "# One row per presentation. block=practice rows are discarded by the procedure and "
    "affect no threshold.\n"
    "# reaction_time_ms runs from the scene-graph event that added the stimulus to the "
    "key event. It includes\n"
    "# toolkit and keyboard latency, is not photodiode-validated, and must not be "
    "reported as a reaction time.\n"
    "# Responses are accepted for window_ms, not display_time_ms, so a press later than "
    "display_time_ms still\n"
    "# counted as seen. extra_presses counts further presses in the same window - this "
    "is what Specvis reports as\n"
    "# \"false-positive answers\"; it is NOT a false-alarm rate, because the procedure "
    "runs no blank catch trials.\n"
    "# michelson_contrast is NA for solid stimuli. All luminance and contrast values are "
    "nominal unless the\n"
    "# luminance scale was measured with a photometer, which Specvis does not record.\n"
)

COLUMNS = (
    "trial", "block", "ms_from_start", "stimulus_index", "x_deg", "y_deg",
    "brightness_value", "brightness_index", "stimulus_luminance_cdm2",
    "michelson_contrast", "seen", "reaction_time_ms", "extra_presses",
    "display_time_ms", "window_ms",
)


@dataclass(frozen=True)
class Entry:
    """One presentation. ``michelson_contrast`` is NaN for a solid stimulus."""

    trial: int
    block: str
    ms_from_start: int
    stimulus_index: int
    x_deg: float
    y_deg: float
    brightness_value: int
    brightness_index: int
    stimulus_luminance: float
    michelson_contrast: float
    seen: bool
    reaction_time_ms: float
    extra_presses: int
    display_time_ms: int
    window_ms: int

    def row(self) -> str:
        return ",".join((
            str(self.trial), self.block, str(self.ms_from_start), str(self.stimulus_index),
            fmt(self.x_deg), fmt(self.y_deg), str(self.brightness_value),
            str(self.brightness_index), fmt(self.stimulus_luminance),
            fmt(self.michelson_contrast), "TRUE" if self.seen else "FALSE",
            fmt(self.reaction_time_ms), str(self.extra_presses), str(self.display_time_ms),
            str(self.window_ms),
        ))


class TrialLog:
    """The log for one session. One instance per session rather than the Java's static."""

    def __init__(self) -> None:
        self._entries: List[Entry] = []
        self._breaks_taken = 0

    def clear(self) -> None:
        """Discard the previous session. Called when a procedure starts."""
        self._entries.clear()
        self._breaks_taken = 0

    def record(self, block: str, ms_from_start: int, stimulus_index: int, x_deg: float,
               y_deg: float, brightness_value: int, brightness_index: int,
               stimulus_luminance: float, michelson_contrast: float, seen: bool,
               reaction_time_ms: float, extra_presses: int, display_time_ms: int,
               window_ms: int) -> None:
        self._entries.append(Entry(
            len(self._entries) + 1, block, int(ms_from_start), stimulus_index, x_deg, y_deg,
            int(brightness_value), brightness_index, stimulus_luminance, michelson_contrast,
            seen, reaction_time_ms, extra_presses, int(display_time_ms), int(window_ms),
        ))

    def record_break(self) -> None:
        self._breaks_taken += 1

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[Entry]:
        return list(self._entries)

    def count_block(self, block: str) -> int:
        return sum(1 for e in self._entries if e.block == block)

    def text(self) -> str:
        """The whole of ``trials.csv``."""
        lines = [",".join(COLUMNS)] + [e.row() for e in self._entries]
        return HEADER + "\n".join(lines) + "\n"

    def write_to(self, directory: pathlib.Path) -> None:
        """Write ``trials.csv`` beside the other session files. Does nothing when empty."""
        if not self._entries:
            return
        path = pathlib.Path(directory) / "trials.csv"
        with path.open("w", newline="") as handle:
            handle.write(self.text())

    def summarise(self) -> str:
        """The trial-log paragraph of the session metadata, worded as the Java words it."""
        if not self._entries:
            return "Trial log: no presentations recorded\n"

        seen = late = extra = rt_n = 0
        rt_sum = 0.0
        for e in self._entries:
            if e.block != MAIN:
                continue
            if e.seen:
                seen += 1
                if not math.isnan(e.reaction_time_ms):
                    rt_sum += e.reaction_time_ms
                    rt_n += 1
                    if e.reaction_time_ms > e.display_time_ms:
                        late += 1
            extra += e.extra_presses

        catch_trials = self.count_block(CATCH)
        catch_presses = sum(1 for e in self._entries if e.block == CATCH and e.seen)
        main = self.count_block(MAIN)

        out = [
            f"Trial log: {len(self._entries)} presentations in trials.csv "
            f"({self.count_block(PRACTICE)} practice, {main} scored, {catch_trials} blank catch)\n",
            f"  responded 'seen' to {seen} of {main} scored presentations "
            f"({format_f(0.0 if main == 0 else 100.0 * seen / main, 1)}%)\n",
        ]
        if rt_n > 0:
            out.append(f"  mean response latency {format_f(rt_sum / rt_n, 0)} ms; {late} of "
                       f"{rt_n} presses arrived after the stimulus had gone\n")
            out.append("    (expected for a brief flash, since the window stays open for "
                       "the whole trial)\n")
        out.append(f"  extra presses inside a response window: {extra}\n")
        if catch_trials > 0:
            out.append(f"  FALSE ALARMS: {catch_presses} presses on {catch_trials} blank trials "
                       f"({format_f(100.0 * catch_presses / catch_trials, 1)}%)\n")
            out.append("    This is a measured false-alarm rate. It is a different quantity "
                       "from\n    the extra-press count above, which is what Specvis calls "
                       "\"false-positive answers\".\n")
        out.append(f"  rests taken: {self._breaks_taken}\n")
        out.append("  NOTE: latencies include toolkit and keyboard delay and are not "
                   "photodiode-validated.\n")
        if catch_trials == 0:
            out.append("  NOTE: no blank catch trials were run, so the false-alarm rate is "
                       "unknown.\n")
        return "".join(out)
