"""Common vocabulary for every threshold method.

Ported from ``org.specvis.logic.threshold`` on the ``refrem-integration`` branch.

A *location threshold* drives one point in the visual field. The procedure asks it
for the next level to present, tells it what the participant answered, and asks
whether it has finished. What "finished" means, and whether the answer is usable,
is the method's business and nobody else's.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass


class Termination(enum.Enum):
    """Why a location stopped.

    The distinction that matters scientifically is between CONVERGED and everything
    else. A run that ended at an edge of the searched range has bracketed nothing,
    and reporting the edge as a threshold reports a number that can be wrong by more
    than the effect being measured.
    """

    CONVERGED = "converged"
    CEILING_NOT_SEEN = "ceiling_not_seen"
    FLOOR_ALWAYS_SEEN = "floor_always_seen"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NOT_FINISHED = "not_finished"


@dataclass(frozen=True)
class Result:
    """What a location produced.

    ``value`` is on whatever axis the method searches -- the 0..100 brightness axis
    for the bisection, Michelson contrast for the Gabor methods. The caller converts.
    ``value`` is ``-1.0`` whenever ``valid`` is false, which is Specvis's own
    convention for "no usable threshold" and is preserved so existing analysis keeps
    working.
    """

    valid: bool
    value: float
    termination: Termination
    trials: int
    note: str = ""

    @classmethod
    def unusable(cls, termination: Termination, trials: int, note: str) -> "Result":
        return cls(False, -1.0, termination, trials, note)


class LocationThreshold(abc.ABC):
    """One point in the visual field, from first presentation to final answer."""

    @abc.abstractmethod
    def next_level(self) -> float:
        """The level to present now. Only meaningful while :meth:`finished` is false."""

    @abc.abstractmethod
    def record(self, seen: bool) -> None:
        """Record the participant's answer to the level just presented."""

    @abc.abstractmethod
    def finished(self) -> bool:
        """Whether this location needs no further presentations."""

    @abc.abstractmethod
    def result(self) -> Result:
        """The threshold, or why there isn't one."""

    @property
    @abc.abstractmethod
    def trials(self) -> int:
        """Presentations recorded so far, practice and catch trials excluded."""

    @property
    @abc.abstractmethod
    def method_name(self) -> str:
        """Written into the session metadata."""
