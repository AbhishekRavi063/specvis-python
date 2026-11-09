"""Specvis's own threshold algorithm.

A direct port of ``org.specvis.logic.BisectionStaircase``, which was itself extracted
verbatim from ``getBrightnessForNextStimulusPresentation`` and
``decideAboutStimulusDeactivation`` in the four ``ProcedureBasicFixMonitor*`` classes.

The port is deliberately literal -- the same index arithmetic in the same order, not a
tidier equivalent -- because its only justification is that it behaves identically to
the Java. ``tests/test_bisection_parity.py`` enumerates every reachable answer sequence
at several vector lengths and compares against output captured from the Java classes.

What the algorithm does, in one sentence: **the reported threshold is the dimmest level
at which the participant said "seen"**. Everything below is an efficient search for that
level in three or four presentations. See
``documentation/ANALYSIS-specvis-threshold-algorithm.md`` for the full analysis,
including the criterion drift and the floor blindness that this method carries.
"""

from __future__ import annotations

from typing import List, Sequence

from .base import LocationThreshold, Result, Termination

#: Answer codes, as stored in the per-location array. These are Specvis's own values.
UNANSWERED = 0
NOT_SEEN = 1
SEEN = 2


def max_answers(vector_len: int) -> int:
    """Presentations a location can receive: upstream's ``(len + 7) / 4``, integer."""
    return (vector_len + 7) // 4


def next_brightness_index(answers: Sequence[int], vector_len: int) -> int:
    """Index into the brightness vector for the next presentation.

    Starts at the middle and replays the whole history. Steps 2 while the direction is
    unchanged and 1 after a reversal, so on a 2.5 dB ladder it is a 5.0-then-2.5 dB
    bracketing search.
    """
    index = vector_len // 2
    latest = -1

    for answer in answers:
        if answer == UNANSWERED:
            break
        if answer == NOT_SEEN:
            index += 1 if latest == SEEN else 2
        elif answer == SEEN:
            index -= 1 if latest == NOT_SEEN else 2
        latest = answer

    return index


def _checked(candidates: List[bool], j: int, vector_len: int) -> bool:
    """Assign False at ``j``, raising when ``j`` is outside the vector.

    Python would happily accept a negative index and write to the far end of the
    list; Java throws. Unreachable in the real procedure either way, but the port is
    only worth anything if it fails where the original fails.
    """
    if j < 0 or j >= vector_len:
        raise IndexError(f"brightness index {j} outside 0..{vector_len - 1}")
    return False


def remaining_levels(answers: Sequence[int], vector_len: int) -> List[bool]:
    """Which levels are still candidates after these answers.

    The two elimination rules are **not** symmetric, and this is the most consequential
    detail in the algorithm:

    * "not seen at level k" eliminates ``0..k`` **inclusive** -- k itself is removed;
    * "seen at level k" eliminates ``k+1..end`` -- k itself **survives**.

    One rung of asymmetry, which is why a path and its mirror image resolve one rung
    apart.
    """
    candidates = [True] * vector_len
    index = vector_len // 2
    latest = -1

    for answer in answers:
        if answer == UNANSWERED:
            break

        if answer == NOT_SEEN:
            # Indexed exactly as the Java does, so an answer array the real procedure
            # could never produce raises here instead of being quietly absorbed.
            # Java throws ArrayIndexOutOfBoundsException on the same inputs.
            for j in range(0, index + 1):
                candidates[j] = _checked(candidates, j, vector_len)
            index += 1 if latest == SEEN else 2
        elif answer == SEEN:
            if index != vector_len - 1:
                for j in range(index + 1, vector_len):
                    candidates[j] = _checked(candidates, j, vector_len)
            index -= 1 if latest == NOT_SEEN else 2

        latest = answer

    return candidates


def surviving_index(answers: Sequence[int], vector_len: int) -> int:
    """The threshold index once finished, or -1 when no candidate survives."""
    remaining = remaining_levels(answers, vector_len)
    index = -1
    for i, alive in enumerate(remaining):
        if alive:
            index = i
    return index if sum(remaining) <= 1 else -1


def is_complete(answers: Sequence[int], vector_len: int) -> bool:
    """True when one candidate or none remains, which is when upstream stops."""
    return sum(remaining_levels(answers, vector_len)) <= 1


class BisectionLocationThreshold(LocationThreshold):
    """Specvis's bisection, wrapped in the common interface.

    .. warning::
       This method **cannot detect a floor effect.** A participant who sees the
       dimmest rung is reported at that rung as a converged, valid threshold, with no
       failure signal -- an observer 18 dB more sensitive than the ladder is reported
       at the ladder floor. The ceiling is handled differently and does return
       "no threshold". That inconsistency is upstream behaviour, preserved here
       deliberately; ``FourTwoStaircase`` detects both ends.
    """

    # Written into the session metadata, so it is the Java's string exactly.
    NAME = "Specvis bisection"

    def __init__(self, brightness_vector: Sequence[float]):
        if len(brightness_vector) < 2:
            raise ValueError("a brightness vector needs at least two rungs")
        self._vector = list(brightness_vector)
        self._answers = [UNANSWERED] * max_answers(len(self._vector))
        self._recorded = 0

    # -- driving ---------------------------------------------------------------

    def next_level(self) -> float:
        index = next_brightness_index(self._answers, len(self._vector))
        # Upstream indexes the vector directly and would throw here; the procedure
        # never reaches it because the candidate set empties first. Clamping instead
        # keeps a simulation from dying on a path the real program cannot take.
        index = max(0, min(index, len(self._vector) - 1))
        return float(self._vector[index])

    def record(self, seen: bool) -> None:
        if self._recorded >= len(self._answers):
            raise RuntimeError("this location has already had its full budget")
        self._answers[self._recorded] = SEEN if seen else NOT_SEEN
        self._recorded += 1

    def finished(self) -> bool:
        return is_complete(self._answers, len(self._vector))

    # -- reporting -------------------------------------------------------------

    def result(self) -> Result:
        surviving = surviving_index(self._answers, len(self._vector))
        if surviving < 0:
            return Result.unusable(
                Termination.CEILING_NOT_SEEN,
                self._recorded,
                "no candidate level survived; Specvis writes -1",
            )
        return Result(
            True,
            float(self._vector[surviving]),
            Termination.CONVERGED,
            self._recorded,
            f"bracket closed on vector index {surviving}",
        )

    @property
    def trials(self) -> int:
        return self._recorded

    @property
    def method_name(self) -> str:
        return self.NAME

    @property
    def answers(self) -> List[int]:
        """The raw answer array, for the trial log and for parity testing."""
        return list(self._answers)
