"""The Python bisection must behave identically to the Java it was ported from.

``java_bisection_reference.csv`` was produced by driving
``org.specvis.logic.BisectionStaircase`` over every answer sequence the procedure can
actually reach, at six vector lengths. It records, at every step of every path, the
next index, the surviving index, whether the location is complete, and the full
candidate mask.

If this test fails, the port is wrong -- not the fixture. Regenerate the fixture only
when the Java itself has deliberately changed.
"""

from __future__ import annotations

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from specvis.threshold.bisection import (  # noqa: E402
    is_complete,
    max_answers,
    next_brightness_index,
    remaining_levels,
    surviving_index,
)

REFERENCE = pathlib.Path(__file__).with_name("java_bisection_reference.csv")


def load():
    with REFERENCE.open() as handle:
        for row in csv.DictReader(handle):
            yield (
                int(row["vector_len"]),
                [int(v) for v in row["answers"].split()],
                int(row["next_index"]),
                int(row["surviving"]),
                row["complete"] == "true",
                [c == "1" for c in row["remaining"]],
            )


def main() -> int:
    rows = list(load())
    if not rows:
        print("FAIL  reference fixture is empty")
        return 1

    failures = []
    lengths = set()

    for vector_len, answers, next_i, surviving, complete, remaining in rows:
        lengths.add(vector_len)

        if len(answers) != max_answers(vector_len):
            failures.append(
                f"len={vector_len}: fixture has {len(answers)} answer slots, "
                f"max_answers says {max_answers(vector_len)}"
            )
            continue

        checks = (
            ("next_brightness_index", next_brightness_index(answers, vector_len), next_i),
            ("surviving_index", surviving_index(answers, vector_len), surviving),
            ("is_complete", is_complete(answers, vector_len), complete),
            ("remaining_levels", remaining_levels(answers, vector_len), remaining),
        )
        for name, got, want in checks:
            if got != want:
                failures.append(
                    f"len={vector_len} answers={answers} {name}: "
                    f"python={got!r} java={want!r}"
                )

    print(f"  compared {len(rows)} reachable states across vector lengths "
          f"{sorted(lengths)}")

    if failures:
        print(f"\nFAILED  {len(failures)} mismatch(es)")
        for line in failures[:20]:
            print("   " + line)
        return 1

    print("  ok   every reachable state matches the Java exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
