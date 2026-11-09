#!/usr/bin/env python3
"""Run every check in tests/. No pytest dependency: these run anywhere Python does."""

from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    tests = sorted(HERE.joinpath("tests").glob("test_*.py"))
    if not tests:
        print("no tests found")
        return 1

    passed, failed = 0, []
    for test in tests:
        result = subprocess.run([sys.executable, str(test)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"PASS  {test.stem}")
            passed += 1
        else:
            print(f"FAIL  {test.stem}")
            for line in (result.stdout + result.stderr).strip().splitlines()[-15:]:
                print("      " + line)
            failed.append(test.stem)

    print()
    print(f"{passed} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
