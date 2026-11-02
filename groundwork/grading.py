"""Dependency-free grading, also copied into isolated Harbor verifiers."""

import json
import sys
from pathlib import Path


def grade(expected, actual):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for key, value in expected.items():
        candidate = actual[key]
        if type(value) is not type(candidate):
            return False
        if isinstance(value, list):
            if any(type(x) is not str for x in candidate) or sorted(candidate) != sorted(value):
                return False
        elif candidate != value:
            return False
    return True


def main():
    expected_path, answer_path, reward_path = map(Path, sys.argv[1:])
    expected = json.loads(expected_path.read_text())
    try:
        if answer_path.stat().st_size > 65536:
            raise ValueError("Answer too large")
        actual = json.loads(answer_path.read_text())
        passed = grade(expected, actual)
    except (OSError, ValueError, TypeError):
        passed = False
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text("1" if passed else "0")
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
