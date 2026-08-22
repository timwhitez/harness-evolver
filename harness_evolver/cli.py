"""Thin console-script wrappers around repository scripts."""

from __future__ import annotations

import sys


def trial() -> int:
    from scripts.run_trial import main

    return main()


def regression() -> int:
    from scripts.regression_check import main

    return main()


def compare() -> int:
    from scripts.compare_trials import main

    return main()


def mission_debug() -> int:
    from scripts.mission_debug import main

    return main()


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "usage: python -m harness_evolver.cli "
            "{trial,regression,compare,mission-debug} [args...]"
        )
        return 0 if len(sys.argv) >= 2 else 2

    command = sys.argv[1]
    commands = {
        "trial": trial,
        "regression": regression,
        "compare": compare,
        "mission-debug": mission_debug,
    }
    if command not in commands:
        print(f"unknown command: {command}", file=sys.stderr)
        return 2

    sys.argv = [f"harness-evolver-{command}", *sys.argv[2:]]
    return commands[command]()


if __name__ == "__main__":
    raise SystemExit(main())
