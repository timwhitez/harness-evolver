"""Thin console-script wrappers with packaged runtime defaults."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

from bench.runtime_resources import bundled_config_path


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(argument == option or argument.startswith(option + "=") for argument in argv)


def _trial_argv_with_runtime_defaults(argv: Sequence[str]) -> list[str]:
    """Add packaged model/trial config only when no explicit or local default exists."""

    updated = list(argv)
    if any(argument in {"-h", "--help"} for argument in updated[1:]):
        return updated

    if not _has_option(updated, "--trials-config") and not Path(
        "config/trials.yaml"
    ).is_file():
        updated.extend(["--trials-config", str(bundled_config_path("trials.yaml"))])

    local_models = (Path("config/local.yaml"), Path("config/models.yaml"))
    if not _has_option(updated, "--models-config") and not any(
        candidate.is_file() for candidate in local_models
    ):
        updated.extend(["--models-config", str(bundled_config_path("models.yaml"))])
    return updated


def trial() -> int:
    sys.argv = _trial_argv_with_runtime_defaults(sys.argv)
    from scripts.run_trial import main

    return main()


def regression() -> int:
    # Regression resolves the same Worker model/trial configuration as a single
    # trial and must therefore receive the installed defaults outside a checkout.
    sys.argv = _trial_argv_with_runtime_defaults(sys.argv)
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
