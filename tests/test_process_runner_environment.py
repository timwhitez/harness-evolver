from __future__ import annotations

import os
import sys

import pytest

from harness.tools.process_runner import run_bounded_argv


@pytest.mark.skipif(os.name == "nt", reason="minimal replacement environment fixture is POSIX-specific")
def test_explicit_environment_replaces_parent_instead_of_overlaying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HL_PARENT_ONLY_SECRET", "must-not-leak")
    program = (
        "import os; "
        "print(os.environ.get('HL_EXPLICIT_ONLY', '<missing>')); "
        "print(os.environ.get('HL_PARENT_ONLY_SECRET', '<missing>'))"
    )

    result = run_bounded_argv(
        [sys.executable, "-c", program],
        timeout_seconds=5.0,
        env={"HL_EXPLICIT_ONLY": "present"},
    )

    assert result.timed_out is False
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["present", "<missing>"]


def test_none_environment_still_inherits_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HL_PARENT_INHERITED", "present")
    program = "import os; print(os.environ.get('HL_PARENT_INHERITED', '<missing>'))"

    result = run_bounded_argv(
        [sys.executable, "-c", program],
        timeout_seconds=5.0,
        env=None,
    )

    assert result.timed_out is False
    assert result.returncode == 0
    assert result.stdout.strip() == "present"
