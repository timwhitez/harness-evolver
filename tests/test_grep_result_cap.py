from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

import harness.tools.search as search_module
from harness.tools.search import GrepTool


def test_schema_publishes_configured_result_window() -> None:
    tool = GrepTool(max_results=3)

    definition = tool.get_schema().parameters["properties"]["max_results"]

    assert definition["minimum"] == 1
    assert definition["maximum"] == 3


def test_over_cap_request_fails_before_path_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = GrepTool(max_results=3)

    def unexpected_path_access(*args: object, **kwargs: object) -> object:
        raise AssertionError("path authorization must not run")

    monkeypatch.setattr(
        search_module,
        "resolve_guarded_path",
        unexpected_path_access,
    )

    result = tool.execute("needle", path="missing", max_results=4)

    assert result.success is False
    assert result.metadata["parameter_validation_failed"] is True
    assert result.metadata["configured_max_results"] == 3
    assert "cannot exceed" in result.error


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2"])
def test_invalid_call_level_result_windows_fail_before_traversal(
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = GrepTool(max_results=3)

    def unexpected_path_access(*args: object, **kwargs: object) -> object:
        raise AssertionError("path authorization must not run")

    monkeypatch.setattr(
        search_module,
        "resolve_guarded_path",
        unexpected_path_access,
    )

    result = tool.execute("needle", path="missing", max_results=value)  # type: ignore[arg-type]

    assert result.success is False
    assert result.metadata["parameter_validation_failed"] is True


def test_invalid_configured_cap_fails_before_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = GrepTool(max_results=0)

    def unexpected_path_access(*args: object, **kwargs: object) -> object:
        raise AssertionError("path authorization must not run")

    monkeypatch.setattr(
        search_module,
        "resolve_guarded_path",
        unexpected_path_access,
    )

    result = tool.execute("needle", path="missing")

    assert result.success is False
    assert result.metadata["parameter_validation_failed"] is True
    assert "configured max_results" in result.error


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="stable no-follow grep traversal requires POSIX",
)
def test_caller_can_lower_but_not_raise_configured_window(tmp_path: Path) -> None:
    target = tmp_path / "matches.txt"
    target.write_text("needle\n" * 5, encoding="utf-8")
    tool = GrepTool(max_results=4)

    lowered = tool.execute("needle", path=str(target), max_results=2)
    defaulted = tool.execute("needle", path=str(target))

    assert lowered.success is True
    assert lowered.metadata["match_count"] == 5
    assert lowered.metadata["returned_count"] == 2
    assert lowered.metadata["omitted_count"] == 3
    assert defaulted.success is True
    assert defaulted.metadata["match_count"] == 5
    assert defaulted.metadata["returned_count"] == 4
    assert defaulted.metadata["omitted_count"] == 1


def test_schema_cap_survives_module_reload() -> None:
    reloaded = importlib.reload(search_module)
    tool = reloaded.GrepTool(max_results=7)

    definition = tool.get_schema().parameters["properties"]["max_results"]

    assert definition["minimum"] == 1
    assert definition["maximum"] == 7
