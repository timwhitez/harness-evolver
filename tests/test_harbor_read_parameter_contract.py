from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import MethodType

import pytest

from bench._canonical_harbor_paths_v4 import (
    HarborFileReadTool,
    _SECURE_READ_STRICT,
)


@pytest.mark.parametrize(
    ("offset", "limit", "message"),
    [
        (0, 1, "offset must be >= 1"),
        (-1, 1, "offset must be >= 1"),
        (1, 0, "limit must be >= 1"),
        (1, -1, "limit must be >= 1"),
        (True, 1, "offset must be an integer >= 1"),
        (1, False, "limit must be an integer >= 1"),
    ],
)
def test_invalid_read_window_fails_before_any_environment_access(
    offset: object,
    limit: object,
    message: str,
) -> None:
    tool = object.__new__(HarborFileReadTool)

    def unexpected_guard(self: HarborFileReadTool, *args: object, **kwargs: object):
        raise AssertionError("path/environment access must not occur for invalid parameters")

    tool._guard_environment_path = MethodType(unexpected_guard, tool)  # type: ignore[method-assign]

    result = tool.execute("/app/file.txt", offset=offset, limit=limit)  # type: ignore[arg-type]

    assert result.success is False
    assert result.output == ""
    assert message in result.error
    assert result.metadata["parameter_validation_failed"] is True


def test_none_limit_keeps_the_documented_default() -> None:
    tool = object.__new__(HarborFileReadTool)
    captured: dict[str, object] = {}

    def fake_guard(
        self: HarborFileReadTool,
        path: str,
        *,
        operation: str,
        must_exist: bool,
    ):
        return path, None

    def fake_run(
        self: HarborFileReadTool,
        script: str,
        *,
        env: dict[str, str],
    ):
        captured["script"] = script
        captured["env"] = env
        from harness.tools.base import ToolResult

        return ToolResult(success=True, output="1\tdata\n", metadata={})

    tool._guard_environment_path = MethodType(fake_guard, tool)  # type: ignore[method-assign]
    tool._run_secure_python = MethodType(fake_run, tool)  # type: ignore[method-assign]

    result = tool.execute("/app/file.txt", offset=1, limit=None)

    assert result.success is True
    assert captured["script"] == _SECURE_READ_STRICT
    assert captured["env"] == {
        "HL_FILE_PATH": "/app/file.txt",
        "HL_OFFSET": "1",
        "HL_LIMIT": "2000",
    }


def test_container_script_rejects_invalid_window_independently(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("data\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-c", _SECURE_READ_STRICT],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HL_FILE_PATH": str(target),
            "HL_OFFSET": "0",
            "HL_LIMIT": "1",
        },
        check=False,
    )

    assert completed.returncode != 0
    assert "offset must be >= 1" in completed.stderr
