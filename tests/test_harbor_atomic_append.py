from __future__ import annotations

import base64
import os
from pathlib import Path
import subprocess
import sys

import pytest

from bench._canonical_harbor_write_v3 import (
    HarborFileWriteTool,
    _SECURE_ATOMIC_WRITE,
)
from harness.tools.base import ToolResult


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="O_NOFOLLOW and hard-link regression tests are POSIX-specific",
)


def test_public_append_builds_one_atomic_replacement_payload() -> None:
    tool = object.__new__(HarborFileWriteTool)
    captured: dict[str, object] = {}

    tool._guard_environment_path = (  # type: ignore[method-assign]
        lambda path, *, operation, must_exist: (path, None)
    )
    tool._secure_raw_read = lambda path: ("existing\n", None)  # type: ignore[method-assign]

    def capture(script: str, *, env: dict[str, str]) -> ToolResult:
        captured["script"] = script
        captured["env"] = env
        return ToolResult(success=True, output="write complete", metadata={})

    tool._run_secure_python = capture  # type: ignore[method-assign]

    result = tool.execute("/tmp/example.txt", "appended\n", append=True)

    assert result.success is True
    assert captured["script"] == _SECURE_ATOMIC_WRITE
    env = captured["env"]
    assert isinstance(env, dict)
    assert base64.b64decode(env["HL_FILE_CONTENT"]) == b"existing\nappended\n"
    assert "HL_APPEND" not in env
    assert result.metadata["atomic_append"] is True
    assert result.metadata["atomic_replace"] is True


def test_atomic_append_replaces_a_hard_link_without_mutating_its_source(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    target = tmp_path / "target.txt"
    os.link(secret, target)

    effective_payload = target.read_text(encoding="utf-8") + "appended\n"
    completed = subprocess.run(
        [sys.executable, "-c", _SECURE_ATOMIC_WRITE],
        env={
            **os.environ,
            "HL_FILE_PATH": str(target),
            "HL_FILE_CONTENT": base64.b64encode(
                effective_payload.encode("utf-8")
            ).decode("ascii"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert secret.read_text(encoding="utf-8") == "classified\n"
    assert target.read_text(encoding="utf-8") == "classified\nappended\n"
    assert secret.stat().st_ino != target.stat().st_ino
