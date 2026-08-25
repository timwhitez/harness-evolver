from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from bench import _canonical_harbor_grep_hardlink as harbor_grep
from bench import _canonical_harbor_hardlink as harbor_files
from harness.tools.file_edit import FileEditTool
from harness.tools.file_read import FileReadTool
from harness.tools.file_write import FileWriteTool
from harness.tools.search import GrepTool


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="hard-link and O_NOFOLLOW isolation is POSIX-specific",
)


def _hardlink_fixture(tmp_path: Path) -> tuple[Path, Path]:
    hidden = tmp_path / "terminal-bench-tasks" / "task" / "tests"
    hidden.mkdir(parents=True)
    secret = hidden / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = workspace / "notes.txt"
    os.link(secret, alias)
    assert alias.stat().st_ino == secret.stat().st_ino
    assert alias.stat().st_nlink == 2
    return secret, alias


def test_local_content_tools_reject_hardlink_aliases(tmp_path: Path) -> None:
    secret, alias = _hardlink_fixture(tmp_path)

    read = FileReadTool().execute(str(alias))
    edit = FileEditTool().execute(str(alias), "classified", "changed")
    grep = GrepTool().execute("classified", str(alias))

    for result in (read, edit, grep):
        assert result.success is False
        assert "classified" not in result.output
        assert str(secret) not in result.error
    assert read.metadata["blocked_by"] == "canonical_path_guard"
    assert edit.metadata["blocked_by"] == "canonical_path_guard"
    assert grep.metadata["blocked_by"] == "canonical_path_guard"
    assert secret.read_text(encoding="utf-8") == "classified\n"


def test_pure_overwrite_safely_dealiases_a_hardlink(tmp_path: Path) -> None:
    secret, alias = _hardlink_fixture(tmp_path)

    result = FileWriteTool().execute(str(alias), "replacement\n")

    assert result.success is True
    assert secret.read_text(encoding="utf-8") == "classified\n"
    assert alias.read_text(encoding="utf-8") == "replacement\n"
    assert alias.stat().st_ino != secret.stat().st_ino
    assert alias.stat().st_nlink == 1
    assert secret.stat().st_nlink == 1


def _run_script(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )


def test_harbor_secure_read_and_append_preread_reject_hardlink_alias(
    tmp_path: Path,
) -> None:
    secret, alias = _hardlink_fixture(tmp_path)

    read = _run_script(
        harbor_files._v2._SECURE_READ,
        {
            "HL_FILE_PATH": str(alias),
            "HL_OFFSET": "1",
            "HL_LIMIT": "10",
        },
    )
    raw = _run_script(
        harbor_files._v2._SECURE_RAW_READ,
        {"HL_FILE_PATH": str(alias)},
    )

    assert read.returncode != 0
    assert raw.returncode != 0
    assert "classified" not in read.stdout
    assert "classified" not in raw.stdout
    assert secret.read_text(encoding="utf-8") == "classified\n"


def test_harbor_grep_script_rejects_hardlink_alias(tmp_path: Path) -> None:
    secret, alias = _hardlink_fixture(tmp_path)

    completed = _run_script(
        harbor_grep._base._HARBOR_GREP_PYTHON,
        {
            "HL_ROOT": str(alias),
            "HL_PATTERN": "classified",
        },
    )

    assert completed.returncode == 74
    assert completed.stdout == ""
    assert "classified" not in completed.stderr
    assert secret.read_text(encoding="utf-8") == "classified\n"


def test_public_harbor_registry_uses_hardlink_safe_classes() -> None:
    from bench.harbor_adapter import (
        HLWorkerHarborAgent,
        HarborFileReadTool,
        HarborGrepTool,
    )

    agent = object.__new__(HLWorkerHarborAgent)
    agent.tool_timeout_seconds = 1.0
    agent._goal_path = lambda: None  # type: ignore[method-assign]
    registry = agent._build_environment_registry(object(), object())

    assert isinstance(registry.get("read"), HarborFileReadTool)
    assert isinstance(registry.get("grep"), HarborGrepTool)
