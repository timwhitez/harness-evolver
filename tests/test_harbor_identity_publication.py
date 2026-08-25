from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import MethodType

import pytest

from bench import _canonical_harbor_identity_guard as identity_guard
from bench.harbor_adapter import HarborFileEditTool, HarborFileWriteTool
from harness.tools.base import ToolResult


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not hasattr(os, "O_NOFOLLOW"),
    reason="race-safe Harbor conditional publication requires Linux renameat2",
)


def _run_script(
    script: str,
    *,
    env: dict[str, str],
    wrapper: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    program = script
    if wrapper:
        program = "\n".join([*wrapper, f"exec({script!r})"])
    return subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _capturing_append_tool(
    snapshot: identity_guard._Snapshot,
    captured: dict[str, object],
) -> HarborFileWriteTool:
    tool = object.__new__(HarborFileWriteTool)
    tool._guard_environment_path = MethodType(  # type: ignore[method-assign]
        lambda self, path, *, operation, must_exist: (path, None),
        tool,
    )
    tool._secure_snapshot = MethodType(  # type: ignore[method-assign]
        lambda self, path: (snapshot, None),
        tool,
    )

    def capture(
        self: HarborFileWriteTool,
        script: str,
        *,
        env: dict[str, str],
    ) -> ToolResult:
        captured["script"] = script
        captured["env"] = env
        return ToolResult(
            success=True,
            output="write complete\ndirectory_synced=1\n",
            metadata={"exit_code": 0},
        )

    tool._run_secure_python = MethodType(capture, tool)  # type: ignore[method-assign]
    return tool


def test_public_append_forwards_existing_snapshot_identity_and_digest() -> None:
    captured: dict[str, object] = {}
    snapshot = identity_guard._Snapshot(
        present=True,
        text="existing\n",
        dev=12,
        ino=34,
        sha256=hashlib.sha256(b"existing\n").hexdigest(),
    )
    tool = _capturing_append_tool(snapshot, captured)

    result = tool.execute("/workspace/target.txt", "tail\n", append=True)

    assert result.success is True
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["HL_EXPECTED_PRESENT"] == "1"
    assert environment["HL_EXPECTED_DEV"] == "12"
    assert environment["HL_EXPECTED_INO"] == "34"
    assert environment["HL_EXPECTED_SHA256"] == snapshot.sha256
    assert base64.b64decode(environment["HL_FILE_CONTENT"]) == b"existing\ntail\n"
    assert result.metadata["target_identity_verified"] is True
    assert result.metadata["directory_fsync"] is True
    assert result.metadata["atomic_append"] is True


def test_public_append_keeps_missing_snapshot_expectation_disjoint() -> None:
    captured: dict[str, object] = {}
    snapshot = identity_guard._Snapshot(
        present=False,
        text="",
        dev=None,
        ino=None,
        sha256=hashlib.sha256(b"").hexdigest(),
    )
    tool = _capturing_append_tool(snapshot, captured)

    result = tool.execute("/workspace/target.txt", "tail\n", append=True)

    assert result.success is True
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert environment["HL_EXPECTED_PRESENT"] == "0"
    assert "HL_EXPECTED_DEV" not in environment
    assert "HL_EXPECTED_INO" not in environment
    assert "HL_EXPECTED_SHA256" not in environment
    assert base64.b64decode(environment["HL_FILE_CONTENT"]) == b"tail\n"


def test_append_script_rejects_replaced_inode(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    old_metadata = target.stat()
    old_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    moved = tmp_path / "old.txt"
    target.replace(moved)
    target.write_text("concurrent\n", encoding="utf-8")

    completed = _run_script(
        identity_guard._v3._SECURE_ATOMIC_WRITE,
        env={
            "HL_FILE_PATH": str(target),
            "HL_FILE_CONTENT": base64.b64encode(b"old\ntail\n").decode("ascii"),
            "HL_EXPECTED_PRESENT": "1",
            "HL_EXPECTED_DEV": str(old_metadata.st_dev),
            "HL_EXPECTED_INO": str(old_metadata.st_ino),
            "HL_EXPECTED_SHA256": old_digest,
        },
    )

    assert completed.returncode != 0
    assert target.read_text(encoding="utf-8") == "concurrent\n"
    assert moved.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".target.txt.tmp-*")) == []


def test_append_script_rejects_target_appearing_after_missing_snapshot(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("concurrent\n", encoding="utf-8")

    completed = _run_script(
        identity_guard._v3._SECURE_ATOMIC_WRITE,
        env={
            "HL_FILE_PATH": str(target),
            "HL_FILE_CONTENT": base64.b64encode(b"tail\n").decode("ascii"),
            "HL_EXPECTED_PRESENT": "0",
        },
    )

    assert completed.returncode != 0
    assert target.read_text(encoding="utf-8") == "concurrent\n"
    assert list(tmp_path.glob(".target.txt.tmp-*")) == []


def test_append_script_creates_target_when_missing_snapshot_remains_missing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"

    completed = _run_script(
        identity_guard._v3._SECURE_ATOMIC_WRITE,
        env={
            "HL_FILE_PATH": str(target),
            "HL_FILE_CONTENT": base64.b64encode(b"tail\n").decode("ascii"),
            "HL_EXPECTED_PRESENT": "0",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert target.read_bytes() == b"tail\n"
    assert "directory_synced=1" in completed.stdout


def test_harbor_directory_fsync_failure_is_success_with_warning(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    wrapper = [
        "import os, stat",
        "real_fsync = os.fsync",
        "def guarded_fsync(fd):",
        "    if stat.S_ISDIR(os.fstat(fd).st_mode):",
        "        raise OSError('injected directory fsync failure')",
        "    return real_fsync(fd)",
        "os.fsync = guarded_fsync",
    ]

    completed = _run_script(
        identity_guard._v3._SECURE_ATOMIC_WRITE,
        env={
            "HL_FILE_PATH": str(target),
            "HL_FILE_CONTENT": base64.b64encode(b"new\n").decode("ascii"),
        },
        wrapper=wrapper,
    )

    assert completed.returncode == 0, completed.stderr
    assert target.read_text(encoding="utf-8") == "new\n"
    assert "directory_synced=0" in completed.stdout


def test_public_edit_reports_post_publication_durability_warning() -> None:
    tool = object.__new__(HarborFileEditTool)

    def completed(
        self: HarborFileEditTool,
        *args: object,
        **kwargs: object,
    ) -> ToolResult:
        return ToolResult(
            success=True,
            output="replaced 1 occurrence(s)\ndirectory_synced=0\n",
            metadata={"exit_code": 0},
        )

    parent = identity_guard._base.HarborFileEditTool
    original = parent.execute
    try:
        parent.execute = completed  # type: ignore[method-assign]
        result = tool.execute("/workspace/target.txt", "old", "new")
    finally:
        parent.execute = original  # type: ignore[method-assign]

    assert result.success is True
    assert result.metadata["directory_fsync"] is False
    assert result.metadata["durability_warning"] is True
    assert "directory_synced=" not in result.output
    assert "atomically published" in result.output


def test_public_registry_uses_identity_bound_harbor_tools() -> None:
    from bench.harbor_adapter import HLWorkerHarborAgent

    agent = object.__new__(HLWorkerHarborAgent)
    agent.tool_timeout_seconds = 1.0
    agent._goal_path = lambda: None  # type: ignore[method-assign]
    registry = agent._build_environment_registry(object(), object())

    assert isinstance(registry.get("write"), HarborFileWriteTool)
    assert isinstance(registry.get("edit"), HarborFileEditTool)
