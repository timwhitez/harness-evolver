from __future__ import annotations

import base64
import os
from pathlib import Path
import stat
import subprocess

import pytest

from bench.harbor_adapter import _ATOMIC_WRITE_SCRIPT
from harness.tools.file_write import FileWriteTool


def test_local_overwrite_is_atomic_and_preserves_mode(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)

    result = FileWriteTool().execute(str(target), "new")

    assert result.success is True
    assert result.metadata["atomic_replace"] is True
    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(tmp_path.glob(".target.txt.hl-write-*")) == []


def test_local_replace_failure_keeps_original_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr("harness.tools.file_write.os.replace", fail_replace)

    result = FileWriteTool().execute(str(target), "new")

    assert result.success is False
    assert "injected replace failure" in result.error
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".target.txt.hl-write-*")) == []


def test_local_new_file_uses_normal_umask_creation_mode(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "created.txt"
    previous_umask = os.umask(0o027)
    try:
        result = FileWriteTool().execute(str(target), "created")
    finally:
        os.umask(previous_umask)

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "created"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(target.parent.glob(".created.txt.hl-write-*")) == []


def test_local_overwrite_does_not_recreate_special_mode_bits(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o4755)

    result = FileWriteTool().execute(str(target), "new")

    assert result.success is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def _run_harbor_script(
    target: Path,
    payload: str,
    *,
    append: bool,
    encoded: bool = True,
) -> subprocess.CompletedProcess[str]:
    content = base64.b64encode(payload.encode()).decode() if encoded else payload
    return subprocess.run(
        ["sh", "-c", _ATOMIC_WRITE_SCRIPT],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HL_FILE_PATH": str(target),
            "HL_FILE_CONTENT": content,
            "HL_APPEND": "1" if append else "0",
        },
    )


def test_harbor_script_atomically_overwrites_and_appends(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)

    overwrite = _run_harbor_script(target, "new", append=False)
    append = _run_harbor_script(target, "+tail", append=True)

    assert overwrite.returncode == 0, overwrite.stderr
    assert append.returncode == 0, append.stderr
    assert target.read_text(encoding="utf-8") == "new+tail"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(tmp_path.glob(".hl-write.*")) == []


def test_harbor_decode_failure_preserves_previous_target(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    failed = _run_harbor_script(
        target,
        "not-valid-base64%%%",
        append=False,
        encoded=False,
    )

    assert failed.returncode != 0
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".hl-write.*")) == []
