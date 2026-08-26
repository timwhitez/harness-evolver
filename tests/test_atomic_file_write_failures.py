from __future__ import annotations

import base64
import os
from pathlib import Path
import stat
import subprocess

import pytest

from bench.harbor_adapter import _ATOMIC_WRITE_SCRIPT
from harness.tools import safe_path_io
from harness.tools.file_write import FileWriteTool


def test_local_fsync_failure_preserves_original_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")

    def fail_fsync(descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(safe_path_io.os, "fsync", fail_fsync)
    result = FileWriteTool().execute(str(target), "new")

    assert result.success is False
    assert "injected fsync failure" in result.error
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".target.txt.hl-write-*")) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell and permission fixture")
def test_harbor_sync_failure_aborts_before_replace_and_cleans_temp(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sync = fake_bin / "sync"
    fake_sync.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    fake_sync.chmod(0o755)

    completed = subprocess.run(
        ["sh", "-c", _ATOMIC_WRITE_SCRIPT],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "HL_FILE_PATH": str(target),
            "HL_FILE_CONTENT": base64.b64encode(b"new").decode("ascii"),
            "HL_APPEND": "0",
        },
        check=False,
    )

    assert completed.returncode == 74
    assert "failed to flush" in completed.stderr
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".hl-write.*")) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode-bit fixture")
def test_harbor_replacement_does_not_recreate_special_mode_bits(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o4755)

    completed = subprocess.run(
        ["sh", "-c", _ATOMIC_WRITE_SCRIPT],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HL_FILE_PATH": str(target),
            "HL_FILE_CONTENT": base64.b64encode(b"new").decode("ascii"),
            "HL_APPEND": "0",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
