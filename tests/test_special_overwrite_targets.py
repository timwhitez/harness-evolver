from __future__ import annotations

import base64
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

import pytest

from bench._canonical_harbor_special_write import (
    HarborFileWriteTool as GuardedHarborFileWriteTool,
    _SECURE_ATOMIC_WRITE,
)
from bench.harbor_adapter import HarborFileWriteTool
from harness.tools.file_write import FileWriteTool


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor-relative special-target checks require POSIX",
)


def _identity(path: Path) -> tuple[int, int, int]:
    metadata = path.lstat()
    return int(metadata.st_dev), int(metadata.st_ino), stat.S_IFMT(metadata.st_mode)


def _temporary_paths(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.tmp-*"))


def _run_harbor_write(path: Path, content: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _SECURE_ATOMIC_WRITE],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "HL_FILE_PATH": str(path),
            "HL_FILE_CONTENT": base64.b64encode(
                content.encode("utf-8")
            ).decode("ascii"),
        },
    )


def test_public_harbor_registry_exports_the_guarded_writer() -> None:
    assert HarborFileWriteTool is GuardedHarborFileWriteTool


def test_local_whole_file_write_rejects_fifo_before_temp_creation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "endpoint.fifo"
    os.mkfifo(target)
    before = _identity(target)

    result = FileWriteTool().execute(str(target), "replacement")

    assert result.success is False
    assert "non-regular overwrite target" in result.error
    assert _identity(target) == before
    assert stat.S_ISFIFO(target.lstat().st_mode)
    assert _temporary_paths(target) == []


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="Unix-domain sockets are unavailable",
)
def test_local_whole_file_write_rejects_unix_socket_before_temp_creation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "endpoint.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(target))
    try:
        before = _identity(target)

        result = FileWriteTool().execute(str(target), "replacement")

        assert result.success is False
        assert "non-regular overwrite target" in result.error
        assert _identity(target) == before
        assert stat.S_ISSOCK(target.lstat().st_mode)
        assert _temporary_paths(target) == []
    finally:
        server.close()


def test_harbor_whole_file_write_rejects_fifo_before_temp_creation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "endpoint.fifo"
    os.mkfifo(target)
    before = _identity(target)

    completed = _run_harbor_write(target, "replacement")

    assert completed.returncode != 0
    assert "target is not a regular file" in completed.stderr
    assert _identity(target) == before
    assert stat.S_ISFIFO(target.lstat().st_mode)
    assert _temporary_paths(target) == []


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="Unix-domain sockets are unavailable",
)
def test_harbor_whole_file_write_rejects_unix_socket_before_temp_creation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "endpoint.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(target))
    try:
        before = _identity(target)

        completed = _run_harbor_write(target, "replacement")

        assert completed.returncode != 0
        assert "target is not a regular file" in completed.stderr
        assert _identity(target) == before
        assert stat.S_ISSOCK(target.lstat().st_mode)
        assert _temporary_paths(target) == []
    finally:
        server.close()


def test_regular_hard_link_overwrite_still_dealiases_only_selected_entry(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.txt"
    target = tmp_path / "target.txt"
    original.write_text("old", encoding="utf-8")
    os.link(original, target)

    result = FileWriteTool().execute(str(target), "new")

    assert result.success is True
    assert original.read_text(encoding="utf-8") == "old"
    assert target.read_text(encoding="utf-8") == "new"
    assert original.stat().st_ino != target.stat().st_ino
    assert _temporary_paths(target) == []


def test_harbor_regular_hard_link_overwrite_still_dealiases_selected_entry(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.txt"
    target = tmp_path / "target.txt"
    original.write_text("old", encoding="utf-8")
    os.link(original, target)

    completed = _run_harbor_write(target, "new")

    assert completed.returncode == 0, completed.stderr
    assert original.read_text(encoding="utf-8") == "old"
    assert target.read_text(encoding="utf-8") == "new"
    assert original.stat().st_ino != target.stat().st_ino
    assert _temporary_paths(target) == []


def test_missing_target_creation_remains_supported_locally_and_in_harbor(
    tmp_path: Path,
) -> None:
    local_target = tmp_path / "local.txt"
    harbor_target = tmp_path / "harbor.txt"

    local = FileWriteTool().execute(str(local_target), "local")
    harbor = _run_harbor_write(harbor_target, "harbor")

    assert local.success is True
    assert local_target.read_text(encoding="utf-8") == "local"
    assert harbor.returncode == 0, harbor.stderr
    assert harbor_target.read_text(encoding="utf-8") == "harbor"
    assert _temporary_paths(local_target) == []
    assert _temporary_paths(harbor_target) == []
