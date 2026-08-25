from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from harness.tools.safe_path_io import atomic_write_text_nofollow


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor-relative no-follow I/O is POSIX-only",
)


def _temporary_paths(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.tmp-*"))


def test_new_atomic_nofollow_file_preserves_umask_creation_mode(tmp_path: Path) -> None:
    target = tmp_path / "created.txt"
    previous = os.umask(0o027)
    try:
        atomic_write_text_nofollow(target, "created")
    finally:
        os.umask(previous)

    assert target.read_text(encoding="utf-8") == "created"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert _temporary_paths(target) == []


def test_fsync_failure_before_replace_keeps_old_bytes_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old", encoding="utf-8")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        atomic_write_text_nofollow(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert _temporary_paths(target) == []


def test_fchmod_failure_before_replace_keeps_old_bytes_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)

    def fail_fchmod(_descriptor: int, _mode: int) -> None:
        raise OSError("injected chmod failure")

    monkeypatch.setattr(os, "fchmod", fail_fchmod)

    with pytest.raises(OSError, match="injected chmod failure"):
        atomic_write_text_nofollow(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert _temporary_paths(target) == []
