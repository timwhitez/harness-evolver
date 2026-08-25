from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

import pytest

import harness.tools.file_edit as file_edit_module
import harness.tools.safe_path_io as safe_path_io
from harness.tools.file_edit import FileEditTool
from harness.tools.file_write import FileWriteTool
from harness.tools.safe_path_io import (
    SafePathError,
    atomic_write_text_nofollow,
    file_identity,
    read_text_nofollow,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor-relative identity checks are POSIX-only",
)


def _temps(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.tmp-*"))


def test_transform_publication_rejects_replaced_inode(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    original, metadata = read_text_nofollow(target, errors="strict")
    moved = tmp_path / "original.txt"
    target.replace(moved)
    target.write_text("concurrent\n", encoding="utf-8")

    with pytest.raises(SafePathError, match="changed identity"):
        atomic_write_text_nofollow(
            target,
            "updated\n",
            mode=metadata.st_mode,
            expected_identity=file_identity(metadata),
            expected_sha256=hashlib.sha256(original.encode()).hexdigest(),
        )

    assert moved.read_text(encoding="utf-8") == "old\n"
    assert target.read_text(encoding="utf-8") == "concurrent\n"
    assert _temps(target) == []


def test_transform_publication_rejects_same_inode_content_change(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    original, metadata = read_text_nofollow(target, errors="strict")
    before = file_identity(metadata)
    target.write_text("concurrent\n", encoding="utf-8")
    assert file_identity(target.stat()) == before

    with pytest.raises(SafePathError, match="content changed"):
        atomic_write_text_nofollow(
            target,
            "updated\n",
            mode=metadata.st_mode,
            expected_identity=before,
            expected_sha256=hashlib.sha256(original.encode()).hexdigest(),
        )

    assert target.read_text(encoding="utf-8") == "concurrent\n"
    assert _temps(target) == []


def test_file_edit_passes_identity_and_digest_to_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old value\n", encoding="utf-8")
    moved = tmp_path / "original.txt"
    real_publish = safe_path_io.atomic_write_text_nofollow
    swapped = False

    def racing_publish(path: object, content: str, **kwargs: object) -> bool:
        nonlocal swapped
        target.replace(moved)
        target.write_text("concurrent value\n", encoding="utf-8")
        swapped = True
        return real_publish(path, content, **kwargs)

    monkeypatch.setattr(file_edit_module, "atomic_write_text_nofollow", racing_publish)

    result = FileEditTool().execute(str(target), "old", "new")

    assert swapped is True
    assert result.success is False
    assert result.metadata["target_identity_verified"] is True
    assert target.read_text(encoding="utf-8") == "concurrent value\n"
    assert moved.read_text(encoding="utf-8") == "old value\n"


def test_directory_fsync_failure_is_post_publication_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old\n", encoding="utf-8")
    real_fsync = safe_path_io.os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(safe_path_io.os, "fsync", fail_directory_fsync)

    result = FileWriteTool().execute(str(target), "new\n")

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "new\n"
    assert result.metadata["atomic_replace"] is True
    assert result.metadata["directory_fsync"] is False
    assert result.metadata["durability_warning"] is True
    assert "atomically published" in result.output
