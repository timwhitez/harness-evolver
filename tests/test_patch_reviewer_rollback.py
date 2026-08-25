from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from meta.reviewer import PatchReviewer


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")


def _commit_all(repo: Path) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "fixture")


def _assert_clean(repo: Path) -> None:
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert status.stdout == ""


def test_staged_rename_is_in_diff_and_fully_rolled_back(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    old = tmp_path / "harness/old name.py"
    old.parent.mkdir(parents=True)
    old.write_text("value = 1\n", encoding="utf-8")
    _commit_all(tmp_path)

    _git(tmp_path, "mv", "harness/old name.py", "harness/new name.py")
    reviewer = PatchReviewer(tmp_path)
    diff = reviewer.diff_text(reviewer.changed_files())

    assert "old name.py" in diff
    assert "new name.py" in diff
    patch = reviewer.save_reverse_patch(tmp_path / "rename.patch")
    assert reviewer.rollback(patch) is True
    assert old.read_text(encoding="utf-8") == "value = 1\n"
    assert not (tmp_path / "harness/new name.py").exists()
    patch.unlink()
    _assert_clean(tmp_path)


def test_staged_rename_plus_edit_restores_original_bytes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    old = tmp_path / "harness/module.py"
    old.parent.mkdir(parents=True)
    old.write_text("value = 1\n", encoding="utf-8")
    _commit_all(tmp_path)

    _git(tmp_path, "mv", "harness/module.py", "harness/renamed.py")
    renamed = tmp_path / "harness/renamed.py"
    renamed.write_text("value = 2\n", encoding="utf-8")
    _git(tmp_path, "add", "harness/renamed.py")

    reviewer = PatchReviewer(tmp_path)
    patch = reviewer.save_reverse_patch(tmp_path / "rename-edit.patch")
    assert reviewer.rollback(patch) is True
    assert old.read_text(encoding="utf-8") == "value = 1\n"
    assert not renamed.exists()
    patch.unlink()
    _assert_clean(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="uses /dev/null no-index binary patch")
def test_untracked_binary_file_is_part_of_reversible_delta(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    tracked = tmp_path / "harness/tracked.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("value = 1\n", encoding="utf-8")
    _commit_all(tmp_path)

    binary = tmp_path / "tests/payload.bin"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x00\xff\x10binary\x00payload")

    reviewer = PatchReviewer(tmp_path)
    patch = reviewer.save_reverse_patch(tmp_path / "binary.patch")
    patch_text = patch.read_text(encoding="utf-8")
    assert "GIT binary patch" in patch_text
    assert reviewer.rollback(patch) is True
    assert not binary.exists()
    patch.unlink()
    _assert_clean(tmp_path)
