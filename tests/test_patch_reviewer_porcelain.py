from __future__ import annotations

from pathlib import Path
import subprocess

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


def test_changed_files_expands_both_rename_endpoints(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    old_path = tmp_path / "harness/old name.py"
    old_path.parent.mkdir(parents=True)
    old_path.write_text("value = 1\n", encoding="utf-8")
    _commit_all(tmp_path)

    _git(tmp_path, "mv", "harness/old name.py", "harness/new name.py")

    reviewer = PatchReviewer(tmp_path)
    changed = reviewer.changed_files()

    assert changed == ["harness/new name.py", "harness/old name.py"]
    diff = reviewer.diff_text(changed)
    assert "old name.py" in diff
    assert "new name.py" in diff


def test_cross_root_rename_checks_source_and_destination(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    source = tmp_path / "harness/module.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    _commit_all(tmp_path)

    (tmp_path / "jobs").mkdir()
    _git(tmp_path, "mv", "harness/module.py", "jobs/module.py")

    result = PatchReviewer(tmp_path).review_worktree()

    assert result.accepted is False
    assert "harness/module.py" in result.changed_files
    assert "jobs/module.py" in result.changed_files
    assert any("forbidden path changed: jobs/module.py" in reason for reason in result.reasons)


def test_nul_parser_preserves_tabs_newlines_unicode_and_copy_endpoints() -> None:
    reviewer = PatchReviewer(".")
    payload = (
        b"R  tests/new\tname.py\0tests/old name.py\0"
        + "C  tests/新\n文件.py\0tests/source.py\0".encode()
        + b"?? tests/untracked\nfile.py\0"
    )

    assert reviewer._parse_porcelain_v1_z(payload) == [
        "tests/new\tname.py",
        "tests/old name.py",
        "tests/新\n文件.py",
        "tests/source.py",
        "tests/untracked\nfile.py",
    ]


def test_untracked_files_use_nul_delimiters(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    special = tmp_path / "tests/untracked\nname.py"
    special.parent.mkdir(parents=True)
    special.write_text("value = 1\n", encoding="utf-8")

    reviewer = PatchReviewer(tmp_path)

    assert reviewer.changed_files() == ["tests/untracked\nname.py"]
    assert reviewer._untracked_files() == ["tests/untracked\nname.py"]
