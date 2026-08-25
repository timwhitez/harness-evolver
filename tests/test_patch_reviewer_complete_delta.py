from __future__ import annotations

from pathlib import Path
import subprocess

from meta.reviewer import PatchReviewer


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=text,
    )


def _init_repo(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")


def _commit_all(repo: Path, message: str = "fixture") -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)


def _status(repo: Path) -> bytes:
    return _git(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        text=False,
    ).stdout


def test_staged_rename_plus_edit_is_fully_rolled_back(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    source = tmp_path / "harness/original.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    _commit_all(tmp_path)

    _git(tmp_path, "mv", "harness/original.py", "harness/renamed.py")
    (tmp_path / "harness/renamed.py").write_text("value = 2\n", encoding="utf-8")
    _git(tmp_path, "add", "harness/renamed.py")

    reviewer = PatchReviewer(tmp_path)
    patch = reviewer.save_reverse_patch(tmp_path / "rename.patch")

    assert "harness/original.py" in reviewer.changed_files()
    assert "harness/renamed.py" in reviewer.changed_files()
    assert reviewer.rollback(patch) is True
    assert (tmp_path / "harness/original.py").read_text(encoding="utf-8") == "value = 1\n"
    assert not (tmp_path / "harness/renamed.py").exists()
    assert _status(tmp_path) == b"?? rename.patch\0"


def test_untracked_binary_delta_is_reviewable_and_reversible(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    tracked = tmp_path / "README.md"
    tracked.write_text("fixture\n", encoding="utf-8")
    _commit_all(tmp_path)

    binary = tmp_path / "tests/payload.bin"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x00\x01\xffbinary\n")

    reviewer = PatchReviewer(tmp_path)
    patch_text = reviewer.diff_text(["tests/payload.bin"])
    assert "GIT binary patch" in patch_text or "Binary files" in patch_text
    patch = tmp_path / "binary.patch"
    patch.write_text(patch_text, encoding="utf-8")

    assert reviewer.rollback(patch) is True
    assert not binary.exists()


def test_path_scoped_rollback_preserves_unrelated_staged_index_state(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    target = tmp_path / "harness/target.py"
    unrelated = tmp_path / "config/keep.yaml"
    target.parent.mkdir(parents=True)
    unrelated.parent.mkdir(parents=True)
    target.write_text("value = 1\n", encoding="utf-8")
    unrelated.write_text("value: old\n", encoding="utf-8")
    _commit_all(tmp_path)

    unrelated.write_text("value: staged\n", encoding="utf-8")
    _git(tmp_path, "add", "config/keep.yaml")
    target.write_text("value = 2\n", encoding="utf-8")

    reviewer = PatchReviewer(tmp_path)
    patch = tmp_path / "target.patch"
    patch.write_text(reviewer.diff_text(["harness/target.py"]), encoding="utf-8")

    assert reviewer.rollback(patch) is True
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    cached = _git(tmp_path, "diff", "--cached", "--", "config/keep.yaml").stdout
    assert "value: staged" in cached
    assert "value: old" in cached
