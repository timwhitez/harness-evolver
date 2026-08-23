from __future__ import annotations

import base64
import os
from pathlib import Path
import subprocess
import sys

import pytest

from bench._canonical_harbor_grep import _HARBOR_GREP_PYTHON
from bench._canonical_harbor_paths_v2 import _SECURE_READ, _SECURE_WRITE
from harness.tools.file_read import FileReadTool
from harness.tools.file_write import FileWriteTool
import harness.tools.safe_path_io as safe_path_io


pytestmark = pytest.mark.skipif(os.name != "posix", reason="O_NOFOLLOW race tests are POSIX-specific")


def test_local_read_rejects_final_symlink_swapped_after_authorization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "victim.txt"
    target.write_text("public\n", encoding="utf-8")
    hidden = tmp_path / "terminal-bench-tasks" / "task" / "tests"
    hidden.mkdir(parents=True)
    secret = hidden / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    real_open = safe_path_io.os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (
            not swapped
            and path == target.name
            and dir_fd is not None
            and flags & os.O_NOFOLLOW
        ):
            target.unlink()
            target.symlink_to(secret)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(safe_path_io.os, "open", racing_open)

    result = FileReadTool().execute(str(target))

    assert swapped is True
    assert result.success is False
    assert "classified" not in result.output
    assert secret.read_text(encoding="utf-8") == "classified\n"


def test_local_atomic_write_replaces_swapped_symlink_without_following_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "victim.txt"
    target.write_text("old\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    real_replace = safe_path_io.os.replace
    swapped = False

    def racing_replace(src, dst, *args, **kwargs):
        nonlocal swapped
        if not swapped and dst == target.name:
            target.unlink()
            target.symlink_to(secret)
            swapped = True
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(safe_path_io.os, "replace", racing_replace)

    result = FileWriteTool().execute(str(target), "new\n")

    assert swapped is True
    assert result.success is True
    assert target.is_symlink() is False
    assert target.read_text(encoding="utf-8") == "new\n"
    assert secret.read_text(encoding="utf-8") == "classified\n"


def test_harbor_secure_read_rejects_post_authorization_symlink(tmp_path: Path) -> None:
    target = tmp_path / "authorized.txt"
    secret = tmp_path / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    target.symlink_to(secret)

    completed = subprocess.run(
        [sys.executable, "-c", _SECURE_READ],
        env={
            **os.environ,
            "HL_FILE_PATH": str(target),
            "HL_OFFSET": "1",
            "HL_LIMIT": "10",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "classified" not in completed.stdout


def test_harbor_secure_write_never_follows_post_authorization_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "authorized.txt"
    secret = tmp_path / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    target.symlink_to(secret)

    completed = subprocess.run(
        [sys.executable, "-c", _SECURE_WRITE],
        env={
            **os.environ,
            "HL_FILE_PATH": str(target),
            "HL_FILE_CONTENT": base64.b64encode(b"overwrite\n").decode("ascii"),
            "HL_APPEND": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert secret.read_text(encoding="utf-8") == "classified\n"


def test_harbor_grep_rejects_file_swapped_after_walk_authorization(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = workspace / "candidate.txt"
    candidate.write_text("public\n", encoding="utf-8")
    hidden = tmp_path / "terminal-bench-tasks" / "task" / "tests"
    hidden.mkdir(parents=True)
    secret = hidden / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")

    wrapper = "\n".join(
        [
            "import os",
            "from pathlib import Path",
            f"target = Path({str(candidate)!r})",
            f"secret = Path({str(secret)!r})",
            "real_open = os.open",
            "swapped = False",
            "def racing_open(path, flags, mode=0o777, *, dir_fd=None):",
            "    global swapped",
            "    if (not swapped and path == target.name and dir_fd is not None "
            "and flags & os.O_NOFOLLOW):",
            "        target.unlink()",
            "        target.symlink_to(secret)",
            "        swapped = True",
            "    return real_open(path, flags, mode, dir_fd=dir_fd)",
            "os.open = racing_open",
            f"exec({_HARBOR_GREP_PYTHON!r})",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", wrapper],
        env={
            **os.environ,
            "HL_ROOT": str(workspace),
            "HL_PATTERN": "classified",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "classified" not in completed.stdout
    assert secret.read_text(encoding="utf-8") == "classified\n"
