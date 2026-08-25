from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.tools.search import GrepTool
import harness.tools.safe_path_io as safe_path_io


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="O_NOFOLLOW race test is POSIX-specific",
)


def test_python_grep_fallback_rejects_post_authorization_symlink_swap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    candidate = workspace / "candidate.txt"
    candidate.write_text("public\n", encoding="utf-8")

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
            and path == candidate.name
            and dir_fd is not None
            and flags & os.O_NOFOLLOW
        ):
            candidate.unlink()
            candidate.symlink_to(secret)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(safe_path_io.os, "open", racing_open)

    result = GrepTool()._python_grep(
        "classified",
        str(workspace),
        None,
        10,
    )

    assert swapped is True
    assert result.success is False
    assert result.metadata["blocked_by"] == "canonical_path_guard"
    assert "classified" not in result.output
    assert secret.read_text(encoding="utf-8") == "classified\n"
