from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.tools.search import GrepTool
import harness.tools.search as search_tools


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="descriptor-relative path tests are POSIX-specific",
)


def test_execute_uses_stable_nofollow_reader_instead_of_external_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("needle\n", encoding="utf-8")

    def unexpected_external_search(*args, **kwargs):
        raise AssertionError("external search process must not receive a mutable path root")

    monkeypatch.setattr(search_tools._base.subprocess, "run", unexpected_external_search)

    result = GrepTool().execute("needle", path=str(tmp_path))

    assert result.success is True
    assert "needle" in result.output
    assert result.metadata["engine"] == "python-stable-nofollow"
    assert result.metadata["stable_root_descriptor"] is True
    assert result.metadata["external_search_disabled_for_path_safety"] is True


def test_execute_blocks_root_replaced_by_symlink_after_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "public.txt").write_text("public\n", encoding="utf-8")

    hidden = tmp_path / "terminal-bench-tasks" / "task" / "tests"
    hidden.mkdir(parents=True)
    secret = hidden / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    parked = tmp_path / "workspace-original"
    swapped = False

    def swap_after_authorization(root: Path, *, action: str):
        nonlocal swapped
        workspace.rename(parked)
        workspace.symlink_to(hidden, target_is_directory=True)
        swapped = True
        return None

    def unexpected_external_search(*args, **kwargs):
        raise AssertionError("external search process must not run")

    monkeypatch.setattr(search_tools._base, "_preflight_symlink_tree", swap_after_authorization)
    monkeypatch.setattr(search_tools._base.subprocess, "run", unexpected_external_search)

    result = GrepTool().execute("classified", path=str(workspace))

    assert swapped is True
    assert result.success is False
    assert result.output == ""
    assert result.metadata["blocked_by"] in {
        "canonical_path_guard",
        "leaderboard_integrity_guard",
    }
    assert "classified" not in result.error
    assert secret.read_text(encoding="utf-8") == "classified\n"


def test_execute_blocks_root_replaced_by_ordinary_sibling_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "public.txt").write_text("public\n", encoding="utf-8")

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    secret = replacement / "secret.txt"
    secret.write_text("ordinary-secret\n", encoding="utf-8")
    parked = tmp_path / "workspace-original"

    def swap_after_authorization(root: Path, *, action: str):
        workspace.rename(parked)
        replacement.rename(workspace)
        return None

    monkeypatch.setattr(search_tools._base, "_preflight_symlink_tree", swap_after_authorization)

    result = GrepTool().execute("ordinary-secret", path=str(workspace))

    assert result.success is False
    assert result.output == ""
    assert result.metadata["blocked_by"] == "canonical_path_guard"
    assert result.metadata["stable_root_descriptor"] is True
    assert "ordinary-secret" not in result.error
    assert secret.exists() is False
    assert (workspace / "secret.txt").read_text(encoding="utf-8") == "ordinary-secret\n"
