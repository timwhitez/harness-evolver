from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.harbor_adapter import HarborFileReadTool, HarborFileWriteTool, HarborGlobTool
from harness.tools.file_edit import FileEditTool
from harness.tools.file_read import FileReadTool
from harness.tools.file_write import FileWriteTool
from harness.tools.search import GlobTool, GrepTool


pytestmark = pytest.mark.skipif(os.name == "nt", reason="symlink policy fixture is POSIX-specific")


def _make_hidden_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    hidden = tmp_path / "terminal-bench-tasks" / "task" / "tests"
    hidden.mkdir(parents=True)
    secret = hidden / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace, hidden, secret


def test_read_write_and_edit_block_a_safe_named_symlink_to_hidden_tests(tmp_path: Path) -> None:
    workspace, _, secret = _make_hidden_fixture(tmp_path)
    alias = workspace / "notes.txt"
    alias.symlink_to(secret)

    results = [
        FileReadTool().execute(str(alias)),
        FileWriteTool().execute(str(alias), "overwritten\n"),
        FileEditTool().execute(str(alias), "classified", "changed"),
    ]

    assert all(result.success is False for result in results)
    assert all(result.metadata["blocked_by"] == "leaderboard_integrity_guard" for result in results)
    assert all(str(secret) not in result.error for result in results)
    assert secret.read_text(encoding="utf-8") == "classified\n"


def test_new_file_write_resolves_an_existing_symlink_parent_before_authorization(
    tmp_path: Path,
) -> None:
    workspace, hidden, _ = _make_hidden_fixture(tmp_path)
    alias = workspace / "generated"
    alias.symlink_to(hidden, target_is_directory=True)
    requested = alias / "new.txt"

    result = FileWriteTool().execute(str(requested), "should not exist")

    assert result.success is False
    assert result.metadata["blocked_by"] == "leaderboard_integrity_guard"
    assert not (hidden / "new.txt").exists()


def test_host_memory_symlink_is_blocked_without_disclosing_the_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory = tmp_path / "trials" / "runs"
    memory.mkdir(parents=True)
    trajectory = memory / "trajectory.jsonl"
    trajectory.write_text("secret", encoding="utf-8")
    alias = workspace / "history.jsonl"
    alias.symlink_to(trajectory)

    result = FileReadTool().execute(str(alias))

    assert result.success is False
    assert result.metadata["blocked_by"] == "host_memory_guard"
    assert str(trajectory) not in result.error
    assert str(trajectory) not in json.dumps(result.metadata)


def test_parent_traversal_is_rejected_before_filesystem_resolution(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    requested = workspace / "missing" / ".." / "file.txt"

    result = FileWriteTool().execute(str(requested), "data")

    assert result.success is False
    assert result.metadata["blocked_by"] == "canonical_path_guard"
    assert "parent traversal" in result.error


def test_glob_rejects_parent_patterns_and_symlink_escape_results(tmp_path: Path) -> None:
    workspace, hidden, _ = _make_hidden_fixture(tmp_path)
    alias = workspace / "external"
    alias.symlink_to(hidden, target_is_directory=True)

    traversal = GlobTool().execute("../terminal-bench-tasks/**/*", str(workspace))
    escaped = GlobTool().execute("external/*", str(workspace))

    assert traversal.success is False
    assert traversal.metadata["blocked_by"] == "canonical_path_guard"
    assert escaped.success is False
    assert escaped.output == ""
    assert escaped.metadata["blocked_by"] in {
        "canonical_path_guard",
        "leaderboard_integrity_guard",
    }


def test_grep_preflights_symlinks_before_searching_content(tmp_path: Path) -> None:
    workspace, _, secret = _make_hidden_fixture(tmp_path)
    (workspace / "safe.txt").write_text("classified\n", encoding="utf-8")
    (workspace / "external.txt").symlink_to(secret)

    result = GrepTool().execute("classified", str(workspace))

    assert result.success is False
    assert result.output == ""
    assert result.metadata["blocked_by"] in {
        "canonical_path_guard",
        "leaderboard_integrity_guard",
    }


def test_safe_internal_symlink_read_uses_the_authorized_canonical_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.txt"
    target.write_text("safe\n", encoding="utf-8")
    alias = workspace / "alias.txt"
    alias.symlink_to(target)

    result = FileReadTool().execute(str(alias))

    assert result.success is True
    assert "safe" in result.output


def test_harbor_read_blocks_the_container_resolved_target_before_delegation(
    tmp_path: Path,
) -> None:
    workspace, _, secret = _make_hidden_fixture(tmp_path)
    alias = workspace / "safe-looking.txt"
    encoded = base64.b64encode(os.fsencode(str(secret))).decode("ascii")
    tool = HarborFileReadTool(environment=object(), loop=object())
    calls: list[str] = []

    def fake_exec(command: str, **_: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout=encoded, stderr="", return_code=0)

    tool._exec = fake_exec  # type: ignore[method-assign]

    result = tool.execute(str(alias))

    assert result.success is False
    assert result.metadata["blocked_by"] == "leaderboard_integrity_guard"
    assert len(calls) == 1
    assert str(secret) not in result.error


def test_harbor_parent_traversal_fails_before_any_environment_call(tmp_path: Path) -> None:
    tool = HarborFileReadTool(environment=object(), loop=object())

    def unexpected_exec(*_: object, **__: object) -> SimpleNamespace:
        raise AssertionError("environment must not be called for lexical traversal")

    tool._exec = unexpected_exec  # type: ignore[method-assign]

    result = tool.execute(str(tmp_path / "a" / ".." / "secret"))

    assert result.success is False
    assert result.metadata["blocked_by"] == "canonical_path_guard"


def test_harbor_write_rechecks_size_policy_on_canonical_alias(tmp_path: Path) -> None:
    alias = tmp_path / "alias" / "gpt2.c"
    resolved = tmp_path / "actual" / "app" / "gpt2.c"
    tool = object.__new__(HarborFileWriteTool)
    tool._guard_environment_path = (  # type: ignore[method-assign]
        lambda *args, **kwargs: (str(resolved), None)
    )
    tool._run_secure_python = (  # type: ignore[method-assign]
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("policy rejection must precede publication")
        )
    )

    result = tool.execute(str(alias), "x" * 5000)

    assert result.success is False
    assert result.metadata["blocked_by"] == "deliverable_size_cap_write_guard"
    assert result.metadata["path"] == str(resolved)
    assert result.metadata["content_bytes"] == 5000
    assert not resolved.exists()


def test_harbor_write_rechecks_staged_policy_on_canonical_alias(tmp_path: Path) -> None:
    alias = tmp_path / "benign.txt"
    resolved = tmp_path / "script.py"
    content = (
        "import urllib.request\n"
        "urllib.request.urlopen('https://files.pythonhosted.org/pkg.whl')\n"
    )
    tool = object.__new__(HarborFileWriteTool)
    tool._guard_environment_path = (  # type: ignore[method-assign]
        lambda *args, **kwargs: (str(resolved), None)
    )
    tool._run_secure_python = (  # type: ignore[method-assign]
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("policy rejection must precede publication")
        )
    )

    result = tool.execute(str(alias), content)

    assert result.success is False
    assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
    assert not resolved.exists()


def test_harbor_glob_revalidates_every_container_match(tmp_path: Path) -> None:
    workspace, _, secret = _make_hidden_fixture(tmp_path)
    tool = HarborGlobTool(environment=object(), loop=object())

    matches, failure = tool._guard_environment_matches(
        [str(secret)],
        requested=str(workspace),
        root=str(workspace.resolve()),
        action="glob result",
    )

    assert matches is None
    assert failure is not None
    assert failure.success is False
    assert failure.metadata["blocked_by"] in {
        "canonical_path_guard",
        "leaderboard_integrity_guard",
    }
