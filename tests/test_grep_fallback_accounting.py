from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess
import sys

import pytest

from bench.harbor_adapter import _HARBOR_GREP_COUNTING_PYTHON
from harness.tools.search import GrepTool
import harness.tools.search as search_tools


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="canonical O_NOFOLLOW grep is POSIX-specific",
)


def test_local_read_failure_never_returns_successful_partial_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_lines = search_tools._bounded_physical_lines

    def stable_targets(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        yield Path("a.txt"), io.BytesIO(b"match-a\n"), object()
        yield Path("b.txt"), io.BytesIO(b"match-b\n"), object()

    scan_count = 0

    def fail_second_scan(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal scan_count
        scan_count += 1
        if scan_count == 2:
            raise OSError("injected stable traversal read failure")
        yield from original_lines(*args, **kwargs)

    monkeypatch.setattr(search_tools, "iter_stable_regular_files", stable_targets)
    monkeypatch.setattr(search_tools, "_bounded_physical_lines", fail_second_scan)

    result = GrepTool()._python_grep(
        "match",
        str(tmp_path),
        None,
        10,
    )

    assert result.success is False
    assert result.metadata["search_failed"] is True
    assert result.metadata["read_error_count"] == 1
    assert result.metadata["partial_results_available"] is True
    assert "match-a" in result.output
    assert "injected stable traversal read failure" in result.error
    assert scan_count == 2


def test_local_retained_match_line_is_independently_bounded(tmp_path: Path) -> None:
    target = tmp_path / "long.txt"
    target.write_text("match" + "x" * 100_000 + "\n", encoding="utf-8")

    result = GrepTool().execute("match", path=str(target), max_results=1)

    assert result.success is True
    assert result.metadata["match_count"] == 1
    assert len(result.output) < 5_000
    assert result.metadata["host_output_bounded"] is True


def test_local_oversized_physical_line_fails_before_materializing_it(
    tmp_path: Path,
) -> None:
    target = tmp_path / "oversized.txt"
    target.write_text("needle" + "x" * 64, encoding="utf-8")
    tool = GrepTool()
    tool.max_input_line_chars = 32

    result = tool._python_grep("needle", str(target), None, 10)

    assert result.success is False
    assert result.output == ""
    assert result.metadata["search_failed"] is True
    assert result.metadata["input_line_limit_exceeded"] is True
    assert result.metadata["physical_input_line_bounded"] is True
    assert result.metadata["max_input_line_chars"] == 32
    assert "physical line 1 exceeds" in result.error


def test_local_exact_input_line_cap_with_newline_remains_valid(
    tmp_path: Path,
) -> None:
    target = tmp_path / "exact.txt"
    target.write_text("n" * 32 + "\n", encoding="utf-8")
    tool = GrepTool()
    tool.max_input_line_chars = 32

    result = tool._python_grep("n+", str(target), None, 10)

    assert result.success is True
    assert result.metadata["match_count"] == 1
    assert result.metadata["max_input_line_chars"] == 32


def _run_harbor_script(
    root: Path,
    pattern: str,
    *,
    max_results: int = 200,
    max_input_line_chars: int = 1_000_000,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _HARBOR_GREP_COUNTING_PYTHON],
        env={
            **os.environ,
            "HL_ROOT": str(root),
            "HL_PATTERN": pattern,
            "HL_MAX_RESULTS": str(max_results),
            "HL_MAX_MATCH_CHARS": "4000",
            "HL_MAX_INPUT_LINE_CHARS": str(max_input_line_chars),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_harbor_script_counts_all_matches_but_returns_only_window(tmp_path: Path) -> None:
    target = tmp_path / "matches.txt"
    target.write_text(
        "".join(f"match-{index}\n" for index in range(205)),
        encoding="utf-8",
    )

    completed = _run_harbor_script(tmp_path, "match", max_results=3)

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[0] == "__HL_GREP_COUNT__205"
    assert len(lines[1:]) == 3


def test_harbor_script_reports_regex_errors_with_nonzero_status(tmp_path: Path) -> None:
    completed = _run_harbor_script(tmp_path, "[")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "invalid regular expression" in completed.stderr


def test_harbor_script_rejects_symlink_root_instead_of_following_it(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "secret.txt").write_text("match\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)

    completed = _run_harbor_script(alias, "match")

    assert completed.returncode != 0
    assert "match" not in completed.stdout


def test_harbor_script_rejects_hardlink_alias_before_content(
    tmp_path: Path,
) -> None:
    hidden = tmp_path / "terminal-bench-tasks" / "task" / "tests"
    hidden.mkdir(parents=True)
    secret = hidden / "secret.txt"
    secret.write_text("classified\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = workspace / "notes.txt"
    os.link(secret, alias)

    completed = _run_harbor_script(alias, "classified")

    assert completed.returncode == 74
    assert completed.stdout == ""
    assert "classified" not in completed.stderr
    assert "multiply-linked regular file" in completed.stderr


def test_harbor_script_fails_closed_on_oversized_physical_line(
    tmp_path: Path,
) -> None:
    target = tmp_path / "oversized.txt"
    target.write_text("needle" + "x" * 64, encoding="utf-8")

    completed = _run_harbor_script(
        target,
        "needle",
        max_input_line_chars=32,
    )

    assert completed.returncode == 75
    assert completed.stdout == ""
    assert "physical line 1 exceeds" in completed.stderr


def test_harbor_script_accepts_exact_line_cap_followed_by_newline(
    tmp_path: Path,
) -> None:
    target = tmp_path / "exact.txt"
    target.write_text("n" * 32 + "\n", encoding="utf-8")

    completed = _run_harbor_script(
        target,
        "n+",
        max_input_line_chars=32,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[0] == "__HL_GREP_COUNT__1"