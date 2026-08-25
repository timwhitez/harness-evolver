from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.harbor_adapter import HarborGrepTool
from harness.tools.search import GrepTool


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="canonical O_NOFOLLOW grep is POSIX-specific",
)


def test_local_invalid_regex_is_a_failure(tmp_path: Path) -> None:
    result = GrepTool().execute("[", path=str(tmp_path))

    assert result.success is False
    assert result.metadata["search_failed"] is True
    assert result.metadata["engine"] == "python-stable-nofollow"
    assert "Invalid regex" in result.error


def test_local_zero_matches_is_success(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("present\n", encoding="utf-8")

    result = GrepTool().execute("missing", path=str(tmp_path))

    assert result.success is True
    assert result.output == "(no matches)"
    assert result.metadata["match_count"] == 0
    assert result.metadata["returned_count"] == 0
    assert result.metadata["omitted_count"] == 0


def test_local_truncation_uses_complete_scan_counts(tmp_path: Path) -> None:
    target = tmp_path / "file.py"
    target.write_text("".join(f"match-{index}\n" for index in range(5)), encoding="utf-8")

    result = GrepTool().execute("match", path=str(tmp_path), max_results=2)

    assert result.success is True
    assert result.metadata["match_count"] == 5
    assert result.metadata["returned_count"] == 2
    assert result.metadata["omitted_count"] == 3
    assert result.metadata["truncated"] is True
    assert result.output.count(str(target)) == 2
    assert "3 more results truncated" in result.output


def test_local_missing_path_is_a_failure() -> None:
    result = GrepTool().execute("match", path="/definitely/missing")

    assert result.success is False
    assert result.output == ""


def _harbor_tool(
    result: SimpleNamespace,
    *,
    root: Path,
) -> HarborGrepTool:
    tool = object.__new__(HarborGrepTool)
    tool._guard_environment_path = (  # type: ignore[method-assign]
        lambda *args, **kwargs: (str(root.resolve()), None)
    )
    tool._exec = lambda *args, **kwargs: result  # type: ignore[method-assign]
    return tool


def test_harbor_invalid_regex_is_not_reported_as_no_matches(tmp_path: Path) -> None:
    result = _harbor_tool(
        SimpleNamespace(
            return_code=2,
            stdout="",
            stderr="invalid regular expression: unterminated character set",
        ),
        root=tmp_path,
    ).execute("[", str(tmp_path))

    assert result.success is False
    assert result.metadata["exit_code"] == 2
    assert result.metadata["search_failed"] is True
    assert result.metadata["host_output_bounded"] is True
    assert "invalid regular expression" in result.error


def test_harbor_hardlink_exit_is_a_canonical_policy_failure(tmp_path: Path) -> None:
    result = _harbor_tool(
        SimpleNamespace(
            return_code=74,
            stdout="classified\n",
            stderr="multiply-linked regular file",
        ),
        root=tmp_path,
    ).execute("classified", str(tmp_path))

    assert result.success is False
    assert result.output == ""
    assert result.metadata["exit_code"] == 74
    assert result.metadata["blocked_by"] == "canonical_path_guard"
    assert result.metadata["hardlink_alias_blocked"] is True
    assert "classified" not in result.error


def test_harbor_zero_matches_remains_success(tmp_path: Path) -> None:
    result = _harbor_tool(
        SimpleNamespace(
            return_code=0,
            stdout="__HL_GREP_COUNT__0\n",
            stderr="",
        ),
        root=tmp_path,
    ).execute("missing", str(tmp_path))

    assert result.success is True
    assert result.output == "(no matches)"
    assert result.metadata["match_count"] == 0
    assert result.metadata["host_output_bounded"] is True


def test_harbor_truncation_uses_count_header_and_bounded_matches(tmp_path: Path) -> None:
    returned = "\n".join(f"file.py:{index}:match" for index in range(200))
    stdout = "__HL_GREP_COUNT__205\n" + returned + "\n"
    result = _harbor_tool(
        SimpleNamespace(return_code=0, stdout=stdout, stderr=""),
        root=tmp_path,
    ).execute("match", str(tmp_path))

    assert result.success is True
    assert result.metadata["match_count"] == 205
    assert result.metadata["returned_count"] == 200
    assert result.metadata["omitted_count"] == 5
    assert result.output.count("file.py:") == 200
    assert "5 more results truncated" in result.output


def test_harbor_success_requires_structured_count_metadata(tmp_path: Path) -> None:
    result = _harbor_tool(
        SimpleNamespace(return_code=0, stdout="file.py:1:match\n", stderr=""),
        root=tmp_path,
    ).execute("match", str(tmp_path))

    assert result.success is False
    assert result.metadata["search_failed"] is True
    assert "malformed count metadata" in result.error