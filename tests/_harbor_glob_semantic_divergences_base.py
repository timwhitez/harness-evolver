from __future__ import annotations

import glob
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from bench import _harbor_glob_issue14 as implementation
from bench.harbor_adapter import HarborGlobTool


def _local_fallback_tool() -> HarborGlobTool:
    tool = object.__new__(HarborGlobTool)

    def execute(
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> SimpleNamespace:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout or 10,
            env={**os.environ, **(env or {})},
        )
        return SimpleNamespace(
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    tool._exec = execute  # type: ignore[method-assign]
    return tool


@pytest.mark.parametrize(
    "pattern",
    [
        "[^a]",
        "[[:alpha:]]",
        "[[.a.]]",
        "[[=a=]]",
        "**/**/x.py",
        "é",
    ],
)
def test_known_python_bash_divergences_fail_closed(
    tmp_path: Path,
    pattern: str,
) -> None:
    for relative in ("a", "b", "^", "x.py", "nested/x.py", "é"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("data", encoding="utf-8")

    result = _local_fallback_tool()._glob_without_python(
        pattern=pattern,
        path=str(tmp_path),
    )

    assert result.success is False
    assert result.output == ""
    assert result.metadata["glob_pattern_unsupported"] is True


def test_multiple_recursive_components_really_duplicate_python_results(
    tmp_path: Path,
) -> None:
    target = tmp_path / "one" / "two" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("data", encoding="utf-8")
    pattern = "**/**/x.py"

    primary = glob.glob(os.path.join(str(tmp_path), pattern), recursive=True)
    result = _local_fallback_tool()._glob_without_python(
        pattern=pattern,
        path=str(tmp_path),
    )

    assert len(primary) > len(set(primary))
    assert result.success is False
    assert "duplicate paths" in result.error


def test_non_ascii_tree_fails_before_bash_locale_matching(tmp_path: Path) -> None:
    (tmp_path / "é.py").write_text("data", encoding="utf-8")

    result = _local_fallback_tool()._glob_without_python(
        pattern="*.py",
        path=str(tmp_path),
    )

    assert result.success is False
    assert result.metadata["glob_filename_unsupported"] is True
    assert "non-ASCII" in result.error


def test_ascii_subset_uses_an_executable_c_locale_preflight() -> None:
    script = implementation._runtime_module._GLOB_FALLBACK_SCRIPT
    assert "export LC_ALL=C" in script
    assert "*[![:print:]]*" in script
    assert "*[! -~]*" not in script


@pytest.mark.parametrize(
    "pattern",
    ["[!a]", "[a-c]", "src/[ab].py", "src/**/test_?.py"],
)
def test_supported_ascii_classes_remain_accepted(pattern: str) -> None:
    assert implementation._unsupported_glob_reason(pattern) == ""
