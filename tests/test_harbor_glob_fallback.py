from __future__ import annotations

import glob
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

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


def _fixture(root: Path) -> None:
    for relative in [
        ".root.py",
        "src/a.py",
        "src/b.txt",
        "src/test_root.py",
        "src/deep/b.py",
        "src/deep/test_deep.py",
        "tests/c.py",
        "src/.hidden/secret.py",
        ".dot/visible_by_explicit_pattern.py",
    ]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")


def _matches(result_output: str) -> list[str]:
    if result_output == "(no matches)":
        return []
    return [line for line in result_output.splitlines() if line]


def _primary_matches(root: str, pattern: str) -> list[str]:
    return glob.glob(os.path.join(root, pattern), recursive=True)[:500]


@pytest.mark.parametrize(
    ("pattern", "expected_relative"),
    [
        ("tests/*.py", "tests/c.py"),
        (".dot/*.py", ".dot/visible_by_explicit_pattern.py"),
        ("src/b.txt", "src/b.txt"),
        ("src/deep/test_*.py", "src/deep/test_deep.py"),
        ("src/[a].py", "src/a.py"),
        ("missing-*.py", None),
    ],
)
def test_zero_or_single_fallback_result_matches_python_exactly(
    tmp_path: Path,
    pattern: str,
    expected_relative: str | None,
) -> None:
    _fixture(tmp_path)
    root = str(tmp_path)

    result = _local_fallback_tool()._glob_without_python(
        pattern=pattern,
        path=root,
    )

    assert result.success is True, result.error
    assert _matches(result.output) == _primary_matches(root, pattern)
    expected = [] if expected_relative is None else [str(tmp_path / expected_relative)]
    assert _matches(result.output) == expected
    assert result.metadata["glob_semantics"] == "python-glob-recursive"
    assert result.metadata["hidden_components"] == "python-glob-default"


def test_multiple_matches_fail_instead_of_returning_bash_order(
    tmp_path: Path,
) -> None:
    # Deliberately use a non-sorted creation order. Python glob follows directory
    # enumeration order while Bash compgen applies shell ordering; either way,
    # more than one result makes the sequence contract unprovable.
    for name in ("z.py", "a.py", "m.py", "b.py"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    primary = _primary_matches(str(tmp_path), "*.py")
    result = _local_fallback_tool()._glob_without_python(
        pattern="*.py",
        path=str(tmp_path),
    )

    assert len(primary) == 4
    assert result.success is False
    assert result.output == ""
    assert result.metadata["glob_result_order_unsupported"] is True
    assert result.metadata["observed_match_count"] == 4
    assert "result order" in result.error


def test_relative_root_spelling_matches_primary_for_one_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "only.py").write_text("data", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = _local_fallback_tool()._glob_without_python(pattern="*.py", path=".")

    assert result.success is True
    assert _matches(result.output) == _primary_matches(".", "*.py") == ["./only.py"]


def test_hidden_components_follow_python_glob_defaults(tmp_path: Path) -> None:
    visible = tmp_path / "visible.py"
    hidden = tmp_path / ".hidden.py"
    nested_visible = tmp_path / "src" / "visible.py"
    nested_hidden = tmp_path / "src" / ".hidden.py"
    explicit = tmp_path / ".dot" / "explicit.py"
    for target in (visible, hidden, nested_visible, nested_hidden, explicit):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("data", encoding="utf-8")

    tool = _local_fallback_tool()
    root = str(tmp_path)
    root_result = tool._glob_without_python(pattern="*.py", path=root)
    nested_result = tool._glob_without_python(pattern="src/*.py", path=root)
    explicit_result = tool._glob_without_python(pattern=".dot/*.py", path=root)

    assert root_result.success is nested_result.success is explicit_result.success is True
    assert _matches(root_result.output) == [str(visible)]
    assert _matches(nested_result.output) == [str(nested_visible)]
    assert _matches(explicit_result.output) == [str(explicit)]


def test_double_star_preserves_zero_component_root_spelling_for_empty_tree(
    tmp_path: Path,
) -> None:
    root = str(tmp_path)

    result = _local_fallback_tool()._glob_without_python(pattern="**", path=root)

    assert result.success is True
    assert _matches(result.output) == _primary_matches(root, "**") == [root + os.sep]


@pytest.mark.skipif(os.name == "nt", reason="POSIX newline filename fixture")
def test_newline_filename_fails_instead_of_splitting_the_path(tmp_path: Path) -> None:
    _fixture(tmp_path)
    newline_file = tmp_path / "src" / "line\nbreak.py"
    newline_file.write_text("data", encoding="utf-8")

    result = _local_fallback_tool()._glob_without_python(
        pattern="**/*.py",
        path=str(tmp_path),
    )

    assert str(newline_file) in _primary_matches(str(tmp_path), "**/*.py")
    assert result.success is False
    assert result.output == ""
    assert result.metadata["glob_filename_unsupported"] is True
    assert "cannot represent" in result.error


def test_recursive_symlink_directory_fails_instead_of_omitting_primary_matches(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real" / "nested"
    real.mkdir(parents=True)
    target = real / "through-link.py"
    target.write_text("data", encoding="utf-8")
    link = tmp_path / "linked"
    try:
        link.symlink_to(tmp_path / "real", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink fixture unavailable: {exc}")

    assert str(link / "nested" / "through-link.py") in _primary_matches(
        str(tmp_path),
        "**/*.py",
    )
    result = _local_fallback_tool()._glob_without_python(
        pattern="**/*.py",
        path=str(tmp_path),
    )

    assert result.success is False
    assert result.metadata["recursive_symlink_glob_unsupported"] is True


def test_nonrecursive_glob_through_symlink_directory_remains_supported(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "file.py").write_text("data", encoding="utf-8")
    link = tmp_path / "linked"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink fixture unavailable: {exc}")

    result = _local_fallback_tool()._glob_without_python(
        pattern="linked/*.py",
        path=str(tmp_path),
    )

    assert result.success is True
    assert _matches(result.output) == _primary_matches(str(tmp_path), "linked/*.py")


def test_more_than_500_matches_is_also_reported_as_order_unsupported(
    tmp_path: Path,
) -> None:
    for index in range(501):
        (tmp_path / f"file-{index:03d}.txt").write_text("data", encoding="utf-8")

    result = _local_fallback_tool()._glob_without_python(
        pattern="*.txt",
        path=str(tmp_path),
    )

    assert len(glob.glob(os.path.join(str(tmp_path), "*.txt"))) == 501
    assert result.success is False
    assert result.metadata["glob_result_limit_unsupported"] is True
    assert result.metadata["glob_result_order_unsupported"] is True


@pytest.mark.parametrize(
    "pattern",
    ["src/[abc.py", "../*.py", "/tmp/*.py", r"src\*.py"],
)
def test_unsupported_patterns_fail_instead_of_broadening(
    tmp_path: Path,
    pattern: str,
) -> None:
    _fixture(tmp_path)

    result = _local_fallback_tool()._glob_without_python(
        pattern=pattern,
        path=str(tmp_path),
    )

    assert result.success is False
    assert result.metadata["glob_pattern_unsupported"] is True
    assert "Unsupported Python-free glob pattern" in result.error
