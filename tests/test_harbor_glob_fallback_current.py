from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from bench._harbor_glob_fallback import (
    HarborGlobTool,
    _GLOB_FALLBACK_SCRIPT,
    _unsupported_glob_reason,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix"
    or shutil.which("bash") is None
    or shutil.which("find") is None,
    reason="Python-free canonical glob fallback requires POSIX Bash/find",
)


def _fixture(root: Path) -> None:
    for relative in [
        "a.py",
        "b.txt",
        ".root.py",
        "src/a.py",
        "src/b.txt",
        "src/.hidden.py",
        "src/deep/b.py",
        "src/deep/test_deep.py",
        ".dot/visible.py",
    ]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")


def _primary(root: Path, pattern: str) -> list[str]:
    return sorted({str(candidate.resolve()) for candidate in root.glob(pattern)})


def _run_fallback(
    root: Path,
    pattern: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    recursive = any(
        component == "**"
        for component in pattern.removeprefix("./").split("/")
    )
    return subprocess.run(
        ["bash", "-c", _GLOB_FALLBACK_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            **(extra_env or {}),
            "HL_ROOT": str(root.resolve()),
            "HL_PATTERN": pattern,
            "HL_RECURSIVE": "1" if recursive else "0",
        },
    )


@pytest.mark.parametrize(
    "pattern",
    [
        "*",
        "*.py",
        "?*.py",
        "[ab].py",
        "[!b].py",
        "**",
        "**/",
        "**/*.py",
        "src/*",
        "src/**",
        "src/**/",
        "src/**/*",
        "src/*.py",
        "src/**/test_*.py",
        ".dot/*.py",
        "./*.py",
        "missing/*.py",
    ],
)
def test_fallback_matches_current_pathlib_primary_in_exact_order(
    tmp_path: Path,
    pattern: str,
) -> None:
    _fixture(tmp_path)

    completed = _run_fallback(tmp_path, pattern)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == _primary(tmp_path, pattern)


@pytest.mark.parametrize(
    "pattern",
    [
        "",
        ".",
        "./",
        "/tmp/*.py",
        "../*.py",
        "src//*.py",
        "[^a].py",
        "[z-a].py",
        "a/**/b/**/c",
        "src\\*.py",
        "é/*.py",
        "@(a|b).py",
    ],
)
def test_unsupported_fallback_patterns_fail_explicitly(pattern: str) -> None:
    assert _unsupported_glob_reason(pattern)


def test_fallback_resets_inherited_case_and_ignore_state(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "B.PY").write_text("b", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("hidden", encoding="utf-8")

    completed = _run_fallback(
        tmp_path,
        "*.py",
        extra_env={
            "BASHOPTS": "nocaseglob:dotglob",
            "GLOBIGNORE": "*",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == _primary(tmp_path, "*.py")


def test_recursive_symlink_tree_fails_instead_of_diverging(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "file.py").write_text("x", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    completed = _run_fallback(tmp_path, "**/*.py")

    assert completed.returncode == 66
    assert completed.stdout == ""
    assert "symlink directories" in completed.stderr


def test_outside_symlink_is_blocked_before_expansion(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "alias").symlink_to(outside, target_is_directory=True)
    try:
        completed = _run_fallback(tmp_path, "alias/*")
    finally:
        shutil.rmtree(outside)

    assert completed.returncode == 73
    assert completed.stdout == ""
    assert "canonical path policy" in completed.stderr


def test_more_than_500_matches_fails_instead_of_selecting_another_subset(
    tmp_path: Path,
) -> None:
    for index in range(501):
        (tmp_path / f"file-{index:03d}.txt").write_text("x", encoding="utf-8")

    completed = _run_fallback(tmp_path, "*.txt")

    assert completed.returncode == 67
    assert completed.stdout == ""
    assert "more than 500" in completed.stderr


def test_public_execute_reaches_reviewed_fallback_when_python_is_missing(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    tool = object.__new__(HarborGlobTool)
    tool._guard_environment_path = (  # type: ignore[method-assign]
        lambda *args, **kwargs: (str(tmp_path.resolve()), None)
    )
    tool._guard_environment_matches = (  # type: ignore[method-assign]
        lambda matches, **kwargs: (list(matches), None)
    )
    calls = 0

    def execute(
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                return_code=127,
                stdout="",
                stderr="python3: command not found",
            )
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout or 10,
            env={**os.environ, **(env or {})},
            check=False,
        )
        return SimpleNamespace(
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    tool._exec = execute  # type: ignore[method-assign]

    result = tool.execute("src/*.py", path=str(tmp_path))

    assert calls == 2
    assert result.success is True
    assert result.output.splitlines() == _primary(tmp_path, "src/*.py")
    assert result.metadata["engine"] == "bash-compgen"
    assert result.metadata["fallback_revalidated"] is True
