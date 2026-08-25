from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from bench import _harbor_glob_fallback as historical
from bench._harbor_glob_literal_root import (
    HarborGlobTool as LiteralRootHarborGlobTool,
    _GLOB_FALLBACK_SCRIPT,
)
from bench.harbor_adapter import HarborGlobTool


pytestmark = pytest.mark.skipif(
    os.name != "posix"
    or shutil.which("bash") is None
    or shutil.which("find") is None,
    reason="Python-free canonical glob fallback requires POSIX Bash/find",
)


def _primary(root: Path, pattern: str) -> list[str]:
    return sorted({str(candidate.resolve()) for candidate in root.glob(pattern)})


def _run_fallback(root: Path, pattern: str) -> subprocess.CompletedProcess[str]:
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
            "HL_ROOT": str(root.resolve()),
            "HL_PATTERN": pattern,
            "HL_RECURSIVE": "1" if recursive else "0",
        },
    )


@pytest.mark.parametrize(
    "root_name",
    [
        "root[1]",
        "root]literal",
        "root*literal",
        "root?literal",
        r"root\literal",
    ],
)
def test_fallback_treats_authorized_root_as_literal_pattern_text(
    tmp_path: Path,
    root_name: str,
) -> None:
    root = tmp_path / root_name
    root.mkdir()
    (root / "a.py").write_text("a", encoding="utf-8")
    (root / "b.txt").write_text("b", encoding="utf-8")

    completed = _run_fallback(root, "*.py")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == _primary(root, "*.py")


def test_public_registry_exports_literal_root_glob_tool() -> None:
    assert HarborGlobTool is LiteralRootHarborGlobTool


def test_public_missing_python_path_uses_literal_root_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root[1]"
    root.mkdir()
    (root / "a.py").write_text("a", encoding="utf-8")
    tool = object.__new__(HarborGlobTool)
    tool._guard_environment_path = (  # type: ignore[method-assign]
        lambda *args, **kwargs: (str(root.resolve()), None)
    )
    tool._guard_environment_matches = (  # type: ignore[method-assign]
        lambda matches, **kwargs: (list(matches), None)
    )
    commands: list[str] = []

    def fake_base_exec(
        self: object,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> SimpleNamespace:
        commands.append(command)
        if len(commands) == 1:
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

    monkeypatch.setattr(
        historical.HarborGlobTool,
        "_exec",
        fake_base_exec,
        raising=False,
    )

    result = tool.execute("*.py", path=str(root))

    assert len(commands) == 2
    assert commands[1] != historical._OLD_FALLBACK_COMMAND if hasattr(
        historical, "_OLD_FALLBACK_COMMAND"
    ) else True
    assert result.success is True
    assert result.output.splitlines() == _primary(root, "*.py")
    assert result.metadata["engine"] == "bash-compgen"
    assert result.metadata["fallback_revalidated"] is True
