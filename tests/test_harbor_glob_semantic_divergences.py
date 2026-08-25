"""Glob semantic-divergence tests plus conservative bracket-range coverage."""

from __future__ import annotations

import glob
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys

import pytest

from bench import _harbor_glob_issue14 as implementation

_BASE_PATH = Path(__file__).with_name("_harbor_glob_semantic_divergences_base.py")
_SPEC = spec_from_file_location("_harness_glob_semantic_divergences_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load glob semantic test base: {_BASE_PATH}")
_base = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _base
_SPEC.loader.exec_module(_base)

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


@pytest.mark.parametrize(
    "pattern",
    [
        "[a-^!]",
        "[a-!!]",
        "[a-3!]",
        "x[a--!]",
        "[z-a]",
        "[9-0]",
        "[a-cx]",
        "[a-cc-d]",
        "[]a]",
        "[a!]",
    ],
)
def test_ambiguous_or_divergent_bracket_recovery_fails_closed(pattern: str) -> None:
    reason = implementation._unsupported_glob_reason(pattern)

    assert reason
    assert "classes" in reason or "Python" in reason


@pytest.mark.parametrize(
    "pattern",
    [
        "[abc]",
        "[!abc]",
        "[a-c]",
        "[!a-c]",
        "[0-9]",
        "[A-Z]",
        "[a_]",
        "src/[ab].py",
    ],
)
def test_verified_ascii_bracket_subset_remains_supported(pattern: str) -> None:
    assert implementation._unsupported_glob_reason(pattern) == ""


@pytest.mark.parametrize(
    "pattern",
    [
        "src//*.py",
        "src///**/*.py",
        "**//x.py",
        "src//",
    ],
)
def test_repeated_separators_fail_instead_of_returning_different_spelling(
    tmp_path: Path,
    pattern: str,
) -> None:
    target = tmp_path / "src" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("data", encoding="utf-8")

    reason = implementation._unsupported_glob_reason(pattern)
    result = _local_fallback_tool()._glob_without_python(
        pattern=pattern,
        path=str(tmp_path),
    )

    assert "repeated path separators" in reason
    assert result.success is False
    assert result.metadata["glob_pattern_unsupported"] is True


def test_empty_root_fails_instead_of_becoming_dot() -> None:
    assert "empty roots" in implementation._unsupported_glob_root_reason("")

    result = _local_fallback_tool()._glob_without_python(pattern="*.py", path="")

    assert result.success is False
    assert result.metadata["glob_root_unsupported"] is True


def test_inherited_nocaseglob_does_not_change_fallback_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lower = tmp_path / "lower.py"
    upper = tmp_path / "upper.PY"
    lower.write_text("data", encoding="utf-8")
    upper.write_text("data", encoding="utf-8")
    monkeypatch.setenv("BASHOPTS", "nocaseglob")
    monkeypatch.setenv("GLOBIGNORE", "lower.py")

    result = _local_fallback_tool()._glob_without_python(
        pattern="*.py",
        path=str(tmp_path),
    )
    primary = set(
        glob.glob(os.path.join(str(tmp_path), "*.py"), recursive=True)[:500]
    )

    assert result.success is True, result.error
    assert set(result.output.splitlines()) == primary == {str(lower)}


def test_fallback_resets_all_relevant_bash_glob_state() -> None:
    script = implementation._runtime_module._GLOB_FALLBACK_SCRIPT

    assert "unset GLOBIGNORE" in script
    assert "shopt -u dotglob nocaseglob failglob extglob" in script
    assert "shopt -s globskipdots" in script
    assert script.index("shopt -u dotglob") < script.index("shopt -s globstar nullglob")
