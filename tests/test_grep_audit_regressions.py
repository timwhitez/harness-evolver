from __future__ import annotations

import io
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from bench.harbor_adapter import HarborGrepTool
from harness.tools.search import GrepTool
import harness.tools.search as search_tools


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="stable descriptor grep is POSIX-specific",
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_input_line_chars", True),
        ("max_input_line_chars", 32.5),
        ("max_input_line_chars", "32"),
    ],
)
def test_local_input_line_bound_requires_a_real_positive_integer(
    field: str,
    value: object,
) -> None:
    tool = GrepTool()
    setattr(tool, field, value)

    result = tool.execute("match", path="/path/must/not/be/touched")

    assert result.success is False
    assert result.output == ""
    assert result.error == "max_input_line_chars must be an integer >= 1"
    assert result.metadata["parameter_validation_failed"] is True


@pytest.mark.parametrize("value", [True, 2.5, "2"])
def test_local_result_bound_requires_a_real_positive_integer(value: object) -> None:
    result = GrepTool().execute(
        "match",
        path="/path/must/not/be/touched",
        max_results=value,  # type: ignore[arg-type]
    )

    assert result.success is False
    assert result.output == ""
    assert result.error == "max_results must be an integer >= 1"
    assert result.metadata["parameter_validation_failed"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_results", True),
        ("max_results", 2.5),
        ("max_match_chars", "4000"),
        ("max_input_line_chars", False),
    ],
)
def test_harbor_bounds_fail_before_environment_access(
    field: str,
    value: object,
) -> None:
    tool = object.__new__(HarborGrepTool)
    tool.max_results = 200
    tool.max_match_chars = 4000
    tool.max_input_line_chars = 1_000_000
    setattr(tool, field, value)

    def unexpected_guard(
        self: HarborGrepTool,
        *args: object,
        **kwargs: object,
    ) -> tuple[str, object]:
        raise AssertionError("invalid bounds must fail before environment access")

    tool._guard_environment_path = MethodType(unexpected_guard, tool)  # type: ignore[method-assign]

    result = tool.execute("match", "/workspace")

    assert result.success is False
    assert result.output == ""
    assert result.error == f"{field} must be an integer >= 1"
    assert result.metadata["parameter_validation_failed"] is True
    assert result.metadata["validated_parameter"] == field


def test_local_failure_count_is_exact_but_diagnostics_are_fixed_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_failures = 100

    def fake_targets(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        for index in range(total_failures):
            yield (
                Path(f"file-{index}.txt"),
                io.BytesIO(b"match\n"),
                SimpleNamespace(),
            )

    def fail_every_line(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        raise OSError("injected-" + "x" * 10_000)
        yield  # pragma: no cover - marks this function as an iterator

    monkeypatch.setattr(search_tools, "iter_stable_regular_files", fake_targets)
    monkeypatch.setattr(search_tools, "_bounded_physical_lines", fail_every_line)

    result = GrepTool()._python_grep(
        "match",
        str(tmp_path),
        None,
        10,
    )

    assert result.success is False
    assert result.metadata["read_error_count"] == total_failures
    assert result.metadata["diagnostic_sample_count"] == 5
    assert result.metadata["diagnostic_sample_limit"] == 5
    assert result.metadata["diagnostics_omitted_count"] == 95
    assert result.metadata["failure_diagnostics_bounded"] is True
    assert "95 additional failures omitted" in result.error
    assert len(result.error) < 11_000
