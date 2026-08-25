from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from harness.tools import process_runner


@pytest.mark.parametrize(
    "argv",
    [
        "echo",
        b"echo",
        [],
        ["echo", 1],
        None,
    ],
)
def test_invalid_argv_is_rejected_before_popen(
    monkeypatch: pytest.MonkeyPatch,
    argv: object,
) -> None:
    popen_called = False

    def unexpected_popen(*args: Any, **kwargs: Any) -> None:
        nonlocal popen_called
        popen_called = True
        raise AssertionError("Popen must not run for invalid argv")

    monkeypatch.setattr(process_runner.subprocess, "Popen", unexpected_popen)

    with pytest.raises(ValueError, match="non-empty sequence of strings"):
        process_runner.run_bounded_argv(  # type: ignore[arg-type]
            argv,
            timeout_seconds=1.0,
        )

    assert popen_called is False


def test_validated_argv_is_a_detached_snapshot() -> None:
    source = ["echo", "before"]

    normalized = process_runner._validated_argv(source)
    source[1] = "after"

    assert normalized == ["echo", "before"]
    assert process_runner._validated_argv(("echo", "tuple")) == ["echo", "tuple"]


def test_validated_argv_consumes_the_source_only_once() -> None:
    class SingleIterationArgv:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self) -> Iterator[str]:
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("argv source was iterated more than once")
            yield "echo"
            yield "ok"

    source = SingleIterationArgv()

    assert process_runner._validated_argv(source) == ["echo", "ok"]
    assert source.iterations == 1
