from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from harness.tools import process_runner


def test_process_runner_facade_retains_namespace_module_reference() -> None:
    assert (
        process_runner._base.__name__
        == "harness.tools._process_runner_issue19_namespace_base"
    )
    assert process_runner._runtime is process_runner._base._base


@pytest.mark.parametrize(
    "argv",
    [
        "echo",
        b"echo",
        [],
        ["echo", 1],
        None,
        {"echo": "ok"},
        {"echo", "ok"},
        (item for item in ("echo", "ok")),
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


def test_validated_argv_consumes_a_sequence_only_once() -> None:
    class SingleIterationArgv(Sequence[str]):
        def __init__(self) -> None:
            self.iterations = 0
            self.items = ("echo", "ok")

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
            return self.items[index]

        def __iter__(self) -> Iterator[str]:
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("argv source was iterated more than once")
            yield from self.items

    source = SingleIterationArgv()

    assert process_runner._validated_argv(source) == ["echo", "ok"]
    assert source.iterations == 1
