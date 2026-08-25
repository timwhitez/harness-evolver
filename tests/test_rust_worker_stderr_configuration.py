from __future__ import annotations

import math

import pytest

from bench.agent import HLAgent


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, "", "not-an-integer", math.inf, -math.inf, math.nan],
)
def test_invalid_stderr_tail_bound_fails_before_worker_discovery(
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = HLAgent(rust_stderr_tail_bytes=value)  # type: ignore[arg-type]

    def unexpected_worker_command() -> list[str]:
        raise AssertionError("Worker discovery must not run for an invalid capture bound")

    monkeypatch.setattr(agent, "_rust_worker_command", unexpected_worker_command)

    with pytest.raises(
        ValueError,
        match="rust_stderr_tail_bytes must be an integer >= 1",
    ):
        agent._run_rust_core("test", {"task_id": "invalid-stderr-bound"})


def test_numeric_string_stderr_tail_bound_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = HLAgent(rust_stderr_tail_bytes="4096")  # type: ignore[arg-type]

    class ExpectedStop(RuntimeError):
        pass

    def stop_after_validation() -> list[str]:
        raise ExpectedStop("validated")

    monkeypatch.setattr(agent, "_rust_worker_command", stop_after_validation)

    with pytest.raises(ExpectedStop, match="validated"):
        agent._run_rust_core("test", {"task_id": "valid-stderr-bound"})
