from __future__ import annotations

import io
import math
import os
from types import MethodType

import pytest

import bench.agent as agent_module
from bench.agent import HLAgent, _validated_stderr_tail_bytes


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.5, "1.5", math.nan, math.inf, None],
)
def test_stderr_tail_bound_rejects_non_integer_or_non_positive_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="integer >= 1"):
        _validated_stderr_tail_bytes(value)


@pytest.mark.parametrize("value", [1, 1.0, "1", "+1", " 1 "])
def test_stderr_tail_bound_accepts_exact_positive_integer_forms(value: object) -> None:
    assert _validated_stderr_tail_bytes(value) == 1


class _FakeProcess:
    pid = 4242

    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.stderr = None
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


def _assert_fds_closed(descriptors: list[int]) -> None:
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_dup_failure_closes_both_pipe_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pipe = os.pipe
    descriptors: list[int] = []

    def tracked_pipe() -> tuple[int, int]:
        pair = real_pipe()
        descriptors.extend(pair)
        return pair

    monkeypatch.setattr(agent_module.os, "pipe", tracked_pipe)
    monkeypatch.setattr(
        agent_module.os,
        "dup",
        lambda _fd: (_ for _ in ()).throw(OSError("injected dup failure")),
    )
    monkeypatch.setattr(
        agent_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Popen must not run after dup failure")
        ),
    )

    with pytest.raises(OSError, match="injected dup failure"):
        HLAgent()._run_rust_core("test", {"task_id": "dup-failure"})

    assert len(descriptors) == 2
    _assert_fds_closed(descriptors)


def test_thread_start_failure_terminates_worker_and_closes_all_pipe_fds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pipe = os.pipe
    real_dup = os.dup
    descriptors: list[int] = []

    def tracked_pipe() -> tuple[int, int]:
        pair = real_pipe()
        descriptors.extend(pair)
        return pair

    def tracked_dup(fd: int) -> int:
        duplicate = real_dup(fd)
        descriptors.append(duplicate)
        return duplicate

    process = _FakeProcess()
    agent = HLAgent()
    terminated: list[_FakeProcess] = []

    def terminate(self: HLAgent, candidate: _FakeProcess) -> None:
        terminated.append(candidate)
        candidate.returncode = -9

    def worker_command(self: HLAgent) -> list[str]:
        return ["fake-worker"]

    monkeypatch.setattr(agent_module.os, "pipe", tracked_pipe)
    monkeypatch.setattr(agent_module.os, "dup", tracked_dup)
    monkeypatch.setattr(agent_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        agent_module.threading.Thread,
        "start",
        lambda self: (_ for _ in ()).throw(RuntimeError("injected thread start failure")),
    )
    agent._rust_worker_command = MethodType(worker_command, agent)  # type: ignore[method-assign]
    agent._terminate_process = MethodType(terminate, agent)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="injected thread start failure"):
        agent._run_rust_core("test", {"task_id": "thread-start-failure"})

    assert terminated == [process]
    assert process.returncode == -9
    assert len(descriptors) == 3
    _assert_fds_closed(descriptors)
