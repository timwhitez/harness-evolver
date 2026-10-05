"""Actual Rust bootstrap requests must compose with the executable shell policy."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.agent import HLAgent
from bench.harbor_adapter import HarborShellTool
from harness.tools.done import DoneTool
from harness.tools.leaderboard_guard import prohibited_command_reason
from harness.tools.process_runner import ProcessOutcome
from harness.tools.registry import ToolRegistry
from harness.tools.shell import ShellTool
from tests import test_worker_metrics as metrics
from tests.test_worker_metrics import (
    assert_result_contract,
    response,
)

worker_binary = metrics.worker_binary  # Reuse source/packaged actual Rust fixture.


@pytest.fixture(params=["default", "harbor"])
def shell_boundary(request, monkeypatch):
    """Keep tool authorization/results real; replace only its execution boundary."""
    calls = []
    mode = {"outcome": "success"}

    def execute(command, timeout):
        calls.append((command, timeout))
        if mode["outcome"] == "timeout" and request.param == "harbor":
            raise TimeoutError("offline executor timed out")
        failed = mode["outcome"] == "failure"
        return SimpleNamespace(
            return_code=1 if failed else 0,
            stdout=("" if failed else "partial workspace output\n"
                    if mode["outcome"] == "timeout" else
                    "PWD: /workspace\nLikely entrypoints:\n./README.md\n"),
            stderr="offline execution failure" if failed else "",
        )

    if request.param == "default":
        tool = ShellTool()

        def run_argv(argv, **kwargs):
            assert argv[:4] == ["bash", "-o", "pipefail", "-c"]
            result = execute(argv[4], kwargs["timeout_seconds"])
            return ProcessOutcome(
                returncode=result.return_code,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=mode["outcome"] == "timeout",
                elapsed_ms=1.0,
                managed_process_group_terminated=mode["outcome"] == "timeout",
            )

        monkeypatch.setattr("harness.tools.shell.process_runner.run_bounded_argv", run_argv)
    else:
        environment = SimpleNamespace(exec=Mock(side_effect=AssertionError("unexpected Harbor exec")))
        loop = Mock(spec=asyncio.AbstractEventLoop)
        tool = HarborShellTool(environment=environment, loop=loop, timeout_seconds=30)
        monkeypatch.setattr(tool, "_exec", lambda command, *, timeout: execute(command, timeout))
    return tool, calls, mode


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_actual_bootstrap_composes_with_shell_policy(
    worker_binary, shell_boundary, monkeypatch, outcome,
):
    tool, calls, mode = shell_boundary
    mode["outcome"] = outcome
    monkeypatch.setenv("HL_WORKER_RUST_BIN", str(worker_binary))
    registry = ToolRegistry()
    registry.register(tool)
    registry.register(DoneTool())
    agent = HLAgent(tool_registry=registry)
    model_requests = []

    def completion(**kwargs):
        model_requests.append(kwargs)
        assert len(model_requests) == 1, "unexpected model request in offline fixture"
        return response(("done", {"summary": "fixture complete"}))

    monkeypatch.setattr("bench.agent.litellm.completion", completion)
    result = agent.run("Inspect the visible workspace", {"task_id": "bootstrap-policy-fixture"})
    bootstrap = result.tool_calls[0]
    command = bootstrap["args"]["command"]
    # On the old producer, actual history records the policy denial and calls=[];
    # this assertion is the RED, not a permissive shell stand-in.
    assert prohibited_command_reason(command) == "", bootstrap
    assert len(calls) == 1 and calls[0] == (command, 15)
    assert "pwd" in command and "ls -la" in command
    assert "find . -maxdepth 2 -type f" in command
    assert "sed -n '1,80p'" in command
    assert bootstrap["id"] == "entrypoint-scan"
    assert bootstrap["phase"] == "bootstrap" and bootstrap["turn"] == 0
    assert bootstrap["success"] is (outcome == "success")
    if outcome == "success":
        assert "./README.md" in bootstrap["output"] and not bootstrap["error"]
    elif outcome == "failure":
        assert bootstrap["error"]
        assert "offline execution failure" in bootstrap["output"]
        assert bootstrap["metadata"]["exit_code"] == 1
    else:
        assert "timed out" in bootstrap["error"]
        assert bootstrap["metadata"]["timed_out"] is True
    assert bootstrap["metadata"].get("blocked_by") != "leaderboard_integrity_guard"
    assert_result_contract(result, 1)
    assert len(result.tool_calls) == 2
    assert [call["tool"] for call in result.tool_calls] == ["bash", "done"]
    assert result.metadata["worker_metrics_observation"]["tool_calls_semantics"] == (
        "logical_outcomes_including_bootstrap"
    )
    events = [event for event in result.trajectory if event["type"] == "entrypoint_scan"]
    assert len(events) == 1 and events[0]["args"] == bootstrap["args"]
    assert events[0]["success"] == bootstrap["success"]
    assert events[0]["error"] == bootstrap["error"]
    assert events[0]["phase"] == "bootstrap" and events[0]["turn"] == 0
    assert events[0]["entrypoint_scan_loop_stop_condition"] is False


@pytest.mark.parametrize("protected", [
    "find /tests -maxdepth 2 -type f",
    "cat /solutions/reference.py",
    "cat /task.toml",
])
def test_protected_tail_still_denies_whole_command_before_execution(shell_boundary, protected):
    tool, calls, _ = shell_boundary
    command = "pwd\nls -la\n" + protected
    assert prohibited_command_reason(command)
    result = tool.execute(command=command, timeout=15)
    assert result.success is False and not result.output
    assert result.metadata["blocked_by"] == "leaderboard_integrity_guard"
    assert "Leaderboard integrity guard blocked command" in result.error
    assert result.metadata["loop_stop_condition"] is False
    assert calls == []  # Neither pwd nor ls was partially executed.


def test_model_protected_command_is_a_failed_logical_outcome_without_execution(
    worker_binary, shell_boundary, monkeypatch,
):
    tool, calls, _ = shell_boundary
    monkeypatch.setenv("HL_WORKER_RUST_BIN", str(worker_binary))
    registry = ToolRegistry()
    registry.register(tool)
    registry.register(DoneTool())
    agent = HLAgent(tool_registry=registry)
    protected = "pwd\nls -la\nfind /tests -maxdepth 2 -type f"
    script = [response(("bash", {"command": protected})), response(("done", {}))]

    def completion(**kwargs):
        assert script, "unexpected model request in offline fixture"
        return script.pop(0)

    monkeypatch.setattr("bench.agent.litellm.completion", completion)
    result = agent.run("Inspect the visible workspace", {"task_id": "bootstrap-policy-fixture"})
    assert_result_contract(result, 2)
    assert [call["tool"] for call in result.tool_calls] == ["bash", "bash", "done"]
    bootstrap, blocked, _ = result.tool_calls
    assert bootstrap["success"] is True and bootstrap["phase"] == "bootstrap"
    assert calls == [(bootstrap["args"]["command"], 15)]
    assert blocked["args"]["command"] == protected
    assert blocked["success"] is False and not blocked["output"]
    assert blocked["metadata"]["blocked_by"] == "leaderboard_integrity_guard"
    assert "Leaderboard integrity guard blocked command" in blocked["error"]
    assert blocked.get("phase") != "bootstrap" and blocked["turn"] == 1
    scans = [event for event in result.trajectory if event["type"] == "entrypoint_scan"]
    assert len(scans) == 1 and scans[0]["success"] is True
    denials = [event for event in result.trajectory if event["type"] == "tool_call"
              and event.get("tool") == "bash"]
    assert len(denials) == 1 and denials[0]["success"] is False
    assert denials[0]["metadata"]["blocked_by"] == "leaderboard_integrity_guard"
