"""Offline metering through the real Rust JSONL worker and public Python adapter."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from bench.agent import HLAgent
from bench.harbor import HarborRunner
from bench.harbor_adapter import HLWorkerHarborAgent
from bench.runtime_resources import worker_crate_root
from harness.tools.base import ToolDef, ToolResult, ToolSchema
from harness.tools.done import DoneTool
from harness.tools.registry import ToolRegistry
from hl.types import TrialResult, TrialStatus


@pytest.fixture(scope="module", params=["source", "packaged"])
def worker_binary(request, tmp_path_factory):
    source = Path(__file__).resolve().parents[1] / "crates/hl-worker-core"
    crate = source if request.param == "source" else worker_crate_root(
        tmp_path_factory.mktemp("metrics-runtime"))
    assert (crate / "src/main.rs").read_bytes() == (source / "src/main.rs").read_bytes()
    target = tmp_path_factory.mktemp("metrics-target")
    subprocess.run(["cargo", "build", "--offline", "--locked", "--quiet",
                    "--manifest-path", str(crate / "Cargo.toml"),
                    "--target-dir", str(target)], check=True)
    return target / "debug/hl-worker-core"


class SpyTool(ToolDef):
    name = "bash"
    description = "Offline shell spy"

    def __init__(self, success=True):
        self.calls = []
        self.success = success

    def get_schema(self):
        return ToolSchema(parameters={"type": "object"}, description=self.description)

    def execute(self, **args):
        self.calls.append(args)
        return ToolResult(success=self.success, output="fixture output",
                          error="" if self.success else "fixture failure",
                          metadata={"spy": True})


def response(*calls):
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
        content="", reasoning_content=None, tool_calls=[SimpleNamespace(
            id=f"call-{index}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
            for index, (name, args) in enumerate(calls)]))])


def scripted_agent(binary, monkeypatch, script, *, scan_success=True, bash=True):
    monkeypatch.setenv("HL_WORKER_RUST_BIN", str(binary))
    spy = SpyTool(scan_success)
    registry = ToolRegistry()
    registry.register(DoneTool())
    if bash:
        registry.register(spy)
    agent = HLAgent(tool_registry=registry)
    observed = []
    def completion(**kwargs):
        observed.append(kwargs)
        if not script:
            raise RuntimeError("Insufficient Balance: offline script exhausted")
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr("bench.agent.litellm.completion", completion)
    return agent, spy, observed


def assert_result_contract(result, turns):
    assert result.turn_count == turns
    assert result.metadata["turn_count"] == turns
    assert result.score == 0 and not result.verified
    assert result.token_usage == {}  # absent usage must not become known zero
    assert TrialResult.model_validate_json(result.model_dump_json()).turn_count == turns


@pytest.mark.parametrize("scan_success", [True, False])
def test_bootstrap_outcome_in_history_once(worker_binary, monkeypatch, scan_success):
    agent, spy, observed = scripted_agent(worker_binary, monkeypatch, [
        response(("bash", {"command": "fixture command"})),
        response(("done", {"summary": "ready"}))], scan_success=scan_success)
    result = agent.run("Finish the task", {"task_id": "fixture"})
    assert len(result.tool_calls) == 3
    assert_result_contract(result, 2)
    assert len(observed) == agent.turn_count == 2
    assert len(spy.calls) == 2
    assert [call["tool"] for call in result.tool_calls] == ["bash", "bash", "done"]
    bootstrap = result.tool_calls[0]
    assert bootstrap["phase"] == "bootstrap" and bootstrap["turn"] == 0
    assert bootstrap["success"] is scan_success
    assert bootstrap["args"] == spy.calls[0]
    assert bootstrap["metadata"]["spy"] is True
    assert bootstrap["duration_ms"] >= 0
    assert len([e for e in result.trajectory if e["type"] == "entrypoint_scan"]) == 1
    assert len([e for e in result.trajectory if e["type"] == "tool_call"]) == 2


def test_failed_model_retry_and_auth_attempts_are_turns(worker_binary, monkeypatch):
    agent, spy, observed = scripted_agent(worker_binary, monkeypatch, [
        TimeoutError("offline transient"),
        response(("bash", {"command": "same fixture"})),
        response(("bash", {"command": "same fixture"})),
        RuntimeError("AuthenticationError: Invalid API key")])
    result = agent.run("Finish the task", {"task_id": "fixture"})
    assert result.status == TrialStatus.ERROR
    assert_result_contract(result, 4)
    assert len(observed) == 4 and len(spy.calls) == 3
    assert len(result.tool_calls) == 3
    assert result.metadata["provider_terminal_error"] is True


def test_without_bash_no_bootstrap_is_invented(worker_binary, monkeypatch):
    agent, spy, observed = scripted_agent(worker_binary, monkeypatch,
        [response(("done", {}))], bash=False)
    result = agent.run("Finish the task", {"task_id": "fixture"})
    assert_result_contract(result, 1)
    assert not spy.calls and len(observed) == 1
    assert [call["tool"] for call in result.tool_calls] == ["done"]


def test_cancel_without_final_keeps_observed_partial_outcomes(worker_binary, monkeypatch):
    agent, spy, observed = scripted_agent(worker_binary, monkeypatch, [])
    def cancel(**kwargs):
        observed.append(kwargs)
        agent.cancel_current_run("offline fixture cancellation")
        raise RuntimeError("AuthenticationError: Invalid API key")
    monkeypatch.setattr("bench.agent.litellm.completion", cancel)
    result = agent.run("Finish the task", {"task_id": "fixture"})
    assert result.status == TrialStatus.ERROR
    assert_result_contract(result, 1)
    assert result.metadata["worker_metrics_observation"]["status"] == "partial"
    assert len(spy.calls) == len(result.tool_calls) == 1
    assert result.tool_calls[0]["phase"] == "bootstrap"
    assert agent._active_process is None


def test_no_model_requests_and_legacy_missing_fields(monkeypatch):
    agent = HLAgent()
    monkeypatch.setattr(agent, "_rust_worker_command", lambda: (_ for _ in ()).throw(
        RuntimeError("offline startup failure")))
    result = agent.run("Finish", {"task_id": "fixture"})
    assert_result_contract(result, 0)
    assert not result.tool_calls
    assert result.metadata["worker_metrics_observation"]["status"] == "partial"
    legacy = HLAgent()._trial_result_from_rust({"status": "unverified"}, {"task_id": "fixture"})
    assert legacy.turn_count is None and legacy.metadata["turn_count"] is None
    old = TrialResult.model_validate({"trial_id": "old", "task_id": "old", "task_domain": "software_engineering",
                                     "task_difficulty": "easy", "status": "unverified"})
    assert old.turn_count is None


def test_zero_and_empty_final_are_authoritative():
    agent = HLAgent()
    agent.turn_count = 9
    agent.tool_call_history = [{"tool": "old"}]
    result = agent._trial_result_from_rust({"status": "unverified", "turn_count": 0,
        "tool_calls": [], "trajectory": []}, {"task_id": "fixture"})
    assert_result_contract(result, 0)
    assert not result.tool_calls and agent.turn_count == 0


def test_repeat_run_resets_metrics(worker_binary, monkeypatch):
    agent, spy, observed = scripted_agent(worker_binary, monkeypatch, [
        response(("done", {})), response(("done", {}))])
    first = agent.run("Finish", {"task_id": "one"})
    second = agent.run("Finish", {"task_id": "two"})
    assert_result_contract(first, 1)
    assert_result_contract(second, 1)
    assert len(first.tool_calls) == len(second.tool_calls) == 2
    assert len(spy.calls) == len(observed) == 2


def test_adapter_and_parser_preserve_turns(worker_binary, monkeypatch, tmp_path):
    agent, _, _ = scripted_agent(worker_binary, monkeypatch, [response(("done", {}))])
    result = agent.run("Finish", {"task_id": "fixture"})
    adapter = HLWorkerHarborAgent(logs_dir=tmp_path / "agent", model_name="test")
    monkeypatch.setattr(agent, "run", lambda *args: result)
    monkeypatch.setattr(adapter, "_build_agent", lambda *args: agent)
    monkeypatch.setattr(adapter, "_build_environment_registry", lambda *args: ToolRegistry())
    context = SimpleNamespace()
    asyncio.run(adapter.run("Finish", SimpleNamespace(environment_name="fixture"), context))
    assert context.metadata["turn_count"] == result.turn_count == 1
    assert context.metadata["tool_calls"] == 2
    def parse(name, items):
        job = tmp_path / name
        job.mkdir()
        (job / "result.json").write_text(json.dumps({"trial_results": [
            {"task_name": "fixture", "trial_name": f"fixture__{i}", "agent_result": item,
             "verifier_result": {"rewards": {"reward": 0.0}}} for i, item in enumerate(items)]}))
        return HarborRunner().parse_job_dir(job, task_id="fixture")
    parsed = parse("single", [vars(context)])
    assert parsed.turn_count == parsed.metadata["turn_count"] == 1
    assert parse("attempts", [vars(context), vars(context)]).turn_count == 2
    assert parse("missing", [vars(context), {}]).turn_count is None
    assert parse("legacy", [{}]).turn_count is None


@pytest.mark.parametrize("value", [None, True, -1, 1.5, "2"])
def test_invalid_payload_counter_stays_unknown(value):
    result = HLAgent()._trial_result_from_rust({"status": "unverified", "turn_count": value,
        "tool_calls": []}, {"task_id": "fixture"})
    assert result.turn_count is None and result.metadata["turn_count"] is None


def test_local_argument_rejection_is_not_a_physical_dispatch(worker_binary, monkeypatch):
    malformed = response(("bash", {}))
    malformed.choices[0].message.tool_calls[0].function.arguments = "{bad json"
    agent, spy, _ = scripted_agent(worker_binary, monkeypatch, [malformed,
        response(("done", {}))])
    result = agent.run("Finish the task", {"task_id": "fixture"})
    assert_result_contract(result, 2)
    assert len(spy.calls) == 1
    assert [c["tool"] for c in result.tool_calls] == ["bash", "bash", "done"]
    assert result.tool_calls[1]["success"] is False
    assert "Malformed JSON arguments" in result.tool_calls[1]["error"]


def test_existing_completion_verification_is_not_double_counted(worker_binary, monkeypatch):
    agent, spy, _ = scripted_agent(worker_binary, monkeypatch, [response(("done", {}))])
    original_request = agent._rust_worker_request
    def request(*args):
        payload = original_request(*args)
        payload["verification_command"] = "fixture verification"
        return payload
    monkeypatch.setattr(agent, "_rust_worker_request", request)
    result = agent.run("Finish the task", {"task_id": "fixture"})
    assert_result_contract(result, 1)
    assert len(spy.calls) == 2  # scan + existing internal verification
    assert len(result.tool_calls) == 3  # also includes logical done
    assert len([e for e in result.trajectory if e["type"] == "completion_verification"]) == 1


def test_local_policy_rejection_preserves_logical_outcome(worker_binary, monkeypatch):
    agent, spy, _ = scripted_agent(worker_binary, monkeypatch, [
        response(("grep", {"pattern": "secret", "path": "/logs/verifier"})),
        response(("done", {}))])
    result = agent.run("Finish the task", {"task_id": "fixture"})
    assert_result_contract(result, 2)
    assert len(spy.calls) == 1
    assert [c["tool"] for c in result.tool_calls] == ["bash", "grep", "done"]
    assert result.tool_calls[1]["success"] is False
    assert "Blocked hidden verifier artifact search" in result.tool_calls[1]["error"]


def test_old_worker_scan_history_does_not_claim_complete_v1_metering():
    result = HLAgent()._trial_result_from_rust({"status": "unverified", "turn_count": 3,
        "tool_calls": [], "trajectory": [{"type": "entrypoint_scan", "success": True,
                                             "output": "legacy scan"}]}, {"task_id": "fixture"})
    assert result.turn_count == 3
    observation = result.metadata["worker_metrics_observation"]
    assert observation["schema"] == "legacy" and observation["status"] == "legacy"
    assert observation["tool_calls_semantics"] == "legacy_unspecified"


def test_harbor_missing_counter_cannot_keep_complete_observation():
    from bench.worker_metrics import harbor_worker_metrics
    metrics = harbor_worker_metrics({"agent_result": {"metadata": {
        "worker_metrics_observation": {"schema": "worker_metrics_v1", "status": "complete"}}}})
    assert metrics["turn_count"] is None
    assert metrics["worker_metrics_observation"]["status"] == "unknown"
    legacy = harbor_worker_metrics({"agent_result": {"metadata": {"turn_count": 3}}})
    assert legacy["turn_count"] == 3
    assert legacy["worker_metrics_observation"]["schema"] == "legacy"



def test_cancel_after_repeated_outcomes_keeps_partial_history(worker_binary, monkeypatch):
    import bench.agent as agent_module
    agent, spy, observed = scripted_agent(worker_binary, monkeypatch, [
        response(("bash", {"command": "same fixture"})),
        response(("bash", {"command": "same fixture"}))])
    scripted_completion = agent_module.litellm.completion
    def cancel_third(**kwargs):
        if len(observed) < 2:
            return scripted_completion(**kwargs)
        observed.append(kwargs)
        agent.cancel_current_run("offline cancellation after outcomes")
        raise RuntimeError("AuthenticationError: Invalid API key")
    monkeypatch.setattr(agent_module.litellm, "completion", cancel_third)
    result = agent.run("Finish", {"task_id": "fixture"})
    assert_result_contract(result, 3)
    assert result.metadata["worker_metrics_observation"]["status"] == "partial"
    assert len(result.tool_calls) == len(spy.calls) == 3


def test_cancel_inflight_scan_does_not_invent_completed_outcome(worker_binary, monkeypatch):
    agent, spy, observed = scripted_agent(worker_binary, monkeypatch, [])
    def cancel_scan(**args):
        spy.calls.append(args)
        agent.cancel_current_run("offline in-flight scan cancellation")
        return ToolResult(success=False, output="", error="cancelled fixture")
    monkeypatch.setattr(spy, "execute", cancel_scan)
    result = agent.run("Finish", {"task_id": "fixture"})
    assert_result_contract(result, 0)
    assert result.status == TrialStatus.ERROR
    assert result.metadata["worker_metrics_observation"]["status"] == "partial"
    assert len(spy.calls) == 1 and not result.tool_calls and not observed
