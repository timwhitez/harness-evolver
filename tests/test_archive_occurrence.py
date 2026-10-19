"""#161: occurrence-specific archive hints through real source/packaged workers."""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import litellm
import pytest

from harness.tools.base import ToolDef, ToolResult, ToolSchema
from hl.types import TrialStatus
from tests.test_tool_result_archive import recover, worker_setup
from tests.test_worker_context_overflow import assert_protocol, history, tool_response
from tests.test_worker_tool_delivery import round_response
from tests.test_worker_usage import worker_binary as worker_binary  # noqa: PLC0414


class OccurrenceRead(ToolDef):
    name = "occurrence_read"
    description = "Read one scripted observation; transport IDs may repeat in later rounds."

    def __init__(self):
        self.calls = []
        self.outputs = []

    def get_schema(self):
        return ToolSchema(parameters={"type": "object"}, description=self.description)

    def execute(self, **kwargs):
        key = kwargs["key"]
        ordinal = sum(previous == key for previous in self.calls)
        assert ordinal < getattr(self, "max_rounds", 2), "a completed occurrence was replayed"
        self.calls.append(key)
        marker = f"round-{ordinal}/{key}:"
        output = marker + ("A" if ordinal == 0 else "B") * (4000 - len(marker))
        self.outputs.append(output)
        return ToolResult(True, output, duration_ms=1)


@pytest.mark.parametrize("rounds", [2, 3])
def test_reused_ids_keep_each_actual_capture_after_old_round_pruning(
    worker_binary, monkeypatch, tmp_path, rounds
):
    worker, _, _, context = worker_setup(worker_binary, monkeypatch, tmp_path, True)
    tool = OccurrenceRead()
    tool.max_rounds = rounds
    worker.tool_registry.register(tool)
    assert worker.ephemeral_tool_output_max_chars == 4000
    assert worker.ephemeral_tool_output_keep_recent == 4
    original = worker._rust_worker_request

    def unseeded(instruction, context):
        payload = original(instruction, context)
        payload["initial_messages"] = history(0)
        return payload

    monkeypatch.setattr(worker, "_rust_worker_request", unseeded)
    observed = []
    ids = [f"call-{index}" for index in range(5)]

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) <= rounds:
            return round_response(ids, [tool.name] * 5)
        return tool_response()

    monkeypatch.setattr("bench.agent.litellm.completion", completion)
    result = worker.run("Inspect two rounds of visible observations.", context)
    assert result.status == TrialStatus.UNVERIFIED, result.error_log
    assert len(observed) == rounds + 1
    assert tool.calls == ids * rounds
    captures = [entry["metadata"]["tool_result_archive"] for entry in result.tool_calls
                if entry["tool"] == tool.name]
    assert len(captures) == 5 * rounds
    assert len({capture["ref"] for capture in captures}) == 5 * rounds
    first_delivery = [m for m in observed[1]["messages"] if m["role"] == "tool"]
    final = [m for m in observed[rounds]["messages"] if m["role"] == "tool"]
    assert len(first_delivery) == 5 and len(final) == 10
    offset = 5 * (rounds - 2)
    for index, message in enumerate(first_delivery):
        assert not message["content"].startswith("[pruned ")
        assert f"ref={captures[index]['ref']}" in message["content"]
    for visible_index, message in enumerate(final):
        index = offset + visible_index
        assert message["tool_call_id"] == ids[index % 5]
        assert message["content"].startswith("[pruned ") == (index < 5 * (rounds - 1))
        assert f"ref={captures[index]['ref']}, capture_status={captures[index]['status']}" in message["content"]
        assert "Capture unavailable/unknown" not in message["content"]
        assert recover(worker._tool_result_archive, captures[index]["ref"])["output"] == tool.outputs[index]
    for index, capture in enumerate(captures):
        assert recover(worker._tool_result_archive, capture["ref"])["output"] == tool.outputs[index]
    if rounds == 3:
        omitted = [m for event in result.trajectory if event["type"] == "context_compaction"
                   for m in event["omitted_history"] if m["role"] == "tool"]
        assert len(omitted) == 5
        for index, message in enumerate(omitted):
            assert f"ref={captures[index]['ref']}" in message["content"]
    assert_protocol(observed[rounds]["messages"])
    assert "__hl_archive_capture" not in json.dumps(observed)
    assert tool.calls == ids * rounds  # Archive page reads never execute environment tools.


def test_legacy_unbound_results_never_borrow_later_reused_id(
    worker_binary, monkeypatch, tmp_path
):
    worker, _, _, context = worker_setup(worker_binary, monkeypatch, tmp_path, True)
    tool = OccurrenceRead()
    worker.tool_registry.register(tool)
    ids = [f"call-{i}" for i in range(5)]
    initial = history(0)
    initial.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": key, "type": "function", "function": {
            "name": tool.name, "arguments": json.dumps({"key": key})}}
        for key in ids]})
    initial.extend({"role": "tool", "tool_call_id": key,
                    "content": "legacy-" + key + "L" * 4001} for key in ids)
    original = worker._rust_worker_request

    def request(instruction, context):
        payload = original(instruction, context)
        payload["initial_messages"] = deepcopy(initial)
        payload["thresholds"]["compaction_char_threshold"] = 10**6
        return payload

    monkeypatch.setattr(worker, "_rust_worker_request", request)
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        return round_response(ids, [tool.name] * 5) if len(observed) == 1 else tool_response()

    monkeypatch.setattr("bench.agent.litellm.completion", completion)
    result = worker.run("Inspect visible observations.", context)
    assert result.status == TrialStatus.UNVERIFIED, result.error_log
    assert len(observed) == 2 and tool.calls == ids
    captures = [x["metadata"]["tool_result_archive"] for x in result.tool_calls
                if x["tool"] == tool.name]
    old = [m for m in observed[-1]["messages"] if m["role"] == "tool"][:5]
    for message in old:
        assert message["content"].startswith("[pruned ")
        assert "Capture unavailable/unknown" in message["content"]
        assert all(capture["ref"] not in message["content"] for capture in captures)
    assert_protocol(observed[-1]["messages"])


# Each mode changes only the later occurrence; the old successful capture must
# retain its own descriptor even when the later result has no capture at all.
@pytest.mark.parametrize("mode", ["different_tool", "withheld", "unavailable", "partial_error",
                                  "archive_read", "archive_index", "transient", "overflow", "unicode_partial"])
def test_later_occurrence_cannot_pollute_old_capture(
    worker_binary, monkeypatch, tmp_path, mode
):
    worker, _, _, context = worker_setup(worker_binary, monkeypatch, tmp_path, True)
    first = OccurrenceRead()
    other = OccurrenceRead()
    other.name = "different_observation"
    worker.tool_registry.register(first)
    worker.tool_registry.register(other)
    original = worker._rust_worker_request

    def request(instruction, context):
        payload = original(instruction, context)
        payload["initial_messages"] = history(0)
        payload["thresholds"]["compaction_char_threshold"] = 10**6
        return payload

    monkeypatch.setattr(worker, "_rust_worker_request", request)
    ids = [f"call-{i}" for i in range(5)]
    original_execute = first.execute

    def execute(**kwargs):
        result = original_execute(**kwargs)
        if len(first.calls) > 5:
            if mode == "withheld":
                result.output = "Bearer abcdefghijklmnop " + result.output
            elif mode in {"partial_error", "unicode_partial"}:
                if mode == "unicode_partial":
                    result.output = "round-unicode:" + "界🙂" * 2000
                result.success, result.error = False, "controlled partial failure"
                result.metadata = {"partial_output_available": True}
        return result

    monkeypatch.setattr(first, "execute", execute)
    from bench.tool_result_archive import ToolResultArchive
    original_capture = ToolResultArchive.capture
    captures_seen = []

    def capture(self, *args, **kwargs):
        if args[1] == "done":
            return original_capture(self, *args, **kwargs)
        if mode == "unavailable" and len(captures_seen) >= 5:
            saved = {"ref": None, "status": "unknown", "reason": "archive_unavailable"}
        else:
            saved = original_capture(self, *args, **kwargs)
        captures_seen.append(saved)
        return saved

    monkeypatch.setattr(ToolResultArchive, "capture", capture)
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        step = len(observed)
        if step == 1:
            return round_response(ids, [first.name] * 5)
        if step == 2:
            if mode.startswith("archive_"):
                name = "tool_result_read" if mode == "archive_read" else "tool_result_index"
                args = [{"ref": item["ref"], "offset": 0, "limit": 16}
                        if mode == "archive_read" else {"offset": 0, "limit": 1}
                        for item in captures_seen[:5]]
                return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
                    content="", reasoning_content=None, tool_calls=[SimpleNamespace(id=key,
                        function=SimpleNamespace(name=name, arguments=json.dumps(value)))
                        for key, value in zip(ids, args, strict=True)]))])
            return round_response(ids, [other.name if mode == "different_tool" else first.name] * 5)
        if step == 3 and mode == "transient":
            raise TimeoutError("controlled local transient")
        if step == 3 and mode == "overflow":
            raise litellm.ContextWindowExceededError("controlled local overflow", "mock", "openai")
        return tool_response()

    monkeypatch.setattr("bench.agent.litellm.completion", completion)
    result = worker.run("Inspect visible observations.", context)
    assert result.status == TrialStatus.UNVERIFIED, result.error_log
    assert len(observed) == (4 if mode in {"transient", "overflow"} else 3)
    assert first.calls + other.calls == (ids if mode.startswith("archive_") else ids + ids)
    old = [m for m in observed[2]["messages"] if m["role"] == "tool"][:5]
    assert len(old) == 5
    for index, message in enumerate(old):
        assert message["content"].startswith("[pruned ")
        assert f"ref={captures_seen[index]['ref']}, capture_status=unknown" in message["content"]
        assert recover(worker._tool_result_archive, captures_seen[index]["ref"])["output"] == first.outputs[index]
    if mode == "transient":
        assert observed[2]["messages"] == observed[3]["messages"][:-1]
        note = observed[3]["messages"][-1]
        assert note["role"] == "user"
        assert note["content"].startswith("Model-provider recovery checkpoint:")
    if mode == "overflow":
        omitted = next(e["omitted_history"] for e in result.trajectory
                       if e["type"] == "context_compaction" and e["reason"] == "context_overflow")
        for index, message in enumerate(m for m in omitted if m["role"] == "tool"):
            assert f"ref={captures_seen[index]['ref']}" in message["content"]
    if mode in {"withheld", "unavailable"}:
        assert all(x["ref"] is None and x["status"] == "unknown" for x in captures_seen[5:])
    if mode in {"partial_error", "unicode_partial"}:
        assert all(x["status"] == "partial" for x in captures_seen[5:])
    for item in observed:
        assert_protocol(item["messages"])
    assert "__hl_archive_capture" not in json.dumps({"requests": observed,
                                                        "trajectory": result.trajectory})
