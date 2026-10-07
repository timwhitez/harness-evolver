"""Saved evidence and scripted Rust bridge checks; no live provider or Harbor."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from bench.trajectory import TrajectoryReader
from tests.test_worker_usage import worker_binary as worker_binary  # noqa: PLC0414


def events():
    return [
        {
            "type": "entrypoint_scan",
            "phase": "bootstrap",
            "turn": 0,
            "id": "scan",
            "tool": "bash",
            "args": {"command": "pwd"},
            "success": True,
            "output": "/app",
        },
        {
            "type": "tool_call",
            "turn": 1,
            "id": "local-1",
            "tool": "bash",
            "args": {"command": "check"},
            "success": True,
            "output": "ok",
        },
        {
            "type": "tool_call",
            "turn": 2,
            "id": "local-2",
            "tool": "done",
            "args": {"summary": "ready"},
            "success": True,
            "output": "ready",
        },
    ]


def trial(name="a", trace=None):
    trace = events() if trace is None else trace
    return {
        "trial_id": name,
        "task_id": "task",
        "status": "unverified",
        "score": 0,
        "task_domain": "software_engineering",
        "task_difficulty": "easy",
        "turn_count": 2,
        "trajectory": trace,
        "tool_calls": [deepcopy(e) for e in trace if isinstance(e, dict) and e.get("tool")],
        "metadata": {
            "worker_metrics_observation": {
                "schema": "worker_metrics_v1",
                "status": "complete",
                "turn_count_semantics": "rust_model_request_attempts",
                "tool_calls_semantics": "logical_outcomes_including_bootstrap",
            }
        },
    }


def save(root, data, *, jsonl=False):
    directory = root / data["trial_id"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    if jsonl:
        (directory / "trajectory.jsonl").write_text(
            "\n".join(json.dumps(e) for e in data["trajectory"]), encoding="utf-8"
        )
    return path


def compare(left, right, **kwargs):
    # Keep collection runnable against the pre-feature code for RED evidence.
    from bench.trajectory_compare import compare_trajectories

    return compare_trajectories(left, right, **kwargs)


def run_cli(memory, *args):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "harness_evolver.cli",
            "compare",
            "--memory-path",
            str(memory),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "change,position,field",
    [
        ("identical", None, None),
        ("args", 2, "args"),
        ("output", 2, "output"),
        ("insert", 2, "type"),
        ("tail", 3, "event_presence"),
    ],
)
def test_first_observable_difference(tmp_path, change, position, field):
    a, b = trial("a"), trial("b")
    if change in {"args", "output"}:
        b["trajectory"][1][change] = {"command": "changed"} if change == "args" else "changed"
    elif change == "insert":
        b["trajectory"].insert(1, {"type": "assistant_message", "turn": 1, "content": "observe"})
    elif change == "tail":
        b["trajectory"].pop()  # counts still describe the saved full execution
    report = compare(save(tmp_path, a, jsonl=True), save(tmp_path, b))
    assert report["status"] == (
        "identical" if change == "identical" else "partial" if change == "tail" else "different"
    )
    difference = report["first_difference"]
    if position is None:
        assert difference is None
    else:
        assert difference["position"] == position
        assert field in difference["fields"]
    assert report["description"] == "first observable difference in the selected projection"
    assert report["left"]["events"][1]["event_index"] == 2
    assert report["left"]["events"][1]["source"].endswith("trajectory.jsonl:2")
    assert report["right"]["events"][1]["source"].endswith("result.json#/trajectory/1")


def aggregate(name, attempts):
    combined = []
    snapshots = []
    for index, attempt in enumerate(attempts, 1):
        combined.extend(
            {**e, "_harbor_attempt_index": index, "_harbor_attempt_trial_id": attempt["trial_id"]}
            for e in attempt["trajectory"]
        )
        snapshots.append(
            {
                **{k: v for k, v in attempt.items() if k != "trajectory"},
                "trajectory_event_count": len(attempt["trajectory"]),
                "canonical_attempt_index": index,
            }
        )
    return {
        **trial(name, combined),
        "metadata": {
            "multi_attempt_aggregate": True,
            "attempt_count": len(attempts),
            "attempt_results": snapshots,
        },
    }


def test_attempt_selection_is_identity_based_and_keeps_local_ids(tmp_path):
    one, two = trial("one"), trial("two")
    two["trajectory"][1]["args"] = {"command": "other attempt"}
    # Both attempts reuse all tool IDs. They must never be joined across attempts.
    left = save(tmp_path, aggregate("left", [one, two]))
    right = save(tmp_path, aggregate("right", [two, one]), jsonl=True)
    required = compare(left, right)
    assert required["status"] == "selection_required"
    assert {a["attempt_id"] for a in required["attempts"]["left"]} == {"one", "two"}
    assert required["first_difference"] is None
    for attempt in ("one", "two"):
        selected = compare(left, right, attempt_a=attempt, attempt_b=attempt, expand=True)
        assert selected["status"] == "identical"
        assert selected["left"]["identity"]["attempt_id"] == attempt
        assert len(selected["left"]["events"]) == 3
    assert (
        compare(left, right, attempt_a="one", attempt_b="absent")["status"] == "identity_mismatch"
    )
    assert compare(left, right, attempt_a="one")["status"] == "selection_required"


def test_unique_aggregate_attempt_can_be_selected_automatically(tmp_path):
    report = compare(
        save(tmp_path, aggregate("a", [trial("one")])),
        save(tmp_path, aggregate("b", [trial("two")])),
    )
    assert report["status"] == "identical"


def test_task_and_event_identity_mismatches_are_refused(tmp_path):
    a, b = trial("a"), trial("b")
    b["task_id"] = "foreign"
    assert compare(save(tmp_path, a), save(tmp_path, b))["status"] == "identity_mismatch"
    b = aggregate("b", [trial("two")])
    b["metadata"]["attempt_results"][0]["task_id"] = "foreign"
    assert compare(save(tmp_path, a), save(tmp_path, b))["status"] == "identity_mismatch"
    b = trial("b")
    b["trajectory"][1]["task_id"] = "foreign"
    assert compare(save(tmp_path, a), save(tmp_path, b))["status"] == "identity_mismatch"


@pytest.mark.parametrize("scan_success", [True, False])
def test_repeated_tools_retries_bootstrap_and_metrics_are_separate(tmp_path, scan_success):
    trace = events()
    trace[0]["success"] = scan_success
    if not scan_success:
        trace[0]["error"] = "local policy rejection"
    trace.insert(1, {"type": "llm_error_recovery_prompt", "turn": 1, "error": "retry"})
    trace.insert(2, deepcopy(trace[0]) | {"type": "tool_call", "turn": 2})
    trace[3]["turn"] = 2
    trace[4]["turn"] = 3
    a, b = trial("a", trace), trial("b", deepcopy(trace))
    a["turn_count"] = b["turn_count"] = 3  # failed request is still a model attempt
    report = compare(save(tmp_path, a), save(tmp_path, b), expand=True)
    assert report["status"] == "identical"
    side = report["left"]
    assert [e["projection"]["turn"] for e in side["events"]] == [0, 1, 2, 2, 3]
    assert side["metrics"]["model_request_attempts"] == 3
    assert side["metrics"]["logical_tool_outcomes"] == 4
    assert side["metrics"]["provider_http_requests"] == "unknown"
    assert side["metrics"]["bootstrap_outcomes"] == 1


def test_compaction_history_is_archived_and_ids_normalized_only_at_named_paths(tmp_path):
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "random-a", "function": {"name": "bash", "arguments": '{"id":"business"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "random-a", "content": "saved output"},
    ]
    compaction = {
        "type": "context_compaction",
        "omitted_history": history,
        "before_bytes": 1000,
        "after_bytes": 400,
        "reason": "context_overflow",
        "estimate_unit": "serialized_request_bytes",
        "omitted_messages": len(history),
        "semantic_summary_generated": False,
    }
    trace = events()
    trace[1:1] = [
        compaction,
        {
            "type": "context_overflow_recovery",
            "turn": 1,
            "error_type": "ContextWindowExceededError",
            "error": "context_length_exceeded",
            "estimate_unit": "serialized_request_bytes",
            "loop_stop_condition": False,
            "time_round_token_limit_driven": False,
            "request_reduced": True,
            "rejected_bytes": 1000,
            "next_bytes": 400,
        },
        {
            "type": "completion_verification",
            "turn": 2,
            "tool": "verify",
            "args": {"command": "check"},
            "output": "ok",
            "success": True,
        },
    ]
    a, b = trial("a", trace), trial("b", deepcopy(trace))
    b["trajectory"][1]["omitted_history"][0]["tool_calls"][0]["id"] = "random-b"
    b["trajectory"][1]["omitted_history"][1]["tool_call_id"] = "random-b"
    b["trajectory"][4]["id"] = "random-tool"
    b["trajectory"][4]["duration_ms"] = 9234
    report = compare(save(tmp_path, a), save(tmp_path, b), expand=True)
    assert report["status"] == "identical"
    assert len(report["left"]["events"]) == 6
    assert report["left"]["metrics"]["logical_tool_outcomes"] == 4
    assert "turn" not in report["left"]["events"][1]["unknown_fields"]
    assert report["left"]["events"][1]["projection"]["semantic_summary_generated"] is False
    assert report["projection_version"] == "saved_trajectory_v2"
    assert (
        "context_compaction.omitted_history[].tool_calls[].id" in report["normalization"]["paths"]
    )
    # Business IDs are significant even inside arguments/omitted history.
    b["trajectory"][4]["args"]["id"] = "business-changed"
    assert compare(save(tmp_path, a), save(tmp_path, b))["status"] == "different"
    b = deepcopy(a) | {"trial_id": "b"}
    b["trajectory"][1]["omitted_history"][0]["tool_calls"][0]["function"]["arguments"] = (
        '{"id":"other"}'
    )
    assert compare(save(tmp_path, a), save(tmp_path, b))["status"] == "different"

    # Legal old compaction evidence names each unavailable field explicitly.
    old_a = deepcopy(a) | {"trial_id": "old-a"}
    old_b = deepcopy(a) | {"trial_id": "old-b"}
    for data in (old_a, old_b):
        data["trajectory"][1].pop("before_bytes")
    partial = compare(save(tmp_path, old_a), save(tmp_path, old_b))
    assert partial["status"] == "partial"
    assert "before_bytes" in partial["left"]["events"][1]["unknown_fields"]


def test_cancel_snapshot_stays_partial_and_is_not_an_execution(tmp_path):
    a = trial("a")
    a["trajectory"].append(
        {
            "type": "worker_metrics_snapshot",
            "run_id": "random-a",
            "task_id": "task",
            "worker_status": "cancelled",
            "turn_count": 2,
            "tool_calls": 3,
            "worker_metrics_observation": {"schema": "worker_metrics_v1", "status": "partial"},
        }
    )
    b = deepcopy(a) | {"trial_id": "b"}
    b["trajectory"][-1]["run_id"] = "random-b"
    report = compare(save(tmp_path, a), save(tmp_path, b), expand=True)
    assert report["status"] == "partial"
    assert report["first_difference"] is None
    assert "cancellation_or_partial_snapshot" in report["left"]["gaps"]
    assert report["left"]["metrics"]["model_request_attempts"] == 2
    assert report["left"]["metrics"]["logical_tool_outcomes"] == 3
    assert len(report["left"]["events"]) == 4


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "empty",
        "bad_json",
        "scalar",
        "bad_type",
        "bad_turn",
        "legacy",
        "unknown",
        "unfinished",
        "unattributed",
    ],
)
def test_incomplete_or_invalid_inputs_never_claim_identical(tmp_path, defect):
    a, b = trial("a"), trial("b")
    if defect == "empty":
        a["trajectory"] = b["trajectory"] = []
    elif defect in {"legacy", "unknown"}:
        a["turn_count"] = b["turn_count"] = None
        a["metadata"] = b["metadata"] = {}
    elif defect == "unfinished":
        a["status"] = b["status"] = "cancelled"
    elif defect == "unattributed":
        a, b = (
            aggregate("a", [trial("one"), trial("two")]),
            aggregate("b", [trial("one"), trial("two")]),
        )
        for data in (a, b):
            data["trajectory"][0].pop("_harbor_attempt_trial_id")
    left, right = save(tmp_path, a, jsonl=True), save(tmp_path, b, jsonl=True)
    if defect == "missing":
        left.unlink()
    elif defect in {"bad_json", "scalar", "bad_type", "bad_turn"}:
        text = {
            "bad_json": "{invalid",
            "scalar": "4",
            "bad_type": '{"type": []}',
            "bad_turn": '{"type":"tool_call","turn":false}',
        }[defect]
        (left.parent / "trajectory.jsonl").write_text(text)
    report = compare(
        left,
        right,
        attempt_a="one" if defect == "unattributed" else None,
        attempt_b="one" if defect == "unattributed" else None,
    )
    expected = (
        "input_error"
        if defect in {"missing", "bad_json", "scalar", "bad_type", "bad_turn"}
        else "partial"
    )
    assert report["status"] == expected


@pytest.mark.parametrize("text", ["{broken", "[1]", '[{"type":4}]'])
def test_strict_diagnostics_do_not_change_legacy_load(tmp_path, text):
    path = tmp_path / "trace.jsonl"
    path.write_text(text)
    before = TrajectoryReader.load(path)
    diagnostic = TrajectoryReader.diagnose(path)
    assert diagnostic["status"] == "input_error"
    assert diagnostic["errors"]
    assert TrajectoryReader.load(path) == before


def test_default_summaries_and_expansion_redact_secrets_without_hiding_differences(tmp_path):
    a, b = trial("a"), trial("b")
    for data in (a, b):
        data["trajectory"][1]["args"] = {
            "api_key": "fixture-key",
            "id": "business",
            "command": "Authorization: Bearer fixture-bearer\nCookie: session=fixture-cookie",
        }
        data["trajectory"][1]["output"] = "password=fixture-password sk-fakefixture123456789"
    b["trajectory"][1]["args"]["api_key"] = "different-key"
    left, right = save(tmp_path, a), save(tmp_path, b)
    for expand in (False, True):
        report = compare(left, right, expand=expand)
        assert report["status"] == "different"
        serialized = json.dumps(report)
        for secret in (
            "fixture-key",
            "fixture-bearer",
            "fixture-cookie",
            "fixture-password",
            "sk-fakefixture123456789",
            "different-key",
        ):
            assert secret not in serialized
        if not expand:
            assert "business" not in serialized
        else:
            assert report["left"]["events"][1]["projection"]["args"]["id"] == "business"

    for data in (a, b):
        data["trajectory"][1]["output"] = (
            'prefix {"api_key":"fixture-json-key","cookie":"fixture-json-cookie",'
            '"authorization":"Bearer fixture-json-auth"} suffix'
        )
    expanded = compare(save(tmp_path, a), save(tmp_path, b), expand=True)
    assert "fixture-json-" not in json.dumps(expanded)


def test_cli_default_json_trajectory_latest_and_read_only(tmp_path):
    memory = tmp_path / "memory"
    save(memory / "runs", trial("a"), jsonl=True)
    save(memory / "runs", trial("b"), jsonl=True)
    before = {p.relative_to(memory): p.read_bytes() for p in memory.rglob("*") if p.is_file()}
    summary = run_cli(memory, "a", "b")
    assert summary.returncode == 0
    assert "Score: 0.0000 → 0.0000" in summary.stdout
    assert "first observable" not in summary.stdout
    structured = run_cli(memory, "a", "b", "--json")
    assert structured.returncode == 0
    assert json.loads(structured.stdout)["score_delta"] == 0
    for args in (
        ("a", "b", "--trajectory", "--json"),
        ("--latest", "--trajectory", "--json"),
        ("a", "b", "--trajectory", "--expand", "--json"),
    ):
        completed = run_cli(memory, *args)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["status"] == "identical"
    text = run_cli(memory, "a", "b", "--trajectory")
    assert "first observable difference in the selected projection" in text.stdout
    assert "entrypoint_scan" in text.stdout
    assert before == {
        p.relative_to(memory): p.read_bytes() for p in memory.rglob("*") if p.is_file()
    }


def test_cli_selection_errors_and_partial_are_structured(tmp_path):
    memory = tmp_path / "memory"
    save(memory / "runs", aggregate("a", [trial("one"), trial("two")]))
    save(memory / "runs", aggregate("b", [trial("two"), trial("one")]))
    required = run_cli(memory, "a", "b", "--trajectory", "--json")
    assert required.returncode == 2
    assert json.loads(required.stdout)["status"] == "selection_required"
    assert "one" in required.stdout and "two" in required.stdout
    selected = run_cli(
        memory, "a", "b", "--trajectory", "--json", "--attempt-a", "one", "--attempt-b", "one"
    )
    assert selected.returncode == 0
    assert json.loads(selected.stdout)["status"] == "identical"
    save(memory / "runs", trial("b", []))
    partial = run_cli(memory, "a", "b", "--trajectory", "--json", "--attempt-a", "one")
    assert partial.returncode == 0
    assert json.loads(partial.stdout)["status"] == "partial"
    (memory / "runs" / "b" / "result.json").write_text('{"secret":"fixture-key",broken')
    invalid = run_cli(memory, "a", "b", "--trajectory", "--json")
    assert invalid.returncode == 2
    assert json.loads(invalid.stdout)["status"] == "input_error"
    assert "fixture-key" not in invalid.stdout + invalid.stderr
    invalid_summary = run_cli(memory, "a", "b", "--json")
    assert invalid_summary.returncode == 2
    assert json.loads(invalid_summary.stdout)["status"] == "input_error"
    assert "fixture-key" not in invalid_summary.stdout + invalid_summary.stderr
    missing_summary = run_cli(memory, "a", "absent", "--json")
    assert missing_summary.returncode == 2
    assert json.loads(missing_summary.stdout)["status"] == "input_error"


@pytest.mark.parametrize(
    "trace",
    [
        [None],
        [{"type": "worker_metrics_snapshot", "worker_metrics_observation": []}],
        [{"type": "context_compaction", "omitted_history": {}}],
        [{"type": "context_compaction", "omitted_history": [42]}],
        [{"type": "tool_call", "metadata": []}],
    ],
)
def test_malformed_nested_event_shapes_are_input_errors(tmp_path, trace):
    report = compare(save(tmp_path, trial("a", trace)), save(tmp_path, trial("b")))
    assert report["status"] == "input_error"
    assert report["errors"]


@pytest.mark.parametrize(
    "trace",
    [
        [{"type": "tool_call", "tool": "bash"}],
        [
            {
                "type": "tool_call",
                "turn": None,
                "tool": "bash",
                "args": {},
                "success": True,
                "output": "ok",
            }
        ],
        [{"tool": "bash", "turn": 1, "args": {}, "success": True, "output": "ok"}],
    ],
)
def test_legal_legacy_event_gaps_are_partial(tmp_path, trace):
    report = compare(save(tmp_path, trial("a", trace)), save(tmp_path, trial("b", trace)))
    assert report["status"] == "partial"
    assert report["first_difference"] is None
    assert report["left"]["gaps"]


def test_metric_only_difference_is_normal_comparison_and_legacy_tools_unknown(tmp_path):
    a, b = trial("a"), trial("b")
    b["turn_count"] = 3
    report = compare(save(tmp_path, a), save(tmp_path, b))
    assert report["status"] == "different"
    assert report["first_difference"] is None
    assert "model_request_attempts" in report["metric_differences"]
    a["metadata"] = b["metadata"] = {}
    report = compare(save(tmp_path, a), save(tmp_path, b))
    assert report["status"] == "partial"
    assert report["left"]["metrics"]["logical_tool_outcomes"] == "unknown"


def test_json_diagnostic_sources_blank_lines_and_invalid_json_values(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(events()))
    diagnostic = TrajectoryReader.diagnose(path)
    assert diagnostic["status"] == "complete"
    assert diagnostic["events"][1]["source"].endswith("trace.json#/1")
    path.write_text(json.dumps(events()[0]))
    singleton = TrajectoryReader.diagnose(path)
    assert singleton["status"] == "complete"
    assert singleton["events"][0]["source"] == f"{path}#"
    assert singleton["events"][0]["event_index"] == 1
    path = tmp_path / "trace.jsonl"
    path.write_text("\n" + json.dumps(events()[0]) + "\n\n" + json.dumps(events()[1]))
    diagnostic = TrajectoryReader.diagnose(path)
    assert diagnostic["events"][1]["event_index"] == 2
    assert diagnostic["events"][1]["source"].endswith("trace.jsonl:4")
    path.write_text('{"type":"tool_call","output":NaN}')
    assert TrajectoryReader.diagnose(path)["status"] == "input_error"


def test_random_ids_and_timing_do_not_mask_payload_ids_or_order(tmp_path):
    a, b = trial("a"), trial("b")
    for index, event in enumerate(b["trajectory"]):
        event["id"] = f"random-{index}"
        event["timestamp"] = "different-time"
        event["duration_ms"] = 100
    assert compare(save(tmp_path, a), save(tmp_path, b))["status"] == "identical"
    a["trajectory"][1]["args"]["timestamp"] = "payload-one"
    b["trajectory"][1]["args"]["timestamp"] = "payload-two"
    report = compare(save(tmp_path, a), save(tmp_path, b))
    assert report["status"] == "different"
    assert report["first_difference"]["position"] == 2
    b = trial("b")
    b["trajectory"][1:3] = reversed(b["trajectory"][1:3])
    assert compare(save(tmp_path, trial("a")), save(tmp_path, b))["status"] == "different"


def test_structured_cookie_and_credential_urls_remain_redacted(tmp_path):
    a = trial("a")
    a["trajectory"][1]["args"] = {
        "cookie": "fixture-cookie",
        "authorization": "fixture-auth",
        "base_url": "https://user:fixture-pass@example.invalid/fixture-path?token=fixture-query",
    }
    report = compare(
        save(tmp_path, a), save(tmp_path, deepcopy(a) | {"trial_id": "b"}), expand=True
    )
    serialized = json.dumps(report)
    for secret in (
        "fixture-cookie",
        "fixture-auth",
        "fixture-pass",
        "fixture-path",
        "fixture-query",
    ):
        assert secret not in serialized


@pytest.mark.parametrize(
    "text",
    [
        '{"type":"tool_call","type":"assistant_message"}',
        '{"type":"tool_call","output":1e400}',
        '{"type":"tool_call","output":Infinity}',
        '{"type":"tool_call","output":"\\ud800"}',
    ],
)
def test_ambiguous_or_unrepresentable_json_is_an_input_error(tmp_path, text):
    path = tmp_path / "trace.jsonl"
    path.write_text(text)
    assert TrajectoryReader.diagnose(path)["status"] == "input_error"


@pytest.mark.parametrize("change", ["status", "event_task", "event_attempt"])
def test_invalid_identity_shapes_are_input_errors(tmp_path, change):
    a = trial("a")
    if change == "status":
        a["status"] = []
    else:
        key = "task_id" if change == "event_task" else "_harbor_attempt_trial_id"
        a["trajectory"][0][key] = []
    report = compare(save(tmp_path, a), save(tmp_path, trial("b")))
    assert report["status"] == "input_error"


@pytest.mark.parametrize(
    "event,unknown",
    [
        ({"type": "context_overflow_recovery", "turn": 1}, "error_type"),
        ({"type": "worker_metrics_snapshot",
          "worker_metrics_observation": {"status": "complete"}}, "worker_status"),
        ({"type": "assistant_message", "content": "observe"}, "turn"),
        ({"type": "context_compaction", "before_bytes": 10, "after_bytes": 1,
          "omitted_history": []}, "reason"),
    ],
)
def test_review_incomplete_saved_evidence_is_partial(tmp_path, event, unknown):
    report = compare(save(tmp_path, trial("a", [event])), save(tmp_path, trial("b", [event])))
    assert report["status"] == "partial"
    assert report["first_difference"] is None
    assert report["left"]["coverage"] == "partial"
    assert report["left"]["gaps"]
    assert unknown in report["left"]["events"][0]["unknown_fields"]
    changed = deepcopy(event) | {"saved_payload": "different"}
    report = compare(save(tmp_path, trial("a", [event])), save(tmp_path, trial("b", [changed])))
    assert report["status"] == "partial"
    assert report["first_difference"]["fields"] == ["saved_payload"]


@pytest.mark.parametrize(
    "event_type,field",
    [("context_overflow_recovery", field) for field in (
        "turn", "error_type", "error", "estimate_unit", "rejected_bytes", "next_bytes",
        "request_reduced", "loop_stop_condition", "time_round_token_limit_driven",
    )] + [("context_compaction", field) for field in (
        "reason", "estimate_unit", "before_bytes", "after_bytes", "omitted_messages",
        "omitted_history", "semantic_summary_generated",
    )] + [("assistant_message", "content")],
)
def test_review_each_producer_evidence_field_is_required(tmp_path, event_type, field):
    event = {
        "context_overflow_recovery": {
            "type": "context_overflow_recovery", "turn": 1,
            "error_type": "ContextWindowExceededError", "error": "context_length_exceeded",
            "estimate_unit": "serialized_request_bytes", "rejected_bytes": 10, "next_bytes": 1,
            "request_reduced": True, "loop_stop_condition": False,
            "time_round_token_limit_driven": False,
        },
        "context_compaction": {
            "type": "context_compaction", "reason": "context_overflow",
            "estimate_unit": "serialized_request_bytes", "before_bytes": 10, "after_bytes": 1,
            "omitted_messages": 0, "omitted_history": [], "semantic_summary_generated": False,
        },
        "assistant_message": {"type": "assistant_message", "turn": 1, "content": "observe"},
    }[event_type]
    report = compare(save(tmp_path, trial("a", [event])), save(tmp_path, trial("b", [event])))
    assert report["status"] == "identical"
    event.pop(field)
    report = compare(save(tmp_path, trial("a", [event])), save(tmp_path, trial("b", [event])))
    assert report["status"] == "partial"
    assert report["left"]["gaps"]
    assert field in report["left"]["events"][0]["unknown_fields"]


def snapshot_event():
    return {
        "type": "worker_metrics_snapshot", "worker_status": "unverified",
        "turn_count": 2, "tool_calls": 0,
        "worker_metrics_observation": deepcopy(trial()["metadata"]["worker_metrics_observation"]),
    }


@pytest.mark.parametrize(
    "field,value,expected",
    [
        (None, None, "identical"),
        ("worker_status", "running", "partial"),
        ("worker_status", None, "partial"),
        ("turn_count", None, "partial"),
        ("tool_calls", None, "partial"),
        ("worker_metrics_observation.schema", None, "partial"),
        ("worker_metrics_observation.status", None, "partial"),
        ("worker_metrics_observation.turn_count_semantics", None, "partial"),
        ("worker_metrics_observation.tool_calls_semantics", None, "partial"),
    ],
)
def test_review_snapshot_requires_complete_terminal_evidence(tmp_path, field, value, expected):
    event = snapshot_event()
    if field:
        keys = field.split(".")
        container = event if len(keys) == 1 else event[keys[0]]
        if value is None:
            container.pop(keys[-1])
        else:
            container[keys[-1]] = value
    report = compare(save(tmp_path, trial("a", [event])), save(tmp_path, trial("b", [event])))
    assert report["status"] == expected
    assert report["left"]["metrics"]["model_request_attempts"] == 2
    assert report["left"]["metrics"]["logical_tool_outcomes"] == 0
    assert "turn" not in report["left"]["events"][0]["unknown_fields"]
    if expected == "partial":
        assert report["left"]["gaps"]
        if value is None:
            assert field in report["left"]["events"][0]["unknown_fields"]


@pytest.mark.parametrize(
    "event",
    [
        {"type": "worker_metrics_snapshot", "worker_status": "unverified",
         "turn_count": [], "tool_calls": {},
         "worker_metrics_observation": {"schema": "worker_metrics_v1", "status": "complete"}},
        {"type": "context_compaction", "reason": "context_overflow", "before_bytes": 10,
         "after_bytes": 1, "semantic_summary_generated": False,
         "omitted_history": [{"role": "assistant", "tool_calls": 42}]},
    ],
)
def test_review_malformed_saved_evidence_is_input_error(tmp_path, event):
    path = save(tmp_path, trial("a", [event]), jsonl=True)
    trace_path = path.parent / "trajectory.jsonl"
    assert TrajectoryReader.load(trace_path) == [event]
    assert TrajectoryReader.diagnose(trace_path)["status"] == "input_error"
    assert compare(path, save(tmp_path, trial("b", [event])))["status"] == "input_error"
    assert TrajectoryReader.load(trace_path) == [event]


@pytest.mark.parametrize(
    "field,value",
    [
        ("turn_count", []), ("turn_count", True), ("turn_count", -1),
        ("tool_calls", {}), ("tool_calls", False), ("tool_calls", -1),
        ("worker_status", []), ("worker_verified", "false"),
        ("worker_metrics_observation.schema", []),
        ("worker_metrics_observation.status", {}),
        ("worker_metrics_observation.turn_count_semantics", 1),
        ("worker_metrics_observation.tool_calls_semantics", False),
    ],
)
def test_review_snapshot_known_types_are_strict(tmp_path, field, value):
    event = snapshot_event()
    keys = field.split(".")
    container = event if len(keys) == 1 else event[keys[0]]
    container[keys[-1]] = value
    assert compare(save(tmp_path, trial("a", [event])), save(tmp_path, trial("b", [event])))[
        "status"
    ] == "input_error"


@pytest.mark.parametrize(
    "calls",
    [42, {}, "calls", [42], [{"id": []}], [{"function": 42}],
     [{"function": {"name": []}}], [{"function": {"arguments": {}}}]],
)
def test_review_archived_tool_call_types_are_strict(tmp_path, calls):
    event = {"type": "context_compaction",
             "omitted_history": [{"role": "assistant", "tool_calls": calls}]}
    assert compare(save(tmp_path, trial("a", [event])), save(tmp_path, trial("b", [event])))[
        "status"
    ] == "input_error"


def test_review_snapshot_null_counters_are_partial_and_payload_types_are_untouched(tmp_path):
    event = snapshot_event() | {"turn_count": None, "tool_calls": None}
    report = compare(save(tmp_path, trial("a", [event])), save(tmp_path, trial("b", [event])))
    assert report["status"] == "partial"
    assert {"turn_count", "tool_calls"} <= set(report["left"]["events"][0]["unknown_fields"])
    trace = events()
    trace[1]["args"] = {"worker_status": [], "tool_calls": 42,
                        "worker_metrics_observation": {"schema": []}}
    report = compare(save(tmp_path, trial("a", trace)), save(tmp_path, trial("b", trace)))
    assert report["status"] == "identical"


@pytest.mark.parametrize("trial_count", [0, 1])
@pytest.mark.parametrize("trajectory", [False, True])
def test_review_latest_json_requires_two_trials(tmp_path, trial_count, trajectory):
    memory = tmp_path / "memory"
    if trial_count:
        save(memory / "runs", trial("a"))
    args = ("--latest", "--json") + (("--trajectory",) if trajectory else ())
    completed = run_cli(memory, *args)
    assert completed.returncode == 2
    diagnostic = json.loads(completed.stdout)
    assert diagnostic["status"] == "selection_required"
    assert diagnostic["errors"] == ["Need at least 2 trials to compare"]
    legacy = run_cli(memory, "--latest", *(("--trajectory",) if trajectory else ()))
    assert legacy.returncode == 0
    assert legacy.stdout == "Need at least 2 trials to compare\n"


@pytest.mark.parametrize(
    "event_type,field,bad",
    [
        ("context_compaction", "before_bytes", []),
        ("context_compaction", "reason", []),
        ("context_compaction", "omitted_messages", {}),
        ("context_compaction", "semantic_summary_generated", "false"),
        ("context_overflow_recovery", "error_type", {}),
        ("context_overflow_recovery", "rejected_bytes", []),
        ("context_overflow_recovery", "request_reduced", "yes"),
    ],
)
def test_review2_malformed_producer_fields_are_input_errors(tmp_path, event_type, field, bad):
    event = {
        "context_compaction": {
            "type": event_type, "reason": "context_overflow",
            "estimate_unit": "serialized_request_bytes", "before_bytes": 1000,
            "after_bytes": 500, "omitted_messages": 1,
            "omitted_history": [{"role": "assistant", "content": "old plan"}],
            "semantic_summary_generated": False,
        },
        "context_overflow_recovery": {
            "type": event_type, "turn": 2, "error_type": "ContextWindowExceededError",
            "error": "scripted overflow", "estimate_unit": "serialized_request_bytes",
            "rejected_bytes": 1000, "next_bytes": 500, "request_reduced": True,
            "loop_stop_condition": False, "time_round_token_limit_driven": False,
        },
    }[event_type]
    event[field] = bad
    left = save(tmp_path, trial("a", [event]), jsonl=True)
    assert TrajectoryReader.load(left.parent / "trajectory.jsonl") == [event]
    assert TrajectoryReader.diagnose(left.parent / "trajectory.jsonl")["status"] == "input_error"
    assert compare(left, save(tmp_path, trial("b", [event])))["status"] == "input_error"


@pytest.mark.parametrize("different_transport_id", [False, True])
def test_review2_real_rust_overflow_delivery_ids(
    worker_binary, tmp_path, monkeypatch, different_transport_id,
):
    from types import SimpleNamespace

    import litellm

    from harness.tools.base import ToolResult
    from tests.test_worker_context_overflow import tool_response
    from tests.test_worker_tool_delivery import setup_worker

    paths = []
    for name, transport_id in (
        ("a", "call-random-1111"),
        ("b", "call-random-2222" if different_transport_id else "call-random-1111"),
    ):
        worker, dispatches, _ = setup_worker(
            worker_binary, monkeypatch, {"semantic-key": ToolResult(True, "visible result " * 400)},
        )
        requests = []

        def completion(requests=requests, transport_id=transport_id, **kwargs):
            requests.append(deepcopy(kwargs))
            if len(requests) == 1:
                return SimpleNamespace(usage=None, choices=[SimpleNamespace(
                    message=SimpleNamespace(content="", reasoning_content=None, tool_calls=[
                        SimpleNamespace(id=transport_id, function=SimpleNamespace(
                            name="read", arguments=json.dumps({"key": "semantic-key"}),
                        )),
                    ]),
                )])
            if len(requests) == 2:
                raise litellm.ContextWindowExceededError("scripted overflow", "mock", "openai")
            assert len(requests) == 3
            return tool_response()

        monkeypatch.setattr("bench.agent.litellm.completion", completion)
        result = worker.run("Inspect visible inputs.", {"task_id": "task-a"})
        assert result.status.value == "unverified"
        assert result.turn_count == 3
        assert len(result.tool_calls) == 2
        assert dispatches == [("read", "semantic-key")]
        types = [event["type"] for event in result.trajectory]
        assert "context_overflow_recovery" in types
        delivery = next(event for event in result.trajectory
                        if event["type"] == "tool_result_delivery_incomplete")
        assert delivery["tool_call_ids"] == [transport_id]
        paths.append(save(tmp_path, result.model_dump(mode="json") | {"trial_id": name}, jsonl=True))
    report = compare(*paths, expand=True)
    assert report["left"]["coverage"] == report["right"]["coverage"] == "complete"
    assert report["status"] == "identical", report["first_difference"]


def typed_event(event_type, path, value):
    event = {"type": event_type} if event_type else {}
    container = event
    parts = path.split(".")
    for part in parts[:-1]:
        if part.endswith("[]"):
            container[part[:-2]] = [{}]
            container = container[part[:-2]][0]
        else:
            container[part] = {}
            container = container[part]
    container[parts[-1]] = value
    return event


def schema_fields():
    from bench.trajectory import COMMON_EVENT_FIELDS, EVENT_SCHEMAS

    return [(event_type, path, kind)
            for event_type, fields in [(None, COMMON_EVENT_FIELDS), *EVENT_SCHEMAS.items()]
            for path, kind in fields.items()]


@pytest.mark.parametrize("event_type,path,kind", schema_fields())
def test_review2_every_declared_field_is_typed(event_type, path, kind):
    from bench.trajectory import diagnose_events

    good = {
        "string": "value", "transport_id": "call-random", "transport_ids": ["call-random"],
        "integer": 0, "number": 0.5, "boolean": False,
        "strings": ["value"], "integers": [0], "object": {}, "objects": [{}],
    }[kind.removeprefix("nullable_")]
    assert diagnose_events([typed_event(event_type, path, good)], "fixture")["status"] == "complete"
    bad_values = ["invalid"] if isinstance(good, (dict, list, bool)) else [{}]
    if kind.removeprefix("nullable_") in {"integer", "number", "integers"}:
        bad_values += [True, -1, 1.5] if kind.endswith("integer") else [True, -1]
    if isinstance(good, list):
        bad_values.append([{}] if kind != "objects" else ["invalid"])
    for bad in bad_values:
        diagnostic = diagnose_events([typed_event(event_type, path, bad)], "fixture")
        assert diagnostic["status"] == "input_error", (event_type, path, bad)
    if kind.startswith("nullable_"):
        assert diagnose_events([typed_event(event_type, path, None)], "fixture")["status"] == "complete"
    # Missing fields are legal historical input, never a parsing error.
    assert diagnose_events([{"type": event_type} if event_type else {}], "fixture")[
        "status"
    ] == "complete"


def test_interpreted_event_names_still_exist_in_current_producers():
    import re
    from pathlib import Path

    from bench.trajectory import INTERPRETED_EVENT_TYPES

    root = Path(__file__).resolve().parents[1]
    rust = (root / "crates/hl-worker-core/src/main.rs").read_text().split("\nmod tests {")[0]
    producer_names = set(re.findall(r'"type"\s*:\s*"([^"]+)"', rust))
    producer_names.update(re.findall(
        r'trajectory_event\.insert\("type"\.to_string\(\), json!\("([^"]+)"\)', rust,
    ))
    for filename in ("bench/_agent_bridge.py", "bench/_harbor_adapter_issue16_base.py"):
        producer_names.update(re.findall(r'"type"\s*:\s*"([^"]+)"', (root / filename).read_text()))
    # Only a rename/removal can disable interpretation. New events and fields
    # must not impose schema work on the Worker/HL loop.
    assert INTERPRETED_EVENT_TYPES <= producer_names, INTERPRETED_EVENT_TYPES - producer_names


@pytest.mark.parametrize("event_type", ["application_event", "failed_verification_completion_gate"])
@pytest.mark.parametrize("field,value", [
    ("reason", {"custom": [None, False, 1]}),
    ("turn", 999),
    ("id", "future-transport-id"),
    ("tool_call_ids", ["future-transport-id"]),
    ("timestamp", "other-time"),
    ("duration_ms", 100),
])
def test_uninterpreted_events_compare_arbitrary_fields_verbatim(tmp_path, event_type, field, value):
    from bench.trajectory import diagnose_events

    event = {"type": event_type, "reason": [], "tool": {}, "turn": "custom",
             "metadata": [], "omitted_history": {}, "success": "application-value",
             "id": {"custom": True}, "tool_call_ids": 42, "timestamp": ["custom"],
             "duration_ms": {"custom": True}}
    assert diagnose_events([event], "fixture")["status"] == "complete"
    left = save(tmp_path, trial("a", [event]))
    right = save(tmp_path, trial("b", [event]))
    report = compare(left, right, expand=True)
    assert report["status"] == "identical"
    assert report["left"]["events"][0]["projection"] == event
    changed = event | {field: value}
    report = compare(left, save(tmp_path, trial("b", [changed])), expand=True)
    assert report["status"] == "different"
    assert report["first_difference"]["fields"] == [field]
    assert report["right"]["events"][0]["projection"] == changed


def test_interpreted_event_undeclared_fields_are_untyped_raw_payload(tmp_path):
    trace = events()
    trace[1]["future_field"] = {"tool_call_id": [], "success": "custom"}
    left = save(tmp_path, trial("a", trace))
    report = compare(left, save(tmp_path, trial("b", trace)), expand=True)
    assert report["status"] == "identical"
    assert report["left"]["events"][1]["projection"]["future_field"] == trace[1]["future_field"]
    trace[1]["future_field"]["tool_call_id"] = "future-transport-id"
    report = compare(left, save(tmp_path, trial("b", trace)))
    assert report["status"] == "different"
    assert report["first_difference"]["fields"] == ["future_field"]


def test_real_cancellation_event_keeps_coverage_partial(tmp_path):
    trace = events() + [{"type": "rust_worker_core_cancelled", "reason": "cancelled by host",
                         "loop_stop_condition": False, "timeout_seconds_stop_condition": False}]
    report = compare(save(tmp_path, trial("a", trace)), save(tmp_path, trial("b", trace)))
    assert report["status"] == "partial"
    assert report["first_difference"] is None
    assert "cancellation_or_partial_snapshot" in report["left"]["gaps"]


def test_review2_id_mapping_preserves_order_aliases_and_attempt_scope(tmp_path):
    trace = [
        {"type": "context_compaction", "reason": "context_overflow",
         "estimate_unit": "serialized_request_bytes", "before_bytes": 1000, "after_bytes": 500,
         "omitted_messages": 3, "semantic_summary_generated": False, "omitted_history": [
             {"role": "assistant", "tool_calls": [
                 {"id": "random-a", "function": {"name": "read", "arguments": '{"id":"business"}'}},
                 {"id": "random-b", "function": {"name": "read", "arguments": "{}"}},
             ]},
             {"role": "tool", "tool_call_id": "random-b", "content": "second"},
             {"role": "tool", "tool_call_id": "random-a", "content": "first"},
         ]},
        {"type": "tool_result_delivery_incomplete", "turn": 2,
         "tool_call_ids": ["random-b", "random-a", "random-b"], "reason": "context_overflow",
         "estimate_unit": "serialized_request_bytes", "target_bytes": 500,
         "before_bytes": 1000, "after_bytes": 500},
    ]
    changed = deepcopy(trace)
    changed[0]["omitted_history"][0]["tool_calls"][0]["id"] = "other-a"
    changed[0]["omitted_history"][0]["tool_calls"][1]["id"] = "other-b"
    changed[0]["omitted_history"][1]["tool_call_id"] = "other-b"
    changed[0]["omitted_history"][2]["tool_call_id"] = "other-a"
    changed[1]["tool_call_ids"] = ["other-b", "other-a", "other-b"]
    left, right = save(tmp_path, trial("a", trace)), save(tmp_path, trial("b", changed))
    report = compare(left, right, expand=True)
    assert report["status"] == "identical"
    archive = report["left"]["events"][0]["projection"]["omitted_history"]
    assert [call["id"] for call in archive[0]["tool_calls"]] == ["local-1", "local-2"]
    assert [message["tool_call_id"] for message in archive[1:]] == ["local-2", "local-1"]
    assert report["left"]["events"][1]["projection"]["tool_call_ids"] == [
        "local-2", "local-1", "local-2",
    ]
    assert archive[0]["tool_calls"][0]["function"]["arguments"] == '{"id":"business"}'
    assert "tool_result_delivery_incomplete.tool_call_ids[]" in report["normalization"]["paths"]
    for refs in (["other-a", "other-b", "other-b"], ["other-b", "other-a", "unrelated"]):
        changed[1]["tool_call_ids"] = refs
        assert compare(left, save(tmp_path, trial("b", changed)))["status"] == "different"
    # Unselected attempts must not populate the selected attempt's mapping.
    selected = trial("selected", trace)
    earlier = trial("earlier", [{"type": "tool_result_delivery_incomplete",
                                 "turn": 1, "tool_call_ids": ["random-b", "random-a"]}])
    aggregate_path = save(tmp_path, aggregate("aggregate", [earlier, selected]))
    report = compare(aggregate_path, left, attempt_a="selected", expand=True)
    assert report["status"] == "identical"
    assert report["left"]["events"][1]["projection"]["tool_call_ids"] == [
        "local-2", "local-1", "local-2",
    ]


@pytest.mark.parametrize(
    "event_type,path,kind",
    [field for field in schema_fields() if field[2] in {"transport_id", "transport_ids"}],
)
def test_review2_every_declared_transport_path_is_normalized(event_type, path, kind):
    from bench.trajectory import event_field_locations
    from bench.trajectory_compare import _normalize

    value = ["random-a", "random-b", "random-a"] if kind == "transport_ids" else "random-a"
    event = typed_event(event_type, path, value)
    event["application_payload"] = {"id": "random-a", "tool_call_ids": ["random-b"]}
    ids = {}
    normalized = _normalize(event, ids)
    container, key = event_field_locations(normalized, path)[0]
    assert container[key] == (["local-1", "local-2", "local-1"]
                              if kind == "transport_ids" else "local-1")
    assert ids["random-a"] == "local-1"
    assert normalized["application_payload"] == event["application_payload"]
    assert event_field_locations(event, path)[0][0][key] == value
