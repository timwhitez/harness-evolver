"""TrajectoryReader — parse ATIF trajectory files from Harbor.

Extracts failure patterns, tool call sequences, and timing data
from agent execution trajectories for meta-agent analysis.

Saved comparison validates only COMMON fields and events the projection interprets.
Other event types and undeclared fields are untyped raw payload, compared verbatim
(JSON object key order ignored). New transport IDs outside declared paths remain
significant: they can conservatively produce different, never false identical.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Strict contract only for events interpreted by the saved comparison projection.
# New producer events/fields need no schema update; load() does not use this table.
# Fields are optional: absence remains unknown; only supplied values are typed.
# nullable_* covers actual producer Options and historical unknown counters.
# transport_id(s) marks ONLY declared run-specific references, never payload IDs.
EVENT_SCHEMA_VERSION = "saved_trajectory_v2"


def _fields(**groups: str) -> dict[str, str]:
    """Compact spelling of the explicit path -> type table below."""
    return {path: kind for kind, paths in groups.items() for path in paths.split()}


COMMON_EVENT_FIELDS = _fields(
    string="type task_id _harbor_attempt_trial_id",
    integer="_harbor_attempt_index",
)

EVENT_SCHEMAS = {
    "ephemeral_tool_output_pruned": _fields(
        nullable_integer="turn",
        integer="pruned_messages",
    ),
    "context_compaction": _fields(
        string=(
            "reason estimate_unit omitted_history[].role omitted_history[].name "
            "omitted_history[].reasoning_content omitted_history[].tool_calls[].type "
            "omitted_history[].tool_calls[].function.name "
            "omitted_history[].tool_calls[].function.arguments"
        ),
        integer="before_bytes after_bytes omitted_messages",
        objects="omitted_history omitted_history[].tool_calls",
        boolean="semantic_summary_generated",
        nullable_string="omitted_history[].content",
        transport_id=(
            "omitted_history[].tool_call_id omitted_history[].tool_calls[].id "
            "omitted_history[].tool_calls[].tool_call_id"
        ),
        object="omitted_history[].tool_calls[].function",
    ),
    "tool_result_delivery_incomplete": _fields(
        nullable_integer="turn",
        transport_ids="tool_call_ids",
        string="reason estimate_unit",
        integer="target_bytes before_bytes after_bytes",
    ),
    "context_overflow_recovery": _fields(
        nullable_integer="turn",
        string="error_type error estimate_unit",
        integer="rejected_bytes next_bytes",
        boolean="request_reduced loop_stop_condition time_round_token_limit_driven",
    ),
    "tool_call": _fields(
        nullable_integer="turn",
        string="tool output error id tool_call_id timestamp",
        object="args metadata",
        boolean="success",
        number="duration duration_ms",
    ),
    "assistant_message": _fields(
        nullable_integer="turn",
        string="content",
    ),
    "entrypoint_scan": _fields(
        string="id phase tool output error tool_call_id timestamp",
        nullable_integer="turn",
        object="args metadata",
        boolean=(
            "success entrypoint_scan_loop_stop_condition master_loop_stop_condition "
            "sub_agent_loop_stop_condition time_round_limit_stop_condition"
        ),
        number="duration duration_ms",
        integer="operation_timeout_seconds_audit_only",
    ),
    "llm_error_recovery_prompt": _fields(
        nullable_integer="turn",
        string="error_type error",
        boolean=(
            "loop_stop_condition time_round_token_limit_driven max_turns_stop_condition "
            "timeout_seconds_stop_condition round_limit_stop_condition "
            "token_budget_stop_condition provider_error_stop_condition"
        ),
    ),
    "completion_verification": _fields(
        nullable_integer="turn",
        string="tool output error id tool_call_id timestamp",
        object="args",
        boolean="success",
        number="duration duration_ms",
    ),
    "rust_worker_core_cancelled": _fields(
        string="reason",
        boolean="loop_stop_condition timeout_seconds_stop_condition",
    ),
    "worker_metrics_snapshot": _fields(
        string=(
            "run_id task_id worker_status worker_metrics_observation.schema "
            "worker_metrics_observation.status worker_metrics_observation.turn_count_semantics "
            "worker_metrics_observation.tool_calls_semantics model token_usage_observation.schema "
            "token_usage_observation.status"
        ),
        boolean="worker_verified",
        nullable_integer="turn_count tool_calls max_turns_audit_only",
        object="worker_metrics_observation token_usage_observation",
        integer="tool_timeout_seconds token_usage_observation.calls",
        strings="error_log token_usage_observation.unknown_fields token_usage_observation.diagnostics",
    ),
}
INTERPRETED_EVENT_TYPES = frozenset(EVENT_SCHEMAS)


class TrajectoryReader:
    """Parse and analyze agent execution trajectories."""

    @staticmethod
    def load(trajectory_path: Path) -> list[dict[str, Any]]:
        """Load a trajectory file (JSONL or JSON)."""
        if not trajectory_path.exists():
            return []

        if trajectory_path.suffix == ".jsonl":
            events = []
            with open(trajectory_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return events

        if trajectory_path.suffix == ".json":
            try:
                data = json.loads(trajectory_path.read_text())
                return data if isinstance(data, list) else [data]
            except (json.JSONDecodeError, FileNotFoundError):
                return []

        return []

    @staticmethod
    def diagnose(trajectory_path: Path) -> dict[str, Any]:
        """Strict read-only diagnostics; load() retains its tolerant semantics."""
        return _diagnose(trajectory_path)

    @staticmethod
    def extract_errors(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract error events from a trajectory."""
        errors = []
        for event in trajectory:
            if event.get("type") == "error":
                errors.append(event)
            elif event.get("success") is False:
                errors.append(event)
            elif "error" in event.get("output", "").lower():
                errors.append(event)
        return errors

    @staticmethod
    def extract_tool_sequence(
        trajectory: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract the sequence of tool calls from a trajectory."""
        tools = []
        for event in trajectory:
            if "tool" in event or "function" in event:
                tools.append(event)
        return tools

    @staticmethod
    def extract_failure_patterns(
        trajectory: list[dict[str, Any]],
    ) -> list[str]:
        """Extract recurring failure patterns from a trajectory.

        These patterns are fed to the meta-agent for root cause analysis.
        """
        patterns: list[str] = []
        errors = TrajectoryReader.extract_errors(trajectory)

        for error in errors:
            msg = str(error.get("error", error.get("output", "")))
            if "command not found" in msg.lower():
                patterns.append("command_not_found")
            elif "permission denied" in msg.lower():
                patterns.append("permission_denied")
            elif "no such file" in msg.lower():
                patterns.append("file_not_found")
            elif "syntax error" in msg.lower():
                patterns.append("syntax_error")
            elif "timeout" in msg.lower():
                patterns.append("timeout")

        return list(set(patterns))

    @staticmethod
    def summarize_timing(
        trajectory: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Summarize timing data from a trajectory."""
        tool_times: dict[str, list[float]] = {}
        for event in trajectory:
            tool_name = event.get("tool", event.get("function", ""))
            duration = event.get("duration_ms", event.get("duration", 0))
            if tool_name and duration:
                tool_times.setdefault(tool_name, []).append(float(duration))

        return {
            tool: sum(times) / len(times)
            for tool, times in tool_times.items()
        }


def diagnose_events(events: Any, source: str, *, lines: list[int] | None = None) -> dict[str, Any]:
    """Strict diagnostic boundary, separate from the legacy tolerant loader."""
    if not isinstance(events, list):
        return {"status": "input_error", "events": [], "errors": [f"{source}: expected event list"]}
    records = []
    errors = []
    for index, event in enumerate(events):
        reference = f"{source}:{lines[index]}" if lines else f"{source}/{index}"
        invalid = not isinstance(event, dict)
        if not invalid:
            fields = COMMON_EVENT_FIELDS | EVENT_SCHEMAS.get(event.get("type"), {}) \
                if isinstance(event.get("type"), str) else COMMON_EVENT_FIELDS
            invalid = (
                ("type" in event and not event["type"])
                or any(not _valid_field(container[key], kind)
                       for path, kind in fields.items()
                       for container, key in event_field_locations(event, path))
            )
        if invalid:
            errors.append(f"{reference}: invalid event shape")
        else:
            records.append({"event_index": index + 1, "source": reference, "event": event})
    return {"status": "input_error" if errors else "complete" if records else "missing",
            "events": records, "errors": errors}


def event_field_locations(value: Any, path: str) -> list[tuple[dict[str, Any], str]]:
    """Locate supplied declared paths; parent/container types are validated separately."""
    head, separator, tail = path.partition(".")
    key = head.removesuffix("[]")
    if not isinstance(value, dict) or key not in value:
        return []
    if not separator:
        return [(value, key)]
    child = value[key]
    if head.endswith("[]"):
        return [location for item in child for location in event_field_locations(item, tail)] \
            if isinstance(child, list) else []
    return event_field_locations(child, tail)


def _valid_field(value: Any, kind: str) -> bool:
    if kind.startswith("nullable_"):
        if value is None:
            return True
        kind = kind.removeprefix("nullable_")
    if kind in {"string", "transport_id"}:
        return isinstance(value, str)
    if kind == "integer":
        return type(value) is int and value >= 0
    if kind == "number":
        return (type(value) is int and value >= 0) or (
            type(value) is float and value >= 0 and math.isfinite(value)
        )
    if kind == "boolean":
        return type(value) is bool
    if kind == "object":
        return isinstance(value, dict)
    if kind in {"strings", "transport_ids", "integers", "objects"}:
        item_kind = {"strings": "string", "transport_ids": "string",
                     "integers": "integer", "objects": "object"}[kind]
        return isinstance(value, list) and all(_valid_field(item, item_kind) for item in value)
    raise ValueError(f"Unknown trajectory field type: {kind}")


def _diagnose(trajectory_path: Path) -> dict[str, Any]:
    """Parse every saved event; never silently skip malformed JSON/JSONL."""
    path = Path(trajectory_path)
    if not path.exists():
        return {"status": "missing", "events": [], "errors": []}
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            events, lines = [], []
            for number, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    try:
                        events.append(parse_saved_json(line))
                    except (ValueError, RecursionError):
                        return {"status": "input_error", "events": [],
                                "errors": [f"{path}:{number}: invalid JSON"]}
                    lines.append(number)
            return diagnose_events(events, str(path), lines=lines)
        if path.suffix == ".json":
            data = parse_saved_json(text)
            if isinstance(data, list):
                return diagnose_events(data, str(path) + "#")
            diagnostic = diagnose_events([data], str(path) + "#")
            for record in diagnostic["events"]:
                record["source"] = f"{path}#"
            if diagnostic["errors"]:
                diagnostic["errors"] = [f"{path}#: invalid event shape"]
            return diagnostic
    except (OSError, ValueError, RecursionError):
        return {"status": "input_error", "events": [], "errors": [f"{path}: unreadable or invalid JSON"]}
    return {"status": "input_error", "events": [], "errors": [f"{path}: unsupported format"]}


def parse_saved_json(text: str) -> Any:
    """Reject ambiguous keys, non-finite numbers and unencodable Unicode."""
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        data = dict(pairs)
        if len(data) != len(pairs):
            raise ValueError("duplicate JSON key")
        return data

    data = json.loads(text, object_pairs_hook=unique_object)
    json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return data
