"""Read-only, attempt-scoped comparison of saved observable events.

No event alignment, replay, inferred counters, or causal interpretation. Display
redaction does not affect equality of the saved, explicitly normalized values.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
from typing import Any

from bench.trajectory import (
    EVENT_SCHEMA_VERSION,
    EVENT_SCHEMAS,
    INTERPRETED_EVENT_TYPES,
    TrajectoryReader,
    diagnose_events,
    parse_saved_json,
)
from bench.worker_metrics import valid_turn_count

PROJECTION_VERSION = EVENT_SCHEMA_VERSION
TOOL_EVENTS = {"tool_call", "entrypoint_scan", "completion_verification"}
NORMALIZATION = {
    "ignored": [
        "event._harbor_attempt_index",
        "event._harbor_attempt_trial_id",
        *[f"{event_type}.{field}" for event_type in sorted(TOOL_EVENTS)
          for field in ("timestamp", "duration", "duration_ms")],
        "worker_metrics_snapshot.run_id",
        "tool_call.id",
        "tool_call.tool_call_id",
        "entrypoint_scan.id",
        "entrypoint_scan.tool_call_id",
        "completion_verification.id",
        "completion_verification.tool_call_id",
    ],
    "paths": [
        f"{event_type}.{path}{'[]' if kind == 'transport_ids' else ''}"
        for event_type, fields in EVENT_SCHEMAS.items()
        for path, kind in fields.items()
        if kind in {"transport_id", "transport_ids"}
    ],
    "rule": "Declared transport references renamed by first occurrence within each selected attempt; payload fields retained",
    "equality": "saved values before display redaction; JSON object key order ignored",
}


class _InputError(ValueError):
    pass


class _IdentityError(ValueError):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _redact_text(text: str) -> str:
    # Campaign text patterns plus quoted credentials in JSON output fragments.
    text = re.sub(
        r'(?i)"([^"\\]*(?:api[_-]?key|authorization|bearer|credential|password|passwd|secret|token|cookie)[^"\\]*)"'
        r'\s*:\s*"(?:\\.|[^"\\])*"',
        r'"\1": "[REDACTED_SECRET]"',
        text,
    )
    text = re.sub(
        r"(?i)\b(authorization\s*[:=]\s*)(?:bearer|basic)?\s*[A-Za-z0-9._~+/=-]{8,}",
        r"\1[REDACTED_SECRET]",
        text,
    )
    text = re.sub(r"(?i)\b(cookie\s*[:=]\s*)[^\n|]{8,}", r"\1[REDACTED_SECRET]", text)
    text = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd|pwd)"
        r"\s*[:=]\s*([^\s'\"&|;]+)",
        r"\1=[REDACTED_SECRET]",
        text,
    )
    return re.sub(
        r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]+"
        r"|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b",
        "[REDACTED_SECRET]",
        text,
    )


def _safe(value: Any) -> Any:
    # Config redaction handles credentials/URLs; text follows campaign analysis.
    from harness.config import _redact_untyped_config

    def text_only(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                _redact_text(key): "<redacted>"
                if key.lower() in {"cookie", "set-cookie"}
                else text_only(child)
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [text_only(child) for child in item]
        return _redact_text(item) if isinstance(item, str) else item

    return text_only(_redact_untyped_config(value))


def _summary(value: Any) -> dict[str, Any]:
    serialized = _json(_safe(value)).encode("utf-8")
    return {
        "kind": type(value).__name__,
        "redacted_bytes": len(serialized),
        "redacted_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _load(path: Path) -> dict[str, Any]:
    try:
        data = parse_saved_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        raise _InputError(f"{path}: unreadable or invalid trial JSON") from None
    if not isinstance(data, dict):
        raise _InputError(f"{path}: expected trial object")
    for key in ("trial_id", "task_id"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise _InputError(f"{path}: missing or invalid {key}")
    if not isinstance(data.get("metadata", {}), dict):
        raise _InputError(f"{path}: invalid metadata")
    if "status" in data and not isinstance(data["status"], str):
        raise _InputError(f"{path}: invalid status")
    return data


def _attempts(data: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = data.get("metadata", {})
    snapshots = metadata.get("attempt_results")
    if snapshots is None and not metadata.get("multi_attempt_aggregate"):
        return [data]
    if not isinstance(snapshots, list) or not snapshots:
        raise _InputError("aggregate missing attempt_results")
    seen = set()
    for snapshot in snapshots:
        if (
            not isinstance(snapshot, dict)
            or not isinstance(snapshot.get("trial_id"), str)
            or not snapshot["trial_id"]
            or not isinstance(snapshot.get("task_id"), str)
        ):
            raise _InputError("invalid attempt identity")
        if snapshot["trial_id"] in seen:
            raise _InputError("duplicate attempt identity")
        seen.add(snapshot["trial_id"])
        if snapshot["task_id"] != data["task_id"]:
            raise _IdentityError("attempt task identity does not match trial")
        if "status" in snapshot and not isinstance(snapshot["status"], str):
            raise _InputError("invalid attempt status")
        if not isinstance(snapshot.get("metadata", {}), dict):
            raise _InputError("invalid attempt metadata")
    return snapshots


def _records(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    for filename in ("trajectory.jsonl", "trajectory.json"):
        candidate = path.parent / filename
        if candidate.exists():
            return TrajectoryReader.diagnose(candidate)
    return diagnose_events(data.get("trajectory", []), f"{path}#/trajectory")


def _normalize(event: dict[str, Any], ids: dict[str, str]) -> dict[str, Any]:
    event = deepcopy(event)
    for key in (
        "_harbor_attempt_index",
        "_harbor_attempt_trial_id",
    ):
        event.pop(key, None)
    if event.get("type") not in INTERPRETED_EVENT_TYPES:
        return event
    if event.get("type") in TOOL_EVENTS:
        for key in ("timestamp", "duration", "duration_ms"):
            event.pop(key, None)
    if event.get("type") == "worker_metrics_snapshot":
        event.pop("run_id", None)

    def rename(value: str) -> str:
        return ids.setdefault(value, f"local-{len(ids) + 1}")

    def normalize_references(container: dict[str, Any], fields: dict[str, str]) -> None:
        # Walk archives message by message, preserving first-occurrence/list order.
        for head in dict.fromkeys(path.split(".")[0] for path in fields):
            key = head.removesuffix("[]")
            if key not in container:
                continue
            value = container[key]
            if head in fields:
                container[key] = [rename(item) for item in value] \
                    if fields[head] == "transport_ids" else rename(value)
            else:
                nested = {path[len(head) + 1:]: kind for path, kind in fields.items()
                          if path.startswith(head + ".")}
                for child in value if head.endswith("[]") else [value]:
                    normalize_references(child, nested)

    normalize_references(event, {
        path: kind for path, kind in EVENT_SCHEMAS.get(event.get("type"), {}).items()
        if kind in {"transport_id", "transport_ids"}
    })
    # Presence of a random transport ID is not an extra semantic observation.
    for key in ("id", "tool_call_id"):
        if event.get("type") in TOOL_EVENTS:
            event.pop(key, None)
    return event


def _project(
    path: Path,
    data: dict[str, Any],
    attempt: dict[str, Any],
    inventory: list[dict[str, Any]],
    expand: bool,
) -> tuple[dict[str, Any], list[dict]]:
    diagnostic = _records(path, data)
    if diagnostic["status"] == "input_error":
        raise _InputError("; ".join(diagnostic["errors"]))
    aggregate = "attempt_results" in data.get("metadata", {})
    gaps = []
    records = []
    attempt_ids = {item["trial_id"] for item in inventory}
    for record in diagnostic["events"]:
        event = record["event"]
        owner = event.get("_harbor_attempt_trial_id")
        if aggregate and owner is None:
            gaps.append("unattributed_aggregate_events")
            continue
        if owner is not None and owner not in attempt_ids:
            raise _IdentityError("trajectory attempt identity does not match saved attempts")
        if aggregate and owner != attempt["trial_id"]:
            continue
        if "task_id" in event and event["task_id"] != attempt["task_id"]:
            raise _IdentityError("trajectory task identity does not match selected attempt")
        records.append(record)
    if not records:
        gaps.append("missing_trajectory")
    expected_count = attempt.get("trajectory_event_count")
    if expected_count is not None and expected_count != len(records):
        gaps.append("trajectory_event_count_mismatch")
    if (
        not aggregate
        and "trajectory" in data
        and isinstance(data["trajectory"], list)
        and len(data["trajectory"]) != len(records)
    ):
        gaps.append("trajectory_source_count_mismatch")
    observation = attempt.get("metadata", {}).get("worker_metrics_observation", {})
    if not isinstance(observation, dict):
        raise _InputError("invalid worker_metrics_observation")
    count = valid_turn_count(attempt.get("turn_count"))
    if count is None:
        gaps.append("unknown_model_request_attempts")
    if observation.get("schema") != "worker_metrics_v1" or observation.get("status") != "complete":
        gaps.append("legacy_or_partial_worker_metrics")
    if attempt.get("status") not in {"passed", "failed", "error", "unverified"}:
        gaps.append("execution_end_unknown")
    outcomes = attempt.get("tool_calls")
    if not isinstance(outcomes, list) or any(not isinstance(item, dict) for item in outcomes):
        if outcomes is not None:
            raise _InputError("invalid logical tool outcomes")
        gaps.append("unknown_logical_tool_outcomes")
    tool_records = [r for r in records if r["event"].get("type") in TOOL_EVENTS]
    if isinstance(outcomes, list) and len(outcomes) != len(tool_records):
        gaps.append("logical_tool_outcome_count_mismatch")
    projected, values, ids = [], [], {}
    for record in records:
        event = record["event"]
        if "type" not in event:
            gaps.append("unknown_event_type")
        # Compaction and adapter snapshots have no turn in the current producers.
        required = ["type"] + {
            "assistant_message": ["turn", "content"],
            "llm_error_recovery_prompt": ["turn", "error"],
            "context_compaction": [
                "reason",
                "estimate_unit",
                "before_bytes",
                "after_bytes",
                "omitted_messages",
                "omitted_history",
                "semantic_summary_generated",
            ],
            "context_overflow_recovery": [
                "turn",
                "error_type",
                "error",
                "estimate_unit",
                "rejected_bytes",
                "next_bytes",
                "request_reduced",
                "loop_stop_condition",
                "time_round_token_limit_driven",
            ],
            "worker_metrics_snapshot": [
                "worker_status",
                "turn_count",
                "tool_calls",
                "worker_metrics_observation",
            ],
        }.get(event.get("type"), [])
        if event.get("type") in TOOL_EVENTS:
            required += ["turn", "tool", "args", "success", "output"]
        unknown = [key for key in required if event.get(key) is None]
        if event.get("type") == "worker_metrics_snapshot":
            snapshot_observation = event.get("worker_metrics_observation", {})
            unknown += [
                f"worker_metrics_observation.{key}"
                for key in ("schema", "status", "turn_count_semantics", "tool_calls_semantics")
                if snapshot_observation.get(key) is None
            ]
            if (
                event.get("worker_status") not in {"passed", "failed", "error", "unverified"}
                or snapshot_observation.get("schema") != "worker_metrics_v1"
                or snapshot_observation.get("status") != "complete"
            ):
                gaps.append("cancellation_or_partial_snapshot")
        if unknown:
            gaps.append("incomplete_tool_event" if event.get("type") in TOOL_EVENTS
                        else f"incomplete_{event.get('type', 'event')}_evidence")
        turn = valid_turn_count(event.get("turn")) \
            if "turn" in EVENT_SCHEMAS.get(event.get("type"), {}) else None
        if count is not None and turn is not None and turn > count:
            gaps.append("event_turn_exceeds_observed_requests")
        if event.get("type") == "rust_worker_core_cancelled":
            gaps.append("cancellation_or_partial_snapshot")
        value = _normalize(event, ids)
        values.append(value)
        display = (
            value
            if expand
            else {
                key: item if key in {"type", "turn", "phase", "tool", "success"} else _summary(item)
                for key, item in value.items()
            }
        )
        projected.append(
            {
                "event_index": record["event_index"],
                "source": record["source"],
                "projection": display,
                "unknown_fields": unknown,
            }
        )
    side = {
        "identity": {
            "trial_id": data["trial_id"],
            "task_id": attempt["task_id"],
            "attempt_id": attempt["trial_id"],
        },
        "coverage": "partial" if gaps else "complete",
        "gaps": sorted(set(gaps)),
        "metrics": {
            "model_request_attempts": count if count is not None else "unknown",
            "logical_tool_outcomes": len(outcomes)
            if isinstance(outcomes, list) and observation.get("schema") == "worker_metrics_v1"
            else "unknown",
            "bootstrap_outcomes": sum(r["event"].get("type") == "entrypoint_scan" for r in records),
            "provider_http_requests": "unknown",
            "observation": observation,
        },
        "events": projected,
    }
    return side, values


def compare_trajectories(
    path_a: Path,
    path_b: Path,
    *,
    attempt_a: str | None = None,
    attempt_b: str | None = None,
    expand: bool = False,
) -> dict[str, Any]:
    """Diagnose saved trials and compare only explicitly selected attempt identities."""
    report: dict[str, Any] = {
        "projection_version": PROJECTION_VERSION,
        "normalization": NORMALIZATION,
        "description": "first observable difference in the selected projection",
        "status": "input_error",
        "attempts": {},
        "first_difference": None,
        "errors": [],
    }
    try:
        paths = [Path(path_a), Path(path_b)]
        trials = [_load(path) for path in paths]
        inventories = [_attempts(data) for data in trials]
        for label, inventory in zip(("left", "right"), inventories):
            report["attempts"][label] = [
                {"attempt_id": item["trial_id"], "task_id": item["task_id"]} for item in inventory
            ]
        if trials[0]["task_id"] != trials[1]["task_id"]:
            raise _IdentityError("selected trials have different task identities")
        selected = []
        for requested, inventory in zip((attempt_a, attempt_b), inventories):
            matches = [item for item in inventory if item["trial_id"] == requested]
            if requested is not None and not matches:
                raise _IdentityError("requested attempt not found")
            if requested is None and len(inventory) != 1:
                report["status"] = "selection_required"
                report["errors"] = [
                    "select both attempts explicitly with --attempt-a and --attempt-b"
                ]
                return _safe(report)
            selected.append(matches[0] if requested is not None else inventory[0])
        sides, values = [], []
        for path, data, attempt, inventory in zip(paths, trials, selected, inventories):
            side, sequence = _project(path, data, attempt, inventory, expand)
            sides.append(side)
            values.append(sequence)
        report["left"], report["right"] = sides
        for index, (left, right) in enumerate(zip_longest(*values), 1):
            if _json(left) != _json(right):
                fields = (
                    ["event_presence"]
                    if left is None or right is None
                    else sorted(
                        key
                        for key in left.keys() | right.keys()
                        if (key in left) != (key in right)
                        or _json(left.get(key)) != _json(right.get(key))
                    )
                )
                report["first_difference"] = {
                    "position": index,
                    "fields": fields,
                    "left": sides[0]["events"][index - 1] if left is not None else None,
                    "right": sides[1]["events"][index - 1] if right is not None else None,
                }
                break
        report["metric_differences"] = [
            key
            for key in sides[0]["metrics"]
            if _json(sides[0]["metrics"][key]) != _json(sides[1]["metrics"][key])
        ]
        report["status"] = (
            "partial"
            if any(side["gaps"] for side in sides)
            else "different"
            if report["first_difference"] or report["metric_differences"]
            else "identical"
        )
    except _IdentityError as error:
        report.update(status="identity_mismatch", errors=[str(error)])
    except (_InputError, RecursionError) as error:
        report.update(
            status="input_error",
            errors=[str(error) if isinstance(error, _InputError) else "input nesting too deep"],
        )
    return _safe(report)
