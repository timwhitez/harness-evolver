"""Worker counters are observations, separate from provider usage/API calls."""
from __future__ import annotations

from typing import Any


def valid_turn_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def worker_metrics_metadata(turn_count: int | None, status: str,
                            schema: str = "worker_metrics_v1") -> dict[str, Any]:
    return {
        "turn_count": turn_count,
        "worker_metrics_observation": {
            "schema": schema,
            "status": status,
            "turn_count_semantics": "rust_model_request_attempts",
            "tool_calls_semantics": "logical_outcomes_including_bootstrap"
                if schema == "worker_metrics_v1" else "legacy_unspecified",
        },
    }


def harbor_worker_metrics(trial: dict[str, Any]) -> dict[str, Any]:
    metadata = (trial.get("agent_result") or {}).get("metadata") or {}
    count = valid_turn_count(metadata.get("turn_count"))
    observation = metadata.get("worker_metrics_observation")
    native = isinstance(observation, dict) and observation.get("schema") == "worker_metrics_v1"
    status = observation.get("status") if isinstance(observation, dict) and native else "legacy"
    if count is None:
        status = "unknown"
    elif status not in {"complete", "partial", "legacy"}:
        status = "unknown"
    return worker_metrics_metadata(count, status, "worker_metrics_v1" if native else "legacy")


def aggregate_worker_metrics(attempts: list[Any]) -> dict[str, Any]:
    counts = [attempt.turn_count for attempt in attempts]
    count = sum(counts) if counts and all(value is not None for value in counts) else None
    statuses = [attempt.metadata.get("worker_metrics_observation", {}).get("status")
                for attempt in attempts]
    native = all(attempt.metadata.get("worker_metrics_observation", {}).get("schema")
                 == "worker_metrics_v1" for attempt in attempts)
    status = "complete" if count is not None and native and all(s == "complete" for s in statuses) \
        else "partial" if count is not None else "unknown"
    return worker_metrics_metadata(count, status, "worker_metrics_v1" if native else "legacy")
