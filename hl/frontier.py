"""Per-task same-model frontier and Codex change-evaluation helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hl.attribution import FailureAttributor
from hl.model_scope import safe_model_scope_name


FRONTIER_SCHEMA_VERSION = 1


def frontier_path(memory_path: str | Path, campaign_id: str, model_scope: str) -> Path:
    safe_campaign = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in campaign_id)
    return (
        Path(memory_path)
        / "summaries"
        / f"{safe_campaign}_frontier_{safe_model_scope_name(model_scope)}.json"
    )


def load_frontier(path: str | Path) -> dict[str, Any]:
    frontier_file = Path(path)
    if not frontier_file.exists():
        return {
            "schema_version": FRONTIER_SCHEMA_VERSION,
            "campaign_id": "",
            "model_scope": "",
            "updated_at": "",
            "tasks": {},
            "aggregate": {},
        }
    try:
        data = json.loads(frontier_file.read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": FRONTIER_SCHEMA_VERSION,
            "campaign_id": "",
            "model_scope": "",
            "updated_at": "",
            "tasks": {},
            "aggregate": {},
        }
    return data if isinstance(data, dict) else {}


def update_frontier(
    frontier: dict[str, Any],
    *,
    trials: list[Any],
    campaign_id: str,
    model_scope: str,
    summary_id: str,
    active_packet_id: str = "",
) -> dict[str, Any]:
    tasks = frontier.setdefault("tasks", {})
    now = datetime.now().isoformat()
    for trial in trials:
        task_id = str(getattr(trial, "task_id", "") or "")
        if not task_id:
            continue
        score = _safe_float(getattr(trial, "score", 0.0))
        status = str(getattr(getattr(trial, "status", None), "value", "") or "")
        entry = tasks.setdefault(
            task_id,
            {
                "task_id": task_id,
                "model_scope": model_scope,
                "best_score": -1.0,
                "best_trial_id": "",
                "best_packet_id": "",
                "last_score": 0.0,
                "last_trial_id": "",
                "last_summary_id": "",
                "last_status": "",
                "attempts": 0,
                "passes": 0,
                "failures": 0,
                "regressed_after_packet": [],
                "history": [],
            },
        )
        previous_best = _safe_float(entry.get("best_score"), default=-1.0)
        if previous_best >= 1.0 and score < previous_best and active_packet_id:
            regressions = entry.setdefault("regressed_after_packet", [])
            if active_packet_id not in regressions:
                regressions.append(active_packet_id)
        entry["attempts"] = int(entry.get("attempts") or 0) + 1
        entry["passes"] = int(entry.get("passes") or 0) + (1 if score >= 1.0 else 0)
        entry["failures"] = int(entry.get("failures") or 0) + (1 if score < 1.0 else 0)
        entry["last_score"] = score
        entry["last_trial_id"] = str(getattr(trial, "trial_id", "") or "")
        entry["last_summary_id"] = summary_id
        entry["last_status"] = status
        entry["updated_at"] = now
        if score > previous_best:
            entry["best_score"] = score
            entry["best_trial_id"] = str(getattr(trial, "trial_id", "") or "")
            entry["best_packet_id"] = active_packet_id
        history = entry.setdefault("history", [])
        history.append(
            {
                "summary_id": summary_id,
                "trial_id": str(getattr(trial, "trial_id", "") or ""),
                "score": score,
                "status": status,
                "packet_id": active_packet_id,
                "recorded_at": now,
            }
        )
        entry["history"] = history[-10:]

    frontier["schema_version"] = FRONTIER_SCHEMA_VERSION
    frontier["campaign_id"] = campaign_id
    frontier["model_scope"] = model_scope
    frontier["updated_at"] = now
    frontier["aggregate"] = frontier_aggregate(frontier)
    return frontier


def write_frontier(path: str | Path, frontier: dict[str, Any]) -> Path:
    frontier_file = Path(path)
    frontier_file.parent.mkdir(parents=True, exist_ok=True)
    frontier_file.write_text(json.dumps(frontier, indent=2))
    return frontier_file


def frontier_aggregate(frontier: dict[str, Any]) -> dict[str, Any]:
    tasks = frontier.get("tasks") if isinstance(frontier, dict) else {}
    if not isinstance(tasks, dict):
        tasks = {}
    total = len(tasks)
    solved = sum(1 for entry in tasks.values() if _safe_float(entry.get("best_score")) >= 1.0)
    regressed = sum(
        1
        for entry in tasks.values()
        if entry.get("regressed_after_packet")
    )
    volatile = sum(
        1
        for entry in tasks.values()
        if int(entry.get("passes") or 0) > 0 and int(entry.get("failures") or 0) > 0
    )
    return {
        "tasks": total,
        "solved_tasks": solved,
        "frontier_score": round(
            sum(max(0.0, _safe_float(entry.get("best_score"))) for entry in tasks.values())
            / total,
            4,
        )
        if total
        else 0.0,
        "regressed_tasks": regressed,
        "volatile_tasks": volatile,
    }


def frontier_summary(frontier: dict[str, Any], *, max_tasks: int = 20) -> dict[str, Any]:
    """Return same-model frontier evidence without task-count truncation.

    ``max_tasks`` is retained for compatibility with older callers, but it is
    audit metadata only. Same-model frontier context feeds Codex update and
    campaign decisions, so a task-count value must not truncate the evidence
    available to master, sub-agent, validation/regression, or Worker loops.
    """
    _ = max_tasks
    tasks = frontier.get("tasks") if isinstance(frontier, dict) else {}
    if not isinstance(tasks, dict):
        tasks = {}
    entries = list(tasks.values())
    unstable = [
        entry
        for entry in entries
        if int(entry.get("passes") or 0) > 0 and int(entry.get("failures") or 0) > 0
    ]
    regressed = [entry for entry in entries if entry.get("regressed_after_packet")]
    recent = sorted(
        entries,
        key=lambda entry: str(entry.get("updated_at") or ""),
        reverse=True,
    )
    return {
        "schema_version": frontier.get("schema_version", FRONTIER_SCHEMA_VERSION),
        "campaign_id": frontier.get("campaign_id", ""),
        "model_scope": frontier.get("model_scope", ""),
        "updated_at": frontier.get("updated_at", ""),
        "aggregate": frontier.get("aggregate") or frontier_aggregate(frontier),
        "max_tasks_audit_only": max_tasks,
        "max_tasks_stop_condition": False,
        "max_tasks_truncates_frontier_evidence": False,
        "recent_tasks": [_task_summary(entry) for entry in recent],
        "volatile_tasks": [_task_summary(entry) for entry in unstable],
        "regressed_tasks": [_task_summary(entry) for entry in regressed],
    }


def evaluate_change_manifest(
    *,
    manifest: dict[str, Any],
    trials: list[Any],
    frontier_before: dict[str, Any],
    summary_id: str,
) -> dict[str, Any]:
    tasks_before = frontier_before.get("tasks") if isinstance(frontier_before, dict) else {}
    if not isinstance(tasks_before, dict):
        tasks_before = {}
    prediction = manifest.get("prediction") if isinstance(manifest, dict) else {}
    if not isinstance(prediction, dict):
        prediction = {}
    expected_classes = [
        str(item).lower()
        for item in prediction.get("expected_fixed_task_classes", [])
        if str(item).strip()
    ]
    risk_classes = [
        str(item).lower()
        for item in prediction.get("risk_task_classes", [])
        if str(item).strip()
    ]

    flipped_pass: list[str] = []
    flipped_fail: list[str] = []
    unchanged_fail: list[str] = []
    unchanged_pass: list[str] = []
    hits: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []

    for trial in trials:
        task_id = str(getattr(trial, "task_id", "") or "")
        if not task_id:
            continue
        current_score = _safe_float(getattr(trial, "score", 0.0))
        before_entry = tasks_before.get(task_id, {})
        previous_score = _safe_float(before_entry.get("last_score"))
        previous_best = _safe_float(before_entry.get("best_score"))
        previous_passed = previous_score >= 1.0 or previous_best >= 1.0
        current_passed = current_score >= 1.0
        labels = _trial_labels(trial)
        expected_matched_classes = _matched_classes(labels, expected_classes)
        risk_matched_classes = _matched_classes(labels, risk_classes)
        expected_match = bool(expected_matched_classes)
        risk_match = bool(risk_matched_classes)

        if not previous_passed and current_passed:
            flipped_pass.append(task_id)
            if expected_match:
                hits.append(
                    _prediction_event(
                        task_id,
                        "flipped_pass",
                        labels,
                        "expected improvement",
                        matched_classes=expected_matched_classes,
                    )
                )
        elif previous_passed and not current_passed:
            flipped_fail.append(task_id)
            if risk_match:
                hits.append(
                    _prediction_event(
                        task_id,
                        "flipped_fail",
                        labels,
                        "declared risk",
                        matched_classes=risk_matched_classes,
                    )
                )
            else:
                misses.append(
                    _prediction_event(task_id, "flipped_fail", labels, "unexpected regression")
                )
        elif not current_passed:
            unchanged_fail.append(task_id)
            if expected_match:
                misses.append(
                    _prediction_event(
                        task_id,
                        "unchanged_fail",
                        labels,
                        "expected fix missed",
                        matched_classes=expected_matched_classes,
                    )
                )
        else:
            unchanged_pass.append(task_id)

        evaluated.append(
            {
                "task_id": task_id,
                "previous_last_score": previous_score,
                "previous_best_score": previous_best,
                "current_score": current_score,
                "labels": labels,
                "expected_match": expected_match,
                "risk_match": risk_match,
                "expected_matched_classes": expected_matched_classes,
                "risk_matched_classes": risk_matched_classes,
            }
        )

    hit_count = len(hits)
    miss_count = len(misses)
    if not expected_classes:
        outcome = "insufficient_prediction"
    elif not evaluated:
        outcome = "insufficient_evidence"
    elif miss_count > hit_count:
        outcome = "prediction_missed"
    elif hit_count > 0 and miss_count == 0:
        outcome = "prediction_supported"
    else:
        outcome = "mixed"
    return {
        "packet_id": str(manifest.get("packet_id") or ""),
        "summary_id": summary_id,
        "evaluated_at": datetime.now().isoformat(),
        "prediction": prediction,
        "evaluated_trials": evaluated,
        "flipped_pass": flipped_pass,
        "flipped_fail": flipped_fail,
        "unchanged_fail": unchanged_fail,
        "unchanged_pass": unchanged_pass,
        "prediction_hits": hits,
        "prediction_misses": misses,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "outcome": outcome,
        "rollback_recommended": bool(miss_count > hit_count and miss_count > 0),
    }


def _trial_labels(trial: Any) -> list[str]:
    metadata = getattr(trial, "metadata", {}) or {}
    task_metadata = metadata.get("task_metadata") if isinstance(metadata, dict) else {}
    if not isinstance(task_metadata, dict):
        task_metadata = {}
    try:
        attribution = FailureAttributor().analyze(trial)
        failure_category = attribution.failure_category
        components = attribution.affected_components
    except Exception:
        failure_category = ""
        components = []
    labels = [
        str(getattr(trial, "task_id", "") or ""),
        str(getattr(getattr(trial, "task_domain", None), "value", "") or ""),
        str(getattr(getattr(trial, "task_difficulty", None), "value", "") or ""),
        str(getattr(getattr(trial, "status", None), "value", "") or ""),
        str(task_metadata.get("task_type") or ""),
        str(metadata.get("timeout_phase") or "") if isinstance(metadata, dict) else "",
        failure_category,
        *[str(component) for component in components],
        *[str(tag) for tag in task_metadata.get("tags", []) if str(tag).strip()],
    ]
    return [label.lower() for label in labels if label]


def _matched_classes(labels: list[str], classes: list[str]) -> list[str]:
    if not classes:
        return []
    matched: list[str] = []
    for expected in classes:
        if any(_class_matches_label(expected, label) for label in labels):
            matched.append(expected)
    return matched


def _class_matches_label(expected: str, label: str) -> bool:
    if expected in label or label in expected:
        return True
    normalized_expected = _normalize_match_text(expected)
    normalized_label = _normalize_match_text(label)
    if not normalized_expected or not normalized_label:
        return False
    if normalized_expected in normalized_label or normalized_label in normalized_expected:
        return True
    compact_expected = normalized_expected.replace(" ", "")
    compact_label = normalized_label.replace(" ", "")
    return bool(
        compact_expected
        and compact_label
        and (compact_expected in compact_label or compact_label in compact_expected)
    )


def _normalize_match_text(value: str) -> str:
    return " ".join(
        "".join(char.lower() if char.isalnum() else " " for char in value).split()
    )


def _matches_any(labels: list[str], classes: list[str]) -> bool:
    return bool(_matched_classes(labels, classes))


def _prediction_event(
    task_id: str,
    event: str,
    labels: list[str],
    reason: str,
    *,
    matched_classes: list[str] | None = None,
) -> dict[str, Any]:
    result = {
        "task_id": task_id,
        "event": event,
        "labels": labels[:12],
        "reason": reason,
    }
    if matched_classes:
        result["matched_classes"] = matched_classes[:8]
    return result


def _task_summary(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": entry.get("task_id", ""),
        "best_score": _safe_float(entry.get("best_score")),
        "best_trial_id": entry.get("best_trial_id", ""),
        "best_packet_id": entry.get("best_packet_id", ""),
        "last_score": _safe_float(entry.get("last_score")),
        "last_trial_id": entry.get("last_trial_id", ""),
        "last_summary_id": entry.get("last_summary_id", ""),
        "attempts": int(entry.get("attempts") or 0),
        "passes": int(entry.get("passes") or 0),
        "failures": int(entry.get("failures") or 0),
        "regressed_after_packet": list(entry.get("regressed_after_packet") or []),
    }


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
