"""Guard hit and benefit audit helpers for guard-reduction work."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from hl.guard_convergence import CAMPAIGN_RUNNER_PATH, RUST_CORE_PATH


GUARDISH_POLICY_NAMES = {
    "large_toolchain_install_plan",
    "network_probe_tool_missing",
    "package_manager_semantic_failure",
    "package_manager_timeout_cap",
}


def build_guard_hit_benefit_audit(
    *,
    repo_root: Path,
    analysis_root: Path,
    campaign_summary_path: Path | None = None,
    campaign_id: str | None = None,
    max_summaries: int | None = None,
    pass_score: float = 1.0,
) -> dict[str, Any]:
    """Build a deterministic guard hit/benefit audit from analysis artifacts.

    Benefit is intentionally strict: a policy is classified as beneficial only
    when a same-campaign, same-task earlier result was below ``pass_score`` and
    a later trial that hit the policy reached ``pass_score``. A policy hit in a
    passing trial is recorded as weaker evidence but does not prove benefit.
    """

    summary_paths = analysis_summary_paths(
        analysis_root,
        campaign_id=campaign_id,
        max_summaries=max_summaries,
    )
    campaign_summary = _load_json(campaign_summary_path) if campaign_summary_path else {}
    result_index = _campaign_result_index(campaign_summary)
    source_catalog = source_guard_policy_catalog(repo_root)
    observed_catalog: dict[str, dict[str, Any]] = {}
    occurrence_records: list[dict[str, Any]] = []

    for summary_path in summary_paths:
        summary = _load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        summary_id = str(summary.get("summary_id") or summary_path.parent.name)
        summary_campaign_id = str(summary.get("campaign_id") or summary_path.parent.parent.name)
        _merge_policy_coverage(observed_catalog, summary.get("policy_coverage"))
        trajectory = summary.get("trajectory_evidence")
        if not isinstance(trajectory, dict):
            continue
        for task_id, evidence in trajectory.items():
            if not isinstance(evidence, dict):
                continue
            trial_result = _match_trial_result(
                result_index,
                summary_id=summary_id,
                task_id=str(task_id),
            )
            policy_counts = evidence.get("policy_counts")
            if not isinstance(policy_counts, dict):
                continue
            for policy_name, raw_count in policy_counts.items():
                name = str(policy_name)
                if not _is_guardish_policy(name):
                    continue
                count = _positive_int(raw_count)
                if count <= 0:
                    continue
                occurrence_records.append(
                    {
                        "policy": name,
                        "count": count,
                        "campaign_id": summary_campaign_id,
                        "summary_id": summary_id,
                        "task_id": str(task_id),
                        "trial_id": trial_result.get("trial_id", ""),
                        "status": trial_result.get("status", ""),
                        "score": _optional_float(trial_result.get("score")),
                        "order": trial_result.get("order"),
                        "summary_path": str(summary_path),
                    }
                )

    source_policy_names = set(source_catalog)
    observed_policy_names = {
        name for name in observed_catalog if _is_guardish_policy(name)
    } | {str(item["policy"]) for item in occurrence_records}
    policy_names = sorted(source_policy_names | observed_policy_names)
    records = _policy_records(
        policy_names,
        source_catalog=source_catalog,
        observed_catalog=observed_catalog,
        occurrences=occurrence_records,
        result_index=result_index,
        pass_score=pass_score,
    )
    classified = _classify_policy_records(records)
    return {
        "schema_version": 1,
        "repo_root": str(repo_root.resolve()),
        "analysis_root": str(analysis_root),
        "campaign_summary_path": str(campaign_summary_path) if campaign_summary_path else "",
        "campaign_id": campaign_id or str(campaign_summary.get("campaign_id") or ""),
        "analysis_summary_count": len(summary_paths),
        "analysis_summary_paths": [str(path) for path in summary_paths],
        "pass_score": pass_score,
        "benefit_definition": (
            "beneficial requires same-campaign same-task earlier score below "
            "pass_score and later policy-hit trial score at or above pass_score"
        ),
        "guard_policy_count": len(records),
        "classification_counts": {
            key: len(value) for key, value in classified.items()
        },
        "zero_hit_guards": classified["zero_hit"],
        "hit_zero_benefit_guards": classified["hit_zero_benefit"],
        "beneficial_guards": classified["beneficial"],
        "deletion_candidate_guards": classified["zero_hit"] + classified["hit_zero_benefit"],
        "policies": records,
    }


def analysis_summary_paths(
    analysis_root: Path,
    *,
    campaign_id: str | None = None,
    max_summaries: int | None = None,
) -> list[Path]:
    if campaign_id:
        base = analysis_root / campaign_id
        paths = sorted(base.glob("summary_*/summary.json"), key=_summary_sort_key)
    else:
        paths = sorted(
            analysis_root.glob("*/summary_*/summary.json"),
            key=lambda path: (path.parent.parent.name, _summary_sort_key(path)),
        )
    if max_summaries is not None and max_summaries > 0:
        return paths[-max_summaries:]
    return paths


def source_guard_policy_catalog(repo_root: Path) -> dict[str, dict[str, Any]]:
    rust_text = _read_text(repo_root / RUST_CORE_PATH)
    runner_text = _read_text(repo_root / CAMPAIGN_RUNNER_PATH)
    catalog: dict[str, dict[str, Any]] = {}
    known_runner_policies = set(re.findall(r'"([A-Za-z0-9_]+(?:_guard|_timeout_phase))"', runner_text))

    for name in sorted(set(re.findall(r"^fn\s+(blocked_repeated_[A-Za-z0-9_]+)\s*\(", rust_text, re.MULTILINE))):
        if name == "blocked_repeated_timeout_path":
            continue
        policy = _policy_from_blocked_repeated_fn(name)
        _add_source(catalog, policy, "rust_blocked_repeated", name)
    for semantic_kind in sorted(
        set(re.findall(r'semantic_failure_kind:\s*"(blocked_repeated_[A-Za-z0-9_]+)"', rust_text))
    ):
        policy = _policy_from_blocked_repeated_fn(semantic_kind)
        _add_source(catalog, policy, "rust_blocked_repeated_spec", semantic_kind)

    for name in sorted(set(re.findall(r"^fn\s+(command_looks_like_[A-Za-z0-9_]+)\s*\(", rust_text, re.MULTILINE))):
        policy = _policy_from_command_classifier(name)
        if policy in known_runner_policies:
            _add_source(catalog, policy, "rust_command_classifier", name)

    for name in sorted(set(re.findall(r"^def\s+(_command_looks_like_[A-Za-z0-9_]+)\s*\(", runner_text, re.MULTILINE))):
        policy = _policy_from_command_classifier(name.lstrip("_"))
        if policy in known_runner_policies:
            _add_source(catalog, policy, "python_command_classifier", name)

    for policy, matcher in sorted(
        set(
            re.findall(
                r'_AnalysisTimeoutPhasePolicy\(\s*"([A-Za-z0-9_]+_timeout_phase)",\s*"(_analysis_command_matches_[A-Za-z0-9_]+)"',
                runner_text,
                re.MULTILINE,
            )
        )
    ):
        _add_source(catalog, policy, "python_timeout_phase_table", matcher)

    for name in sorted(set(re.findall(r'"([A-Za-z0-9_]+_timeout_phase)"', runner_text))):
        _add_source(catalog, name, "python_timeout_phase_policy", name)

    return catalog


def _policy_records(
    policy_names: list[str],
    *,
    source_catalog: dict[str, dict[str, Any]],
    observed_catalog: dict[str, dict[str, Any]],
    occurrences: list[dict[str, Any]],
    result_index: dict[str, Any],
    pass_score: float,
) -> dict[str, dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for occurrence in occurrences:
        by_policy.setdefault(str(occurrence["policy"]), []).append(occurrence)

    records: dict[str, dict[str, Any]] = {}
    for name in policy_names:
        policy_occurrences = by_policy.get(name, [])
        source = source_catalog.get(name, {})
        observed = observed_catalog.get(name, {})
        hit_count = sum(int(item.get("count") or 0) for item in policy_occurrences)
        coverage_count = _positive_int(observed.get("coverage_count"))
        total_hits = max(hit_count, coverage_count)
        tasks = sorted(
            {
                str(task)
                for item in policy_occurrences
                for task in [item.get("task_id")]
                if task
            }
            | {str(task) for task in observed.get("tasks", []) if task}
        )
        passed_hit_count = sum(
            int(item.get("count") or 0)
            for item in policy_occurrences
            if _is_passing_score(item.get("score"), pass_score)
        )
        failed_hit_count = sum(
            int(item.get("count") or 0)
            for item in policy_occurrences
            if not _is_passing_score(item.get("score"), pass_score)
        )
        transition_examples = _fail_to_pass_transition_examples(
            policy_occurrences,
            result_index=result_index,
            pass_score=pass_score,
        )
        records[name] = {
            "policy": name,
            "classification": _classification(total_hits, transition_examples),
            "hit_count": total_hits,
            "trajectory_hit_count": hit_count,
            "coverage_hit_count": coverage_count,
            "task_count": len(tasks),
            "tasks": tasks,
            "passed_hit_count": passed_hit_count,
            "failed_or_unknown_hit_count": failed_hit_count,
            "fail_to_pass_transition_count": len(transition_examples),
            "fail_to_pass_examples": transition_examples[:5],
            "source_kinds": sorted(source.get("source_kinds", [])),
            "source_identifiers": sorted(source.get("source_identifiers", [])),
            "description": str(observed.get("description") or ""),
            "examples": observed.get("examples", [])[:5],
        }
    return records


def _classify_policy_records(
    records: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    classified = {"zero_hit": [], "hit_zero_benefit": [], "beneficial": []}
    for name, record in sorted(records.items()):
        classification = str(record.get("classification") or "")
        if classification in classified:
            classified[classification].append(name)
    return classified


def _classification(total_hits: int, transition_examples: list[dict[str, Any]]) -> str:
    if total_hits <= 0:
        return "zero_hit"
    if transition_examples:
        return "beneficial"
    return "hit_zero_benefit"


def _fail_to_pass_transition_examples(
    occurrences: list[dict[str, Any]],
    *,
    result_index: dict[str, Any],
    pass_score: float,
) -> list[dict[str, Any]]:
    task_results = result_index.get("task_results_by_task", {})
    examples: list[dict[str, Any]] = []
    for occurrence in sorted(occurrences, key=lambda item: _order_or_max(item.get("order"))):
        score = occurrence.get("score")
        order = occurrence.get("order")
        task_id = str(occurrence.get("task_id") or "")
        if not task_id or order is None or not _is_passing_score(score, pass_score):
            continue
        previous = [
            item
            for item in task_results.get(task_id, [])
            if _order_or_max(item.get("order")) < _order_or_max(order)
        ]
        if not any(not _is_passing_score(item.get("score"), pass_score) for item in previous):
            continue
        previous_best = max(
            (_optional_float(item.get("score")) or 0.0 for item in previous),
            default=None,
        )
        examples.append(
            {
                "campaign_id": occurrence.get("campaign_id", ""),
                "summary_id": occurrence.get("summary_id", ""),
                "task_id": task_id,
                "trial_id": occurrence.get("trial_id", ""),
                "previous_best_score": previous_best,
                "score": _optional_float(score),
            }
        )
    return examples


def _campaign_result_index(campaign_summary: dict[str, Any]) -> dict[str, Any]:
    task_results = campaign_summary.get("task_results")
    if not isinstance(task_results, list):
        task_results = []
    by_trial_id: dict[str, dict[str, Any]] = {}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for order, raw in enumerate(task_results):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["order"] = order
        trial_id = str(item.get("trial_id") or "")
        task_id = str(item.get("task_id") or "")
        if trial_id:
            by_trial_id[trial_id] = item
        if task_id:
            by_task.setdefault(task_id, []).append(item)

    summary_trial_ids: dict[str, list[str]] = {}
    for key in ("iteration_summaries", "summaries"):
        raw_summaries = campaign_summary.get(key)
        if not isinstance(raw_summaries, list):
            continue
        for raw_summary in raw_summaries:
            if not isinstance(raw_summary, dict):
                continue
            summary_id = str(raw_summary.get("summary_id") or "")
            trial_ids = [str(item) for item in raw_summary.get("trial_ids") or [] if item]
            if summary_id and trial_ids:
                summary_trial_ids.setdefault(summary_id, trial_ids)
    return {
        "results_by_trial_id": by_trial_id,
        "task_results_by_task": by_task,
        "summary_trial_ids": summary_trial_ids,
    }


def _match_trial_result(
    result_index: dict[str, Any],
    *,
    summary_id: str,
    task_id: str,
) -> dict[str, Any]:
    by_trial_id = result_index.get("results_by_trial_id", {})
    for trial_id in result_index.get("summary_trial_ids", {}).get(summary_id, []):
        item = by_trial_id.get(trial_id)
        if isinstance(item, dict) and str(item.get("task_id") or "") == task_id:
            return item
    task_results = result_index.get("task_results_by_task", {}).get(task_id, [])
    if len(task_results) == 1:
        return task_results[0]
    return {}


def _merge_policy_coverage(
    observed_catalog: dict[str, dict[str, Any]],
    policy_coverage: Any,
) -> None:
    if not isinstance(policy_coverage, dict):
        return
    policies = policy_coverage.get("policies")
    if not isinstance(policies, dict):
        return
    for name, raw in policies.items():
        policy = str(name)
        if not _is_guardish_policy(policy) or not isinstance(raw, dict):
            continue
        entry = observed_catalog.setdefault(
            policy,
            {"coverage_count": 0, "tasks": set(), "examples": [], "description": ""},
        )
        entry["coverage_count"] += _positive_int(raw.get("count"))
        if not entry.get("description") and raw.get("description"):
            entry["description"] = str(raw.get("description"))
        tasks = raw.get("tasks")
        if isinstance(tasks, list):
            entry["tasks"].update(str(task) for task in tasks if task)
        examples = raw.get("examples")
        if isinstance(examples, list):
            for example in examples:
                clean = _clean_example(example)
                if clean and clean not in entry["examples"]:
                    entry["examples"].append(clean)


def _clean_example(example: Any) -> dict[str, str]:
    if not isinstance(example, dict):
        return {}
    task_id = _redact_text(str(example.get("task_id") or ""))
    command = _redact_text(str(example.get("command") or ""))
    return {key: value for key, value in {"task_id": task_id, "command": command}.items() if value}


def _add_source(
    catalog: dict[str, dict[str, Any]],
    policy: str,
    source_kind: str,
    source_identifier: str,
) -> None:
    if not policy:
        return
    entry = catalog.setdefault(policy, {"source_kinds": set(), "source_identifiers": set()})
    entry["source_kinds"].add(source_kind)
    entry["source_identifiers"].add(source_identifier)


def _policy_from_blocked_repeated_fn(name: str) -> str:
    stem = re.sub(r"^blocked_", "", name)
    return f"{stem}_guard"


def _policy_from_command_classifier(name: str) -> str:
    stem = re.sub(r"^command_looks_like_", "", name)
    if stem == "vm_service_readiness_validation":
        stem = "vm_service_readiness"
    return f"{stem}_timeout_phase"


def _is_guardish_policy(name: str) -> bool:
    return (
        name.endswith("_guard")
        or name.endswith("_timeout_phase")
        or name in GUARDISH_POLICY_NAMES
    )


def _summary_sort_key(path: Path) -> tuple[int, str]:
    text = path.parent.name
    match = re.search(r"(\d+)$", text)
    return (int(match.group(1)) if match else 0, text)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def _positive_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_passing_score(value: Any, pass_score: float) -> bool:
    parsed = _optional_float(value)
    return parsed is not None and parsed >= pass_score


def _order_or_max(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 2**31 - 1


def _redact_text(text: str) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED_SECRET]", text)
    text = re.sub(r"AKIA[0-9A-Z]{16}", "[REDACTED_SECRET]", text)
    text = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)=([^\s'\"]+)",
        r"\1=[REDACTED_SECRET]",
        text,
    )
    return text
