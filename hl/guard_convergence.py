"""Guard-convergence safety net and deterministic audit helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


RUST_CORE_PATH = Path("crates/hl-worker-core/src/main.rs")
CAMPAIGN_RUNNER_PATH = Path("scripts/run_campaign.py")


def load_trials_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data if isinstance(data, dict) else {}


def guard_convergence_config(trials_config: dict[str, Any]) -> dict[str, Any]:
    raw = trials_config.get("guard_convergence", {})
    return raw if isinstance(raw, dict) else {}


def fixed_eval_task_ids(trials_config: dict[str, Any]) -> list[str]:
    fixed_eval = guard_convergence_config(trials_config).get("fixed_eval", {})
    if not isinstance(fixed_eval, dict):
        return []
    tasks = fixed_eval.get("tasks", [])
    task_ids: list[str] = []
    if isinstance(tasks, str):
        task_ids.extend(item.strip() for item in tasks.split(","))
    elif isinstance(tasks, list):
        for item in tasks:
            if isinstance(item, str):
                task_ids.append(item.strip())
            elif isinstance(item, dict):
                task_ids.append(
                    str(item.get("task_id") or item.get("id") or "").strip()
                )
    return [task_id for task_id in dict.fromkeys(task_ids) if task_id]


def fixed_eval_audit(trials_config: dict[str, Any]) -> dict[str, Any]:
    fixed_eval = guard_convergence_config(trials_config).get("fixed_eval", {})
    if not isinstance(fixed_eval, dict):
        fixed_eval = {}
    raw_tasks = fixed_eval.get("tasks", [])
    task_entries = raw_tasks if isinstance(raw_tasks, list) else []
    task_ids = fixed_eval_task_ids(trials_config)
    domains: set[str] = set()
    known_pass = 0
    known_fail = 0
    baseline_scores: list[float] = []
    for item in task_entries:
        if not isinstance(item, dict):
            continue
        domain = str(item.get("domain") or "").strip()
        if domain:
            domains.add(domain)
        score = item.get("baseline_score")
        if score is None:
            continue
        try:
            parsed_score = float(score)
        except (TypeError, ValueError):
            continue
        baseline_scores.append(parsed_score)
        if parsed_score >= 1.0:
            known_pass += 1
        elif parsed_score <= 0.0:
            known_fail += 1
    baseline_score = fixed_eval.get("baseline_score")
    if baseline_score is None and baseline_scores:
        baseline_score = sum(baseline_scores) / len(baseline_scores)
    minimum_accept_score = fixed_eval.get("minimum_accept_score", baseline_score)
    domain_count = len(domains)
    return {
        "name": fixed_eval.get("name", ""),
        "source_campaign_id": fixed_eval.get("source_campaign_id", ""),
        "source_summary_id": fixed_eval.get("source_summary_id", ""),
        "tasks": task_ids,
        "task_count": len(task_ids),
        "domains": sorted(domains),
        "domain_count": domain_count,
        "known_pass_count": known_pass,
        "known_fail_count": known_fail,
        "baseline_score": _optional_float(baseline_score),
        "minimum_accept_score": _optional_float(minimum_accept_score),
        "coverage_ok": domain_count >= 4,
        "has_known_pass_and_fail": known_pass > 0 and known_fail > 0,
        "valid": (
            len(task_ids) > 0
            and domain_count >= 4
            and known_pass > 0
            and known_fail > 0
        ),
    }


def count_guard_surface(repo_root: Path) -> dict[str, Any]:
    rust_text = _read_text(repo_root / RUST_CORE_PATH)
    runner_text = _read_text(repo_root / CAMPAIGN_RUNNER_PATH)
    rust_fn_names = set(
        re.findall(r"^fn\s+([A-Za-z0-9_]+)\s*\(", rust_text, re.MULTILINE)
    )
    rust_guard_names = sorted(
        name
        for name in rust_fn_names
        if "guard" in name or "block" in name or name.startswith("blocked_")
    )
    rust_repeated_timeout_names = sorted(
        name
        for name in rust_fn_names
        if name.startswith("blocked_repeated_") and name.endswith("_timeout_path")
    )
    rust_looks_like_names = sorted(
        set(
            re.findall(
                r"^fn\s+(command_looks_like_[A-Za-z0-9_]+)\s*\(",
                rust_text,
                re.MULTILINE,
            )
        )
    )
    python_looks_like_names = sorted(
        set(
            re.findall(
                r"^def\s+(_command_looks_like_[A-Za-z0-9_]+)\s*\(",
                runner_text,
                re.MULTILINE,
            )
        )
    )
    python_timeout_phase_keys = sorted(
        set(re.findall(r'"([A-Za-z0-9_]+_timeout_phase)"', runner_text))
    )
    total_guard_surface = (
        len(rust_guard_names) + len(rust_looks_like_names) + len(python_looks_like_names)
    )
    return {
        "rust_guard_functions": len(rust_guard_names),
        "rust_blocked_repeated_timeout_path_functions": len(rust_repeated_timeout_names),
        "rust_command_looks_like_functions": len(rust_looks_like_names),
        "python_command_looks_like_functions": len(python_looks_like_names),
        "python_timeout_phase_keys": len(python_timeout_phase_keys),
        "total_guard_surface": total_guard_surface,
        "sample_names": {
            "rust_guard_functions": rust_guard_names[:10],
            "rust_blocked_repeated_timeout_path_functions": rust_repeated_timeout_names[:10],
            "rust_command_looks_like_functions": rust_looks_like_names[:10],
            "python_command_looks_like_functions": python_looks_like_names[:10],
            "python_timeout_phase_keys": python_timeout_phase_keys[:10],
        },
    }


def guard_budget_audit(
    trials_config: dict[str, Any],
    *,
    current_counts: dict[str, Any],
) -> dict[str, Any]:
    raw_budget = guard_convergence_config(trials_config).get("guard_budget", {})
    if not isinstance(raw_budget, dict):
        raw_budget = {}
    baseline_counts = raw_budget.get("baseline_counts", {})
    if not isinstance(baseline_counts, dict):
        baseline_counts = {}
    baseline_total = _optional_int(baseline_counts.get("total_guard_surface"))
    current_total = int(current_counts.get("total_guard_surface") or 0)
    reduction_target_fraction = _optional_float(raw_budget.get("target_reduction_fraction"))
    target_total = _optional_int(raw_budget.get("target_total_guard_surface"))
    if (
        target_total is None
        and baseline_total is not None
        and reduction_target_fraction is not None
    ):
        target_total = int(baseline_total * (1.0 - reduction_target_fraction))
    reduction_from_baseline = None
    if baseline_total:
        reduction_from_baseline = (baseline_total - current_total) / baseline_total
    return {
        "baseline_counts": baseline_counts,
        "baseline_total_guard_surface": baseline_total,
        "current_total_guard_surface": current_total,
        "target_reduction_fraction": reduction_target_fraction,
        "target_total_guard_surface": target_total,
        "reduction_from_baseline": reduction_from_baseline,
        "target_met": target_total is not None and current_total <= target_total,
        "baseline_not_exceeded": baseline_total is not None and current_total <= baseline_total,
        "valid": baseline_total is not None and target_total is not None,
    }


def score_history_from_report(path: Path | None) -> list[float]:
    if path is None:
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        history = data.get("score_history")
        if isinstance(history, list):
            scores = []
            for entry in history:
                if isinstance(entry, dict) and "score" in entry:
                    parsed = _optional_float(entry.get("score"))
                    if parsed is not None:
                        scores.append(parsed)
                elif isinstance(entry, (int, float)):
                    scores.append(float(entry))
            return scores
        state_summaries = data.get("summaries")
        if isinstance(state_summaries, list):
            scores = []
            for entry in state_summaries:
                if not isinstance(entry, dict):
                    continue
                score = _optional_float(entry.get("overall_score"))
                if score is not None:
                    scores.append(score)
            if scores:
                return scores
        campaign_state = data.get("campaign_state")
        if isinstance(campaign_state, dict):
            scores = score_history_from_report_dict(campaign_state)
            if scores:
                return scores
        if "overall_score" in data:
            score = _optional_float(data.get("overall_score"))
            return [] if score is None else [score]
        if "score" in data:
            score = _optional_float(data.get("score"))
            return [] if score is None else [score]
    return []


def score_history_from_report_dict(data: dict[str, Any]) -> list[float]:
    state_summaries = data.get("summaries")
    if not isinstance(state_summaries, list):
        return []
    scores: list[float] = []
    for entry in state_summaries:
        if not isinstance(entry, dict):
            continue
        score = _optional_float(entry.get("overall_score"))
        if score is not None:
            scores.append(score)
    return scores


def fixed_eval_state_report_paths(repo_root: Path, fixed_eval: dict[str, Any]) -> list[Path]:
    summaries_dir = repo_root / "trials" / "summaries"
    if not summaries_dir.exists():
        return []
    expected_tasks = list(fixed_eval.get("tasks") or [])
    paths: list[tuple[str, str, Path]] = []
    for path in summaries_dir.glob("guard-convergence-*_campaign_state.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if expected_tasks and list(data.get("tasks") or []) != expected_tasks:
            continue
        summaries = data.get("summaries")
        if not isinstance(summaries, list) or not summaries:
            continue
        first_summary = next((item for item in summaries if isinstance(item, dict)), {})
        recorded_at = str(first_summary.get("recorded_at") or data.get("updated_at") or "")
        campaign_id = str(data.get("campaign_id") or path.name)
        paths.append((recorded_at, campaign_id, path))
    return [path for _, _, path in sorted(paths)]


def score_history_from_state_reports(paths: list[Path]) -> list[float]:
    scores: list[float] = []
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            scores.extend(score_history_from_report_dict(data))
    return scores


def convergence_audit(
    trials_config: dict[str, Any],
    *,
    score_history: list[float],
    fixed_eval: dict[str, Any],
    guard_budget: dict[str, Any],
) -> dict[str, Any]:
    raw = guard_convergence_config(trials_config).get("convergence", {})
    if not isinstance(raw, dict):
        raw = {}
    plateau_rounds = _optional_int(raw.get("plateau_rounds")) or 3
    min_score_delta = _optional_float(raw.get("min_score_delta"))
    if min_score_delta is None:
        min_score_delta = 0.02
    improvements = [
        max(0.0, score_history[index] - score_history[index - 1])
        for index in range(1, len(score_history))
    ]
    recent_improvements = improvements[-plateau_rounds:]
    plateau_met = (
        len(recent_improvements) >= plateau_rounds
        and all(delta < min_score_delta for delta in recent_improvements)
    )
    latest_score = score_history[-1] if score_history else None
    minimum_accept_score = fixed_eval.get("minimum_accept_score")
    score_gate_met = (
        latest_score is not None
        and minimum_accept_score is not None
        and latest_score >= float(minimum_accept_score)
    )
    converged = bool(score_gate_met and guard_budget.get("target_met") and plateau_met)
    return {
        "plateau_rounds": plateau_rounds,
        "min_score_delta": min_score_delta,
        "score_history": score_history,
        "recent_improvements": recent_improvements,
        "score_gate_met": score_gate_met,
        "guard_budget_target_met": bool(guard_budget.get("target_met")),
        "plateau_met": plateau_met,
        "converged": converged,
        "stop_condition": (
            "fixed-eval score is non-regressed, guard target is met, "
            "and recent score gains are below epsilon"
            if converged
            else "not converged; continue fixed-eval-gated guard reduction"
        ),
    }


def build_guard_convergence_audit(
    *,
    repo_root: Path,
    trials_config_path: Path,
    fixed_eval_report: Path | None = None,
) -> dict[str, Any]:
    trials_config = load_trials_config(trials_config_path)
    current_counts = count_guard_surface(repo_root)
    fixed_eval = fixed_eval_audit(trials_config)
    guard_budget = guard_budget_audit(trials_config, current_counts=current_counts)
    state_report_paths: list[Path] = []
    if fixed_eval_report is None:
        state_report_paths = fixed_eval_state_report_paths(repo_root, fixed_eval)
        score_history = score_history_from_state_reports(state_report_paths)
    else:
        score_history = score_history_from_report(fixed_eval_report)
    convergence = convergence_audit(
        trials_config,
        score_history=score_history,
        fixed_eval=fixed_eval,
        guard_budget=guard_budget,
    )
    ready = bool(fixed_eval.get("valid") and guard_budget.get("valid"))
    return {
        "schema_version": 1,
        "repo_root": str(repo_root.resolve()),
        "trials_config_path": str(trials_config_path),
        "fixed_eval_report": str(fixed_eval_report) if fixed_eval_report else "",
        "fixed_eval_state_reports": [str(path) for path in state_report_paths],
        "ready_for_guard_reduction": ready,
        "guard_deletion_acceptance_gate": "latest fixed-eval score must be >= fixed_eval.minimum_accept_score",
        "fixed_eval": fixed_eval,
        "guard_counts": current_counts,
        "guard_budget": guard_budget,
        "convergence": convergence,
    }


def _read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
