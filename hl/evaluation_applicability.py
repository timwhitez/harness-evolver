"""Applicability conditions and provenance helpers for change evaluations.

Distinguishes three independent layers for historical change evaluations:
1. Comparable result conditions: actual Worker config, task/split identity,
   evaluator, and tool/environment interface.
2. Updater provenance: which outer updater runner/model/reasoning produced the patch.
3. Harness change identity: base -> candidate commit/diff/packet identity.

Changing updater provenance or candidate commit alone never makes an evaluation
non-comparable. Missing, partial, conflicting, or non-unique source records
resolve strictly to ``unknown`` without backfilling from the current run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hl.model_scope import MODEL_SCOPE_KEYS, model_scope_from_config, normalize_model_config

COMPARABILITY_DIMENSIONS = (
    "worker_config",
    "task_split",
    "evaluator",
    "tool_env_interface",
)


def parse_model_scope_string(model_scope: str) -> dict[str, str]:
    """Parse a canonical ``key=value|...`` model_scope string into a dict."""
    parsed: dict[str, str] = {}
    for segment in str(model_scope or "").split("|"):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in MODEL_SCOPE_KEYS and value:
            parsed[key] = value
    return parsed


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _normalize_worker_config(
    raw_config: Mapping[str, Any] | None,
    *,
    fallback_model: str = "",
    raw_model_scope: str = "",
) -> tuple[dict[str, str] | None, str | None, dict[str, str]]:
    """Return (canonical_dict, canonical_scope, partial_hints) or (None, None, hints)."""
    merged: dict[str, Any] = {}
    if raw_model_scope:
        merged.update(parse_model_scope_string(raw_model_scope))
    if isinstance(raw_config, Mapping):
        for key, value in raw_config.items():
            if value is not None and str(value) != "":
                merged[key] = value
    normalized = normalize_model_config(merged, fallback_model=fallback_model)
    if not normalized:
        return None, None, {}
    # Require both provider and model for a minimum known Worker identity.
    if not normalized.get("provider") or not normalized.get("model"):
        return None, None, dict(normalized)
    scope = model_scope_from_config(normalized)
    if not scope:
        return None, None, dict(normalized)
    return normalized, scope, {}


def _normalize_task_split(mapping: Mapping[str, Any]) -> tuple[Any | None, bool]:
    explicit = mapping.get("task_split")
    if isinstance(explicit, Mapping):
        cleaned = {
            str(k): str(v).strip()
            for k, v in explicit.items()
            if v is not None and str(v).strip() != ""
        }
        has_identity = any(
            k in cleaned for k in ("split", "task_split", "holdout_mode", "dataset", "task_pool")
        )
        return (cleaned, False) if has_identity else (None, bool(cleaned))
    if isinstance(explicit, str) and explicit.strip():
        return {"split": explicit.strip()}, False

    parts: dict[str, str] = {}
    for key in ("split", "holdout_mode", "task_pool"):
        val = mapping.get(key)
        if val is not None and str(val).strip():
            parts[key] = str(val).strip()
    dataset = mapping.get("dataset") or mapping.get("dataset_path")
    if dataset is not None and str(dataset).strip():
        raw_dataset = str(dataset).strip().rstrip("/")
        parts["dataset"] = Path(raw_dataset).name if "/" in raw_dataset else raw_dataset
    task_meta = mapping.get("task_metadata")
    if isinstance(task_meta, Mapping):
        for key in ("split", "task_split", "holdout_mode", "dataset"):
            val = task_meta.get(key)
            if val is not None and str(val).strip() and key not in parts:
                parts[key] = str(val).strip()
    if parts:
        for extra_key in ("holdout_fraction", "holdout_seed"):
            val = mapping.get(extra_key)
            if val is not None and str(val).strip() != "":
                parts[extra_key] = str(val).strip()
        return parts, False

    partial = any(
        mapping.get(k) is not None and str(mapping.get(k)).strip() != ""
        for k in ("holdout_fraction", "holdout_seed")
    )
    return None, partial


def _normalize_evaluator(mapping: Mapping[str, Any]) -> tuple[Any | None, bool]:
    explicit = mapping.get("evaluator")
    if isinstance(explicit, Mapping):
        cleaned = {
            str(k): str(v).strip()
            for k, v in explicit.items()
            if v is not None and str(v).strip() != ""
        }
        return (cleaned, False) if cleaned else (None, False)
    if isinstance(explicit, str) and explicit.strip():
        return {"evaluator": explicit.strip()}, False

    parts: dict[str, str] = {}
    for key in ("verifier", "harbor_version", "evaluator_interface", "agent"):
        val = mapping.get(key)
        if val is not None and str(val).strip():
            parts[key] = str(val).strip()
    if parts:
        return parts, False
    return None, False


def _normalize_tool_env_interface(mapping: Mapping[str, Any]) -> tuple[Any | None, bool]:
    explicit = mapping.get("tool_env_interface") or mapping.get("tool_interface") or mapping.get("environment_interface")
    if isinstance(explicit, Mapping):
        cleaned = {
            str(k): (v if isinstance(v, bool) else str(v).strip())
            for k, v in explicit.items()
            if v is not None and str(v).strip() != ""
        }
        return (cleaned, False) if cleaned else (None, False)
    if isinstance(explicit, str) and explicit.strip():
        return {"interface": explicit.strip()}, False

    parts: dict[str, Any] = {}
    for key in (
        "tool_result_archive_enabled",
        "network_hardened_environment",
        "sandbox",
        "environment_type",
        "harness_interface",
    ):
        if key in mapping and mapping.get(key) is not None:
            val = mapping.get(key)
            if isinstance(val, bool) or str(val).strip():
                parts[key] = val if isinstance(val, bool) else str(val).strip()
    if parts:
        return parts, False
    return None, False


def extract_condition_observation(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract a single-artifact condition observation (never backfilling)."""
    if not isinstance(source, Mapping):
        return {"values": {}, "partial": {}, "partial_worker": {}}

    values: dict[str, Any] = {}
    partial: dict[str, str] = {}
    partial_worker: dict[str, str] = {}

    nested = source.get("comparability_conditions") or source.get("evaluation_conditions")
    merged_source: dict[str, Any] = dict(source)
    if isinstance(nested, Mapping):
        for k, v in nested.items():
            if k not in merged_source or merged_source[k] in (None, "", {}):
                merged_source[k] = v

    raw_worker = (
        merged_source.get("worker_config")
        or merged_source.get("model_config")
        or merged_source.get("agent_config")
    )
    fallback_model = str(merged_source.get("model_used") or merged_source.get("model") or "")
    raw_scope = str(merged_source.get("model_scope") or "")
    worker_dict, worker_scope, worker_partial = _normalize_worker_config(
        raw_worker if isinstance(raw_worker, Mapping) else None,
        fallback_model=fallback_model,
        raw_model_scope=raw_scope,
    )
    if worker_scope and worker_dict is not None:
        values["worker_config"] = {"scope": worker_scope, "config": worker_dict}
    elif worker_partial:
        partial["worker_config"] = "partial_worker_config"
        partial_worker = worker_partial

    task_split, split_partial = _normalize_task_split(merged_source)
    if task_split is not None:
        values["task_split"] = task_split
    elif split_partial:
        partial["task_split"] = "partial_task_split"

    evaluator, eval_partial = _normalize_evaluator(merged_source)
    if evaluator is not None:
        values["evaluator"] = evaluator
    elif eval_partial:
        partial["evaluator"] = "partial_evaluator"

    tool_env, tool_partial = _normalize_tool_env_interface(merged_source)
    if tool_env is not None:
        values["tool_env_interface"] = tool_env
    elif tool_partial:
        partial["tool_env_interface"] = "partial_tool_env_interface"

    return {
        "values": values,
        "partial": partial,
        "partial_worker": partial_worker,
    }


def extract_trial_observation(trial: Any) -> dict[str, Any]:
    """Extract condition observation from a TrialResult or trial summary dict."""
    if isinstance(trial, Mapping):
        metadata = trial.get("metadata") if isinstance(trial.get("metadata"), Mapping) else {}
        combined = {**dict(metadata), **dict(trial)}
        return extract_condition_observation(combined)
    metadata = getattr(trial, "metadata", {}) or {}
    combined: dict[str, Any] = dict(metadata) if isinstance(metadata, Mapping) else {}
    model_used = str(getattr(trial, "model_used", "") or "")
    if model_used and "model_used" not in combined:
        combined["model_used"] = model_used
    return extract_condition_observation(combined)


def merge_condition_observations(
    observations: list[dict[str, Any]],
    *,
    ambiguous_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Merge condition observations, marking missing/partial/conflicting as unknown."""
    merged_values: dict[str, Any] = {}
    status_by_dim: dict[str, str] = {}
    reasons: list[str] = list(ambiguous_reasons or [])

    if ambiguous_reasons:
        for dim in COMPARABILITY_DIMENSIONS:
            status_by_dim[dim] = "unknown"
        return {
            "complete": False,
            "conditions": {},
            "status_by_dimension": status_by_dim,
            "reasons": sorted(set(reasons)),
        }

    for dim in COMPARABILITY_DIMENSIONS:
        distinct_by_key: dict[str, Any] = {}
        partial_reason = ""
        partial_worker_hints: list[dict[str, str]] = []
        for obs in observations:
            val = (obs.get("values") or {}).get(dim)
            if val is not None:
                key = val["scope"] if dim == "worker_config" else _canonical_json(val)
                distinct_by_key[key] = val
            p_reason = (obs.get("partial") or {}).get(dim)
            if p_reason and not partial_reason:
                partial_reason = p_reason
            if dim == "worker_config" and obs.get("partial_worker"):
                partial_worker_hints.append(dict(obs["partial_worker"]))

        if len(distinct_by_key) > 1:
            status_by_dim[dim] = "unknown"
            reasons.append(f"conflicting_{dim}")
        elif len(distinct_by_key) == 1:
            only_val = next(iter(distinct_by_key.values()))
            if dim == "worker_config" and partial_worker_hints:
                # A partial hint from another source conflicts if any key disagrees.
                only_cfg = only_val.get("config") or {}
                conflict = any(
                    hint_val != only_cfg.get(hint_key)
                    for hint in partial_worker_hints
                    for hint_key, hint_val in hint.items()
                    if hint_key in only_cfg and hint_val
                )
                if conflict:
                    status_by_dim[dim] = "unknown"
                    reasons.append("conflicting_worker_config")
                    continue
            merged_values[dim] = only_val
            status_by_dim[dim] = "known"
        else:
            status_by_dim[dim] = "unknown"
            reasons.append(partial_reason or f"missing_{dim}")

    complete = all(status_by_dim[dim] == "known" for dim in COMPARABILITY_DIMENSIONS)
    return {
        "complete": complete,
        "conditions": merged_values,
        "status_by_dimension": status_by_dim,
        "reasons": sorted(set(reasons)),
    }


def _dimension_canonical_key(dim: str, value: Any) -> str:
    if dim == "worker_config" and isinstance(value, Mapping):
        return str(value.get("scope") or "")
    return _canonical_json(value)


def compare_evaluation_applicability(
    historical: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare historical vs current comparability conditions.

    Empty equality never proves compatibility: if either side is missing,
    partial, conflicting, or ambiguous on any dimension without a known
    mismatch on another dimension, applicability is ``unknown``.
    """
    hist_conds = historical.get("conditions") or {}
    curr_conds = current.get("conditions") or {}
    hist_reasons = list(historical.get("reasons") or [])
    curr_reasons = list(current.get("reasons") or [])

    mismatched: list[str] = []
    missing_or_conflicting: list[str] = []

    for dim in COMPARABILITY_DIMENSIONS:
        h_val = hist_conds.get(dim)
        c_val = curr_conds.get(dim)
        if h_val is not None and c_val is not None:
            if _dimension_canonical_key(dim, h_val) != _dimension_canonical_key(dim, c_val):
                mismatched.append(dim)
        else:
            missing_or_conflicting.append(dim)

    if mismatched:
        status = "cross_environment"
        comparable = False
        reasons = [f"mismatched_{dim}" for dim in mismatched]
    elif missing_or_conflicting or hist_reasons or curr_reasons:
        status = "unknown"
        comparable = False
        reasons = sorted(
            set(
                hist_reasons
                + [f"current_{r}" for r in curr_reasons]
                + [f"incomplete_{dim}" for dim in missing_or_conflicting]
            )
        )
    else:
        status = "applicable"
        comparable = True
        reasons = ["all_minimum_conditions_matched"]

    return {
        "status": status,
        "comparable": comparable,
        "mismatched_dimensions": mismatched,
        "missing_or_conflicting_dimensions": missing_or_conflicting,
        "reasons": reasons,
        "historical_conditions": dict(hist_conds),
        "current_conditions": dict(curr_conds),
    }


def extract_updater_provenance(source: Mapping[str, Any] | None) -> dict[str, str]:
    """Extract outer updater provenance separately from result comparability."""
    if not isinstance(source, Mapping):
        return {}
    nested = source.get("updater_provenance")
    merged = dict(source)
    if isinstance(nested, Mapping):
        merged.update(nested)
    provenance: dict[str, str] = {}
    for out_key, in_keys in (
        ("updater_runner", ("updater_runner", "runner")),
        ("updater_model", ("updater_model", "model")),
        ("updater_reasoning_effort", ("updater_reasoning_effort", "reasoning_effort")),
    ):
        for key in in_keys:
            val = merged.get(key)
            if val is not None and str(val).strip():
                provenance[out_key] = str(val).strip()
                break
    return provenance


def extract_harness_change_identity(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract base -> candidate harness change identity separately from comparability."""
    if not isinstance(source, Mapping):
        return {}
    nested = source.get("harness_change_identity")
    merged = dict(source)
    if isinstance(nested, Mapping):
        merged.update(nested)
    identity: dict[str, Any] = {}
    for key in ("packet_id", "base_commit", "candidate_commit", "diff_sha256", "base_harness_version"):
        val = merged.get(key)
        if val is not None and str(val).strip():
            identity[key] = str(val).strip()
    changed_files = merged.get("changed_files")
    if isinstance(changed_files, list) and changed_files:
        identity["changed_files"] = [str(item) for item in changed_files[:20]]
    return identity
