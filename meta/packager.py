"""Build structured Codex work packets from HL failure artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from hl.attribution import FailureAttributor
from hl.failure_mechanisms import (
    DEPENDENCY_LOOP_BASE_REPLACEMENT_NEUTRAL_MECHANISM_NAMES,
    PRIMARY_VERIFIER_CONTRACT_MECHANISM_NAMES,
    affected_components_for_failure_mechanism,
    dependency_loop_failure_category_for_trial,
    dependency_loop_mechanism_for_failure_category,
    failure_mechanisms_replace_base_components,
    failure_mechanisms_for_trial,
)
from hl.frontier import frontier_summary
from hl.goals import GoalStore
from hl.loop_limits import (
    normalize_legacy_limit_driven_skip_event,
    sub_agent_creation_policy,
)
from hl.model_scope import model_scope_from_trial, safe_model_scope_name
from hl.types import TrialResult
from hl.web_references import WECHAT_ARTICLE_USER_AGENT
from meta.missions import MissionFeatureCandidate, MissionPlanner
from meta.mechanism_coverage import covered_mechanism_evidence
from meta.reviewer import PatchReviewer
from meta.update_policy import validation_ladder_contract


DEFAULT_REJECTED_UPDATE_BUFFER_LIMIT = 12
DEFAULT_CAMPAIGN_STATE_CONTEXT_LIMIT = 12
DEFAULT_CAMPAIGN_SUMMARY_CONTEXT_LIMIT = 12
DEFAULT_CAMPAIGN_TRIAL_CONTEXT_LIMIT = 120
DEFAULT_ANALYSIS_REPORT_CONTEXT_LIMIT = 12
DEFAULT_CODEX_UPDATE_EVENT_CONTEXT_LIMIT = 24
DEFAULT_UPDATE_HISTORY_CONTEXT_LIMIT = 12


MECHANISM_CATEGORY_OVERRIDES = {
    "async_cancellation_cleanup_contract": "async_cancellation_cleanup_contract",
    "model_extraction_matrix_contract": "model_extraction_matrix_contract",
    "stan_dependency_stack_pivot_mechanism": "stan_dependency_stack_pivot_mechanism",
    "fasttext_artifact_pivot_mechanism": "fasttext_artifact_pivot_mechanism",
    "cross_arch_toolchain_pivot_mechanism": "cross_arch_toolchain_pivot_mechanism",
    "cython_extension_optional_import_pivot_mechanism": (
        "cython_extension_optional_import_pivot_mechanism"
    ),
    "numpy_eigensolver_dependency_pivot_mechanism": (
        "numpy_eigensolver_dependency_pivot_mechanism"
    ),
    "terminal_environment_unavailable_after_dependency_loop_mechanism": (
        "terminal_environment_unavailable_after_dependency_loop"
    ),
    "dependency_loop_without_deliverable_progress_mechanism": (
        "dependency_loop_without_deliverable_progress"
    ),
}


INFRASTRUCTURE_PHASE_ATTRIBUTION = {
    "environment_start": (
        "environment_start_timeout",
        ("bench/harbor", "bench/network_environment"),
    ),
    "environment_build": (
        "environment_build_timeout",
        ("bench/harbor", "bench/network_environment"),
    ),
    "verifier_runtime_prepare": (
        "verifier_runtime_prepare_timeout",
        ("bench/harbor", "bench/network_environment"),
    ),
    "harbor_process": (
        "harbor_process_timeout",
        ("bench/harbor", "orchestration/run_campaign"),
    ),
    "harbor_cancelled": (
        "harbor_process_timeout",
        ("bench/harbor", "orchestration/run_campaign"),
    ),
}


def _demote_discouraged_patterns(
    patterns: list[dict[str, Any]],
    discouraged_categories: set[str],
) -> list[dict[str, Any]]:
    """Rank failure patterns, demoting repeatedly-failed (discouraged) categories.

    Codex updates that keep targeting the highest-count failure class can stall
    when that class is an already-repeatedly-failed direction (a treadmill). This
    keeps the default count ordering but pushes discouraged categories below fresh
    ones, so the loop pivots to a different, not-yet-exhausted surface. Discouraged
    patterns are demoted, never dropped: if every candidate is discouraged, the
    normal count ordering still applies so a target always exists.
    """
    normalized_discouraged = {
        str(category).strip().lower()
        for category in discouraged_categories
        if str(category).strip()
    }

    def sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
        category = str(item.get("failure_category") or "")
        is_discouraged = 1 if category.strip().lower() in normalized_discouraged else 0
        return (is_discouraged, -int(item.get("count") or 0), category)

    return sorted(patterns, key=sort_key)


class CodexWorkPacket(BaseModel):
    packet_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    hl_goal: str
    failing_tasks: list[dict[str, Any]]
    trajectory_slices: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    verifier_outputs: dict[str, str] = Field(default_factory=dict)
    tool_failures: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    failure_artifacts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    campaign_context: dict[str, Any] = Field(default_factory=dict)
    failure_pattern_digest: dict[str, Any] = Field(default_factory=dict)
    current_harness: dict[str, Any] = Field(default_factory=dict)
    allowed_edit_paths: list[str] = Field(default_factory=list)
    forbidden_paths: list[str] = Field(default_factory=list)
    regression_contracts: list[str] = Field(default_factory=list)
    required_validation_commands: list[str] = Field(default_factory=list)
    expected_report_schema: dict[str, Any] = Field(default_factory=dict)
    mission_debug: dict[str, Any] = Field(default_factory=dict)
    generalization_contract: dict[str, Any] = Field(default_factory=dict)
    leaderboard_compliance_contract: dict[str, Any] = Field(default_factory=dict)
    heuristic_learning_contract: dict[str, Any] = Field(default_factory=dict)
    update_memory_contract: dict[str, Any] = Field(default_factory=dict)
    framework_comparison_contract: dict[str, Any] = Field(default_factory=dict)
    architecture_update_contract: dict[str, Any] = Field(default_factory=dict)
    official_evaluation_contract: dict[str, Any] = Field(default_factory=dict)
    cross_round_update_contract: dict[str, Any] = Field(default_factory=dict)
    harness_reference_contract: dict[str, Any] = Field(default_factory=dict)
    validation_ladder_contract: dict[str, Any] = Field(default_factory=dict)
    same_model_frontier: dict[str, Any] = Field(default_factory=dict)
    runner_pivot_policy: dict[str, Any] = Field(default_factory=dict)
    mission_selection_contract: dict[str, Any] = Field(default_factory=dict)
    update_history: dict[str, Any] = Field(default_factory=dict)
    change_evaluation_digest: dict[str, Any] = Field(default_factory=dict)
    policy_recurrence_signals: list[dict[str, Any]] = Field(default_factory=list)
    infrastructure_triage: dict[str, Any] = Field(default_factory=dict)
    self_harness_improvement_queue: dict[str, Any] = Field(default_factory=dict)
    update_search_policy: dict[str, Any] = Field(default_factory=dict)
    self_iteration_contract: dict[str, Any] = Field(default_factory=dict)
    rejected_update_buffer: list[dict[str, Any]] = Field(default_factory=list)
    prior_update_lessons: list[str] = Field(default_factory=list)
    prior_update_lesson_entries: list[dict[str, Any]] = Field(default_factory=list)
    external_research_policy: dict[str, Any] = Field(default_factory=dict)
    sub_agent_creation_policy: dict[str, Any] = Field(default_factory=dict)
    report_contract_rules: dict[str, Any] = Field(default_factory=dict)
    report_value_budget: dict[str, Any] = Field(default_factory=dict)


class WorkPacketBuilder:
    """Create the structured input contract for one bounded Codex update."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        goal_store: GoalStore | None = None,
        memory_path: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.goal_store = goal_store or GoalStore()
        if memory_path is None:
            self.memory_path = self.repo_root / "trials"
        else:
            raw_memory_path = Path(memory_path)
            self.memory_path = (
                raw_memory_path
                if raw_memory_path.is_absolute()
                else self.repo_root / raw_memory_path
            )
        self._campaign_states_cache: list[dict[str, Any]] | None = None
        self._completed_trial_full_context_threshold = 64

    def build(
        self,
        *,
        failures: list[TrialResult],
        current_harness: dict[str, Any],
        allowed_edit_paths: list[str] | None = None,
        regression_contracts: list[str] | None = None,
        required_validation_commands: list[str] | None = None,
    ) -> CodexWorkPacket:
        packet_id = f"codex_packet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        goal_prompt = self.goal_store.continuation_prompt()
        allowed = allowed_edit_paths or [
            "bench",
            "harness",
            "crates",
            "hl",
            "meta",
            "scripts",
            "config",
            "tests",
        ]
        forbidden = [
            "terminal-bench-tasks",
            "terminal-bench",
            "trials/runs",
            "trials/regressions",
            "trials/submissions",
            "jobs",
        ]
        self._campaign_states_cache = self._load_campaign_states(
            limit=DEFAULT_CAMPAIGN_STATE_CONTEXT_LIMIT
        )
        try:
            campaign_context = self._campaign_context(
                failures,
                summary_limit=DEFAULT_CAMPAIGN_SUMMARY_CONTEXT_LIMIT,
                trial_limit=DEFAULT_CAMPAIGN_TRIAL_CONTEXT_LIMIT,
            )
            failure_pattern_digest = self._failure_pattern_digest(
                failures,
                campaign_context,
            )
            failing_task_summaries = [self._trial_summary(trial) for trial in failures]
            policy_recurrence_signals = self._policy_recurrence_signals_from_reports(
                campaign_context.get("recent_analysis_reports")
            )
            infrastructure_triage = self._infrastructure_triage(
                failing_task_summaries,
                campaign_context,
            )
            self_harness_improvement_queue = self._self_harness_improvement_queue(
                failure_pattern_digest=failure_pattern_digest,
                policy_recurrence_signals=policy_recurrence_signals,
                infrastructure_triage=infrastructure_triage,
                campaign_context=campaign_context,
                failing_tasks=failing_task_summaries,
            )
            update_history = self._update_history(
                limit=DEFAULT_UPDATE_HISTORY_CONTEXT_LIMIT
            )
            mission_debug = self._mission_debug_from_failures(
                packet_id,
                failing_task_summaries,
            )
            rejected_update_buffer = self._rejected_update_buffer(
                limit=DEFAULT_REJECTED_UPDATE_BUFFER_LIMIT
            )
            return CodexWorkPacket(
                packet_id=packet_id,
                hl_goal=goal_prompt,
                failing_tasks=failing_task_summaries,
                trajectory_slices={
                    trial.trial_id: self._full_trajectory(trial) for trial in failures
                },
                verifier_outputs={
                    trial.trial_id: self._full_verifier_output(trial)
                    for trial in failures
                },
                tool_failures={
                    trial.trial_id: self._failed_tool_calls(trial)
                    for trial in failures
                },
                failure_artifacts={
                    trial.trial_id: self._failure_artifacts(trial)
                    for trial in failures
                },
                campaign_context=campaign_context,
                failure_pattern_digest=failure_pattern_digest,
                current_harness=current_harness,
                allowed_edit_paths=allowed,
                forbidden_paths=forbidden,
                regression_contracts=regression_contracts or [],
                required_validation_commands=required_validation_commands
                or ["pytest tests/ -v"],
                expected_report_schema=self.report_schema(),
                mission_debug=mission_debug,
                generalization_contract=self._generalization_contract(failures),
                leaderboard_compliance_contract=self._leaderboard_compliance_contract(),
                heuristic_learning_contract=self._heuristic_learning_contract(),
                update_memory_contract=self._update_memory_contract(),
                framework_comparison_contract=self._framework_comparison_contract(),
                architecture_update_contract=self._architecture_update_contract(),
                official_evaluation_contract=self._official_evaluation_contract(),
                cross_round_update_contract=self._cross_round_update_contract(),
                harness_reference_contract=self._harness_reference_contract(),
                validation_ladder_contract=validation_ladder_contract(),
                same_model_frontier=self._same_model_frontier(failures),
                runner_pivot_policy=self._runner_pivot_policy(),
                mission_selection_contract=self._mission_selection_contract(),
                update_history=update_history,
                change_evaluation_digest=self._change_evaluation_digest(),
                policy_recurrence_signals=policy_recurrence_signals,
                infrastructure_triage=infrastructure_triage,
                self_harness_improvement_queue=self_harness_improvement_queue,
                update_search_policy=self._update_search_policy(),
                self_iteration_contract=self._self_iteration_contract(),
                rejected_update_buffer=rejected_update_buffer,
                prior_update_lessons=self._prior_update_lessons(),
                prior_update_lesson_entries=self._prior_update_lesson_entries(),
                external_research_policy=self._external_research_policy(update_history),
                sub_agent_creation_policy=self._sub_agent_creation_policy(),
                report_contract_rules=self._report_contract_rules(),
                report_value_budget=self._report_value_budget(
                    failure_pattern_digest=failure_pattern_digest,
                    mission_debug=mission_debug,
                    rejected_update_buffer=rejected_update_buffer,
                ),
            )
        finally:
            self._campaign_states_cache = None

    def write(self, packet: CodexWorkPacket, path: str | Path) -> Path:
        packet_path = Path(path)
        packet_path.parent.mkdir(parents=True, exist_ok=True)
        packet_path.write_text(packet.model_dump_json(indent=2))
        return packet_path

    def _mission_debug_from_failures(
        self,
        packet_id: str,
        failing_task_summaries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        planner = MissionPlanner()
        task_results = [
            self._mission_debug_task_result(summary)
            for summary in failing_task_summaries
        ]
        summary = {
            "campaign_id": packet_id,
            "task_results": task_results,
            "tasks_completed": len(task_results),
            "tasks_pending": 0,
            "score_history": [self._mission_debug_score_history(task_results)],
        }
        coverage_evidence = covered_mechanism_evidence(
            self.repo_root,
            self.memory_path,
        )
        mission = planner.from_campaign_summary(
            summary,
            source_path=packet_id,
            covered_mechanism_signatures=[
                str(entry.get("signature") or "") for entry in coverage_evidence
            ],
        ).model_dump(mode="json")
        mission["evidence_summary"]["covered_mechanism_evidence"] = coverage_evidence
        return self.prepare_mission_debug(mission)

    def prepare_mission_debug(self, mission_debug: dict[str, Any]) -> dict[str, Any]:
        """Select one uncovered candidate or preserve a structured skip decision."""

        mission = json.loads(json.dumps(mission_debug))
        candidates = self._mission_candidate_dicts(mission)
        audit = mission.get("candidate_audit")
        if not isinstance(audit, list) or not audit:
            mission["candidate_audit"] = list(candidates)
        evidence = mission.setdefault("evidence_summary", {})
        coverage_evidence = covered_mechanism_evidence(
            self.repo_root,
            self.memory_path,
        )
        evidence["covered_mechanism_evidence"] = coverage_evidence
        signatures = [
            str(entry.get("signature") or "") for entry in coverage_evidence
        ]
        planner = MissionPlanner()
        validated_candidates = [
            MissionFeatureCandidate.model_validate(candidate)
            for candidate in candidates
        ]
        kept, filtered_ids = planner.filter_covered_candidates(
            validated_candidates,
            signatures,
        )
        candidates = [candidate.model_dump(mode="json") for candidate in kept]
        if filtered_ids:
            evidence["filtered_covered_candidate_ids"] = filtered_ids
        if not candidates and filtered_ids:
            evidence["all_candidates_covered"] = True
            evidence["skip_codex_update"] = {
                "reason": "all mission candidates are already covered by current policy/tests or accepted update memory",
                "covered_candidate_ids": filtered_ids,
                "covered_mechanism_signatures": signatures,
            }
        if evidence.get("all_candidates_covered") or not candidates:
            evidence.setdefault(
                "skip_codex_update",
                {
                    "reason": "no uncovered mission candidate is available",
                    "covered_candidate_ids": evidence.get(
                        "filtered_covered_candidate_ids", []
                    ),
                },
            )
            mission["feature_candidates"] = []
            evidence["selected_candidate_id"] = ""
            return mission
        selected = candidates[0]
        selected_id = str(selected.get("id") or "").strip()
        if not selected_id:
            evidence["skip_codex_update"] = {
                "reason": "selected mission candidate has no stable id",
                "covered_candidate_ids": [],
            }
            mission["feature_candidates"] = []
            evidence["selected_candidate_id"] = ""
            return mission
        mission["feature_candidates"] = [selected]
        evidence["selected_candidate_id"] = selected_id
        evidence["selection_reason"] = (
            "highest-ranked uncovered candidate from deterministic mission ordering"
        )
        return mission

    def _mission_debug_score_history(
        self,
        task_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        counts = {"passed": 0, "failed": 0, "timeout": 0, "error": 0}
        for item in task_results:
            status = str(item.get("status") or "").lower()
            if status in counts:
                counts[status] += 1
        completed = max(len(task_results), 1)
        return {
            "score": round(counts["passed"] / completed, 4),
            **counts,
        }

    def _mission_debug_task_result(self, summary: dict[str, Any]) -> dict[str, Any]:
        result = dict(summary)
        failure_category = str(result.get("failure_category") or "").lower()
        status = str(result.get("status") or "").lower()
        mechanisms = result.get("failure_mechanisms") or []
        if failure_category and failure_category == status and not mechanisms:
            result.pop("failure_category", None)
        return result

    def report_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": ["edited", "noop", "rejected"]},
                "summary": {"type": "string"},
                "changed_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Must exactly match the reviewed git diff file list. "
                        "Do not omit changed files or include files not changed by the diff."
                    ),
                },
                "validation_commands": {"type": "array", "items": {"type": "string"}},
                "skipped_validation_reason": {"type": "string"},
                "strategy_confidence": {
                    "type": "string",
                    "description": (
                        "For edited patches, report high/medium/low confidence after "
                        "loophole review and validation planning."
                    ),
                    "enum": ["high", "medium", "low"],
                },
                "loophole_review": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For edited patches, list at least one concrete reviewed "
                        "loophole, counterexample, regression risk, or benchmark "
                        "integrity risk."
                    ),
                },
                "loophole_fixes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For edited patches, list at least one concrete mitigation "
                        "for the reviewed loopholes or risks."
                    ),
                },
                "component_type": {
                    "type": "string",
                    "description": (
                        "Must match the primary changed-file layer derived from the "
                        "actual diff, using this schema's coarse categories."
                    ),
                    "enum": [
                        "prompt",
                        "tool",
                        "planning",
                        "recovery",
                        "context",
                        "config",
                        "adapter",
                        "memory",
                        "verification",
                        "architecture",
                        "orchestration",
                        "harbor_integration",
                        "other",
                    ],
                },
                "implementation_scope": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "primary_layer": {
                            "type": "string",
                            "description": (
                                "Must match the primary changed-file layer derived "
                                "from the actual diff unless component_type already "
                                "does."
                            ),
                            "enum": [
                                "prompt",
                                "tool",
                                "planning",
                                "recovery",
                                "context",
                                "config",
                                "adapter",
                                "memory",
                                "verification",
                                "architecture",
                                "orchestration",
                                "harbor_integration",
                                "other",
                            ],
                        },
                        "architectural_change_considered": {"type": "boolean"},
                        "structural_files_changed": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Must exactly list structural non-test, non-doc files "
                                "changed by the reviewed diff; tests and docs are not "
                                "required here."
                            ),
                        },
                        "why_prompt_only_is_sufficient": {"type": "string"},
                    },
                    "required": [
                        "primary_layer",
                        "architectural_change_considered",
                        "structural_files_changed",
                        "why_prompt_only_is_sufficient",
                    ],
                },
                "generalization": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "problem_class": {
                            "type": "string",
                            "description": (
                                "Name the reusable class using concrete "
                                "failure_pattern_digest labels when available."
                            ),
                        },
                        "applies_to": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List reusable task classes or components, grounded "
                                "in failure_pattern_digest labels when available."
                            ),
                        },
                        "anti_overfit_checks": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "why_not_task_specific": {"type": "string"},
                    },
                    "required": [
                        "problem_class",
                        "applies_to",
                        "anti_overfit_checks",
                        "why_not_task_specific",
                    ],
                },
                "cross_round_evidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "used": {"type": "boolean"},
                        "recent_summary_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "dominant_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Use concrete failure_pattern_digest labels, "
                                "such as failure_category or affected_components."
                            ),
                        },
                        "selected_problem_class": {
                            "type": "string",
                            "description": (
                                "Name the failure_pattern_digest label selected for "
                                "this bounded update."
                            ),
                        },
                        "why_this_slice_generalizes": {"type": "string"},
                    },
                    "required": [
                        "used",
                        "recent_summary_ids",
                        "dominant_patterns",
                        "selected_problem_class",
                        "why_this_slice_generalizes",
                    ],
                },
                "memory_record": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "concise": {"type": "string"},
                        "detailed": {"type": "string"},
                        "failed_directions_to_avoid": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "When rejected_update_buffer is non-empty, include "
                                "at least one concrete mission_candidate_id, packet_id, "
                                "failure_class, or component_layer marker for every "
                                "rejected buffer entry. "
                                "When runner_pivot_policy.discouraged is non-empty, "
                                "cover each discouraged failure_class or component_layer. "
                                "When runner_pivot_policy.layer_pressure is non-empty, "
                                "also cover each pressured component_layer plus a recent "
                                "packet_id or failure_class when available. "
                                "When a rejected buffer entry includes loophole_review "
                                "or loophole_fixes evidence, reference at least one prior "
                                "risk or mitigation so the next patch mutates the reviewed "
                                "gap instead of only naming the packet. "
                                "When a rejected buffer entry includes required_mutation, "
                                "reference that guidance with a concrete marker such as "
                                "no-diff update, tracked Worker/harness change, required "
                                "validation, allowed edit roots, fresh trajectory, verifier "
                                "evidence, or missing evidence. "
                                "When change_evaluation_digest.miss_classes is non-empty, "
                                "cover the top miss-heavy class labels so repeated poor "
                                "directions are explicitly avoided or mutated."
                            ),
                        },
                        "supported_directions_to_preserve": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "When runner_pivot_policy.supported is non-empty, "
                                "include concrete packet_id, failure_class, or "
                                "component_layer markers for supported directions the "
                                "patch preserves or intentionally extends."
                            ),
                        },
                    },
                    "required": [
                        "concise",
                        "detailed",
                        "failed_directions_to_avoid",
                        "supported_directions_to_preserve",
                    ],
                },
                "framework_comparison": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "before": {"type": "string"},
                        "after": {"type": "string"},
                        "expected_effect": {"type": "string"},
                        "rollback_trigger": {"type": "string"},
                    },
                    "required": [
                        "before",
                        "after",
                        "expected_effect",
                        "rollback_trigger",
                    ],
                },
                "prediction": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "expected_fixed_task_classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Use concrete labels from failure_pattern_digest, "
                                "change_evaluation_digest, or rejected_update_buffer; "
                                "generic task-type guesses are rejected when evidence "
                                "labels are available."
                            ),
                        },
                        "risk_task_classes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "expected_metric_delta": {"type": "number"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "falsification_window": {"type": "string"},
                    },
                    "required": [
                        "expected_fixed_task_classes",
                        "risk_task_classes",
                        "expected_metric_delta",
                        "confidence",
                        "falsification_window",
                    ],
                },
                "leaderboard_compliance": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "harbor_official_harness_preserved": {"type": "boolean"},
                        "self_owned_worker_preserved": {"type": "boolean"},
                        "benchmark_integrity_preserved": {"type": "boolean"},
                        "timeouts_resources_unchanged": {"type": "boolean"},
                        "submit_gate_preserved": {"type": "boolean"},
                        "official_dataset_preserved": {"type": "boolean"},
                        "five_attempts_per_task_preserved": {"type": "boolean"},
                        "no_prohibited_terminal_bench_access": {"type": "boolean"},
                        "upload_artifacts_trace_preserved": {"type": "boolean"},
                    },
                    "required": [
                        "harbor_official_harness_preserved",
                        "self_owned_worker_preserved",
                        "benchmark_integrity_preserved",
                        "timeouts_resources_unchanged",
                        "submit_gate_preserved",
                        "official_dataset_preserved",
                        "five_attempts_per_task_preserved",
                        "no_prohibited_terminal_bench_access",
                        "upload_artifacts_trace_preserved",
                    ],
                },
                "external_research": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "used": {"type": "boolean"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "When used is true, every source must be copied "
                                "from external_research_policy.web_sources or "
                                "external_research_policy.local_read_only_refs; "
                                "unlisted sources are rejected. Sources matching "
                                "external_research_policy.fetch_requirements must "
                                "be fetched with the listed required_user_agent "
                                "before reporting them unavailable."
                            ),
                        },
                        "fetches": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "source": {"type": "string"},
                                    "headers": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "User-Agent": {"type": "string"},
                                        },
                                        "required": ["User-Agent"],
                                    },
                                    "result": {"type": "string"},
                                },
                                "required": ["source", "headers", "result"],
                            },
                            "description": (
                                "Required for any source matching "
                                "external_research_policy.fetch_requirements. "
                                "Record the actual request headers used; "
                                "mp.weixin.qq.com must include a User-Agent "
                                "containing MicroMessenger."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "Explain why the listed packet source was relevant "
                                "to this evidence-backed update slice."
                            ),
                        },
                        "impact": {
                            "type": "string",
                            "description": (
                                "When used is true, name the concrete local harness "
                                "or updater decision changed by the reference and "
                                "include at least one marker from "
                                "external_research_policy.research_focus_areas, "
                                "such as action space, shell/file, done, verifier, "
                                "context, provider, or handoff."
                            ),
                        },
                    },
                    "required": ["used", "sources", "fetches", "reason", "impact"],
                },
            },
            "required": [
                "status",
                "summary",
                "changed_files",
                "validation_commands",
                "skipped_validation_reason",
                "strategy_confidence",
                "loophole_review",
                "loophole_fixes",
                "component_type",
                "implementation_scope",
                "generalization",
                "cross_round_evidence",
                "memory_record",
                "framework_comparison",
                "prediction",
                "leaderboard_compliance",
                "external_research",
            ],
        }

    def _trial_summary(self, trial: TrialResult) -> dict[str, Any]:
        attribution = FailureAttributor().analyze(trial)
        metadata = trial.metadata if isinstance(trial.metadata, dict) else {}
        base_failure_category = self._base_failure_category(
            trial,
            attribution.failure_category,
        )
        mechanisms = [mechanism.as_dict() for mechanism in failure_mechanisms_for_trial(trial)]
        mechanism_names = [item["name"] for item in mechanisms if item.get("name")]
        phase_attribution = self._infrastructure_phase_attribution(trial)
        if phase_attribution is not None:
            failure_category = phase_attribution[0]
            affected_components = sorted(phase_attribution[1])
        else:
            failure_category = self._mechanism_enhanced_failure_category(
                attribution.failure_category,
                mechanism_names,
            )
            failure_category = self._dependency_loop_enhanced_failure_category(
                trial,
                failure_category,
                mechanism_names,
            )
            affected_components = self._mechanism_enhanced_components(
                attribution.affected_components,
                mechanism_names,
                failure_category=failure_category,
            )
        summary = {
            "trial_id": trial.trial_id,
            "task_id": trial.task_id,
            "status": trial.status.value,
            "score": trial.score,
            "verified": trial.verified,
            "failure_category": failure_category,
            "base_failure_category": base_failure_category,
            "affected_components": affected_components,
            "component_confidence": attribution.component_confidence,
            "timeout_phase": str(metadata.get("timeout_phase") or ""),
            "failure_mechanisms": mechanisms,
            "infra_error_detected": bool(metadata.get("infra_error_detected")),
            "environment_start_attribution_hint": str(
                metadata.get("environment_start_attribution_hint") or ""
            ),
            "docker_image_validation_failed": bool(
                metadata.get("docker_image_validation_failed")
            ),
            "heavy_dockerfile_install_detected": bool(
                metadata.get("heavy_dockerfile_install_detected")
            ),
            "errors": list(trial.error_log),
            "harbor_job_dir": trial.harbor_job_dir,
            "harbor_trial_dir": trial.harbor_trial_dir,
        }
        warmup_commands = [
            str(command)
            for command in metadata.get("prebuilt_image_cache_warmup_commands") or []
        ][:3]
        if metadata.get("prebuilt_image_cache_miss_detected"):
            summary["prebuilt_image_cache_miss_detected"] = True
        if warmup_commands:
            summary["prebuilt_image_cache_warmup_commands"] = warmup_commands
        if metadata.get("network_preflight_recommended"):
            summary["network_preflight_recommended"] = True
        return summary

    def _full_trajectory(self, trial: TrialResult) -> list[dict[str, Any]]:
        return list(trial.trajectory)

    def _full_verifier_output(self, trial: TrialResult) -> str:
        return trial.verifier_output

    def _failed_tool_calls(self, trial: TrialResult) -> list[dict[str, Any]]:
        return [call for call in trial.tool_calls if call.get("success") is False]

    def _failure_artifacts(self, trial: TrialResult) -> dict[str, Any]:
        metadata_verifier_logs = str(trial.metadata.get("verifier_logs") or "")
        artifact = {
            "trial_id": trial.trial_id,
            "task_id": trial.task_id,
            "source_paths": {
                "harbor_job_dir": trial.harbor_job_dir,
                "harbor_trial_dir": trial.harbor_trial_dir,
            },
            "direct_logs": {
                "error_log_tail": self._tail_strings(trial.error_log, max_items=8),
                "harbor_stdout_tail": self._tail_text(trial.harbor_stdout, 6000),
                "harbor_stderr_tail": self._tail_text(trial.harbor_stderr, 6000),
                "verifier_output_tail": self._tail_text(trial.verifier_output, 6000),
                "verifier_logs_tail": self._tail_text(metadata_verifier_logs, 6000),
            },
            "primary_evidence_policy": {
                "trajectory_slices_are_full": True,
                "trajectory_window_audit_only": None,
                "trajectory_window_stop_condition": False,
                "tool_failures_are_full": True,
                "tool_failure_window_audit_only": None,
                "tool_failure_window_stop_condition": False,
                "verifier_outputs_are_full": True,
                "verifier_output_window_audit_only": None,
                "verifier_output_window_stop_condition": False,
                "error_log_is_full": True,
                "error_log_window_audit_only": None,
                "error_log_window_stop_condition": False,
            },
            "environment_start_evidence": trial.metadata.get(
                "environment_start_evidence"
            )
            or {},
            "artifact_files": self._failure_artifact_files(trial),
            "artifact_file_policy": {
                "allowed_roots": [
                    "trial.harbor_trial_dir",
                    "trial.harbor_job_dir",
                ],
                "excluded_roots": [
                    "terminal-bench-tasks",
                    "terminal-bench",
                ],
                "max_files": 12,
                "max_chars_per_file": 6000,
            },
        }
        artifact["available_artifacts"] = list(trial.artifacts)[:40]
        return artifact

    def _failure_artifact_files(self, trial: TrialResult) -> list[dict[str, Any]]:
        roots = [
            Path(trial.harbor_trial_dir) if trial.harbor_trial_dir else None,
            Path(trial.harbor_job_dir) if trial.harbor_job_dir else None,
        ]
        seen: set[Path] = set()
        entries: list[dict[str, Any]] = []
        for root in roots:
            if root is None:
                continue
            root = self._safe_artifact_root(root)
            if root is None:
                continue
            for path in self._candidate_artifact_paths(root):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                entry = self._artifact_file_entry(root, path)
                if entry:
                    entries.append(entry)
                if len(entries) >= 12:
                    return entries
        return entries

    def _safe_artifact_root(self, root: Path) -> Path | None:
        candidate = root if root.is_absolute() else self.repo_root / root
        try:
            resolved = candidate.resolve()
        except OSError:
            return None
        if not resolved.exists() or not resolved.is_dir():
            return None
        forbidden_roots = [
            self.repo_root / "terminal-bench-tasks",
            self.repo_root / "terminal-bench",
        ]
        for forbidden in forbidden_roots:
            try:
                resolved.relative_to(forbidden.resolve())
                return None
            except ValueError:
                continue
        return resolved

    def _candidate_artifact_paths(self, root: Path) -> list[Path]:
        preferred = [
            "harbor_stdout.txt",
            "harbor_stderr.txt",
            "verifier_output.txt",
            "trajectory.jsonl",
            "tool_calls.jsonl",
            "agent/trajectory.jsonl",
            "agent/trajectory.json",
            "verifier/test-stdout.txt",
            "verifier/test-stderr.txt",
            "verifier/reward.txt",
            "logs/verifier/test-stdout.txt",
            "logs/verifier/test-stderr.txt",
            "logs/verifier/reward.txt",
        ]
        paths = [root / name for name in preferred if (root / name).is_file()]
        if paths:
            return paths
        fallback_names = {
            "harbor_stdout.txt",
            "harbor_stderr.txt",
            "verifier_output.txt",
            "trajectory.jsonl",
            "trajectory.json",
            "tool_calls.jsonl",
            "test-stdout.txt",
            "test-stderr.txt",
            "reward.txt",
        }
        return [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name in fallback_names
        ][:12]

    def _artifact_file_entry(self, root: Path, path: Path) -> dict[str, Any] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        if stat.st_size > 1_000_000:
            return {
                "path": str(path.relative_to(root)),
                "size_bytes": stat.st_size,
                "skipped_reason": "file exceeds 1000000 byte artifact summary limit",
            }
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            return {
                "path": str(path.relative_to(root)),
                "skipped_reason": f"could not read artifact: {exc}",
            }
        return {
            "path": str(path.relative_to(root)),
            "size_bytes": stat.st_size,
            "tail": self._tail_text(text, 6000),
        }

    def _tail_strings(self, values: list[str], *, max_items: int) -> list[str]:
        return [self._tail_text(str(value), 2000) for value in values[-max_items:]]

    def _tail_text(self, text: str, max_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return f"[... omitted {omitted} chars ...]\n" + text[-max_chars:]

    def _campaign_context(
        self,
        failures: list[TrialResult],
        *,
        summary_limit: int | None = None,
        trial_limit: int | None = None,
    ) -> dict[str, Any]:
        summaries = self._recent_summaries(limit=summary_limit)
        trigger_trial_ids = {trial.trial_id for trial in failures}
        completed = self._recent_completed_trials(
            summaries=summaries,
            trigger_trial_ids=trigger_trial_ids,
            limit=trial_limit,
        )
        completed_full_context_count = sum(
            1 for item in completed if item.get("full_result_context")
        )
        recent_analysis_reports = self._recent_analysis_reports(
            limit=DEFAULT_ANALYSIS_REPORT_CONTEXT_LIMIT
        )
        policy_recurrence_signals = self._policy_recurrence_signals_from_reports(
            recent_analysis_reports
        )
        recent_codex_update_events = self._recent_codex_update_events(
            limit=DEFAULT_CODEX_UPDATE_EVENT_CONTEXT_LIMIT
        )
        return {
            "purpose": (
                "Bounded recent-campaign evidence for choosing a reusable "
                "Worker/harness improvement. Treat task ids as evidence labels, "
                "not targets for special cases. Trigger trials are always retained; "
                "the filesystem memory remains the complete archive."
            ),
            "summary_window": summary_limit,
            "trial_window": trial_limit,
            "summary_window_audit_only": summary_limit,
            "trial_window_audit_only": trial_limit,
            "summary_window_stop_condition": False,
            "trial_window_stop_condition": False,
            "evidence_count_stop_condition": False,
            "recent_summaries": summaries,
            "recent_completed_trials": completed,
            "recent_completed_trial_count": len(completed),
            "recent_completed_full_result_context_count": completed_full_context_count,
            "recent_completed_full_result_context_threshold_audit_only": (
                self._completed_trial_full_context_threshold
            ),
            "recent_completed_full_result_context_stop_condition": False,
            "recent_completed_context_note": (
                "The packet keeps a deterministic recent trial window plus every "
                "trigger trial. Expensive result.json hydration is applied to "
                "trigger trials, or to every retained row only when the context is "
                "small. Complete campaign_state and run artifacts remain available "
                "on disk; this is packet compression, not a loop stop condition."
            ),
            "recent_analysis_reports": recent_analysis_reports,
            "policy_recurrence_signals": policy_recurrence_signals,
            "recent_codex_update_events": recent_codex_update_events,
            "recent_codex_update_event_count": len(recent_codex_update_events),
            "recent_codex_update_events_stop_condition": False,
            "legacy_limit_driven_skip_events_normalized": sum(
                1
                for event in recent_codex_update_events
                if event.get("legacy_limit_driven_skip_normalized") is True
            ),
            "trigger_trial_ids": sorted(trigger_trial_ids),
            "source": "latest campaign_state.json under the active memory path when available",
        }

    def _recent_codex_update_events(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for state in reversed(self._recent_campaign_states(limit=None)):
            raw_events = state.get("codex_update_events")
            if not isinstance(raw_events, list):
                continue
            events.extend(event for event in raw_events if isinstance(event, dict))
        if limit is None:
            return events
        limit_value = max(0, int(limit))
        return events[-limit_value:] if limit_value else []

    def _recent_analysis_reports(
        self,
        limit: int | None = None,
        max_chars: int = 6000,
    ) -> list[dict[str, Any]]:
        analysis_root = self.memory_path / "analysis"
        if not analysis_root.exists():
            return []
        if limit is not None and int(limit) <= 0:
            return []
        reports: list[dict[str, Any]] = []
        candidates = sorted(
            analysis_root.glob("*/*/overview.md"),
            key=lambda path: path.stat().st_mtime,
        )
        for path in reversed(candidates):
            try:
                text = path.read_text(errors="replace").strip()
            except OSError:
                continue
            summary_path = path.with_name("summary.json")
            summary: dict[str, Any] = {}
            if summary_path.exists():
                try:
                    raw_summary = self._read_json(summary_path)
                except (OSError, ValueError):
                    raw_summary = {}
                if isinstance(raw_summary, dict):
                    summary = raw_summary
            failure_buckets = self._analysis_failure_bucket_entries(
                summary.get("failure_buckets")
            )
            raw_candidate_update_classes = [
                str(item) for item in summary.get("candidate_update_classes") or []
            ][:8]
            candidate_update_classes = self._normalized_analysis_candidate_classes(
                raw_candidate_update_classes,
                failure_buckets,
            )
            trajectory_evidence = self._analysis_trajectory_evidence_entries(
                summary.get("trajectory_evidence")
            )
            mechanism_update_classes = self._analysis_mechanism_update_classes(
                summary.get("mechanism_update_classes"),
            )
            mechanism_update_entries = self._analysis_mechanism_update_entries(
                summary.get("mechanism_update_entries")
            ) or self._analysis_mechanism_update_entries_from_evidence(
                failure_buckets=failure_buckets,
                trajectory_evidence=trajectory_evidence,
            )
            if not mechanism_update_classes:
                mechanism_update_classes = self._format_analysis_mechanism_update_classes(
                    mechanism_update_entries
                )
            existing_weakness_signatures = self._analysis_weakness_signature_entries(
                summary.get("weakness_signatures")
            )
            synthesized_weakness_signatures = self._legacy_analysis_weakness_signatures(
                summary_id=str(summary.get("summary_id") or ""),
                failure_buckets=failure_buckets,
                trajectory_evidence=trajectory_evidence,
            )
            weakness_signatures = self._normalized_analysis_weakness_signatures(
                existing=existing_weakness_signatures,
                synthesized=synthesized_weakness_signatures,
                failure_buckets=failure_buckets,
            )
            report = {
                "path": str(path),
                "summary_path": str(summary_path) if summary_path.exists() else "",
                "summary_id": str(summary.get("summary_id") or ""),
                "overall_score": self._safe_float(summary.get("overall_score")),
                "trial_count": int(summary.get("trial_count") or 0),
                "infrastructure_failure_count": int(
                    summary.get("infrastructure_failure_count") or 0
                ),
                "terminal_environment_signal_count": int(
                    summary.get("terminal_environment_signal_count") or 0
                ),
                "candidate_update_classes": candidate_update_classes,
                "mechanism_update_entries": mechanism_update_entries,
                "mechanism_update_classes": mechanism_update_classes,
                "failure_buckets": failure_buckets,
                "weakness_signatures": weakness_signatures,
                "policy_coverage": self._analysis_policy_coverage(
                    summary.get("policy_coverage")
                ),
                "trajectory_evidence": trajectory_evidence,
                "failure_mechanisms": self._analysis_failure_mechanism_entries(
                    summary.get("trajectory_evidence")
                ),
                "policy_recurrence_signals": self._analysis_policy_recurrence_signals(
                    summary,
                    failure_buckets,
                ),
                "detail_paths": {
                    str(task_id): str(detail_path)
                    for task_id, detail_path in (
                        summary.get("detail_paths") or {}
                    ).items()
                }
                if isinstance(summary.get("detail_paths"), dict)
                else {},
                "tail": text[-max_chars:],
            }
            if raw_candidate_update_classes != candidate_update_classes:
                report["raw_candidate_update_classes"] = raw_candidate_update_classes
                report["candidate_update_classes_normalized_from"] = (
                    "failure_buckets"
                )
            if (
                existing_weakness_signatures
                and weakness_signatures != existing_weakness_signatures
            ):
                report["raw_weakness_signatures"] = existing_weakness_signatures
                report["weakness_signatures_normalized_from"] = "failure_buckets"
            reports.append(report)
            if limit is not None and len(reports) >= max(0, int(limit)):
                break
        return reports

    def _normalized_analysis_candidate_classes(
        self,
        raw_candidates: list[str],
        failure_buckets: list[dict[str, Any]],
    ) -> list[str]:
        bucket_candidates = self._analysis_candidate_classes_from_buckets(
            failure_buckets,
        )
        if not raw_candidates or not bucket_candidates:
            return raw_candidates or bucket_candidates
        bucket_categories = {
            str(bucket.get("failure_category") or "") for bucket in failure_buckets
        }
        candidate_categories = {
            self._candidate_update_class_category(candidate)
            for candidate in raw_candidates
        }
        candidate_categories.discard("")
        if candidate_categories and candidate_categories.issubset(bucket_categories):
            return raw_candidates
        return bucket_candidates

    def _analysis_candidate_classes_from_buckets(
        self,
        failure_buckets: list[dict[str, Any]],
    ) -> list[str]:
        candidates: list[str] = []
        for bucket in failure_buckets[:8]:
            category = str(bucket.get("failure_category") or "").strip()
            if not category:
                continue
            components = ", ".join(
                str(item)
                for item in bucket.get("affected_components") or ["unknown"]
            )
            prefix = "infrastructure " if bucket.get("infrastructure") else ""
            count = int(bucket.get("count") or 0)
            candidates.append(
                f"{prefix}{category} -> {components} ({count} trial(s))"
            )
        return candidates

    def _analysis_mechanism_update_classes(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        classes: list[str] = []
        for raw_item in value:
            if len(classes) >= 8:
                break
            item = self._tail_text(str(raw_item or ""), 500).strip()
            if item:
                classes.append(item)
        return classes

    def _analysis_mechanism_update_entries(
        self,
        value: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        entries: list[dict[str, Any]] = []
        for raw_entry in value[:8]:
            if not isinstance(raw_entry, dict):
                continue
            category = self._tail_text(
                str(raw_entry.get("failure_category") or ""),
                180,
            )
            mechanism = self._tail_text(str(raw_entry.get("mechanism") or ""), 180)
            if not category or not mechanism:
                continue
            entries.append(
                {
                    "failure_category": category,
                    "mechanism": mechanism,
                    "count": int(raw_entry.get("count") or 0),
                    "task_ids": [
                        str(item) for item in raw_entry.get("task_ids") or []
                    ][:12],
                    "affected_components": [
                        str(item)
                        for item in raw_entry.get("affected_components") or []
                    ][:12],
                }
            )
        entries.sort(
            key=lambda item: (
                -int(item.get("count") or 0),
                str(item.get("failure_category") or ""),
                str(item.get("mechanism") or ""),
            )
        )
        return entries

    def _analysis_mechanism_update_entries_from_evidence(
        self,
        *,
        failure_buckets: list[dict[str, Any]],
        trajectory_evidence: dict[str, Any],
    ) -> list[str]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for bucket in failure_buckets:
            category = str(bucket.get("failure_category") or "").strip()
            if not category:
                continue
            for task_id in bucket.get("task_ids") or []:
                task_text = str(task_id).strip()
                if not task_text:
                    continue
                evidence = trajectory_evidence.get(task_text)
                if not isinstance(evidence, dict):
                    continue
                for mechanism in evidence.get("failure_mechanisms") or []:
                    if not isinstance(mechanism, dict):
                        continue
                    mechanism_name = str(mechanism.get("name") or "").strip()
                    if not mechanism_name:
                        continue
                    key = (category, mechanism_name)
                    entry = grouped.setdefault(
                        key,
                        {
                            "failure_category": category,
                            "mechanism": mechanism_name,
                            "task_ids": set(),
                            "affected_components": set(),
                        },
                    )
                    entry["task_ids"].add(task_text)
                    entry["affected_components"].update(
                        affected_components_for_failure_mechanism(mechanism_name)
                    )
                    entry["affected_components"].update(
                        self._analysis_mechanism_component_overrides(mechanism_name)
                    )

        entries: list[dict[str, Any]] = []
        for entry in grouped.values():
            components = sorted(entry["affected_components"])
            entries.append(
                {
                    "failure_category": entry["failure_category"],
                    "mechanism": entry["mechanism"],
                    "count": len(entry["task_ids"]),
                    "task_ids": sorted(entry["task_ids"]),
                    "affected_components": components or ["unknown"],
                }
            )
        entries.sort(
            key=lambda item: (
                -int(item["count"]),
                str(item["failure_category"]),
                str(item["mechanism"]),
            )
        )
        return entries[:8]

    def _format_analysis_mechanism_update_classes(
        self,
        entries: list[dict[str, Any]],
    ) -> list[str]:
        return [
            f"{entry['failure_category']} / {entry['mechanism']} -> "
            f"{', '.join(entry.get('affected_components') or ['unknown'])} "
            f"({entry['count']} trial(s))"
            for entry in entries[:8]
        ]

    def _analysis_mechanism_component_overrides(self, mechanism_name: str) -> set[str]:
        if mechanism_name == "missing_output_artifact_contract":
            return {
                "bench/agent",
                "harness/tools/verify",
                "recovery/patterns",
                "verification/checks",
            }
        return set()

    def _candidate_update_class_category(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        left = text.split("->", 1)[0].strip()
        if left.startswith("infrastructure "):
            left = left[len("infrastructure ") :].strip()
        return left

    def _analysis_failure_bucket_entries(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        entries: list[dict[str, Any]] = []
        for raw_entry in value[:8]:
            if not isinstance(raw_entry, dict):
                continue
            entries.append(
                {
                    "failure_category": str(raw_entry.get("failure_category") or ""),
                    "count": int(raw_entry.get("count") or 0),
                    "infrastructure": bool(raw_entry.get("infrastructure")),
                    "task_ids": [
                        str(item) for item in raw_entry.get("task_ids") or []
                    ][:12],
                    "affected_components": [
                        str(item)
                        for item in raw_entry.get("affected_components") or []
                    ][:8],
                    "timeout_phases": [
                        str(item) for item in raw_entry.get("timeout_phases") or []
                    ][:8],
                }
            )
            failure_mechanisms = [
                str(item) for item in raw_entry.get("failure_mechanisms") or []
            ][:12]
            if failure_mechanisms:
                entries[-1]["failure_mechanisms"] = failure_mechanisms
                entries[-1]["failure_mechanism_count_stop_condition"] = False
        return self._normalize_analysis_failure_bucket_entries(entries)

    def _analysis_weakness_signature_entries(
        self,
        value: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        entries: list[dict[str, Any]] = []
        for raw_entry in value[:12]:
            if not isinstance(raw_entry, dict):
                continue
            signature = self._tail_text(str(raw_entry.get("signature") or ""), 600)
            category = self._tail_text(
                str(raw_entry.get("failure_category") or ""),
                160,
            )
            if not signature and not category:
                continue
            entries.append(
                {
                    "signature": signature,
                    "verifier_failure": self._tail_text(
                        str(raw_entry.get("verifier_failure") or ""),
                        240,
                    ),
                    "agent_contribution": self._tail_text(
                        str(raw_entry.get("agent_contribution") or ""),
                        240,
                    ),
                    "reusable_mechanism": self._tail_text(
                        str(raw_entry.get("reusable_mechanism") or ""),
                        240,
                    ),
                    "failure_category": category,
                    "count": int(raw_entry.get("count") or 0),
                    "task_ids": [
                        str(item) for item in raw_entry.get("task_ids") or []
                    ][:12],
                    "affected_components": [
                        str(item)
                        for item in raw_entry.get("affected_components") or []
                    ][:12],
                    "timeout_phases": [
                        str(item) for item in raw_entry.get("timeout_phases") or []
                    ][:8],
                    "failure_mechanisms": [
                        str(item)
                        for item in raw_entry.get("failure_mechanisms") or []
                    ][:12],
                    "evidence_sources": [
                        str(item) for item in raw_entry.get("evidence_sources") or []
                    ][:12],
                    "loop_stop_condition": bool(
                        raw_entry.get("loop_stop_condition", False)
                    ),
                    "time_round_token_limit_driven": bool(
                        raw_entry.get("time_round_token_limit_driven", False)
                    ),
                }
            )
        entries.sort(
            key=lambda item: (
                -int(item.get("count") or 0),
                str(item.get("signature") or ""),
            )
        )
        return entries

    def _legacy_analysis_weakness_signatures(
        self,
        *,
        summary_id: str,
        failure_buckets: list[dict[str, Any]],
        trajectory_evidence: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not failure_buckets:
            return []
        grouped: dict[str, dict[str, Any]] = {}
        for bucket in failure_buckets:
            category = str(bucket.get("failure_category") or "").strip()
            if not category:
                continue
            task_ids = [str(item) for item in bucket.get("task_ids") or []]
            count_increment = int(bucket.get("count") or 0) if len(task_ids) <= 1 else 1
            for task_id in task_ids or [""]:
                evidence = trajectory_evidence.get(task_id)
                if not isinstance(evidence, dict):
                    evidence = {}
                mechanisms = self._legacy_analysis_weakness_mechanisms(evidence)
                verifier_failure = self._legacy_analysis_verifier_failure(
                    category,
                    bucket,
                )
                agent_contribution = self._legacy_analysis_agent_contribution(
                    evidence,
                    bucket=bucket,
                )
                reusable_mechanism = self._legacy_analysis_reusable_mechanism(
                    category,
                    mechanisms=mechanisms,
                    bucket=bucket,
                )
                signature = (
                    f"verifier={verifier_failure}|agent={agent_contribution}|"
                    f"mechanism={reusable_mechanism}"
                )
                entry = grouped.setdefault(
                    signature,
                    {
                        "signature": signature,
                        "verifier_failure": verifier_failure,
                        "agent_contribution": agent_contribution,
                        "reusable_mechanism": reusable_mechanism,
                        "failure_category": category,
                        "count": 0,
                        "task_ids": set(),
                        "affected_components": set(),
                        "timeout_phases": set(),
                        "failure_mechanisms": set(),
                        "evidence_sources": set(),
                    },
                )
                entry["count"] = int(entry.get("count") or 0) + max(
                    count_increment,
                    1,
                )
                if task_id:
                    entry["task_ids"].add(task_id)
                for field, target in (
                    ("affected_components", "affected_components"),
                    ("timeout_phases", "timeout_phases"),
                ):
                    for value in bucket.get(field) or []:
                        text = str(value).strip()
                        if text:
                            entry[target].add(text)
                for mechanism in mechanisms:
                    entry["failure_mechanisms"].add(mechanism)
                for source in self._legacy_analysis_evidence_sources(evidence):
                    entry["evidence_sources"].add(source)
                if summary_id:
                    entry.setdefault("source_summary_ids", set()).add(summary_id)
        entries: list[dict[str, Any]] = []
        for entry in grouped.values():
            result = {
                "signature": entry["signature"],
                "verifier_failure": entry["verifier_failure"],
                "agent_contribution": entry["agent_contribution"],
                "reusable_mechanism": entry["reusable_mechanism"],
                "failure_category": entry["failure_category"],
                "count": int(entry.get("count") or 0),
                "task_ids": sorted(entry["task_ids"])[:12],
                "affected_components": sorted(entry["affected_components"])[:12],
                "timeout_phases": sorted(entry["timeout_phases"])[:8],
                "failure_mechanisms": sorted(entry["failure_mechanisms"])[:12],
                "evidence_sources": sorted(entry["evidence_sources"])[:12],
                "loop_stop_condition": False,
                "time_round_token_limit_driven": False,
                "synthesized_from_legacy_analysis": True,
            }
            source_summary_ids = sorted(entry.get("source_summary_ids", set()))
            if source_summary_ids:
                result["source_summary_ids"] = source_summary_ids
            entries.append(result)
        entries.sort(
            key=lambda item: (
                -int(item.get("count") or 0),
                str(item.get("signature") or ""),
            )
        )
        return entries

    def _legacy_analysis_weakness_mechanisms(
        self,
        evidence: dict[str, Any],
    ) -> list[str]:
        mechanisms: list[str] = []
        for raw_item in evidence.get("failure_mechanisms") or []:
            if not isinstance(raw_item, dict):
                continue
            name = str(raw_item.get("name") or "").strip()
            if name:
                mechanisms.append(name)
        return sorted(dict.fromkeys(mechanisms))

    def _legacy_analysis_verifier_failure(
        self,
        category: str,
        bucket: dict[str, Any],
    ) -> str:
        if bucket.get("infrastructure"):
            phases = [str(item) for item in bucket.get("timeout_phases") or [] if item]
            if phases:
                return "infra_timeout_phase:" + "+".join(sorted(dict.fromkeys(phases)))
            return "infrastructure_failure:" + category
        if "timeout" in category and not category.endswith("_contract"):
            return "worker_or_verifier_timeout:" + category
        return "verifier_assertion:" + category

    def _legacy_analysis_agent_contribution(
        self,
        evidence: dict[str, Any],
        *,
        bucket: dict[str, Any] | None = None,
    ) -> str:
        policy_counts: dict[str, int] = {}
        for raw_name, raw_count in (evidence.get("policy_counts") or {}).items():
            name = str(raw_name).strip()
            if not name or name == "artifact_check_deliverable_progress":
                continue
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                policy_counts[name] = policy_counts.get(name, 0) + count
        if policy_counts:
            name, count = sorted(
                policy_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
            return f"policy:{name}:{count}"
        for label, field in (
            ("dependency_loop", "dependency_and_toolchain_evidence"),
            ("blocked_guard", "blocked_guards"),
            ("timed_out_command", "timed_out_commands"),
            ("semantic_mechanism", "failure_mechanisms"),
            ("terminal_environment", "terminal_environment_markers"),
            ("deliverable_progress", "deliverable_progress"),
        ):
            items = evidence.get(field)
            if isinstance(items, list) and items:
                return f"evidence:{label}:{len(items)}"
        if bucket and bool(bucket.get("infrastructure")):
            phases = [
                str(phase)
                for phase in bucket.get("timeout_phases") or []
                if str(phase).strip()
            ]
            count = int(bucket.get("count") or 0)
            if phases:
                return (
                    "infrastructure:timeout_phase:"
                    + "+".join(sorted(dict.fromkeys(phases)))
                    + f":{count}"
                )
            category = str(bucket.get("failure_category") or "infrastructure_failure")
            return f"infrastructure:{category}:{count}"
        return "agent_behavior:unclassified"

    def _normalized_analysis_weakness_signatures(
        self,
        *,
        existing: list[dict[str, Any]],
        synthesized: list[dict[str, Any]],
        failure_buckets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not existing:
            return synthesized
        if not synthesized:
            return existing

        bucket_categories = {
            str(bucket.get("failure_category") or "")
            for bucket in failure_buckets
            if str(bucket.get("failure_category") or "")
        }
        raw_category_map: dict[str, set[str]] = {}
        for bucket in failure_buckets:
            category = str(bucket.get("failure_category") or "")
            if not category:
                continue
            for raw_category in bucket.get("raw_failure_categories") or []:
                raw_text = str(raw_category)
                if raw_text:
                    raw_category_map.setdefault(raw_text, set()).add(category)

        kept: list[dict[str, Any]] = []
        stale: list[dict[str, Any]] = []
        for entry in existing:
            if self._weakness_signature_needs_bucket_resynthesis(
                entry,
                bucket_categories=bucket_categories,
                raw_bucket_categories=set(raw_category_map),
            ):
                stale.append(entry)
            else:
                kept.append(entry)
        if not stale:
            return existing

        stale_categories = {
            str(entry.get("failure_category") or "")
            for entry in stale
            if str(entry.get("failure_category") or "")
        }
        replacement_categories = set(stale_categories)
        for category in stale_categories:
            replacement_categories.update(raw_category_map.get(category, set()))
        stale_task_ids = {
            str(task_id)
            for entry in stale
            for task_id in entry.get("task_ids") or []
            if str(task_id)
        }
        stale_timeout_phases = {
            str(phase)
            for entry in stale
            for phase in entry.get("timeout_phases") or []
            if str(phase)
        }
        kept_signatures = {str(entry.get("signature") or "") for entry in kept}
        replacements = [
            entry
            for entry in synthesized
            if str(entry.get("signature") or "") not in kept_signatures
            and self._synthesized_weakness_matches_stale_entry(
                entry,
                replacement_categories=replacement_categories,
                stale_task_ids=stale_task_ids,
                stale_timeout_phases=stale_timeout_phases,
            )
        ]
        if not replacements and not kept:
            replacements = synthesized
        normalized = [*kept, *replacements]
        normalized.sort(
            key=lambda item: (
                -int(item.get("count") or 0),
                str(item.get("signature") or ""),
            )
        )
        return normalized

    def _weakness_signature_needs_bucket_resynthesis(
        self,
        entry: dict[str, Any],
        *,
        bucket_categories: set[str],
        raw_bucket_categories: set[str],
    ) -> bool:
        contribution = str(entry.get("agent_contribution") or "")
        signature = str(entry.get("signature") or "")
        if contribution == "agent_behavior:unclassified":
            return True
        if "agent=agent_behavior:unclassified" in signature:
            return True
        category = str(entry.get("failure_category") or "")
        return bool(
            category
            and category not in bucket_categories
            and category in raw_bucket_categories
        )

    def _synthesized_weakness_matches_stale_entry(
        self,
        entry: dict[str, Any],
        *,
        replacement_categories: set[str],
        stale_task_ids: set[str],
        stale_timeout_phases: set[str],
    ) -> bool:
        category = str(entry.get("failure_category") or "")
        if category and category in replacement_categories:
            return True
        task_ids = {str(item) for item in entry.get("task_ids") or [] if str(item)}
        if stale_task_ids and task_ids & stale_task_ids:
            return True
        timeout_phases = {
            str(item) for item in entry.get("timeout_phases") or [] if str(item)
        }
        return bool(stale_timeout_phases and timeout_phases & stale_timeout_phases)

    def _legacy_analysis_reusable_mechanism(
        self,
        category: str,
        *,
        mechanisms: list[str],
        bucket: dict[str, Any],
    ) -> str:
        if mechanisms:
            return "mechanism:" + "+".join(mechanisms[:4])
        components = [
            str(item) for item in bucket.get("affected_components") or [] if item
        ]
        if components:
            return "components:" + "+".join(components[:4])
        return "category:" + category

    def _legacy_analysis_evidence_sources(
        self,
        evidence: dict[str, Any],
    ) -> list[str]:
        sources: set[str] = set()
        for field in (
            "failure_mechanisms",
            "policy_counts",
            "timed_out_commands",
            "blocked_guards",
            "dependency_and_toolchain_evidence",
            "deliverable_progress",
            "terminal_environment_markers",
        ):
            value = evidence.get(field)
            if isinstance(value, dict) and value:
                sources.add(field)
            elif isinstance(value, list) and value:
                sources.add(field)
        return sorted(sources)

    def _normalize_analysis_failure_bucket_entries(
        self,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, bool], dict[str, Any]] = {}
        order: list[tuple[str, bool]] = []
        for entry in entries:
            normalized = self._normalize_analysis_failure_bucket_entry(entry)
            key = (
                str(normalized.get("failure_category") or ""),
                bool(normalized.get("infrastructure")),
            )
            if key not in merged:
                merged[key] = normalized
                order.append(key)
                continue
            target = merged[key]
            target["count"] = int(target.get("count") or 0) + int(
                normalized.get("count") or 0
            )
            target["infrastructure"] = bool(
                target.get("infrastructure") or normalized.get("infrastructure")
            )
            for field in (
                "task_ids",
                "affected_components",
                "timeout_phases",
                "raw_failure_categories",
                "normalized_from_timeout_phases",
            ):
                self._merge_bucket_list_field(target, normalized, field)
        return [merged[key] for key in order]

    def _normalize_analysis_failure_bucket_entry(
        self,
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(entry)
        phase_attribution = self._infrastructure_phase_for_bucket(entry)
        if phase_attribution is None:
            return normalized
        timeout_phase, (category, components) = phase_attribution
        original_category = str(normalized.get("failure_category") or "")
        original_components = [
            str(component) for component in normalized.get("affected_components") or []
        ]
        original_infrastructure = bool(normalized.get("infrastructure"))
        canonical_components = list(components)
        changed = (
            original_category != category
            or not original_infrastructure
            or original_components != canonical_components
        )
        normalized["failure_category"] = category
        normalized["infrastructure"] = True
        normalized["affected_components"] = canonical_components
        if original_category and original_category != category:
            normalized["raw_failure_categories"] = [original_category]
        if changed:
            normalized["normalized_from_timeout_phases"] = [timeout_phase]
        return normalized

    def _infrastructure_phase_for_bucket(
        self,
        entry: dict[str, Any],
    ) -> tuple[str, tuple[str, tuple[str, ...]]] | None:
        for phase in entry.get("timeout_phases") or []:
            phase_text = str(phase)
            attribution = INFRASTRUCTURE_PHASE_ATTRIBUTION.get(phase_text)
            if attribution is not None:
                return phase_text, attribution
        return None

    def _merge_bucket_list_field(
        self,
        target: dict[str, Any],
        source: dict[str, Any],
        field: str,
    ) -> None:
        values = [str(item) for item in target.get(field) or []]
        seen = set(values)
        for item in source.get(field) or []:
            text = str(item)
            if text and text not in seen:
                values.append(text)
                seen.add(text)
        if values:
            target[field] = values

    def _analysis_policy_coverage(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        policies = value.get("policies")
        policy_entries: list[dict[str, Any]] = []
        if isinstance(policies, dict):
            for raw_name, raw_entry in policies.items():
                if not isinstance(raw_entry, dict):
                    continue
                name = str(raw_name).strip()
                if not name:
                    continue
                try:
                    count = int(raw_entry.get("count") or 0)
                except (TypeError, ValueError):
                    continue
                if count <= 0:
                    continue
                policy_entries.append(
                    {
                        "policy": name,
                        "count": count,
                        "description": self._tail_text(
                            str(raw_entry.get("description") or ""),
                            240,
                        ),
                        "tasks": [
                            str(task_id)
                            for task_id in raw_entry.get("tasks") or []
                        ][:12],
                        "examples": self._analysis_evidence_items(
                            raw_entry.get("examples"),
                            limit=3,
                        ),
                    }
                )
        policy_entries.sort(key=lambda item: (-int(item["count"]), item["policy"]))
        uncovered, currently_covered, recheck = (
            self._analysis_uncovered_timeout_examples(
                value.get("uncovered_timeout_examples"),
                limit=6,
            )
        )
        result: dict[str, Any] = {}
        if policy_entries:
            result["top_policies"] = policy_entries[:12]
        if uncovered:
            result["uncovered_timeout_examples"] = uncovered
        if currently_covered:
            result["currently_covered_timeout_examples"] = currently_covered
        if recheck["legacy_uncovered_count"]:
            result["uncovered_timeout_recheck"] = recheck
        return result

    def _analysis_policy_recurrence_signals(
        self,
        summary: dict[str, Any],
        failure_buckets: list[dict[str, Any]],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        _ = limit
        if not isinstance(summary, dict) or not failure_buckets:
            return []
        coverage_counts = self._policy_counts_from_coverage(
            summary.get("policy_coverage")
        )
        trajectory_counts, mechanisms_by_task = self._trajectory_policy_and_mechanism_counts(
            summary.get("trajectory_evidence")
        )
        summary_id = str(summary.get("summary_id") or "")
        signals: dict[tuple[str, str, str], dict[str, Any]] = {}

        def add_signal(
            *,
            bucket: dict[str, Any],
            policy_name: str,
            mechanism_item: dict[str, str] | None = None,
        ) -> None:
            if not policy_name:
                return
            coverage_count = int(
                coverage_counts.get(policy_name, {}).get("count") or 0
            )
            trajectory_count = int(trajectory_counts.get(policy_name) or 0)
            sources: set[str] = set()
            category = str(bucket.get("failure_category") or "")
            if coverage_count:
                sources.add("policy_coverage")
            if trajectory_count:
                sources.add("trajectory_policy_counts")
            if mechanism_item:
                sources.add("failure_mechanism")
            if not sources:
                return
            if category == policy_name:
                sources.add("failure_bucket")
            key = (summary_id, category, policy_name)
            task_ids = [str(item) for item in bucket.get("task_ids") or [] if item]
            signal = signals.setdefault(
                key,
                {
                    "summary_id": summary_id,
                    "failure_category": category,
                    "policy": policy_name,
                    "mechanism": "",
                    "count": int(bucket.get("count") or 0),
                    "infrastructure": bool(bucket.get("infrastructure")),
                    "task_ids": task_ids[:12],
                    "affected_components": [
                        str(item)
                        for item in bucket.get("affected_components") or []
                    ][:8],
                    "policy_coverage_count": coverage_count,
                    "trajectory_policy_count": trajectory_count,
                    "evidence_sources": [],
                    "interpretation": (
                        "A current analysis policy or failure mechanism recognized "
                        "this class, yet the same class remains in the failing "
                        "bucket. Treat this as recurrence under existing coverage: "
                        "inspect trigger timing, gating strength, and validation "
                        "placement before adding another duplicate prompt or rule."
                    ),
                    "update_hint": (
                        "Prefer a bounded harness/updater change that makes the "
                        "recognized policy actionable earlier or more enforceable, "
                        "or explain why the recurrence is only noise."
                    ),
                },
            )
            signal["evidence_sources"] = sorted(
                set(signal.get("evidence_sources") or []) | sources
            )
            if mechanism_item and not signal.get("mechanism"):
                signal["mechanism"] = policy_name
                evidence = str(mechanism_item.get("evidence") or "")
                if evidence:
                    signal["mechanism_evidence"] = self._tail_text(evidence, 240)
                description = str(mechanism_item.get("description") or "")
                if description:
                    signal["mechanism_description"] = self._tail_text(
                        description,
                        240,
                    )

        for bucket in failure_buckets:
            category = str(bucket.get("failure_category") or "").strip()
            if category:
                add_signal(bucket=bucket, policy_name=category)
            bucket_tasks = {str(item) for item in bucket.get("task_ids") or []}
            for task_id in bucket_tasks:
                for mechanism_item in mechanisms_by_task.get(task_id, []):
                    mechanism_name = str(mechanism_item.get("name") or "").strip()
                    if not self._mechanism_matches_failure_category(
                        mechanism_name,
                        category,
                    ):
                        continue
                    add_signal(
                        bucket=bucket,
                        policy_name=mechanism_name,
                        mechanism_item=mechanism_item,
                    )

        ordered = sorted(
            signals.values(),
            key=lambda item: (
                -int(item.get("count") or 0),
                -int(item.get("policy_coverage_count") or 0),
                -int(item.get("trajectory_policy_count") or 0),
                str(item.get("policy") or ""),
            ),
        )
        return ordered

    def _mechanism_matches_failure_category(
        self,
        mechanism_name: str,
        failure_category: str,
    ) -> bool:
        mechanism = mechanism_name.strip()
        category = failure_category.strip()
        if not mechanism or not category:
            return False
        override_category = MECHANISM_CATEGORY_OVERRIDES.get(mechanism)
        if override_category:
            return category in {mechanism, override_category}
        if category == mechanism:
            return True
        return category in {"verifier_mismatch", "agent_timeout_with_verifier_mismatch"}

    def _policy_counts_from_coverage(self, value: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            return {}
        policies = value.get("policies")
        if not isinstance(policies, dict):
            return {}
        counts: dict[str, dict[str, Any]] = {}
        for raw_name, raw_entry in policies.items():
            if not isinstance(raw_entry, dict):
                continue
            name = str(raw_name).strip()
            if not name:
                continue
            try:
                count = int(raw_entry.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            if count <= 0:
                continue
            counts[name] = {
                "count": count,
                "tasks": [str(item) for item in raw_entry.get("tasks") or []],
            }
        return counts

    def _trajectory_policy_and_mechanism_counts(
        self,
        value: Any,
    ) -> tuple[dict[str, int], dict[str, list[dict[str, str]]]]:
        if not isinstance(value, dict):
            return {}, {}
        policy_counts: dict[str, int] = {}
        mechanisms_by_task: dict[str, list[dict[str, str]]] = {}
        for task_id, raw_entry in value.items():
            if not isinstance(raw_entry, dict):
                continue
            raw_counts = raw_entry.get("policy_counts")
            if isinstance(raw_counts, dict):
                for raw_name, raw_count in raw_counts.items():
                    name = str(raw_name).strip()
                    if not name:
                        continue
                    try:
                        count = int(raw_count or 0)
                    except (TypeError, ValueError):
                        count = 0
                    if count > 0:
                        policy_counts[name] = policy_counts.get(name, 0) + count
            mechanisms = self._analysis_failure_mechanism_items(
                raw_entry.get("failure_mechanisms"),
                limit=8,
            )
            if mechanisms:
                mechanisms_by_task[str(task_id)] = mechanisms
        return policy_counts, mechanisms_by_task

    def _policy_recurrence_signals_from_reports(
        self,
        reports: Any,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        _ = limit
        if not isinstance(reports, list):
            return []
        signals: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for report in reports:
            if not isinstance(report, dict):
                continue
            for raw_signal in report.get("policy_recurrence_signals") or []:
                if not isinstance(raw_signal, dict):
                    continue
                key = (
                    str(raw_signal.get("summary_id") or ""),
                    str(raw_signal.get("failure_category") or ""),
                    str(raw_signal.get("policy") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                signals.append(dict(raw_signal))
        return signals

    def _analysis_uncovered_timeout_examples(
        self,
        value: Any,
        *,
        limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int | str]]:
        if not isinstance(value, list):
            return [], [], {
                "legacy_uncovered_count": 0,
                "currently_covered_count": 0,
                "still_uncovered_count": 0,
                "source": "current_harness_policy_recheck",
            }
        uncovered: list[dict[str, Any]] = []
        currently_covered: list[dict[str, Any]] = []
        legacy_count = 0
        covered_count = 0
        still_uncovered_count = 0
        for raw_item in value:
            items = self._analysis_evidence_items([raw_item], limit=1)
            if not items:
                continue
            legacy_count += 1
            item: dict[str, Any] = dict(items[0])
            matches = self._current_policy_matches_for_uncovered_timeout(item)
            if matches:
                covered_count += 1
                if len(currently_covered) < limit:
                    item["current_policy_matches"] = matches[:8]
                    currently_covered.append(item)
            else:
                still_uncovered_count += 1
                if len(uncovered) < limit:
                    uncovered.append(item)
        return uncovered, currently_covered, {
            "legacy_uncovered_count": legacy_count,
            "currently_covered_count": covered_count,
            "still_uncovered_count": still_uncovered_count,
            "source": "current_harness_policy_recheck",
        }

    def _current_policy_matches_for_uncovered_timeout(
        self,
        item: dict[str, Any],
    ) -> list[str]:
        command = str(item.get("command") or "").strip()
        if not command:
            return []
        output = str(item.get("output_tail") or "")
        try:
            from scripts.run_campaign import _policy_matches_for_event
        except Exception:
            return []
        event = {
            "task_id": str(item.get("task_id") or ""),
            "tool": str(item.get("tool") or ""),
            "command": command,
            "file_path": "",
            "content": "",
            "output": output,
            "success": False,
            "timed_out": True,
            "metadata": {},
            "artifacts": [],
            "expected_artifacts": [],
        }
        try:
            matches = _policy_matches_for_event(event)
        except Exception:
            return []
        return sorted({str(match) for match in matches if str(match)})

    def _analysis_trajectory_evidence_entries(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        entries: dict[str, Any] = {}
        for task_id, raw_entry in list(value.items())[:8]:
            if not isinstance(raw_entry, dict):
                continue
            entry = {
                "policy_counts": self._analysis_policy_counts(
                    raw_entry.get("policy_counts")
                ),
                "timed_out_commands": self._analysis_evidence_items(
                    raw_entry.get("timed_out_commands"),
                    limit=4,
                ),
                "blocked_guards": self._analysis_evidence_items(
                    raw_entry.get("blocked_guards"),
                    limit=4,
                ),
                "dependency_and_toolchain_evidence": self._analysis_evidence_items(
                    raw_entry.get("dependency_and_toolchain_evidence"),
                    limit=4,
                ),
                "deliverable_progress": self._analysis_evidence_items(
                    raw_entry.get("deliverable_progress"),
                    limit=3,
                ),
                "terminal_environment_markers": self._analysis_evidence_items(
                    raw_entry.get("terminal_environment_markers"),
                    limit=3,
                ),
                "failure_mechanisms": self._analysis_failure_mechanism_items(
                    raw_entry.get("failure_mechanisms"),
                    limit=4,
                ),
            }
            if any(entry.values()):
                entries[str(task_id)] = entry
        return entries

    def _analysis_failure_mechanism_entries(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, dict):
            return []
        entries: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for task_id, raw_entry in list(value.items())[:12]:
            if not isinstance(raw_entry, dict):
                continue
            for item in self._analysis_failure_mechanism_items(
                raw_entry.get("failure_mechanisms"),
                limit=8,
            ):
                name = item.get("name", "")
                if not name:
                    continue
                scoped = dict(item)
                scoped.setdefault("task_id", str(task_id))
                key = (scoped.get("task_id", ""), name)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(scoped)
                if len(entries) >= 12:
                    return entries
        return entries

    def _analysis_failure_mechanism_items(
        self,
        value: Any,
        *,
        limit: int,
    ) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        items: list[dict[str, str]] = []
        for raw_item in value:
            if len(items) >= limit:
                break
            if not isinstance(raw_item, dict):
                continue
            item = {
                "task_id": self._tail_text(str(raw_item.get("task_id") or ""), 120),
                "name": self._tail_text(str(raw_item.get("name") or ""), 160),
                "description": self._tail_text(
                    str(raw_item.get("description") or ""),
                    300,
                ),
                "evidence": self._tail_text(str(raw_item.get("evidence") or ""), 300),
            }
            compact = {key: value for key, value in item.items() if value}
            if compact.get("name"):
                items.append(compact)
        return items

    def _analysis_policy_counts(self, value: Any) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        counts: list[tuple[str, int]] = []
        for raw_name, raw_count in value.items():
            name = str(raw_name).strip()
            if not name:
                continue
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count <= 0:
                continue
            counts.append((name, count))
        counts.sort(key=lambda item: (-item[1], item[0]))
        return {name: count for name, count in counts[:12]}

    def _analysis_evidence_items(self, value: Any, *, limit: int) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        items: list[dict[str, str]] = []
        for raw_item in value:
            if len(items) >= limit:
                break
            if not isinstance(raw_item, dict):
                text = str(raw_item).strip()
                if text:
                    items.append({"command": self._tail_text(text, 300)})
                continue
            command = self._tail_text(str(raw_item.get("command") or ""), 300)
            output_tail = self._tail_text(str(raw_item.get("output_tail") or ""), 300)
            item = {
                "task_id": self._tail_text(str(raw_item.get("task_id") or ""), 120),
                "tool": self._tail_text(str(raw_item.get("tool") or ""), 80),
                "command": command,
                "timed_out": self._tail_text(str(raw_item.get("timed_out") or ""), 20),
                "success": self._tail_text(str(raw_item.get("success") or ""), 20),
                "output_tail": output_tail,
                "guards": self._tail_text(str(raw_item.get("guards") or ""), 300),
                "policies": self._tail_text(str(raw_item.get("policies") or ""), 300),
            }
            compact = {key: value for key, value in item.items() if value}
            if compact:
                items.append(compact)
        return items

    def _recent_summaries(self, *, limit: int | None) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for state in self._recent_campaign_states(limit=None):
            campaign_id = str(state.get("campaign_id") or "")
            raw = state.get("summaries") if state else []
            summaries = raw if isinstance(raw, list) else []
            for summary in reversed(summaries):
                if not isinstance(summary, dict):
                    continue
                summary_id = str(summary.get("summary_id") or "")
                entries.append(
                    {
                        "campaign_id": campaign_id,
                        "summary_id": summary_id,
                        "summary_key": self._summary_key(campaign_id, summary_id),
                        "trial_ids": [
                            str(item) for item in summary.get("trial_ids") or []
                        ],
                        "overall_score": self._safe_float(summary.get("overall_score")),
                        "patches_applied": [
                            str(item) for item in summary.get("patches_applied") or []
                        ],
                        "codex_update_packet_id": str(
                            summary.get("codex_update_packet_id") or ""
                        ),
                        "recorded_at": str(summary.get("recorded_at") or ""),
                    }
                )
        ordered = list(reversed(entries))
        if limit is None:
            return ordered
        limit_value = max(0, int(limit))
        return ordered[-limit_value:] if limit_value else []

    def _recent_completed_trials(
        self,
        *,
        summaries: list[dict[str, Any]],
        trigger_trial_ids: set[str],
        limit: int | None,
    ) -> list[dict[str, Any]]:
        summary_keys = {
            str(summary.get("summary_key") or "")
            for summary in summaries
            if summary.get("summary_key")
        }
        trial_ids = {
            str(trial_id)
            for summary in summaries
            for trial_id in summary.get("trial_ids", [])
        }
        selected: list[dict[str, Any]] = []
        raw_selected: list[tuple[dict[str, Any], str, str]] = []
        seen_trial_ids: set[str] = set()
        states = list(reversed(self._recent_campaign_states(limit=None)))
        for match_mode in ("summary_key", "trial_id", "trigger"):
            for state in states:
                campaign_id = str(state.get("campaign_id") or "")
                completed_raw = state.get("completed") if state else []
                completed = completed_raw if isinstance(completed_raw, list) else []
                for entry in completed:
                    if not isinstance(entry, dict):
                        continue
                    trial_id = str(entry.get("trial_id") or "")
                    summary_id = str(entry.get("summary_id") or "")
                    summary_key = self._summary_key(campaign_id, summary_id)
                    if match_mode == "summary_key":
                        matched = summary_key in summary_keys
                    elif match_mode == "trial_id":
                        matched = trial_id in trial_ids
                    else:
                        matched = trial_id in trigger_trial_ids
                    if not matched:
                        continue
                    if match_mode != "summary_key" and summary_key in summary_keys:
                        continue
                    if match_mode == "trigger" and trial_id in trial_ids:
                        continue
                    if trial_id in seen_trial_ids:
                        continue
                    seen_trial_ids.add(trial_id)
                    raw_selected.append((entry, campaign_id, match_mode))
        hydrate_all = len(raw_selected) <= self._completed_trial_full_context_threshold
        for entry, campaign_id, match_mode in raw_selected:
            trial_id = str(entry.get("trial_id") or "")
            selected.append(
                self._completed_trial_context(
                    entry,
                    campaign_id,
                    hydrate_result=hydrate_all or trial_id in trigger_trial_ids,
                    match_mode=match_mode,
                )
            )
        if limit is None:
            return selected
        limit_value = max(0, int(limit))
        if len(selected) <= limit_value:
            return selected
        trigger_items = [
            item for item in selected if item.get("trial_id") in trigger_trial_ids
        ]
        retained_trigger_ids = {
            str(item.get("trial_id") or "") for item in trigger_items
        }
        regular_items = [
            item
            for item in selected
            if str(item.get("trial_id") or "") not in retained_trigger_ids
        ]
        regular_budget = max(0, limit_value - len(trigger_items))
        retained_regular = regular_items[-regular_budget:] if regular_budget else []
        return retained_regular + trigger_items

    def _completed_trial_context(
        self,
        entry: dict[str, Any],
        campaign_id: str,
        *,
        hydrate_result: bool = True,
        match_mode: str = "",
    ) -> dict[str, Any]:
        trial_id = str(entry.get("trial_id") or "")
        summary_id = str(entry.get("summary_id") or "")
        context = {
            "campaign_id": campaign_id,
            "task_id": str(entry.get("task_id") or ""),
            "trial_id": trial_id,
            "iteration": entry.get("iteration"),
            "summary_id": summary_id,
            "summary_key": self._summary_key(campaign_id, summary_id),
            "status": str(entry.get("status") or ""),
            "score": self._safe_float(entry.get("score")),
            "verified": bool(entry.get("verified")),
            "wall_time_seconds": self._safe_float(entry.get("wall_time_seconds")),
            "failure_category": "",
            "affected_components": [],
            "timeout_phase": "",
            "failure_mechanisms": [],
            "full_result_context": False,
            "full_result_context_deferred": not hydrate_result,
            "context_match_mode": match_mode,
        }
        if not hydrate_result:
            return context
        try:
            trial = self._load_trial_result(trial_id)
        except (OSError, ValueError):
            return context
        context["full_result_context"] = True
        context["full_result_context_deferred"] = False
        attribution = FailureAttributor().analyze(trial)
        base_failure_category = self._base_failure_category(
            trial,
            attribution.failure_category,
        )
        mechanisms = [mechanism.as_dict() for mechanism in failure_mechanisms_for_trial(trial)]
        mechanism_names = [item["name"] for item in mechanisms if item.get("name")]
        phase_attribution = self._infrastructure_phase_attribution(trial)
        if phase_attribution is not None:
            failure_category = phase_attribution[0]
            affected_components = sorted(phase_attribution[1])
        else:
            failure_category = self._mechanism_enhanced_failure_category(
                attribution.failure_category,
                mechanism_names,
            )
            failure_category = self._dependency_loop_enhanced_failure_category(
                trial,
                failure_category,
                mechanism_names,
            )
            affected_components = self._mechanism_enhanced_components(
                attribution.affected_components,
                mechanism_names,
                failure_category=failure_category,
            )
        context.update(
            {
                "failure_category": failure_category,
                "base_failure_category": base_failure_category,
                "affected_components": affected_components,
                "timeout_phase": str(trial.metadata.get("timeout_phase") or ""),
                "failure_mechanisms": mechanisms,
            }
        )
        return context

    def _base_failure_category(self, trial: TrialResult, attribution_category: str) -> str:
        metadata = trial.metadata if isinstance(trial.metadata, dict) else {}
        if (
            str(metadata.get("timeout_phase") or "") == "agent_execution"
            and trial.status.value == "timeout"
        ):
            return "agent_execution_timeout"
        return attribution_category

    def _infrastructure_phase_attribution(
        self,
        trial: TrialResult,
    ) -> tuple[str, tuple[str, ...]] | None:
        metadata = trial.metadata if isinstance(trial.metadata, dict) else {}
        timeout_phase = str(metadata.get("timeout_phase") or "")
        return INFRASTRUCTURE_PHASE_ATTRIBUTION.get(timeout_phase)

    def _infrastructure_triage(
        self,
        failing_tasks: list[dict[str, Any]],
        campaign_context: dict[str, Any],
        *,
        limit: int = 10,
    ) -> dict[str, Any]:
        trigger_items = self._infrastructure_triage_items_from_tasks(
            failing_tasks,
            source="trigger_failures",
            limit=limit,
        )
        trigger_infrastructure_count = sum(
            1
            for task in failing_tasks
            if isinstance(task, dict) and self._task_summary_is_infrastructure(task)
        )
        recent_items = self._infrastructure_triage_items_from_reports(
            campaign_context.get("recent_analysis_reports"),
            limit=limit,
        )
        infrastructure_categories = sorted(
            {
                str(item.get("failure_category") or "")
                for item in [*trigger_items, *recent_items]
                if str(item.get("failure_category") or "")
            }
        )
        non_infrastructure_categories = sorted(
            {
                str(item.get("failure_category") or "")
                for item in self._non_infrastructure_failure_items(
                    failing_tasks,
                    campaign_context,
                )
                if str(item.get("failure_category") or "")
            }
        )
        total_trigger_failures = len(failing_tasks)
        trigger_all_infrastructure = (
            total_trigger_failures > 0
            and trigger_infrastructure_count == total_trigger_failures
        )
        recommended_layers = sorted(
            {
                component
                for item in [*trigger_items, *recent_items]
                for component in item.get("affected_components") or []
                if str(component).startswith("bench/")
            }
        )
        if not recommended_layers and (trigger_items or recent_items):
            recommended_layers = ["bench/harbor", "bench/network_environment"]
        avoid_layers: list[str] = []
        if trigger_all_infrastructure and not non_infrastructure_categories:
            avoid_layers = [
                "crates/hl-worker-core",
                "harness/prompts",
                "harness/tools",
                "recovery/patterns",
            ]
        if trigger_items or recent_items:
            guidance = (
                "Infrastructure-only timeout phases are Harbor/environment evidence, "
                "not proof of a Worker reasoning or prompt failure. Prefer bounded "
                "changes in Harbor adapter, network/environment attribution, retry "
                "evidence, or packet routing. Touch Worker/harness policy only when "
                "a separate non-infrastructure bucket or trajectory mechanism proves "
                "a Worker-owned recovery gap."
            )
        else:
            guidance = "No infrastructure-only timeout buckets were detected in this packet."
        return {
            "trigger_infrastructure_count": trigger_infrastructure_count,
            "trigger_total_failures": total_trigger_failures,
            "trigger_all_infrastructure": trigger_all_infrastructure,
            "infrastructure_categories": infrastructure_categories,
            "non_infrastructure_categories": non_infrastructure_categories,
            "trigger_items": trigger_items,
            "recent_items": recent_items,
            "recommended_layers": recommended_layers,
            "avoid_worker_policy_layers_when_infrastructure_only": avoid_layers,
            "selection_guidance": guidance,
            "loop_stop_condition": False,
            "time_round_token_limit_driven": False,
            "attempt_count_stop_condition": False,
            "timeout_seconds_stop_condition": False,
        }

    def _self_harness_improvement_queue(
        self,
        *,
        failure_pattern_digest: dict[str, Any],
        policy_recurrence_signals: list[dict[str, Any]],
        infrastructure_triage: dict[str, Any],
        campaign_context: dict[str, Any],
        failing_tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for raw_signature in failure_pattern_digest.get("weakness_signatures") or []:
            if not isinstance(raw_signature, dict):
                continue
            candidate = self._self_harness_candidate_from_weakness(raw_signature)
            if candidate:
                candidates.append(candidate)
        for raw_pattern in failure_pattern_digest.get("mechanism_patterns") or []:
            if not isinstance(raw_pattern, dict):
                continue
            candidate = self._self_harness_candidate_from_mechanism(raw_pattern)
            if candidate:
                candidates.append(candidate)
        for raw_entry in self._mechanism_update_entries_from_reports(
            campaign_context.get("recent_analysis_reports")
        ):
            candidate = self._self_harness_candidate_from_mechanism_update_entry(
                raw_entry
            )
            if candidate:
                candidates.append(candidate)
        for raw_signal in policy_recurrence_signals:
            if not isinstance(raw_signal, dict):
                continue
            candidate = self._self_harness_candidate_from_policy_recurrence(raw_signal)
            if candidate:
                candidates.append(candidate)
        for raw_item in infrastructure_triage.get("trigger_items") or []:
            if isinstance(raw_item, dict):
                candidate = self._self_harness_candidate_from_infrastructure(raw_item)
                if candidate:
                    candidates.append(candidate)
        for raw_item in infrastructure_triage.get("recent_items") or []:
            if isinstance(raw_item, dict):
                candidate = self._self_harness_candidate_from_infrastructure(raw_item)
                if candidate:
                    candidates.append(candidate)

        merged = self._merge_self_harness_candidates(candidates)
        merged.sort(
            key=lambda item: (
                -int(item.get("score_audit_only") or 0),
                str(item.get("candidate_id") or ""),
            )
        )
        for index, candidate in enumerate(merged, start=1):
            candidate["rank_audit_only"] = index
            candidate["rank_stop_condition"] = False
            candidate["selection_stop_condition"] = False
            candidate["loop_stop_condition"] = False
            candidate["time_round_token_limit_driven"] = False

        trigger_categories = sorted(
            {
                str(task.get("failure_category") or "")
                for task in failing_tasks
                if isinstance(task, dict) and str(task.get("failure_category") or "")
            }
        )
        summary_ids = sorted(
            {
                str(report.get("summary_id") or "")
                for report in campaign_context.get("recent_analysis_reports") or []
                if isinstance(report, dict) and str(report.get("summary_id") or "")
            }
        )
        return {
            "objective": (
                "Self-Harness candidate queue built from verifier-grounded recent "
                "weakness signatures, mechanism patterns, mechanism update classes, "
                "policy recurrence, and infrastructure triage. Use it to choose one "
                "bounded Worker/harness or updater policy slice; do not treat the "
                "queue as permission to rewrite the framework or solve task ids directly."
            ),
            "source_practices": [
                "weakness_mining_from_real_trajectories",
                "mechanism_signature_clustering",
                "bounded_harness_proposal",
                "deterministic_validation_or_rollback",
                "same_model_frontier_or_regression_comparison",
            ],
            "trigger_failure_categories": trigger_categories,
            "recent_summary_ids": summary_ids,
            "candidates": merged,
            "candidate_count_audit_only": len(merged),
            "candidate_count_stop_condition": False,
            "selection_limit_stop_condition": False,
            "time_round_attempt_limit_stop_condition": False,
            "loop_stop_condition": False,
            "time_round_token_limit_driven": False,
            "selection_guidance": [
                "Prefer the top candidate only when its evidence matches the trigger or recent comparable failures.",
                "Choose exactly one bounded slice and cite candidate_id in cross_round_evidence.selected_problem_class when used.",
                "If infrastructure candidates dominate and no non-infrastructure evidence exists, route to Harbor/environment attribution instead of Worker prompt changes.",
                "If the selected candidate is recurrence_under_existing_policy, strengthen timing, gating, or validation placement before adding another duplicate recognizer.",
                "Validate with the candidate validation_surfaces and keep all *_stop_condition fields false.",
            ],
        }

    def _self_harness_candidate_from_weakness(
        self,
        signature: dict[str, Any],
    ) -> dict[str, Any]:
        label = str(signature.get("signature") or signature.get("failure_category") or "").strip()
        category = str(signature.get("failure_category") or "").strip()
        if not label and not category:
            return {}
        components = [str(item) for item in signature.get("affected_components") or []]
        mechanisms = [str(item) for item in signature.get("failure_mechanisms") or []]
        count = int(signature.get("count") or 0)
        return self._base_self_harness_candidate(
            source="weakness_signature",
            label=label or category,
            failure_category=category,
            score=count * 4 + len(set(signature.get("task_ids") or [])) * 2,
            task_ids=[str(item) for item in signature.get("task_ids") or []],
            affected_components=components,
            timeout_phases=[str(item) for item in signature.get("timeout_phases") or []],
            failure_mechanisms=mechanisms,
            evidence={
                "verifier_failure": str(signature.get("verifier_failure") or ""),
                "agent_contribution": str(signature.get("agent_contribution") or ""),
                "reusable_mechanism": str(signature.get("reusable_mechanism") or ""),
                "summary_ids": [str(item) for item in signature.get("summary_ids") or []],
                "evidence_sources": [str(item) for item in signature.get("evidence_sources") or []],
            },
        )

    def _self_harness_candidate_from_mechanism(
        self,
        pattern: dict[str, Any],
    ) -> dict[str, Any]:
        label = str(pattern.get("signature") or "").strip()
        category = str(pattern.get("failure_category") or "").strip()
        if not label and not category:
            return {}
        count = int(pattern.get("count") or 0)
        return self._base_self_harness_candidate(
            source="mechanism_pattern",
            label=label or category,
            failure_category=category,
            score=count * 3 + len(set(pattern.get("task_ids") or [])),
            task_ids=[str(item) for item in pattern.get("task_ids") or []],
            affected_components=[str(item) for item in pattern.get("affected_components") or []],
            timeout_phases=[str(pattern.get("timeout_phase") or "")],
            failure_mechanisms=[str(item) for item in pattern.get("failure_mechanisms") or []],
            evidence={
                "status": str(pattern.get("status") or ""),
                "summary_ids": [str(item) for item in pattern.get("summary_ids") or []],
                "verified_failures": int(pattern.get("verified_failures") or 0),
            },
        )

    def _mechanism_update_entries_from_reports(self, reports: Any) -> list[dict[str, Any]]:
        if not isinstance(reports, list):
            return []
        entries: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for report in reports:
            if not isinstance(report, dict):
                continue
            summary_id = str(report.get("summary_id") or "")
            for raw_entry in report.get("mechanism_update_entries") or []:
                if not isinstance(raw_entry, dict):
                    continue
                category = str(raw_entry.get("failure_category") or "").strip()
                mechanism = str(raw_entry.get("mechanism") or "").strip()
                if not category or not mechanism:
                    continue
                key = (summary_id, category, mechanism)
                if key in seen:
                    continue
                seen.add(key)
                entry = dict(raw_entry)
                if summary_id:
                    entry.setdefault("summary_id", summary_id)
                entries.append(entry)
        return entries

    def _self_harness_candidate_from_mechanism_update_entry(
        self,
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        category = str(entry.get("failure_category") or "").strip()
        mechanism = str(entry.get("mechanism") or "").strip()
        if not category or not mechanism:
            return {}
        count = int(entry.get("count") or 0)
        candidate = self._base_self_harness_candidate(
            source="mechanism_update_class",
            label="|".join([category, mechanism]),
            failure_category=category,
            score=count * 5 + len(set(entry.get("task_ids") or [])) * 2,
            task_ids=[str(item) for item in entry.get("task_ids") or []],
            affected_components=[
                str(item) for item in entry.get("affected_components") or []
            ],
            timeout_phases=[],
            failure_mechanisms=[mechanism],
            evidence={
                "summary_id": str(entry.get("summary_id") or ""),
                "mechanism": mechanism,
                "source": "analysis_mechanism_update_classes",
            },
        )
        candidate["proposal_kind"] = "mechanism_targeted_harness_policy_slice"
        candidate["why_candidate_generalizes"] = (
            "The analysis bucket may be broad, but this candidate is keyed by a "
            "specific extracted failure mechanism and its affected Worker/harness "
            "surfaces, so it can target reusable behavior without task-id branching."
        )
        return candidate

    def _self_harness_candidate_from_policy_recurrence(
        self,
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        category = str(signal.get("failure_category") or "").strip()
        policy = str(signal.get("policy") or "").strip()
        if not category and not policy:
            return {}
        count = int(signal.get("count") or 0)
        coverage = int(signal.get("policy_coverage_count") or 0)
        trajectory = int(signal.get("trajectory_policy_count") or 0)
        candidate = self._base_self_harness_candidate(
            source="policy_recurrence_signal",
            label="|".join(part for part in [category, policy] if part),
            failure_category=category,
            score=count * 4 + coverage * 2 + trajectory * 2,
            task_ids=[str(item) for item in signal.get("task_ids") or []],
            affected_components=[str(item) for item in signal.get("affected_components") or []],
            timeout_phases=[str(item) for item in signal.get("timeout_phases") or []],
            failure_mechanisms=[str(signal.get("mechanism") or "")],
            evidence={
                "summary_id": str(signal.get("summary_id") or ""),
                "policy": policy,
                "policy_coverage_count": coverage,
                "trajectory_policy_count": trajectory,
                "interpretation": str(signal.get("interpretation") or ""),
                "update_hint": str(signal.get("update_hint") or ""),
            },
        )
        candidate["proposal_kind"] = "strengthen_existing_policy"
        candidate["recurrence_under_existing_policy"] = True
        candidate["why_candidate_generalizes"] = (
            "The current analysis policy already recognizes this class, but the "
            "same class remains in failing buckets; improve when the policy becomes "
            "actionable, gated, or validated rather than adding a duplicate label."
        )
        return candidate

    def _self_harness_candidate_from_infrastructure(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        category = str(item.get("failure_category") or "").strip()
        if not category:
            return {}
        count = int(item.get("count") or 1)
        evidence = {
            "source": str(item.get("source") or ""),
            "summary_id": str(item.get("summary_id") or ""),
            "routing": str(item.get("routing") or ""),
            "environment_start_attribution_hint": str(
                item.get("environment_start_attribution_hint") or ""
            ),
        }
        warmup_commands = [
            str(command)
            for command in item.get("prebuilt_image_cache_warmup_commands") or []
        ][:3]
        if item.get("prebuilt_image_cache_miss_detected"):
            evidence["prebuilt_image_cache_miss_detected"] = True
        if warmup_commands:
            evidence["prebuilt_image_cache_warmup_commands"] = warmup_commands
        if item.get("network_preflight_recommended"):
            evidence["network_preflight_recommended"] = True
        candidate = self._base_self_harness_candidate(
            source="infrastructure_triage",
            label=category,
            failure_category=category,
            score=count * 3 + len(set(item.get("task_ids") or [])),
            task_ids=[str(task_id) for task_id in item.get("task_ids") or []],
            affected_components=[str(component) for component in item.get("affected_components") or []],
            timeout_phases=[str(phase) for phase in item.get("timeout_phases") or []],
            failure_mechanisms=[],
            evidence=evidence,
        )
        candidate["proposal_kind"] = "infrastructure_attribution_or_routing"
        candidate["avoid_layers"] = [
            "crates/hl-worker-core",
            "harness/prompts",
            "harness/tools",
        ]
        candidate["why_candidate_generalizes"] = (
            "Infrastructure timeout phases are recurring Harbor/environment evidence; "
            "improve attribution, retry evidence, packet routing, or environment "
            "diagnostics before treating them as Worker reasoning failures."
        )
        return candidate

    def _base_self_harness_candidate(
        self,
        *,
        source: str,
        label: str,
        failure_category: str,
        score: int,
        task_ids: list[str],
        affected_components: list[str],
        timeout_phases: list[str],
        failure_mechanisms: list[str],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        clean_components = sorted(dict.fromkeys(item for item in affected_components if item))
        clean_mechanisms = sorted(dict.fromkeys(item for item in failure_mechanisms if item))
        clean_timeout_phases = sorted(dict.fromkeys(item for item in timeout_phases if item))
        edit_surfaces = self._self_harness_edit_surfaces(clean_components, clean_timeout_phases)
        validation_surfaces = self._self_harness_validation_surfaces(
            clean_components,
            clean_mechanisms,
            clean_timeout_phases,
        )
        candidate_id = self._self_harness_candidate_id(source, label or failure_category)
        return {
            "candidate_id": candidate_id,
            "source": source,
            "evidence_label": label or failure_category,
            "failure_category": failure_category,
            "proposal_kind": "bounded_harness_policy_slice",
            "score_audit_only": max(score, 0),
            "score_stop_condition": False,
            "task_ids": sorted(dict.fromkeys(task_id for task_id in task_ids if task_id))[:12],
            "affected_components": clean_components[:12],
            "timeout_phases": clean_timeout_phases[:8],
            "failure_mechanisms": clean_mechanisms[:12],
            "evidence": self._compact_self_harness_evidence(evidence),
            "recommended_edit_surfaces": edit_surfaces,
            "validation_surfaces": validation_surfaces,
            "why_candidate_generalizes": (
                "The candidate is keyed by verifier-grounded category, mechanism, "
                "component, and repeated task evidence rather than a single task id."
            ),
            "anti_overfit_checks": [
                "Do not branch on task_id or benchmark fixture names.",
                "Keep benchmark tasks/tests/solutions/verifier definitions untouched.",
                "Tie prediction.expected_fixed_task_classes to this candidate evidence_label or failure_category.",
            ],
            "benchmark_integrity_stop_condition": False,
        }

    def _self_harness_edit_surfaces(
        self,
        components: list[str],
        timeout_phases: list[str],
    ) -> list[str]:
        surfaces: set[str] = set()
        for component in components:
            if component.startswith("bench/harbor") or component.startswith("bench/network"):
                surfaces.update({"bench/harbor.py", "bench/harbor_adapter.py", "hl/attribution.py"})
            elif component.startswith("bench/agent"):
                surfaces.update({"bench/agent.py", "crates/hl-worker-core/src/main.rs"})
            elif component.startswith("harness"):
                surfaces.add(component)
            elif component.startswith("recovery"):
                surfaces.update({"harness/recovery", "harness/tools"})
            elif component.startswith("context"):
                surfaces.update({"harness/context", "crates/hl-worker-core/src/main.rs"})
            elif component:
                surfaces.add(component)
        if any(phase in INFRASTRUCTURE_PHASE_ATTRIBUTION for phase in timeout_phases):
            surfaces.update({"bench/harbor.py", "bench/harbor_adapter.py"})
        if not surfaces:
            surfaces.update({"meta/packager.py", "scripts/run_campaign.py", "tests"})
        surfaces.add("tests")
        return sorted(surfaces)

    def _self_harness_validation_surfaces(
        self,
        components: list[str],
        mechanisms: list[str],
        timeout_phases: list[str],
    ) -> list[str]:
        surfaces = {"targeted pytest for touched policy", "git diff --check"}
        if any(component.startswith("bench/harbor") for component in components) or any(
            phase in INFRASTRUCTURE_PHASE_ATTRIBUTION for phase in timeout_phases
        ):
            surfaces.add("tests/test_roadmap_harbor.py targeted infra attribution tests")
        if any(component.startswith("bench/agent") for component in components):
            surfaces.add("tests/test_models_and_worker_policy.py targeted Worker policy tests")
        if any(component.startswith("harness/tools") for component in components):
            surfaces.add("tests/test_tool_registry.py targeted tool policy tests")
        if mechanisms:
            surfaces.add("failure mechanism fixture or analysis replay for " + ",".join(mechanisms[:3]))
        surfaces.add("loop-limit regression for master/sub-agent/Worker stop-condition metadata")
        return sorted(surfaces)

    def _self_harness_candidate_id(self, source: str, label: str) -> str:
        normalized = "".join(
            char.lower() if char.isalnum() else "-" for char in label
        ).strip("-")
        while "--" in normalized:
            normalized = normalized.replace("--", "-")
        if not normalized:
            normalized = "unlabeled"
        return "self-harness-" + source.replace("_", "-") + "-" + normalized[:96]

    def _compact_self_harness_evidence(self, evidence: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for key, value in evidence.items():
            if value in ("", [], {}, None):
                continue
            if isinstance(value, str):
                compact[key] = self._tail_text(value, 500)
            elif isinstance(value, list):
                compact[key] = [str(item) for item in value if str(item).strip()][:12]
            else:
                compact[key] = value
        return compact

    def _merge_self_harness_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                continue
            if candidate_id not in merged:
                merged[candidate_id] = dict(candidate)
                order.append(candidate_id)
                continue
            target = merged[candidate_id]
            target["score_audit_only"] = int(target.get("score_audit_only") or 0) + int(
                candidate.get("score_audit_only") or 0
            )
            target["score_stop_condition"] = False
            for field in (
                "task_ids",
                "affected_components",
                "timeout_phases",
                "failure_mechanisms",
                "recommended_edit_surfaces",
                "validation_surfaces",
                "anti_overfit_checks",
                "avoid_layers",
            ):
                self._merge_candidate_list_field(target, candidate, field)
            if candidate.get("recurrence_under_existing_policy"):
                target["recurrence_under_existing_policy"] = True
            target.setdefault("evidence_sources", [])
            sources = list(target.get("evidence_sources") or [])
            source = str(candidate.get("source") or "")
            if source and source not in sources:
                sources.append(source)
            target["evidence_sources"] = sources
        return [merged[key] for key in order]

    def _merge_candidate_list_field(
        self,
        target: dict[str, Any],
        source: dict[str, Any],
        field: str,
    ) -> None:
        values = [str(item) for item in target.get(field) or []]
        seen = set(values)
        for item in source.get(field) or []:
            text = str(item)
            if text and text not in seen:
                values.append(text)
                seen.add(text)
        if values:
            target[field] = values[:12]

    def _infrastructure_triage_items_from_tasks(
        self,
        tasks: list[dict[str, Any]],
        *,
        source: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for task in tasks:
            if len(items) >= limit:
                break
            if not isinstance(task, dict) or not self._task_summary_is_infrastructure(task):
                continue
            category = str(task.get("failure_category") or "").strip()
            item = {
                "source": source,
                "failure_category": category,
                "task_ids": [str(task.get("task_id") or "")],
                "trial_ids": [str(task.get("trial_id") or "")],
                "timeout_phases": [str(task.get("timeout_phase") or "")],
                "affected_components": [
                    str(component)
                    for component in task.get("affected_components") or []
                ],
                "infra_error_detected": bool(task.get("infra_error_detected")),
                "environment_start_attribution_hint": self._tail_text(
                    str(task.get("environment_start_attribution_hint") or ""),
                    300,
                ),
                "routing": "infrastructure_harbor_or_environment",
            }
            warmup_commands = [
                str(command)
                for command in task.get("prebuilt_image_cache_warmup_commands") or []
            ][:3]
            if task.get("prebuilt_image_cache_miss_detected"):
                item["prebuilt_image_cache_miss_detected"] = True
            if warmup_commands:
                item["prebuilt_image_cache_warmup_commands"] = warmup_commands
            if task.get("network_preflight_recommended"):
                item["network_preflight_recommended"] = True
            items.append(self._compact_infrastructure_triage_item(item))
        return items

    def _infrastructure_triage_items_from_reports(
        self,
        reports: Any,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(reports, list):
            return []
        items: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for report in reports:
            if len(items) >= limit:
                break
            if not isinstance(report, dict):
                continue
            summary_id = str(report.get("summary_id") or "")
            for bucket in report.get("failure_buckets") or []:
                if len(items) >= limit:
                    break
                if not isinstance(bucket, dict) or not self._failure_bucket_is_infrastructure(bucket):
                    continue
                category = str(bucket.get("failure_category") or "").strip()
                key = (summary_id, category)
                if key in seen:
                    continue
                seen.add(key)
                item = {
                    "source": "recent_analysis_reports",
                    "summary_id": summary_id,
                    "failure_category": category,
                    "count": int(bucket.get("count") or 0),
                    "task_ids": [
                        str(task_id) for task_id in bucket.get("task_ids") or []
                    ][:12],
                    "timeout_phases": [
                        str(phase) for phase in bucket.get("timeout_phases") or []
                    ][:8],
                    "affected_components": [
                        str(component)
                        for component in bucket.get("affected_components") or []
                    ][:8],
                    "routing": "infrastructure_harbor_or_environment",
                }
                items.append(self._compact_infrastructure_triage_item(item))
        return items

    def _non_infrastructure_failure_items(
        self,
        failing_tasks: list[dict[str, Any]],
        campaign_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for task in failing_tasks:
            if isinstance(task, dict) and not self._task_summary_is_infrastructure(task):
                items.append(task)
        for report in campaign_context.get("recent_analysis_reports") or []:
            if not isinstance(report, dict):
                continue
            for bucket in report.get("failure_buckets") or []:
                if isinstance(bucket, dict) and not self._failure_bucket_is_infrastructure(bucket):
                    items.append(bucket)
        return items

    def _task_summary_is_infrastructure(self, task: dict[str, Any]) -> bool:
        if bool(task.get("infra_error_detected")):
            return True
        return self._infrastructure_category_or_phase(
            str(task.get("failure_category") or ""),
            [str(task.get("timeout_phase") or "")],
        )

    def _failure_bucket_is_infrastructure(self, bucket: dict[str, Any]) -> bool:
        if bool(bucket.get("infrastructure")):
            return True
        return self._infrastructure_category_or_phase(
            str(bucket.get("failure_category") or ""),
            [str(phase) for phase in bucket.get("timeout_phases") or []],
        )

    def _infrastructure_category_or_phase(
        self,
        category: str,
        timeout_phases: list[str],
    ) -> bool:
        category = category.strip()
        if category in {"environment_start_timeout", "verifier_runtime_prepare_timeout"}:
            return True
        return any(phase in INFRASTRUCTURE_PHASE_ATTRIBUTION for phase in timeout_phases)

    def _compact_infrastructure_triage_item(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        compact = {}
        for key, value in item.items():
            if value in ("", [], {}, None):
                continue
            compact[key] = value
        return compact

    def _mechanism_enhanced_failure_category(
        self,
        base_category: str,
        mechanism_names: list[str],
    ) -> str:
        mechanism_name_set = set(mechanism_names)
        if (
            "terminal_environment_unavailable_after_dependency_loop_mechanism"
            in mechanism_name_set
            and "cython_extension_optional_import_pivot_mechanism" in mechanism_name_set
        ):
            return "terminal_environment_unavailable_after_dependency_loop"
        protected_verifier_contract = bool(
            mechanism_name_set & PRIMARY_VERIFIER_CONTRACT_MECHANISM_NAMES
        )
        for mechanism_name, override in MECHANISM_CATEGORY_OVERRIDES.items():
            if mechanism_name in mechanism_name_set:
                if protected_verifier_contract and mechanism_name not in {
                    "async_cancellation_cleanup_contract",
                    "model_extraction_matrix_contract",
                }:
                    continue
                return override
        return base_category

    def _dependency_loop_enhanced_failure_category(
        self,
        trial: TrialResult,
        base_category: str,
        mechanism_names: list[str],
    ) -> str:
        category = dependency_loop_failure_category_for_trial(
            trial,
            mechanism_names,
        )
        return category or base_category

    def _mechanism_enhanced_components(
        self,
        base_components: list[str],
        mechanism_names: list[str],
        *,
        failure_category: str = "",
    ) -> list[str]:
        enhanced_mechanism_names = list(mechanism_names)
        category_mechanism = dependency_loop_mechanism_for_failure_category(
            failure_category
        )
        if category_mechanism and category_mechanism not in enhanced_mechanism_names:
            enhanced_mechanism_names.insert(0, category_mechanism)
        replace_base = failure_mechanisms_replace_base_components(
            enhanced_mechanism_names
        )
        components = set() if replace_base else {
            str(component) for component in base_components if component
        }
        for mechanism_name in enhanced_mechanism_names:
            if (
                replace_base
                and mechanism_name in DEPENDENCY_LOOP_BASE_REPLACEMENT_NEUTRAL_MECHANISM_NAMES
            ):
                continue
            components.update(affected_components_for_failure_mechanism(mechanism_name))
        return sorted(components)

    def _failure_pattern_digest(
        self,
        failures: list[TrialResult],
        campaign_context: dict[str, Any],
    ) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        mechanism_groups: dict[str, dict[str, Any]] = {}
        report_weakness_signatures = self._weakness_signatures_from_reports(
            campaign_context.get("recent_analysis_reports")
        )
        for entry in campaign_context.get("recent_completed_trials", []) or []:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status") or "")
            if status == "passed":
                continue
            category = str(entry.get("failure_category") or status or "unknown")
            affected_components = [
                str(component)
                for component in entry.get("affected_components") or []
                if component
            ]
            timeout_phase = str(entry.get("timeout_phase") or "")
            failure_mechanisms = [
                str(item.get("name") or "")
                for item in entry.get("failure_mechanisms") or []
                if isinstance(item, dict) and str(item.get("name") or "")
            ]
            group = groups.setdefault(
                category,
                {
                    "failure_category": category,
                    "count": 0,
                    "task_ids": set(),
                    "trial_ids": set(),
                    "statuses": {},
                    "affected_components": set(),
                    "timeout_phases": set(),
                    "summary_ids": set(),
                    "failure_mechanisms": set(),
                    "verified_failures": 0,
                    "total_wall_time_seconds": 0.0,
                },
            )
            group["count"] += 1
            group["task_ids"].add(str(entry.get("task_id") or ""))
            group["trial_ids"].add(str(entry.get("trial_id") or ""))
            group["summary_ids"].add(str(entry.get("summary_id") or ""))
            group.setdefault("summary_keys", set()).add(
                str(entry.get("summary_key") or "")
            )
            group["total_wall_time_seconds"] += self._safe_float(
                entry.get("wall_time_seconds")
            )
            if entry.get("verified"):
                group["verified_failures"] += 1
            group["statuses"][status] = int(group["statuses"].get(status, 0)) + 1
            if timeout_phase:
                group["timeout_phases"].add(timeout_phase)
            for mechanism_name in failure_mechanisms:
                group["failure_mechanisms"].add(mechanism_name)
            for component in affected_components:
                group["affected_components"].add(component)

            signature = self._failure_mechanism_signature(
                failure_category=category,
                status=status,
                timeout_phase=timeout_phase,
                affected_components=affected_components,
                failure_mechanisms=failure_mechanisms,
            )
            mechanism = mechanism_groups.setdefault(
                signature,
                {
                    "signature": signature,
                    "failure_category": category,
                    "status": status,
                    "timeout_phase": timeout_phase,
                    "affected_components": set(affected_components),
                    "failure_mechanisms": set(failure_mechanisms),
                    "count": 0,
                    "task_ids": set(),
                    "trial_ids": set(),
                    "summary_ids": set(),
                    "verified_failures": 0,
                    "total_wall_time_seconds": 0.0,
                },
            )
            mechanism["count"] += 1
            mechanism["task_ids"].add(str(entry.get("task_id") or ""))
            mechanism["trial_ids"].add(str(entry.get("trial_id") or ""))
            mechanism["summary_ids"].add(str(entry.get("summary_id") or ""))
            mechanism["total_wall_time_seconds"] += self._safe_float(
                entry.get("wall_time_seconds")
            )
            if entry.get("verified"):
                mechanism["verified_failures"] += 1
            for mechanism_name in failure_mechanisms:
                mechanism["failure_mechanisms"].add(mechanism_name)

        patterns: list[dict[str, Any]] = []
        for group in groups.values():
            count = int(group["count"])
            total_wall = float(group["total_wall_time_seconds"])
            patterns.append(
                {
                    "failure_category": group["failure_category"],
                    "count": count,
                    "task_ids": sorted(group["task_ids"])[:12],
                    "trial_ids": sorted(group["trial_ids"])[:12],
                    "summary_ids": sorted(group["summary_ids"]),
                    "summary_keys": sorted(group.get("summary_keys", set())),
                    "statuses": dict(sorted(group["statuses"].items())),
                    "affected_components": sorted(group["affected_components"])[:12],
                    "timeout_phases": sorted(group["timeout_phases"]),
                    "failure_mechanisms": sorted(group["failure_mechanisms"]),
                    "verified_failures": int(group["verified_failures"]),
                    "average_wall_time_seconds": (
                        round(total_wall / count, 3) if count else 0.0
                    ),
                }
            )
        patterns.sort(
            key=lambda item: (
                -int(item["count"]),
                str(item["failure_category"]),
            )
        )
        patterns = _demote_discouraged_patterns(
            patterns,
            self._discouraged_failure_categories(),
        )
        mechanism_patterns: list[dict[str, Any]] = []
        for group in mechanism_groups.values():
            count = int(group["count"])
            total_wall = float(group["total_wall_time_seconds"])
            mechanism_patterns.append(
                {
                    "signature": group["signature"],
                    "failure_category": group["failure_category"],
                    "status": group["status"],
                    "timeout_phase": group["timeout_phase"],
                    "affected_components": sorted(group["affected_components"])[:12],
                    "failure_mechanisms": sorted(group["failure_mechanisms"]),
                    "count": count,
                    "task_ids": sorted(group["task_ids"])[:12],
                    "trial_ids": sorted(group["trial_ids"])[:12],
                    "summary_ids": sorted(group["summary_ids"]),
                    "verified_failures": int(group["verified_failures"]),
                    "average_wall_time_seconds": (
                        round(total_wall / count, 3) if count else 0.0
                    ),
                }
            )
        mechanism_patterns.sort(
            key=lambda item: (
                -int(item["count"]),
                str(item["signature"]),
            )
        )
        trigger_categories = sorted(
            {
                FailureAttributor().analyze(trial).failure_category
                for trial in failures
                if trial.status.value != "passed"
            }
        )
        return {
            "purpose": (
                "Use dominant cross-round patterns to pick one generalizable "
                "policy improvement; do not optimize only for the trigger tasks."
            ),
            "trigger_failure_categories": trigger_categories,
            "patterns": patterns[:8],
            "dominant_pattern": patterns[0] if patterns else {},
            "mechanism_patterns": mechanism_patterns[:12],
            "dominant_mechanism_pattern": (
                mechanism_patterns[0] if mechanism_patterns else {}
            ),
            "weakness_signatures": report_weakness_signatures[:12],
            "dominant_weakness_signature": (
                report_weakness_signatures[0] if report_weakness_signatures else {}
            ),
            "mechanism_signature_contract": {
                "fields": [
                    "failure_category",
                    "status",
                    "timeout_phase",
                    "affected_components",
                    "failure_mechanisms",
                ],
                "purpose": (
                    "Verifier-grounded weakness mining: separate failures that "
                    "share a broad category but differ in agent behavior, timeout "
                    "phase, or affected harness surface."
                ),
            },
            "selection_guidance": [
                "Prefer a pattern recurring across multiple tasks or summaries.",
                "Use mechanism_patterns to avoid merging broad categories with different root mechanisms.",
                "Use weakness_signatures when available to require the same verifier failure, agent behavior contribution, and reusable mechanism before merging a cluster.",
                "If the trigger failures are outliers, choose a no-op or gather more evidence.",
                "A patch should improve the Worker/harness capability behind the pattern.",
            ],
        }

    def _weakness_signatures_from_reports(self, reports: Any) -> list[dict[str, Any]]:
        if not isinstance(reports, list):
            return []
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for report in reports:
            if not isinstance(report, dict):
                continue
            summary_id = str(report.get("summary_id") or "")
            for raw_entry in report.get("weakness_signatures") or []:
                if not isinstance(raw_entry, dict):
                    continue
                signature = str(raw_entry.get("signature") or "").strip()
                category = str(raw_entry.get("failure_category") or "").strip()
                if not signature and category:
                    signature = "category=" + category
                if not signature:
                    continue
                entry = merged.setdefault(
                    signature,
                    {
                        "signature": signature,
                        "verifier_failure": str(
                            raw_entry.get("verifier_failure") or ""
                        ),
                        "agent_contribution": str(
                            raw_entry.get("agent_contribution") or ""
                        ),
                        "reusable_mechanism": str(
                            raw_entry.get("reusable_mechanism") or ""
                        ),
                        "failure_category": category,
                        "count": 0,
                        "task_ids": set(),
                        "affected_components": set(),
                        "timeout_phases": set(),
                        "failure_mechanisms": set(),
                        "evidence_sources": set(),
                        "summary_ids": set(),
                        "loop_stop_condition": False,
                        "time_round_token_limit_driven": False,
                        "synthesized_from_legacy_analysis": False,
                    },
                )
                if signature not in order:
                    order.append(signature)
                try:
                    count = int(raw_entry.get("count") or 0)
                except (TypeError, ValueError):
                    count = 0
                entry["count"] = int(entry.get("count") or 0) + max(count, 0)
                if summary_id:
                    entry["summary_ids"].add(summary_id)
                for field in (
                    "task_ids",
                    "affected_components",
                    "timeout_phases",
                    "failure_mechanisms",
                    "evidence_sources",
                ):
                    for value in raw_entry.get(field) or []:
                        text = str(value).strip()
                        if text:
                            entry[field].add(text)
                entry["loop_stop_condition"] = bool(
                    entry.get("loop_stop_condition")
                    or raw_entry.get("loop_stop_condition")
                )
                entry["time_round_token_limit_driven"] = bool(
                    entry.get("time_round_token_limit_driven")
                    or raw_entry.get("time_round_token_limit_driven")
                )
                entry["synthesized_from_legacy_analysis"] = bool(
                    entry.get("synthesized_from_legacy_analysis")
                    or raw_entry.get("synthesized_from_legacy_analysis")
                )
        entries: list[dict[str, Any]] = []
        for signature in order:
            entry = merged[signature]
            result = {
                "signature": entry["signature"],
                "verifier_failure": entry["verifier_failure"],
                "agent_contribution": entry["agent_contribution"],
                "reusable_mechanism": entry["reusable_mechanism"],
                "failure_category": entry["failure_category"],
                "count": int(entry.get("count") or 0),
                "task_ids": sorted(entry["task_ids"])[:12],
                "affected_components": sorted(entry["affected_components"])[:12],
                "timeout_phases": sorted(entry["timeout_phases"])[:8],
                "failure_mechanisms": sorted(entry["failure_mechanisms"])[:12],
                "evidence_sources": sorted(entry["evidence_sources"])[:12],
                "summary_ids": sorted(entry["summary_ids"]),
                "loop_stop_condition": bool(entry.get("loop_stop_condition")),
                "time_round_token_limit_driven": bool(
                    entry.get("time_round_token_limit_driven")
                ),
            }
            if entry.get("synthesized_from_legacy_analysis"):
                result["synthesized_from_legacy_analysis"] = True
            entries.append(result)
        entries.sort(
            key=lambda item: (
                -int(item.get("count") or 0),
                str(item.get("signature") or ""),
            )
        )
        return entries

    def _failure_mechanism_signature(
        self,
        *,
        failure_category: str,
        status: str,
        timeout_phase: str,
        affected_components: list[str],
        failure_mechanisms: list[str] | None = None,
    ) -> str:
        components = ",".join(sorted(dict.fromkeys(affected_components))) or "none"
        phase = timeout_phase or "none"
        mechanisms = ",".join(sorted(dict.fromkeys(failure_mechanisms or []))) or "none"
        signature = (
            f"category={failure_category}|status={status or 'unknown'}|"
            f"phase={phase}|components={components}"
        )
        if mechanisms != "none":
            signature += f"|mechanisms={mechanisms}"
        return signature

    def _recent_campaign_states(self, *, limit: int | None) -> list[dict[str, Any]]:
        if self._campaign_states_cache is not None:
            states = list(self._campaign_states_cache)
            if limit is None:
                return states
            return states[: max(0, int(limit))]
        return self._load_campaign_states(limit=limit)

    def _load_campaign_states(self, *, limit: int | None) -> list[dict[str, Any]]:
        summaries_dir = self.memory_path / "summaries"
        if not summaries_dir.exists():
            return []
        if limit is not None and int(limit) <= 0:
            return []
        candidates = sorted(
            summaries_dir.glob("*campaign_state.json"),
            key=lambda path: path.stat().st_mtime,
        )
        states: list[dict[str, Any]] = []
        for path in reversed(candidates):
            try:
                data = self._read_json(path)
            except (OSError, ValueError):
                continue
            if data:
                states.append(self._normalized_campaign_state(data))
                if limit is not None and len(states) >= max(0, int(limit)):
                    break
        return states

    def _normalized_campaign_state(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(state)
        events = state.get("codex_update_events")
        if isinstance(events, list):
            normalized["codex_update_events"] = [
                normalize_legacy_limit_driven_skip_event(event)
                for event in events
                if isinstance(event, dict)
            ]
        return normalized

    def _summary_key(self, campaign_id: str, summary_id: str) -> str:
        return f"{campaign_id}:{summary_id}" if campaign_id else summary_id

    def _summary_trial_key(
        self,
        campaign_id: str,
        summary_id: str,
        trial_ids: list[str],
    ) -> str:
        summary_key = self._summary_key(campaign_id, summary_id)
        return f"{summary_key}:{json.dumps(trial_ids, separators=(',', ':'))}"

    def _load_trial_result(self, trial_id: str) -> TrialResult:
        path = self.memory_path / "runs" / trial_id / "result.json"
        return TrialResult.model_validate_json(path.read_text())

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _generalization_contract(self, failures: list[TrialResult]) -> dict[str, Any]:
        domains = sorted({trial.task_domain.value for trial in failures})
        difficulties = sorted({trial.task_difficulty.value for trial in failures})
        return {
            "objective": (
                "Codex updates must improve a reusable Worker/harness failure "
                "class, not hard-code behavior for one TerminalBench task."
            ),
            "failure_tasks_are_evidence_not_targets": [
                trial.task_id for trial in failures
            ],
            "observed_domains": domains,
            "observed_difficulties": difficulties,
            "required_final_report_fields": [
                "generalization.problem_class",
                "generalization.applies_to",
                "generalization.anti_overfit_checks",
                "generalization.why_not_task_specific",
            ],
            "anti_patterns": [
                "branching on a benchmark task id",
                "editing task tests, oracle solutions, verifier, or task definitions",
                "raising official timeout/resource limits",
                "logging-only patches that do not change Worker or harness behavior",
            ],
        }

    def _leaderboard_compliance_contract(self) -> dict[str, Any]:
        return {
            "terminal_bench_version": "terminal-bench@2.0",
            "harbor_is_official_harness": True,
            "custom_agent_entrypoint": (
                "Use Harbor custom agent import path for this repo's self-owned "
                "Worker, for example bench.harbor_adapter:HLWorkerHarborAgent."
            ),
            "official_leaderboard_run_shape": (
                "Leaderboard candidate evaluation must use Harbor with the "
                "terminal-bench@2.0 dataset or the official local 2.0 dataset path, "
                "this repo's custom agent import path, and at least --n-attempts/-k 5 "
                "per task before upload."
            ),
            "must_preserve": [
                "Do not modify official task timeouts or resources.",
                "Do not edit terminal-bench task tests, solutions, verifier, or task definitions.",
                "Do not delegate benchmark execution to Codex, openai-codex, ForgeCode, Claude Code, Factory, Droid, or another external agent.",
                "Do not create nested sub-agents; only the master HL orchestrator may create sub-agents.",
                "Do not let the evaluated Worker access the Terminal-Bench website, Terminal-Bench GitHub repository, or Harbor/Terminal-Bench internals during task solving.",
                "Keep complete Harbor/ATIF trajectory and artifact evidence for every submitted run.",
                "Keep submit opt-in and one-shot per campaign.",
                "Store verifier-grounded trajectory/artifact evidence for review.",
            ],
            "required_final_report_fields": [
                "leaderboard_compliance.harbor_official_harness_preserved",
                "leaderboard_compliance.self_owned_worker_preserved",
                "leaderboard_compliance.benchmark_integrity_preserved",
                "leaderboard_compliance.timeouts_resources_unchanged",
                "leaderboard_compliance.submit_gate_preserved",
                "leaderboard_compliance.official_dataset_preserved",
                "leaderboard_compliance.five_attempts_per_task_preserved",
                "leaderboard_compliance.no_prohibited_terminal_bench_access",
                "leaderboard_compliance.upload_artifacts_trace_preserved",
            ],
        }

    def _sub_agent_creation_policy(self) -> dict[str, Any]:
        policy = sub_agent_creation_policy()
        policy["work_packet_guardrail"] = (
            "This Codex update is itself a master-created sub-agent. It must "
            "not spawn or invoke Codex CLI, OpenAI Codex/openai-codex, Claude, "
            "ForgeCode, Factory Droid, Factory/factory, Droid/droid, Gemini, "
            "OpenCode, Aider, Amp, Cursor Agent, or external coding-agent "
            "process."
        )
        policy["blocked_actions"] = [
            "Do not run codex, codex exec, codex run, openai-codex, claude, claude-code, forgecode, factory-droid, factory, droid, gemini, opencode, aider, amp, cursor-agent, or another coding-agent CLI from this update.",
            "Do not add Worker tools, scripts, prompts, or configs that allow sub-agents to create further sub-agents.",
            "Keep all sub-agent creation owned by the master HL campaign orchestrator.",
        ]
        return policy

    def _report_contract_rules(self) -> dict[str, Any]:
        """Render the report-contract registry into the packet.

        The same final-report rule definitions drive packet guidance, host
        review, and report_lint. Validator bindings and severity therefore
        cannot drift into parallel string-classification lists.
        """

        from meta.report_contract import final_report_rules

        rules = [
            {
                "id": rule.id,
                "severity": rule.severity,
                "description": rule.description,
                "binding": rule.binding,
            }
            for rule in final_report_rules()
        ]
        return {
            "objective": (
                "Final-report validation is registry-driven. 'fatal' violations "
                "reject and roll back the patch; 'report' violations are advisory. "
                "Unknown rule ids fail closed as internal.contract_error."
            ),
            "rules": rules,
            "self_check": (
                "Before delivering, run: python scripts/report_lint.py "
                "--packet-dir <packet-run-dir> --report <draft-report.json>. This "
                "loads packet evidence and the isolated worktree delta, prints the "
                "post-edit valid layer budget, and exits non-zero on fatal findings."
            ),
        }

    def _report_value_budget(
        self,
        *,
        failure_pattern_digest: dict[str, Any],
        mission_debug: dict[str, Any],
        rejected_update_buffer: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Separate pre-edit choices from values computed from the final diff."""

        problem_class_labels = self._failure_pattern_marker_labels(
            failure_pattern_digest
        )
        for candidate in self._mission_candidate_dicts(mission_debug):
            for key in ("id", "failure_category"):
                value = str(candidate.get(key) or "").strip()
                if len(value) >= 3 and value not in problem_class_labels:
                    problem_class_labels.append(value)
        rejected_ids = [
            str(entry.get("packet_id") or entry.get("mission_candidate_id") or "").strip()
            for entry in rejected_update_buffer
            if str(entry.get("packet_id") or entry.get("mission_candidate_id") or "").strip()
        ]
        candidate_ids = [
            str(candidate.get("id") or "").strip()
            for candidate in self._mission_candidate_dicts(mission_debug)
            if str(candidate.get("id") or "").strip()
        ]
        selected_candidate_id = str(
            (mission_debug.get("evidence_summary") or {}).get("selected_candidate_id")
            or ""
        ).strip()
        selected_ids = [selected_candidate_id] if selected_candidate_id else []
        return {
            "pre_edit": {
                "selected_feature_candidate_id": selected_candidate_id,
                "selected_problem_class_labels": problem_class_labels,
                "rejected_update_buffer_ids_to_cover": list(
                    dict.fromkeys(rejected_ids)
                ),
                "rejected_update_buffer_limit": DEFAULT_REJECTED_UPDATE_BUFFER_LIMIT,
            },
            "post_edit": {
                "valid_primary_layers": [],
                "source": (
                    "computed from the isolated diff by scripts/report_lint.py "
                    "before final delivery"
                ),
            },
            # Compatibility aliases remain explicit but no longer pretend the
            # pre-edit packet knows the eventual changed-file layer.
            "valid_primary_layers": [],
            "selected_problem_class_labels": problem_class_labels,
            "rejected_update_buffer_ids_to_cover": list(dict.fromkeys(rejected_ids)),
            "rejected_update_buffer_limit": DEFAULT_REJECTED_UPDATE_BUFFER_LIMIT,
            "mission_feature_candidate_ids": candidate_ids,
            "attributed_feature_candidate_ids": selected_ids,
            "selected_feature_candidate_id": selected_candidate_id,
            "rules": [
                "After editing, run report_lint with --packet-dir and choose primary_layer "
                "only from post_edit.valid_primary_layers in its output.",
                "Set selected_problem_class / dominant_patterns / generalization.problem_class "
                "from selected_problem_class_labels (a concrete mechanism signature is preferred "
                "over a generic status label).",
                "memory_record.failed_directions_to_avoid must cover every packaged id "
                "in rejected_update_buffer_ids_to_cover; the buffer is a deterministic "
                "priority-ordered recent window, not the full historical archive.",
                "Reference the single selected_feature_candidate_id as the slice; an empty "
                "id means the packet is a structured skip and Codex must not run.",
                "Never write a TerminalBench task id literal into production code; name the "
                "mechanism signature or failure class instead.",
            ],
            "task_id_literal_policy": {
                "rule": (
                    "Production code (bench/, harness/, crates/, hl/, meta/, scripts/, config/) "
                    "must never branch on or embed a specific TerminalBench task id literal; "
                    "that is benchmark leakage and is a fatal review violation."
                ),
                "bad_example": (
                    'if task_id == "path-tracing": enable_special_case()  # forbidden: task-id literal'
                ),
                "good_example": (
                    "# Generalize by mechanism/failure class instead of a task name:\n"
                    'if failure_class == "missing_output_artifact_contract": '
                    "enforce_artifact_preflight()"
                ),
            },
        }

    def _failure_pattern_marker_labels(
        self,
        failure_pattern_digest: dict[str, Any],
    ) -> list[str]:
        labels: list[str] = []

        def _add(value: Any) -> None:
            text = str(value or "").strip()
            if len(text) >= 3 and text not in labels:
                labels.append(text)

        dominant = failure_pattern_digest.get("dominant_pattern")
        if isinstance(dominant, dict):
            _add(dominant.get("failure_category"))
            for component in dominant.get("affected_components") or []:
                _add(component)
        for key in ("dominant_mechanism_pattern", "dominant_weakness_signature"):
            entry = failure_pattern_digest.get(key)
            if isinstance(entry, dict):
                _add(entry.get("signature"))
        for pattern in failure_pattern_digest.get("patterns") or []:
            if isinstance(pattern, dict):
                _add(pattern.get("failure_category"))
        for key in ("mechanism_patterns", "weakness_signatures"):
            for pattern in failure_pattern_digest.get(key) or []:
                if isinstance(pattern, dict):
                    _add(pattern.get("signature"))
        return labels

    def _mission_candidate_dicts(
        self,
        mission_debug: dict[str, Any],
    ) -> list[dict[str, Any]]:
        raw = mission_debug.get("feature_candidates")
        if not isinstance(raw, list):
            return []
        return [candidate for candidate in raw if isinstance(candidate, dict)]

    def _heuristic_learning_contract(self) -> dict[str, Any]:
        return {
            "source": "https://trinkle23897.github.io/learning-beyond-gradients/",
            "principles": [
                "Absorb feedback by writing failures, logs, rewards, and update outcomes into trials memory.",
                "Protect old capabilities with regression tests, golden traces, version diffs, and failed-direction records.",
                "Compress accumulated local patches into simpler maintainable representations when coupling grows.",
                "Treat tests, observability, rollback cost, and modular boundaries as part of the learnable system.",
            ],
        }

    def _update_memory_contract(self) -> dict[str, Any]:
        return {
            "required_artifacts": [
                "codex_update_packet.json",
                "codex_events.jsonl",
                "final_message.json",
                "git.diff",
                "review.json",
                "update_record.json",
                "update_summary.md",
            ],
            "final_report_must_include": [
                "memory_record.concise",
                "memory_record.detailed",
                "memory_record.failed_directions_to_avoid",
                "memory_record.supported_directions_to_preserve",
            ],
            "purpose": (
                "Each update must leave both a short handoff and a detailed "
                "pitfall/support record so later updates avoid repeating failed "
                "directions and preserve validated ones."
            ),
        }

    def _framework_comparison_contract(self) -> dict[str, Any]:
        return {
            "before_source": "current_harness plus git diff before Codex executes",
            "after_source": "reviewed Codex delta and required validation commands",
            "final_report_must_include": [
                "framework_comparison.before",
                "framework_comparison.after",
                "framework_comparison.expected_effect",
                "framework_comparison.rollback_trigger",
            ],
            "rollback_rule": (
                "If post-update regression or next comparable evaluation is worse "
                "than the baseline, roll back the saved Codex diff and record the failed direction."
            ),
        }

    def _architecture_update_contract(self) -> dict[str, Any]:
        return {
            "objective": (
                "Codex may and should change Worker/harness architecture, modules, "
                "adapters, validation, configuration, or orchestration when evidence "
                "shows the failure is structural; prompt-only patches are not the default."
            ),
            "learnable_surfaces": [
                "Worker loop control flow",
                "tool APIs and tool descriptions",
                "todo/goal/recovery/context policies",
                "Harbor adapter and artifact parsing",
                "verification and regression gates",
                "configuration schemas and provider validation",
                "observability, memory, compression, and rollback records",
                "prompts, only when the defect is model-facing",
            ],
            "prompt_only_rule": (
                "A prompt-only update is acceptable only when the final report explains "
                "why structural/tool/config changes were considered and rejected as unnecessary."
            ),
            "required_final_report_fields": [
                "changed_files exactly matching the reviewed git diff",
                "component_type matching the primary changed-file layer",
                "implementation_scope.primary_layer",
                "implementation_scope.architectural_change_considered",
                "implementation_scope.structural_files_changed exactly matching structural diff files",
                "implementation_scope.why_prompt_only_is_sufficient",
            ],
            "host_report_gates": [
                "final_report.changed_files must equal the actual reviewed diff file list",
                "implementation_scope.structural_files_changed must neither omit nor overreport structural files",
                "implementation_scope.primary_layer or component_type must match the primary actual changed-file layer",
            ],
        }

    def _official_evaluation_contract(self) -> dict[str, Any]:
        return {
            "source_of_truth": [
                "https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard",
                "https://www.tbench.ai/leaderboard/terminal-bench/2.0",
                "https://www.tbench.ai/news/leaderboard-integrity-update",
                "harbor run --help on the installed CLI",
            ],
            "required_for_leaderboard_candidate": [
                "Use the official Terminal-Bench 2.0 dataset via Harbor.",
                "Run each evaluated task with at least 5 attempts/trials.",
                "Do not modify task timeouts or resources.",
                "Do not edit benchmark tasks, tests, solutions, verifier, or definitions.",
                "Do not let the evaluated Worker access the Terminal-Bench website, Terminal-Bench GitHub repository, or Harbor/Terminal-Bench internals.",
                "Do not fetch public internet solutions or inject them through AGENTS.md or another setup artifact.",
                "Ensure every passing trial has a complete ATIF trajectory; do not upload incomplete traces.",
                "Do not package task tests or official solutions into the agent setup or artifacts.",
                "Preserve complete uploadable Harbor job artifacts and trajectories.",
            ],
        }

    def _cross_round_update_contract(self) -> dict[str, Any]:
        return {
            "objective": (
                "Before editing, compare the trigger-round failures with the recent "
                "campaign_context and failure_pattern_digest. Select one bounded "
                "improvement that addresses a recurring Worker/harness capability "
                "gap, not a task-specific symptom."
            ),
            "packet_fields_to_review": [
                "campaign_context.recent_summaries",
                "campaign_context.recent_completed_trials",
                "campaign_context.recent_analysis_reports.policy_recurrence_signals",
                "policy_recurrence_signals",
                "infrastructure_triage",
                "failure_pattern_digest.patterns",
                "failure_pattern_digest.dominant_pattern",
                "failure_pattern_digest.weakness_signatures",
                "failure_pattern_digest.dominant_weakness_signature",
            ],
            "selection_rules": [
                "Prefer failure categories recurring across multiple tasks or summaries.",
                "When weakness_signatures are available, prefer clusters with the same verifier failure, agent contribution, and reusable mechanism over broader status-only buckets.",
                "If the current trigger failures are outliers, use them as examples but do not optimize only for them.",
                "Before repeating a component layer or failure class, inspect rejected_update_buffer and explain what new evidence changes the decision.",
                "If the last accepted direction was prediction_missed or rolled back, mutate the approach or pivot to a different learnable surface.",
                "If evidence is too mixed for a safe bounded change, return noop/rejected with the missing evidence.",
            ],
            "required_final_report_fields": [
                "cross_round_evidence.used",
                "cross_round_evidence.recent_summary_ids",
                "cross_round_evidence.dominant_patterns",
                "cross_round_evidence.selected_problem_class",
                "cross_round_evidence.why_this_slice_generalizes",
                "prediction.expected_fixed_task_classes",
                "prediction.risk_task_classes",
                "prediction.expected_metric_delta",
                "prediction.confidence",
            ],
        }

    def _update_search_policy(self) -> dict[str, Any]:
        return {
            "inspired_by": [
                "SkillOpt-style skill optimization: propose edits, validate on held-out evidence, keep successful variants, and learn from rejected variants.",
                "Heuristic Learning: absorb update outcomes into memory, then mutate or pivot instead of repeating failed local patches.",
            ],
            "objective": (
                "Make Codex updates converge by treating recent update outcomes as a "
                "search history over Worker/harness policies, not as isolated patches."
            ),
            "candidate_generation_rules": [
                "Start from mission_selection_contract and mission_debug.feature_candidates; choose exactly one mission candidate or explain why all candidates are unsafe or stale.",
                "When mission_debug.feature_candidates include mission-attributed-* entries, prefer their failure_category, affected_components, allowed_edit_paths, and validation_contracts over broad status-only buckets.",
                "Generate one primary bounded candidate and at least one rejected alternative in the final rationale.",
                "Prefer executable policy changes with measurable prediction targets over observability-only patches unless evidence quality is the explicit blocker.",
                "Read change_evaluation_digest before choosing a candidate; preserve hit-backed classes and mutate or avoid miss-heavy classes.",
                "When policy_recurrence_signals is non-empty, treat it as evidence that an existing recognizer or policy is too late, too weak, or insufficiently enforced before adding a duplicate rule.",
                "When infrastructure_triage.trigger_all_infrastructure is true, do not turn the update into a Worker prompt/reasoning patch unless a separate non-infrastructure category proves a Worker-owned failure; prefer Harbor/environment attribution, retry evidence, or packet-routing changes.",
                "Read prior_update_lesson_entries before choosing a candidate; if they name failed packet_id, outcome, or mission_candidate_id, mutate that direction or explain the fresh evidence that justifies revisiting it.",
                "If a prior candidate missed its prediction, do not submit a cosmetic variant of the same direction.",
                "If runner_pivot_policy.layer_pressure is non-empty, avoid another same-layer patch unless fresh evidence explains why the new surface is different; memory_record.failed_directions_to_avoid must name the pressured layer plus a recent packet_id or failure_class when available.",
                "If runner_pivot_policy.supported is non-empty, consider whether the bounded candidate should preserve or extend that supported direction before pivoting elsewhere.",
            ],
            "validation_rules": [
                "Tie prediction.expected_fixed_task_classes to concrete labels in failure_pattern_digest, change_evaluation_digest, rejected_update_buffer, or prior_update_lesson_entries.",
                "Use required validation commands plus the dynamic validation ladder; do not weaken regression or submit gates.",
                "Define a falsification window that can be evaluated by the next comparable summary or same-model frontier update.",
                "For edited patches, loophole_review and loophole_fixes must each contain at least one concrete risk and mitigation; strategy_confidence must be high, medium, or low.",
            ],
            "rejected_buffer_rules": [
                "Read rejected_update_buffer before editing.",
                "When rejected_update_buffer is non-empty, memory_record.failed_directions_to_avoid must cover every entry in the packaged buffer by naming its packet_id, failure_class, or component_layer.",
                "When a rejected_update_buffer entry includes loophole_review or loophole_fixes, memory_record.failed_directions_to_avoid must name at least one prior reviewed risk or mitigation for that entry.",
                "When a rejected_update_buffer entry includes required_mutation, memory_record.failed_directions_to_avoid must reference that mutation guidance, not just the packet_id or layer.",
                "When runner_pivot_policy.discouraged is non-empty, memory_record.failed_directions_to_avoid must cover each discouraged failure_class or component_layer.",
                "When a discouraged entry includes mission_candidate_id, memory_record.failed_directions_to_avoid must name that exact candidate id rather than only its failure category or layer.",
                "When prior_update_lesson_entries include failed packet_id, outcome, or mission_candidate_id markers, memory_record.failed_directions_to_avoid must reference those markers.",
                "Avoid repeated failure_class/component_layer pairs unless the packet contains fresh trajectory or verifier evidence.",
                "When repeating a direction, state the mutation relative to the failed attempt in memory_record.failed_directions_to_avoid.",
            ],
            "supported_direction_rules": [
                "Read runner_pivot_policy.supported before editing.",
                "When runner_pivot_policy.supported is non-empty, memory_record.supported_directions_to_preserve must reference the supported packet_id, failure_class, or component_layer and explain preservation, extension, or evidence-backed pivot.",
                "When a supported entry includes mission_candidate_id, memory_record.supported_directions_to_preserve must name that exact candidate id rather than only its failure category or layer.",
                "Preserve or extend supported directions only when the new candidate keeps the same verifier/frontier evidence discipline.",
                "Do not use a supported direction to bypass rejected_update_buffer, regression gates, or prediction falsification requirements.",
            ],
            "exploration_rules": [
                "When update_history.research_recommended is true, use local reference contracts or web references as design evidence.",
                "Treat external references as patterns to adapt into this repo's Worker/harness interfaces, not code to copy.",
                "When external_research.used is true, external_research.sources must come from external_research_policy.web_sources or local_read_only_refs.",
                "When external_research.used is true, external_research.impact must explain the concrete local harness or updater decision changed by the reference and reference a marker from external_research_policy.research_focus_areas.",
                "Before using or dismissing a web source, apply any matching external_research_policy.fetch_requirements; mp.weixin.qq.com sources require the MicroMessenger required_user_agent to avoid WeChat environment verification pages.",
                "Consider non-local surfaces such as candidate selection, rollback policy, validation contracts, and memory compression when Worker/tool patches plateau.",
            ],
        }

    def _mission_selection_contract(self) -> dict[str, Any]:
        return {
            "objective": (
                "Use mission_debug as the bounded feature-selection surface before "
                "editing Worker/harness policy. Mission candidates are external-loop "
                "choices, not permission to solve benchmark tasks directly."
            ),
            "packet_fields_to_review": [
                "mission_debug.feature_candidates",
                "mission_debug.validation_contracts",
                "mission_debug.evidence_summary",
                "failure_pattern_digest.patterns",
                "runner_pivot_policy",
                "change_evaluation_digest",
            ],
            "selection_rules": [
                "Choose exactly one mission_debug.feature_candidates entry as the primary slice before editing.",
                "Prefer mission-attributed-* candidates over broad status-only candidates when both are present, because they carry task-level failure_category and affected_components evidence.",
                "Keep edits within the selected candidate's allowed_edit_paths plus tests; if another path is necessary, return noop/rejected unless the packet evidence explicitly justifies the expansion.",
                "Run or cite the selected candidate's validation_contracts through required_validation_commands or skipped_validation_reason.",
                "If no mission candidate is safe, report noop/rejected and name the missing evidence instead of making a speculative patch.",
            ],
            "final_report_expectations": [
                "cross_round_evidence.selected_problem_class should name the selected mission candidate id or failure_category.",
                "prediction.expected_fixed_task_classes should include the selected mission candidate's failure_category when available.",
                "implementation_scope.structural_files_changed must stay within the selected candidate's allowed_edit_paths unless explicitly justified.",
                "memory_record.detailed should mention how the selected mission candidate was preserved, extended, or rejected.",
            ],
        }

    def _self_iteration_contract(self) -> dict[str, Any]:
        return {
            "inspired_by": (
                "DeliAutoResearch continual-learning/self-iteration discussion: "
                "post-deployment systems should update from feedback without "
                "forgetting stable capabilities."
            ),
            "objective": (
                "Make each Codex update a grounded self-iteration step: absorb "
                "new campaign evidence, preserve known-good behavior, and only "
                "parameterize lessons when repeated evidence supports them."
            ),
            "update_axes": {
                "what": [
                    "worker skill",
                    "tool/recovery policy",
                    "context or memory policy",
                    "verification/runner policy",
                ],
                "how": [
                    "bounded code patch",
                    "validation ladder",
                    "same-model frontier comparison",
                    "rejected-direction mutation or pivot",
                ],
                "when": [
                    "event-triggered by verified failures",
                    "periodic Codex update interval",
                    "rollback-triggered cooldown",
                    "memory compression after repeated patterns",
                ],
            },
            "grounding_signals": [
                "Harbor/verifier score and status, never Worker self-report.",
                "Host-run required validation commands and dynamic validation ladder.",
                "Same-model solved-task frontier and post-update regression snapshots.",
                "Rejected update buffer, rollback records, and falsified predictions.",
            ],
            "memory_rules": [
                "Keep one-off trajectory details as episodic trial memory.",
                "Promote only repeated failure patterns or validated fixes into long-term policy.",
                "Use regression/replay evidence before changing behavior that can affect previously solved task classes.",
                "Record failed directions explicitly so later updates mutate or pivot rather than repeat cosmetic variants.",
            ],
            "convergence_rules": [
                "Prefer independent validation signals over self-critique when deciding whether an update improved the harness.",
                "Balance stability and plasticity: broaden capability only when regression gates protect existing passes.",
                "If grounding is weak or mixed, return noop/rejected and request the missing evidence instead of shipping a speculative patch.",
            ],
        }

    def _same_model_frontier(self, failures: list[TrialResult]) -> dict[str, Any]:
        model_scope = ""
        for trial in failures:
            model_scope = model_scope_from_trial(trial)
            if model_scope:
                break
        if not model_scope:
            return {
                "available": False,
                "reason": "trigger failures do not include model scope metadata",
            }

        summaries_dir = self.memory_path / "summaries"
        if not summaries_dir.exists():
            return {
                "available": False,
                "model_scope": model_scope,
                "reason": "no summaries directory exists yet",
            }
        suffix = f"_frontier_{safe_model_scope_name(model_scope)}.json"
        candidates = sorted(
            summaries_dir.glob(f"*{suffix}"),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            return {
                "available": False,
                "model_scope": model_scope,
                "reason": "no same-model frontier file exists yet",
            }
        frontier_path = candidates[-1]
        try:
            data = self._read_json(frontier_path)
        except (OSError, ValueError):
            return {
                "available": False,
                "model_scope": model_scope,
                "path": str(frontier_path),
                "reason": "frontier file could not be parsed",
            }
        summary = frontier_summary(data)
        summary.update(
            {
                "available": True,
                "path": str(frontier_path),
                "purpose": (
                    "Same-model per-task frontier for comparing rotating-task "
                    "updates without relying only on aggregate score."
                ),
            }
        )
        return summary

    def _harness_reference_contract(self) -> dict[str, Any]:
        refs = [
            self._reference_source(
                name="Agentic Harness Engineering",
                url="https://github.com/china-qijizhifeng/agentic-Harness-engineering",
                local_path="/tmp/harness-evolver-refs/agentic-harness-engineering",
                practices=[
                    "Hold the base model fixed while evolving harness components.",
                    "Treat traces and sourced analysis, not pass rate alone, as the update unit.",
                    "Require each edit to state failure evidence, root cause, targeted fix, and predicted impact.",
                    "Use next-iteration task flips to falsify predictions and decide keep/improve/rollback.",
                ],
                local_surfaces=[
                    "meta/packager.py",
                    "meta/codex_update.py",
                    "hl/loop.py",
                    "trials/diffs",
                    "harness/context",
                ],
            ),
            self._reference_source(
                name="Meta-Harness",
                url="https://github.com/stanford-iris-lab/meta-harness",
                local_path="/tmp/harness-evolver-refs/meta-harness",
                practices=[
                    "Compare candidates against same-model baselines and a per-task frontier.",
                    "Use a cheap bring-up ladder before expensive full runs: smoke, hard subset, then full evaluation.",
                    "Parse trial-level reward, token, cache, cost, turn, and API-call metrics from job artifacts.",
                    "Treat high concurrency API throughput failures as infrastructure signals before blaming reasoning.",
                ],
                local_surfaces=[
                    "scripts/run_campaign.py",
                    "bench/harbor.py",
                    "bench/scoring.py",
                    "hl/memory.py",
                    "trials/summaries",
                ],
            ),
            self._reference_source(
                name="TACO",
                url="https://github.com/multimodal-art-projection/TACO",
                local_path="/tmp/harness-evolver-refs/TACO",
                practices=[
                    "Compress terminal observations with reusable rules instead of blind truncation.",
                    "Flag long uncovered outputs as candidates for new compression rules.",
                    "Keep a reusable rule pool, with freeze/local-only modes for ablations and reproducibility.",
                    "Preserve trajectory structure even when content is shortened.",
                ],
                local_surfaces=[
                    "harness/context/compaction.py",
                    "harness/context/trajectory_pack.py",
                    "bench/trajectory.py",
                    "tests/test_goal_submit_compression.py",
                ],
            ),
            self._reference_source(
                name="OpenClacky",
                url="https://github.com/clacky-ai/openclacky",
                local_path="/tmp/harness-evolver-refs/openclacky",
                practices=[
                    "Keep the system prompt and tool schema stable; route extensibility through a small stable surface.",
                    "Use cache-aware insert-then-compress and guard against duplicate system prompt rebuilds.",
                    "Escalate context overflow recovery in layers instead of retrying the same compression.",
                    "Prefer direct repo search/read over stale vector indexes for local coding context.",
                ],
                local_surfaces=[
                    "harness/prompts",
                    "harness/tools/registry.py",
                    "harness/skill_loading",
                    "harness/context/compaction.py",
                    "harness/recovery",
                ],
            ),
            {
                "name": "SkillOpt",
                "url": "https://github.com/microsoft/SkillOpt",
                "site": "https://microsoft.github.io/SkillOpt/",
                "local_path": "",
                "local_reference_status": {
                    "path": "",
                    "exists": False,
                    "fallback": (
                        "Use the GitHub repository and project site as read-only "
                        "design references."
                    ),
                },
                "practices": [
                    "Treat skills or policies as text artifacts optimized by proposal, validation, and feedback.",
                    "Keep a rejected-variant buffer so future candidates avoid previously failed directions.",
                    "Use iterative mutation and selection rather than one-shot prompt edits.",
                    "Evaluate candidates against held-out or cross-task evidence before keeping them.",
                ],
                "local_surfaces": [
                    "meta/packager.py",
                    "meta/codex_update.py",
                    "hl/frontier.py",
                    "trials/diffs",
                    "trials/summaries",
                ],
            },
            {
                "name": "Self-Harness article",
                "url": "https://mp.weixin.qq.com/s/sgP8m1nnW7JhsDT7Ki7nVw",
                "paper": "https://arxiv.org/abs/2606.09498",
                "local_path": "",
                "local_reference_status": {
                    "path": "",
                    "exists": False,
                    "fallback": (
                        "Fetch the WeChat source with a MicroMessenger mobile "
                        "User-Agent, then use the linked paper as read-only "
                        "design evidence."
                    ),
                },
                "practices": [
                    "Mine weaknesses from real verifier-grounded trajectories before proposing harness edits.",
                    "Cluster failures by exact mechanism signature so broad timeout buckets do not merge different root causes.",
                    "Generate bounded candidate harness changes against declared edit surfaces instead of rewriting the whole loop.",
                    "Validate proposals against same-model held-in/held-out, frontier, or regression evidence before promotion.",
                    "Keep rejected proposals as future negative evidence rather than mutating the active harness.",
                ],
                "local_surfaces": [
                    "meta/packager.py",
                    "meta/codex_update.py",
                    "scripts/run_campaign.py",
                    "hl/frontier.py",
                    "trials/analysis",
                    "trials/diffs",
                ],
            },
            {
                "name": "Claude Code large-codebase practices",
                "url": (
                    "https://claude.com/blog/how-claude-code-works-in-large-codebases-"
                    "best-practices-and-where-to-start"
                ),
                "local_path": "",
                "local_reference_status": {
                    "path": "",
                    "exists": False,
                    "fallback": "Use the official blog URL.",
                },
                "practices": [
                    "Maintain lightweight repo context such as AGENTS.md or CLAUDE.md and keep it reviewed.",
                    "Use progressive-disclosure skills and focused sub-contexts for specialized workflows.",
                    "Use deterministic hooks/checks for formatting, tests, and policy enforcement.",
                    "Review and prune agent configuration periodically so accumulated guidance stays accurate.",
                ],
                "local_surfaces": [
                    "AGENTS.md",
                    "docs/architecture.md",
                    "harness/skill_loading",
                    "meta/reviewer.py",
                    "scripts/run_campaign.py",
                ],
            },
        ]
        return {
            "objective": (
                "Use external harness projects as design evidence for reusable "
                "Worker/harness capability improvements, not as code to copy or as "
                "a way to delegate Terminal-Bench solving to another agent."
            ),
            "sources": refs,
            "transfer_rules": [
                "Translate a reference pattern into this repo's existing Worker/harness interfaces.",
                "Tie every adopted pattern to local failure evidence or cross-round inefficiency evidence.",
                "Prefer observability, validation, recovery, context, or architecture changes when the failure is structural.",
                "Keep same-model before/after comparison; do not compare a flash update against pro history.",
                "Do not copy reference source into this repo unless a small adapted interface is justified and tested.",
                "Do not add benchmark-task-specific logic, hidden task data, solution artifacts, or internet solution retrieval.",
            ],
            "anti_patterns": [
                "Adding links to prompts without changing an executable policy or validation contract.",
                "Copying a reference agent or framework into this repository instead of adapting a small local interface.",
                "Replacing the self-owned Worker with a reference agent.",
                "Growing tool schemas or prompts without a cache/context rationale.",
                "Treating high concurrency provider timeouts as task-solving failures without attribution.",
                "Compressing trajectories so aggressively that verifier, tool-call, and timing evidence is lost.",
            ],
            "expected_codex_behavior": [
                "If external research is used, cite concrete sources in external_research.sources.",
                "State which practice was adopted or rejected in the summary, implementation_scope, or memory_record.",
                "If no external source is used, explain why local packet evidence was sufficient.",
            ],
        }

    def _reference_source(
        self,
        *,
        name: str,
        url: str,
        local_path: str,
        practices: list[str],
        local_surfaces: list[str],
    ) -> dict[str, Any]:
        return {
            "name": name,
            "url": url,
            "local_path": local_path,
            "local_reference_status": self._local_reference_status(local_path),
            "practices": practices,
            "local_surfaces": local_surfaces,
        }

    def _update_history(self, limit: int | None = None) -> dict[str, Any]:
        diffs_dir = self.memory_path / "diffs"
        entries: list[dict[str, Any]] = []
        states = self._recent_campaign_states(limit=None)
        update_status_by_packet = self._update_history_status_by_packet(states)
        if diffs_dir.exists():
            review_paths = sorted(
                diffs_dir.glob("codex_packet_*/review.json"),
                key=lambda path: path.stat().st_mtime,
            )
            if limit is not None:
                limit_value = max(0, int(limit))
                review_paths = review_paths[-limit_value:] if limit_value else []
            for review_path in review_paths:
                try:
                    review = self._read_json(review_path)
                except (OSError, ValueError):
                    continue
                update_record = self._codex_update_record(review_path.parent)
                reasons = list(review.get("reasons") or [])[:5]
                changed_files = list(review.get("changed_files") or [])[:10]
                entry = {
                    "packet_id": review_path.parent.name,
                    "accepted": bool(review.get("accepted")),
                    "reasons": reasons,
                    "changed_files": changed_files,
                    "strategy_confidence": str(
                        update_record.get("strategy_confidence") or ""
                    ),
                    "loophole_review": [
                        str(item)
                        for item in update_record.get("loophole_review") or []
                    ][:3],
                    "loophole_fixes": [
                        str(item)
                        for item in update_record.get("loophole_fixes") or []
                    ][:3],
                    "external_research": self._external_research_record(
                        update_record,
                    ),
                    "validation_failed": self._validation_failed(review_path.parent),
                    "rolled_back": bool(
                        update_status_by_packet.get(review_path.parent.name, {}).get(
                            "rolled_back"
                        )
                    ),
                    "score_declined": bool(
                        update_status_by_packet.get(review_path.parent.name, {}).get(
                            "score_declined"
                        )
                    ),
                    "evaluation_outcome": str(
                        update_status_by_packet.get(review_path.parent.name, {}).get(
                            "evaluation_outcome",
                            "",
                        )
                    ),
                }
                if self._rejected_event_guidance(reasons, changed_files).get(
                    "superseded_by_current_reviewer"
                ):
                    entry["superseded_by_current_reviewer"] = True
                entries.append(entry)
        unsuccessful = sum(
            1
            for entry in entries
            if (
                (
                    not entry["accepted"]
                    and not entry.get("superseded_by_current_reviewer")
                )
                or entry.get("validation_failed")
                or entry.get("rolled_back")
                or entry.get("score_declined")
                or entry.get("evaluation_outcome") in {"prediction_missed", "mixed"}
            )
        )
        return {
            "recent_codex_updates": entries,
            "recent_unsuccessful_updates": unsuccessful,
            "recent_codex_updates_window_audit_only": limit,
            "recent_codex_updates_window_stop_condition": False,
            "update_history_count_stop_condition": False,
            "poor_update_threshold_for_research": 2,
            "research_recommended": unsuccessful >= 2,
        }

    def _codex_update_record(self, run_dir: Path) -> dict[str, Any]:
        record_path = run_dir / "update_record.json"
        if not record_path.is_file():
            return {}
        try:
            record = self._read_json(record_path)
        except (OSError, ValueError):
            return {}
        return record if isinstance(record, dict) else {}

    def _external_research_record(self, update_record: dict[str, Any]) -> dict[str, Any]:
        research = update_record.get("external_research")
        if not isinstance(research, dict):
            return {}
        used = research.get("used")
        if not isinstance(used, bool):
            used = False
        return {
            "used": used,
            "sources": [str(item) for item in research.get("sources") or []][:5],
            "fetches": self._external_research_fetch_records(research),
            "reason": str(research.get("reason") or ""),
            "impact": str(research.get("impact") or ""),
        }

    def _external_research_fetch_records(
        self,
        research: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        fetches = research.get("fetches")
        if not isinstance(fetches, list):
            return records
        for fetch in fetches[:5]:
            if not isinstance(fetch, dict):
                continue
            headers = fetch.get("headers")
            if not isinstance(headers, dict):
                headers = {}
            records.append(
                {
                    "source": str(fetch.get("source") or ""),
                    "headers": {
                        str(key): str(value) for key, value in headers.items()
                    },
                    "result": str(fetch.get("result") or ""),
                }
            )
        return records

    def _loophole_record_fields(self, packet_id: str) -> dict[str, Any]:
        if not packet_id:
            return {}
        record = self._codex_update_record(self.memory_path / "diffs" / packet_id)
        if not record:
            return {}
        fields: dict[str, Any] = {}
        strategy_confidence = str(record.get("strategy_confidence") or "")
        if strategy_confidence:
            fields["strategy_confidence"] = strategy_confidence
        loophole_review = [
            str(item) for item in record.get("loophole_review") or []
        ][:3]
        if loophole_review:
            fields["loophole_review"] = loophole_review
        loophole_fixes = [str(item) for item in record.get("loophole_fixes") or []][:3]
        if loophole_fixes:
            fields["loophole_fixes"] = loophole_fixes
        return fields

    def _change_evaluation_digest(self, limit: int | None = None) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        hit_classes: dict[str, int] = {}
        miss_classes: dict[str, int] = {}
        risk_classes: dict[str, int] = {}
        states = self._recent_campaign_states(limit=None)
        events_by_packet_id = self._codex_update_events_by_packet(states)
        for state in states:
            for evaluation in reversed(state.get("change_evaluations") or []):
                if not isinstance(evaluation, dict):
                    continue
                packet_id = str(evaluation.get("packet_id") or "")
                if not packet_id:
                    continue
                event = self._codex_update_event_for_packet(
                    state,
                    packet_id,
                    events_by_packet_id=events_by_packet_id,
                )
                prediction = evaluation.get("prediction")
                if not isinstance(prediction, dict):
                    prediction = {}
                hits = self._prediction_task_events(evaluation.get("prediction_hits"))
                misses = self._prediction_task_events(evaluation.get("prediction_misses"))
                expected_classes = [
                    str(item)
                    for item in prediction.get("expected_fixed_task_classes") or []
                    if str(item).strip()
                ][:8]
                risks = [
                    str(item)
                    for item in prediction.get("risk_task_classes") or []
                    if str(item).strip()
                ][:8]
                mission_candidate_id = str(
                    evaluation.get("mission_candidate_id")
                    or event.get("mission_candidate_id")
                    or ""
                )
                mission_failure_category = str(
                    evaluation.get("mission_failure_category")
                    or event.get("mission_failure_category")
                    or ""
                )
                mission_markers = [
                    marker
                    for marker in (mission_candidate_id, mission_failure_category)
                    if marker
                ]
                outcome = str(evaluation.get("outcome") or "")
                outcome_expected_hits = list(
                    expected_classes
                    if self._evaluation_outcome_supported(outcome)
                    else []
                )
                outcome_expected_misses = list(
                    expected_classes
                    if self._evaluation_outcome_missed(outcome)
                    else []
                )
                if self._evaluation_outcome_supported(outcome):
                    outcome_expected_hits.extend(mission_markers)
                if self._evaluation_outcome_missed(outcome):
                    outcome_expected_misses.extend(mission_markers)
                for class_name in self._matched_event_classes(hits) + outcome_expected_hits:
                    hit_classes[class_name] = hit_classes.get(class_name, 0) + 1
                for class_name in (
                    self._matched_event_classes(misses) + outcome_expected_misses
                ):
                    miss_classes[class_name] = miss_classes.get(class_name, 0) + 1
                for class_name in risks:
                    risk_classes[class_name] = risk_classes.get(class_name, 0) + 1
                entries.append(
                    {
                        "packet_id": packet_id,
                        "summary_id": str(evaluation.get("summary_id") or ""),
                        "iteration": event.get("iteration"),
                        "outcome": outcome,
                        "failure_class": str(event.get("failure_class") or ""),
                        "component_layer": str(event.get("component_layer") or ""),
                        "mission_candidate_id": mission_candidate_id,
                        "mission_failure_category": mission_failure_category,
                        "hit_count": int(evaluation.get("hit_count") or 0),
                        "miss_count": int(evaluation.get("miss_count") or 0),
                        "prediction_hits": hits[:4],
                        "prediction_misses": misses[:4],
                        "expected_fixed_task_classes": expected_classes,
                        "risk_task_classes": risks,
                    }
                )
        return {
            "recent_evaluations": entries,
            "recent_evaluations_window_audit_only": limit,
            "recent_evaluations_window_stop_condition": False,
            "change_evaluation_count_stop_condition": False,
            "hit_classes": self._ranked_counts(hit_classes),
            "miss_classes": self._ranked_counts(miss_classes),
            "risk_classes": self._ranked_counts(risk_classes),
            "selection_guidance": [
                "Use miss_classes to avoid cosmetic repeats of evaluated poor directions.",
                "Use hit_classes only when the new patch preserves the same verifier/frontier evidence discipline.",
                "Use risk_classes to define explicit validation and rollback checks before accepting a related patch.",
            ],
        }

    def _matched_event_classes(self, events: list[dict[str, Any]]) -> list[str]:
        classes: list[str] = []
        for event in events:
            for class_name in event.get("matched_classes") or []:
                class_name = str(class_name).strip()
                if class_name:
                    classes.append(class_name)
        return classes

    def _evaluation_outcome_supported(self, outcome: str) -> bool:
        return outcome in {"prediction_supported", "prediction_hit"}

    def _evaluation_outcome_missed(self, outcome: str) -> bool:
        return outcome in {"prediction_missed", "mixed"}

    def _ranked_counts(
        self,
        counts: dict[str, int],
        *,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        return [
            {"class": class_name, "count": count}
            for class_name, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:limit]
        ]

    def _rejected_update_buffer(self, limit: int | None = None) -> list[dict[str, Any]]:
        buffer: list[dict[str, Any]] = []
        seen: set[str] = set()
        states = self._recent_campaign_states(limit=None)
        events_by_packet_id = self._codex_update_events_by_packet(states)
        packet_ids_by_summary = self._codex_update_packet_ids_by_summary(states)
        for state in states:
            evaluations = state.get("change_evaluations")
            if isinstance(evaluations, list):
                for evaluation in reversed(evaluations):
                    if not isinstance(evaluation, dict):
                        continue
                    if not (
                        evaluation.get("rollback_recommended")
                        or evaluation.get("rollback_applied")
                        or evaluation.get("outcome") == "prediction_missed"
                        or evaluation.get("outcome") == "mixed"
                    ):
                        continue
                    packet_id = str(evaluation.get("packet_id") or "")
                    key = f"evaluation:{packet_id}"
                    if key in seen:
                        continue
                    seen.add(key)
                    buffer.append(
                        self._rejected_evaluation_entry(
                            evaluation,
                            state,
                            events_by_packet_id=events_by_packet_id,
                        )
                    )
            validation_failures = state.get("codex_validation_failures")
            if isinstance(validation_failures, list):
                for failure in reversed(validation_failures):
                    if not isinstance(failure, dict):
                        continue
                    if not failure.get("rolled_back"):
                        continue
                    packet_id = self._validation_failure_packet_id(
                        failure,
                        state,
                        packet_ids_by_summary=packet_ids_by_summary,
                    )
                    if not packet_id:
                        continue
                    key = f"validation_failure:{packet_id}"
                    if key in seen:
                        continue
                    seen.add(key)
                    buffer.append(
                        self._rejected_validation_failure_entry(
                            failure,
                            state,
                            packet_id,
                            events_by_packet_id=events_by_packet_id,
                        )
                    )
            frontier_events = state.get("frontier_regression_events")
            if isinstance(frontier_events, list):
                for event in reversed(frontier_events):
                    if not isinstance(event, dict):
                        continue
                    packet_id = str(event.get("packet_id") or "")
                    if not packet_id:
                        continue
                    key = f"frontier_regression:{packet_id}"
                    if key in seen:
                        continue
                    seen.add(key)
                    buffer.append(
                        self._rejected_frontier_regression_entry(
                            event,
                            state,
                            events_by_packet_id=events_by_packet_id,
                        )
                    )
            for event in reversed(state.get("codex_update_events") or []):
                if not isinstance(event, dict):
                    continue
                if event.get("action") != "rejected":
                    continue
                packet_id = str(event.get("packet_id") or "")
                key = f"event:{packet_id}"
                if key in seen:
                    continue
                seen.add(key)
                reasons = [str(item) for item in event.get("reasons") or []][:6]
                changed_files = [
                    str(path) for path in event.get("changed_files") or []
                ][:10]
                buffer.append(
                    {
                        "source": "codex_update_event",
                        "packet_id": packet_id,
                        "summary_id": str(event.get("summary_id") or ""),
                        "iteration": event.get("iteration"),
                        "changed_files": changed_files,
                        "failure_class": str(event.get("failure_class") or ""),
                        "component_layer": str(event.get("component_layer") or ""),
                        "outcome": "rejected",
                        "reasons": reasons,
                        **self._rejected_event_guidance(reasons, changed_files),
                        **self._loophole_record_fields(packet_id),
                    }
                )
        review_limit: int | None = None
        if limit is not None:
            buffered_packet_ids = {
                str(entry.get("packet_id") or "")
                for entry in buffer
                if str(entry.get("packet_id") or "")
            }
            review_limit = max(0, int(limit) - len(buffered_packet_ids))
        self._append_review_based_rejections(
            buffer,
            seen,
            limit=review_limit,
        )
        sorted_buffer = sorted(
            buffer,
            key=self._rejected_update_entry_sort_key,
        )
        unique_buffer: list[dict[str, Any]] = []
        seen_packet_ids: set[str] = set()
        for entry in sorted_buffer:
            packet_id = str(entry.get("packet_id") or "")
            if packet_id:
                if packet_id in seen_packet_ids:
                    continue
                seen_packet_ids.add(packet_id)
            unique_buffer.append(entry)
        if limit is None:
            return unique_buffer
        return unique_buffer[: max(0, int(limit))]

    def _rejected_update_entry_sort_key(
        self,
        entry: dict[str, Any],
    ) -> tuple[int, int, str, str]:
        source_priority = {
            "frontier_regression": 0,
            "codex_validation_failure": 1,
            "change_evaluation": 2,
            "codex_update_event": 3,
            "review": 4,
        }.get(str(entry.get("source") or ""), 5)
        return (
            source_priority,
            -int(entry.get("iteration") or 0),
            str(entry.get("packet_id") or ""),
            str(entry.get("summary_id") or ""),
        )

    def _rejected_event_guidance(
        self,
        reasons: list[str],
        changed_files: list[str],
    ) -> dict[str, Any]:
        reason_text = "\n".join(reasons).lower()
        if "no files changed" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a report-only or no-diff update. The next candidate "
                    "must either make a bounded tracked Worker/harness change tied to "
                    "packet evidence, or return noop/rejected with the missing evidence "
                    "instead of claiming an edit."
                ),
            }
        if "baseline worktree has uncommitted changes" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a Codex update attempt from a dirty baseline. "
                    "Before the next real update, make git status clean by committing, "
                    "stashing, or removing unrelated local changes; if dirty-baseline "
                    "is intentionally allowed, rerun with explicit allow_dirty_baseline "
                    "and keep the baseline-delta evidence separate from the Codex patch."
                ),
            }
        if "required validation commands missing" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a patch without the packet's required validation. "
                    "The next candidate must include the required command, a host-run "
                    "equivalent, or a concrete skipped_validation_reason accepted by the gate."
                ),
            }
        if (
            "validation commands were skipped without explanation" in reason_text
            or "codex final report skipped_validation_reason must be a string" in reason_text
        ):
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an edited report that leaves validation ambiguous. "
                    "The next candidate must list validation_commands, cite a host-run "
                    "equivalent for the packet's required commands, or provide a concrete "
                    "string skipped_validation_reason that explains why validation could not run."
                ),
            }
        if (
            "strategy_confidence must be high, medium, or low" in reason_text
            or "loophole_review must list at least one reviewed risk" in reason_text
            or "loophole_fixes must list at least one mitigation" in reason_text
        ):
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an edited report without explicit loophole review. "
                    "The next candidate must set strategy_confidence to high, medium, "
                    "or low, list at least one concrete regression/counterexample risk "
                    "in loophole_review, and list the matching mitigation in loophole_fixes."
                ),
            }
        if (
            "exactly one mission_debug.feature_candidates" in reason_text
            or "matched multiple candidates" in reason_text
        ):
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an ambiguous mission selection. The next candidate "
                    "must explicitly choose one mission_debug.feature_candidates id, "
                    "keep prediction and memory_record tied to that single candidate, "
                    "or return noop/rejected if no candidate is safe."
                ),
            }
        if (
            "mp.weixin.qq.com" in reason_text
            and "required_user_agent" in reason_text
        ):
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a WeChat article fetch with a generic or partial "
                    "User-Agent. The next external_research.fetches entry for "
                    "mp.weixin.qq.com must use the packet's exact required_user_agent "
                    "before using, dismissing, or citing the source."
                ),
            }
        if (
            "external_research.fetches must record fetch_requirements" in reason_text
            or "external_research.fetches headers must include required" in reason_text
        ):
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an external research report that cites a constrained "
                    "source without recording the required fetch. The next candidate must "
                    "add an external_research.fetches entry for each matching source, "
                    "including the required headers from external_research_policy.fetch_requirements, "
                    "or set external_research.used=false with a concrete skip reason."
                ),
            }
        if "external_research.sources must come from packet external_research_policy" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an external research citation from outside the packet policy. "
                    "The next candidate must either cite only exact sources from "
                    "external_research_policy.web_sources or local_read_only_refs, or set "
                    "external_research.used=false with a concrete skip reason."
                ),
            }
        if "external_research.sources required when research was used" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an external research report that marks research "
                    "used without naming the cited sources. The next candidate must "
                    "either set external_research.used=false with a concrete skip "
                    "reason, or list the exact allowed external_research.sources "
                    "from external_research_policy that informed the patch."
                ),
            }
        if "external_research.impact required when research was used" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an external research report that cites sources "
                    "without stating the local impact. The next candidate must "
                    "either set external_research.used=false with a concrete skip "
                    "reason, or fill external_research.impact with the specific "
                    "Worker/harness or updater decision changed by the cited source."
                ),
            }
        if "external_research.impact must reference a packet research_focus_area" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an external research impact claim that is detached "
                    "from the packet focus areas. The next candidate must name a concrete "
                    "external_research_policy.research_focus_areas marker in external_research.impact "
                    "and explain the local Worker/harness or updater decision it changed."
                ),
            }
        if "external research was recommended after poor updates but no skip reason was reported" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not ignore recommended external research after unsuccessful "
                    "updates. The next candidate must either use an allowed source "
                    "from external_research_policy and report its impact, or set "
                    "external_research.used=false with a concrete reason why local "
                    "evidence was sufficient."
                ),
            }
        if "changed files exceed selected mission candidate allowed_edit_paths" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a mission-selected patch that edits outside the "
                    "selected candidate's allowed_edit_paths. The next candidate must "
                    "either keep structural changes inside that selected mission scope "
                    "plus tests, choose a different mission candidate whose allowed paths "
                    "cover the edit, or return noop/rejected with the missing evidence."
                ),
            }
        if "implementation_scope.primary_layer or component_type must match" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a report that misclassifies the changed-file layer. "
                    "The next candidate must set implementation_scope.primary_layer "
                    "or component_type to match the actual structural diff layer, "
                    "and keep the summary, prediction, and validation rationale tied "
                    "to that same layer."
                ),
            }
        if "implementation_scope.structural_files_changed" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an implementation_scope manifest that omits or "
                    "overreports structural files. The next candidate must list exactly "
                    "the structural files changed by the diff in "
                    "implementation_scope.structural_files_changed, excluding tests/docs, "
                    "and explain any prompt-only or non-structural edits separately."
                ),
            }
        if "prediction.expected_fixed_task_classes must reference a concrete label" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an evidence-free prediction. The next candidate "
                    "must bind prediction.expected_fixed_task_classes to a concrete "
                    "label from failure_pattern_digest, change_evaluation_digest, "
                    "rejected_update_buffer, or prior_update_lesson_entries, or return "
                    "noop/rejected if no measurable class is supported."
                ),
            }
        if "generalization.problem_class or applies_to must reference a concrete" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a task-specific or ungrounded generalization claim. "
                    "The next candidate must name a concrete failure_pattern_digest label "
                    "in generalization.problem_class or generalization.applies_to, include "
                    "anti_overfit_checks that would catch task-id or fixture-specific logic, "
                    "and explain why the slice is reusable beyond the trigger task."
                ),
            }
        if (
            "cross_round_evidence.dominant_patterns must reference a concrete" in reason_text
            or "cross_round_evidence.selected_problem_class must reference a concrete" in reason_text
        ):
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a cross-round evidence claim that is not tied to "
                    "failure_pattern_digest. The next candidate must name a concrete "
                    "failure_pattern_digest label in cross_round_evidence.dominant_patterns "
                    "and selected_problem_class, then explain why this update slice generalizes, "
                    "or return noop/rejected when the recent summaries do not support it."
                ),
            }
        if "prediction.risk_task_classes must reference top change_evaluation_digest.risk_classes" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a prediction that drops known risk classes. The "
                    "next candidate must include the top change_evaluation_digest.risk_classes "
                    "in prediction.risk_task_classes and describe how validation protects "
                    "those classes, or return noop/rejected if the risk cannot be controlled."
                ),
            }
        if "memory_record.failed_directions_to_avoid must reference top change_evaluation_digest.miss_classes" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an update summary that ignores recent prediction misses. "
                    "The next candidate must name the top change_evaluation_digest.miss_classes "
                    "in memory_record.failed_directions_to_avoid and explain the concrete "
                    "mutation that prevents repeating those missed classes, or return noop/rejected."
                ),
            }
        if "memory_record.supported_directions_to_preserve must reference each runner_pivot_policy.supported" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a patch that drops supported directions. The next "
                    "candidate must name each runner_pivot_policy.supported packet_id, "
                    "mission_candidate_id, failure_class, or component_layer in "
                    "memory_record.supported_directions_to_preserve and explain whether "
                    "the patch preserves, extends, or evidence-backed pivots away from it."
                ),
            }
        if "memory_record.failed_directions_to_avoid must reference each runner_pivot_policy.discouraged" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a direction that ignores runner_pivot_policy.discouraged. "
                    "The next candidate must name each discouraged failure_class, "
                    "component_layer, packet_id, or mission_candidate_id in "
                    "memory_record.failed_directions_to_avoid, then describe the "
                    "concrete mutation or fresh evidence that makes the direction different."
                ),
            }
        if "memory_record.failed_directions_to_avoid must reference each runner_pivot_policy.layer_pressure" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a same-layer patch that ignores runner_pivot_policy.layer_pressure. "
                    "The next candidate must name the pressured component_layer plus a recent "
                    "packet_id or failure_class in memory_record.failed_directions_to_avoid, "
                    "then pivot to another layer or cite fresh trajectory/verifier evidence "
                    "for why staying on that layer is materially different."
                ),
            }
        if "memory_record.failed_directions_to_avoid must reference each prior_update_lesson_entries" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a direction that ignores structured prior update lessons. "
                    "The next candidate must name each relevant prior_update_lesson_entries "
                    "packet_id, outcome, or mission_candidate_id in memory_record.failed_directions_to_avoid, "
                    "then describe the concrete mutation or fresh evidence that justifies revisiting it."
                ),
            }
        if "memory_record.failed_directions_to_avoid must reference the required_mutation guidance" in reason_text:
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat a memory record that only names rejected packet metadata. "
                    "The next candidate must quote or substantively cover each prior "
                    "rejected_update_buffer.required_mutation in memory_record.failed_directions_to_avoid, "
                    "then explain the concrete mutation, noop/rejected decision, or fresh evidence "
                    "that satisfies that guidance."
                ),
            }
        if "outside allowed edit roots" in reason_text:
            if changed_files and PatchReviewer(self.repo_root).review_delta(
                changed_files,
                "",
            ).accepted:
                return {
                    "avoid_repeating": False,
                    "superseded_by_current_reviewer": True,
                    "required_mutation": (
                        "This old outside-allowed-roots rejection is superseded by "
                        "the current reviewer, which now accepts these changed files. "
                        "A future candidate may touch this scope when the packet allows "
                        "it, but must still explain the current allowed edit roots and "
                        "run required validation."
                    ),
                }
            return {
                "avoid_repeating": True,
                "required_mutation": (
                    "Do not repeat an out-of-scope edit. The next candidate must target "
                    "an allowed Worker/harness path or first update the policy that "
                    "defines the allowed edit roots with review evidence."
                ),
            }
        return {
            "avoid_repeating": True,
            "required_mutation": (
                "Do not repeat this rejected update unchanged. Mutate the candidate to "
                "address the recorded review reasons, or return noop/rejected with the "
                "missing evidence."
            ),
        }

    def _rejected_evaluation_entry(
        self,
        evaluation: dict[str, Any],
        campaign_state: dict[str, Any],
        *,
        events_by_packet_id: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        packet_id = str(evaluation.get("packet_id") or "")
        event = self._codex_update_event_for_packet(
            campaign_state,
            packet_id,
            events_by_packet_id=events_by_packet_id,
        )
        prediction = evaluation.get("prediction")
        if not isinstance(prediction, dict):
            prediction = {}
        prediction_hits = self._prediction_task_events(
            evaluation.get("prediction_hits")
        )
        prediction_misses = self._prediction_task_events(
            evaluation.get("prediction_misses")
        )
        rollback_recommended = bool(evaluation.get("rollback_recommended"))
        rollback_applied = bool(evaluation.get("rollback_applied"))
        return {
            "source": "change_evaluation",
            "packet_id": packet_id,
            "summary_id": str(evaluation.get("summary_id") or ""),
            "iteration": event.get("iteration"),
            "changed_files": [
                str(path) for path in event.get("changed_files") or []
            ][:10],
            "evaluated_at": str(evaluation.get("evaluated_at") or ""),
            "failure_class": str(event.get("failure_class") or ""),
            "component_layer": str(event.get("component_layer") or ""),
            "mission_candidate_id": str(
                evaluation.get("mission_candidate_id")
                or event.get("mission_candidate_id")
                or ""
            ),
            "mission_failure_category": str(
                evaluation.get("mission_failure_category")
                or event.get("mission_failure_category")
                or ""
            ),
            "outcome": str(evaluation.get("outcome") or ""),
            "rollback_recommended": rollback_recommended,
            "rollback_applied": rollback_applied,
            "hit_count": int(evaluation.get("hit_count") or 0),
            "miss_count": int(evaluation.get("miss_count") or 0),
            "prediction_hits": prediction_hits,
            "prediction_misses": prediction_misses,
            "expected_fixed_task_classes": [
                str(item)
                for item in prediction.get("expected_fixed_task_classes") or []
            ][:8],
            "risk_task_classes": [
                str(item) for item in prediction.get("risk_task_classes") or []
            ][:8],
            "avoid_repeating": True,
            **self._loophole_record_fields(packet_id),
            "required_mutation": self._change_evaluation_required_mutation(
                evaluation,
                prediction_misses,
                rollback_recommended=rollback_recommended,
                rollback_applied=rollback_applied,
            ),
        }

    def _change_evaluation_required_mutation(
        self,
        evaluation: dict[str, Any],
        prediction_misses: list[dict[str, Any]],
        *,
        rollback_recommended: bool,
        rollback_applied: bool,
    ) -> str:
        miss_tasks = [
            str(event.get("task_id") or "").strip()
            for event in prediction_misses
            if str(event.get("task_id") or "").strip()
        ][:4]
        miss_classes: list[str] = []
        for event in prediction_misses:
            for class_name in event.get("matched_classes") or []:
                class_text = str(class_name).strip()
                if class_text and class_text not in miss_classes:
                    miss_classes.append(class_text)
                if len(miss_classes) >= 6:
                    break
            if len(miss_classes) >= 6:
                break
        parts: list[str] = []
        if miss_tasks:
            parts.append("Missed tasks: " + ", ".join(miss_tasks) + ".")
        if miss_classes:
            parts.append("Missed classes: " + ", ".join(miss_classes) + ".")
        parts.append(
            "Do not repeat this change_evaluation direction unless the next "
            "candidate names the missed evaluation evidence and explains a "
            "concrete Worker/harness mutation."
        )
        if rollback_recommended or rollback_applied:
            parts.append(
                "Because rollback was recommended or applied, the next candidate "
                "must include an explicit rollback/risk-control check before "
                "reattempting this direction."
            )
        if not miss_tasks and not miss_classes:
            outcome = str(evaluation.get("outcome") or "").strip()
            if outcome:
                parts.append("Prior outcome: " + outcome + ".")
        return " ".join(parts)

    def _rejected_frontier_regression_entry(
        self,
        regression_event: dict[str, Any],
        campaign_state: dict[str, Any],
        *,
        events_by_packet_id: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        packet_id = str(regression_event.get("packet_id") or "")
        event = self._codex_update_event_for_packet(
            campaign_state,
            packet_id,
            events_by_packet_id=events_by_packet_id,
        )
        return {
            "source": "frontier_regression",
            "packet_id": packet_id,
            "summary_id": str(regression_event.get("summary_id") or ""),
            "iteration": event.get("iteration"),
            "changed_files": [
                str(path) for path in event.get("changed_files") or []
            ][:10],
            "failure_class": str(event.get("failure_class") or ""),
            "component_layer": str(event.get("component_layer") or ""),
            "mission_candidate_id": str(
                regression_event.get("mission_candidate_id")
                or event.get("mission_candidate_id")
                or ""
            ),
            "mission_failure_category": str(
                regression_event.get("mission_failure_category")
                or event.get("mission_failure_category")
                or ""
            ),
            "outcome": "same_model_frontier_regression",
            "regression_count": int(regression_event.get("regression_count") or 0),
            "regressed_tasks": [
                str(task_id)
                for task_id in regression_event.get("regressed_tasks") or []
            ][:12],
            "rollback_applied": bool(regression_event.get("rollback_applied")),
            "avoid_repeating": True,
            **self._loophole_record_fields(packet_id),
            "required_mutation": (
                "Do not repeat this update direction after same-model frontier "
                "regression unless the next proposal names the regressed tasks, "
                "explains a concrete risk control, and preserves solved-task regression gates."
            ),
        }

    def _validation_failure_packet_id(
        self,
        failure: dict[str, Any],
        campaign_state: dict[str, Any],
        *,
        packet_ids_by_summary: dict[str, str] | None = None,
    ) -> str:
        packet_id = str(failure.get("packet_id") or "")
        if packet_id:
            return packet_id
        summary_id = str(failure.get("summary_id") or "")
        if not summary_id:
            return ""
        campaign_id = str(campaign_state.get("campaign_id") or "")
        failure_trial_ids = [
            str(trial_id) for trial_id in failure.get("trial_ids") or []
        ]
        matching_summary_packet_ids: list[str] = []
        for summary in campaign_state.get("summaries") or []:
            if not isinstance(summary, dict):
                continue
            if str(summary.get("summary_id") or "") != summary_id:
                continue
            summary_trial_ids = [
                str(trial_id) for trial_id in summary.get("trial_ids") or []
            ]
            if failure_trial_ids and summary_trial_ids != failure_trial_ids:
                continue
            packet_id = str(summary.get("codex_update_packet_id") or "")
            if packet_id:
                if failure_trial_ids:
                    return packet_id
                matching_summary_packet_ids.append(packet_id)
        if not failure_trial_ids:
            unique_summary_packet_ids = set(matching_summary_packet_ids)
            if len(unique_summary_packet_ids) == 1:
                return matching_summary_packet_ids[0]
            if len(unique_summary_packet_ids) > 1:
                return ""
        if packet_ids_by_summary is not None and failure_trial_ids:
            trial_key = self._summary_trial_key(campaign_id, summary_id, failure_trial_ids)
            keyed_packet_id = packet_ids_by_summary.get(trial_key)
            if keyed_packet_id:
                return keyed_packet_id
            unscoped_trial_key = self._summary_trial_key("", summary_id, failure_trial_ids)
            keyed_packet_id = packet_ids_by_summary.get(unscoped_trial_key)
            if keyed_packet_id:
                return keyed_packet_id
        for event in reversed(campaign_state.get("codex_update_events") or []):
            if not isinstance(event, dict):
                continue
            if str(event.get("summary_id") or "") == summary_id:
                return self._unique_summary_packet_id(
                    packet_ids_by_summary,
                    campaign_id,
                    summary_id,
                )
        if packet_ids_by_summary is not None:
            return self._unique_summary_packet_id(
                packet_ids_by_summary,
                campaign_id,
                summary_id,
            )
        return ""

    def _unique_summary_packet_id(
        self,
        packet_ids_by_summary: dict[str, str] | None,
        campaign_id: str,
        summary_id: str,
    ) -> str:
        if packet_ids_by_summary is None:
            return ""
        scoped_prefix = f"{self._summary_key(campaign_id, summary_id)}:"
        scoped_packet_ids = {
            packet_id
            for key, packet_id in packet_ids_by_summary.items()
            if packet_id
            and (
                key == self._summary_key(campaign_id, summary_id)
                or key.startswith(scoped_prefix)
            )
        }
        if len(scoped_packet_ids) == 1:
            return next(iter(scoped_packet_ids))
        if len(scoped_packet_ids) > 1:
            return ""
        unscoped_prefix = f"{self._summary_key('', summary_id)}:"
        unscoped_packet_ids = {
            packet_id
            for key, packet_id in packet_ids_by_summary.items()
            if packet_id and (key == summary_id or key.startswith(unscoped_prefix))
        }
        if len(unscoped_packet_ids) == 1:
            return next(iter(unscoped_packet_ids))
        return ""

    def _rejected_validation_failure_entry(
        self,
        failure: dict[str, Any],
        campaign_state: dict[str, Any],
        packet_id: str,
        *,
        events_by_packet_id: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        event = self._codex_update_event_for_packet(
            campaign_state,
            packet_id,
            events_by_packet_id=events_by_packet_id,
        )
        reason = str(failure.get("reason") or "")
        return {
            "source": "codex_validation_failure",
            "packet_id": packet_id,
            "summary_id": str(failure.get("summary_id") or ""),
            "iteration": event.get("iteration"),
            "changed_files": [
                str(path) for path in event.get("changed_files") or []
            ][:10],
            "failure_class": str(event.get("failure_class") or ""),
            "component_layer": str(event.get("component_layer") or ""),
            "outcome": "rolled_back_validation_failure",
            "failure_marker": str(failure.get("failure_marker") or ""),
            "reason": reason,
            "exit_code": int(failure.get("exit_code") or 0),
            "rolled_back": bool(failure.get("rolled_back")),
            "avoid_repeating": True,
            **self._loophole_record_fields(packet_id),
            "required_mutation": self._validation_failure_required_mutation(reason),
        }

    def _validation_failure_required_mutation(self, reason: str) -> str:
        reason_text = reason.lower()
        if "pre-update regression failed" in reason_text:
            return (
                "Do not treat a pre-update regression failure as evidence about a "
                "new Codex patch direction. The next candidate must first refresh or "
                "quarantine the stale baseline snapshot, prove the same-model pre-update "
                "gate is stable, or return noop/rejected with the unstable baseline evidence."
            )
        if "post-update regression failed" in reason_text:
            return (
                "Do not repeat this accepted update direction after post-update "
                "regression unless fresh verifier evidence explains the specific "
                "mutation, names the regressed solved-task class, and adds a regression "
                "risk control before reattempting the patch."
            )
        return (
            "Do not repeat this accepted update direction after regression or "
            "validation rollback unless fresh verifier evidence explains a "
            "specific mutation and regression risk control."
        )

    def _codex_update_event_for_packet(
        self,
        campaign_state: dict[str, Any],
        packet_id: str,
        *,
        events_by_packet_id: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        for event in reversed(campaign_state.get("codex_update_events") or []):
            if isinstance(event, dict) and str(event.get("packet_id") or "") == packet_id:
                return event
        if events_by_packet_id is not None:
            return events_by_packet_id.get(packet_id, {})
        return {}

    def _codex_update_events_by_packet(
        self,
        campaign_states: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        events_by_packet_id: dict[str, dict[str, Any]] = {}
        for state in reversed(campaign_states):
            for event in state.get("codex_update_events") or []:
                if not isinstance(event, dict):
                    continue
                packet_id = str(event.get("packet_id") or "")
                if packet_id:
                    events_by_packet_id[packet_id] = event
        return events_by_packet_id

    def _codex_update_packet_ids_by_summary(
        self,
        campaign_states: list[dict[str, Any]],
    ) -> dict[str, str]:
        packet_ids_by_summary_id: dict[str, str] = {}
        for state in reversed(campaign_states):
            campaign_id = str(state.get("campaign_id") or "")
            for summary in state.get("summaries") or []:
                if not isinstance(summary, dict):
                    continue
                summary_id = str(summary.get("summary_id") or "")
                packet_id = str(summary.get("codex_update_packet_id") or "")
                if summary_id and packet_id:
                    trial_ids = [str(item) for item in summary.get("trial_ids") or []]
                    if trial_ids:
                        self._record_summary_packet_id(
                            packet_ids_by_summary_id,
                            self._summary_trial_key(campaign_id, summary_id, trial_ids),
                            packet_id,
                        )
                        self._record_summary_packet_id(
                            packet_ids_by_summary_id,
                            self._summary_trial_key("", summary_id, trial_ids),
                            packet_id,
                        )
                    self._record_summary_packet_id(
                        packet_ids_by_summary_id,
                        self._summary_key(campaign_id, summary_id),
                        packet_id,
                    )
                    self._record_summary_packet_id(
                        packet_ids_by_summary_id,
                        summary_id,
                        packet_id,
                    )
            for event in state.get("codex_update_events") or []:
                if not isinstance(event, dict):
                    continue
                summary_id = str(event.get("summary_id") or "")
                packet_id = str(event.get("packet_id") or "")
                if summary_id and packet_id:
                    self._record_summary_packet_id(
                        packet_ids_by_summary_id,
                        self._summary_key(campaign_id, summary_id),
                        packet_id,
                    )
                    self._record_summary_packet_id(
                        packet_ids_by_summary_id,
                        summary_id,
                        packet_id,
                    )
        return packet_ids_by_summary_id

    def _record_summary_packet_id(
        self,
        packet_ids_by_summary_id: dict[str, str],
        key: str,
        packet_id: str,
    ) -> None:
        if not key or not packet_id:
            return
        existing = packet_ids_by_summary_id.get(key)
        if existing is not None and existing != packet_id:
            packet_ids_by_summary_id[key] = ""
            return
        packet_ids_by_summary_id[key] = packet_id

    def _prediction_task_events(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        events: list[dict[str, Any]] = []
        for item in value[:8]:
            if not isinstance(item, dict):
                continue
            event = {
                "task_id": str(item.get("task_id") or ""),
                "event": str(item.get("event") or ""),
                "reason": str(item.get("reason") or ""),
            }
            raw_classes = item.get("matched_classes")
            if not isinstance(raw_classes, list) or not raw_classes:
                raw_classes = item.get("labels")
            matched_classes = [
                str(class_name)
                for class_name in raw_classes or []
                if str(class_name).strip()
            ]
            if matched_classes:
                event["matched_classes"] = matched_classes[:8]
            events.append(event)
        return events

    def _append_review_based_rejections(
        self,
        buffer: list[dict[str, Any]],
        seen: set[str],
        *,
        limit: int | None = None,
    ) -> None:
        if limit is not None and int(limit) <= 0:
            return
        diffs_dir = self.memory_path / "diffs"
        if not diffs_dir.exists():
            return
        review_paths = sorted(
            diffs_dir.glob("codex_packet_*/review.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        appended = 0
        for review_path in review_paths:
            packet_id = review_path.parent.name
            key = f"review:{packet_id}"
            if key in seen:
                continue
            try:
                review = self._read_json(review_path)
            except (OSError, ValueError):
                continue
            validation_failed = self._validation_failed(review_path.parent)
            if bool(review.get("accepted")) and not validation_failed:
                continue
            reasons = [str(item) for item in review.get("reasons") or []][:6]
            changed_files = [
                str(item) for item in review.get("changed_files") or []
            ][:10]
            seen.add(key)
            buffer.append(
                {
                    "source": "review",
                    "packet_id": packet_id,
                    "outcome": "validation_failed" if validation_failed else "rejected",
                    "accepted": bool(review.get("accepted")),
                    "reasons": reasons,
                    "changed_files": changed_files,
                    "validation_failed": validation_failed,
                    **self._rejected_event_guidance(reasons, changed_files),
                    **self._loophole_record_fields(packet_id),
                }
            )
            appended += 1
            if limit is not None and appended >= int(limit):
                break

    def _discouraged_failure_categories(self) -> set[str]:
        """Failure categories that repeatedly failed recent Codex updates.

        Derived from runner_pivot_policy.discouraged so pattern ranking can demote
        treadmilled directions. Both the free-form failure_class tokens and any
        mission_failure_category are included so digest failure_category values
        (which may be a single category token) can match.
        """
        categories: set[str] = set()
        policy = self._runner_pivot_policy()
        for entry in policy.get("discouraged", []) or []:
            if not isinstance(entry, dict):
                continue
            failure_class = str(entry.get("failure_class") or "").strip()
            if failure_class:
                categories.add(failure_class)
                for token in failure_class.split():
                    if token:
                        categories.add(token)
            mission_category = str(entry.get("mission_failure_category") or "").strip()
            if mission_category:
                categories.add(mission_category)
        return categories

    def _runner_pivot_policy(self) -> dict[str, Any]:
        attempts: list[dict[str, Any]] = []
        states = list(reversed(self._recent_campaign_states(limit=None)))
        inferred_results = self._inferred_next_eval_results_from_states(states)
        events_by_packet_id = self._codex_update_events_by_packet(states)
        for state in states:
            raw = state.get("failure_class_attempts")
            if isinstance(raw, list):
                attempts.extend(
                    self._attempt_with_inferred_next_eval_result(item, inferred_results)
                    for item in raw
                    if isinstance(item, dict)
                )
        discouraged: list[dict[str, Any]] = []
        layer_pressure: list[dict[str, Any]] = []
        unsuccessful_by_direction: dict[tuple[str, str], int] = {}
        unsuccessful_by_layer: dict[str, list[dict[str, Any]]] = {}
        unsuccessful_by_mission: dict[str, list[dict[str, Any]]] = {}
        latest_result_by_direction: dict[tuple[str, str], dict[str, Any]] = {}
        latest_attempt_by_layer: dict[str, dict[str, Any]] = {}
        latest_attempt_by_mission: dict[str, dict[str, Any]] = {}
        for attempt in attempts:
            failure_class = str(attempt.get("failure_class") or "")
            component_layer = str(attempt.get("component_layer") or "")
            mission_candidate_id = str(attempt.get("mission_candidate_id") or "")
            key = (failure_class, component_layer)
            if component_layer:
                latest_attempt_by_layer[component_layer] = attempt
            if mission_candidate_id:
                latest_attempt_by_mission[mission_candidate_id] = attempt
            if failure_class and str(attempt.get("next_eval_result") or ""):
                latest_result_by_direction[key] = attempt
            if self._attempt_is_superseded_scope_rejection(
                attempt,
                events_by_packet_id,
            ):
                continue
            if not failure_class or not self._attempt_was_unsuccessful(attempt):
                continue
            unsuccessful_by_direction[key] = unsuccessful_by_direction.get(key, 0) + 1
            if component_layer:
                unsuccessful_by_layer.setdefault(component_layer, []).append(attempt)
            if mission_candidate_id:
                unsuccessful_by_mission.setdefault(mission_candidate_id, []).append(attempt)
        for (failure_class, component_layer), count in unsuccessful_by_direction.items():
            if count < 2:
                continue
            latest_result = latest_result_by_direction.get(
                (failure_class, component_layer),
                {},
            )
            if latest_result and not self._attempt_was_unsuccessful(latest_result):
                continue
            discouraged.append(
                {
                    "failure_class": failure_class,
                    "component_layer": component_layer,
                    "unsuccessful_attempts": count,
                    "reason": (
                        "The same failure class and component layer failed in "
                        "multiple recent Codex update attempts; pivot to another "
                        "layer unless fresh evidence justifies staying."
                    ),
                }
            )
        for mission_candidate_id, mission_attempts in unsuccessful_by_mission.items():
            if len(mission_attempts) < 2:
                continue
            latest_attempt = latest_attempt_by_mission.get(mission_candidate_id, {})
            if latest_attempt and self._attempt_was_supported(latest_attempt):
                continue
            failure_categories = sorted(
                {
                    str(item.get("mission_failure_category") or "")
                    for item in mission_attempts
                    if str(item.get("mission_failure_category") or "")
                }
            )[:5]
            discouraged.append(
                {
                    "failure_class": " ".join(failure_categories),
                    "component_layer": "mission_selection",
                    "mission_candidate_id": mission_candidate_id,
                    "mission_failure_category": failure_categories[0]
                    if failure_categories
                    else "",
                    "unsuccessful_attempts": len(mission_attempts),
                    "recent_packet_ids": [
                        str(item.get("packet_id") or "")
                        for item in mission_attempts
                        if str(item.get("packet_id") or "")
                    ],
                    "reason": (
                        "The same mission-selected candidate failed in multiple "
                        "recent Codex update attempts; mutate the candidate, "
                        "choose another mission slice, or name fresh evidence for "
                        "retrying it."
                    ),
                }
            )
        for component_layer, layer_attempts in unsuccessful_by_layer.items():
            if len(layer_attempts) < 3:
                continue
            latest_attempt = latest_attempt_by_layer.get(component_layer, {})
            if latest_attempt and self._attempt_was_supported(latest_attempt):
                continue
            layer_pressure.append(
                {
                    "component_layer": component_layer,
                    "unsuccessful_attempts": len(layer_attempts),
                    "recent_packet_ids": [
                        str(item.get("packet_id") or "")
                        for item in layer_attempts
                        if str(item.get("packet_id") or "")
                    ],
                    "failure_classes": sorted(
                        {
                            str(item.get("failure_class") or "")
                            for item in layer_attempts
                            if str(item.get("failure_class") or "")
                        }
                    )[:5],
                    "reason": (
                        "Multiple recent Codex update attempts failed within this "
                        "component layer even when their failure_class wording differed; "
                        "pivot to another layer or name fresh evidence for a distinct surface."
                    ),
                }
            )
        supported = [
            self._supported_attempt_entry(attempt)
            for attempt in latest_result_by_direction.values()
            if self._attempt_was_supported(attempt)
        ]
        return {
            "attempts": attempts,
            "attempt_history_count": len(attempts),
            "attempt_history_truncated": False,
            "attempt_history_truncation_stop_condition": False,
            "discouraged": discouraged,
            "layer_pressure": layer_pressure,
            "supported": supported,
            "rule": (
                "After repeated unsuccessful updates for the same failure class at "
                "the same component layer, prefer rollback/pivot to another layer. "
                "This uses the full available campaign-state attempt history; "
                "attempt count is not a sub-agent stop or truncation condition."
            ),
        }

    def _inferred_next_eval_results_from_states(
        self,
        campaign_states: list[dict[str, Any]],
    ) -> dict[str, str]:
        inferred: dict[str, str] = {}
        packet_ids_by_summary = self._codex_update_packet_ids_by_summary(campaign_states)
        for state in campaign_states:
            for packet_id, outcome in self._inferred_next_eval_results_by_packet(
                state,
                packet_ids_by_summary=packet_ids_by_summary,
            ).items():
                self._record_inferred_next_eval_result(inferred, packet_id, outcome)
        return inferred

    def _inferred_next_eval_results_by_packet(
        self,
        campaign_state: dict[str, Any],
        *,
        packet_ids_by_summary: dict[str, str] | None = None,
    ) -> dict[str, str]:
        inferred: dict[str, str] = {}
        for evaluation in campaign_state.get("change_evaluations") or []:
            if not isinstance(evaluation, dict):
                continue
            packet_id = str(evaluation.get("packet_id") or "")
            if not packet_id:
                continue
            outcome = "rollback_applied" if evaluation.get("rollback_applied") else str(
                evaluation.get("outcome") or ""
            )
            self._record_inferred_next_eval_result(inferred, packet_id, outcome)
        for failure in campaign_state.get("codex_validation_failures") or []:
            if not isinstance(failure, dict):
                continue
            if not failure.get("rolled_back"):
                continue
            packet_id = self._validation_failure_packet_id(
                failure,
                campaign_state,
                packet_ids_by_summary=packet_ids_by_summary,
            )
            if packet_id:
                self._record_inferred_next_eval_result(
                    inferred,
                    packet_id,
                    "validation_failed",
                )
        for event in campaign_state.get("frontier_regression_events") or []:
            if not isinstance(event, dict):
                continue
            packet_id = str(event.get("packet_id") or "")
            if packet_id:
                self._record_inferred_next_eval_result(
                    inferred,
                    packet_id,
                    "frontier_regression",
                )
        return inferred

    def _record_inferred_next_eval_result(
        self,
        inferred: dict[str, str],
        packet_id: str,
        outcome: str,
    ) -> None:
        if self._next_eval_result_should_replace(inferred.get(packet_id, ""), outcome):
            inferred[packet_id] = outcome

    def _attempt_with_inferred_next_eval_result(
        self,
        attempt: dict[str, Any],
        inferred_results: dict[str, str],
    ) -> dict[str, Any]:
        packet_id = str(attempt.get("packet_id") or "")
        inferred = inferred_results.get(packet_id, "")
        if not inferred:
            return attempt
        current = str(attempt.get("next_eval_result") or "")
        if not self._next_eval_result_should_replace(current, inferred):
            return attempt
        normalized = dict(attempt)
        normalized["next_eval_result"] = inferred
        normalized["next_eval_result_inferred"] = True
        return normalized

    def _next_eval_result_should_replace(self, current: str, incoming: str) -> bool:
        if not current:
            return bool(incoming)
        if not incoming:
            return False
        return self._next_eval_result_priority(incoming) > self._next_eval_result_priority(
            current
        )

    def _next_eval_result_priority(self, outcome: str) -> int:
        return {
            "": 0,
            "insufficient_prediction": 1,
            "insufficient_evidence": 1,
            "mixed": 2,
            "prediction_supported": 3,
            "prediction_missed": 4,
            "rollback_applied": 5,
            "validation_failed": 5,
            "frontier_regression": 5,
        }.get(outcome, 2)

    def _supported_attempt_entry(self, attempt: dict[str, Any]) -> dict[str, Any]:
        entry = {
            "failure_class": str(attempt.get("failure_class") or ""),
            "component_layer": str(attempt.get("component_layer") or ""),
            "packet_id": str(attempt.get("packet_id") or ""),
            "summary_id": str(attempt.get("summary_id") or ""),
            "reason": (
                "This recent update direction matched its declared prediction; "
                "preserve or extend it only with the same verifier/frontier "
                "evidence discipline."
            ),
        }
        mission_candidate_id = str(attempt.get("mission_candidate_id") or "")
        mission_failure_category = str(
            attempt.get("mission_failure_category") or ""
        )
        if mission_candidate_id:
            entry["mission_candidate_id"] = mission_candidate_id
        if mission_failure_category:
            entry["mission_failure_category"] = mission_failure_category
        return entry

    def _attempt_was_supported(self, attempt: dict[str, Any]) -> bool:
        return bool(attempt.get("accepted")) and str(
            attempt.get("next_eval_result") or ""
        ) == "prediction_supported"

    def _attempt_was_unsuccessful(self, attempt: dict[str, Any]) -> bool:
        if not bool(attempt.get("accepted")):
            return True
        return str(attempt.get("next_eval_result") or "") in {
            "mixed",
            "prediction_missed",
            "rollback_applied",
            "frontier_regression",
            "validation_failed",
        }

    def _attempt_is_superseded_scope_rejection(
        self,
        attempt: dict[str, Any],
        events_by_packet_id: dict[str, dict[str, Any]],
    ) -> bool:
        if bool(attempt.get("accepted")):
            return False
        packet_id = str(attempt.get("packet_id") or "")
        if not packet_id:
            return False
        event = events_by_packet_id.get(packet_id, {})
        if not event:
            return False
        reasons = [str(item) for item in event.get("reasons") or []]
        changed_files = [str(path) for path in event.get("changed_files") or []]
        guidance = self._rejected_event_guidance(reasons, changed_files)
        return bool(guidance.get("superseded_by_current_reviewer"))

    def _prior_update_lessons(
        self,
        limit: int | None = None,
        max_chars: int = 4000,
    ) -> list[str]:
        _ = limit
        lessons_dir = self.memory_path / "memory" / "component_lessons"
        if not lessons_dir.exists():
            return []
        lessons: list[str] = []
        lesson_paths = sorted(lessons_dir.glob("*.md"))
        for path in lesson_paths:
            try:
                text = path.read_text(errors="replace").strip()
            except OSError:
                continue
            if text:
                lessons.append(f"{path.name}: {text[-max_chars:]}")
        return lessons

    def _prior_update_lesson_entries(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        _ = limit
        lessons_dir = self.memory_path / "memory" / "component_lessons"
        if not lessons_dir.exists():
            return []
        entries: list[dict[str, Any]] = []
        for path in sorted(lessons_dir.glob("*.md"), key=lambda item: item.stat().st_mtime):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            entries.extend(self._structured_lesson_entries(path.name, text))
        return entries

    def _structured_lesson_entries(
        self,
        filename: str,
        text: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        current_header = ""
        current_fields: dict[str, Any] = {}
        current_reason_lines: list[str] = []
        in_reason_block = False

        def flush() -> None:
            nonlocal current_header, current_fields, current_reason_lines, in_reason_block
            if not current_fields:
                current_header = ""
                current_reason_lines = []
                in_reason_block = False
                return
            entry = {
                "file": filename,
                "recorded_at": current_header.lstrip("# ").strip(),
                **current_fields,
            }
            if current_reason_lines:
                entry["reason"] = "\n".join(current_reason_lines).strip()
            entries.append(entry)
            current_header = ""
            current_fields = {}
            current_reason_lines = []
            in_reason_block = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                flush()
                current_header = line
                continue
            key = ""
            value = ""
            if ":" in line:
                raw_key, raw_value = line.split(":", 1)
                key = raw_key.strip()
                value = raw_value.strip()
            if not key:
                if in_reason_block and raw_line.strip():
                    current_reason_lines.append(raw_line.strip())
                continue
            if key not in {
                "source",
                "packet_id",
                "outcome",
                "summary_id",
                "rollback_applied",
                "mission_candidate_id",
                "mission_failure_category",
                "regressed_tasks",
                "reason",
            }:
                if in_reason_block and raw_line.strip():
                    current_reason_lines.append(raw_line.strip())
                continue
            if key != "reason":
                in_reason_block = False
            if key == "rollback_applied":
                current_fields[key] = value.lower() == "true"
            elif key == "regressed_tasks":
                current_fields[key] = [
                    item.strip() for item in value.split(",") if item.strip()
                ]
            elif key == "reason":
                current_reason_lines.append(value)
                in_reason_block = True
            else:
                current_fields[key] = value
        flush()
        return entries

    def _external_research_policy(
        self,
        update_history: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        history = update_history if update_history is not None else self._update_history()
        recommended = bool(history.get("research_recommended"))
        refs = [
            self._local_reference_status("/tmp/harness-evolver-refs/codex"),
            self._local_reference_status("/tmp/harness-evolver-refs/forgecode"),
            self._local_reference_status(
                "/tmp/harness-evolver-refs/agentic-harness-engineering"
            ),
            self._local_reference_status("/tmp/harness-evolver-refs/meta-harness"),
            self._local_reference_status("/tmp/harness-evolver-refs/TACO"),
            self._local_reference_status("/tmp/harness-evolver-refs/openclacky"),
            self._local_reference_status("/tmp/harness-evolver-refs/SkillOpt"),
        ]
        return {
            "status": "recommended" if recommended else "available_if_needed",
            "trigger": (
                "Use external research when recent Codex updates are repeatedly "
                "rejected, no-op, or fail to improve comparable validation."
            ),
            "recent_unsuccessful_updates": history.get("recent_unsuccessful_updates", 0),
            "local_read_only_refs": [ref["path"] for ref in refs],
            "local_reference_status": refs,
            "web_sources": [
                "https://mp.weixin.qq.com/s/sgP8m1nnW7JhsDT7Ki7nVw",
                "https://www.harborframework.com/docs/tutorials/running-terminal-bench",
                "https://www.tbench.ai/leaderboard/terminal-bench/2.0",
                "https://www.tbench.ai/news/leaderboard-integrity-update",
                "https://trinkle23897.github.io/learning-beyond-gradients/",
                "https://vix.codes/",
                "https://github.com/kirby88/vix-releases",
                "https://github.com/schpet/jjagent",
                "https://github.com/china-qijizhifeng/agentic-harness-engineering",
                "https://capy.ai/",
                "https://capy.ai/articles",
                "https://github.com/octu0/polaris",
                "https://www.wozcode.com/blog",
                "https://github.com/WithWoz/wozcode-plugin",
                "https://github.com/stanford-iris-lab/meta-harness",
                "https://github.com/multimodal-art-projection/TACO",
                "https://github.com/clacky-ai/openclacky",
                "https://github.com/microsoft/SkillOpt",
                "https://microsoft.github.io/SkillOpt/",
                "https://claude.com/blog/how-claude-code-works-in-large-codebases-best-practices-and-where-to-start",
                "https://github.com/openai/codex",
                "https://github.com/harbor-framework/terminal-bench/tree/main/terminal_bench/agents/terminus_2",
                "https://docs.anthropic.com/en/docs/claude-code/memory",
                "https://factory.ai/news/missions-architecture",
                "https://docs.langchain.com/oss/python/langgraph/durable-execution",
                "https://browser-use.com/posts/bitter-lesson-agent-frameworks",
                "https://mariozechner.at/posts/2025-11-30-pi-coding-agent/",
            ],
            "research_focus_areas": [
                (
                    "Prefer a small Worker loop with broad shell/file action space; "
                    "add restrictions from campaign evidence instead of prebuilding "
                    "large brittle abstractions."
                ),
                (
                    "Preserve explicit completion via done plus verifier gates; do "
                    "not let plain-text completion or Codex updater summaries become "
                    "benchmark pass evidence."
                ),
                (
                    "Keep context and tool-output management observable: prune or "
                    "compress bulky transient outputs while leaving trajectory events "
                    "that explain what was removed."
                ),
                (
                    "Expose provider, reasoning, token, cache, validation, and update "
                    "handoff decisions in artifacts so model or provider switches can "
                    "be audited instead of hidden behind framework state."
                ),
                (
                    "Self-Harness weakness mining must start from real campaign, "
                    "trajectory, change-evaluation, or regression evidence before "
                    "proposing a Worker or harness policy change."
                ),
                (
                    "Self-Harness bounded harness proposal: change one narrow "
                    "Worker, tool, context, recovery, verifier-gate, or updater "
                    "policy slice rather than adding broad framework machinery."
                ),
                (
                    "Self-Harness proposal validation must name the deterministic "
                    "test, regression, Harbor/verifier evidence, or report gate that "
                    "accepts or rejects the proposed harness change."
                ),
                (
                    "Self-Harness same-model self-improvement must compare the "
                    "candidate against same-model frontier or validation artifacts, "
                    "not hide model/provider switches inside the update."
                ),
            ],
            "fetch_requirements": [
                {
                    "url_prefix": "https://mp.weixin.qq.com/",
                    "required_user_agent": WECHAT_ARTICLE_USER_AGENT,
                    "required_header": "User-Agent",
                    "failure_signature": "environment abnormal verification page",
                    "reason": (
                        "WeChat public-account articles can return an environment "
                        "verification page to generic crawlers; fetch them with a "
                        "WeChat mobile client User-Agent before treating the source "
                        "as unavailable."
                    ),
                }
            ],
            "final_report_expectation": (
                "If research was used, list the concrete sources and how they "
                "changed the patch. If skipped, explain why local evidence was sufficient."
            ),
        }

    def _local_reference_status(self, path: str) -> dict[str, Any]:
        ref_path = Path(path)
        return {
            "path": path,
            "exists": ref_path.exists(),
            "fallback": (
                "Use the matching GitHub/web source from web_sources when this local "
                "read-only checkout is absent."
                if not ref_path.exists()
                else ""
            ),
        }

    def _validation_failed(self, run_dir: Path) -> bool:
        validation_path = run_dir / "validation_results.json"
        if not validation_path.exists():
            return False
        try:
            data = self._read_json(validation_path)
        except (OSError, ValueError):
            return True
        commands = data.get("commands")
        if not isinstance(commands, list):
            return True
        return any(
            bool(command.get("timed_out")) or int(command.get("returncode", 1)) != 0
            for command in commands
            if isinstance(command, dict)
        )

    def _update_history_status_by_packet(
        self,
        states: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        rollback_markers = (
            "codex_update:rolled_back_worse_score",
            "codex_update:rolled_back_regression_gate",
            "codex_update:rolled_back_pre_regression_gate",
            "codex_update:rolled_back_pending_regression_gate",
            "codex_update:rolled_back_prediction_miss",
            "codex_update:rolled_back_frontier_regression",
        )
        statuses: dict[str, dict[str, Any]] = {}
        packet_ids_by_summary = self._codex_update_packet_ids_by_summary(states)

        def status_for(packet_id: str) -> dict[str, Any]:
            return statuses.setdefault(
                packet_id,
                {
                    "rolled_back": False,
                    "score_declined": False,
                    "evaluation_outcome": "",
                },
            )

        for state in states:
            for failure in state.get("codex_validation_failures") or []:
                if not (isinstance(failure, dict) and failure.get("rolled_back")):
                    continue
                packet_id = self._validation_failure_packet_id(
                    failure,
                    state,
                    packet_ids_by_summary=packet_ids_by_summary,
                )
                if packet_id:
                    status_for(packet_id)["rolled_back"] = True

            for evaluation in state.get("change_evaluations") or []:
                if not isinstance(evaluation, dict):
                    continue
                packet_id = str(evaluation.get("packet_id") or "")
                if not packet_id:
                    continue
                current = str(evaluation.get("outcome") or "")
                if evaluation.get("rollback_applied"):
                    current = "rollback_applied"
                    status_for(packet_id)["rolled_back"] = True
                existing = str(status_for(packet_id).get("evaluation_outcome") or "")
                if self._next_eval_result_should_replace(existing, current):
                    status_for(packet_id)["evaluation_outcome"] = current

            for event in state.get("frontier_regression_events") or []:
                if not isinstance(event, dict):
                    continue
                packet_id = str(event.get("packet_id") or "")
                if packet_id and bool(event.get("rollback_applied")):
                    status_for(packet_id)["rolled_back"] = True

            summaries = state.get("summaries")
            if not isinstance(summaries, list):
                continue
            for index, summary in enumerate(summaries):
                if not isinstance(summary, dict):
                    continue
                packet_id = str(summary.get("codex_update_packet_id") or "")
                if not packet_id:
                    continue
                if self._summary_has_rollback_marker(
                    summary,
                    packet_id,
                    rollback_markers,
                ):
                    status_for(packet_id)["rolled_back"] = True
                if index >= len(summaries) - 1:
                    continue
                current_score = self._summary_score(summary)
                next_score = self._summary_score(summaries[index + 1])
                if (
                    current_score is not None
                    and next_score is not None
                    and next_score < current_score
                ):
                    status_for(packet_id)["score_declined"] = True
        return statuses

    def _codex_update_was_rolled_back(self, packet_id: str) -> bool:
        states = self._recent_campaign_states(limit=None)
        return bool(
            self._update_history_status_by_packet(states)
            .get(packet_id, {})
            .get("rolled_back")
        )

    def _codex_update_evaluation_outcome(self, packet_id: str) -> str:
        states = self._recent_campaign_states(limit=None)
        return str(
            self._update_history_status_by_packet(states)
            .get(packet_id, {})
            .get("evaluation_outcome", "")
        )

    def _codex_update_score_declined(self, packet_id: str) -> bool:
        states = self._recent_campaign_states(limit=None)
        return bool(
            self._update_history_status_by_packet(states)
            .get(packet_id, {})
            .get("score_declined")
        )

    def _summary_has_rollback_marker(
        self,
        summary: Any,
        packet_id: str,
        markers: tuple[str, ...],
    ) -> bool:
        if not isinstance(summary, dict):
            return False
        if str(summary.get("codex_update_packet_id") or "") != packet_id:
            return False
        text = json.dumps(summary)
        return any(marker in text for marker in markers)

    def _summary_score(self, summary: Any) -> float | None:
        if not isinstance(summary, dict):
            return None
        value = summary.get("overall_score")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _read_json(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
