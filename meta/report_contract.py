"""Typed, single-source contract for Codex update review findings.

The registry in this module owns the stable rule id, severity, packet wording,
and validator binding for every final-report gate.  Host review and
``scripts/report_lint.py`` execute the same bindings; neither path recovers
severity by matching human-readable reason text.

Non-report review findings (diff integrity, Codex execution, host validation,
rollback, and skip decisions) also use stable ids from this registry so every
``review.json`` finding is structured.  Unknown ids fail closed through the
``internal.contract_error`` rule instead of producing an empty id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from meta.update_policy import classify_component_delta


FATAL = "fatal"
REPORT = "report"
FINAL_REPORT_SCOPE = "final_report"
REVIEW_SCOPE = "review"
INTERNAL_RULE_ID = "internal.contract_error"


@dataclass(frozen=True)
class ReportContractRule:
    """One registered review rule and its validator binding."""

    id: str
    severity: str
    description: str
    binding: str
    scope: str = FINAL_REPORT_SCOPE

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("ReportContractRule id must not be empty")
        if self.severity not in {FATAL, REPORT}:
            raise ValueError(
                f"ReportContractRule {self.id!r} has invalid severity {self.severity!r}"
            )
        if not self.description.strip():
            raise ValueError(f"ReportContractRule {self.id!r} needs a description")
        if not self.binding.strip():
            raise ValueError(f"ReportContractRule {self.id!r} needs a binding")
        if self.scope not in {FINAL_REPORT_SCOPE, REVIEW_SCOPE}:
            raise ValueError(
                f"ReportContractRule {self.id!r} has invalid scope {self.scope!r}"
            )


@dataclass(frozen=True)
class ReportViolation:
    """Structured finding emitted directly by a registered rule."""

    rule_id: str
    severity: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "reason": self.reason,
            "rule_id": self.rule_id,
            "severity": self.severity,
        }


@dataclass
class ReportValidationContext:
    """Complete host/lint context used by final-report validators."""

    changed_files: list[str] = field(default_factory=list)
    ignore_files: set[str] = field(default_factory=set)
    required_validation_commands: list[str] = field(default_factory=list)
    host_validation_commands: list[str] = field(default_factory=list)
    failure_pattern_digest: dict[str, Any] = field(default_factory=dict)
    mission_debug: dict[str, Any] = field(default_factory=dict)
    rejected_update_buffer: list[dict[str, Any]] = field(default_factory=list)
    runner_pivot_policy: dict[str, Any] = field(default_factory=dict)
    change_evaluation_digest: dict[str, Any] = field(default_factory=dict)
    prior_update_lesson_entries: list[dict[str, Any]] = field(default_factory=list)
    external_research_recommended: bool = False
    external_research_policy: dict[str, Any] = field(default_factory=dict)


class ViolationCollector:
    """Collect typed findings while preserving legacy ``reasons`` access."""

    def __init__(self, violations: Iterable[ReportViolation] = ()) -> None:
        self._violations = list(violations)

    @property
    def violations(self) -> list[ReportViolation]:
        return list(self._violations)

    @property
    def reasons(self) -> list[str]:
        return [item.reason for item in self._violations]

    @property
    def blocking(self) -> list[ReportViolation]:
        return [item for item in self._violations if item.severity == FATAL]

    def add(self, rule_id: str, reason: str) -> ReportViolation:
        finding = violation(rule_id, reason)
        self._violations.append(finding)
        return finding

    def extend(self, violations: Iterable[ReportViolation]) -> None:
        self._violations.extend(violations)


def _rule(
    rule_id: str,
    severity: str,
    description: str,
    binding: str,
    *,
    scope: str = FINAL_REPORT_SCOPE,
) -> ReportContractRule:
    return ReportContractRule(
        id=rule_id,
        severity=severity,
        description=description,
        binding=binding,
        scope=scope,
    )


_RULES: tuple[ReportContractRule, ...] = (
    # Final-report rules. Each binding is executed by both host review and lint.
    _rule(
        "report.present",
        FATAL,
        "A Codex final report must be present and parse as the requested JSON object.",
        "report_presence",
    ),
    _rule(
        "report.status",
        FATAL,
        "status must be edited, noop, or rejected.",
        "report_status",
    ),
    _rule(
        "report.status_changed_files",
        FATAL,
        "A report with actual changed files must use status=edited.",
        "report_status_changed_files",
    ),
    _rule(
        "report.basic_fields",
        FATAL,
        "summary is required and skipped_validation_reason, when present, must be a string.",
        "basic_report",
    ),
    _rule(
        "report.changed_files",
        FATAL,
        "changed_files must be a list that exactly matches the isolated Codex diff.",
        "changed_files",
    ),
    _rule(
        "report.validation_shape",
        FATAL,
        "validation_commands must be a list; skipped validation needs an explanation when host validation is unavailable.",
        "validation_shape",
    ),
    _rule(
        "report.required_validation",
        FATAL,
        "Every required validation command must be reported, supplied by host validation, or explicitly skipped.",
        "required_validation",
    ),
    _rule(
        "report.loophole_review",
        FATAL,
        "Edited patches must include strategy_confidence plus non-empty loophole_review and loophole_fixes.",
        "loophole_review",
    ),
    _rule(
        "report.generalization_structure",
        FATAL,
        "generalization must contain problem_class, applies_to, anti_overfit_checks, and why_not_task_specific.",
        "generalization_structure",
    ),
    _rule(
        "report.generalization_evidence",
        REPORT,
        "generalization.problem_class or applies_to should reference a concrete packet failure or mission mechanism label.",
        "generalization_evidence",
    ),
    _rule(
        "report.cross_round_structure",
        FATAL,
        "cross_round_evidence must cite summaries, dominant patterns, the selected problem class, and why the slice generalizes.",
        "cross_round_structure",
    ),
    _rule(
        "report.cross_round_patterns",
        REPORT,
        "cross_round_evidence.dominant_patterns should reference a concrete packet failure or mission mechanism label.",
        "cross_round_patterns",
    ),
    _rule(
        "report.cross_round_problem_class",
        REPORT,
        "cross_round_evidence.selected_problem_class should reference a concrete packet failure or mission mechanism label.",
        "cross_round_problem_class",
    ),
    _rule(
        "report.memory_structure",
        FATAL,
        "memory_record must contain concise and detailed records plus list-valued failed and supported directions.",
        "memory_structure",
    ),
    _rule(
        "report.memory_failed_directions",
        REPORT,
        "memory_record.failed_directions_to_avoid should cover rejected updates, pivot pressure, evaluation misses, and prior failed lessons.",
        "memory_failed_directions",
    ),
    _rule(
        "report.memory_supported_directions",
        REPORT,
        "memory_record.supported_directions_to_preserve should cover runner_pivot_policy supported markers.",
        "memory_supported_directions",
    ),
    _rule(
        "report.framework_comparison",
        FATAL,
        "framework_comparison must contain before, after, expected_effect, and rollback_trigger.",
        "framework_comparison",
    ),
    _rule(
        "report.prediction_structure",
        FATAL,
        "prediction must contain non-empty expected_fixed_task_classes, list-valued risk_task_classes, numeric expected_metric_delta, confidence, and falsification_window.",
        "prediction_structure",
    ),
    _rule(
        "report.prediction_window",
        REPORT,
        "prediction.falsification_window should name an evaluable next summary, frontier, regression, or rerun window.",
        "prediction_window",
    ),
    _rule(
        "report.change_evaluation_misses",
        REPORT,
        "memory_record.failed_directions_to_avoid should reference top change_evaluation_digest.miss_classes.",
        "change_evaluation_misses",
    ),
    _rule(
        "report.change_evaluation_risks",
        REPORT,
        "prediction.risk_task_classes should reference top change_evaluation_digest.risk_classes.",
        "change_evaluation_risks",
    ),
    _rule(
        "report.prediction_evidence",
        REPORT,
        "prediction.expected_fixed_task_classes should reference a concrete packet failure, evaluation, rejected-update, prior-lesson, or mission label.",
        "prediction_evidence",
    ),
    _rule(
        "report.mission_selection",
        REPORT,
        "The final report should reference the packet's single selected mission candidate.",
        "mission_selection",
    ),
    _rule(
        "report.mission_scope",
        FATAL,
        "Changed files must stay within the selected mission candidate allowed_edit_paths plus tests.",
        "mission_scope",
    ),
    _rule(
        "report.implementation_scope",
        FATAL,
        "implementation_scope must use valid fields and exactly report structural files from the isolated diff.",
        "implementation_scope",
    ),
    _rule(
        "report.implementation_layer",
        REPORT,
        "implementation_scope.primary_layer or component_type should match the allowed layer values computed from the isolated diff.",
        "implementation_layer",
    ),
    _rule(
        "report.leaderboard_compliance",
        FATAL,
        "leaderboard_compliance must explicitly preserve every required Harbor, Worker, benchmark-integrity, resource, attempt, artifact, and submit gate.",
        "leaderboard_compliance",
    ),
    _rule(
        "report.external_research",
        FATAL,
        "external_research must have valid shape and obey packet source, focus, fetch, and skip-reason requirements.",
        "external_research",
    ),
    # Diff/update/review findings. These are typed for review.json consistency.
    _rule("patch.no_files_changed", FATAL, "No tracked files changed.", "patch_review", scope=REVIEW_SCOPE),
    _rule("patch.forbidden_path", FATAL, "A forbidden benchmark/runtime path changed.", "patch_review", scope=REVIEW_SCOPE),
    _rule("patch.outside_allowed_roots", FATAL, "A changed path is outside allowed edit roots.", "patch_review", scope=REVIEW_SCOPE),
    _rule("patch.memory_only", FATAL, "The patch changes only memory or log artifacts.", "patch_review", scope=REVIEW_SCOPE),
    _rule("patch.gate_weakening", FATAL, "The patch appears to weaken verifier, regression, or submit gates.", "patch_review", scope=REVIEW_SCOPE),
    _rule("patch.nested_agent", FATAL, "The production diff creates a nested or external coding-agent launch path.", "patch_review", scope=REVIEW_SCOPE),
    _rule("patch.task_id_hardcoding", FATAL, "The production diff hardcodes TerminalBench task ids.", "patch_review", scope=REVIEW_SCOPE),
    _rule("update.dry_run", FATAL, "The update was a dry run and made no accepted change.", "update_flow", scope=REVIEW_SCOPE),
    _rule("update.dirty_baseline", FATAL, "The update was blocked by a dirty baseline.", "update_flow", scope=REVIEW_SCOPE),
    _rule("update.binary_delta", FATAL, "The isolated Codex delta contains an unreviewable binary change.", "update_flow", scope=REVIEW_SCOPE),
    _rule("update.rollback_applied", FATAL, "A rejected Codex delta was rolled back.", "update_flow", scope=REVIEW_SCOPE),
    _rule("update.rollback_failed", FATAL, "A rejected Codex delta could not be rolled back.", "update_flow", scope=REVIEW_SCOPE),
    _rule("update.skip_all_candidates_covered", FATAL, "Codex execution was skipped because every mission candidate is already covered.", "update_flow", scope=REVIEW_SCOPE),
    _rule("validation.host_commands_required", FATAL, "Changed files require host validation commands.", "host_validation", scope=REVIEW_SCOPE),
    _rule("validation.host_command_failed", FATAL, "A host validation command failed.", "host_validation", scope=REVIEW_SCOPE),
    _rule("codex.exec_failed", FATAL, "codex exec returned a non-zero exit code.", "codex_execution", scope=REVIEW_SCOPE),
    _rule("codex.provider_failure", FATAL, "Codex events indicate an upstream provider or authentication failure.", "codex_execution", scope=REVIEW_SCOPE),
    _rule("codex.nested_agent", FATAL, "The Codex updater attempted to launch a nested or external coding agent.", "codex_execution", scope=REVIEW_SCOPE),
    _rule(
        INTERNAL_RULE_ID,
        FATAL,
        "Internal contract error: a finding used an unregistered rule id.",
        "internal_contract",
        scope=REVIEW_SCOPE,
    ),
)


_RULES_BY_ID = {rule.id: rule for rule in _RULES}
if len(_RULES_BY_ID) != len(_RULES):
    raise ValueError("Report contract rule ids must be unique")


def report_contract_rules(*, scope: str | None = None) -> tuple[ReportContractRule, ...]:
    """Return registered rules, optionally restricted by scope."""

    if scope is None:
        return _RULES
    return tuple(rule for rule in _RULES if rule.scope == scope)


def final_report_rules() -> tuple[ReportContractRule, ...]:
    return report_contract_rules(scope=FINAL_REPORT_SCOPE)


def rule_for_id(rule_id: str) -> ReportContractRule | None:
    return _RULES_BY_ID.get(str(rule_id))


def violation(rule_id: str, reason: str) -> ReportViolation:
    """Create a typed finding, failing closed for an unknown rule id."""

    rule = rule_for_id(rule_id)
    if rule is None:
        internal = _RULES_BY_ID[INTERNAL_RULE_ID]
        return ReportViolation(
            rule_id=internal.id,
            severity=internal.severity,
            reason=(
                f"internal contract error: unregistered rule id {rule_id!r}: "
                f"{str(reason)}"
            ),
        )
    return ReportViolation(
        rule_id=rule.id,
        severity=rule.severity,
        reason=str(reason),
    )


def untyped_violation(reason: str) -> ReportViolation:
    """Fail closed when legacy/external code supplies only a reason string."""

    return violation(INTERNAL_RULE_ID, str(reason))


def is_blocking_violation(finding: ReportViolation) -> bool:
    return finding.severity == FATAL


_REPORT_LAYERS_BY_COMPONENT: dict[str, tuple[str, ...]] = {
    "worker_loop": (
        "planning",
        "tool",
        "recovery",
        "verification",
        "context",
        "adapter",
        "architecture",
    ),
    "harbor_adapter": ("adapter", "harbor_integration"),
    "scoring": ("verification",),
    "trajectory": ("memory", "verification"),
    "task_catalog": ("harbor_integration",),
    "prompt": ("prompt",),
    "tool_schema": ("tool",),
    "tool_impl": ("tool",),
    "planning": ("planning",),
    "context_compaction": ("context",),
    "recovery": ("recovery",),
    "entrypoint": ("planning",),
    "verification": ("verification",),
    "skill_loading": ("tool",),
    "campaign_loop": ("orchestration",),
    "memory": ("memory",),
    "coupling": ("memory",),
    "submit_gate": ("orchestration",),
    "codex_update": ("orchestration",),
    "codex_packet": ("orchestration",),
    "diff_review": ("orchestration",),
    "mission_debug": ("orchestration",),
    "meta_updater": ("orchestration",),
    "campaign_runner": ("orchestration",),
    "regression_gate": ("verification", "orchestration"),
    "script_orchestration": ("orchestration",),
    "config": ("config",),
}


def valid_primary_layers_for_changed_files(changed_files: list[str]) -> list[str]:
    """Return the exact report-layer choices allowed by the isolated diff."""

    if not changed_files:
        return []
    delta = classify_component_delta(changed_files)
    primary_layer = str(delta.get("primary_layer") or "")
    layers = _REPORT_LAYERS_BY_COMPONENT.get(primary_layer)
    if layers:
        return list(layers)
    if primary_layer in {"tests", "docs", "repo_guidance", "other"}:
        return ["other"]
    return ["other"]
