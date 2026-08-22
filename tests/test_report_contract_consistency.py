"""Registry, packet, validator reachability, and per-rule fixtures."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

from hl.types import TaskDifficulty, TaskDomain, TrialResult, TrialStatus
from meta import report_contract
from meta.codex_update import CodexUpdateEngine
from meta.packager import WorkPacketBuilder


def _failed_trial() -> TrialResult:
    return TrialResult(
        trial_id="t1",
        task_id="task-a",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.EASY,
        status=TrialStatus.FAILED,
        score=0.0,
        verified=True,
        verifier_output="reward 0",
        trajectory=[{"type": "tool_call", "tool": "bash", "error": "file not found"}],
    )


def _valid_report() -> dict:
    candidate_id = "mission-attributed-entrypoint-miss"
    return {
        "status": "edited",
        "summary": f"selected {candidate_id} for a bounded worker edit",
        "changed_files": ["bench/agent.py"],
        "validation_commands": ["pytest tests/ -v"],
        "skipped_validation_reason": "",
        "component_type": "adapter",
        "strategy_confidence": "medium",
        "loophole_review": ["reviewed regression risk"],
        "loophole_fixes": ["kept verifier and regression gates"],
        "generalization": {
            "problem_class": "entrypoint_miss",
            "applies_to": [candidate_id, "software engineering tasks"],
            "anti_overfit_checks": ["no task-id branch"],
            "why_not_task_specific": "the edit is mechanism-based",
        },
        "cross_round_evidence": {
            "used": True,
            "recent_summary_ids": ["summary_001"],
            "dominant_patterns": ["entrypoint_miss"],
            "selected_problem_class": candidate_id,
            "why_this_slice_generalizes": "entrypoint discovery is reusable",
        },
        "memory_record": {
            "concise": "bounded entrypoint policy edit",
            "detailed": f"selected {candidate_id} and preserved prior gates",
            "failed_directions_to_avoid": [],
            "supported_directions_to_preserve": [],
        },
        "framework_comparison": {
            "before": "old entrypoint policy",
            "after": "bounded entrypoint policy",
            "expected_effect": "fewer entrypoint misses",
            "rollback_trigger": "next regression loses a solved task",
        },
        "prediction": {
            "expected_fixed_task_classes": [candidate_id, "entrypoint_miss"],
            "risk_task_classes": ["regression_gate"],
            "expected_metric_delta": 0.1,
            "confidence": "medium",
            "falsification_window": "next comparable summary or frontier update",
        },
        "implementation_scope": {
            "primary_layer": "adapter",
            "architectural_change_considered": True,
            "structural_files_changed": ["bench/agent.py"],
            "why_prompt_only_is_sufficient": "not prompt-only",
        },
        "leaderboard_compliance": {
            "harbor_official_harness_preserved": True,
            "self_owned_worker_preserved": True,
            "benchmark_integrity_preserved": True,
            "timeouts_resources_unchanged": True,
            "submit_gate_preserved": True,
            "official_dataset_preserved": True,
            "five_attempts_per_task_preserved": True,
            "no_prohibited_terminal_bench_access": True,
            "upload_artifacts_trace_preserved": True,
        },
        "external_research": {
            "used": False,
            "sources": [],
            "fetches": [],
            "reason": "local evidence is sufficient",
            "impact": "",
        },
    }


def _valid_context() -> report_contract.ReportValidationContext:
    candidate_id = "mission-attributed-entrypoint-miss"
    return report_contract.ReportValidationContext(
        changed_files=["bench/agent.py"],
        required_validation_commands=["pytest tests/ -v"],
        failure_pattern_digest={
            "dominant_pattern": {
                "failure_category": "entrypoint_miss",
                "affected_components": ["bench/agent"],
            },
            "patterns": [{"failure_category": "entrypoint_miss"}],
        },
        mission_debug={
            "evidence_summary": {"selected_candidate_id": candidate_id},
            "feature_candidates": [
                {
                    "id": candidate_id,
                    "failure_category": "entrypoint_miss",
                    "allowed_edit_paths": ["bench", "tests"],
                }
            ],
        },
    )


def _negative_fixture(
    rule_id: str,
) -> tuple[dict, report_contract.ReportValidationContext]:
    report = copy.deepcopy(_valid_report())
    context = copy.deepcopy(_valid_context())
    if rule_id == "report.present":
        report = {}
    elif rule_id == "report.status":
        report["status"] = "invalid"
    elif rule_id == "report.status_changed_files":
        report["status"] = "noop"
    elif rule_id == "report.basic_fields":
        report["summary"] = ""
    elif rule_id == "report.changed_files":
        report["changed_files"] = []
    elif rule_id == "report.validation_shape":
        report["validation_commands"] = "pytest tests/ -v"
    elif rule_id == "report.required_validation":
        report["validation_commands"] = []
        context = replace(
            context,
            required_validation_commands=["pytest tests/ -v", "python missing_check.py"],
        )
    elif rule_id == "report.loophole_review":
        report["loophole_review"] = []
    elif rule_id == "report.generalization_structure":
        report.pop("generalization")
    elif rule_id == "report.generalization_evidence":
        report["generalization"]["problem_class"] = "generic tasks"
        report["generalization"]["applies_to"] = ["generic tasks"]
    elif rule_id == "report.cross_round_structure":
        report.pop("cross_round_evidence")
    elif rule_id == "report.cross_round_patterns":
        report["cross_round_evidence"]["dominant_patterns"] = ["generic pattern"]
    elif rule_id == "report.cross_round_problem_class":
        report["cross_round_evidence"]["selected_problem_class"] = "generic class"
    elif rule_id == "report.memory_structure":
        report.pop("memory_record")
    elif rule_id == "report.memory_failed_directions":
        context = replace(
            context,
            rejected_update_buffer=[
                {"packet_id": "codex_packet_rejected", "failure_class": "bad slice"}
            ],
        )
    elif rule_id == "report.memory_supported_directions":
        context = replace(
            context,
            runner_pivot_policy={
                "supported": [
                    {
                        "packet_id": "codex_packet_good",
                        "failure_class": "entrypoint_miss",
                        "component_layer": "adapter",
                    }
                ]
            },
        )
    elif rule_id == "report.framework_comparison":
        report.pop("framework_comparison")
    elif rule_id == "report.prediction_structure":
        report.pop("prediction")
    elif rule_id == "report.prediction_window":
        report["prediction"]["falsification_window"] = "eventually"
    elif rule_id == "report.change_evaluation_misses":
        context = replace(
            context,
            change_evaluation_digest={
                "miss_classes": [{"class": "missed_regression"}]
            },
        )
    elif rule_id == "report.change_evaluation_risks":
        context = replace(
            context,
            change_evaluation_digest={
                "risk_classes": [{"class": "risky_adapter"}]
            },
        )
    elif rule_id == "report.prediction_evidence":
        report["prediction"]["expected_fixed_task_classes"] = ["generic tasks"]
    elif rule_id == "report.mission_selection":
        context.mission_debug = {
            "evidence_summary": {
                "selected_candidate_id": "mission-attributed-mission-only-gap"
            },
            "feature_candidates": [
                {
                    "id": "mission-attributed-mission-only-gap",
                    "failure_category": "mission_only_gap",
                    "allowed_edit_paths": ["bench", "tests"],
                }
            ],
        }
    elif rule_id == "report.mission_scope":
        context.mission_debug["feature_candidates"][0]["allowed_edit_paths"] = [
            "harness",
            "tests",
        ]
    elif rule_id == "report.implementation_scope":
        report.pop("implementation_scope")
    elif rule_id == "report.implementation_layer":
        report["implementation_scope"]["primary_layer"] = "config"
        report["component_type"] = "config"
    elif rule_id == "report.leaderboard_compliance":
        report["leaderboard_compliance"]["submit_gate_preserved"] = False
    elif rule_id == "report.external_research":
        report.pop("external_research")
    else:  # pragma: no cover - guarded by the registry equality assertion below
        raise AssertionError(f"missing negative fixture for {rule_id}")
    return report, context


def test_registry_ids_bindings_and_packet_rendering_are_single_source(tmp_path):
    rules = report_contract.final_report_rules()
    ids = [rule.id for rule in rules]
    bindings = [rule.binding for rule in rules]
    assert len(ids) == len(set(ids))
    assert len(bindings) == len(set(bindings))

    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    assert set(bindings) == set(engine._report_validator_bindings())

    packet = WorkPacketBuilder(repo_root=tmp_path).build(
        failures=[_failed_trial()],
        current_harness={"version": "x"},
    )
    rendered = json.dumps(packet.report_contract_rules)
    for rule in rules:
        assert rule.id in rendered
        assert rule.description in rendered
        assert rule.binding in rendered


def test_every_final_report_rule_has_passing_and_failing_fixture(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    rule_ids = {rule.id for rule in report_contract.final_report_rules()}
    fixture_ids = set()
    for rule_id in sorted(rule_ids):
        passing = engine.validate_report_rule(rule_id, _valid_report(), _valid_context())
        assert passing == [], (rule_id, passing)

        report, context = _negative_fixture(rule_id)
        failing = engine.validate_report_rule(rule_id, report, context)
        assert failing, rule_id
        assert {finding.rule_id for finding in failing} == {rule_id}, failing
        fixture_ids.add(rule_id)
    assert fixture_ids == rule_ids


def test_packet_advises_context_complete_report_lint(tmp_path):
    packet = WorkPacketBuilder(repo_root=tmp_path).build(
        failures=[_failed_trial()],
        current_harness={"version": "x"},
    )
    self_check = packet.report_contract_rules.get("self_check", "")
    assert "scripts/report_lint.py" in self_check
    assert "--packet-dir" in self_check
