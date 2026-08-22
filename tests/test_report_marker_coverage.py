"""T1 regression: mechanism-signature problem classes must satisfy marker gates.

The historical Codex packets 023213 and 180021 were both rolled back because the
``cross_round_evidence.selected_problem_class`` gate demanded the report echo a
*generic* digest label (``timeout``) while Codex correctly named a *specific,
operable* mechanism signature (``missing_output_artifact_contract``). The gate
punished the better choice. These tests pin the fixed behavior: a concrete
mechanism signature drawn from the report's own dominant_patterns or from a
mission feature candidate is accepted as covering the evidence, while an
unrelated invented label is still rejected.
"""

from meta import report_contract
from meta.codex_update import CodexUpdateEngine


def _engine(tmp_path):
    return CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "events")


def _digest_timeout_only() -> dict:
    return {
        "dominant_pattern": {
            "failure_category": "timeout",
            "affected_components": [],
        },
        "patterns": [{"failure_category": "timeout"}],
    }


def _mission_debug_missing_artifact() -> dict:
    return {
        "feature_candidates": [
            {
                "id": "mission-attributed-missing-output-artifact-contract",
                "failure_category": "timeout",
                "allowed_edit_paths": ["crates/hl-worker-core/"],
            }
        ]
    }


def _rule_reasons(
    engine: CodexUpdateEngine,
    rule_id: str,
    final_report: dict,
    *,
    mission_debug: dict,
) -> list[str]:
    context = report_contract.ReportValidationContext(
        changed_files=["crates/hl-worker-core/src/main.rs"],
        failure_pattern_digest=_digest_timeout_only(),
        mission_debug=mission_debug,
    )
    return [
        violation.reason
        for violation in engine.validate_report_rule(rule_id, final_report, context)
    ]


def test_selected_problem_class_accepts_mission_mechanism_signature(tmp_path):
    engine = _engine(tmp_path)
    final_report = {
        "cross_round_evidence": {
            "used": True,
            "recent_summary_ids": ["summary_001"],
            "dominant_patterns": ["missing_output_artifact_contract"],
            "selected_problem_class": (
                "missing_output_artifact_contract artifact-first semantic preflight"
            ),
            "why_this_slice_generalizes": "reusable across artifact-producing tasks",
        }
    }
    reasons = _rule_reasons(
        engine,
        "report.cross_round_problem_class",
        final_report,
        mission_debug=_mission_debug_missing_artifact(),
    )
    assert not any("selected_problem_class must reference" in r for r in reasons), reasons


def test_selected_problem_class_accepts_report_dominant_pattern_signature(tmp_path):
    engine = _engine(tmp_path)
    final_report = {
        "cross_round_evidence": {
            "used": True,
            "recent_summary_ids": ["summary_001"],
            "dominant_patterns": ["timeout", "missing_output_artifact_contract"],
            "selected_problem_class": "missing_output_artifact_contract",
            "why_this_slice_generalizes": "reusable across artifact-producing tasks",
        }
    }
    reasons = _rule_reasons(
        engine,
        "report.cross_round_problem_class",
        final_report,
        mission_debug={},
    )
    assert not any("selected_problem_class must reference" in r for r in reasons), reasons


def test_selected_problem_class_still_rejects_unrelated_label(tmp_path):
    engine = _engine(tmp_path)
    final_report = {
        "cross_round_evidence": {
            "used": True,
            "recent_summary_ids": ["summary_001"],
            "dominant_patterns": ["timeout"],
            "selected_problem_class": "totally_unrelated_invented_label",
            "why_this_slice_generalizes": "reusable across tasks",
        }
    }
    reasons = _rule_reasons(
        engine,
        "report.cross_round_problem_class",
        final_report,
        mission_debug=_mission_debug_missing_artifact(),
    )
    assert any("selected_problem_class must reference" in r for r in reasons), reasons


def test_generalization_accepts_mission_mechanism_signature(tmp_path):
    engine = _engine(tmp_path)
    final_report = {
        "generalization": {
            "problem_class": "missing_output_artifact_contract",
            "applies_to": ["artifact-producing software_engineering tasks"],
            "anti_overfit_checks": ["does not branch on task id"],
            "why_not_task_specific": "mechanism is shared across tasks",
        }
    }
    reasons = _rule_reasons(
        engine,
        "report.generalization_evidence",
        final_report,
        mission_debug=_mission_debug_missing_artifact(),
    )
    assert not any("must reference a concrete" in r for r in reasons), reasons
