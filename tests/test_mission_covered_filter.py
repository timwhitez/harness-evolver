"""T5: mission selection filters already-covered mechanism signatures."""

import json
from pathlib import Path

from meta.codex_update import CodexUpdateEngine
from meta.mechanism_coverage import deterministic_worker_policy_coverage
from meta.missions import MissionFeatureCandidate, MissionPlanner
from meta.packager import CodexWorkPacket, WorkPacketBuilder
from tests.test_missions_debug import campaign_summary


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_NOOP = (
    ROOT
    / "trials"
    / "diffs"
    / "codex_packet_20260709_123630"
    / "codex_update_packet.json"
)


def _attributed_summary():
    summary = campaign_summary()
    summary["task_results"][1].update(
        {
            "failure_category": "environment_start_timeout",
            "affected_components": ["bench/harbor", "bench/network_environment"],
            "timeout_phase": "environment_start",
            "infra_error_detected": True,
            "score_exclusion_reason": "infrastructure_error",
        }
    )
    summary["task_results"][2].update(
        {
            "failure_category": "harbor_environment_error",
            "affected_components": ["bench/harbor", "bench/network_environment"],
            "infra_error_detected": True,
            "score_exclusion_reason": "infrastructure_error",
        }
    )
    summary["task_results"][3].update(
        {
            "failure_category": "verifier_mismatch",
            "affected_components": ["verification/checks", "harness/tools/verify"],
        }
    )
    return summary


def test_covered_signature_filters_matching_candidate():
    summary = _attributed_summary()
    packet = MissionPlanner().from_campaign_summary(
        summary,
        covered_mechanism_signatures=["verifier_mismatch"],
    )
    feature_ids = [feature.id for feature in packet.feature_candidates]
    assert "mission-attributed-verifier-mismatch" not in feature_ids
    # Non-covered candidates survive.
    assert "mission-attributed-environment-start-timeout" in feature_ids
    assert (
        "mission-attributed-verifier-mismatch"
        in packet.evidence_summary["filtered_covered_candidate_ids"]
    )
    assert packet.evidence_summary["covered_mechanism_signatures"] == ["verifier_mismatch"]


def test_all_candidates_covered_removes_active_candidates_and_records_skip():
    summary = _attributed_summary()
    covered = [
        "environment_start_timeout",
        "harbor_environment_error",
        "verifier_mismatch",
    ]
    packet = MissionPlanner().from_campaign_summary(
        summary,
        covered_mechanism_signatures=covered,
    )
    assert packet.feature_candidates == []
    assert packet.candidate_audit
    assert packet.evidence_summary.get("all_candidates_covered") is True
    skip = packet.evidence_summary["skip_codex_update"]
    assert skip["covered_candidate_ids"]
    assert "already covered" in skip["reason"]


def test_no_covered_signatures_is_backward_compatible():
    summary = _attributed_summary()
    packet = MissionPlanner().from_campaign_summary(summary)
    feature_ids = [feature.id for feature in packet.feature_candidates]
    assert "mission-attributed-verifier-mismatch" in feature_ids
    assert packet.evidence_summary["filtered_covered_candidate_ids"] == []


def test_current_policy_filters_mechanisms_from_recent_no_diff_campaign():
    coverage = deterministic_worker_policy_coverage(ROOT)
    signatures = {entry["signature"] for entry in coverage}
    expected = {
        "tokenized_output_file_contract",
        "dataset_shard_generalization_contract",
        "deliverable_size_cap_contract",
        "cross_arch_toolchain_pivot_mechanism",
        "terminal_environment_unavailable_after_dependency_loop_mechanism",
        "verifier_runtime_prepare_timeout",
    }
    assert expected <= signatures

    candidate_ids = [
        "mission-attributed-tokenized-output-file-contract-tokenized-output-file-contract",
        "mission-attributed-terminal-environment-unavailable-after-dependency-loop-terminal-environment-unavailable-after-dependency-loop-mechanism",
        "mission-attributed-verifier-runtime-prepare-timeout-cross-arch-toolchain-pivot-mechanism",
        "mission-attributed-dataset-shard-generalization-contract-deliverable-size-cap-contract-dataset-shard-generalization-contract",
        "mission-attributed-cross-arch-toolchain-pivot-mechanism-cross-arch-toolchain-pivot-mechanism",
    ]
    candidates = [
        MissionFeatureCandidate(
            id=candidate_id,
            title="covered candidate",
            rationale="already implemented policy candidate",
            success_signal="no duplicate update is launched",
        )
        for candidate_id in candidate_ids
    ]

    kept, filtered = MissionPlanner().filter_covered_candidates(
        candidates,
        sorted(signatures),
    )

    assert kept == []
    assert filtered == candidate_ids


def test_historical_noop_packet_is_automatically_all_covered():
    data = json.loads(HISTORICAL_NOOP.read_text())
    builder = WorkPacketBuilder(repo_root=ROOT, memory_path=ROOT / "trials")

    mission = builder.prepare_mission_debug(data["mission_debug"])

    assert mission["feature_candidates"] == []
    assert len(mission["candidate_audit"]) == 17
    assert len(mission["evidence_summary"]["filtered_covered_candidate_ids"]) == 17
    assert mission["evidence_summary"]["all_candidates_covered"] is True


def test_historical_noop_packet_skips_before_codex_exec(tmp_path):
    data = json.loads(HISTORICAL_NOOP.read_text())
    builder = WorkPacketBuilder(repo_root=ROOT, memory_path=ROOT / "trials")
    data["mission_debug"] = builder.prepare_mission_debug(data["mission_debug"])
    packet = CodexWorkPacket.model_validate(data)
    engine = CodexUpdateEngine(
        repo_root=ROOT,
        codex_bin=str(tmp_path / "must-not-execute"),
        events_dir=tmp_path / "diffs",
    )
    engine.packet_builder.build = lambda **_kwargs: packet

    result = engine.run_update(failures=[], current_harness={})

    assert result.review.accepted is False
    assert result.review.reason_details[0]["rule_id"] == (
        "update.skip_all_candidates_covered"
    )
    assert "already covered" in result.review.reasons[0]
    assert Path(result.events_path).read_text() == ""


def test_work_packet_selects_exactly_one_uncovered_candidate(tmp_path):
    mission = {
        "evidence_summary": {},
        "feature_candidates": [
            {
                "id": "mission-attributed-novel-parser-gap",
                "title": "Repair novel parser gap",
                "rationale": "novel parser evidence",
                "target_tasks": ["task-a"],
                "affected_components": ["harness/tools/verify"],
                "allowed_edit_paths": ["harness", "tests"],
                "validation_contracts": ["contract-unit-suite"],
                "success_signal": "parser gap is covered",
                "priority": "P1",
            },
            {
                "id": "mission-attributed-novel-context-gap",
                "title": "Repair novel context gap",
                "rationale": "novel context evidence",
                "target_tasks": ["task-b"],
                "affected_components": ["harness/context"],
                "allowed_edit_paths": ["harness", "tests"],
                "validation_contracts": ["contract-unit-suite"],
                "success_signal": "context gap is covered",
                "priority": "P2",
            },
        ],
    }
    builder = WorkPacketBuilder(repo_root=tmp_path, memory_path=tmp_path / "trials")

    selected = builder.prepare_mission_debug(mission)
    budget = builder._report_value_budget(
        failure_pattern_digest={},
        mission_debug=selected,
        rejected_update_buffer=[],
    )

    assert len(selected["feature_candidates"]) == 1
    assert len(selected["candidate_audit"]) == 2
    assert selected["evidence_summary"]["selected_candidate_id"] == (
        "mission-attributed-novel-parser-gap"
    )
    assert budget["selected_feature_candidate_id"] == (
        "mission-attributed-novel-parser-gap"
    )
    assert budget["attributed_feature_candidate_ids"] == [
        "mission-attributed-novel-parser-gap"
    ]
    assert budget["valid_primary_layers"] == []
