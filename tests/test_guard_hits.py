import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from hl.guard_hits import build_guard_hit_benefit_audit, source_guard_policy_catalog


def test_source_guard_policy_catalog_includes_repeated_and_classifier_policies():
    catalog = source_guard_policy_catalog(Path("."))

    assert "repeated_dependency_timeout_path_guard" in catalog
    assert "repeated_vm_service_readiness_timeout_path_guard" in catalog
    assert "vm_service_readiness_timeout_phase" in catalog
    assert "simulation_timeout_phase" not in catalog


def test_guard_hit_audit_classifies_zero_hit_zero_benefit_and_beneficial(tmp_path):
    analysis_root = tmp_path / "analysis"
    summary_dir_1 = analysis_root / "campaign-a" / "summary_001"
    summary_dir_2 = analysis_root / "campaign-a" / "summary_002"
    summary_dir_1.mkdir(parents=True)
    summary_dir_2.mkdir(parents=True)
    _write_summary(
        summary_dir_1 / "summary.json",
        summary_id="summary_001",
        policies={
            "manual_dependency_download_guard": {
                "description": "blocks manual downloads",
                "count": 1,
                "tasks": ["task-a"],
                "examples": [{"task_id": "task-a", "command": "curl https://example.invalid/pkg"}],
            },
            "background_package_command_guard": {
                "description": "blocks detached package commands",
                "count": 1,
                "tasks": ["task-b"],
                "examples": [{"task_id": "task-b", "command": "apt-get install foo &"}],
            },
        },
        trajectory={
            "task-a": {"policy_counts": {"manual_dependency_download_guard": 1}},
            "task-b": {"policy_counts": {"background_package_command_guard": 1}},
        },
    )
    _write_summary(
        summary_dir_2 / "summary.json",
        summary_id="summary_002",
        policies={
            "manual_dependency_download_guard": {
                "description": "blocks manual downloads",
                "count": 2,
                "tasks": ["task-a"],
                "examples": [{"task_id": "task-a", "command": "wget https://example.invalid/pkg"}],
            }
        },
        trajectory={"task-a": {"policy_counts": {"manual_dependency_download_guard": 2}}},
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "campaign_id": "campaign-a",
                "iteration_summaries": [
                    {"summary_id": "summary_001", "trial_ids": ["task-a__1", "task-b__1"]},
                    {"summary_id": "summary_002", "trial_ids": ["task-a__2"]},
                ],
                "task_results": [
                    {"trial_id": "task-a__1", "task_id": "task-a", "status": "failed", "score": 0.0},
                    {"trial_id": "task-b__1", "task_id": "task-b", "status": "failed", "score": 0.0},
                    {"trial_id": "task-a__2", "task_id": "task-a", "status": "passed", "score": 1.0},
                ],
            }
        )
    )

    audit = build_guard_hit_benefit_audit(
        repo_root=Path("."),
        analysis_root=analysis_root,
        campaign_summary_path=campaign_path,
        campaign_id="campaign-a",
    )

    assert "manual_dependency_download_guard" in audit["beneficial_guards"]
    assert audit["policies"]["manual_dependency_download_guard"]["hit_count"] == 3
    assert audit["policies"]["manual_dependency_download_guard"][
        "fail_to_pass_transition_count"
    ] == 1
    assert "background_package_command_guard" in audit["hit_zero_benefit_guards"]
    assert "repeated_vm_service_readiness_timeout_path_guard" in audit["zero_hit_guards"]
    assert audit["analysis_summary_count"] == 2


def test_guard_hit_cli_json_is_machine_readable(tmp_path):
    analysis_root = tmp_path / "analysis"
    summary_dir = analysis_root / "campaign-a" / "summary_001"
    summary_dir.mkdir(parents=True)
    _write_summary(
        summary_dir / "summary.json",
        summary_id="summary_001",
        policies={"background_package_command_guard": {"count": 1, "tasks": ["task-a"]}},
        trajectory={"task-a": {"policy_counts": {"background_package_command_guard": 1}}},
    )
    summary_payload = json.loads((summary_dir / "summary.json").read_text())
    summary_payload["policy_coverage"]["policies"]["background_package_command_guard"][
        "examples"
    ] = [{"task_id": "task-a", "command": "echo sk-testsecret1234567890"}]
    (summary_dir / "summary.json").write_text(json.dumps(summary_payload))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_guard_hits.py",
            "--json",
            "--analysis-root",
            str(analysis_root),
            "--campaign-id",
            "campaign-a",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == 1
    assert payload["classification_counts"]["hit_zero_benefit"] >= 1
    assert "sk-testsecret1234567890" not in completed.stdout
    assert "[REDACTED_SECRET]" in completed.stdout


def _write_summary(path, *, summary_id, policies, trajectory):
    path.write_text(
        json.dumps(
            {
                "campaign_id": "campaign-a",
                "summary_id": summary_id,
                "overall_score": 0.5,
                "policy_coverage": {"policies": policies},
                "trajectory_evidence": trajectory,
            }
        )
    )


_MISSING = object()


@pytest.mark.parametrize(
    ("prior_entries", "later_score", "pass_score", "expect_transition", "expected_best"),
    [
        ([_MISSING], 1.0, 1.0, False, None),
        ([None], 1.0, 1.0, False, None),
        (["bad"], 1.0, 1.0, False, None),
        (["NaN"], 1.0, 1.0, False, None),
        ([None, 1.0], 1.0, 1.0, False, None),
        ([0], 1.0, 1.0, True, 0.0),
        (["0"], 1.0, 1.0, True, 0.0),
        ([None, 0.25], 1.0, 1.0, True, 0.25),
        ([None, -0.5], 1.0, 1.0, True, -0.5),
        (["NaN", 0.25], 1.0, 1.0, True, 0.25),
        ([0.25, "NaN"], 1.0, 1.0, True, 0.25),
        ([0.25, 1.0, None], 1.0, 1.0, True, 1.0),
        ([0.25], 0.75, 0.75, True, 0.25),
        (["inf"], 1.0, 1.0, False, None),
        (["-inf"], 1.0, 1.0, True, float("-inf")),
        ([0.25, "inf"], 1.0, 1.0, True, float("inf")),
        ([False], 1.0, 1.0, True, 0.0),
        ([True], 1.0, 1.0, False, None),
        ([], 1.0, 1.0, False, None),
        ([1.0, 1.0], 1.0, 1.0, False, None),
        ([0.0], "NaN", 1.0, False, None),
        ([0.0], None, 1.0, False, None),
    ],
)
def test_guard_hit_audit_prior_score_comparability_table(
    tmp_path,
    prior_entries,
    later_score,
    pass_score,
    expect_transition,
    expected_best,
):
    analysis_root = tmp_path / "analysis"
    summary_dir = analysis_root / "campaign-a" / "summary_002"
    summary_dir.mkdir(parents=True)
    guard_name = "manual_dependency_download_guard"
    _write_summary(
        summary_dir / "summary.json",
        summary_id="summary_002",
        policies={guard_name: {"count": 1, "tasks": ["task-a"]}},
        trajectory={"task-a": {"policy_counts": {guard_name: 1}}},
    )

    task_results = []
    for idx, score_val in enumerate(prior_entries):
        row = {"trial_id": f"task-a__p{idx}", "task_id": "task-a", "status": "failed"}
        if score_val is not _MISSING:
            row["score"] = score_val
        task_results.append(row)
    later_row = {"trial_id": "task-a__later", "task_id": "task-a", "status": "passed"}
    if later_score is not _MISSING:
        later_row["score"] = later_score
    task_results.append(later_row)

    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "campaign_id": "campaign-a",
                "iteration_summaries": [
                    {"summary_id": "summary_002", "trial_ids": ["task-a__later"]},
                ],
                "task_results": task_results,
            }
        )
    )

    audit = build_guard_hit_benefit_audit(
        repo_root=Path("."),
        analysis_root=analysis_root,
        campaign_summary_path=campaign_path,
        campaign_id="campaign-a",
        pass_score=pass_score,
    )
    record = audit["policies"][guard_name]
    if expect_transition:
        assert record["classification"] == "beneficial"
        assert record["fail_to_pass_transition_count"] == 1
        actual_best = record["fail_to_pass_examples"][0]["previous_best_score"]
        if math.isinf(expected_best):
            assert actual_best == expected_best
        else:
            assert actual_best == expected_best
    else:
        assert record["classification"] == "hit_zero_benefit"
        assert record["fail_to_pass_transition_count"] == 0


def test_guard_hit_audit_ignores_other_task_low_and_post_later_low(tmp_path):
    analysis_root = tmp_path / "analysis"
    summary_dir = analysis_root / "campaign-a" / "summary_001"
    summary_dir.mkdir(parents=True)
    guard_name = "manual_dependency_download_guard"
    _write_summary(
        summary_dir / "summary.json",
        summary_id="summary_001",
        policies={guard_name: {"count": 1, "tasks": ["task-a"]}},
        trajectory={"task-a": {"policy_counts": {guard_name: 1}}},
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "campaign_id": "campaign-a",
                "iteration_summaries": [
                    {"summary_id": "summary_001", "trial_ids": ["task-a__later"]},
                ],
                "task_results": [
                    {"trial_id": "task-b__0", "task_id": "task-b", "score": 0.0},
                    {"trial_id": "task-a__later", "task_id": "task-a", "score": 1.0},
                    {"trial_id": "task-a__after", "task_id": "task-a", "score": 0.0},
                ],
            }
        )
    )
    audit = build_guard_hit_benefit_audit(
        repo_root=Path("."),
        analysis_root=analysis_root,
        campaign_summary_path=campaign_path,
        campaign_id="campaign-a",
    )
    record = audit["policies"][guard_name]
    assert record["classification"] == "hit_zero_benefit"
    assert record["fail_to_pass_transition_count"] == 0


def test_guard_hit_cli_missing_prior_score_is_hit_zero_benefit(tmp_path):
    analysis_root = tmp_path / "analysis"
    summary_dir = analysis_root / "campaign-a" / "summary_001"
    summary_dir.mkdir(parents=True)
    guard_name = "background_package_command_guard"
    _write_summary(
        summary_dir / "summary.json",
        summary_id="summary_001",
        policies={guard_name: {"count": 1, "tasks": ["task-a"]}},
        trajectory={"task-a": {"policy_counts": {guard_name: 1}}},
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "campaign_id": "campaign-a",
                "iteration_summaries": [
                    {"summary_id": "summary_001", "trial_ids": ["task-a__1"]},
                ],
                "task_results": [
                    {"trial_id": "task-a__0", "task_id": "task-a", "status": "failed"},
                    {"trial_id": "task-a__1", "task_id": "task-a", "status": "passed", "score": 1.0},
                ],
            }
        )
    )

    completed_json = subprocess.run(
        [
            sys.executable,
            "scripts/audit_guard_hits.py",
            "--json",
            "--analysis-root",
            str(analysis_root),
            "--campaign-summary",
            str(campaign_path),
            "--campaign-id",
            "campaign-a",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed_json.stdout)
    assert guard_name in payload["hit_zero_benefit_guards"]
    assert payload["policies"][guard_name]["fail_to_pass_transition_count"] == 0

    completed_text = subprocess.run(
        [
            sys.executable,
            "scripts/audit_guard_hits.py",
            "--analysis-root",
            str(analysis_root),
            "--campaign-summary",
            str(campaign_path),
            "--campaign-id",
            "campaign-a",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Hit but zero strict benefit:" in completed_text.stdout
    assert guard_name in completed_text.stdout

