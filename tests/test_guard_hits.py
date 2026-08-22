import json
import subprocess
import sys
from pathlib import Path

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
