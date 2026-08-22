import json
import subprocess
import sys
from pathlib import Path

import yaml

from hl.guard_convergence import build_guard_convergence_audit


def test_guard_convergence_audit_reports_fixed_eval_and_guard_budget(tmp_path):
    report_path = tmp_path / "fixed-eval-report.json"
    report_path.write_text(
        json.dumps(
            {
                "score_history": [
                    {"score": 0.50},
                    {"score": 0.51},
                    {"score": 0.515},
                    {"score": 0.52},
                ]
            }
        )
    )

    audit = build_guard_convergence_audit(
        repo_root=Path("."),
        trials_config_path=Path("config/trials.yaml"),
        fixed_eval_report=report_path,
    )

    assert audit["ready_for_guard_reduction"] is True
    assert audit["fixed_eval"]["task_count"] == 10
    assert audit["fixed_eval"]["domain_count"] >= 4
    assert audit["fixed_eval"]["known_pass_count"] > 0
    assert audit["fixed_eval"]["known_fail_count"] > 0
    assert audit["guard_budget"]["baseline_total_guard_surface"] == 167
    assert audit["guard_budget"]["target_total_guard_surface"] == 116
    assert audit["guard_counts"]["total_guard_surface"] <= 167
    assert audit["convergence"]["score_gate_met"] is True
    assert audit["convergence"]["plateau_met"] is True


def test_guard_convergence_audit_reads_campaign_state_report(tmp_path):
    report_path = tmp_path / "campaign_state.json"
    report_path.write_text(
        json.dumps(
            {
                "campaign_id": "guard-convergence-example",
                "summaries": [
                    {"summary_id": "summary_001", "overall_score": 0.5},
                    {"summary_id": "summary_002", "overall_score": 0.505},
                    {"summary_id": "summary_003", "overall_score": 0.51},
                    {"summary_id": "summary_004", "overall_score": 0.515},
                ],
            }
        )
    )

    audit = build_guard_convergence_audit(
        repo_root=Path("."),
        trials_config_path=Path("config/trials.yaml"),
        fixed_eval_report=report_path,
    )

    assert audit["convergence"]["score_history"] == [0.5, 0.505, 0.51, 0.515]
    assert audit["convergence"]["plateau_met"] is True


def test_guard_convergence_audit_discovers_fixed_eval_state_reports(tmp_path):
    config = yaml.safe_load(Path("config/trials.yaml").read_text())
    trials_path = tmp_path / "trials.yaml"
    trials_path.write_text(yaml.safe_dump(config, sort_keys=False))
    summaries_dir = tmp_path / "trials" / "summaries"
    summaries_dir.mkdir(parents=True)
    tasks = [
        item["task_id"] for item in config["guard_convergence"]["fixed_eval"]["tasks"]
    ]
    for index, score in enumerate([0.5, 0.505, 0.51, 0.515], start=1):
        (summaries_dir / f"guard-convergence-t{index}_campaign_state.json").write_text(
            json.dumps(
                {
                    "campaign_id": f"guard-convergence-t{index}",
                    "tasks": tasks,
                    "summaries": [
                        {
                            "summary_id": "summary_001",
                            "overall_score": score,
                            "recorded_at": f"2026-06-23T00:0{index}:00",
                        }
                    ],
                }
            )
        )

    audit = build_guard_convergence_audit(
        repo_root=tmp_path,
        trials_config_path=trials_path,
    )

    assert audit["convergence"]["score_history"] == [0.5, 0.505, 0.51, 0.515]
    assert len(audit["fixed_eval_state_reports"]) == 4
    assert audit["convergence"]["plateau_met"] is True


def test_guard_convergence_cli_json_is_machine_readable(tmp_path):
    report_path = tmp_path / "fixed-eval-report.json"
    report_path.write_text(json.dumps({"score_history": [{"score": 0.5}]}))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_guard_convergence.py",
            "--json",
            "--fixed-eval-report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == 1
    assert payload["fixed_eval"]["valid"] is True
    assert payload["guard_budget"]["valid"] is True
    assert "sk-" not in completed.stdout


def test_run_campaign_guard_convergence_dry_run_uses_fixed_eval(tmp_path):
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        """
roles:
  worker:
    provider: openai_compatible
    base_url: http://127.0.0.1:8000/v1
    api_key_env: TEST_API_KEY
    model: test-model
    reasoning:
      effort: none
"""
    )
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_API_KEY=\n")

    config = yaml.safe_load(Path("config/trials.yaml").read_text())
    trials_path = tmp_path / "trials.yaml"
    trials_path.write_text(yaml.safe_dump(config, sort_keys=False))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_campaign.py",
            "--guard-convergence-eval",
            "--dry-run",
            "--models-config",
            str(models_path),
            "--env-file",
            str(env_path),
            "--trials-config",
            str(trials_path),
            "--memory-path",
            str(tmp_path / "trials"),
            "--campaign-id",
            "guard-eval-test",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    fixed_tasks = [
        item["task_id"]
        for item in config["guard_convergence"]["fixed_eval"]["tasks"]
    ]
    assert payload["tasks"] == fixed_tasks
    assert payload["pending_tasks"] == fixed_tasks
    assert payload["task_rotation"]["mode"] == "fixed_guard_convergence_eval"
    assert payload["guard_convergence"]["valid"] is True
    assert payload["goal"]["goal"]["guard_convergence_score_floor"] == 0.5
    assert payload["goal"]["goal"]["guard_budget_target_total"] == 116
