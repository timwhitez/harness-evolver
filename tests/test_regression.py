"""Tests for regression detection and validation contracts."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta

from hl.memory import FileSystemMemory
from hl.model_scope import model_scope_from_config
from hl.types import TrialResult, TrialStatus, RegressionSnapshot, TaskDomain, TaskDifficulty
from scripts.regression_check import (
    _run_regression_snapshots,
    _select_regression_snapshots,
    record_snapshot_from_trial,
)


class TestRegressionDetection:
    def test_adaptive_regression_selection_treats_cap_as_audit_only(self):
        cooldown_slow_flaky = RegressionSnapshot(
            task_id="aaa-cooldown-slow-flaky",
            harness_version="0.1.0",
            validation_status="stable",
            regression_transient_failures=9,
            regression_failures=4,
            regression_cooldown_until=datetime.now() + timedelta(days=1),
            last_regression_wall_time_seconds=500,
        )
        fast_stable = RegressionSnapshot(
            task_id="bbb-fast-stable",
            harness_version="0.1.0",
            validation_status="stable",
            last_regression_wall_time_seconds=5,
        )

        selected = _select_regression_snapshots(
            [fast_stable, cooldown_slow_flaky],
            argparse.Namespace(lane="smoke", cap=1, selection_policy="adaptive"),
        )

        assert [snapshot.task_id for snapshot in selected] == [
            "aaa-cooldown-slow-flaky",
            "bbb-fast-stable",
        ]

    def test_regression_lane_does_not_apply_implicit_snapshot_count_cap(self):
        snapshots = [
            RegressionSnapshot(task_id=f"task-{index}", harness_version="0.1.0")
            for index in range(5)
        ]

        selected = _select_regression_snapshots(
            snapshots,
            argparse.Namespace(lane="smoke", cap=None, selection_policy="stable-order"),
        )

        assert [snapshot.task_id for snapshot in selected] == [
            "task-0",
            "task-1",
            "task-2",
            "task-3",
            "task-4",
        ]

    def test_explicit_regression_cap_is_audit_only_not_selection_gate(self):
        snapshots = [
            RegressionSnapshot(task_id=f"task-{index}", harness_version="0.1.0")
            for index in range(5)
        ]

        selected = _select_regression_snapshots(
            snapshots,
            argparse.Namespace(lane="full", cap=2, selection_policy="stable-order"),
        )

        assert [snapshot.task_id for snapshot in selected] == [
            "task-0",
            "task-1",
            "task-2",
            "task-3",
            "task-4",
        ]

    def test_regression_runner_passes_harbor_timeout_as_audit_only(self):
        snapshots = [RegressionSnapshot(task_id="task-a", harness_version="0.1.0")]

        class Runner:
            def __init__(self):
                self.calls = []

            def run_task(self, **kwargs):
                self.calls.append(kwargs)
                return TrialResult(
                    trial_id="trial-a",
                    task_id=kwargs["task_id"],
                    task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                    task_difficulty=TaskDifficulty.EASY,
                    status=TrialStatus.PASSED,
                    score=1.0,
                    verified=True,
                    metadata={
                        "outer_harbor_timeout_stop_condition": False,
                        "outer_harbor_timeout_seconds_audit_only": kwargs[
                            "timeout_audit"
                        ],
                    },
                )

        runner = Runner()
        results = _run_regression_snapshots(
            snapshots=snapshots,
            runner=runner,
            agent_config={},
            timeout_audit=7,
            jobs_dir="jobs",
            run_stamp=123,
            task_concurrency=1,
        )

        assert len(results) == 1
        assert runner.calls[0]["timeout_audit"] == 7
        assert "timeout" not in runner.calls[0]
        assert results[0][1].metadata["outer_harbor_timeout_stop_condition"] is False
        assert results[0][1].metadata["outer_harbor_timeout_seconds_audit_only"] == 7

    def test_regression_check_accepts_campaign_forwarded_worker_flags(self, tmp_path):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/regression_check.py",
                "--dry-run",
                "--n-attempts",
                "5",
                "--tool-timeout-seconds",
                "66",
                "--max-turns-audit",
                "9",
                "--memory-path",
                str(tmp_path / "trials"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert "No regression snapshots found." in completed.stdout

    def test_regression_check_holdout_mode_splits_snapshots(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        scope = model_scope_from_config(
            {
                "provider": "openai_compatible",
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-v4-flash",
                "reasoning_effort": "max",
                "max_output_tokens": "8000",
            }
        )
        tasks = [f"task-{i}" for i in range(40)]
        for task in tasks:
            memory.save_regression(
                task,
                RegressionSnapshot(
                    task_id=task,
                    harness_version="0.1.0",
                    validation_status="stable",
                    model_scope=scope,
                ),
            )

        def selected_tasks(holdout_mode: str) -> set[str]:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/regression_check.py",
                    "--dry-run",
                    "--snapshot-status",
                    "stable",
                    "--holdout-mode",
                    holdout_mode,
                    "--holdout-fraction",
                    "0.25",
                    "--holdout-seed",
                    "0",
                    "--memory-path",
                    str(tmp_path / "trials"),
                    "--model",
                    "deepseek-v4-flash",
                    "--provider",
                    "openai_compatible",
                    "--base-url",
                    "https://api.deepseek.com/v1",
                    "--reasoning-effort",
                    "max",
                    "--max-output-tokens",
                    "8000",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(completed.stdout)
            return {row["task_id"] for row in payload["selected_snapshots"]}

        held_in = selected_tasks("held_in")
        held_out = selected_tasks("held_out")

        assert held_in
        assert held_out
        # Disjoint and complete: held-in + held-out == all solved tasks.
        assert held_in.isdisjoint(held_out)
        assert held_in | held_out == set(tasks)

    def test_regression_check_holdout_matches_partition_helper(self, tmp_path):
        from hl.regression_split import is_holdout_task

        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        scope = model_scope_from_config(
            {
                "provider": "openai_compatible",
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-v4-flash",
                "reasoning_effort": "max",
                "max_output_tokens": "8000",
            }
        )
        tasks = [f"task-{i}" for i in range(30)]
        for task in tasks:
            memory.save_regression(
                task,
                RegressionSnapshot(
                    task_id=task,
                    harness_version="0.1.0",
                    validation_status="stable",
                    model_scope=scope,
                ),
            )

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/regression_check.py",
                "--dry-run",
                "--holdout-mode",
                "held_out",
                "--holdout-fraction",
                "0.3",
                "--holdout-seed",
                "5",
                "--memory-path",
                str(tmp_path / "trials"),
                "--model",
                "deepseek-v4-flash",
                "--provider",
                "openai_compatible",
                "--base-url",
                "https://api.deepseek.com/v1",
                "--reasoning-effort",
                "max",
                "--max-output-tokens",
                "8000",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        selected = {row["task_id"] for row in payload["selected_snapshots"]}
        expected = {
            task for task in tasks if is_holdout_task(task, fraction=0.3, seed=5)
        }
        assert selected == expected

    def test_regression_when_previously_solved_fails(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))

        # Save a golden snapshot
        memory.save_regression("task_solved", RegressionSnapshot(
            task_id="task_solved",
            harness_version="0.1.0",
            required_assertions=["test must pass"],
            solved_at=datetime.now(),
        ))

        # New trial fails
        failed_trial = TrialResult(
            trial_id="t_new",
            task_id="task_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.FAILED,
            score=0.0,
        )

        is_regression = memory.check_regression("task_solved", failed_trial)
        assert is_regression is True

    def test_no_regression_when_still_solved(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))

        memory.save_regression("task_solved", RegressionSnapshot(
            task_id="task_solved",
            harness_version="0.1.0",
        ))

        passed_trial = TrialResult(
            trial_id="t_new",
            task_id="task_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
            score=1.0,
        )

        is_regression = memory.check_regression("task_solved", passed_trial)
        assert is_regression is False

    def test_no_snapshot_no_regression(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))

        failed_trial = TrialResult(
            trial_id="t_new",
            task_id="never_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.FAILED,
        )

        is_regression = memory.check_regression("never_solved", failed_trial)
        assert is_regression is False  # Wasn't previously solved, so not a regression

    def test_record_snapshot_from_verified_passed_trial(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path))
        passed_trial = TrialResult(
            trial_id="t_passed",
            task_id="task_solved",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
            score=1.0,
            verified=True,
            verifier_output='{"reward": 1.0}',
            model_used="deepseek-v4-flash",
            metadata={
                "model_config": {
                    "provider": "openai_compatible",
                    "base_url_host": "api.deepseek.com",
                    "model": "deepseek-v4-flash",
                    "reasoning_effort": "max",
                    "max_output_tokens": "8000",
                }
            },
        )
        memory.record_trial(passed_trial)

        snapshot = record_snapshot_from_trial(memory, "t_passed")

        assert snapshot.task_id == "task_solved"
        assert "score >= 1.0" in snapshot.required_assertions[0]
        assert snapshot.source_trial_id == "t_passed"
        assert snapshot.validation_status == "stable"
        assert snapshot.scope_config["model"] == "deepseek-v4-flash"
        assert memory.get_regression_snapshot(
            "task_solved",
            model_scope=snapshot.model_scope,
        ) is not None

    def test_regression_dry_run_only_uses_same_model_scope(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        memory.save_regression(
            "pro-task",
            RegressionSnapshot(
                task_id="pro-task",
                harness_version="0.1.0",
                validation_status="stable",
                model_scope=model_scope_from_config(
                    {
                        "provider": "openai_compatible",
                        "base_url": "https://api.deepseek.com/v1",
                        "model": "deepseek-v4-pro",
                        "reasoning_effort": "max",
                        "max_output_tokens": "8000",
                    }
                ),
            ),
        )
        memory.save_regression(
            "flash-task",
            RegressionSnapshot(
                task_id="flash-task",
                harness_version="0.1.0",
                validation_status="stable",
                model_scope=model_scope_from_config(
                    {
                        "provider": "openai_compatible",
                        "base_url": "https://api.deepseek.com/v1",
                        "model": "deepseek-v4-flash",
                        "reasoning_effort": "max",
                        "max_output_tokens": "8000",
                    }
                ),
            ),
        )
        models_path = tmp_path / "models.yaml"
        models_path.write_text(
            """
roles:
  worker_deepseek:
    provider: openai_compatible
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-v4-flash
    reasoning:
      effort: max
    max_output_tokens: 8000
"""
        )

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/regression_check.py",
                "--dry-run",
                "--memory-path",
                str(tmp_path / "trials"),
                "--models-config",
                str(models_path),
                "--worker-role",
                "worker_deepseek",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        payload = json.loads(completed.stdout)
        assert len(payload["commands"]) == 1
        assert "--include-task-name flash-task" in payload["commands"][0]
        assert "pro-task" not in completed.stdout

    def test_regression_dry_run_with_no_same_model_snapshots_is_empty(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        memory.save_regression(
            "pro-task",
            RegressionSnapshot(
                task_id="pro-task",
                harness_version="0.1.0",
                validation_status="stable",
                model_scope=model_scope_from_config(
                    {
                        "provider": "openai_compatible",
                        "base_url": "https://api.deepseek.com/v1",
                        "model": "deepseek-v4-pro",
                        "reasoning_effort": "max",
                        "max_output_tokens": "8000",
                    }
                ),
            ),
        )

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/regression_check.py",
                "--dry-run",
                "--memory-path",
                str(tmp_path / "trials"),
                "--model",
                "deepseek-v4-flash",
                "--provider",
                "openai_compatible",
                "--base-url",
                "https://api.deepseek.com/v1",
                "--reasoning-effort",
                "max",
                "--max-output-tokens",
                "8000",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert "No regression snapshots found." in completed.stdout

    def test_regression_check_can_dry_run_pending_snapshot(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        models_path = tmp_path / "models.yaml"
        models_path.write_text(
            """
roles:
  pending_worker:
    provider: openai_compatible
    base_url: http://127.0.0.1:8000/v1
    api_key_env: TEST_API_KEY
    model: pending-test-model
    reasoning:
      effort: none
    max_output_tokens: 8000
"""
        )
        scope = model_scope_from_config(
            {
                "provider": "openai_compatible",
                "base_url": "http://127.0.0.1:8000/v1",
                "model": "pending-test-model",
                "reasoning_effort": "none",
                "max_output_tokens": "8000",
            }
        )
        memory.save_regression(
            "stable-task",
            RegressionSnapshot(
                task_id="stable-task",
                harness_version="0.1.0",
                validation_status="stable",
                model_scope=scope,
            ),
        )
        memory.save_regression(
            "pending-task",
            RegressionSnapshot(
                task_id="pending-task",
                harness_version="0.1.0",
                validation_status="pending",
                source_summary_id="summary_1",
                model_scope=scope,
            ),
        )

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/regression_check.py",
                "--dry-run",
                "--memory-path",
                str(tmp_path / "trials"),
                "--snapshot-status",
                "pending",
                "--models-config",
                str(models_path),
                "--worker-role",
                "pending_worker",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        payload = json.loads(completed.stdout)
        assert payload["snapshot_status"] == "pending"
        assert len(payload["commands"]) == 1
        assert "--include-task-name pending-task" in payload["commands"][0]
        assert payload["skipped_snapshots"] == [
            {
                "task_id": "stable-task",
                "validation_status": "stable",
                "invalidation_reason": "",
                "model_scope": scope,
            }
        ]
