"""Tests for regression detection and validation contracts."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

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

    @pytest.mark.parametrize(
        (
            "status",
            "score",
            "verified",
            "expected_exit_code",
            "expected_message",
            "expected_failures",
        ),
        [
            (
                TrialStatus.FAILED,
                0.0,
                True,
                1,
                "Regressions detected:\n- pending-task",
                1,
            ),
            (
                TrialStatus.PASSED,
                1.0,
                True,
                0,
                "Regression lane smoke passed for 1 snapshot(s).",
                0,
            ),
        ],
    )
    def test_regression_check_enforces_pending_snapshot_validation(
        self,
        tmp_path,
        monkeypatch,
        capsys,
        status,
        score,
        verified,
        expected_exit_code,
        expected_message,
        expected_failures,
    ):
        from scripts import regression_check as regression_check_module

        memory_path = tmp_path / "trials"
        memory = FileSystemMemory(base_path=str(memory_path))
        models_path = tmp_path / "models.yaml"
        models_path.write_text(
            """
roles:
  pending_worker:
    provider: openai_compatible
    base_url: https://api.example/v1
    api_key_env: TEST_API_KEY
    model: pending-test-model
    reasoning:
      effort: none
    max_output_tokens: 8000
"""
        )
        model_config = {
            "provider": "openai_compatible",
            "base_url": "https://api.example/v1",
            "model": "pending-test-model",
            "reasoning_effort": "none",
            "max_output_tokens": "8000",
        }
        scope = model_scope_from_config(model_config)
        memory.save_regression(
            "pending-task",
            RegressionSnapshot(
                task_id="pending-task",
                harness_version="0.1.0",
                validation_status="pending",
                source_summary_id="summary-1",
                model_scope=scope,
            ),
        )

        class FakeHarborRunner:
            def __init__(self, *args, **kwargs):
                pass

            def build_command(self, *args, **kwargs):
                return object()

            def run_task(self, *, task_id, agent_config, **kwargs):
                return TrialResult(
                    trial_id="pending-validation",
                    task_id=task_id,
                    task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                    task_difficulty=TaskDifficulty.EASY,
                    status=status,
                    score=score,
                    verified=verified,
                    model_used="pending-test-model",
                    metadata={"model_config": model_config},
                )

        monkeypatch.setattr("bench.harbor.HarborRunner", FakeHarborRunner)
        monkeypatch.setenv("TEST_API_KEY", "test-secret")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "regression_check.py",
                "--task",
                "pending-task",
                "--snapshot-status",
                "pending",
                "--memory-path",
                str(memory_path),
                "--models-config",
                str(models_path),
                "--worker-role",
                "pending_worker",
            ],
        )

        assert regression_check_module.main() == expected_exit_code
        assert expected_message in capsys.readouterr().out
        persisted = memory.get_regression_snapshot(
            "pending-task",
            model_scope=scope,
        )
        assert persisted is not None
        assert persisted.validation_status == "pending"
        assert persisted.regression_runs == 1
        assert persisted.regression_failures == expected_failures

    @pytest.mark.parametrize("dry_run", [False, True])
    @pytest.mark.parametrize("mode", ["corrupt_only", "mixed_valid_and_corrupt", "specific_task"])
    def test_regression_check_aborts_on_incomplete_snapshots_before_runner(
        self,
        tmp_path,
        monkeypatch,
        capsys,
        dry_run,
        mode,
    ):
        from scripts import regression_check as regression_check_module

        memory_path = tmp_path / "trials"
        memory = FileSystemMemory(base_path=str(memory_path))
        model_config = {
            "provider": "openai_compatible",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "max",
            "max_output_tokens": "8000",
        }
        scope = model_scope_from_config(model_config)
        if mode in {"mixed_valid_and_corrupt", "specific_task"}:
            memory.save_regression(
                "valid-task",
                RegressionSnapshot(
                    task_id="valid-task",
                    harness_version="0.1.0",
                    validation_status="stable",
                    model_scope=scope,
                ),
            )
        corrupt_path = memory._regression_path("broken-task", model_scope=scope)
        corrupt_path.parent.mkdir(parents=True, exist_ok=True)
        corrupt_path.write_text('{"task_id": "broken-task", "token": "SECRET_MARKER_163"')

        calls = {"runner": 0, "api_key": 0, "project_test": 0}

        class SpyHarborRunner:
            def __init__(self, *args, **kwargs):
                calls["runner"] += 1

        def spy_require_key(*args, **kwargs):
            calls["api_key"] += 1

        def spy_subprocess_run(*args, **kwargs):
            calls["project_test"] += 1
            raise AssertionError("project test must not run when snapshots are incomplete")

        monkeypatch.setattr("bench.harbor.HarborRunner", SpyHarborRunner)
        monkeypatch.setattr("scripts.run_trial._require_worker_api_key", spy_require_key)
        monkeypatch.setattr(subprocess, "run", spy_subprocess_run)
        argv = [
            "regression_check.py",
            "--lane",
            "full",
            "--memory-path",
            str(memory_path),
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
        ]
        if mode == "specific_task":
            argv.extend(["--task", "valid-task"])
        if dry_run:
            argv.append("--dry-run")
        monkeypatch.setattr(sys, "argv", argv)

        exit_code = regression_check_module.main()
        captured = capsys.readouterr()

        assert exit_code != 0
        assert "incomplete" in captured.err.lower()
        assert "not compared" in captured.err.lower()
        assert "broken-task.json" in captured.err
        assert "SECRET_MARKER_163" not in captured.err
        assert "SECRET_MARKER_163" not in captured.out
        assert "No regression snapshots found." not in captured.out
        assert calls == {"runner": 0, "api_key": 0, "project_test": 0}


class TestUnifiedRegressionIdentityIssue170:
    """Issue #170: unified dataset-scoped task identity and port-preserving route identity."""

    @staticmethod
    def _create_task(dataset_dir, *relative_parts: str) -> None:
        task_dir = dataset_dir.joinpath(*relative_parts)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.toml").write_text('[metadata]\ncategory = "git"\ndifficulty = "easy"\n')

    @staticmethod
    def _write_models_yaml(path, *, base_url: str) -> None:
        path.write_text(
            f"""
roles:
  cpa_worker:
    provider: openai_compatible
    base_url: {base_url}
    api_key_env: TEST_API_KEY
    model: deepseek-flash
    reasoning:
      effort: high
    max_output_tokens: 8000
"""
        )

    def test_harbor_parse_job_dir_and_cli_pass_on_equivalent_task_and_port_representations(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        """Same confirmed task & port-bearing route differing only in representation passes CLI."""
        from bench.harbor import HarborRunner
        from scripts import regression_check as regression_check_module

        dataset_dir = tmp_path / "terminal-bench"
        self._create_task(dataset_dir, "fix-git")
        self._create_task(dataset_dir, "vulnerable-secret")

        memory_path = tmp_path / "trials"
        memory = FileSystemMemory(base_path=str(memory_path))
        models_path = tmp_path / "models.yaml"
        self._write_models_yaml(models_path, base_url="http://127.0.0.1:24635/v1")

        # Snapshot saved using namespaced task_id and port-bearing URL (equivalent to CLI route)
        scope = model_scope_from_config(
            {
                "provider": "openai_compatible",
                "base_url": "http://127.0.0.1:24635/v1",
                "model": "deepseek-flash",
                "reasoning_effort": "high",
                "max_output_tokens": "8000",
            }
        )
        assert "base_url_host=127.0.0.1:24635" in scope
        memory.save_regression(
            "terminal-bench/fix-git",
            RegressionSnapshot(
                task_id="terminal-bench/fix-git",
                harness_version="0.1.0",
                validation_status="stable",
                model_scope=scope,
            ),
        )
        # Second snapshot uses bare task_id while receipt will return namespaced task_id
        memory.save_regression(
            "vulnerable-secret",
            RegressionSnapshot(
                task_id="vulnerable-secret",
                harness_version="0.1.0",
                validation_status="stable",
                model_scope=scope,
            ),
        )

        # Build real synthetic Harbor job dirs parsed by the real HarborRunner.parse_job_dir!
        jobs_root = tmp_path / "synthetic_jobs"

        class SyntheticHarborRunner(HarborRunner):
            def run_task(self, *, task_id, agent_config, job_name=None, jobs_dir=None, **kwargs):
                cmd = self.build_command(task_id, agent_config, job_name=job_name, jobs_dir=jobs_dir)
                # Namespaced snapshot in terminal-bench must pass bare filter to Harbor
                if task_id == "terminal-bench/fix-git":
                    assert "--include-task-name" in cmd.argv
                    idx = cmd.argv.index("--include-task-name")
                    assert cmd.argv[idx + 1] == "fix-git"
                    job_dir = jobs_root / "job_fix_git"
                    job_dir.mkdir(parents=True, exist_ok=True)
                    (job_dir / "result.json").write_text(
                        json.dumps(
                            {
                                "trial_results": [
                                    {
                                        "trial_name": "fix-git__a1b2c3d",
                                        "task_name": "fix-git",
                                        "task_id": {
                                            "name": "fix-git",
                                            "path": str(dataset_dir / "fix-git"),
                                        },
                                        "verifier_result": {"rewards": {"reward": 1.0}},
                                    }
                                ]
                            }
                        )
                    )
                    return self.parse_job_dir(
                        job_dir,
                        task_id=task_id,
                        returncode=0,
                        agent_config=agent_config,
                    )
                # Bare snapshot receiving namespaced Harbor task_name
                job_dir = jobs_root / "job_vuln"
                job_dir.mkdir(parents=True, exist_ok=True)
                (job_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "trial_results": [
                                {
                                    "trial_name": "vulnerable-secret__e4f5g6h",
                                    "task_name": "terminal-bench/vulnerable-secret",
                                    "verifier_result": {"rewards": {"reward": 1.0}},
                                }
                            ]
                        }
                    )
                )
                return self.parse_job_dir(
                    job_dir,
                    task_id=task_id,
                    returncode=0,
                    agent_config=agent_config,
                )

        monkeypatch.setattr("bench.harbor.HarborRunner", SyntheticHarborRunner)
        monkeypatch.setenv("TEST_API_KEY", "test-secret")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "regression_check.py",
                "--path",
                str(dataset_dir),
                "--memory-path",
                str(memory_path),
                "--models-config",
                str(models_path),
                "--worker-role",
                "cpa_worker",
            ],
        )

        exit_code = regression_check_module.main()
        captured = capsys.readouterr()
        assert exit_code == 0, captured.out + captured.err
        assert "Regression lane smoke passed for 2 snapshot(s)." in captured.out

        # Also verify --task bare filter selects namespaced snapshot uniquely within dataset
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "regression_check.py",
                "--task",
                "fix-git",
                "--path",
                str(dataset_dir),
                "--memory-path",
                str(memory_path),
                "--models-config",
                str(models_path),
                "--worker-role",
                "cpa_worker",
            ],
        )
        assert regression_check_module.main() == 0
        assert "passed for 1 snapshot(s)." in capsys.readouterr().out

    def test_default_port_and_redacted_port_representations_match(self, tmp_path):
        """https://Gateway.Example:443/v1 == https://gateway.example/v1 and host+port == host:port."""
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        dataset_dir = tmp_path / "terminal-bench"
        self._create_task(dataset_dir, "fix-git")

        snap_https = RegressionSnapshot(
            task_id="terminal-bench/fix-git",
            harness_version="0.1.0",
            validation_status="stable",
            model_scope=model_scope_from_config(
                {
                    "provider": "openai_compatible",
                    "base_url": "https://Gateway.Example:443/v1",
                    "model": "deepseek-flash",
                }
            ),
        )
        trial_https = TrialResult(
            trial_id="t_https",
            task_id="fix-git",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
            score=1.0,
            verified=True,
            metadata={
                "model_config": {
                    "provider": "openai_compatible",
                    "base_url_host": "gateway.example",
                    "model": "deepseek-flash",
                }
            },
        )
        res_https = memory.diagnose_snapshot_validation(
            snap_https, trial_https, dataset_path=dataset_dir
        )
        assert res_https.passed is True

        snap_port = RegressionSnapshot(
            task_id="terminal-bench/fix-git",
            harness_version="0.1.0",
            validation_status="stable",
            model_scope=model_scope_from_config(
                {
                    "provider": "openai_compatible",
                    "base_url": "http://127.0.0.1:24635/v1",
                    "model": "deepseek-flash",
                }
            ),
        )
        trial_redacted_port = TrialResult(
            trial_id="t_port",
            task_id="fix-git",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
            score=1.0,
            verified=True,
            metadata={
                "model_config": {
                    "provider": "openai_compatible",
                    "base_url_host": "127.0.0.1",
                    "base_url_port": 24635,
                    "base_url_scheme": "http",
                    "model": "deepseek-flash",
                }
            },
        )
        res_port = memory.diagnose_snapshot_validation(
            snap_port, trial_redacted_port, dataset_path=dataset_dir
        )
        assert res_port.passed is True

    @pytest.mark.parametrize(
        ("case", "expected_kind", "expected_absent_kind"),
        [
            ("ambiguous_bare_cli_filter", "Task identity ambiguous", "model_route_mismatch"),
            ("ambiguous_bare_receipt", "task_identity_ambiguous", "model_route_mismatch"),
            ("wrong_dataset_task_mismatch", "task_identity_mismatch", "model_route_mismatch"),
            ("missing_task_identity", "task_identity_missing", "model_route_mismatch"),
            ("different_port_route_mismatch", "model_route_mismatch", "task_identity_mismatch"),
            ("portless_vs_port_route_mismatch", "model_route_mismatch", "task_identity_mismatch"),
            ("missing_route_identity", "model_route_missing", "task_identity_mismatch"),
            ("failing_verifier", "verifier_failed", "task_identity_mismatch"),
        ],
    )
    def test_cli_rejects_ambiguous_mismatched_missing_and_failing_receipts_with_distinct_errors(
        self,
        tmp_path,
        monkeypatch,
        capsys,
        case,
        expected_kind,
        expected_absent_kind,
    ):
        """CLI rejects ambiguous bare names, route/port mismatches, missing IDs, and failed verifiers."""
        from scripts import regression_check as regression_check_module

        memory_path = tmp_path / "trials"
        memory = FileSystemMemory(base_path=str(memory_path))
        models_path = tmp_path / "models.yaml"
        self._write_models_yaml(models_path, base_url="http://127.0.0.1:24635/v1")

        route_cfg = {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:24635/v1",
            "model": "deepseek-flash",
            "reasoning_effort": "high",
            "max_output_tokens": "8000",
        }
        scope = model_scope_from_config(route_cfg)

        if case in {"ambiguous_bare_cli_filter", "ambiguous_bare_receipt"}:
            dataset_dir = tmp_path / "multi-datasets"
            self._create_task(dataset_dir, "dataset-a", "fix-git")
            self._create_task(dataset_dir, "dataset-b", "fix-git")
            snap_task_id = "dataset-a/fix-git"
        else:
            dataset_dir = tmp_path / "terminal-bench"
            self._create_task(dataset_dir, "fix-git")
            snap_task_id = "terminal-bench/fix-git"

        memory.save_regression(
            snap_task_id,
            RegressionSnapshot(
                task_id=snap_task_id,
                harness_version="0.1.0",
                validation_status="stable",
                model_scope=scope,
            ),
        )

        def _make_receipt(task_id: str) -> TrialResult:
            if case == "ambiguous_bare_receipt":
                return TrialResult(
                    trial_id="t_ambig",
                    task_id="fix-git",
                    task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                    task_difficulty=TaskDifficulty.EASY,
                    status=TrialStatus.PASSED,
                    score=1.0,
                    verified=True,
                    metadata={"model_config": route_cfg},
                )
            if case == "wrong_dataset_task_mismatch":
                return TrialResult(
                    trial_id="t_wrong_ds",
                    task_id="other-dataset/fix-git",
                    task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                    task_difficulty=TaskDifficulty.EASY,
                    status=TrialStatus.PASSED,
                    score=1.0,
                    verified=True,
                    metadata={"model_config": route_cfg},
                )
            if case == "missing_task_identity":
                return TrialResult(
                    trial_id="t_no_task",
                    task_id="",
                    task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                    task_difficulty=TaskDifficulty.EASY,
                    status=TrialStatus.PASSED,
                    score=1.0,
                    verified=True,
                    metadata={"model_config": route_cfg},
                )
            if case == "different_port_route_mismatch":
                wrong_route = {**route_cfg, "base_url": "http://127.0.0.1:24636/v1"}
                return TrialResult(
                    trial_id="t_wrong_port",
                    task_id="fix-git",
                    task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                    task_difficulty=TaskDifficulty.EASY,
                    status=TrialStatus.PASSED,
                    score=1.0,
                    verified=True,
                    metadata={"model_config": wrong_route},
                )
            if case == "portless_vs_port_route_mismatch":
                portless_route = {
                    "provider": "openai_compatible",
                    "base_url_host": "127.0.0.1",
                    "model": "deepseek-flash",
                    "reasoning_effort": "high",
                    "max_output_tokens": "8000",
                }
                return TrialResult(
                    trial_id="t_portless",
                    task_id="fix-git",
                    task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                    task_difficulty=TaskDifficulty.EASY,
                    status=TrialStatus.PASSED,
                    score=1.0,
                    verified=True,
                    metadata={"model_config": portless_route},
                )
            if case == "missing_route_identity":
                return TrialResult(
                    trial_id="t_no_route",
                    task_id="fix-git",
                    task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                    task_difficulty=TaskDifficulty.EASY,
                    status=TrialStatus.PASSED,
                    score=1.0,
                    verified=True,
                    metadata={},
                )
            return TrialResult(
                trial_id="t_fail_verif",
                task_id="fix-git",
                task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                task_difficulty=TaskDifficulty.EASY,
                status=TrialStatus.FAILED,
                score=0.0,
                verified=True,
                metadata={"model_config": route_cfg},
            )

        class FakeRunner:
            def __init__(self, *args, **kwargs):
                pass

            def build_command(self, *args, **kwargs):
                return object()

            def run_task(self, *, task_id, agent_config, **kwargs):
                return _make_receipt(task_id)

        monkeypatch.setattr("bench.harbor.HarborRunner", FakeRunner)
        monkeypatch.setenv("TEST_API_KEY", "test-secret")
        argv = [
            "regression_check.py",
            "--path",
            str(dataset_dir),
            "--memory-path",
            str(memory_path),
            "--models-config",
            str(models_path),
            "--worker-role",
            "cpa_worker",
        ]
        if case == "ambiguous_bare_cli_filter":
            argv.extend(["--task", "fix-git"])
        monkeypatch.setattr(sys, "argv", argv)

        exit_code = regression_check_module.main()
        captured = capsys.readouterr()
        combined_output = captured.out + "\n" + captured.err
        assert exit_code == 1
        assert expected_kind in combined_output
        assert expected_absent_kind not in combined_output


