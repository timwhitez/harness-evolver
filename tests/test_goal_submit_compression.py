import json
import subprocess
import sys

from hl.compression import CompressionEngine
from hl.goals import GoalStore
from hl.submit import SubmitConfig, SubmitGate


def write_uploadable_job(job_dir, *, trial_name="task-a__1", reward=1.0, trajectory=True):
    trial_dir = job_dir / trial_name
    (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "trial_name": trial_name,
                        "task_name": "task-a",
                        "verifier_result": {"rewards": {"reward": reward}},
                    }
                ]
            }
        )
    )
    if trajectory:
        (trial_dir / "agent" / "trajectory.jsonl").write_text(
            json.dumps({"type": "tool_call", "tool": "bash", "args": {"command": "true"}})
            + "\n"
        )


def test_default_goal_store_is_in_memory_and_does_not_touch_trials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    store = GoalStore()

    assert store.path is None
    assert store.get_goal() is None
    assert not (tmp_path / "trials").exists()

    store.create_goal("local smoke", token_budget=10)

    assert store.get_goal() is not None
    assert not (tmp_path / "trials").exists()


def test_submit_once_is_idempotent_per_campaign(tmp_path):
    harbor_log = tmp_path / "harbor.log"
    harbor_bin = tmp_path / "harbor"
    harbor_bin.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$@\" >> \"$HARBOR_LOG\"\n"
        "exit 0\n"
    )
    harbor_bin.chmod(0o755)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    write_uploadable_job(job_dir)

    gate = SubmitGate(
        SubmitConfig(
            enabled=True,
            trigger_score=0.5,
            min_tasks_evaluated=1,
            min_attempts_per_task=1,
            require_full_regression=False,
            require_clean_git=False,
            require_no_uncommitted_harness_diff=False,
            harbor_upload=True,
        ),
        submissions_dir=tmp_path / "submissions",
        harbor_bin=str(harbor_bin),
    )

    import os

    old_log = os.environ.get("HARBOR_LOG")
    os.environ["HARBOR_LOG"] = str(harbor_log)
    try:
        first = gate.submit_once(
            campaign_id="camp1",
            best_job_dir=job_dir,
            score=0.9,
            tasks_evaluated=1,
            full_regression_passed=True,
        )
        second = gate.submit_once(
            campaign_id="camp1",
            best_job_dir=job_dir,
            score=0.9,
            tasks_evaluated=1,
            full_regression_passed=True,
        )
    finally:
        if old_log is None:
            os.environ.pop("HARBOR_LOG", None)
        else:
            os.environ["HARBOR_LOG"] = old_log

    assert first.eligible is True
    assert second.eligible is False
    assert "campaign already has a submit result" in second.reasons
    assert harbor_log.read_text().splitlines() == [f"auth status", f"upload {job_dir} --private"]


def test_goal_token_budget_is_per_iteration_by_default(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score", token_budget=10)
    goal = store.update_usage(worker_tokens={"input": 8, "output": 3})
    assert goal is not None
    assert goal.status == "active"
    assert goal.token_budget_scope == "iteration"
    assert goal.usage.total_tokens == 11
    assert goal.usage.last_iteration_tokens == 11
    assert goal.usage.token_budget_overruns == 1
    assert goal.remaining_tokens() == 10
    assert goal.latest_iteration_remaining_tokens() == 0


def test_wall_time_budget_is_opt_in(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score")
    goal = store.update_usage(harbor_wall_time_seconds=999999)

    assert goal is not None
    assert goal.status == "active"
    assert goal.wall_time_budget_seconds is None
    assert goal.remaining_wall_time() is None


def test_non_positive_goal_budgets_are_disabled_audit_fields(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    created = store.create_goal(
        "reach score",
        token_budget=0,
        wall_time_budget_seconds=-5,
    )

    assert created.token_budget is None
    assert created.wall_time_budget_seconds is None
    assert created.remaining_tokens() is None
    assert created.remaining_wall_time() is None


def test_update_budget_disables_non_positive_goal_budgets(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score", token_budget=10, wall_time_budget_seconds=5)

    updated = store.update_budget(token_budget=0, wall_time_budget_seconds=-5)

    assert updated.token_budget is None
    assert updated.wall_time_budget_seconds is None
    assert updated.remaining_tokens() is None
    assert updated.remaining_wall_time() is None


def test_campaign_scope_goal_budget_exhaustion_is_not_completion(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score", token_budget=10, token_budget_scope="campaign")
    goal = store.update_usage(worker_tokens={"input": 8, "output": 3})
    assert goal is not None
    assert goal.status == "active"
    assert "audit observation only" in goal.completion_reason
    assert "loops remain active" in goal.completion_reason
    assert goal.usage.token_budget_observations == 1
    assert goal.remaining_tokens() == 0


def test_budget_exhausted_goal_reopens_when_budget_is_raised(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score", token_budget=10, token_budget_scope="campaign")
    exhausted = store.update_usage(worker_tokens={"input": 11})
    assert exhausted is not None
    assert exhausted.status == "active"

    reopened = store.update_budget(token_budget=100)

    assert reopened.status == "active"
    assert reopened.token_budget == 100
    assert reopened.remaining_tokens() == 89


def test_wall_time_budget_exhausted_goal_reopens_when_budget_is_disabled(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score", wall_time_budget_seconds=5)
    exhausted = store.update_usage(harbor_wall_time_seconds=6)
    assert exhausted is not None
    assert exhausted.status == "active"
    assert "audit observation only" in exhausted.completion_reason
    assert "loops remain active" in exhausted.completion_reason
    assert exhausted.usage.wall_time_budget_observations == 1

    reopened = store.update_budget(wall_time_budget_seconds=None)

    assert reopened.status == "active"
    assert reopened.wall_time_budget_seconds is None
    assert reopened.remaining_wall_time() is None


def test_goal_read_normalizes_legacy_budget_to_active_state(tmp_path):
    path = tmp_path / "current.json"
    store = GoalStore(path)
    store.create_goal("reach score", token_budget=10, token_budget_scope="campaign")
    exhausted = store.update_usage(worker_tokens={"input": 11})
    assert exhausted is not None
    assert exhausted.status == "active"

    data = json.loads(path.read_text())
    data["status"] = "budget_exhausted"
    data["token_budget_scope"] = "iteration"
    path.write_text(json.dumps(data))

    normalized = store.get_goal()

    assert normalized is not None
    assert normalized.status == "active"
    assert "Legacy budget_exhausted status was normalized" in normalized.completion_reason


def test_goal_read_normalizes_legacy_limit_stopped_state(tmp_path):
    path = tmp_path / "current.json"
    path.write_text(
        json.dumps(
            {
                "objective": "reach score",
                "status": "stopped",
                "completion_reason": "wall time budget exhausted before score target",
                "token_budget_scope": "campaign",
            }
        )
    )
    store = GoalStore(path)

    normalized = store.get_goal()

    assert normalized is not None
    assert normalized.status == "active"
    assert "Legacy stopped status" in normalized.completion_reason
    assert "audit metadata only" in normalized.completion_reason


def test_goal_read_normalizes_legacy_limit_complete_state(tmp_path):
    path = tmp_path / "current.json"
    path.write_text(
        json.dumps(
            {
                "objective": "reach score",
                "status": "complete",
                "completion_reason": "token budget and round limit exhausted",
                "completed_at": "2026-06-20T00:00:00",
            }
        )
    )
    store = GoalStore(path)

    normalized = store.get_goal()

    assert normalized is not None
    assert normalized.status == "active"
    assert normalized.completed_at is None
    assert "Legacy complete status" in normalized.completion_reason
    assert "audit metadata only" in normalized.completion_reason


def test_goal_read_normalizes_non_explicit_complete_state(tmp_path):
    path = tmp_path / "current.json"
    path.write_text(
        json.dumps(
            {
                "objective": "reach score",
                "status": "complete",
                "completion_reason": "scheduler marked campaign complete",
                "completed_at": "2026-06-20T00:00:00",
            }
        )
    )
    store = GoalStore(path)

    normalized = store.get_goal()

    assert normalized is not None
    assert normalized.status == "active"
    assert normalized.completed_at is None
    assert "Non-explicit complete status" in normalized.completion_reason
    assert "may only stop on explicit user/operator completion" in normalized.completion_reason


def test_goal_read_preserves_explicit_user_stopped_state(tmp_path):
    path = tmp_path / "current.json"
    path.write_text(
        json.dumps(
            {
                "objective": "reach score",
                "status": "stopped",
                "completion_reason": "explicit user stopped campaign",
            }
        )
    )
    store = GoalStore(path)

    goal = store.get_goal()

    assert goal is not None
    assert goal.status == "stopped"
    assert goal.completion_reason == "explicit user stopped campaign"


def test_goal_read_normalizes_non_explicit_stopped_state(tmp_path):
    path = tmp_path / "current.json"
    path.write_text(
        json.dumps(
            {
                "objective": "reach score",
                "status": "stopped",
                "completion_reason": "scheduler halted campaign",
            }
        )
    )
    store = GoalStore(path)

    goal = store.get_goal()

    assert goal is not None
    assert goal.status == "active"
    assert "Non-explicit stopped status" in goal.completion_reason
    assert "validation/regression" in goal.completion_reason


def test_update_goal_treats_budget_exhausted_as_audit_only(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score", token_budget=10, token_budget_scope="campaign")

    goal = store.update_goal("budget_exhausted")

    assert goal.status == "active"
    assert "audit metadata only" in goal.completion_reason


def test_update_goal_treats_limit_stopped_as_audit_only(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score")

    goal = store.update_goal("stopped", reason="token budget and timeout exhausted")

    assert goal.status == "active"
    assert "Limit-driven stopped status was normalized" in goal.completion_reason
    assert "validation/regression" in goal.completion_reason
    assert "must not stop on time" in goal.completion_reason


def test_update_goal_treats_limit_complete_as_audit_only(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score")

    goal = store.update_goal("complete", reason="round limit and timeout exhausted")

    assert goal.status == "active"
    assert goal.completed_at is None
    assert "Limit-driven complete status was normalized" in goal.completion_reason
    assert "must not stop on time" in goal.completion_reason


def test_update_goal_treats_non_explicit_complete_as_audit_only(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score")

    goal = store.update_goal(
        "complete",
        reason="scheduler marked campaign complete after local bookkeeping",
    )

    assert goal.status == "active"
    assert goal.completed_at is None
    assert "Non-explicit complete status was normalized" in goal.completion_reason
    assert "must not stop on time" in goal.completion_reason


def test_update_goal_preserves_explicit_complete_state(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score")

    goal = store.update_goal(
        "complete",
        reason="explicit user marks campaign goal complete",
    )

    assert goal.status == "complete"
    assert goal.completed_at is not None
    assert goal.completion_reason == "explicit user marks campaign goal complete"


def test_update_goal_preserves_explicit_user_stopped_state(tmp_path):
    store = GoalStore(tmp_path / "current.json")
    store.create_goal("reach score")

    goal = store.update_goal("stopped", reason="explicit user stopped campaign")

    assert goal.status == "stopped"
    assert goal.completion_reason == "explicit user stopped campaign"


def test_submit_gate_default_disabled(tmp_path):
    gate = SubmitGate(SubmitConfig(enabled=False), submissions_dir=tmp_path)
    result = gate.check(
        campaign_id="camp1",
        best_job_dir=tmp_path / "missing-job",
        score=1.0,
        tasks_evaluated=89,
        full_regression_passed=True,
    )
    assert result.eligible is False
    assert "submit.enabled is false" in result.reasons


def test_submit_gate_upload_command_supports_share_confirmation(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    write_uploadable_job(job_dir)
    gate = SubmitGate(
        SubmitConfig(
            enabled=True,
            trigger_score=0.5,
            min_tasks_evaluated=1,
            min_attempts_per_task=1,
            require_full_regression=False,
            require_clean_git=False,
            require_no_uncommitted_harness_diff=False,
            harbor_upload=True,
            visibility="private",
            share_orgs=["TimWhite-AGI"],
            share_users=["timwhitez"],
            share_yes=True,
        ),
        submissions_dir=tmp_path / "submissions",
        harbor_bin="harbor",
    )

    result = gate.check(
        campaign_id="camp-share",
        best_job_dir=job_dir,
        score=0.9,
        tasks_evaluated=1,
        full_regression_passed=True,
    )

    assert result.command == [
        "harbor",
        "upload",
        str(job_dir),
        "--private",
        "--share-org",
        "TimWhite-AGI",
        "--share-user",
        "timwhitez",
        "--yes",
    ]


def test_submit_once_does_not_upload_when_harbor_upload_disabled(tmp_path):
    harbor_log = tmp_path / "harbor.log"
    harbor_bin = tmp_path / "harbor"
    harbor_bin.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$@\" >> \"$HARBOR_LOG\"\n"
        "exit 0\n"
    )
    harbor_bin.chmod(0o755)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    write_uploadable_job(job_dir)

    gate = SubmitGate(
        SubmitConfig(
            enabled=True,
            trigger_score=0.5,
            min_tasks_evaluated=1,
            min_attempts_per_task=1,
            require_full_regression=False,
            require_clean_git=False,
            require_no_uncommitted_harness_diff=False,
            harbor_upload=False,
        ),
        submissions_dir=tmp_path / "submissions",
        harbor_bin=str(harbor_bin),
    )

    result = gate.submit_once(
        campaign_id="camp-no-upload",
        best_job_dir=job_dir,
        score=0.9,
        tasks_evaluated=1,
        full_regression_passed=True,
    )

    result_payload = json.loads((tmp_path / "submissions" / "camp-no-upload.json").read_text())
    assert result.eligible is True
    assert result.command == []
    assert result_payload["submitted"] is False
    assert result_payload["upload_skipped"] is True
    assert not harbor_log.exists()


def test_submit_once_reports_terminal_upload_failure(tmp_path):
    harbor_bin = tmp_path / "harbor"
    harbor_bin.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"auth\" && \"$2\" == \"status\" ]]; then exit 0; fi\n"
        "echo upload failed >&2\n"
        "exit 42\n"
    )
    harbor_bin.chmod(0o755)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    write_uploadable_job(job_dir)

    gate = SubmitGate(
        SubmitConfig(
            enabled=True,
            trigger_score=0.5,
            min_tasks_evaluated=1,
            min_attempts_per_task=1,
            require_full_regression=False,
            require_clean_git=False,
            require_no_uncommitted_harness_diff=False,
            harbor_upload=True,
            stop_after_submit_attempt=True,
        ),
        submissions_dir=tmp_path / "submissions",
        harbor_bin=str(harbor_bin),
    )

    result = gate.submit_once(
        campaign_id="camp-fail",
        best_job_dir=job_dir,
        score=0.9,
        tasks_evaluated=1,
        full_regression_passed=True,
    )

    result_payload = json.loads((tmp_path / "submissions" / "camp-fail.json").read_text())
    assert result.eligible is True
    assert result.attempted is True
    assert result.submitted is False
    assert result.returncode == 42
    assert result.terminal is True
    assert result_payload["submitted"] is False
    assert result_payload["returncode"] == 42
    assert result_payload["stderr"] == "upload failed\n"


def test_submit_once_cli_loads_config_and_prints_json(tmp_path):
    config_path = tmp_path / "trials.yaml"
    config_path.write_text(
        """
submit:
  enabled: true
  trigger_score: 0.5
  min_tasks_evaluated: 1
  min_attempts_per_task: 1
  require_full_regression: false
  require_clean_git: false
  require_no_uncommitted_harness_diff: false
  harbor_upload: false
  share_orgs:
    - OldOrg
  share_users:
    - old-user
  share_yes: false
"""
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/submit_once.py",
            "--campaign-id",
            "camp-cli",
            "--best-job-dir",
            str(job_dir),
            "--score",
            "0.9",
            "--tasks-evaluated",
            "1",
            "--config",
            str(config_path),
            "--submissions-dir",
            str(tmp_path / "submissions"),
            "--share-org",
            "TimWhite-AGI",
            "--share-user",
            "timwhitez",
            "--share-yes",
            "--json",
            "--dry-run",
            "--record-dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["eligible"] is True
    assert payload["harbor_upload"] is False
    assert payload["command"] == []
    assert payload["share_orgs"] == ["TimWhite-AGI"]
    assert payload["share_users"] == ["timwhitez"]
    assert payload["share_yes"] is True
    record_path = tmp_path / "submissions" / "camp-cli.dry_run.json"
    assert payload["dry_run_record_path"] == str(record_path)
    assert record_path.exists()
    assert not (tmp_path / "submissions" / "camp-cli.json").exists()
    record = json.loads(record_path.read_text())
    assert record["non_terminal_marker"] is True
    assert record["dry_run"] is True


def test_submit_gate_requires_official_attempt_evidence(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    gate = SubmitGate(
        SubmitConfig(
            enabled=True,
            trigger_score=0.5,
            min_tasks_evaluated=2,
            min_attempts_per_task=5,
            require_full_regression=False,
            require_clean_git=False,
            require_no_uncommitted_harness_diff=False,
            harbor_upload=False,
        ),
        submissions_dir=tmp_path / "submissions",
    )

    missing = gate.check(
        campaign_id="camp-attempts",
        best_job_dir=job_dir,
        score=0.9,
        tasks_evaluated=2,
        full_regression_passed=True,
    )
    short = gate.check(
        campaign_id="camp-attempts",
        best_job_dir=job_dir,
        score=0.9,
        tasks_evaluated=2,
        full_regression_passed=True,
        attempts_per_task={"task-a": 5, "task-b": 4},
    )
    enough = gate.check(
        campaign_id="camp-attempts",
        best_job_dir=job_dir,
        score=0.9,
        tasks_evaluated=2,
        full_regression_passed=True,
        attempts_per_task={"task-a": 5, "task-b": 5},
    )

    assert missing.eligible is False
    assert "attempt evidence missing for leaderboard submit gate" in missing.reasons
    assert short.eligible is False
    assert "tasks below minimum attempts 5: task-b" in short.reasons
    assert enough.eligible is True


def test_submit_gate_rejects_missing_harbor_result_for_upload(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    gate = SubmitGate(
        SubmitConfig(
            enabled=True,
            trigger_score=0.5,
            min_tasks_evaluated=1,
            min_attempts_per_task=1,
            require_full_regression=False,
            require_clean_git=False,
            require_no_uncommitted_harness_diff=False,
            harbor_upload=True,
        ),
        submissions_dir=tmp_path / "submissions",
        harbor_bin=str(tmp_path / "missing-harbor"),
    )

    result = gate.check(
        campaign_id="camp-missing-result",
        best_job_dir=job_dir,
        score=0.9,
        tasks_evaluated=1,
        full_regression_passed=True,
    )

    assert result.eligible is False
    assert "Harbor job result.json missing; cannot verify upload integrity" in result.reasons


def test_submit_gate_rejects_passing_trial_without_atif_trajectory(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    write_uploadable_job(job_dir, trajectory=False)
    gate = SubmitGate(
        SubmitConfig(
            enabled=True,
            trigger_score=0.5,
            min_tasks_evaluated=1,
            min_attempts_per_task=1,
            require_full_regression=False,
            require_clean_git=False,
            require_no_uncommitted_harness_diff=False,
            harbor_upload=True,
        ),
        submissions_dir=tmp_path / "submissions",
        harbor_bin=str(tmp_path / "missing-harbor"),
    )

    result = gate.check(
        campaign_id="camp-no-atif",
        best_job_dir=job_dir,
        score=0.9,
        tasks_evaluated=1,
        full_regression_passed=True,
    )

    assert result.eligible is False
    assert "passing Harbor trial task-a__1 is missing ATIF trajectory" in result.reasons


def test_submit_gate_rejects_solution_url_in_agent_trajectory(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    write_uploadable_job(job_dir)
    trajectory_path = job_dir / "task-a__1" / "agent" / "trajectory.jsonl"
    trajectory_path.write_text(
        json.dumps(
            {
                "type": "tool_call",
                "tool": "bash",
                "args": {
                    "command": "curl https://example.com/solutions/task-a.py > AGENTS.md"
                },
            }
        )
        + "\n"
    )
    gate = SubmitGate(
        SubmitConfig(
            enabled=True,
            trigger_score=0.5,
            min_tasks_evaluated=1,
            min_attempts_per_task=1,
            require_full_regression=False,
            require_clean_git=False,
            require_no_uncommitted_harness_diff=False,
            harbor_upload=True,
        ),
        submissions_dir=tmp_path / "submissions",
        harbor_bin=str(tmp_path / "missing-harbor"),
    )

    result = gate.check(
        campaign_id="camp-solution-url",
        best_job_dir=job_dir,
        score=0.9,
        tasks_evaluated=1,
        full_regression_passed=True,
    )

    assert result.eligible is False
    assert any("External solution URL access" in reason for reason in result.reasons)


def test_compression_preserves_evidence_paths(tmp_path):
    engine = CompressionEngine(repo_root=tmp_path)
    plan = engine.dry_run(patch_metadata=[{"touched_components": ["a", "b"]}])
    assert "trials/regressions" in plan.preserved_paths
    assert "trials/submissions" in plan.preserved_paths
