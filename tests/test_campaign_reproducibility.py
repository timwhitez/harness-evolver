"""Offline contracts for campaign invocation provenance and checkout observations."""

import copy
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bench.harbor import HarborRunner
from hl.memory import FileSystemMemory
from hl.types import (
    ComponentVersion,
    TaskDifficulty,
    TaskDomain,
    TrialResult,
    TrialStatus,
    TrialSummary,
)
from meta.missions import MissionPlanner
from scripts import run_campaign
from scripts.audit_roadmap import _check_campaign_scale


@pytest.fixture
def offline_git(monkeypatch):
    """Reject every unexpected subprocess and freeze the report collection time."""
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 10, 8, 0, 0, 0, tzinfo=UTC)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    monkeypatch.setattr(run_campaign, "datetime", FrozenDatetime)

    def install(mode="clean"):
        calls = []

        def run(command, **kwargs):
            assert command in (
                ["git", "rev-parse", "HEAD"],
                ["git", "status", "--short"],
            ), f"unexpected subprocess: {command}"
            calls.append((command, kwargs))
            secret = "fixture-private-value" * 100
            if mode == "missing":
                raise FileNotFoundError(secret)
            if mode == "permission":
                raise PermissionError(secret)
            failed = mode == "nonzero" or (
                mode == "status-failed" and command[1] == "status"
            ) or (mode == "commit-failed" and command[1] == "rev-parse")
            if failed:
                return subprocess.CompletedProcess(command, 128, secret, secret)
            output = "current-head\n" if command[1] == "rev-parse" else ""
            if mode == "dirty" and command[1] == "status":
                output = " M source.py\n?? fixture-private-value.txt\n"
            return subprocess.CompletedProcess(command, 0, output, "")

        monkeypatch.setattr(run_campaign.subprocess, "run", run)
        return calls

    return install


def _report(tmp_path, **kwargs):
    return run_campaign._build_campaign_report(
        campaign_id="provenance",
        tasks=[],
        iteration_limit=1,
        summaries=[],
        memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
        memory_path=tmp_path / "trials",
        regression_plan={"lane": "none"},
        submit_results=[],
        codex_update=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("mode", "commit", "dirty", "errors"),
    [
        ("clean", "current-head", False, {}),
        ("dirty", "current-head", True, {}),
        ("nonzero", None, None, {
            "git_commit": "git exited with code 128",
            "git_dirty": "git exited with code 128",
        }),
        ("missing", None, None, {
            "git_commit": "git unavailable: FileNotFoundError",
            "git_dirty": "git unavailable: FileNotFoundError",
        }),
        ("permission", None, None, {
            "git_commit": "git unavailable: PermissionError",
            "git_dirty": "git unavailable: PermissionError",
        }),
        ("status-failed", "current-head", None, {
            "git_dirty": "git exited with code 128",
        }),
        ("commit-failed", None, False, {
            "git_commit": "git exited with code 128",
        }),
    ],
)
def test_report_git_observations_are_scoped_and_honest(
    tmp_path, monkeypatch, offline_git, mode, commit, dirty, errors,
):
    monkeypatch.chdir(tmp_path)
    calls = offline_git(mode)
    report = _report(tmp_path)
    provenance = report["reproducibility"]

    assert provenance["git_commit"] == commit
    assert provenance["git_dirty"] is dirty
    assert provenance["git_errors"] == errors
    assert provenance["git_scope"] == "report_generation_checkout"
    assert provenance["git_query_cwd"] == str(tmp_path.resolve())
    assert provenance["git_observed_at"] == "2026-10-08T00:00:00+00:00"
    assert len(calls) == 2
    assert all(call[1]["cwd"] == tmp_path.resolve() for call in calls)
    assert "fixture-private-value" not in json.dumps(report)
    for reason in errors.values():
        assert len(reason) < 100
    path = run_campaign._write_campaign_report(tmp_path / "trials", "provenance", report)
    assert json.loads(path.read_text())["reproducibility"] == provenance


def test_report_git_query_subdirectory_observes_enclosing_checkout(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "--quiet", "--initial-branch=main", str(repo)], check=True)
    subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.com",
         "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
         "commit", "--quiet", "--allow-empty", "-m", "fixture"],
        cwd=repo, check=True,
    )
    commit = (repo / ".git/refs/heads/main").read_text().strip()
    (repo / "source.py").write_text("# untracked file outside the query directory\n")
    query_dir = repo / "scripts"
    query_dir.mkdir()
    monkeypatch.chdir(query_dir)
    calls = []
    real_run = subprocess.run

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return real_run(command, **kwargs)

    monkeypatch.setattr(run_campaign.subprocess, "run", run)
    provenance = _report(tmp_path)["reproducibility"]

    assert provenance["git_query_cwd"] == str(query_dir.resolve())
    assert "git_repo_root" not in provenance
    assert provenance["git_commit"] == commit
    assert provenance["git_dirty"] is True
    assert provenance["git_errors"] == {}
    assert [command for command, _ in calls] == [
        ["git", "rev-parse", "HEAD"], ["git", "status", "--short"],
    ]
    assert all(kwargs["cwd"] == query_dir.resolve() for _, kwargs in calls)


@pytest.mark.parametrize(
    ("selection", "git_mode"),
    [("custom", "clean"), ("local", "clean"), ("default", "clean"),
     ("cli-only", "clean"), ("process-role", "clean"), ("dotenv-role", "clean"),
     ("custom", "nonzero"), ("custom", "missing")],
)
def test_normal_campaign_final_and_checkpoint_record_resolved_sources(
    tmp_path, monkeypatch, capsys, offline_git, selection, git_mode,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HL_WORKER_ROLE", "")
    monkeypatch.setenv("TEST_API_KEY", "fixture-api-key")
    monkeypatch.setenv("TEST_PROVIDER_URL", "https://fixture-private-endpoint/v1")
    offline_git(git_mode)
    (tmp_path / "config").mkdir()
    trials_path = Path("custom-trials.yaml") if selection == "custom" else Path("config/trials.yaml")
    trials_path.write_text("private_note: fixture-yaml-secret\nexecution:\n  tool_timeout_seconds: 17\n")
    env_path = tmp_path / ".env.selected"
    env_path.write_text("TEST_API_KEY=fixture-dotenv-secret\n")
    role = "worker"
    role_source = "default"
    extra = []
    if selection == "custom":
        models_path = "custom-models.yaml"
        extra = ["--models-config", models_path, "--worker-role", role]
        role_source = "cli"
    elif selection == "cli-only":
        models_path = None
        extra = ["--model", "fixture-cli-model", "--provider", "openai",
                 "--api-key-env", "TEST_API_KEY"]
    else:
        models_path = "config/models.yaml" if selection == "default" else "config/local.yaml"
    if selection == "process-role":
        role = "worker_gpt"
        role_source = "process_environment"
        monkeypatch.setenv("HL_WORKER_ROLE", role)
    if selection == "dotenv-role":
        role = "worker_gpt"
        role_source = "selected_dotenv"
        env_path.write_text("HL_WORKER_ROLE=worker_gpt\nTEST_API_KEY=fixture-dotenv-secret\n")
    if models_path:
        Path(models_path).write_text(
            f"roles:\n  {role}:\n    provider: openai_compatible\n"
            "    model: fixture-model\n    api_key_env: TEST_API_KEY\n"
            "    base_url: https://fixture-private-endpoint/v1?token=fixture-url-secret\n"
        )
    if selection == "default":
        Path("config/local.yaml").write_text("roles:\n  other:\n    model: unused-model\n")
    run_configs = []

    def run_task(self, **kwargs):
        run_configs.append(kwargs["agent_config"])
        return TrialResult(
            trial_id="task__1", task_id=kwargs["task_id"],
            task_domain=TaskDomain.DEVOPS, task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED, score=1.0, verified=True,
            model_used=str(kwargs["agent_config"]["model"]),
        )

    monkeypatch.setattr(HarborRunner, "run_task", run_task)
    monkeypatch.setattr(sys, "argv", [
        "scripts/run_campaign.py", "--task", "task", "--iterations", "1",
        "--campaign-id", "provenance", "--regression-lane", "none",
        "--skip-network-preflight", "--env-file", str(env_path),
        "--trials-config", str(trials_path), *extra,
    ])
    assert run_campaign.main() == 0
    stdout = json.loads(capsys.readouterr().out)
    final = json.loads(Path("trials/summaries/provenance_campaign.json").read_text())
    checkpoint = json.loads(Path("trials/summaries/provenance_campaign.checkpoint.json").read_text())
    assert stdout == final
    assert len(run_configs) == 1
    assert run_configs[0]["tool_timeout_seconds"] == 17
    for report in (final, checkpoint):
        provenance = report["reproducibility"]
        assert provenance["trials_config"] == str(trials_path)
        assert provenance["models_config_path"] == models_path
        assert provenance["worker_role"] == role
        assert provenance["worker_role_source"] == role_source
        assert provenance["config_scope"] == "report_invocation"
        assert provenance["models_config_priority"] == ["config/local.yaml", "config/models.yaml"]
        assert provenance["models_config_priority_scope"] == "discovery_search_order"
        assert provenance["models_config_path"] == run_configs[0]["models_config_path"]
        assert provenance["worker_role_source"] == run_configs[0]["worker_role_source"]
        assert provenance["git_dirty"] is (False if git_mode == "clean" else None)
        for secret in ("fixture-api-key", "fixture-dotenv-secret", "fixture-yaml-secret",
                       "https://fixture-private-endpoint", "fixture-url-secret"):
            assert secret not in json.dumps(report)
    assert final["best"]["score"] == 1.0
    assert final["task_results"][0]["verified"] is True
    assert checkpoint["checkpoint"] is True


def test_stored_state_report_preserves_historical_evidence(tmp_path, monkeypatch, offline_git):
    monkeypatch.chdir(tmp_path)
    offline_git()
    memory_path = tmp_path / "trials"
    memory = FileSystemMemory(base_path=str(memory_path))
    trial = TrialResult(
        trial_id="historical", task_id="task", status=TrialStatus.PASSED,
        task_domain=TaskDomain.DEVOPS, task_difficulty=TaskDifficulty.EASY,
        score=1.0, verified=True, harness_version="historical-harness",
        component_versions={"prompt": ComponentVersion(
            name="prompt", version="v1", git_commit="historical-head", content_hash="original-hash",
        )},
        metadata={"execution_git_commit": "historical-head"},
    )
    memory.record_trial(trial)
    state = run_campaign._new_campaign_state("provenance", ["task"])
    run_campaign._record_campaign_trial(state, trial, iteration=1, summary_id="old-summary")
    run_campaign._record_campaign_summary(state, TrialSummary(
        summary_id="old-summary", trial_ids=["historical"], total_tasks=1,
        passed=1, overall_score=1.0, patches_applied=["historical-patch"],
    ))
    state_path = run_campaign._write_campaign_state(memory_path, "provenance", state)
    loaded_state = json.loads(state_path.read_text())
    state_before = copy.deepcopy(loaded_state)
    paths = list((memory_path / "runs" / "historical").iterdir()) + [state_path]
    before = {path: path.read_bytes() for path in paths}

    report = run_campaign._build_campaign_report_from_state(
        campaign_id="provenance", tasks=["task"], iteration_limit=1,
        campaign_state=loaded_state, summaries=[], memory=memory,
        memory_path=memory_path, regression_plan={"lane": "none"},
        submit_results=[], codex_update=False, stopped_reason="stored state", checkpoint=False,
    )

    provenance = report["reproducibility"]
    assert provenance["git_commit"] == "current-head"
    assert provenance["git_scope"] == "report_generation_checkout"
    assert provenance["git_query_cwd"] == str(tmp_path)
    assert provenance["trials_config"] is None
    assert provenance["models_config_path"] is None
    assert provenance["worker_role_source"] is None
    assert provenance["worker_role"] is None
    assert loaded_state == state_before
    assert {path: path.read_bytes() for path in paths} == before
    assert memory.get_trial("historical").model_dump() == trial.model_dump()
    assert report["patch_lineage"][0]["patches_applied"] == ["historical-patch"]
    assert report["task_results"][0]["verified"] is True


@pytest.mark.parametrize("provenance", [None, {"git_commit": "old-head"}, {
    "git_commit": None, "git_dirty": None, "git_errors": {"git_dirty": "git unavailable"},
}])
def test_legacy_and_unknown_reports_keep_reader_semantics(tmp_path, provenance):
    tasks = [f"task-{index}" for index in range(89)]
    report = {"campaign_id": "old", "tasks": tasks,
              "task_results": [{"task_id": task} for task in tasks],
              "patch_lineage": [{"iteration": 1}]}
    if provenance is not None:
        report["reproducibility"] = provenance
    before = copy.deepcopy(report)
    path = tmp_path / "summaries" / "old_campaign.json"
    path.parent.mkdir()
    path.write_text(json.dumps(report))
    loaded = json.loads(path.read_text())
    packet = MissionPlanner().from_campaign_summary(loaded)
    audit = _check_campaign_scale(tmp_path)

    assert packet.evidence_summary["has_reproducibility"] is bool(provenance)
    assert audit.status == ("pass" if provenance else "partial")
    assert loaded.get("reproducibility", {}).get("git_dirty") is None
    assert loaded.get("reproducibility", {}).get("models_config_path") is None
    assert loaded == before
    assert json.loads(path.read_text()) == before
