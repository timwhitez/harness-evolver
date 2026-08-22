import json
import subprocess
import sys
from pathlib import Path


def test_roadmap_audit_reports_real_evidence_without_claiming_complete(tmp_path):
    memory = tmp_path / "trials"
    trial_dir = memory / "runs" / "fix-git__pass"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_id": "fix-git__pass",
                "task_id": "fix-git",
                "status": "passed",
                "score": 1.0,
                "verified": True,
                "harbor_job_dir": str(tmp_path / "jobs" / "smoke"),
            }
        )
    )
    for name in ("trajectory.jsonl", "verifier_output.txt", "feedback.json", "harness_snapshot.json", "handoff.md"):
        (trial_dir / name).write_text("{}\n")
    regressions = memory / "regressions"
    regressions.mkdir(parents=True)
    (regressions / "fix-git.json").write_text(json.dumps({"task_id": "fix-git"}))
    submissions = memory / "submissions"
    submissions.mkdir(parents=True)
    (submissions / "camp.dry_run.json").write_text(
        json.dumps({"dry_run": True, "eligible": True, "non_terminal_marker": True})
    )

    jobs = tmp_path / "jobs"
    regression_job = jobs / "regression_123_fix-git"
    regression_job.mkdir(parents=True)
    (regression_job / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {"verifier_result": {"rewards": {"reward": 1.0}}}
                ]
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_roadmap.py",
            "--json",
            "--memory-path",
            str(memory),
            "--jobs-dir",
            str(jobs),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    by_id = {item["id"]: item for item in payload["checklist"]}
    assert by_id["phase2_1.verified_pass"]["status"] == "pass"
    assert by_id["phase5.regression_snapshots"]["status"] == "pass"
    assert by_id["phase5.real_regression_runs"]["status"] == "pass"
    assert by_id["phase6.submit_evidence"]["status"] == "pass"
    assert by_id["phase3.real_codex_update"]["status"] == "missing"
    assert by_id["phase7.campaign_scale_evidence"]["status"] == "missing"
    assert payload["roadmap_complete"] is False
    assert "sk-" not in completed.stdout


def test_roadmap_audit_strict_fails_when_runtime_evidence_is_missing(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_roadmap.py",
            "--strict",
            "--memory-path",
            str(tmp_path / "trials"),
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "[missing] phase2_1.verified_trial" in completed.stdout


def test_roadmap_audit_has_static_secret_hygiene_guard():
    from scripts.audit_roadmap import build_audit

    audit = build_audit(
        repo_root=Path("."),
        memory_path=Path("/tmp/hl-roadmap-audit-empty-trials"),
        jobs_dir=Path("/tmp/hl-roadmap-audit-empty-jobs"),
        task_path=Path("terminal-bench-tasks/terminal-bench"),
    )

    by_id = {item.id: item for item in audit.checklist}
    assert by_id["phase2_2.secret_hygiene"].status == "pass"
    assert all("sk-" not in evidence for evidence in by_id["phase2_2.provider_reasoning_config"].evidence)


def test_secret_hygiene_allows_fixtures_but_flags_real_leaks(tmp_path):
    from scripts.audit_roadmap import _check_secret_hygiene, _is_fixture_secret_match

    # Build the key-shaped tokens via concatenation so this test file itself
    # never contains a contiguous "sk-<16+ chars>" literal that the repo-wide
    # secret scan would (correctly) flag.
    real_shaped = "sk-" + "aB3dEf9hKmNpQrStUvWx"
    fake_shaped = "sk-" + "testsecret1234567890"

    # Deliberately fake fixtures and redaction assertions must not be flagged.
    assert _is_fixture_secret_match(fake_shaped, "command: echo " + fake_shaped)
    assert _is_fixture_secret_match(real_shaped, repr(real_shaped) + " not in completed.stdout")
    # A high-entropy key on an ordinary source line is still a leak.
    assert not _is_fixture_secret_match(real_shaped, "client = OpenAI(api_key=key)")

    # End to end: a planted real-looking key under a scanned dir is reported.
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "leak.py").write_text("API_KEY = " + repr(real_shaped) + "\n")
    item = _check_secret_hygiene(tmp_path)
    assert item.status == "missing"
    assert any("scripts/leak.py" in note for note in item.notes)


def test_campaign_scale_requires_full_task_results(tmp_path):
    from scripts.audit_roadmap import _check_campaign_scale

    memory = tmp_path / "trials"
    summaries = memory / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "full_campaign.json").write_text(
        json.dumps(
            {
                "tasks": [f"task-{index:03d}" for index in range(89)],
                "task_results": [{"task_id": "task-000", "trial_id": "trial-000"}],
                "patch_lineage": [{"iteration": 1, "patches_applied": []}],
                "reproducibility": {"git_commit": "abc"},
            }
        )
    )

    item = _check_campaign_scale(memory)

    assert item.status == "partial"
    assert "completed task results" in item.missing[0]
