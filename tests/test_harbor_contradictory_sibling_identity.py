from __future__ import annotations

import json
from pathlib import Path

from bench.harbor import HarborRunner


def _attempt(
    *,
    trial_name: str,
    task_name: str,
    task_path: str,
    score: float,
) -> dict[str, object]:
    return {
        "trial_name": trial_name,
        "task_name": task_name,
        "task_id": {"name": task_name, "path": task_path},
        "verifier_result": {"rewards": {"reward": score}},
    }


def test_valid_exact_attempt_does_not_hide_contradictory_sibling(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    accepted = _attempt(
        trial_name="wanted__attempt-1",
        task_name="wanted-task",
        task_path="/datasets/current/wanted-task",
        score=1.0,
    )
    contradictory = _attempt(
        trial_name="stale__attempt-1",
        task_name="wanted-task",
        task_path="/datasets/stale/other-task",
        score=0.0,
    )
    attempts = [accepted, contradictory]
    (job / "result.json").write_text(
        json.dumps({"trial_results": attempts}),
        encoding="utf-8",
    )
    for attempt in attempts:
        trial_dir = job / str(attempt["trial_name"])
        trial_dir.mkdir()
        (trial_dir / "result.json").write_text(
            json.dumps(attempt),
            encoding="utf-8",
        )

    result = HarborRunner().parse_job_dir(job, task_id="wanted-task")

    assert result.metadata.get("multi_attempt_aggregate") is not True
    assert result.metadata["task_identity_match_failed"] is True
    assert result.metadata["observed_task_identities"] == [
        "name:wanted-task",
        "path:/datasets/current/wanted-task",
        "path:/datasets/stale/other-task",
    ]


def test_malformed_trial_results_cannot_recover_convenient_directory(
    tmp_path: Path,
) -> None:
    malformed_shapes = ["malformed", {"bad": "shape"}, [1], 1, True]
    for index, malformed in enumerate(malformed_shapes):
        job = tmp_path / f"job-{index}"
        trial_dir = job / "wanted-task__abcdefg"
        trial_dir.mkdir(parents=True)
        (trial_dir / "exception.txt").write_text(
            "harbor.trial.trial.AgentTimeoutError: Agent execution timed out\n",
            encoding="utf-8",
        )
        (job / "result.json").write_text(
            json.dumps({"trial_results": malformed}),
            encoding="utf-8",
        )

        result = HarborRunner().parse_job_dir(job, task_id="wanted-task")

        assert result.status.value == "error"
        assert result.harbor_trial_dir == ""
        assert result.metadata["task_identity_match_failed"] is True
        assert result.metadata["malformed_trial_results"] is True


def test_contradictory_top_level_cannot_recover_convenient_directory(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    trial_dir = job / "wanted-task__abcdefg"
    trial_dir.mkdir(parents=True)
    (trial_dir / "exception.txt").write_text(
        "harbor.trial.trial.AgentTimeoutError: Agent execution timed out\n",
        encoding="utf-8",
    )
    contradictory = _attempt(
        trial_name="stale__attempt-1",
        task_name="wanted-task",
        task_path="/datasets/stale/other-task",
        score=0.0,
    )
    (job / "result.json").write_text(
        json.dumps({"trial_results": [contradictory]}),
        encoding="utf-8",
    )

    result = HarborRunner().parse_job_dir(job, task_id="wanted-task")

    assert result.harbor_trial_dir == ""
    assert result.metadata["task_identity_match_failed"] is True
