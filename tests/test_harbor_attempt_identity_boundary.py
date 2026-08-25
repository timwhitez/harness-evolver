from __future__ import annotations

import json
from pathlib import Path

from bench.harbor import HarborRunner


def _attempt(path: str, name: str, score: float) -> dict[str, object]:
    return {
        "trial_name": name,
        "task_name": "shared-task",
        "task_id": {"name": "shared-task", "path": path},
        "verifier_result": {"rewards": {"reward": score}},
    }


def _write_job(job: Path, attempts: list[dict[str, object]]) -> None:
    job.mkdir()
    (job / "result.json").write_text(
        json.dumps({"trial_results": attempts}),
        encoding="utf-8",
    )
    for attempt in attempts:
        trial = job / str(attempt["trial_name"])
        trial.mkdir()
        (trial / "result.json").write_text(json.dumps(attempt), encoding="utf-8")


def test_aggregation_rejects_same_basename_from_distinct_paths(tmp_path: Path) -> None:
    job = tmp_path / "job"
    attempts = [
        _attempt("/datasets/one/shared-task", "one__attempt-1", 1.0),
        _attempt("/datasets/two/shared-task", "two__attempt-1", 0.0),
    ]
    _write_job(job, attempts)

    result = HarborRunner().parse_job_dir(job, task_id="shared-task")

    assert result.metadata.get("multi_attempt_aggregate") is not True
    assert result.metadata["task_identity_match_failed"] is True
    assert result.metadata["observed_task_identities"] == [
        "name:shared-task",
        "path:/datasets/one/shared-task",
        "path:/datasets/two/shared-task",
    ]


def test_aggregation_keeps_multiple_attempts_for_one_path(tmp_path: Path) -> None:
    job = tmp_path / "job"
    attempts = [
        _attempt("/datasets/one/shared-task", "shared__attempt-2", 1.0),
        _attempt("/datasets/one/shared-task", "shared__attempt-1", 0.0),
    ]
    _write_job(job, attempts)

    result = HarborRunner().parse_job_dir(job, task_id="shared-task")

    assert result.metadata["multi_attempt_aggregate"] is True
    assert result.metadata["attempt_count"] == 2
    assert result.score == 0.5
