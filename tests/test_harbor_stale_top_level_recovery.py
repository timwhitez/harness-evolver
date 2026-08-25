from __future__ import annotations

import json
from pathlib import Path

from bench.harbor import HarborRunner
from hl.types import TrialStatus


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


def _write_subdir_result(job: Path, attempt: dict[str, object]) -> None:
    trial = job / str(attempt["trial_name"])
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(json.dumps(attempt), encoding="utf-8")


def test_unrelated_valid_top_level_recovers_exact_subdir_attempts(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    stale = _attempt(
        trial_name="other__attempt-1",
        task_name="other-task",
        task_path="/datasets/old/other-task",
        score=1.0,
    )
    (job / "result.json").write_text(
        json.dumps({"trial_results": [stale]}),
        encoding="utf-8",
    )

    first = _attempt(
        trial_name="wanted__attempt-1",
        task_name="wanted-task",
        task_path="/datasets/current/wanted-task",
        score=0.0,
    )
    second = _attempt(
        trial_name="wanted__attempt-2",
        task_name="wanted-task",
        task_path="/datasets/current/wanted-task",
        score=1.0,
    )
    _write_subdir_result(job, first)
    _write_subdir_result(job, second)

    result = HarborRunner().parse_job_dir(job, task_id="wanted-task")

    assert result.status == TrialStatus.FAILED
    assert result.score == 0.5
    assert result.metadata["multi_attempt_aggregate"] is True
    assert result.metadata["attempt_count"] == 2
    assert result.metadata["attempt_result_source"] == (
        "trial_subdirs_after_unrelated_top_level"
    )
    assert result.metadata["top_level_job_result_valid"] is True
    assert result.metadata["top_level_job_result_matched_task"] is False
    assert result.metadata["subdirectory_attempt_recovery"] is True


def test_conflicting_top_level_identity_is_not_hidden_by_subdir_subset(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    conflicting = [
        _attempt(
            trial_name="one__attempt-1",
            task_name="shared-task",
            task_path="/datasets/one/shared-task",
            score=1.0,
        ),
        _attempt(
            trial_name="two__attempt-1",
            task_name="shared-task",
            task_path="/datasets/two/shared-task",
            score=0.0,
        ),
    ]
    (job / "result.json").write_text(
        json.dumps({"trial_results": conflicting}),
        encoding="utf-8",
    )

    apparently_convenient = _attempt(
        trial_name="one__attempt-2",
        task_name="shared-task",
        task_path="/datasets/one/shared-task",
        score=1.0,
    )
    _write_subdir_result(job, apparently_convenient)

    result = HarborRunner().parse_job_dir(job, task_id="shared-task")

    assert result.metadata.get("multi_attempt_aggregate") is not True
    assert result.metadata["task_identity_match_failed"] is True
    assert result.metadata["observed_task_identities"] == [
        "name:shared-task",
        "path:/datasets/one/shared-task",
        "path:/datasets/two/shared-task",
    ]
