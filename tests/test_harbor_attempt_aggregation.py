from __future__ import annotations

import json
from pathlib import Path

from bench.harbor import HarborRunner
from hl.types import TrialStatus


def _attempt(
    name: str,
    score: float | None,
    *,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    exception: dict[str, str] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "trial_name": name,
        "task_name": "example-task",
        "task_id": {"name": "example-task", "path": "/tasks/example-task"},
        "agent_result": {
            "n_input_tokens": input_tokens,
            "n_output_tokens": output_tokens,
            "cost_usd": cost_usd,
        },
    }
    if score is not None:
        result["verifier_result"] = {"rewards": {"reward": score}}
    if exception is not None:
        result["exception_info"] = exception
    return result


def _write_job(job_dir: Path, attempts: list[dict[str, object]]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "trial_results": attempts,
                "stats": {"n_completed_trials": len(attempts)},
            }
        ),
        encoding="utf-8",
    )
    for index, attempt in enumerate(attempts):
        trial_name = str(attempt["trial_name"])
        trial_dir = job_dir / trial_name
        (trial_dir / "agent").mkdir(parents=True, exist_ok=True)
        (trial_dir / "agent" / "trajectory.jsonl").write_text(
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "read",
                    "sequence": index,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts = trial_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / f"artifact-{index}.txt").write_text("artifact", encoding="utf-8")


def _write_per_trial_results(job_dir: Path, attempts: list[dict[str, object]]) -> None:
    _write_job(job_dir, attempts)
    for attempt in attempts:
        trial_dir = job_dir / str(attempt["trial_name"])
        (trial_dir / "result.json").write_text(
            json.dumps(attempt),
            encoding="utf-8",
        )


def test_multiple_attempts_use_an_order_independent_mean(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    failed = _attempt(
        "example-task__attempt-1",
        0.0,
        input_tokens=10,
        output_tokens=2,
        cost_usd=0.1,
    )
    passed = _attempt(
        "example-task__attempt-2",
        1.0,
        input_tokens=20,
        output_tokens=4,
        cost_usd=0.2,
    )
    _write_job(job_dir, [passed, failed])

    first = HarborRunner().parse_job_dir(job_dir, task_id="example-task", wall_time=12.5)
    _write_job(job_dir, [failed, passed])
    reversed_result = HarborRunner().parse_job_dir(
        job_dir,
        task_id="example-task",
        wall_time=12.5,
    )

    assert first.score == reversed_result.score == 0.5
    assert first.status == reversed_result.status == TrialStatus.FAILED
    assert first.verified is reversed_result.verified is True
    assert first.trial_id == reversed_result.trial_id
    assert first.token_usage == reversed_result.token_usage == {"input": 30, "output": 6}
    assert first.metadata["attempt_scores"] == reversed_result.metadata["attempt_scores"]
    assert first.metadata["attempt_pass_rate"] == 0.5
    assert first.metadata["attempt_verified_rate"] == 1.0
    assert first.metadata["attempt_order_independent"] is True
    assert first.metadata["attempt_aggregation_policy"] == "arithmetic_mean_all_attempts"
    assert first.metadata["trial_metrics"]["cost_usd"] == 0.3


def test_every_attempt_result_trajectory_and_artifact_remains_visible(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    attempts = [
        _attempt(
            "example-task__attempt-2",
            1.0,
            input_tokens=2,
            output_tokens=1,
            cost_usd=0.02,
        ),
        _attempt(
            "example-task__attempt-1",
            0.0,
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.01,
        ),
    ]
    _write_job(job_dir, attempts)

    result = HarborRunner().parse_job_dir(job_dir, task_id="example-task")

    snapshots = result.metadata["attempt_results"]
    assert len(snapshots) == 2
    assert [snapshot["trial_id"] for snapshot in snapshots] == [
        "example-task__attempt-1",
        "example-task__attempt-2",
    ]
    assert all(snapshot["trajectory_event_count"] == 1 for snapshot in snapshots)
    assert len(result.trajectory) == 2
    assert {event["_harbor_attempt_trial_id"] for event in result.trajectory} == {
        "example-task__attempt-1",
        "example-task__attempt-2",
    }
    assert result.artifacts == [
        "example-task__attempt-1/artifact-1.txt",
        "example-task__attempt-2/artifact-0.txt",
    ]


def test_unverified_error_attempt_is_included_as_zero_and_controls_status(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    passed = _attempt(
        "example-task__attempt-1",
        1.0,
        input_tokens=10,
        output_tokens=2,
        cost_usd=0.1,
    )
    errored = _attempt(
        "example-task__attempt-2",
        None,
        input_tokens=5,
        output_tokens=1,
        cost_usd=0.05,
        exception={"exception_type": "RuntimeError", "exception_message": "boom"},
    )
    _write_job(job_dir, [passed, errored])

    result = HarborRunner().parse_job_dir(job_dir, task_id="example-task")

    assert result.score == 0.5
    assert result.verified is False
    assert result.status == TrialStatus.ERROR
    assert result.metadata["attempt_verified_count"] == 1
    assert result.metadata["attempt_status_counts"] == {"error": 1, "passed": 1}
    assert len(result.metadata["attempt_results"]) == 2


def test_single_attempt_keeps_the_existing_parser_path(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    attempt = _attempt(
        "example-task__attempt-1",
        1.0,
        input_tokens=10,
        output_tokens=2,
        cost_usd=0.1,
    )
    _write_job(job_dir, [attempt])

    result = HarborRunner().parse_job_dir(job_dir, task_id="example-task")

    assert result.trial_id == "example-task__attempt-1"
    assert result.score == 1.0
    assert result.status == TrialStatus.PASSED
    assert "multi_attempt_aggregate" not in result.metadata


def test_verified_raw_verifier_text_is_normalized_in_every_attempt(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    attempts = [
        _attempt(
            f"example-task__attempt-{index}",
            0.0,
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.01,
        )
        for index in (1, 2)
    ]
    _write_job(job_dir, attempts)
    for attempt in attempts:
        verifier = job_dir / str(attempt["trial_name"]) / "verifier"
        verifier.mkdir()
        (verifier / "test-stdout.txt").write_text(
            "Verifier runtime network preparation timed out after 90 seconds\n",
            encoding="utf-8",
        )

    result = HarborRunner().parse_job_dir(job_dir, task_id="example-task")

    assert result.verified is True
    assert result.metadata["infra_error_detected"] is False
    for snapshot in result.metadata["attempt_results"]:
        metadata = snapshot["metadata"]
        assert metadata["verifier_infra_error"] is False
        assert metadata["infra_error_detected"] is False
        assert "score_exclusion_reason" not in metadata


def test_mixed_aggregate_normalizes_each_untrusted_attempt_marker(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    verified = _attempt(
        "example-task__attempt-1",
        0.0,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
    )
    errored = _attempt(
        "example-task__attempt-2",
        None,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.01,
        exception={
            "exception_type": "RuntimeError",
            "exception_message": "Command timed out after 90 seconds",
            "exception_traceback": (
                "bench/network_environment.py in _prepare_verifier_runtime"
            ),
        },
    )
    attempts = [verified, errored]
    _write_job(job_dir, attempts)
    verifier = job_dir / str(verified["trial_name"]) / "verifier"
    verifier.mkdir()
    (verifier / "test-stdout.txt").write_text(
        "Verifier runtime network preparation timed out after 90 seconds\n",
        encoding="utf-8",
    )

    result = HarborRunner().parse_job_dir(job_dir, task_id="example-task")

    assert result.verified is False
    assert result.metadata["infra_error_detected"] is False
    for snapshot in result.metadata["attempt_results"]:
        metadata = snapshot["metadata"]
        assert metadata["verifier_infra_error"] is False
        assert metadata["infra_error_detected"] is False
        assert "score_exclusion_reason" not in metadata


def test_missing_top_level_result_recovers_and_aggregates_subdirectory_attempts(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    attempts = [
        _attempt(
            "example-task__attempt-2",
            1.0,
            input_tokens=20,
            output_tokens=4,
            cost_usd=0.2,
        ),
        _attempt(
            "example-task__attempt-1",
            0.0,
            input_tokens=10,
            output_tokens=2,
            cost_usd=0.1,
        ),
    ]
    _write_per_trial_results(job_dir, attempts)
    (job_dir / "result.json").unlink()

    result = HarborRunner().parse_job_dir(job_dir, task_id="example-task")

    assert result.score == 0.5
    assert result.status == TrialStatus.FAILED
    assert result.verified is True
    assert result.metadata["attempt_count"] == 2
    assert result.metadata["attempt_result_source"] == "trial_subdirs_without_top_level"
    assert result.metadata["top_level_job_result_present"] is False
    assert result.metadata["top_level_job_result_valid"] is False
    assert result.metadata["subdirectory_attempt_recovery"] is True
    assert all(
        snapshot["metadata"]["normalized_from_subdirectory_result"] is True
        for snapshot in result.metadata["attempt_results"]
    )


def test_malformed_top_level_result_recovers_only_when_multiple_subdir_attempts_exist(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    attempts = [
        _attempt(
            "example-task__attempt-1",
            1.0,
            input_tokens=4,
            output_tokens=2,
            cost_usd=0.04,
        ),
        _attempt(
            "example-task__attempt-2",
            0.0,
            input_tokens=6,
            output_tokens=3,
            cost_usd=0.06,
        ),
    ]
    _write_per_trial_results(job_dir, attempts)
    (job_dir / "result.json").write_text("{not-json", encoding="utf-8")

    result = HarborRunner().parse_job_dir(job_dir, task_id="example-task")

    assert result.score == 0.5
    assert result.metadata["attempt_count"] == 2
    assert result.metadata["attempt_result_source"] == (
        "trial_subdirs_after_invalid_top_level"
    )
    assert result.metadata["top_level_job_result_present"] is True
    assert result.metadata["top_level_job_result_valid"] is False
