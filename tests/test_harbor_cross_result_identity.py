from __future__ import annotations

import json
from pathlib import Path

from bench.harbor import HarborRunner


def _path_attempt(path: str, trial_name: str) -> dict[str, object]:
    return {
        "trial_name": trial_name,
        "task_name": "shared-task",
        "task_id": {"name": "shared-task", "path": path},
    }


def test_simple_name_rejects_distinct_structured_paths_across_results() -> None:
    runner = HarborRunner()
    first = _path_attempt("/datasets/one/shared-task", "one__attempt-1")
    second = _path_attempt("/datasets/two/shared-task", "two__attempt-1")

    # Each record matches the requested basename in isolation. Selection must
    # nevertheless fail because the complete structured identities disagree.
    assert runner._trial_result_matches_task(first, "shared-task") is True
    assert runner._trial_result_matches_task(second, "shared-task") is True
    assert runner._matching_trial_results([first, second], "shared-task") == []
    assert runner._select_trial_result([first, second], "shared-task") is None


def test_multiple_attempts_for_one_structured_path_remain_selectable() -> None:
    runner = HarborRunner()
    first = _path_attempt("/datasets/one/shared-task", "shared__attempt-1")
    second = _path_attempt("/datasets/one/shared-task", "shared__attempt-2")

    assert runner._matching_trial_results([second, first], "shared-task") == [
        second,
        first,
    ]
    assert runner._select_trial_result([second, first], "shared-task") is second


def test_cross_result_ambiguity_is_persisted_in_parser_diagnostics(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    attempts = [
        _path_attempt("/datasets/one/shared-task", "one__attempt-1"),
        _path_attempt("/datasets/two/shared-task", "two__attempt-1"),
    ]
    (job / "result.json").write_text(
        json.dumps({"trial_results": attempts}),
        encoding="utf-8",
    )

    result = HarborRunner().parse_job_dir(job, task_id="shared-task")

    assert result.metadata["task_identity_match_failed"] is True
    assert result.metadata["observed_task_identities"] == [
        "name:shared-task",
        "path:/datasets/one/shared-task",
        "path:/datasets/two/shared-task",
    ]
    assert any(
        "path:/datasets/one/shared-task" in error
        and "path:/datasets/two/shared-task" in error
        for error in result.error_log
    )
