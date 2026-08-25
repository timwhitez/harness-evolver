from __future__ import annotations

import json
from pathlib import Path

from bench.harbor import HarborRunner


def test_full_path_request_rejects_name_only_result_identity() -> None:
    runner = HarborRunner()
    name_only = {"task_name": "shared-task", "task_id": {"name": "shared-task"}}

    assert runner._trial_result_matches_task(
        name_only,
        "/datasets/one/shared-task",
    ) is False


def test_full_path_fallback_rejects_name_only_recorded_identity(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "shared-task__A1b2C3d"
    candidate.mkdir()
    (candidate / "result.json").write_text(
        json.dumps({"task_name": "shared-task"}),
        encoding="utf-8",
    )

    assert HarborRunner()._fallback_trial_dir(
        tmp_path,
        "/datasets/one/shared-task",
    ) is None


def test_full_path_request_accepts_equal_structured_path() -> None:
    exact = {"task_id": {"path": "/datasets/one/shared-task"}}

    assert HarborRunner()._trial_result_matches_task(
        exact,
        "/datasets/one/shared-task",
    ) is True
