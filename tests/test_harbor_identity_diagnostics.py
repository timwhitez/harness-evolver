from __future__ import annotations

import json
from pathlib import Path

from bench.harbor import (
    HarborRunner,
    _TaskIdentity,
    _fallback_candidate_matches,
    _identity_matches_requested,
)


def test_multiple_distinct_names_or_paths_are_ambiguous() -> None:
    assert (
        _identity_matches_requested(
            _TaskIdentity(names={"fix-git", "fix-git-history"}, paths=set()),
            "fix-git",
        )
        is False
    )
    assert (
        _identity_matches_requested(
            _TaskIdentity(
                names=set(),
                paths={"/dataset-a/fix-git", "/dataset-b/fix-git"},
            ),
            "fix-git",
        )
        is False
    )


def test_failed_selection_reports_every_observed_typed_identity(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "trial_name": "fix-git-history__ABCDEFG",
                        "task_name": "fix-git-history",
                    },
                    {
                        "trial_name": "other__ABCDEFG",
                        "task_id": {"path": "/datasets/other"},
                    },
                    {
                        "trial_name": "missing__ABCDEFG",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = HarborRunner().parse_job_dir(job, task_id="fix-git")

    assert result.metadata["task_identity_match_failed"] is True
    assert result.metadata["requested_task_identity"] == "fix-git"
    assert result.metadata["observed_task_identities"] == [
        "name:fix-git-history",
        "path:/datasets/other",
    ]
    assert any(
        "Requested task identity 'fix-git'" in error
        and "name:fix-git-history" in error
        and "path:/datasets/other" in error
        for error in result.error_log
    )


def test_conflicting_fallback_result_and_config_paths_are_rejected(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "fix-git__ABCDEFG"
    candidate.mkdir()
    (candidate / "result.json").write_text(
        json.dumps({"task_id": {"path": "/dataset-a/fix-git"}}),
        encoding="utf-8",
    )
    (candidate / "config.json").write_text(
        json.dumps({"task_id": {"path": "/dataset-b/fix-git"}}),
        encoding="utf-8",
    )

    assert _fallback_candidate_matches(candidate, "fix-git") is False
