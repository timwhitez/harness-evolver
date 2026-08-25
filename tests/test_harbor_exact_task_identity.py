from __future__ import annotations

import json
from pathlib import Path

from bench.harbor import HarborRunner


def test_similarly_prefixed_task_names_do_not_match() -> None:
    runner = HarborRunner()
    requested = {"task_name": "fix-git", "trial_name": "fix-git__1"}
    other = {"task_name": "fix-git-history", "trial_name": "fix-git-history__1"}

    assert runner._trial_result_matches_task(requested, "fix-git") is True
    assert runner._trial_result_matches_task(other, "fix-git") is False
    assert runner._select_trial_result([other, requested], "fix-git") is requested


def test_missing_exact_identity_does_not_fall_back_to_first_result() -> None:
    runner = HarborRunner()
    unrelated = {"task_name": "prefix-fix-git-suffix"}

    assert runner._select_trial_result([unrelated], "fix-git") is None


def test_structured_task_path_matches_a_simple_requested_name_by_tail() -> None:
    runner = HarborRunner()
    exact = {"task_id": {"path": "/datasets/terminal-bench/fix-git"}}
    similar = {"task_id": {"path": "/datasets/terminal-bench/fix-git-history"}}

    assert runner._trial_result_matches_task(exact, "fix-git") is True
    assert runner._trial_result_matches_task(similar, "fix-git") is False


def test_full_structured_paths_with_the_same_basename_do_not_match() -> None:
    runner = HarborRunner()
    exact = {"task_id": {"path": "/datasets/one/shared-task"}}
    other = {"task_id": {"path": "/datasets/two/shared-task"}}

    assert runner._trial_result_matches_task(exact, "/datasets/one/shared-task") is True
    assert runner._trial_result_matches_task(other, "/datasets/one/shared-task") is False
    assert runner._select_trial_result(
        [other, exact],
        "/datasets/one/shared-task",
    ) is exact


def test_structured_path_evidence_overrides_a_matching_but_ambiguous_name() -> None:
    runner = HarborRunner()
    contradictory = {
        "task_name": "shared-task",
        "task_id": {"path": "/datasets/two/shared-task"},
    }

    assert runner._trial_result_matches_task(
        contradictory,
        "/datasets/one/shared-task",
    ) is False


def test_structured_path_evidence_also_overrides_a_conflicting_simple_name() -> None:
    runner = HarborRunner()
    contradictory = {
        "task_name": "fix-git",
        "task_id": {"path": "/datasets/terminal-bench/fix-git-history"},
    }
    consistent = {
        "task_name": "legacy-display-name",
        "task_id": {"path": "/datasets/terminal-bench/fix-git"},
    }

    assert runner._trial_result_matches_task(contradictory, "fix-git") is False
    assert runner._trial_result_matches_task(consistent, "fix-git") is True
    assert runner._select_trial_result(
        [contradictory, consistent],
        "fix-git",
    ) is consistent


def test_string_task_ids_support_normalized_paths_without_substrings() -> None:
    runner = HarborRunner()

    assert runner._trial_result_matches_task({"task_id": "fix-git"}, "fix-git") is True
    assert runner._trial_result_matches_task(
        {"task_id": "C:\\datasets\\fix-git"},
        "fix-git",
    ) is True
    assert runner._trial_result_matches_task(
        {"task_id": "C:\\datasets\\fix-git"},
        "C:/datasets/fix-git",
    ) is True
    assert runner._trial_result_matches_task(
        {"task_id": "C:\\other\\fix-git"},
        "C:/datasets/fix-git",
    ) is False
    assert runner._trial_result_matches_task(
        {"task_id": "C:\\datasets\\prefix-fix-git-suffix"},
        "fix-git",
    ) is False


def test_fallback_directory_accepts_harbors_default_shortuuid_suffix(tmp_path: Path) -> None:
    runner = HarborRunner()
    wrong = tmp_path / "fix-git-history__A1b2C3d"
    exact = tmp_path / "fix-git__A1b2C3d"
    wrong.mkdir()
    exact.mkdir()

    assert runner._fallback_trial_dir(tmp_path, "fix-git") == exact


def test_full_path_fallback_requires_exact_recorded_path(tmp_path: Path) -> None:
    runner = HarborRunner()
    candidate = tmp_path / "shared-task__A1b2C3d"
    candidate.mkdir()

    assert runner._fallback_trial_dir(tmp_path, "/datasets/one/shared-task") is None

    (candidate / "result.json").write_text(
        json.dumps({"task_id": {"path": "/datasets/two/shared-task"}}),
        encoding="utf-8",
    )
    assert runner._fallback_trial_dir(tmp_path, "/datasets/one/shared-task") is None
    assert runner._fallback_trial_dir(tmp_path, "/datasets/two/shared-task") == candidate


def test_legacy_numeric_attempt_suffix_remains_supported(tmp_path: Path) -> None:
    runner = HarborRunner()
    exact = tmp_path / "fix-git__attempt-2"
    exact.mkdir()

    assert runner._fallback_trial_dir(tmp_path, "fix-git") == exact


def test_ambiguous_exact_fallback_directories_are_rejected(tmp_path: Path) -> None:
    runner = HarborRunner()
    (tmp_path / "fix-git__A1b2C3d").mkdir()
    (tmp_path / "fix-git__D4e5F6g").mkdir()

    assert runner._fallback_trial_dir(tmp_path, "fix-git") is None


def test_fallback_preserves_double_underscore_inside_task_id(tmp_path: Path) -> None:
    runner = HarborRunner()
    exact = tmp_path / "archive__repair__A1b2C3d"
    misleading_prefix = tmp_path / "archive__A1b2C3d"
    exact.mkdir()
    misleading_prefix.mkdir()

    assert runner._fallback_trial_dir(tmp_path, "archive__repair") == exact


def test_truncated_harbor_name_requires_exact_structured_identity(tmp_path: Path) -> None:
    runner = HarborRunner()
    task_id = "task-name-that-is-longer-than-thirty-two-characters"
    expected_prefix = task_id[:32].rstrip("_-")
    exact = tmp_path / f"{expected_prefix}__A1b2C3d"
    exact.mkdir()

    assert runner._fallback_trial_dir(tmp_path, task_id) is None

    (exact / "config.json").write_text(
        json.dumps({"dataset": {"include_task_name": task_id}}),
        encoding="utf-8",
    )
    assert runner._fallback_trial_dir(tmp_path, task_id) == exact


def test_colliding_truncated_task_prefix_is_not_misattributed(tmp_path: Path) -> None:
    runner = HarborRunner()
    shared = "task-name-with-a-common-prefix-1234"
    first = shared + "-first"
    second = shared + "-second"
    assert first[:32].rstrip("_-") == second[:32].rstrip("_-")

    candidate = tmp_path / f"{first[:32].rstrip('_-')}__A1b2C3d"
    candidate.mkdir()
    (candidate / "result.json").write_text(
        json.dumps({"task_name": second, "trial_name": candidate.name}),
        encoding="utf-8",
    )

    assert runner._fallback_trial_dir(tmp_path, first) is None
    assert runner._fallback_trial_dir(tmp_path, second) == candidate


def test_result_metadata_overrides_a_contradictory_short_directory_name(
    tmp_path: Path,
) -> None:
    runner = HarborRunner()
    candidate = tmp_path / "fix-git__A1b2C3d"
    candidate.mkdir()
    (candidate / "result.json").write_text(
        json.dumps({"task_name": "fix-git-history"}),
        encoding="utf-8",
    )

    assert runner._fallback_trial_dir(tmp_path, "fix-git") is None


def test_unknown_suffix_is_not_guessed_as_an_attempt(tmp_path: Path) -> None:
    runner = HarborRunner()
    (tmp_path / "fix-git__history__attempt-1").mkdir()
    (tmp_path / "fix-git__abcdefgh").mkdir()

    assert runner._fallback_trial_dir(tmp_path, "fix-git") is None
