"""Scoring unit tests — infrastructure failures must not lower Worker score."""

from __future__ import annotations

from bench.scoring import Scoring
from hl.types import TaskDifficulty, TaskDomain, TrialResult, TrialStatus
from tests.infra_fixtures import finalized_infra_metadata


def _trial(
    task_id: str,
    status: TrialStatus,
    score: float,
    *,
    metadata: dict | None = None,
    domain: TaskDomain = TaskDomain.SOFTWARE_ENGINEERING,
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM,
) -> TrialResult:
    return TrialResult(
        trial_id=f"{task_id}__x",
        task_id=task_id,
        task_domain=domain,
        task_difficulty=difficulty,
        status=status,
        score=score,
        metadata=metadata or {},
    )


def test_infrastructure_error_excluded_from_overall_score():
    trials = [
        _trial("a", TrialStatus.PASSED, 1.0),
        _trial("b", TrialStatus.FAILED, 0.0),
        _trial(
            "c",
            TrialStatus.ERROR,
            0.0,
            metadata=finalized_infra_metadata(),
        ),
    ]
    summary = Scoring.build_summary("s1", trials)
    # Only a (pass) and b (fail) are scored -> 0.5, infra c excluded.
    assert summary.overall_score == 0.5
    assert summary.scored_tasks == 2
    assert summary.infrastructure_excluded == 1
    # Full status counts are preserved regardless of scoring exclusion.
    assert summary.total_tasks == 3
    assert summary.error == 1


def test_environment_start_timeout_excluded_from_score():
    trials = [
        _trial("a", TrialStatus.PASSED, 1.0),
        _trial(
            "b",
            TrialStatus.ERROR,
            0.0,
            metadata=finalized_infra_metadata(timeout_phase="environment_start"),
        ),
    ]
    summary = Scoring.build_summary("s2", trials)
    assert summary.overall_score == 1.0
    assert summary.scored_tasks == 1
    assert summary.infrastructure_excluded == 1


def test_all_infrastructure_failures_score_zero_without_crash():
    trials = [
        _trial(
            "a",
            TrialStatus.ERROR,
            0.0,
            metadata=finalized_infra_metadata(),
        ),
    ]
    summary = Scoring.build_summary("s3", trials)
    assert summary.overall_score == 0.0
    assert summary.scored_tasks == 0
    assert summary.infrastructure_excluded == 1


def test_per_domain_and_difficulty_exclude_infrastructure():
    trials = [
        _trial("a", TrialStatus.PASSED, 1.0, domain=TaskDomain.SECURITY),
        _trial(
            "b",
            TrialStatus.ERROR,
            0.0,
            metadata=finalized_infra_metadata(timeout_phase="environment_build"),
            domain=TaskDomain.SECURITY,
        ),
    ]
    summary = Scoring.build_summary("s4", trials)
    assert summary.per_domain_scores["security"] == 1.0
    assert summary.per_difficulty_scores["medium"] == 1.0


def test_normal_failures_still_count():
    trials = [
        _trial("a", TrialStatus.PASSED, 1.0),
        _trial("b", TrialStatus.FAILED, 0.0),
        _trial("c", TrialStatus.TIMEOUT, 0.0, metadata={"timeout_phase": "agent_execution"}),
    ]
    summary = Scoring.build_summary("s5", trials)
    # No infra exclusion: agent_execution timeout is a real worker failure.
    assert summary.overall_score == round(1.0 / 3.0, 4)
    assert summary.scored_tasks == 3
    assert summary.infrastructure_excluded == 0
