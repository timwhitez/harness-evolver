from __future__ import annotations

import pytest

from bench.scoring import Scoring
from hl.types import (
    TaskDifficulty,
    TaskDomain,
    TrialResult,
    TrialStatus,
    trial_is_infrastructure_failure,
)


def _trial(
    trial_id: str,
    *,
    status: TrialStatus,
    score: float,
    verified: bool = False,
    metadata: dict[str, object] | None = None,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        task_id=trial_id,
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=status,
        score=score,
        verified=verified,
        metadata=metadata or {},
    )


def _final_infra_metadata(**extra: object) -> dict[str, object]:
    return {
        "infra_attribution_finalized": True,
        "infra_attribution_policy": "phase_owned_evidence_v2",
        "infra_error_detected": True,
        "score_exclusion_reason": "infrastructure_error",
        **extra,
    }


@pytest.mark.parametrize(
    "phase",
    [
        "harbor_process",
        "harbor_cancelled",
        "environment_start",
        "environment_build",
        "verifier_runtime_prepare",
    ],
)
def test_timeout_phase_alone_never_establishes_score_exclusion(phase: str) -> None:
    trial = _trial(
        phase,
        status=(
            TrialStatus.CANCELLED
            if phase == "harbor_cancelled"
            else TrialStatus.TIMEOUT
        ),
        score=0.0,
        metadata={"timeout_phase": phase},
    )

    assert trial_is_infrastructure_failure(trial) is False


def test_worker_timeout_remains_in_score_denominator() -> None:
    passed = _trial(
        "passed",
        status=TrialStatus.PASSED,
        score=1.0,
        verified=True,
    )
    worker_hang = _trial(
        "worker-hang",
        status=TrialStatus.TIMEOUT,
        score=0.0,
        metadata={"timeout_phase": "harbor_process"},
    )

    summary = Scoring.build_summary("example", [passed, worker_hang])

    assert summary.total_tasks == 2
    assert summary.scored_tasks == 2
    assert summary.infrastructure_excluded == 0
    assert summary.overall_score == 0.5


@pytest.mark.parametrize(
    "metadata",
    [
        {"infra_error_detected": True},
        {"score_exclusion_reason": "infrastructure_error"},
        {
            "infra_error_detected": True,
            "score_exclusion_reason": "infrastructure_error",
        },
        {
            "infra_attribution_finalized": True,
            "infra_attribution_policy": "phase_owned_evidence_v1",
            "infra_error_detected": True,
            "score_exclusion_reason": "infrastructure_error",
        },
        {
            "infra_attribution_finalized": True,
            "infra_attribution_policy": "phase_owned_evidence_v2",
            "infra_error_detected": False,
            "score_exclusion_reason": "infrastructure_error",
        },
    ],
)
def test_partial_or_stale_markers_are_not_authoritative(
    metadata: dict[str, object],
) -> None:
    trial = _trial(
        "stale",
        status=TrialStatus.ERROR,
        score=0.0,
        metadata=metadata,
    )

    assert trial_is_infrastructure_failure(trial) is False


def test_pr39_finalized_infrastructure_decision_is_excluded() -> None:
    trial = _trial(
        "structured-infra",
        status=TrialStatus.TIMEOUT,
        score=0.0,
        metadata=_final_infra_metadata(
            timeout_phase="environment_start",
            environment_start_evidence="Environment start timed out",
        ),
    )

    assert trial_is_infrastructure_failure(trial) is True


@pytest.mark.parametrize(
    "metadata",
    [
        _final_infra_metadata(),
        _final_infra_metadata(timeout_phase="verifier_runtime_prepare"),
    ],
)
def test_verified_trial_is_never_excluded_by_finalized_failure_metadata(
    metadata: dict[str, object],
) -> None:
    trial = _trial(
        "verified",
        status=TrialStatus.FAILED,
        score=0.25,
        verified=True,
        metadata=metadata,
    )

    assert trial_is_infrastructure_failure(trial) is False


def test_pr39_all_infrastructure_aggregate_is_excluded() -> None:
    trial = _trial(
        "aggregate-infra",
        status=TrialStatus.ERROR,
        score=0.0,
        metadata=_final_infra_metadata(
            multi_attempt_aggregate=True,
            attempt_results=[
                {
                    "verified": False,
                    "metadata": {
                        "timeout_phase": "environment_start",
                        "environment_start_evidence": "Environment start timed out",
                    },
                },
                {
                    "verified": False,
                    "metadata": {
                        "timeout_phase": "environment_build",
                        "environment_build_evidence": "Environment build timed out",
                    },
                },
            ],
        ),
    )

    assert trial_is_infrastructure_failure(trial) is True


def test_pr39_mixed_aggregate_remains_in_score_denominator() -> None:
    passed = _trial(
        "passed",
        status=TrialStatus.PASSED,
        score=1.0,
        verified=True,
    )
    mixed = _trial(
        "aggregate-mixed",
        status=TrialStatus.ERROR,
        score=0.0,
        metadata={
            "multi_attempt_aggregate": True,
            "infra_attribution_finalized": True,
            "infra_attribution_policy": "phase_owned_evidence_v2",
            "infra_error_detected": False,
            "attempt_results": [
                {
                    "verified": False,
                    "metadata": {
                        "timeout_phase": "environment_start",
                        "environment_start_evidence": "Environment start timed out",
                    },
                },
                {
                    "verified": False,
                    "metadata": {"timeout_phase": "worker_execution"},
                },
            ],
        },
    )

    summary = Scoring.build_summary("example", [passed, mixed])

    assert summary.scored_tasks == 2
    assert summary.infrastructure_excluded == 0
    assert summary.overall_score == 0.5
