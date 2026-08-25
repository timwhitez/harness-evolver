from __future__ import annotations

from bench.harbor import HarborRunner
from hl.types import TaskDifficulty, TaskDomain, TrialResult, TrialStatus


def _trial(
    metadata: dict[str, object],
    *,
    verified: bool = False,
) -> TrialResult:
    return TrialResult(
        trial_id="trial",
        task_id="task",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.ERROR,
        score=0.0,
        verified=verified,
        metadata=dict(metadata),
    )


def test_finalized_decision_persists_structured_infrastructure_attribution() -> None:
    trial = _trial(
        {
            "timeout_phase": "environment_start",
            "environment_start_evidence": "Environment start timed out",
        }
    )

    HarborRunner()._mark_retry_policy_finite(trial)

    assert trial.metadata["infra_attribution_finalized"] is True
    assert trial.metadata["infra_attribution_policy"] == "phase_owned_evidence_v2"
    assert trial.metadata["infra_error_detected"] is True
    assert trial.metadata["score_exclusion_reason"] == "infrastructure_error"


def test_finalized_decision_clears_stale_phase_and_exclusion_markers() -> None:
    trial = _trial(
        {
            "timeout_phase": "environment_start",
            "infra_error_detected": True,
            "score_exclusion_reason": "infrastructure_error",
        }
    )

    HarborRunner()._mark_retry_policy_finite(trial)

    assert trial.metadata["infra_attribution_finalized"] is True
    assert trial.metadata["infra_error_detected"] is False
    assert "score_exclusion_reason" not in trial.metadata


def test_verified_result_cannot_be_finalized_as_infrastructure() -> None:
    trial = _trial(
        {
            "environment_start_evidence": "Environment start timed out",
            "score_exclusion_reason": "infrastructure_error",
        },
        verified=True,
    )

    HarborRunner()._mark_retry_policy_finite(trial)

    assert trial.metadata["infra_attribution_finalized"] is True
    assert trial.metadata["infra_error_detected"] is False
    assert "score_exclusion_reason" not in trial.metadata
