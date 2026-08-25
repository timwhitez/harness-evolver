from __future__ import annotations

from bench.harbor import HarborRunner
from hl.types import TaskDifficulty, TaskDomain, TrialResult, TrialStatus


def _aggregate(attempts: list[dict[str, object]]) -> TrialResult:
    return TrialResult(
        trial_id="aggregate",
        task_id="task",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.ERROR,
        score=0.0,
        verified=False,
        metadata={
            "multi_attempt_aggregate": True,
            "attempt_results": attempts,
        },
    )


def _attempt(
    *,
    verified: bool = False,
    timeout_phase: str = "",
    infrastructure_phase: str = "",
    evidence_key: str = "",
    evidence: object | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if timeout_phase:
        metadata["timeout_phase"] = timeout_phase
    if infrastructure_phase:
        metadata["infrastructure_phase"] = infrastructure_phase
    if evidence_key:
        metadata[evidence_key] = evidence
    return {
        "verified": verified,
        "metadata": metadata,
    }


def test_all_structured_infrastructure_attempts_are_excluded() -> None:
    trial = _aggregate(
        [
            _attempt(
                timeout_phase="environment_start",
                evidence_key="environment_start_evidence",
                evidence="Environment start timed out",
            ),
            _attempt(
                timeout_phase="verifier_runtime_prepare",
                evidence_key="verifier_runtime_prepare_evidence",
                evidence="Verifier runtime prepare timed out",
            ),
        ]
    )

    assert HarborRunner().is_infra_error(trial) is True


def test_all_trusted_early_phase_attempts_are_excluded() -> None:
    trial = _aggregate(
        [
            _attempt(
                infrastructure_phase="harbor_launch",
                evidence_key="harbor_launch_evidence",
                evidence={
                    "kind": "executable_not_found",
                    "executable": "harbor",
                },
            ),
            _attempt(
                infrastructure_phase="environment_setup",
                evidence_key="early_environment_setup_evidence",
                evidence=(
                    "registry-1.docker.io: Temporary failure resolving "
                    "registry-1.docker.io"
                ),
            ),
        ]
    )

    assert HarborRunner().is_infra_error(trial) is True


def test_mixed_early_and_timeout_infrastructure_attempts_are_excluded() -> None:
    trial = _aggregate(
        [
            _attempt(
                infrastructure_phase="harbor_launch",
                evidence_key="harbor_launch_evidence",
                evidence={"kind": "executable_not_permitted"},
            ),
            _attempt(
                timeout_phase="environment_build",
                evidence_key="environment_build_evidence",
                evidence="Environment build timed out",
            ),
        ]
    )

    assert HarborRunner().is_infra_error(trial) is True


def test_phase_labels_without_owned_evidence_do_not_exclude_aggregate() -> None:
    trial = _aggregate(
        [
            _attempt(infrastructure_phase="harbor_launch"),
            _attempt(timeout_phase="environment_build"),
        ]
    )

    assert HarborRunner().is_infra_error(trial) is False


def test_mixed_infrastructure_and_worker_attempts_remain_scored() -> None:
    trial = _aggregate(
        [
            _attempt(
                timeout_phase="environment_start",
                evidence_key="environment_start_evidence",
                evidence="Environment start timed out",
            ),
            _attempt(timeout_phase="harbor_process"),
        ]
    )

    assert HarborRunner().is_infra_error(trial) is False


def test_verified_attempt_prevents_aggregate_infrastructure_exclusion() -> None:
    trial = _aggregate(
        [
            _attempt(
                timeout_phase="environment_start",
                evidence_key="environment_start_evidence",
                evidence="Environment start timed out",
            ),
            _attempt(
                verified=True,
                timeout_phase="environment_build",
                evidence_key="environment_build_evidence",
                evidence="Environment build timed out",
            ),
        ]
    )

    assert HarborRunner().is_infra_error(trial) is False


def test_malformed_attempt_snapshot_fails_closed_to_scored_result() -> None:
    trial = _aggregate(
        [
            _attempt(
                timeout_phase="environment_start",
                evidence_key="environment_start_evidence",
                evidence="Environment start timed out",
            ),
            {"verified": False},
        ]
    )

    assert HarborRunner().is_infra_error(trial) is False
