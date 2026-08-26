from __future__ import annotations

from pathlib import Path
import subprocess
from types import MethodType

from bench.harbor import HarborRunner
from hl.types import TaskDifficulty, TaskDomain, TrialResult, TrialStatus


def _trial(
    message: str,
    *,
    status: TrialStatus = TrialStatus.ERROR,
    metadata: dict[str, object] | None = None,
) -> TrialResult:
    return TrialResult(
        trial_id="trial",
        task_id="task",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=status,
        score=0.0,
        verified=False,
        error_log=[message],
        metadata=dict(metadata or {}),
    )


def test_worker_selected_missing_apt_package_is_not_infrastructure() -> None:
    runner = HarborRunner()
    trial = _trial("E: Unable to locate package package-that-does-not-exist")

    assert runner.is_infra_error(trial) is False
    assert runner._should_retry_infra_failure(trial, infra_error=False) is False


def test_worker_selected_missing_pypi_distribution_is_not_infrastructure() -> None:
    runner = HarborRunner()
    trial = _trial(
        "ERROR: Could not find a version that satisfies the requirement made-up-package\n"
        "ERROR: No matching distribution found for made-up-package"
    )

    assert runner.is_infra_error(trial) is False


def test_verifier_text_alone_does_not_reclassify_a_dependency_failure() -> None:
    runner = HarborRunner()
    trial = _trial(
        "No matching distribution found for verifier-selected-package",
        status=TrialStatus.FAILED,
        metadata={"verifier_infra_error": True},
    )
    trial.verifier_output = "No matching distribution found for verifier-selected-package"

    assert runner.is_infra_error(trial) is False


def test_verifier_endpoint_text_alone_is_not_runner_provenance() -> None:
    runner = HarborRunner()
    trial = _trial("verifier failed", status=TrialStatus.FAILED)
    trial.verifier_output = (
        "https://pypi.org/simple failed: Temporary failure resolving pypi.org"
    )

    assert runner.is_infra_error(trial) is False


def test_raw_harbor_endpoint_text_is_not_phase_owned_evidence() -> None:
    runner = HarborRunner()
    trial = _trial("Harbor process captured Worker output")
    trial.harbor_stderr = (
        "pip index https://pypi.org/simple failed: Temporary failure resolving pypi.org"
    )

    assert runner.is_infra_error(trial) is False


def test_raw_phase_looking_harbor_text_is_not_structured_evidence() -> None:
    runner = HarborRunner()
    trial = _trial("Worker can emit arbitrary process text")
    trial.harbor_stderr = "Environment start timed out"

    assert runner.is_infra_error(trial) is False


def test_worker_prebuilt_warmup_lookalike_is_not_environment_provenance() -> None:
    runner = HarborRunner()
    trial = _trial(
        "Prebuilt Docker image cache warmup failed: 403 Forbidden",
        metadata={
            "agent_exception_type": "RuntimeError",
            "agent_exception_message": (
                "Prebuilt Docker image cache warmup failed: 403 Forbidden"
            ),
        },
    )

    assert runner.is_infra_error(trial) is False
    assert runner._should_retry_infra_failure(trial, infra_error=False) is False


def test_worker_traceback_markers_cannot_forge_environment_provenance(
    tmp_path: Path,
) -> None:
    message = (
        "Prebuilt Docker image cache warmup failed: 403 Forbidden; "
        "diagnostic mentions bench/network_environment.py and "
        "_warm_prebuilt_image_cache_if_needed"
    )
    exception = {
        "exception_type": "RuntimeError",
        "exception_message": message,
        "exception_traceback": (
            'Traceback (most recent call last):\n'
            '  File "bench/harbor_adapter.py", line 1, in run\n'
            "RuntimeError: " + message
        ),
    }

    assert (
        HarborRunner._environment_exception_evidence(
            exception,
            trial_dir=tmp_path,
        )
        == {}
    )


def test_prefixed_worker_timeout_type_is_not_harbor_provenance(
    tmp_path: Path,
) -> None:
    exception = {
        "exception_type": "worker.fake.EnvironmentStartTimeoutError",
        "exception_message": "Environment start timed out",
    }

    assert (
        HarborRunner._environment_exception_evidence(
            exception,
            trial_dir=tmp_path,
        )
        == {}
    )


def test_network_marker_without_failure_evidence_is_not_infrastructure() -> None:
    runner = HarborRunner()
    trial = _trial(
        "ordinary task failure",
        metadata={
            "network_hardened_environment_marker": True,
            "environment_start_evidence": "network hardening configured",
        },
    )

    assert runner.is_infra_error(trial) is False


def test_stale_infra_flags_require_current_provenance() -> None:
    runner = HarborRunner()
    trial = _trial(
        "agent command failed",
        metadata={
            "infra_error_detected": True,
            "score_exclusion_reason": "infrastructure_error",
            "trusted_infrastructure_failure": True,
        },
    )

    assert runner.is_infra_error(trial) is False


def test_timeout_phase_without_owned_evidence_is_not_infrastructure() -> None:
    runner = HarborRunner()
    trial = _trial(
        "build process exited",
        status=TrialStatus.TIMEOUT,
        metadata={"timeout_phase": "environment_build"},
    )

    assert runner.is_infra_error(trial) is False


def test_verifier_prepare_flag_without_owned_evidence_is_not_infrastructure() -> None:
    runner = HarborRunner()
    trial = _trial(
        "verifier preparation failed",
        metadata={
            "timeout_phase": "verifier_runtime_prepare",
            "verifier_runtime_prepare_timeout": True,
        },
    )

    assert runner.is_infra_error(trial) is False


def test_trusted_phase_label_without_owned_evidence_is_not_infrastructure() -> None:
    runner = HarborRunner()
    trial = _trial(
        "environment setup failed",
        metadata={"infrastructure_phase": "environment_setup"},
    )

    assert runner.is_infra_error(trial) is False


def test_phase_owned_endpoint_plus_transport_failure_is_infrastructure() -> None:
    runner = HarborRunner()
    trial = _trial(
        "environment startup failed",
        metadata={
            "environment_start_evidence": (
                "pip index https://pypi.org/simple failed: "
                "Temporary failure resolving pypi.org"
            )
        },
    )

    assert runner.is_infra_error(trial) is True


def test_structured_environment_phase_is_infrastructure() -> None:
    runner = HarborRunner()
    trial = _trial(
        "build process exited",
        status=TrialStatus.TIMEOUT,
        metadata={
            "timeout_phase": "environment_build",
            "environment_build_evidence": "Environment build timed out",
        },
    )

    assert runner.is_infra_error(trial) is True


def test_missing_harbor_executable_is_trusted_launch_infrastructure(tmp_path: Path) -> None:
    runner = HarborRunner(
        harbor_bin=str(tmp_path / "missing-harbor"),
        jobs_dir=tmp_path,
    )

    trial = runner._run_task_once(
        task_id="task",
        agent_config={},
        timeout_audit=30,
        job_name="missing-cli",
        jobs_dir=tmp_path,
    )

    assert trial.status == TrialStatus.ERROR
    assert trial.metadata["infrastructure_phase"] == "harbor_launch"
    assert trial.metadata["harbor_launch_evidence"]["kind"] == "executable_not_found"
    assert runner.is_infra_error(trial) is True
    assert runner._should_retry_infra_failure(trial, infra_error=True) is False
    assert (
        trial.metadata["infra_retry_suppressed_reason"]
        == "deterministic_harbor_launch_failure"
    )


def test_pre_result_docker_text_is_not_trusted_process_provenance(
    tmp_path: Path,
) -> None:
    runner = HarborRunner(jobs_dir=tmp_path)

    def fake_run_command(
        self: HarborRunner,
        argv: list[str],
        *,
        timeout_audit: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            1,
            "",
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
            "Is the docker daemon running?",
        )

    runner._run_command = MethodType(fake_run_command, runner)  # type: ignore[method-assign]
    trial = runner._run_task_once(
        task_id="task",
        agent_config={},
        timeout_audit=30,
        job_name="docker-looking-text",
        jobs_dir=tmp_path,
    )

    assert "infrastructure_phase" not in trial.metadata
    assert "harbor_launch_evidence" not in trial.metadata
    assert runner.is_infra_error(trial) is False


def test_pre_result_network_text_is_not_trusted_process_provenance(
    tmp_path: Path,
) -> None:
    runner = HarborRunner(jobs_dir=tmp_path)

    def fake_run_command(
        self: HarborRunner,
        argv: list[str],
        *,
        timeout_audit: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            1,
            "",
            "registry-1.docker.io: Temporary failure resolving registry-1.docker.io",
        )

    runner._run_command = MethodType(fake_run_command, runner)  # type: ignore[method-assign]
    trial = runner._run_task_once(
        task_id="task",
        agent_config={},
        timeout_audit=30,
        job_name="network-looking-text",
        jobs_dir=tmp_path,
    )

    assert "infrastructure_phase" not in trial.metadata
    assert "early_environment_setup_evidence" not in trial.metadata
    assert runner.is_infra_error(trial) is False


def test_process_text_is_not_reclassified_after_trial_evidence_exists(
    tmp_path: Path,
) -> None:
    runner = HarborRunner(jobs_dir=tmp_path)
    trial_root = tmp_path / "has-trial" / "task__attempt" / "agent"
    trial_root.mkdir(parents=True)

    def fake_run_command(
        self: HarborRunner,
        argv: list[str],
        *,
        timeout_audit: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            1,
            "",
            "Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
        )

    runner._run_command = MethodType(fake_run_command, runner)  # type: ignore[method-assign]
    trial = runner._run_task_once(
        task_id="task",
        agent_config={},
        timeout_audit=30,
        job_name="has-trial",
        jobs_dir=tmp_path,
    )

    assert "infrastructure_phase" not in trial.metadata
    assert runner.is_infra_error(trial) is False


def test_configured_retry_reference_does_not_stop_recovery() -> None:
    runner = HarborRunner()
    trial = _trial(
        "Temporary failure resolving pypi.org",
        metadata={
            "infra_retry_attempt": 1,
            "infra_retries_audit_only": 1,
        },
    )

    assert runner._should_retry_infra_failure(trial, infra_error=True) is True
    assert "infra_retry_limit_reached" not in trial.metadata
    assert "infra_retry_suppressed_reason" not in trial.metadata


def test_repeated_failure_signature_does_not_stop_recovery() -> None:
    runner = HarborRunner()
    first = _trial(
        "Temporary failure resolving pypi.org",
        metadata={
            "infra_retry_attempt": 0,
            "infra_retries_audit_only": 5,
            "timeout_phase": "environment_start",
            "environment_start_evidence": (
                "https://pypi.org/simple failed: Temporary failure resolving pypi.org"
            ),
        },
    )
    repeated = _trial(
        "Temporary failure resolving pypi.org",
        metadata={
            "infra_retry_attempt": 1,
            "infra_retries_audit_only": 5,
            "timeout_phase": "environment_start",
            "environment_start_evidence": (
                "https://pypi.org/simple failed: Temporary failure resolving pypi.org"
            ),
        },
    )

    assert runner._should_retry_infra_failure(first, infra_error=True) is True
    assert runner._should_retry_infra_failure(repeated, infra_error=True) is True
    assert "infra_retry_signature_repeated" not in repeated.metadata
    assert "infra_retry_suppressed_reason" not in repeated.metadata


def test_run_task_continues_repeated_infrastructure_until_task_evidence() -> None:
    runner = HarborRunner()
    calls = 0

    def fake_run_once(self: HarborRunner, **kwargs: object) -> TrialResult:
        nonlocal calls
        calls += 1
        infrastructure = calls < 3
        trial = TrialResult(
            trial_id=f"trial-{calls}",
            task_id="task",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.MEDIUM,
            status=TrialStatus.ERROR if infrastructure else TrialStatus.FAILED,
            score=0.0,
            verified=not infrastructure,
            error_log=["Harbor environment startup failed"] if infrastructure else [],
            metadata=(
                {
                    "timeout_phase": "environment_start",
                    "environment_start_evidence": (
                        "https://pypi.org/simple failed: "
                        "Temporary failure resolving pypi.org"
                    ),
                }
                if infrastructure
                else {}
            ),
        )
        # The raw stream is deliberately redundant; phase-owned evidence above,
        # not this Worker-visible text, is what permits attribution.
        trial.harbor_stderr = (
            "https://pypi.org/simple failed: Temporary failure resolving pypi.org"
        )
        return trial

    def ignore_materialization(self: HarborRunner, trial: TrialResult) -> None:
        return None

    runner._run_task_once = MethodType(fake_run_once, runner)  # type: ignore[method-assign]
    runner._materialize_trial = MethodType(  # type: ignore[method-assign]
        ignore_materialization,
        runner,
    )

    result = runner.run_task(
        "task",
        {"infra_retries": 5},
        job_name="deterministic-infra",
    )

    assert calls == 3
    assert result.metadata["infra_retry_limit_enforced"] is False
    assert result.metadata["infra_retry_unbounded_by_attempt_count"] is True
    assert result.metadata["infra_retry_failure_signature_deduplication"] is False
    assert len(result.metadata["infra_retry_attempts"]) == 3
