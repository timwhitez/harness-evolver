"""Harbor orchestration with provenance-aware infrastructure attribution."""

from __future__ import annotations

from contextvars import ContextVar
import errno
import hashlib
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from bench import _harbor_issue5_base as _base

for _name in dir(_base):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_base, _name))


_INFRA_PHASE_TEXT_PATTERNS = (
    "docker compose command failed for environment",
    "environment start timed out",
    "environmentstarttimeouterror",
    "environment build timed out",
    "environmentbuildtimeouterror",
    "verifier runtime network preparation timed out",
    "verifier runtime prepare timed out",
    "prebuilt docker image cache warmup failed",
    "prebuilt docker image cache warmup timed out",
    "failed to write to the distribution cache",
    "failed to rename file from /tmp/hl-verifier-cache",
)
_INFRA_TIMEOUT_PHASES = frozenset(
    {
        "environment_start",
        "environment_build",
        "verifier_runtime_prepare",
    }
)
_TRUSTED_INFRA_PHASES = _INFRA_TIMEOUT_PHASES | {
    "harbor_launch",
    "environment_setup",
}
_STRUCTURED_EVIDENCE_KEYS = (
    "harbor_launch_evidence",
    "early_environment_setup_evidence",
    "environment_start_evidence",
    "environment_build_evidence",
    "environment_failure_evidence",
    "verifier_runtime_prepare_evidence",
    "docker_image_validation_evidence",
    "prebuilt_image_cache_evidence",
)
_DOCKER_DAEMON_PATTERNS = (
    "cannot connect to the docker daemon",
    "docker daemon is not running",
    "error during connect",
    "permission denied while trying to connect to the docker daemon socket",
    "is the docker daemon running",
)
_DETERMINISTIC_LAUNCH_KINDS = frozenset(
    {
        "executable_not_found",
        "executable_not_permitted",
        "executable_format_error",
    }
)
_SEEN_INFRA_SIGNATURES: ContextVar[set[str] | None] = ContextVar(
    "harness_evolver_seen_infra_signatures",
    default=None,
)


def _normalise_failure_text(text: str) -> str:
    compact = " ".join(str(text).lower().split())
    compact = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", compact)
    return compact[-4000:]


class HarborRunner(_base.HarborRunner):
    """HarborRunner with finite, provenance-aware infrastructure recovery."""

    def run_task(
        self,
        task_id: str,
        agent_config: dict[str, Any],
        timeout: int | None = None,
        *,
        timeout_audit: int | None = None,
        job_name: str | None = None,
        jobs_dir: str | _base.Path | None = None,
    ) -> _base.TrialResult:
        token = _SEEN_INFRA_SIGNATURES.set(set())
        try:
            trial = super().run_task(
                task_id,
                agent_config,
                timeout,
                timeout_audit=timeout_audit,
                job_name=job_name,
                jobs_dir=jobs_dir,
            )
            self._mark_retry_policy_finite(trial)
            # The baseline materializes before returning. Persist corrected
            # attribution and finite-policy metadata as the final snapshot.
            self._materialize_trial(trial)
            return trial
        finally:
            _SEEN_INFRA_SIGNATURES.reset(token)

    def _run_task_once(
        self,
        *,
        task_id: str,
        agent_config: dict[str, Any],
        timeout_audit: int,
        job_name: str | None,
        jobs_dir: str | Path | None,
    ) -> _base.TrialResult:
        """Run one Harbor command while recording provenance at the failure site."""

        command = self.build_command(
            task_id,
            agent_config,
            job_name=job_name,
            jobs_dir=jobs_dir,
        )
        start_time = time.time()

        try:
            completed = self._run_command(command.argv, timeout_audit=timeout_audit)
            wall_time = time.time() - start_time
            trial = self.parse_job_dir(
                command.job_dir,
                task_id=task_id,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                wall_time=wall_time,
                agent_config=agent_config,
            )
            evidence = self._trusted_early_process_failure(command, completed)
            if evidence is not None:
                phase, payload = evidence
                trial.metadata["infrastructure_phase"] = phase
                key = (
                    "harbor_launch_evidence"
                    if phase == "harbor_launch"
                    else "early_environment_setup_evidence"
                )
                trial.metadata[key] = payload
            return trial
        except FileNotFoundError:
            return self._launch_exception_trial(
                command=command,
                task_id=task_id,
                agent_config=agent_config,
                start_time=start_time,
                kind="executable_not_found",
            )
        except PermissionError:
            return self._launch_exception_trial(
                command=command,
                task_id=task_id,
                agent_config=agent_config,
                start_time=start_time,
                kind="executable_not_permitted",
            )
        except OSError as exc:
            launch_kind = {
                errno.ENOENT: "executable_not_found",
                errno.EACCES: "executable_not_permitted",
                errno.ENOEXEC: "executable_format_error",
            }.get(exc.errno)
            if launch_kind is not None:
                return self._launch_exception_trial(
                    command=command,
                    task_id=task_id,
                    agent_config=agent_config,
                    start_time=start_time,
                    kind=launch_kind,
                )
            return self._generic_runner_exception_trial(
                command=command,
                task_id=task_id,
                agent_config=agent_config,
                start_time=start_time,
                exc=exc,
            )
        except subprocess.TimeoutExpired as exc:
            partial = self._partial_timeout_trial(
                job_dir=command.job_dir,
                task_id=task_id,
                timeout_audit=timeout_audit,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                agent_config=agent_config,
            )
            if partial is not None:
                return partial
            return _base.TrialResult(
                trial_id=command.job_name,
                task_id=task_id,
                task_domain=_base.TaskDomain.SOFTWARE_ENGINEERING,
                task_difficulty=_base.TaskDifficulty.MEDIUM,
                status=_base.TrialStatus.TIMEOUT,
                error_log=[
                    "Harbor command entered the external-interruption "
                    f"compatibility path after audit reference {timeout_audit}s"
                ],
                wall_time_seconds=float(timeout_audit),
                harbor_job_dir=str(command.job_dir),
                harbor_stdout=exc.stdout or "",
                harbor_stderr=exc.stderr or "",
                metadata={
                    "model_config": self._model_config_metadata(agent_config),
                    "timeout_phase": "harbor_process",
                    "timeout_source": "outer_harbor_command_interrupted",
                    "timeout_seconds": timeout_audit,
                    "outer_harbor_timeout_seconds_audit_only": timeout_audit,
                    "outer_harbor_timeout_stop_condition": False,
                    "outer_harbor_timeout_loop_stop_condition": False,
                    "timeout_expired_exception_stop_condition": False,
                    "time_round_token_limit_driven": False,
                    "partial_harbor_artifacts": False,
                },
            )
        except Exception as exc:
            return self._generic_runner_exception_trial(
                command=command,
                task_id=task_id,
                agent_config=agent_config,
                start_time=start_time,
                exc=exc,
            )

    def _trusted_early_process_failure(
        self,
        command: _base.HarborCommand,
        completed: subprocess.CompletedProcess[str],
    ) -> tuple[str, dict[str, Any]] | None:
        """Classify only runner-owned failures before any trial result exists."""

        if completed.returncode == 0 or self._job_has_trial_evidence(command.job_dir):
            return None
        text = "\n".join([completed.stdout or "", completed.stderr or ""])
        lowered = text.lower()
        tail = _normalise_failure_text(text)
        if any(pattern in lowered for pattern in _DOCKER_DAEMON_PATTERNS):
            return (
                "harbor_launch",
                {
                    "kind": "docker_daemon_unavailable",
                    "returncode": completed.returncode,
                    "evidence_tail": tail,
                },
            )
        if self._structured_text_is_infrastructure(text):
            return (
                "environment_setup",
                {
                    "kind": "early_network_or_environment_failure",
                    "returncode": completed.returncode,
                    "evidence_tail": tail,
                },
            )
        return None

    @staticmethod
    def _job_has_trial_evidence(job_dir: Path) -> bool:
        if (job_dir / "result.json").is_file():
            return True
        if not job_dir.exists():
            return False
        try:
            return any(job_dir.glob("*/result.json")) or any(
                (path / "agent").exists() or (path / "verifier").exists()
                for path in job_dir.iterdir()
                if path.is_dir()
            )
        except OSError:
            return False

    def _launch_exception_trial(
        self,
        *,
        command: _base.HarborCommand,
        task_id: str,
        agent_config: dict[str, Any],
        start_time: float,
        kind: str,
    ) -> _base.TrialResult:
        executable = str(command.argv[0]) if command.argv else self.harbor_bin
        return _base.TrialResult(
            trial_id=command.job_name,
            task_id=task_id,
            task_domain=_base.TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=_base.TaskDifficulty.MEDIUM,
            status=_base.TrialStatus.ERROR,
            error_log=[f"Harbor launch failed: {kind} ({executable})"],
            wall_time_seconds=time.time() - start_time,
            harbor_job_dir=str(command.job_dir),
            metadata={
                "model_config": self._model_config_metadata(agent_config),
                "infrastructure_phase": "harbor_launch",
                "harbor_launch_evidence": {
                    "kind": kind,
                    "executable": executable,
                },
            },
        )

    def _generic_runner_exception_trial(
        self,
        *,
        command: _base.HarborCommand,
        task_id: str,
        agent_config: dict[str, Any],
        start_time: float,
        exc: Exception,
    ) -> _base.TrialResult:
        return _base.TrialResult(
            trial_id=command.job_name,
            task_id=task_id,
            task_domain=_base.TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=_base.TaskDifficulty.MEDIUM,
            status=_base.TrialStatus.ERROR,
            error_log=[str(exc)],
            wall_time_seconds=time.time() - start_time,
            harbor_job_dir=str(command.job_dir),
            metadata={"model_config": self._model_config_metadata(agent_config)},
        )

    def is_infra_error(self, trial: _base.TrialResult) -> bool:
        """Require a trusted phase, flag, or phase-owned evidence field."""

        metadata = trial.metadata or {}
        phase = str(metadata.get("infrastructure_phase") or "")
        if phase in _TRUSTED_INFRA_PHASES:
            return True
        timeout_phase = str(metadata.get("timeout_phase") or "")
        if timeout_phase in _INFRA_TIMEOUT_PHASES:
            return True
        if metadata.get("verifier_runtime_prepare_timeout"):
            return True
        if metadata.get("terminal_environment_unavailable"):
            return True
        if metadata.get("docker_image_validation_failed"):
            return True
        if metadata.get("prebuilt_image_cache_miss_detected"):
            return True

        # Harbor stdout/stderr, verifier logs, and Worker error text are untyped
        # streams. Only runner-owned phase fields may carry textual evidence.
        return self._structured_text_is_infrastructure(
            self._structured_environment_evidence(metadata)
        )

    def _is_infra_text(self, text: str) -> bool:
        """Never infer infrastructure from an untyped verifier/Worker log."""

        return False

    def _structured_text_is_infrastructure(self, text: str) -> bool:
        lowered = str(text).lower()
        if not lowered.strip():
            return False
        if any(pattern in lowered for pattern in _INFRA_PHASE_TEXT_PATTERNS):
            return True
        return (
            any(
                str(pattern).lower() in lowered
                for pattern in _base.INFRA_NETWORK_ENDPOINT_PATTERNS
            )
            and any(
                str(pattern).lower() in lowered
                for pattern in _base.INFRA_NETWORK_FAILURE_PATTERNS
            )
        )

    def _structured_environment_evidence(self, metadata: dict[str, Any]) -> str:
        return "\n".join(
            str(metadata.get(key) or "") for key in _STRUCTURED_EVIDENCE_KEYS
        )

    def _should_retry_infra_failure(
        self,
        trial: _base.TrialResult,
        *,
        infra_error: bool,
    ) -> bool:
        if not super()._should_retry_infra_failure(trial, infra_error=infra_error):
            return False

        metadata = trial.metadata
        launch_evidence = metadata.get("harbor_launch_evidence")
        if isinstance(launch_evidence, dict) and launch_evidence.get("kind") in (
            _DETERMINISTIC_LAUNCH_KINDS
        ):
            metadata["infra_retry_suppressed_reason"] = "deterministic_harbor_launch_failure"
            metadata["infra_retry_loop_stop_condition"] = True
            return False

        attempt_index = self._nonnegative_int(metadata.get("infra_retry_attempt"), 0)
        configured_retries = self._nonnegative_int(
            metadata.get("infra_retries_audit_only"),
            self.default_infra_retries,
        )
        metadata["infra_retry_limit"] = configured_retries
        metadata["infra_retry_limit_enforced"] = True
        metadata["infra_retry_unbounded_by_attempt_count"] = False

        if attempt_index >= configured_retries:
            metadata["infra_retry_suppressed_reason"] = (
                "configured_infrastructure_retry_limit"
            )
            metadata["infra_retry_limit_reached"] = True
            metadata["infra_retries_stop_condition"] = True
            metadata["infra_retry_attempt_count_stop_condition"] = True
            metadata["infra_retry_loop_stop_condition"] = True
            return False

        signature = self._infra_failure_signature(trial)
        metadata["infra_failure_signature"] = signature
        seen = _SEEN_INFRA_SIGNATURES.get()
        if seen is None:
            seen = getattr(self, "_seen_infra_failure_signatures", None)
            if seen is None:
                seen = set()
                self._seen_infra_failure_signatures = seen
        if signature in seen:
            metadata["infra_retry_suppressed_reason"] = (
                "repeated_infrastructure_failure_signature"
            )
            metadata["infra_retry_signature_repeated"] = True
            metadata["infra_retries_stop_condition"] = True
            metadata["infra_retry_attempt_count_stop_condition"] = False
            metadata["infra_retry_loop_stop_condition"] = True
            return False

        seen.add(signature)
        metadata["infra_retry_signature_repeated"] = False
        metadata["infra_retries_stop_condition"] = False
        return True

    def _infra_failure_signature(self, trial: _base.TrialResult) -> str:
        metadata = trial.metadata or {}
        evidence = "\n".join(
            [
                str(metadata.get("infrastructure_phase") or ""),
                str(metadata.get("timeout_phase") or ""),
                self._structured_environment_evidence(metadata),
            ]
        )
        material = "|".join(
            [
                str(trial.status.value),
                str(metadata.get("infrastructure_phase") or ""),
                str(metadata.get("timeout_phase") or ""),
                str(bool(metadata.get("terminal_environment_unavailable"))),
                _normalise_failure_text(evidence),
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _mark_retry_policy_finite(self, trial: _base.TrialResult) -> None:
        metadata = trial.metadata
        configured_retries = self._nonnegative_int(
            metadata.get("infra_retries_configured"),
            self.default_infra_retries,
        )
        metadata.update(
            {
                "infra_retry_limit": configured_retries,
                "infra_retry_limit_enforced": True,
                "infra_retry_unbounded_by_attempt_count": False,
                "infra_retry_failure_signature_deduplication": True,
                "infra_retry_loop_stop_condition": True,
            }
        )
        current_infra = self.is_infra_error(trial)
        metadata["infra_error_detected"] = current_infra
        if not current_infra:
            metadata["verifier_infra_error"] = False
            if metadata.get("score_exclusion_reason") == "infrastructure_error":
                metadata.pop("score_exclusion_reason", None)
        for attempt in metadata.get("infra_retry_attempts") or []:
            if not isinstance(attempt, dict):
                continue
            attempt["infra_retry_limit"] = configured_retries
            attempt["infra_retry_limit_enforced"] = True
            attempt["infra_retry_unbounded_by_attempt_count"] = False
            attempt["infra_retry_failure_signature_deduplication"] = True

    @staticmethod
    def _nonnegative_int(value: Any, default: int) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return max(0, int(default))
