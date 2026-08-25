"""HarborRunner: TerminalBench/Harbor orchestration and artifact parsing.

The runner deliberately treats Harbor job directories as the source of truth.
Worker self-reports are never converted into passes; a pass requires verifier
reward data in Harbor's result artifacts.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bench.trajectory import TrajectoryReader
from bench.network_environment import (
    DEFAULT_DOWNLOAD_URL_REWRITES,
    DEFAULT_DOCKER_CPUS,
    DEFAULT_DOCKER_HUB_MIRROR,
    DEFAULT_DOCKER_LABELS,
    DEFAULT_DOCKER_LOG_MAX_FILE,
    DEFAULT_DOCKER_LOG_MAX_SIZE,
    DEFAULT_DOCKER_MEMORY,
    DEFAULT_DOCKER_MEMORY_SWAP,
    DEFAULT_DOCKER_PIDS_LIMIT,
    DEFAULT_PREBUILT_DOCKER_HUB_MIRROR,
    DEFAULT_PYPI_INDEX_URL,
    DEFAULT_PYPI_TRUSTED_HOST,
    _parse_docker_labels,
)
from hl.types import TaskDifficulty, TaskDomain, TrialResult, TrialStatus
from harness.tools.shell import external_agent_command_reason


DEFAULT_WORKER_IMPORT_PATH = "bench.harbor_adapter:HLWorkerHarborAgent"
DEFAULT_NETWORK_ENVIRONMENT_IMPORT_PATH = (
    "bench.network_environment:AptMirrorDockerEnvironment"
)
WORKER_TLS_ENV_KEYS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
EXTERNAL_AGENT_NAMES = {
    "aider",
    "amp",
    "claude",
    "claude-code",
    "codex",
    "cursor-agent",
    "droid",
    "factory",
    "factory-droid",
    "forgecode",
    "gemini",
    "gemini-cli",
    "openai-codex",
    "opencode",
}

INFRA_ERROR_PATTERNS = [
    "Docker compose command failed for environment",
    "Environment start timed out",
    "EnvironmentStartTimeoutError",
    "Verifier runtime network preparation timed out",
    "verifier runtime prepare timed out",
    "hl-verifier-network-prepared",
    "Prebuilt Docker image cache warmup failed",
    "Prebuilt Docker image cache warmup timed out",
    "failed to solve: process",
    "did not complete successfully: exit code: 5",
    "403 Forbidden",
    "pull access denied",
    "denied: requested access",
    "manifest unknown",
    "502 Bad Gateway",
    "504 Gateway Timeout",
    "no such host",
    "Temporary failure resolving",
    "Could not resolve host",
    "failed to resolve source metadata",
    "failed to resolve reference",
    "failed to fetch",
    "Unable to fetch some archives",
    "Could not get lock",
    "Unable to acquire the dpkg frontend lock",
    "Unable to lock directory",
    "is another process using it",
    "held by process",
    "Unable to locate package",
    "E: Failed to fetch",
    "Hash Sum mismatch",
    "Clearsigned file isn't valid",
    "does not have a Release file",
    "network is unreachable",
    "connection timed out",
    "TLS handshake timeout",
    "i/o timeout",
    "SSLCertVerificationError",
    "certificate verify failed",
    "SSL certificate problem",
    "unable to get local issuer certificate",
    "curl: (60)",
    "/root/.local/bin/env",
    "uvx: command not found",
    "uvx command not found",
    "Failed to write to the distribution cache",
    "failed to rename file from /tmp/hl-verifier-cache",
    "No matching distribution found",
]
INFRA_NETWORK_ENDPOINT_PATTERNS = [
    "dl-cdn.alpinelinux.org",
    "registry-1.docker.io",
    "deb.debian.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "pypi.org",
    "files.pythonhosted.org",
]
INFRA_NETWORK_FAILURE_PATTERNS = [
    "502 bad gateway",
    "504 gateway timeout",
    "no such host",
    "temporary failure resolving",
    "could not resolve host",
    "failed to resolve",
    "failed to fetch",
    "unable to fetch some archives",
    "network is unreachable",
    "connection timed out",
    "tls handshake timeout",
    "i/o timeout",
    "certificate verify failed",
    "ssl certificate problem",
    "curl: (6)",
    "curl: (7)",
    "curl: (28)",
    "curl: (35)",
    "curl: (56)",
    "curl: (60)",
]

VERIFIER_LOG_MAX_CHARS = 12000
ENVIRONMENT_EVIDENCE_LOG_MAX_CHARS = 12000
ENVIRONMENT_EVIDENCE_DOCKERFILE_MAX_CHARS = 20000
ENVIRONMENT_CONFIG_KWARG_EVIDENCE_KEYS = (
    "apt_mirror_enabled",
    "debian_mirror",
    "debian_security_mirror",
    "ubuntu_mirror",
    "docker_hub_mirror",
    "prebuilt_docker_hub_mirror",
    "docker_image_overrides",
    "download_url_rewrites",
    "pypi_index_url",
    "pypi_trusted_host",
    "apt_retries",
    "apt_timeout_seconds",
    "pip_retries",
    "pip_timeout_seconds",
    "prebuilt_docker_pull_timeout_seconds",
    "bootstrap_ca_certificates",
    "download_retry_wrapper",
    "inject_host_ca_into_build",
    "host_ca_cert_bundle",
)
DISALLOWED_TIMEOUT_RESOURCE_CONFIG_KEYS = (
    "timeout_multiplier",
    "agent_timeout_multiplier",
    "verifier_timeout_multiplier",
    "agent_setup_timeout_multiplier",
    "environment_build_timeout_multiplier",
    "storage",
    "storage_gb",
    "resources",
    "override_storage_mb",
    "override_gpus",
)


@dataclass(frozen=True)
class HarborCommand:
    """A concrete Harbor CLI invocation plus expected output locations."""

    argv: list[str]
    job_name: str
    jobs_dir: Path
    job_dir: Path
    config: dict[str, Any]

    def shell_command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.argv)


@dataclass
class HarborRunner:
    """Runs TerminalBench tasks via the installed Harbor CLI."""

    harbor_bin: str = "harbor"
    dataset_path: str | None = "terminal-bench-tasks/terminal-bench"
    dataset_name: str | None = None
    jobs_dir: Path = Path("jobs")
    output_dir: Path = Path("trials/runs")
    worker_import_path: str = DEFAULT_WORKER_IMPORT_PATH
    max_parallel: int = 1
    # Compatibility/audit value only. The harness must not impose a host-side
    # wall-clock stop on Harbor; official Harbor/TerminalBench limits own task
    # termination.
    default_timeout: int = 1800
    # Compatibility/audit value only. Infrastructure recovery attempts are not
    # capped by this value; recovery keeps going until Harbor produces verifier
    # evidence, a non-infrastructure task result, or the process is cancelled.
    default_infra_retries: int = 0
    default_network_hardened_environment: bool = True

    _running_tasks: dict[str, subprocess.Popen] = field(default_factory=dict)

    def build_command(
        self,
        task_id: str,
        agent_config: dict[str, Any] | None = None,
        *,
        job_name: str | None = None,
        jobs_dir: str | Path | None = None,
    ) -> HarborCommand:
        """Build an installed-Harbor-compatible command for one task."""
        agent_config = agent_config or {}
        self._reject_timeout_resource_overrides(agent_config)
        resolved_jobs_dir = Path(jobs_dir) if jobs_dir is not None else self.jobs_dir
        resolved_job_name = job_name or self._default_job_name(task_id)
        agent_name = agent_config.get("agent") or agent_config.get("agent_type") or "hl-worker"
        self._reject_external_agent_delegate(str(agent_name))
        model = agent_config.get("model") or agent_config.get("model_name")

        argv = [
            self.harbor_bin,
            "run",
            "--job-name",
            resolved_job_name,
            "--jobs-dir",
            str(resolved_jobs_dir),
            "--n-concurrent",
            str(self.max_parallel),
        ]
        n_attempts = self._n_attempts(agent_config)
        if n_attempts is not None:
            argv.extend(["--n-attempts", str(n_attempts)])

        env_file = agent_config.get("env_file")
        if env_file:
            argv.extend(["--env-file", str(env_file)])

        force_build = bool(agent_config.get("force_build"))
        if force_build:
            argv.append("--force-build")
        argv.append("--delete")

        docker_resources = self._docker_resource_config(agent_config)
        if docker_resources["enabled"]:
            cpus = docker_resources.get("cpus")
            memory_mb = docker_resources.get("memory_mb")
            if cpus is not None:
                argv.extend(["--override-cpus", str(cpus)])
            if memory_mb is not None:
                argv.extend(["--override-memory-mb", str(memory_mb)])

        mounts_json = agent_config.get("mounts_json")
        if mounts_json is not None:
            argv.extend(["--mounts-json", self._mounts_json_argument(mounts_json)])

        if agent_config.get("yes"):
            argv.append("--yes")

        environment_import_path = agent_config.get("environment_import_path")
        network_hardened = agent_config.get(
            "network_hardened_environment",
            self.default_network_hardened_environment,
        )
        if network_hardened and not environment_import_path:
            environment_import_path = DEFAULT_NETWORK_ENVIRONMENT_IMPORT_PATH
        if environment_import_path:
            argv.extend(["--environment-import-path", str(environment_import_path)])

        environment_kwargs = self._environment_kwargs(agent_config)
        if environment_import_path == DEFAULT_NETWORK_ENVIRONMENT_IMPORT_PATH:
            environment_kwargs.setdefault("docker_hub_mirror", DEFAULT_DOCKER_HUB_MIRROR)
            environment_kwargs.setdefault(
                "prebuilt_docker_hub_mirror",
                DEFAULT_PREBUILT_DOCKER_HUB_MIRROR,
            )
            environment_kwargs.setdefault("download_url_rewrites", DEFAULT_DOWNLOAD_URL_REWRITES)
            environment_kwargs.setdefault("pypi_index_url", DEFAULT_PYPI_INDEX_URL)
            environment_kwargs.setdefault("pypi_trusted_host", DEFAULT_PYPI_TRUSTED_HOST)
        for key, value in environment_kwargs.items():
            if value is None:
                continue
            argv.extend(["--environment-kwarg", f"{key}={self._kwarg_value(value)}"])

        if self.dataset_path:
            argv.extend(["--path", self.dataset_path, "--include-task-name", task_id])
            dataset_config: dict[str, Any] = {
                "path": self.dataset_path,
                "include_task_name": task_id,
            }
        elif self.dataset_name:
            argv.extend(["--dataset", self.dataset_name, "--include-task-name", task_id])
            dataset_config = {
                "dataset": self.dataset_name,
                "include_task_name": task_id,
            }
        else:
            argv.extend(["--task", task_id])
            dataset_config = {"task": task_id}

        if agent_name == "hl-worker":
            argv.extend(["--agent-import-path", self.worker_import_path])
        else:
            argv.extend(["--agent", agent_name])

        if model:
            argv.extend(["--model", str(model)])

        for key, value in self._agent_kwargs(agent_config).items():
            if value is None:
                continue
            argv.extend(["--agent-kwarg", f"{key}={value}"])

        for key, value in self._agent_env(agent_config).items():
            if value is None:
                continue
            argv.extend(["--agent-env", f"{key}={value}"])

        verifier_env = self._env_list(agent_config.get("verifier_env"))
        for item in verifier_env:
            argv.extend(["--verifier-env", item])

        return HarborCommand(
            argv=argv,
            job_name=resolved_job_name,
            jobs_dir=resolved_jobs_dir,
            job_dir=resolved_jobs_dir / resolved_job_name,
            config={
                "job_name": resolved_job_name,
                "jobs_dir": str(resolved_jobs_dir),
                "dataset": dataset_config,
                "agent": {
                    "name": agent_name,
                    "import_path": self.worker_import_path if agent_name == "hl-worker" else None,
                    "role": agent_config.get("worker_role"),
                    "model": model,
                    "kwargs": self._agent_kwargs(agent_config),
                    "env": self._agent_env(agent_config),
                },
                "n_concurrent_trials": self.max_parallel,
                "n_attempts": n_attempts,
                "env_file": str(env_file) if env_file else None,
                "timeouts": {},
                "environment": {
                    "force_build": force_build,
                    "delete": True,
                    "build_timeout_multiplier": None,
                    "docker_resources": docker_resources,
                    "mounts_json": mounts_json,
                    "import_path": str(environment_import_path) if environment_import_path else None,
                    "kwargs": environment_kwargs,
                },
                "verifier_env": verifier_env,
                "yes": bool(agent_config.get("yes")),
            },
        )

    def _reject_timeout_resource_overrides(self, agent_config: dict[str, Any]) -> None:
        disallowed = [
            key
            for key in DISALLOWED_TIMEOUT_RESOURCE_CONFIG_KEYS
            if agent_config.get(key) is not None
        ]
        if disallowed:
            joined = ", ".join(sorted(disallowed))
            raise ValueError(
                "Terminal-Bench 2.0 leaderboard runs must keep official task "
                "timeouts/resources unchanged; remove Harbor override(s): "
                f"{joined}. Timeout failures must be fixed by Worker strategy "
                "within the task's official limits."
            )

    def _reject_external_agent_delegate(self, agent_name: str) -> None:
        normalized = Path(agent_name).name.lower()
        external_agent_reason = external_agent_command_reason(agent_name)
        if normalized in EXTERNAL_AGENT_NAMES or external_agent_reason:
            raise ValueError(
                "Benchmark execution must use this repo's self-owned hl-worker. "
                "External coding agents are only master-orchestrator update/research "
                "tools and must not be selected as the Harbor task-solving agent. "
                "Only the master HL orchestrator may create sub-agents."
            )

    def run_task(
        self,
        task_id: str,
        agent_config: dict[str, Any],
        timeout: int | None = None,
        *,
        timeout_audit: int | None = None,
        job_name: str | None = None,
        jobs_dir: str | Path | None = None,
    ) -> TrialResult:
        """Run one task and normalize Harbor output into a TrialResult.

        ``timeout`` is retained for older callers as an audit reference only.
        New master/regression loop callers should pass ``timeout_audit`` so the
        parameter cannot be mistaken for a harness wall-clock deadline.
        """
        timeout_audit_value = timeout_audit if timeout_audit is not None else timeout
        timeout_audit_value = timeout_audit_value or self.default_timeout
        infra_retry_reference = self._infra_retries(agent_config)
        base_job_name = job_name or self._default_job_name(task_id)
        attempts: list[dict[str, Any]] = []
        final_trial: TrialResult | None = None

        attempt_index = 0
        while True:
            # Infrastructure retries are an evidence-recovery loop, not an
            # attempt-count loop. ``infra_retries`` is retained only as an audit
            # reference so repeated registry/network startup failures do not
            # end the campaign before Harbor produces usable task evidence.
            attempt_job_name = self._attempt_job_name(base_job_name, attempt_index)
            trial = self._run_task_once(
                task_id=task_id,
                agent_config=agent_config,
                timeout_audit=timeout_audit_value,
                job_name=attempt_job_name,
                jobs_dir=jobs_dir,
            )
            infra_error = self.is_infra_error(trial)
            trial.metadata["infra_error_detected"] = infra_error
            trial.metadata["infra_retry_attempt"] = attempt_index
            trial.metadata["infra_retry_attempt_index_audit_only"] = attempt_index
            trial.metadata["infra_retries_audit_only"] = infra_retry_reference
            trial.metadata["infra_retries_stop_condition"] = False
            trial.metadata["infra_retry_attempt_count_stop_condition"] = False
            trial.metadata["infra_retry_reference_stop_condition"] = False
            trial.metadata["infra_retry_loop_stop_condition"] = False
            trial.metadata["infra_retry_unbounded_by_attempt_count"] = True
            trial.metadata["time_round_token_limit_driven"] = False
            trial.metadata["infra_retry_reference_exceeded"] = (
                attempt_index > infra_retry_reference
            )
            attempts.append(
                {
                    "trial_id": trial.trial_id,
                    "job_dir": trial.harbor_job_dir,
                    "status": trial.status.value,
                    "verified": trial.verified,
                    "score": trial.score,
                    "infra_error_detected": infra_error,
                    "infra_retry_reference_exceeded": (
                        attempt_index > infra_retry_reference
                    ),
                    "infra_retry_attempt_index_audit_only": attempt_index,
                    "infra_retries_stop_condition": False,
                    "infra_retry_attempt_count_stop_condition": False,
                    "infra_retry_reference_stop_condition": False,
                    "infra_retry_loop_stop_condition": False,
                    "infra_retry_unbounded_by_attempt_count": True,
                    "time_round_token_limit_driven": False,
                }
            )
            final_trial = trial
            if self._should_retry_infra_failure(
                trial,
                infra_error=infra_error,
            ):
                retry_delay_seconds = 30 if attempt_index >= 5 else 2**attempt_index
                trial.metadata["infra_retry_scheduled"] = True
                trial.metadata["infra_retry_reference_remaining_audit_only"] = max(
                    0,
                    infra_retry_reference - attempt_index,
                )
                trial.metadata["infra_retry_delay_seconds_audit_only"] = retry_delay_seconds
                trial.metadata["infra_retry_delay_stop_condition"] = False
                trial.metadata["infra_retry_cooldown_stop_condition"] = False
                trial.metadata["infra_retry_delay_runtime_wait_condition"] = False
                trial.metadata["infra_retry_delay_wait_executed"] = False
                attempts[-1]["infra_retry_scheduled"] = True
                attempts[-1]["infra_retry_delay_seconds_audit_only"] = retry_delay_seconds
                attempts[-1]["infra_retry_delay_stop_condition"] = False
                attempts[-1]["infra_retry_cooldown_stop_condition"] = False
                attempts[-1]["infra_retry_delay_runtime_wait_condition"] = False
                attempts[-1]["infra_retry_delay_wait_executed"] = False
                self._materialize_trial(trial)
                # The backoff value is audit-only. Do not sleep or throttle this
                # recovery loop based on retry count or cooldown metadata.
                attempt_index += 1
                continue
            break

        assert final_trial is not None
        final_trial.metadata["outer_harbor_timeout_seconds_audit_only"] = timeout_audit_value
        final_trial.metadata["outer_harbor_timeout_stop_condition"] = False
        final_trial.metadata["infra_retry_attempts"] = attempts
        final_trial.metadata["infra_retries_configured"] = infra_retry_reference
        final_trial.metadata["infra_retries_audit_only"] = infra_retry_reference
        final_trial.metadata["infra_retries_stop_condition"] = False
        final_trial.metadata["infra_retry_attempt_count_stop_condition"] = False
        final_trial.metadata["infra_retry_reference_stop_condition"] = False
        final_trial.metadata["infra_retry_loop_stop_condition"] = False
        final_trial.metadata["infra_retry_unbounded_by_attempt_count"] = True
        final_trial.metadata["time_round_token_limit_driven"] = False
        final_trial.metadata["infra_retry_reference_exceeded"] = any(
            bool(attempt.get("infra_retry_reference_exceeded"))
            for attempt in attempts
        )
        if (
            (
                final_trial.status in {TrialStatus.ERROR, TrialStatus.TIMEOUT}
                and not final_trial.verified
            )
            or (
                final_trial.status == TrialStatus.FAILED
                and bool(final_trial.metadata.get("verifier_infra_error"))
            )
            or bool(final_trial.metadata.get("terminal_environment_unavailable"))
        ) and (
            final_trial.metadata.get("infra_error_detected")
            or self._is_retryable_infra_timeout_phase(final_trial)
            or bool(final_trial.metadata.get("verifier_infra_error"))
            or bool(final_trial.metadata.get("terminal_environment_unavailable"))
        ):
            final_trial.metadata["score_exclusion_reason"] = "infrastructure_error"
        self._materialize_trial(final_trial)
        return final_trial

    def _should_retry_infra_failure(
        self,
        trial: TrialResult,
        *,
        infra_error: bool,
    ) -> bool:
        """Return whether the current evidence is still infrastructure-only.

        This intentionally does not read ``infra_retries`` or an attempt count;
        those values are audit references, not retry-loop stop conditions.
        """
        if trial.verified or not infra_error:
            return False
        if self._is_nonretryable_prebuilt_warmup_failure(trial):
            trial.metadata["prebuilt_warmup_failure_nonretryable"] = True
            trial.metadata["infra_retry_suppressed_reason"] = (
                "deterministic_prebuilt_image_warmup_failure"
            )
            trial.metadata["infra_retry_suppressed_stop_condition"] = False
            return False
        if trial.status == TrialStatus.ERROR:
            return True
        if trial.status == TrialStatus.FAILED and bool(
            trial.metadata.get("verifier_infra_error")
        ):
            return True
        if bool(trial.metadata.get("terminal_environment_unavailable")):
            return True
        return self._is_retryable_infra_timeout_phase(trial)

    def _is_nonretryable_prebuilt_warmup_failure(self, trial: TrialResult) -> bool:
        text = self._trial_infra_text(trial).lower()
        if "prebuilt docker image cache warmup failed" not in text:
            return False
        return any(
            marker in text
            for marker in [
                "403 forbidden",
                "pull access denied",
                "denied: requested access",
                "manifest unknown",
                "repository does not exist",
                "not found",
            ]
        )

    def _trial_infra_text(self, trial: TrialResult) -> str:
        return "\n".join(
            [
                *trial.error_log,
                trial.harbor_stdout or "",
                trial.harbor_stderr or "",
                trial.verifier_output or "",
                str(trial.metadata.get("verifier_logs") or ""),
                str(trial.metadata.get("agent_exception_message") or ""),
            ]
        )

    def _is_retryable_infra_timeout_phase(self, trial: TrialResult) -> bool:
        if trial.status != TrialStatus.TIMEOUT:
            return False
        timeout_phase = str(trial.metadata.get("timeout_phase") or "")
        return timeout_phase in {
            "environment_start",
            "environment_build",
            "verifier_runtime_prepare",
        }

    def _run_task_once(
        self,
        *,
        task_id: str,
        agent_config: dict[str, Any],
        timeout_audit: int,
        job_name: str | None,
        jobs_dir: str | Path | None,
    ) -> TrialResult:
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
            return self.parse_job_dir(
                command.job_dir,
                task_id=task_id,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                wall_time=wall_time,
                agent_config=agent_config,
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
            return TrialResult(
                trial_id=command.job_name,
                task_id=task_id,
                task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                task_difficulty=TaskDifficulty.MEDIUM,
                status=TrialStatus.TIMEOUT,
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
            return TrialResult(
                trial_id=command.job_name,
                task_id=task_id,
                task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                task_difficulty=TaskDifficulty.MEDIUM,
                status=TrialStatus.ERROR,
                error_log=[str(exc)],
                wall_time_seconds=time.time() - start_time,
                harbor_job_dir=str(command.job_dir),
                metadata={"model_config": self._model_config_metadata(agent_config)},
            )

    def _run_command(
        self,
        argv: list[str],
        *,
        timeout_audit: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run Harbor without a harness wall-clock timeout.

        ``timeout_audit`` is retained as an audit/compatibility reference for
        callers and tests. Official Harbor/TerminalBench environment behavior
        is observed as task evidence; this harness wait does not add a
        time-based loop stop condition.
        """
        _ = timeout_audit
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name != "nt"),
        )
        try:
            stdout, stderr = process.communicate()
            return subprocess.CompletedProcess(
                argv,
                process.returncode,
                stdout,
                stderr,
            )
        except subprocess.TimeoutExpired as exc:
            self._terminate_process_tree(process)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                argv,
                timeout_audit,
                output=stdout or exc.output,
                stderr=stderr or exc.stderr,
            ) from exc

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
                return
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        else:
            process.kill()

    def _partial_timeout_trial(
        self,
        *,
        job_dir: Path,
        task_id: str,
        timeout_audit: int,
        stdout: str,
        stderr: str,
        agent_config: dict[str, Any] | None,
    ) -> TrialResult | None:
        """Recover Harbor artifacts after an external/caller interruption.

        ``timeout_audit`` is a reference retained for legacy callers. The
        normal Harbor runner does not enforce it as a wall-clock loop limit.
        """
        job_result = self.parse_job_dir(
            job_dir,
            task_id=task_id,
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            wall_time=float(timeout_audit),
            agent_config=agent_config,
        )
        if not self._has_partial_timeout_evidence(job_result):
            return None

        if job_result.status != TrialStatus.PASSED:
            job_result.status = TrialStatus.TIMEOUT
        errors = [
            "Harbor command entered the external-interruption compatibility "
            f"path after audit reference {timeout_audit}s"
        ]
        errors.extend(error for error in job_result.error_log if error not in errors)
        job_result.error_log = errors
        job_result.wall_time_seconds = float(timeout_audit)
        job_result.harbor_stdout = stdout or job_result.harbor_stdout
        job_result.harbor_stderr = stderr or job_result.harbor_stderr
        job_result.metadata.update(
            {
                "outer_harbor_timeout": True,
                "outer_harbor_interrupted": True,
                "outer_harbor_timeout_seconds_audit_only": timeout_audit,
                "outer_harbor_timeout_stop_condition": False,
                "outer_harbor_timeout_loop_stop_condition": False,
                "timeout_expired_exception_stop_condition": False,
                "time_round_token_limit_driven": False,
                "partial_harbor_artifacts": True,
            }
        )
        job_result.metadata.setdefault(
            "timeout_phase",
            self._timeout_phase(
                status=job_result.status,
                errors=job_result.error_log,
                stdout=job_result.harbor_stdout,
                stderr=job_result.harbor_stderr,
                verifier_output=job_result.verifier_output,
                verifier_logs=str(job_result.metadata.get("verifier_logs") or ""),
                exception={
                    "exception_type": job_result.metadata.get("agent_exception_type"),
                    "exception_message": job_result.metadata.get("agent_exception_message"),
                },
                timed_out_process=True,
            ),
        )
        job_result.metadata.setdefault("timeout_source", "outer_harbor_command_interrupted")
        return job_result

    def _has_partial_timeout_evidence(self, trial: TrialResult) -> bool:
        return bool(
            trial.harbor_trial_dir
            or trial.trajectory
            or trial.verifier_output
            or trial.metadata.get("agent_exception_type")
            or trial.metadata.get("job_result_status_counts")
        )

    def parse_job_dir(
        self,
        job_dir: str | Path,
        *,
        task_id: str,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        wall_time: float = 0.0,
        agent_config: dict[str, Any] | None = None,
    ) -> TrialResult:
        """Parse Harbor ``jobs/<job>/result.json`` and trial artifacts."""
        job_path = Path(job_dir)
        job_result_path = job_path / "result.json"
        if not job_result_path.exists():
            return self._parse_result(
                trial_id=job_path.name,
                task_id=task_id,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                wall_time=wall_time,
                agent_config=agent_config,
            )

        try:
            job_result = json.loads(job_result_path.read_text())
        except json.JSONDecodeError as exc:
            return TrialResult(
                trial_id=job_path.name,
                task_id=task_id,
                task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                task_difficulty=TaskDifficulty.MEDIUM,
                status=TrialStatus.ERROR,
                error_log=[f"Could not parse Harbor result.json: {exc}"],
                wall_time_seconds=wall_time,
                harbor_job_dir=str(job_path),
                harbor_stdout=stdout,
                harbor_stderr=stderr,
                metadata={"model_config": self._model_config_metadata(agent_config)},
            )

        trial_results = job_result.get("trial_results") or []
        if not trial_results:
            trial_results = self._load_trial_results_from_subdirs(job_path)
        status_counts = self._job_status_counts(job_result, trial_results)
        selected = self._select_trial_result(trial_results, task_id)
        if selected is None:
            fallback_trial_dir = self._fallback_trial_dir(job_path, task_id)
            fallback_errors = ["Harbor result.json did not contain a matching trial result"]
            if status_counts:
                fallback_errors.append(f"Harbor job status counts: {status_counts}")
            fallback_exception = self._read_exception_tail(fallback_trial_dir)
            if fallback_exception:
                fallback_errors.append(fallback_exception)
            fallback_trajectory = (
                self._load_trajectory(fallback_trial_dir)
                if fallback_trial_dir is not None
                else []
            )
            fallback_status = self._status_from_incomplete_job(
                returncode=returncode,
                status_counts=status_counts,
                exception_text=fallback_exception,
            )
            fallback_timeout_phase = self._timeout_phase_from_incomplete(
                status_counts=status_counts,
                exception_text=fallback_exception,
            )
            fallback_metadata = {
                "model_config": self._model_config_metadata(agent_config),
                "job_result_path": str(job_result_path),
                "job_result_status_counts": status_counts,
                "timeout_phase": fallback_timeout_phase,
                "partial_harbor_artifacts": fallback_trial_dir is not None,
            }
            if fallback_trial_dir is not None:
                fallback_metadata.update(
                    self._verifier_runtime_prepare_evidence_metadata(
                        job_path=job_path,
                        trial_dir=fallback_trial_dir,
                        timeout_phase=fallback_timeout_phase,
                    )
                )
            return TrialResult(
                trial_id=fallback_trial_dir.name if fallback_trial_dir else job_path.name,
                task_id=task_id,
                task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                task_difficulty=TaskDifficulty.MEDIUM,
                status=fallback_status,
                error_log=fallback_errors,
                trajectory=fallback_trajectory,
                wall_time_seconds=wall_time,
                harbor_job_dir=str(job_path),
                harbor_trial_dir=str(fallback_trial_dir) if fallback_trial_dir else "",
                harbor_stdout=stdout,
                harbor_stderr=stderr,
                metadata=self._with_verifier_runtime_prepare_timeout_metadata(
                    fallback_metadata
                ),
            )

        score, verifier_output, verifier_logs = self._score_from_harbor_trial(selected, job_path)
        exception = selected.get("exception_info")
        verified = selected.get("verifier_result") is not None
        status = self._status_from_harbor(score, verified, exception, returncode)
        trial_name = selected.get("trial_name") or job_path.name
        task_name = selected.get("task_name") or task_id
        trial_dir = job_path / trial_name
        trajectory = self._load_trajectory(trial_dir)
        artifacts = self._list_artifacts(trial_dir)
        token_usage = self._token_usage(selected)
        trial_metrics = self._trial_metrics(selected, token_usage)
        errors: list[str] = []
        if stderr:
            errors.append(stderr)
        if exception:
            errors.append(str(exception.get("exception_message") or exception))
        if verifier_output and score < 1.0:
            errors.append(verifier_output[:4000])
        if verifier_logs and score < 1.0:
            errors.append(verifier_logs[:4000])
        verifier_infra_error = self._is_infra_text(verifier_logs)
        verified_pass_with_exception = bool(exception and verified and score >= 1.0)
        timeout_phase = self._timeout_phase(
            status=status,
            errors=errors,
            stdout=stdout,
            stderr=stderr,
            verifier_output=verifier_output,
            verifier_logs=verifier_logs,
            exception=exception,
            timed_out_process=False,
        )
        done_after_worker_completion = bool(
            exception
            and timeout_phase
            not in {
                "verifier_runtime_prepare",
                "verifier",
                "environment_start",
                "environment_build",
            }
            and self._has_successful_done_event(trajectory)
        )
        if exception:
            if verified and score >= 1.0:
                errors.append(
                    "Harbor recorded an agent exception after verifier reward "
                    "1.0; preserving it as completion hygiene evidence."
                )
            elif done_after_worker_completion:
                errors.append(
                    "Harbor recorded an agent exception after the Worker called "
                    "done; preserving it as post-completion exception evidence."
                )

        metadata = {
            "harbor_returncode": returncode,
            "job_result_path": str(job_result_path),
            "agent_info": selected.get("agent_info"),
            "model_config": self._model_config_metadata(agent_config, selected),
            "task_metadata": self._task_metadata(selected),
            "raw_rewards": (selected.get("verifier_result") or {}).get("rewards"),
            "verifier_logs": verifier_logs,
            "verifier_infra_error": verifier_infra_error,
            "agent_exception_type": (
                str(exception.get("exception_type") or "") if exception else ""
            ),
            "agent_exception_message": (
                str(exception.get("exception_message") or "") if exception else ""
            ),
            "verified_pass_with_agent_exception": verified_pass_with_exception,
            "completion_hygiene_warning": verified_pass_with_exception,
            "post_completion_agent_exception": done_after_worker_completion,
            "harbor_trial_result_count": len(trial_results),
            "job_result_status_counts": status_counts,
            "timeout_phase": timeout_phase,
            "attempts_observed_for_task": len(
                [
                    result
                    for result in trial_results
                    if self._trial_result_matches_task(result, task_id)
                ]
            )
            or len(trial_results),
            "trial_metrics": trial_metrics,
        }
        metadata.update(
            self._environment_start_evidence_metadata(
                job_path=job_path,
                trial_dir=trial_dir,
                timeout_phase=timeout_phase,
            )
        )
        metadata.update(
            self._verifier_runtime_prepare_evidence_metadata(
                job_path=job_path,
                trial_dir=trial_dir,
                timeout_phase=timeout_phase,
            )
        )
        metadata = self._with_verifier_runtime_prepare_timeout_metadata(metadata)

        return TrialResult(
            trial_id=trial_name,
            task_id=task_name,
            task_domain=self._coerce_domain(selected),
            task_difficulty=self._coerce_difficulty(selected),
            status=status,
            score=score,
            trajectory=trajectory,
            error_log=errors,
            wall_time_seconds=wall_time,
            model_used=self._model_used(selected),
            token_usage=token_usage,
            verified=verified,
            verifier_output=verifier_output,
            harbor_job_dir=str(job_path),
            harbor_trial_dir=str(trial_dir) if trial_dir.exists() else "",
            harbor_stdout=stdout,
            harbor_stderr=stderr,
            artifacts=artifacts,
            metadata=metadata,
        )

    def _parse_result(
        self,
        trial_id: str,
        task_id: str,
        returncode: int,
        stdout: str,
        stderr: str,
        wall_time: float,
        agent_config: dict[str, Any] | None = None,
    ) -> TrialResult:
        """Legacy parser for older fixtures and incomplete Harbor runs."""
        score = 0.0
        status = TrialStatus.FAILED
        error_log = [stderr] if stderr else []
        verifier_output = ""

        trial_path = self.output_dir / trial_id
        reward_paths = [
            trial_path / "verifier" / "reward.txt",
            trial_path / "logs" / "verifier" / "reward.txt",
        ]
        for reward_path in reward_paths:
            if reward_path.exists():
                verifier_output = reward_path.read_text(errors="replace").strip()
                try:
                    score = float(verifier_output)
                    status = TrialStatus.PASSED if score >= 1.0 else TrialStatus.FAILED
                except ValueError:
                    status = TrialStatus.FAILED
                break

        return TrialResult(
            trial_id=trial_id,
            task_id=task_id,
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.MEDIUM,
            status=status,
            score=score,
            error_log=error_log,
            wall_time_seconds=wall_time,
            verified=bool(verifier_output),
            verifier_output=verifier_output,
            harbor_stdout=stdout,
            harbor_stderr=stderr,
            metadata={"model_config": self._model_config_metadata(agent_config)},
        )

    def run_batch(
        self,
        task_ids: list[str],
        agent_config: dict[str, Any],
    ) -> list[TrialResult]:
        """Run multiple tasks serially by default."""
        return [self.run_task(task_id, agent_config) for task_id in task_ids]

    def _materialize_trial(self, trial: TrialResult) -> None:
        trial_dir = self.output_dir / trial.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(trial.model_dump_json(indent=2))
        if trial.trajectory:
            (trial_dir / "trajectory.jsonl").write_text(
                "\n".join(json.dumps(event) for event in trial.trajectory)
            )
        if trial.harbor_stdout:
            (trial_dir / "harbor_stdout.txt").write_text(trial.harbor_stdout)
        if trial.harbor_stderr:
            (trial_dir / "harbor_stderr.txt").write_text(trial.harbor_stderr)
        if trial.verifier_output:
            (trial_dir / "verifier_output.txt").write_text(trial.verifier_output)
        if trial.harbor_trial_dir:
            src = Path(trial.harbor_trial_dir) / "artifacts"
            dst = trial_dir / "artifacts"
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)

    def _agent_kwargs(self, agent_config: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "provider",
            "base_url",
            "api_key_env",
            "reasoning_effort",
            "reasoning_max_tokens",
            "max_output_tokens",
            "max_turns_audit",
            "timeout_seconds",
            "tool_timeout_seconds",
            "max_retries",
            "custom_llm_provider",
            "harness_config",
            "goal_path",
            "memory_path",
        ]
        return {key: agent_config.get(key) for key in keys if key in agent_config}

    def _agent_env(self, agent_config: dict[str, Any]) -> dict[str, Any]:
        env: dict[str, Any] = {}
        api_key_env = agent_config.get("api_key_env")
        if api_key_env:
            env["HL_WORKER_API_KEY_ENV"] = api_key_env
        verifier_env = self._env_dict(agent_config.get("verifier_env"))
        for key in WORKER_TLS_ENV_KEYS:
            if key in verifier_env:
                env[key] = verifier_env[key]
        return env

    def _mounts_json_argument(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, separators=(",", ":"))

    def _env_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [f"{key}={item}" for key, item in value.items()]
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    def _env_dict(self, value: Any) -> dict[str, str]:
        env: dict[str, str] = {}
        for item in self._env_list(value):
            text = str(item)
            if "=" not in text:
                continue
            key, raw = text.split("=", 1)
            key = key.strip()
            if key:
                env[key] = raw
        return env

    def _default_job_name(self, task_id: str) -> str:
        safe_task = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task_id)
        return f"hl_{int(time.time())}_{safe_task[:48]}"

    def _attempt_job_name(self, base_job_name: str | None, attempt_index: int) -> str | None:
        if attempt_index == 0:
            return base_job_name
        base = base_job_name or f"hl_{int(time.time())}"
        return f"{base}_infra_retry{attempt_index}"

    def _infra_retries(self, agent_config: dict[str, Any]) -> int:
        raw = agent_config.get("infra_retries", self.default_infra_retries)
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            return self.default_infra_retries
        return max(0, parsed)

    def _environment_kwargs(self, agent_config: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        raw = agent_config.get("environment_kwargs")
        if isinstance(raw, dict):
            kwargs.update(raw)
        for config_key, env_key in [
            ("apt_mirror_enabled", "apt_mirror_enabled"),
            ("debian_mirror", "debian_mirror"),
            ("debian_security_mirror", "debian_security_mirror"),
            ("ubuntu_mirror", "ubuntu_mirror"),
            ("docker_hub_mirror", "docker_hub_mirror"),
            ("prebuilt_docker_hub_mirror", "prebuilt_docker_hub_mirror"),
            ("docker_image_overrides", "docker_image_overrides"),
            ("download_url_rewrites", "download_url_rewrites"),
            ("pypi_index_url", "pypi_index_url"),
            ("pypi_trusted_host", "pypi_trusted_host"),
            ("apt_retries", "apt_retries"),
            ("apt_timeout_seconds", "apt_timeout_seconds"),
            ("pip_retries", "pip_retries"),
            ("pip_timeout_seconds", "pip_timeout_seconds"),
            ("prebuilt_docker_pull_timeout_seconds", "prebuilt_docker_pull_timeout_seconds"),
            ("bootstrap_ca_certificates", "bootstrap_ca_certificates"),
            ("download_retry_wrapper", "download_retry_wrapper"),
            ("inject_host_ca_into_build", "inject_host_ca_into_build"),
            ("host_ca_cert_bundle", "host_ca_cert_bundle"),
            ("docker_resource_enabled", "docker_resource_enabled"),
            ("docker_memory", "docker_memory"),
            ("docker_memory_swap", "docker_memory_swap"),
            ("docker_cpus", "docker_cpus"),
            ("docker_pids_limit", "docker_pids_limit"),
            ("docker_labels", "docker_labels"),
            ("docker_log_max_size", "docker_log_max_size"),
            ("docker_log_max_file", "docker_log_max_file"),
        ]:
            if config_key in agent_config and agent_config[config_key] is not None:
                kwargs[env_key] = agent_config[config_key]
        return kwargs

    def _docker_resource_config(self, agent_config: dict[str, Any]) -> dict[str, Any]:
        enabled = self._truthy(agent_config.get("docker_resource_enabled", True))
        memory = str(agent_config.get("docker_memory") or DEFAULT_DOCKER_MEMORY)
        memory_swap = str(
            agent_config.get("docker_memory_swap") or memory or DEFAULT_DOCKER_MEMORY_SWAP
        )
        cpus = self._positive_int(agent_config.get("docker_cpus") or DEFAULT_DOCKER_CPUS)
        pids_limit = self._positive_int(
            agent_config.get("docker_pids_limit") or DEFAULT_DOCKER_PIDS_LIMIT
        )
        memory_mb = self._memory_to_mb(memory)
        labels = _parse_docker_labels(agent_config.get("docker_labels"))
        return {
            "enabled": enabled,
            "memory": memory,
            "memory_swap": memory_swap,
            "memory_mb": memory_mb,
            "cpus": cpus,
            "pids_limit": pids_limit,
            "labels": labels,
            "log_max_size": str(
                agent_config.get("docker_log_max_size") or DEFAULT_DOCKER_LOG_MAX_SIZE
            ),
            "log_max_file": str(
                agent_config.get("docker_log_max_file") or DEFAULT_DOCKER_LOG_MAX_FILE
            ),
            "delete_volumes": False,
        }

    def _positive_int(self, value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _memory_to_mb(self, value: Any) -> int | None:
        text = str(value or "").strip().lower()
        if not text:
            return None
        match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgt]?i?b?)?", text)
        if not match:
            return None
        amount = float(match.group(1))
        unit = (match.group(2) or "m").rstrip("b")
        multipliers = {
            "": 1 / (1024 * 1024),
            "k": 1 / 1024,
            "ki": 1 / 1024,
            "m": 1,
            "mi": 1,
            "g": 1024,
            "gi": 1024,
            "t": 1024 * 1024,
            "ti": 1024 * 1024,
        }
        multiplier = multipliers.get(unit)
        if multiplier is None:
            return None
        return max(1, int(amount * multiplier))

    def _truthy(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off", ""}

    def _kwarg_value(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return str(value)

    def _n_attempts(self, agent_config: dict[str, Any]) -> int | None:
        raw = agent_config.get("n_attempts")
        if raw is None:
            return None
        try:
            attempts = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_attempts must be a positive integer") from exc
        if attempts <= 0:
            return None
        return attempts

    def is_infra_error(self, trial: TrialResult) -> bool:
        metadata = trial.metadata or {}
        if metadata.get("score_exclusion_reason") == "infrastructure_error":
            return True
        if metadata.get("infra_error_detected") and not trial.verified:
            return True
        if metadata.get("verifier_runtime_prepare_timeout"):
            return True
        if metadata.get("timeout_phase") == "verifier_runtime_prepare":
            return True
        text = "\n".join(
            [
                self._trial_infra_text(trial),
            ]
        )
        return self._is_infra_text(text)

    def _is_infra_text(self, text: str) -> bool:
        lowered = text.lower()
        if any(pattern.lower() in lowered for pattern in INFRA_ERROR_PATTERNS):
            return True
        return (
            any(pattern in lowered for pattern in INFRA_NETWORK_ENDPOINT_PATTERNS)
            and any(pattern in lowered for pattern in INFRA_NETWORK_FAILURE_PATTERNS)
        )

    def _environment_start_evidence_metadata(
        self,
        *,
        job_path: Path,
        trial_dir: Path,
        timeout_phase: str,
    ) -> dict[str, Any]:
        if timeout_phase not in {"environment_start", "environment_build"}:
            return {}
        evidence_text = self._environment_evidence_text(job_path, trial_dir)
        docker_image_validation = self._docker_image_validation_events(evidence_text)
        heavy_steps = self._heavy_dockerfile_install_steps(trial_dir)
        patch_marker = self._environment_patch_marker(trial_dir)
        config_evidence = self._environment_config_evidence(job_path, trial_dir)
        cache_warmup = self._prebuilt_image_cache_warmup_evidence(
            docker_image_validation,
            config_evidence,
            patch_marker,
        )
        metadata: dict[str, Any] = {
            "environment_start_evidence": {
                "docker_image_validation": docker_image_validation,
                "prebuilt_image_cache_warmup": cache_warmup,
                "heavy_dockerfile_install_steps": heavy_steps,
                "patched_environment_marker": patch_marker,
                "environment_config": config_evidence,
            },
            "environment_start_attribution_hint": self._environment_start_attribution_hint(
                docker_image_validation,
                heavy_steps,
                patch_marker,
                config_evidence,
                cache_warmup,
            ),
        }
        if docker_image_validation:
            metadata["docker_image_validation_failed"] = True
            metadata["docker_image_validation_events"] = docker_image_validation
        if cache_warmup:
            metadata["prebuilt_image_cache_miss_detected"] = True
            metadata["prebuilt_image_cache_warmup"] = cache_warmup
            metadata["prebuilt_image_cache_warmup_targets"] = cache_warmup[
                "targets"
            ]
            metadata["prebuilt_image_cache_warmup_commands"] = cache_warmup[
                "commands"
            ]
            metadata["network_preflight_recommended"] = True
        if heavy_steps:
            metadata["heavy_dockerfile_install_detected"] = True
            metadata["heavy_dockerfile_install_steps"] = heavy_steps
        if patch_marker:
            metadata["network_hardened_environment_marker"] = patch_marker
        if config_evidence:
            metadata["environment_config_evidence"] = config_evidence
            if self._environment_config_is_network_hardened(config_evidence):
                metadata["network_hardened_environment_config"] = True
        return metadata

    def _environment_config_evidence(
        self,
        job_path: Path,
        trial_dir: Path,
    ) -> dict[str, Any]:
        configs: list[tuple[str, Path, dict[str, Any]]] = []
        for label, path in [
            ("job_config_path", job_path / "config.json"),
            ("trial_config_path", trial_dir / "config.json"),
        ]:
            data = self._read_json_object(path)
            if data:
                configs.append((label, path, data))
        if not configs:
            return {}

        evidence: dict[str, Any] = {}
        environment_kwargs: dict[str, Any] = {}
        for label, path, data in configs:
            evidence[label] = str(path)
            environment = data.get("environment")
            if isinstance(environment, dict):
                if "environment_import_path" not in evidence and environment.get("import_path"):
                    evidence["environment_import_path"] = str(environment.get("import_path"))
                if "force_build" not in evidence and "force_build" in environment:
                    evidence["force_build"] = bool(environment.get("force_build"))
                if "mounts_json_count" not in evidence:
                    mounts_json = environment.get("mounts_json")
                    mounts_count = self._environment_mounts_json_count(mounts_json)
                    if mounts_count is not None:
                        evidence["mounts_json_count"] = mounts_count
                raw_kwargs = environment.get("kwargs")
                if isinstance(raw_kwargs, dict):
                    for key in ENVIRONMENT_CONFIG_KWARG_EVIDENCE_KEYS:
                        if key in raw_kwargs and raw_kwargs[key] is not None:
                            environment_kwargs.setdefault(key, raw_kwargs[key])
            dataset = data.get("dataset")
            if isinstance(dataset, dict) and "task_path" not in evidence:
                task_path = dataset.get("path") or dataset.get("task_path")
                if task_path:
                    evidence["task_path"] = str(task_path)

        if environment_kwargs:
            evidence["environment_kwargs"] = environment_kwargs
        return evidence

    def _read_json_object(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return {"parse_error": True, "path": str(path)}
        return data if isinstance(data, dict) else {}

    def _environment_mounts_json_count(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, list):
            return len(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, list):
                return len(parsed)
        return None

    def _environment_config_is_network_hardened(self, config_evidence: dict[str, Any]) -> bool:
        import_path = str(config_evidence.get("environment_import_path") or "")
        if import_path == DEFAULT_NETWORK_ENVIRONMENT_IMPORT_PATH:
            return True
        kwargs = config_evidence.get("environment_kwargs")
        if not isinstance(kwargs, dict):
            return False
        return any(
            kwargs.get(key)
            for key in [
                "docker_hub_mirror",
                "prebuilt_docker_hub_mirror",
                "download_url_rewrites",
                "pypi_index_url",
                "debian_mirror",
                "ubuntu_mirror",
            ]
        )

    def _environment_evidence_text(self, job_path: Path, trial_dir: Path) -> str:
        parts: list[str] = []
        for path in [job_path / "job.log", trial_dir / "trial.log", trial_dir / "exception.txt"]:
            if path.exists():
                parts.append(path.read_text(errors="replace")[-ENVIRONMENT_EVIDENCE_LOG_MAX_CHARS:])
        return "\n".join(parts)

    def _docker_image_validation_events(self, text: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        pattern = re.compile(
            r"Skipping image OS validation for (?P<image>\S+): "
            r"docker inspect returned (?P<returncode>\d+)",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            image = match.group("image")
            returncode = match.group("returncode")
            key = ("docker_inspect", image, returncode)
            if key in seen:
                continue
            seen.add(key)
            events.append(
                {
                    "operation": "docker_inspect",
                    "image": image,
                    "returncode": int(returncode),
                }
            )
        warmup_timeout_pattern = re.compile(
            r"Prebuilt Docker image cache warmup timed out after "
            r"(?P<seconds>\d+(?:\.\d+)?) seconds for image (?P<image>\S+)",
            flags=re.IGNORECASE,
        )
        for match in warmup_timeout_pattern.finditer(text):
            image = match.group("image").rstrip(". ,;:\")'`]")
            seconds = match.group("seconds")
            key = ("docker_pull", image, seconds)
            if key in seen:
                continue
            seen.add(key)
            timeout_value = float(seconds)
            events.append(
                {
                    "operation": "docker_pull",
                    "image": image,
                    "timeout_seconds": int(timeout_value)
                    if timeout_value.is_integer()
                    else timeout_value,
                }
            )
        return events

    def _prebuilt_image_cache_warmup_evidence(
        self,
        docker_image_validation: list[dict[str, Any]],
        config_evidence: dict[str, Any],
        patch_marker: dict[str, Any],
    ) -> dict[str, Any]:
        targets: list[dict[str, Any]] = []
        commands: list[str] = []
        seen: set[str] = set()
        prebuilt_mirror = self._configured_prebuilt_image_mirror(
            config_evidence,
            patch_marker,
        )
        for event in docker_image_validation:
            image = str(event.get("image") or "").strip()
            if not image or image in seen:
                continue
            seen.add(image)
            command = f"docker pull {shlex.quote(image)}"
            target = {
                "effective_image": image,
                "docker_pull_command": command,
                "operation": str(event.get("operation") or "docker_inspect"),
            }
            if event.get("returncode") is not None:
                target["returncode"] = event.get("returncode")
            if event.get("timeout_seconds") is not None:
                target["timeout_seconds"] = event.get("timeout_seconds")
            if prebuilt_mirror:
                target["configured_prebuilt_docker_hub_mirror"] = prebuilt_mirror
                original = self._original_prebuilt_image_name(image, prebuilt_mirror)
                if original != image:
                    target["original_image"] = original
            targets.append(target)
            commands.append(command)
        if not targets:
            return {}
        return {
            "cache_miss_detected": True,
            "network_preflight_recommended": True,
            "source": "docker_image_validation_events",
            "targets": targets,
            "commands": commands,
            "preflight_command": "python scripts/network_preflight.py --quick",
        }

    def _configured_prebuilt_image_mirror(
        self,
        config_evidence: dict[str, Any],
        patch_marker: dict[str, Any],
    ) -> str:
        kwargs = config_evidence.get("environment_kwargs")
        if isinstance(kwargs, dict):
            mirror = str(kwargs.get("prebuilt_docker_hub_mirror") or "").strip()
            if mirror:
                return mirror.rstrip("/")
        mirror = str(patch_marker.get("prebuilt_docker_hub_mirror") or "").strip()
        return mirror.rstrip("/")

    def _original_prebuilt_image_name(self, image: str, prebuilt_mirror: str) -> str:
        prefix = prebuilt_mirror.rstrip("/") + "/"
        if image.startswith(prefix):
            return image[len(prefix) :]
        return image

    def _heavy_dockerfile_install_steps(self, trial_dir: Path) -> list[dict[str, Any]]:
        env_dir = trial_dir / "hl_patched_environment"
        if not env_dir.exists():
            return []
        steps: list[dict[str, Any]] = []
        for dockerfile in sorted(env_dir.glob("Dockerfile*")):
            text = dockerfile.read_text(errors="replace")[-ENVIRONMENT_EVIDENCE_DOCKERFILE_MAX_CHARS:]
            for instruction in self._dockerfile_run_instructions(text):
                packages = self._heavy_dependency_names(instruction)
                if not packages:
                    continue
                steps.append(
                    {
                        "file": str(dockerfile.relative_to(trial_dir)),
                        "instruction": self._truncate_one_line(instruction, 500),
                        "packages": packages,
                    }
                )
        return steps

    def _dockerfile_run_instructions(self, text: str) -> list[str]:
        instructions: list[str] = []
        current: list[str] = []
        in_run = False
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not in_run:
                if re.match(r"^RUN\s+", stripped, flags=re.IGNORECASE):
                    current = [re.sub(r"^RUN\s+", "", stripped, flags=re.IGNORECASE)]
                    in_run = stripped.endswith("\\")
                    if not in_run:
                        instructions.append(" ".join(current))
                        current = []
                continue
            current.append(stripped.rstrip("\\").strip())
            in_run = stripped.endswith("\\")
            if not in_run:
                instructions.append(" ".join(current))
                current = []
        if current:
            instructions.append(" ".join(current))
        return instructions

    def _heavy_dependency_names(self, instruction: str) -> list[str]:
        lowered = instruction.lower()
        if not any(
            marker in lowered
            for marker in [
                "pip install",
                "python -m pip install",
                "uv pip install",
                "apt-get install",
                "apt install",
            ]
        ):
            return []
        heavy_markers = [
            "torch",
            "tensorflow",
            "transformers",
            "datasets",
            "sentence-transformers",
            "spacy",
            "opencv",
            "cuda",
            "nvidia",
            "pytorch",
            "llama",
            "llm",
            "scipy",
            "scikit-learn",
        ]
        found = [marker for marker in heavy_markers if re.search(rf"\b{re.escape(marker)}\b", lowered)]
        return sorted(dict.fromkeys(found))

    def _environment_patch_marker(self, trial_dir: Path) -> dict[str, Any]:
        marker_path = trial_dir / "hl_patched_environment" / ".hl_apt_mirror.json"
        if not marker_path.exists():
            return {}
        try:
            marker = json.loads(marker_path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return {"path": str(marker_path.relative_to(trial_dir)), "parse_error": True}
        return {
            "path": str(marker_path.relative_to(trial_dir)),
            "patched_files": [str(item) for item in marker.get("patched_files") or []],
            "docker_hub_mirror": marker.get("docker_hub_mirror"),
            "prebuilt_docker_hub_mirror": marker.get("prebuilt_docker_hub_mirror"),
            "pypi_index_url": marker.get("pypi_index_url"),
            "apt_timeout_seconds": marker.get("apt_timeout_seconds"),
            "pip_timeout_seconds": marker.get("pip_timeout_seconds"),
            "prebuilt_docker_pull_timeout_seconds": marker.get(
                "prebuilt_docker_pull_timeout_seconds"
            ),
        }

    def _environment_start_attribution_hint(
        self,
        docker_image_validation: list[dict[str, Any]],
        heavy_steps: list[dict[str, Any]],
        patch_marker: dict[str, Any],
        config_evidence: dict[str, Any],
        cache_warmup: dict[str, Any],
    ) -> str:
        hints: list[str] = []
        if docker_image_validation:
            images = ", ".join(
                str(event.get("image")) for event in docker_image_validation[:3]
            )
            hints.append(f"prebuilt image inspect failed for {images}")
        if cache_warmup:
            images = ", ".join(
                str(target.get("effective_image"))
                for target in cache_warmup.get("targets", [])[:3]
            )
            hints.append(f"pre-pull/cache-warm prebuilt image(s): {images}")
        if heavy_steps:
            packages = sorted(
                {
                    package
                    for step in heavy_steps
                    for package in step.get("packages", [])
                }
            )
            hints.append("heavy Dockerfile dependency install: " + ", ".join(packages[:8]))
        if patch_marker:
            hints.append("network-hardened copied environment was present")
        if config_evidence:
            import_path = config_evidence.get("environment_import_path")
            if import_path == DEFAULT_NETWORK_ENVIRONMENT_IMPORT_PATH:
                hints.append(f"network-hardened environment configured: {import_path}")
            kwargs = config_evidence.get("environment_kwargs")
            if isinstance(kwargs, dict):
                prebuilt_mirror = kwargs.get("prebuilt_docker_hub_mirror")
                docker_mirror = kwargs.get("docker_hub_mirror")
                if prebuilt_mirror:
                    hints.append(f"prebuilt image mirror configured: {prebuilt_mirror}")
                if docker_mirror:
                    hints.append(f"Docker Hub mirror configured: {docker_mirror}")
        if not hints:
            return "environment startup/build timed out before Worker trajectory evidence"
        return "; ".join(hints)

    def _verifier_runtime_prepare_evidence_metadata(
        self,
        *,
        job_path: Path,
        trial_dir: Path,
        timeout_phase: str,
    ) -> dict[str, Any]:
        if timeout_phase != "verifier_runtime_prepare":
            return {}
        evidence_text = self._verifier_runtime_prepare_evidence_text(job_path, trial_dir)
        config_evidence = self._environment_config_evidence(job_path, trial_dir)
        patch_marker = self._environment_patch_marker(trial_dir)
        docker_image_validation = self._docker_image_validation_events(evidence_text)
        cache_warmup = self._prebuilt_image_cache_warmup_evidence(
            docker_image_validation,
            config_evidence,
            patch_marker,
        )
        timeout_seconds = self._verifier_runtime_prepare_timeout_seconds(evidence_text)
        evidence: dict[str, Any] = {
            "timeout_seconds": timeout_seconds,
            "prepare_marker_path": "/tmp/hl-verifier-network-prepared",
            "prepare_function": "bench.network_environment:AptMirrorDockerEnvironment._prepare_verifier_runtime",
            "evidence_text_tail": self._truncate_one_line(evidence_text, 2000),
            "environment_config": config_evidence,
            "patched_environment_marker": patch_marker,
            "docker_image_validation": docker_image_validation,
            "prebuilt_image_cache_warmup": cache_warmup,
        }
        metadata: dict[str, Any] = {
            "verifier_runtime_prepare_evidence": evidence,
            "verifier_runtime_prepare_attribution_hint": (
                self._verifier_runtime_prepare_attribution_hint(
                    timeout_seconds=timeout_seconds,
                    config_evidence=config_evidence,
                    patch_marker=patch_marker,
                    docker_image_validation=docker_image_validation,
                    cache_warmup=cache_warmup,
                )
            ),
        }
        if timeout_seconds is not None:
            metadata["verifier_runtime_prepare_timeout_seconds_observed"] = timeout_seconds
        if config_evidence:
            metadata["verifier_runtime_prepare_environment_config"] = config_evidence
            if self._environment_config_is_network_hardened(config_evidence):
                metadata["verifier_runtime_prepare_network_hardened_config"] = True
        if patch_marker:
            metadata["verifier_runtime_prepare_network_hardened_marker"] = patch_marker
        if docker_image_validation:
            metadata["verifier_runtime_prepare_docker_image_validation_events"] = (
                docker_image_validation
            )
        if cache_warmup:
            metadata["prebuilt_image_cache_miss_detected"] = True
            metadata["prebuilt_image_cache_warmup"] = cache_warmup
            metadata["prebuilt_image_cache_warmup_targets"] = cache_warmup[
                "targets"
            ]
            metadata["prebuilt_image_cache_warmup_commands"] = cache_warmup[
                "commands"
            ]
            metadata["verifier_runtime_prepare_prebuilt_image_cache_miss_detected"] = True
            metadata["verifier_runtime_prepare_prebuilt_image_cache_warmup"] = (
                cache_warmup
            )
            metadata["verifier_runtime_prepare_prebuilt_image_cache_warmup_commands"] = (
                cache_warmup["commands"]
            )
            metadata["network_preflight_recommended"] = True
        return metadata

    def _verifier_runtime_prepare_evidence_text(
        self,
        job_path: Path,
        trial_dir: Path,
    ) -> str:
        parts: list[str] = []
        for path in [
            job_path / "job.log",
            trial_dir / "trial.log",
            trial_dir / "exception.txt",
        ]:
            if path.exists():
                parts.append(path.read_text(errors="replace")[-ENVIRONMENT_EVIDENCE_LOG_MAX_CHARS:])
        verifier_dir = trial_dir / "verifier"
        if verifier_dir.exists():
            for path in sorted(verifier_dir.iterdir()):
                if path.is_file() and path.name != "reward.txt":
                    parts.append(path.read_text(errors="replace")[-ENVIRONMENT_EVIDENCE_LOG_MAX_CHARS:])
        return "\n".join(part for part in parts if part)

    def _verifier_runtime_prepare_timeout_seconds(self, text: str) -> float | None:
        match = re.search(
            r"(?:timed out after|Command timed out after)\s+(?P<seconds>\d+(?:\.\d+)?)\s+seconds",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        value = float(match.group("seconds"))
        return int(value) if value.is_integer() else value

    def _verifier_runtime_prepare_attribution_hint(
        self,
        *,
        timeout_seconds: float | None,
        config_evidence: dict[str, Any],
        patch_marker: dict[str, Any],
        docker_image_validation: list[dict[str, Any]],
        cache_warmup: dict[str, Any],
    ) -> str:
        hints: list[str] = []
        if timeout_seconds is not None:
            hints.append(f"verifier runtime network preparation timed out after {timeout_seconds}s")
        if config_evidence:
            import_path = config_evidence.get("environment_import_path")
            if import_path == DEFAULT_NETWORK_ENVIRONMENT_IMPORT_PATH:
                hints.append(f"runtime network-hardened environment configured: {import_path}")
            kwargs = config_evidence.get("environment_kwargs")
            if isinstance(kwargs, dict):
                pypi_index = kwargs.get("pypi_index_url")
                apt_timeout = kwargs.get("apt_timeout_seconds")
                pip_timeout = kwargs.get("pip_timeout_seconds")
                if pypi_index:
                    hints.append(f"runtime PyPI mirror configured: {pypi_index}")
                if apt_timeout is not None or pip_timeout is not None:
                    hints.append(
                        "runtime package manager timeouts configured: "
                        f"apt={apt_timeout}, pip={pip_timeout}"
                    )
        if patch_marker:
            hints.append("network-hardened copied environment was present")
        if docker_image_validation:
            images = ", ".join(
                str(event.get("image")) for event in docker_image_validation[:3]
            )
            hints.append(f"prebuilt image inspect failed for {images}")
        if cache_warmup:
            images = ", ".join(
                str(target.get("effective_image"))
                for target in cache_warmup.get("targets", [])[:3]
            )
            hints.append(f"pre-pull/cache-warm prebuilt image(s): {images}")
        if not hints:
            return "verifier runtime network preparation timed out before verifier result evidence"
        return "; ".join(hints)

    def _truncate_one_line(self, value: str, max_chars: int) -> str:
        text = " ".join(value.split())
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _select_trial_result(
        self, trial_results: list[dict[str, Any]], task_id: str
    ) -> dict[str, Any] | None:
        if not trial_results:
            return None
        for result in trial_results:
            if self._trial_result_matches_task(result, task_id):
                return result
        return trial_results[0]

    def _trial_result_matches_task(self, result: dict[str, Any], task_id: str) -> bool:
        task_id_value = result.get("task_id") or {}
        task_path = task_id_value.get("path") if isinstance(task_id_value, dict) else task_id_value
        task_name = task_id_value.get("name") if isinstance(task_id_value, dict) else ""
        names = {
            str(result.get("task_name") or ""),
            str(task_path or ""),
            str(task_name or ""),
        }
        return any(task_id in name for name in names)

    def _load_trial_results_from_subdirs(self, job_path: Path) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for result_path in sorted(job_path.glob("*/result.json")):
            try:
                data = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                results.append(data)
        return results

    def _score_from_harbor_trial(
        self, trial: dict[str, Any], job_path: Path
    ) -> tuple[float, str, str]:
        trial_name = trial.get("trial_name")
        verifier_logs = self._read_verifier_logs(job_path, trial_name)
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        if rewards:
            if "reward" in rewards:
                return float(rewards["reward"]), json.dumps(rewards, indent=2), verifier_logs
            numeric = [float(value) for value in rewards.values() if isinstance(value, int | float)]
            if numeric:
                return sum(numeric) / len(numeric), json.dumps(rewards, indent=2), verifier_logs

        if trial_name:
            reward_path = job_path / trial_name / "verifier" / "reward.txt"
            if reward_path.exists():
                text = reward_path.read_text(errors="replace").strip()
                try:
                    return float(text), text, verifier_logs
                except ValueError:
                    return 0.0, text, verifier_logs
        return 0.0, "", verifier_logs

    def _read_verifier_logs(self, job_path: Path, trial_name: str | None) -> str:
        if not trial_name:
            return ""
        verifier_dir = job_path / trial_name / "verifier"
        if not verifier_dir.exists():
            return ""
        parts: list[str] = []
        for path in sorted(verifier_dir.iterdir()):
            if not path.is_file() or path.name == "reward.txt":
                continue
            text = path.read_text(errors="replace").strip()
            if not text:
                continue
            parts.append(f"## {path.name}\n{text}")
        return "\n\n".join(parts)[:VERIFIER_LOG_MAX_CHARS]

    def _status_from_harbor(
        self,
        score: float,
        verified: bool,
        exception: dict[str, Any] | None,
        returncode: int,
    ) -> TrialStatus:
        if verified and score >= 1.0:
            return TrialStatus.PASSED
        if exception:
            exc_type = str(exception.get("exception_type") or "")
            if "Timeout" in exc_type:
                return TrialStatus.TIMEOUT
            if "Cancelled" in exc_type:
                return TrialStatus.CANCELLED
            return TrialStatus.ERROR
        if verified:
            return TrialStatus.PASSED if score >= 1.0 else TrialStatus.FAILED
        if returncode != 0:
            return TrialStatus.ERROR
        return TrialStatus.FAILED

    def _load_trajectory(self, trial_dir: Path) -> list[dict[str, Any]]:
        candidates = [
            trial_dir / "agent" / "trajectory.jsonl",
            trial_dir / "agent" / "trajectory.json",
            trial_dir / "trajectory.jsonl",
            trial_dir / "trajectory.json",
        ]
        for path in candidates:
            events = TrajectoryReader.load(path)
            if events:
                return events
        return []

    def _has_successful_done_event(self, trajectory: list[dict[str, Any]]) -> bool:
        for event in trajectory:
            if not isinstance(event, dict):
                continue
            if event.get("type") != "tool_call" or event.get("tool") != "done":
                continue
            if event.get("success") is False:
                continue
            return True
        return False

    def _list_artifacts(self, trial_dir: Path) -> list[str]:
        artifacts_dir = trial_dir / "artifacts"
        if not artifacts_dir.exists():
            return []
        return sorted(
            str(path.relative_to(artifacts_dir))
            for path in artifacts_dir.rglob("*")
            if path.is_file()
        )

    def _token_usage(self, trial: dict[str, Any]) -> dict[str, int]:
        agent_result = trial.get("agent_result") or {}
        mapping = {
            "input": agent_result.get("n_input_tokens"),
            "cache": agent_result.get("n_cache_tokens"),
            "output": agent_result.get("n_output_tokens"),
        }
        return {key: int(value) for key, value in mapping.items() if isinstance(value, int)}

    def _trial_metrics(
        self,
        trial: dict[str, Any],
        token_usage: dict[str, int],
    ) -> dict[str, Any]:
        agent_result = trial.get("agent_result") or {}
        metrics: dict[str, Any] = {}
        numeric_fields = {
            "cost_usd": ("cost_usd", "total_cost_usd", "cost"),
            "n_turns": ("n_turns", "turns"),
            "n_api_calls": ("n_api_calls", "api_calls"),
            "api_error_count": ("api_error_count", "n_api_errors"),
            "provider_latency_ms": ("provider_latency_ms", "latency_ms"),
        }
        for target, candidates in numeric_fields.items():
            value = self._first_numeric(agent_result, candidates)
            if value is not None:
                metrics[target] = value
        input_tokens = int(token_usage.get("input", 0) or 0)
        cache_tokens = int(token_usage.get("cache", 0) or 0)
        total_prompt_tokens = input_tokens + cache_tokens
        if total_prompt_tokens > 0:
            metrics["cache_hit_ratio"] = round(cache_tokens / total_prompt_tokens, 4)
        return metrics

    def _first_numeric(
        self,
        data: dict[str, Any],
        candidates: tuple[str, ...],
    ) -> int | float | None:
        for key in candidates:
            value = data.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return round(value, 6)
        return None

    def _model_used(self, trial: dict[str, Any]) -> str:
        model_info = (trial.get("agent_info") or {}).get("model_info") or {}
        provider = model_info.get("provider")
        name = model_info.get("name")
        if provider and name:
            return f"{provider}/{name}"
        return name or ""

    def _model_config_metadata(
        self,
        agent_config: dict[str, Any] | None,
        trial: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        agent_config = agent_config or {}
        model_info = ((trial or {}).get("agent_info") or {}).get("model_info") or {}
        base_url = agent_config.get("base_url")
        metadata = {
            "worker_role": agent_config.get("worker_role"),
            "provider": agent_config.get("provider") or model_info.get("provider"),
            "model": agent_config.get("model") or model_info.get("name"),
            "base_url_host": self._base_url_host(str(base_url)) if base_url else None,
            "api_key_env": agent_config.get("api_key_env"),
            "reasoning_effort": agent_config.get("reasoning_effort"),
            "reasoning_max_tokens": agent_config.get("reasoning_max_tokens"),
            "max_output_tokens": agent_config.get("max_output_tokens"),
            "timeout_seconds": agent_config.get("timeout_seconds"),
            "tool_timeout_seconds": agent_config.get("tool_timeout_seconds"),
            "max_retries": agent_config.get("max_retries"),
        }
        return {key: value for key, value in metadata.items() if value is not None}

    def _job_status_counts(
        self,
        job_result: dict[str, Any],
        trial_results: list[dict[str, Any]],
    ) -> dict[str, int]:
        stats = job_result.get("stats") if isinstance(job_result, dict) else {}
        counts: dict[str, int] = {}
        if isinstance(stats, dict):
            for key, name in [
                ("n_completed_trials", "completed"),
                ("n_errored_trials", "errored"),
                ("n_running_trials", "running"),
                ("n_pending_trials", "pending"),
                ("n_cancelled_trials", "cancelled"),
            ]:
                value = stats.get(key)
                if isinstance(value, int) and value:
                    counts[name] = value
        if trial_results:
            counts["trial_results"] = len(trial_results)
        return counts

    def _fallback_trial_dir(self, job_path: Path, task_id: str) -> Path | None:
        if not job_path.exists():
            return None
        candidates = [
            path
            for path in sorted(job_path.iterdir())
            if path.is_dir() and not path.name.startswith(".")
        ]
        for path in candidates:
            if task_id in path.name:
                return path
        return candidates[0] if len(candidates) == 1 else None

    def _read_exception_tail(self, trial_dir: Path | None) -> str:
        if trial_dir is None:
            return ""
        path = trial_dir / "exception.txt"
        if not path.exists():
            return ""
        return path.read_text(errors="replace")[-4000:].strip()

    def _status_from_incomplete_job(
        self,
        *,
        returncode: int,
        status_counts: dict[str, int],
        exception_text: str,
    ) -> TrialStatus:
        lowered = exception_text.lower()
        if "agenttimeouterror" in lowered or "timed out" in lowered:
            return TrialStatus.TIMEOUT
        if status_counts.get("cancelled"):
            return TrialStatus.CANCELLED
        if status_counts.get("running"):
            return TrialStatus.TIMEOUT if returncode else TrialStatus.RUNNING
        if returncode != 0:
            return TrialStatus.ERROR
        return TrialStatus.FAILED

    def _timeout_phase_from_incomplete(
        self,
        *,
        status_counts: dict[str, int],
        exception_text: str,
    ) -> str:
        lowered = exception_text.lower()
        if self._has_verifier_runtime_prepare_timeout_marker(lowered):
            return "verifier_runtime_prepare"
        if "agenttimeouterror" in lowered or "agent execution timed out" in lowered:
            return "agent_execution"
        if (
            "environmentstarttimeouterror" in lowered
            or "environment start timed out" in lowered
            or "_start_environment" in lowered
            or "_setup_environment" in lowered
        ):
            return "environment_start"
        if "environmentbuildtimeouterror" in lowered or (
            "environment" in lowered and "build" in lowered and "timed out" in lowered
        ):
            return "environment_build"
        if self._has_verifier_timeout_marker(
            lowered,
            timed_out_or_cancelled=bool(
                status_counts.get("cancelled")
                or status_counts.get("running")
                or "timed out" in lowered
                or "cancellederror" in lowered
            ),
        ):
            return "verifier"
        if "cancelled" in lowered or status_counts.get("cancelled"):
            return "harbor_cancelled"
        if status_counts.get("running"):
            return "harbor_process"
        return ""

    def _timeout_phase(
        self,
        *,
        status: TrialStatus,
        errors: list[str],
        stdout: str,
        stderr: str,
        verifier_output: str,
        verifier_logs: str,
        exception: dict[str, Any] | None = None,
        timed_out_process: bool = False,
    ) -> str:
        exception = exception or {}
        exception_type = str(exception.get("exception_type") or "").lower()
        exception_message = str(exception.get("exception_message") or "").lower()
        exception_traceback = str(exception.get("exception_traceback") or "").lower()
        text = "\n".join(
            [
                exception_type,
                exception_message,
                exception_traceback,
                *errors,
                stdout or "",
                stderr or "",
                verifier_output or "",
                verifier_logs or "",
            ]
        ).lower()
        if self._has_verifier_runtime_prepare_timeout_marker(text):
            return "verifier_runtime_prepare"
        if "agenttimeouterror" in text or "agent execution timed out" in text:
            return "agent_execution"
        if (
            "environmentstarttimeouterror" in text
            or "environment start timed out" in text
            or "_start_environment" in text
            or "_setup_environment" in text
        ):
            return "environment_start"
        if "environmentbuildtimeouterror" in text or (
            "environment" in text and "build" in text and "timed out" in text
        ):
            return "environment_build"
        if self._has_verifier_timeout_marker(
            text,
            timed_out_or_cancelled=(
                timed_out_process
                or status in {TrialStatus.TIMEOUT, TrialStatus.CANCELLED}
                or "timed out" in text
                or "cancellederror" in text
            ),
        ):
            return "verifier"
        if timed_out_process or status == TrialStatus.TIMEOUT:
            return "harbor_process"
        return ""

    def _has_verifier_timeout_marker(
        self,
        text: str,
        *,
        timed_out_or_cancelled: bool,
    ) -> bool:
        if "verifiertimeouterror" in text:
            return True
        if not timed_out_or_cancelled:
            return False
        return any(
            marker in text
            for marker in [
                "_run_verification",
                "_verify_with_retry",
                "verifier.py",
                "/verifier",
                " verifier",
            ]
        )

    def _has_verifier_runtime_prepare_timeout_marker(self, text: str) -> bool:
        return (
            (
                "verifier runtime network preparation" in text
                or "verifier runtime prepare" in text
                or "hl-verifier-network-prepared" in text
                or "_prepare_verifier_runtime" in text
            )
            and ("timed out" in text or "timeout" in text or "cancellederror" in text)
        )

    def _with_verifier_runtime_prepare_timeout_metadata(
        self, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        if metadata.get("timeout_phase") == "verifier_runtime_prepare":
            metadata["verifier_runtime_prepare_timeout"] = True
            metadata["verifier_infra_error"] = True
            metadata["infra_error_detected"] = True
            metadata["score_exclusion_reason"] = "infrastructure_error"
        return metadata

    def _base_url_host(self, base_url: str) -> str:
        parsed = urlparse(base_url)
        if parsed.netloc:
            return parsed.netloc
        return base_url.split("/")[0]

    def _coerce_domain(self, trial: dict[str, Any]) -> TaskDomain:
        metadata = self._task_metadata(trial)
        raw = str(metadata.get("category") or "").lower().replace("-", "_")
        try:
            return TaskDomain(raw)
        except ValueError:
            pass
        if "security" in raw:
            return TaskDomain.SECURITY
        return TaskDomain.SOFTWARE_ENGINEERING

    def _coerce_difficulty(self, trial: dict[str, Any]) -> TaskDifficulty:
        metadata = self._task_metadata(trial)
        raw = str(metadata.get("difficulty") or "").lower()
        try:
            return TaskDifficulty(raw)
        except ValueError:
            return TaskDifficulty.MEDIUM

    def _task_metadata(self, trial: dict[str, Any]) -> dict[str, Any]:
        task_config = ((trial.get("config") or {}).get("task") or {})
        metadata = task_config.get("metadata")
        if not isinstance(metadata, dict):
            task_path = task_config.get("path")
            metadata = self._task_toml_metadata(task_path) if task_path else {}

        tags = metadata.get("tags") if isinstance(metadata, dict) else []
        if not isinstance(tags, list):
            tags = []
        task_type = (
            metadata.get("task_type")
            or metadata.get("type")
            or (str(tags[0]) if tags else "")
        )
        keys = [
            "category",
            "difficulty",
            "author_name",
            "expert_time_estimate_min",
            "junior_time_estimate_min",
        ]
        normalized = {
            key: metadata.get(key)
            for key in keys
            if isinstance(metadata, dict) and metadata.get(key) is not None
        }
        normalized["tags"] = [str(tag) for tag in tags]
        if task_type:
            normalized["task_type"] = str(task_type)
        return normalized

    def _task_toml_metadata(self, task_path: Any) -> dict[str, Any]:
        path = Path(str(task_path)) / "task.toml"
        if not path.exists():
            return {}
        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError:
            return {}
        metadata = data.get("metadata")
        return metadata if isinstance(metadata, dict) else {}
