"""Multi-attempt Harbor aggregation with incomplete-job recovery.

The primary implementation is retained in
:mod:`bench._harbor_issue6_aggregate_base`. This facade adds one missing path:
when the top-level job ``result.json`` is absent, malformed, or not an object,
Harbor may still have written complete per-trial ``*/result.json`` artifacts.
Multiple exact task attempts are normalized through the same inherited helper
contracts and aggregated instead of falling back to one/error-only result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench import _harbor_issue6_aggregate_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


class HarborRunner(_base.HarborRunner):
    """Aggregate exact attempts even when only subdirectory results survive."""

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
    ) -> _base.TrialResult:
        job_path = Path(job_dir)
        result_path = job_path / "result.json"

        if result_path.exists():
            try:
                top_level = json.loads(result_path.read_text())
            except (json.JSONDecodeError, OSError):
                source = "trial_subdirs_after_invalid_top_level"
            else:
                if isinstance(top_level, dict):
                    return super().parse_job_dir(
                        job_path,
                        task_id=task_id,
                        returncode=returncode,
                        stdout=stdout,
                        stderr=stderr,
                        wall_time=wall_time,
                        agent_config=agent_config,
                    )
                source = "trial_subdirs_after_non_object_top_level"
        else:
            source = "trial_subdirs_without_top_level"

        trial_results = self._load_trial_results_from_subdirs(job_path)
        matching = [
            result
            for result in trial_results
            if isinstance(result, dict)
            and self._trial_result_matches_task(result, task_id)
        ]
        if len(matching) <= 1:
            # Preserve the inherited single-attempt/error behavior. This PR only
            # changes the multi-attempt data-loss case from Issue #6.
            return super().parse_job_dir(
                job_path,
                task_id=task_id,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                wall_time=wall_time,
                agent_config=agent_config,
            )

        ordered_raw = sorted(matching, key=self._attempt_sort_key)
        status_counts = self._job_status_counts({}, trial_results)
        attempts = [
            self._normalize_subdir_attempt(
                raw_attempt=raw_attempt,
                job_path=job_path,
                result_path=result_path,
                task_id=task_id,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                wall_time=wall_time,
                agent_config=agent_config,
                total_result_count=len(trial_results),
                status_counts=status_counts,
            )
            for raw_attempt in ordered_raw
        ]
        aggregate = self._aggregate_attempts(
            attempts=attempts,
            raw_attempts=ordered_raw,
            job_path=job_path,
            task_id=task_id,
            result_path=result_path,
            total_result_count=len(trial_results),
            status_counts=status_counts,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            wall_time=wall_time,
            agent_config=agent_config,
        )
        aggregate.metadata.update(
            {
                "attempt_result_source": source,
                "top_level_job_result_present": result_path.exists(),
                "top_level_job_result_valid": False,
                "subdirectory_attempt_recovery": True,
            }
        )
        return aggregate

    def _normalize_subdir_attempt(
        self,
        *,
        raw_attempt: dict[str, Any],
        job_path: Path,
        result_path: Path,
        task_id: str,
        returncode: int,
        stdout: str,
        stderr: str,
        wall_time: float,
        agent_config: dict[str, Any] | None,
        total_result_count: int,
        status_counts: dict[str, int],
    ) -> _base.TrialResult:
        """Normalize one raw attempt using the inherited parser helper contract."""

        score, verifier_output, verifier_logs = self._score_from_harbor_trial(
            raw_attempt,
            job_path,
        )
        exception = raw_attempt.get("exception_info")
        verified = raw_attempt.get("verifier_result") is not None
        status = self._status_from_harbor(score, verified, exception, returncode)
        trial_name = str(raw_attempt.get("trial_name") or job_path.name)
        task_name = str(raw_attempt.get("task_name") or task_id)
        trial_dir = job_path / trial_name
        trajectory = self._load_trajectory(trial_dir)
        artifacts = self._list_artifacts(trial_dir)
        token_usage = self._token_usage(raw_attempt)
        trial_metrics = self._trial_metrics(raw_attempt, token_usage)

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
                    "Harbor recorded an agent exception after verifier reward 1.0; "
                    "preserving it as completion hygiene evidence."
                )
            elif done_after_worker_completion:
                errors.append(
                    "Harbor recorded an agent exception after the Worker called done; "
                    "preserving it as post-completion exception evidence."
                )

        metadata: dict[str, Any] = {
            "harbor_returncode": returncode,
            "job_result_path": str(result_path),
            "agent_info": raw_attempt.get("agent_info"),
            "model_config": self._model_config_metadata(agent_config, raw_attempt),
            "task_metadata": self._task_metadata(raw_attempt),
            "raw_rewards": (raw_attempt.get("verifier_result") or {}).get("rewards"),
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
            "harbor_trial_result_count": total_result_count,
            "job_result_status_counts": status_counts,
            "timeout_phase": timeout_phase,
            "attempts_observed_for_task": 1,
            "trial_metrics": trial_metrics,
            "normalized_from_subdirectory_result": True,
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

        return _base.TrialResult(
            trial_id=trial_name,
            task_id=task_name,
            task_domain=self._coerce_domain(raw_attempt),
            task_difficulty=self._coerce_difficulty(raw_attempt),
            status=status,
            score=score,
            trajectory=trajectory,
            error_log=errors,
            wall_time_seconds=wall_time,
            model_used=self._model_used(raw_attempt),
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
