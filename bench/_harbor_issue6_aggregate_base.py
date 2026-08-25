"""Harbor runner with exact identity and explicit multi-attempt aggregation.

The exact single-attempt parser from PR #33 is retained in
:mod:`bench._harbor_issue6_base`. Each matching Harbor attempt is normalized
through that parser independently, then combined with an order-independent
arithmetic-mean policy instead of silently selecting the first result.
"""

from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
import json
import re
from pathlib import Path
from typing import Any

from bench import _harbor_issue6_base as _base

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


_FORCED_TRIAL_RESULT: ContextVar[dict[str, Any] | None] = ContextVar(
    "harness_evolver_forced_harbor_trial_result",
    default=None,
)


class HarborRunner(_base.HarborRunner):
    """Normalize and aggregate every exact-identity result from one Harbor job."""

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
        if not result_path.exists():
            return super().parse_job_dir(
                job_path,
                task_id=task_id,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                wall_time=wall_time,
                agent_config=agent_config,
            )

        try:
            job_result = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            return super().parse_job_dir(
                job_path,
                task_id=task_id,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                wall_time=wall_time,
                agent_config=agent_config,
            )
        if not isinstance(job_result, dict):
            return super().parse_job_dir(
                job_path,
                task_id=task_id,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                wall_time=wall_time,
                agent_config=agent_config,
            )

        trial_results = job_result.get("trial_results") or []
        if not trial_results:
            trial_results = self._load_trial_results_from_subdirs(job_path)
        matching = [
            result
            for result in trial_results
            if isinstance(result, dict)
            and self._trial_result_matches_task(result, task_id)
        ]
        if len(matching) <= 1:
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
        attempts: list[_base.TrialResult] = []
        for raw_attempt in ordered_raw:
            token = _FORCED_TRIAL_RESULT.set(raw_attempt)
            try:
                attempts.append(
                    super().parse_job_dir(
                        job_path,
                        task_id=task_id,
                        returncode=returncode,
                        stdout=stdout,
                        stderr=stderr,
                        wall_time=wall_time,
                        agent_config=agent_config,
                    )
                )
            finally:
                _FORCED_TRIAL_RESULT.reset(token)

        return self._aggregate_attempts(
            attempts=attempts,
            raw_attempts=ordered_raw,
            job_path=job_path,
            task_id=task_id,
            result_path=result_path,
            total_result_count=len(trial_results),
            status_counts=self._job_status_counts(job_result, trial_results),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            wall_time=wall_time,
            agent_config=agent_config,
        )

    def _select_trial_result(
        self,
        trial_results: list[dict[str, Any]],
        task_id: str,
    ) -> dict[str, Any] | None:
        forced = _FORCED_TRIAL_RESULT.get()
        if forced is not None:
            return forced
        return super()._select_trial_result(trial_results, task_id)

    def _attempt_sort_key(self, result: dict[str, Any]) -> tuple[str, str, str]:
        task_id_value = result.get("task_id") or {}
        if isinstance(task_id_value, dict):
            task_identity = str(
                task_id_value.get("name") or task_id_value.get("path") or ""
            )
        else:
            task_identity = str(task_id_value)
        stable_json = json.dumps(result, sort_keys=True, separators=(",", ":"), default=str)
        return (
            str(result.get("trial_name") or ""),
            str(result.get("task_name") or task_identity),
            stable_json,
        )

    def _aggregate_attempts(
        self,
        *,
        attempts: list[_base.TrialResult],
        raw_attempts: list[dict[str, Any]],
        job_path: Path,
        task_id: str,
        result_path: Path,
        total_result_count: int,
        status_counts: dict[str, int],
        returncode: int,
        stdout: str,
        stderr: str,
        wall_time: float,
        agent_config: dict[str, Any] | None,
    ) -> _base.TrialResult:
        attempt_count = len(attempts)
        score = sum(float(attempt.score) for attempt in attempts) / attempt_count
        verified_count = sum(bool(attempt.verified) for attempt in attempts)
        pass_count = sum(
            bool(attempt.verified) and float(attempt.score) >= 1.0
            for attempt in attempts
        )
        aggregate_verified = verified_count == attempt_count
        aggregate_status = self._aggregate_status(attempts, aggregate_verified)

        token_usage: dict[str, int] = {}
        for attempt in attempts:
            for key, value in attempt.token_usage.items():
                if isinstance(value, int):
                    token_usage[key] = token_usage.get(key, 0) + value

        metric_totals: dict[str, int | float] = {}
        for attempt in attempts:
            metrics = attempt.metadata.get("trial_metrics") or {}
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                if key == "cache_hit_ratio" or isinstance(value, bool):
                    continue
                if isinstance(value, int | float):
                    metric_totals[key] = metric_totals.get(key, 0) + value
        for key, value in list(metric_totals.items()):
            if isinstance(value, float):
                metric_totals[key] = round(value, 6)
        prompt_tokens = token_usage.get("input", 0) + token_usage.get("cache", 0)
        if prompt_tokens:
            metric_totals["cache_hit_ratio"] = round(
                token_usage.get("cache", 0) / prompt_tokens,
                4,
            )

        combined_trajectory: list[dict[str, Any]] = []
        combined_artifacts: list[str] = []
        attempt_snapshots: list[dict[str, Any]] = []
        error_log: list[str] = []
        verifier_summaries: list[dict[str, Any]] = []
        verifier_log_parts: list[str] = []

        for index, attempt in enumerate(attempts, start=1):
            for event in attempt.trajectory:
                if isinstance(event, dict):
                    annotated = dict(event)
                    annotated["_harbor_attempt_index"] = index
                    annotated["_harbor_attempt_trial_id"] = attempt.trial_id
                    combined_trajectory.append(annotated)
            combined_artifacts.extend(
                f"{attempt.trial_id}/{artifact}" for artifact in attempt.artifacts
            )
            for error in attempt.error_log:
                prefixed = f"[{attempt.trial_id}] {error}"
                if prefixed not in error_log:
                    error_log.append(prefixed)

            snapshot = attempt.model_dump(
                mode="json",
                exclude={"trajectory", "harbor_stdout", "harbor_stderr", "timestamp"},
            )
            snapshot["canonical_attempt_index"] = index
            snapshot["trajectory_event_count"] = len(attempt.trajectory)
            attempt_snapshots.append(snapshot)
            verifier_summaries.append(
                {
                    "canonical_attempt_index": index,
                    "trial_id": attempt.trial_id,
                    "status": attempt.status.value,
                    "score": attempt.score,
                    "verified": attempt.verified,
                    "verifier_output": attempt.verifier_output[:4000],
                }
            )
            verifier_logs = str(attempt.metadata.get("verifier_logs") or "").strip()
            if verifier_logs:
                verifier_log_parts.append(f"## {attempt.trial_id}\n{verifier_logs}")

        status_counter = Counter(attempt.status.value for attempt in attempts)
        timeout_phases = [
            str(attempt.metadata.get("timeout_phase") or "") for attempt in attempts
        ]
        nonempty_timeout_phases = sorted({phase for phase in timeout_phases if phase})
        if nonempty_timeout_phases and len(set(timeout_phases)) == 1:
            aggregate_timeout_phase = nonempty_timeout_phases[0]
        elif nonempty_timeout_phases:
            aggregate_timeout_phase = "mixed_attempts"
        else:
            aggregate_timeout_phase = ""

        verifier_output = json.dumps(
            {
                "aggregation": "arithmetic_mean_all_attempts",
                "attempt_count": attempt_count,
                "mean_score": score,
                "pass_rate": pass_count / attempt_count,
                "verified_rate": verified_count / attempt_count,
                "attempts": verifier_summaries,
            },
            indent=2,
            sort_keys=True,
        )
        verifier_logs = "\n\n".join(verifier_log_parts)[: _base.VERIFIER_LOG_MAX_CHARS]
        all_verifier_infra = all(
            bool(attempt.metadata.get("verifier_infra_error")) for attempt in attempts
        )

        unique_models = sorted({attempt.model_used for attempt in attempts if attempt.model_used})
        unique_domains = sorted({attempt.task_domain.value for attempt in attempts})
        unique_difficulties = sorted(
            {attempt.task_difficulty.value for attempt in attempts}
        )
        safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_") or "task"
        if len(unique_models) == 1:
            aggregate_model = unique_models[0]
        elif unique_models:
            aggregate_model = "multiple"
        else:
            aggregate_model = ""

        metadata: dict[str, Any] = {
            "harbor_returncode": returncode,
            "job_result_path": str(result_path),
            "model_config": self._model_config_metadata(agent_config, raw_attempts[0]),
            "task_metadata": self._task_metadata(raw_attempts[0]),
            "harbor_trial_result_count": total_result_count,
            "job_result_status_counts": status_counts,
            "attempts_observed_for_task": attempt_count,
            "attempt_aggregation_policy": "arithmetic_mean_all_attempts",
            "attempt_order_independent": True,
            "attempt_ordering": "trial_name_then_task_identity_then_stable_json",
            "attempt_count": attempt_count,
            "attempt_verified_count": verified_count,
            "attempt_pass_count": pass_count,
            "attempt_pass_rate": pass_count / attempt_count,
            "attempt_verified_rate": verified_count / attempt_count,
            "attempt_status_counts": dict(sorted(status_counter.items())),
            "attempt_scores": [attempt.score for attempt in attempts],
            "attempt_results": attempt_snapshots,
            "raw_rewards": [
                attempt.metadata.get("raw_rewards") for attempt in attempts
            ],
            "verifier_logs": verifier_logs,
            "verifier_infra_error": all_verifier_infra,
            "timeout_phase": aggregate_timeout_phase,
            "trial_metrics": metric_totals,
            "attempt_metric_totals": metric_totals,
            "attempt_models": unique_models,
            "attempt_domains": unique_domains,
            "attempt_difficulties": unique_difficulties,
            "multi_attempt_aggregate": True,
        }
        metadata = self._with_verifier_runtime_prepare_timeout_metadata(metadata)

        return _base.TrialResult(
            trial_id=f"{job_path.name}__{safe_task_id}__aggregate",
            task_id=task_id,
            task_domain=attempts[0].task_domain,
            task_difficulty=attempts[0].task_difficulty,
            status=aggregate_status,
            score=score,
            trajectory=combined_trajectory,
            error_log=error_log,
            wall_time_seconds=wall_time,
            model_used=aggregate_model,
            token_usage=token_usage,
            verified=aggregate_verified,
            verifier_output=verifier_output,
            harbor_job_dir=str(job_path),
            harbor_trial_dir="",
            harbor_stdout=stdout,
            harbor_stderr=stderr,
            artifacts=sorted(combined_artifacts),
            metadata=metadata,
        )

    def _aggregate_status(
        self,
        attempts: list[_base.TrialResult],
        aggregate_verified: bool,
    ) -> _base.TrialStatus:
        if aggregate_verified:
            return (
                _base.TrialStatus.PASSED
                if all(attempt.score >= 1.0 for attempt in attempts)
                else _base.TrialStatus.FAILED
            )
        statuses = {attempt.status for attempt in attempts}
        for status in (
            _base.TrialStatus.ERROR,
            _base.TrialStatus.TIMEOUT,
            _base.TrialStatus.CANCELLED,
            _base.TrialStatus.RUNNING,
            _base.TrialStatus.UNVERIFIED,
        ):
            if status in statuses:
                return status
        return _base.TrialStatus.FAILED
