"""Multi-attempt aggregation constrained by one canonical task identity.

The aggregation and incomplete-job recovery implementations are retained in
:mod:`bench._harbor_issue6_recovery_base`. This facade computes the globally
allowed attempt set once and makes every inherited selection path use it. A
valid but stale top-level result that has no evidence for the requested task may
recover exact attempts from per-trial result files; contradictory or ambiguous
top-level identity evidence remains fail-closed.
"""

from __future__ import annotations

from contextvars import ContextVar
import json
from pathlib import Path
from typing import Any

from bench import _harbor_issue6_recovery_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, _value)


_ALLOWED_ATTEMPT_KEYS: ContextVar[frozenset[str] | None] = ContextVar(
    "harness_evolver_allowed_harbor_attempt_keys",
    default=None,
)


def _stable_attempt_key(result: dict[str, Any]) -> str:
    return json.dumps(result, sort_keys=True, separators=(",", ":"), default=str)


def _identity_mentions_requested(result: dict[str, Any], task_id: str) -> bool:
    """Return whether raw identity evidence refers to the requested task at all.

    This predicate is intentionally broader than an accepted identity match. It
    distinguishes a completely unrelated stale top-level result (safe to ignore
    in favour of exact subdirectory evidence) from a contradictory/ambiguous
    record that mentions the requested name or path and must remain fail-closed.
    """

    requested = str(task_id or "").strip()
    if not requested:
        return False
    identity = _base._task_identity(result)
    if _base._looks_path_like(requested):
        normalized = _base._normalise_path(requested)
        tail = _base._path_tail(normalized)
        return (
            normalized in identity.paths
            or requested in identity.names
            or tail in identity.names
            or any(_base._path_tail(path) == tail for path in identity.paths)
        )
    return requested in identity.names or any(
        _base._path_tail(path) == requested for path in identity.paths
    )


def _top_level_has_rejected_requested_evidence(
    candidates: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    task_id: str,
) -> bool:
    """Reject a convenient valid subset when a sibling contradicts the task.

    A top-level object is one authoritative Harbor snapshot. If any record in
    that snapshot mentions the requested identity but is excluded by the exact
    singleton matcher, the object is internally inconsistent. Selecting only
    the accepted records would make attribution depend on filtering order and
    could hide a stale or cross-dataset result.
    """

    accepted_keys = {_stable_attempt_key(item) for item in accepted}
    return any(
        _identity_mentions_requested(item, task_id)
        and _stable_attempt_key(item) not in accepted_keys
        for item in candidates
    )


class HarborRunner(_base.HarborRunner):
    """Aggregate attempts only after exact cross-record identity validation."""

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
        top_level_valid = False
        top_level_candidates: list[dict[str, Any]] = []

        if result_path.exists():
            try:
                top_level = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                top_level = None
            if isinstance(top_level, dict):
                top_level_valid = True
                raw_results = top_level.get("trial_results") or []
                if isinstance(raw_results, list):
                    top_level_candidates = [
                        item for item in raw_results if isinstance(item, dict)
                    ]

        force_no_match = False
        candidates: list[dict[str, Any]]
        if top_level_valid:
            accepted_top_level = self._matching_trial_results(
                top_level_candidates,
                task_id,
            )
            if accepted_top_level and not _top_level_has_rejected_requested_evidence(
                top_level_candidates,
                accepted_top_level,
                task_id,
            ):
                candidates = top_level_candidates
            elif accepted_top_level or any(
                _identity_mentions_requested(item, task_id)
                for item in top_level_candidates
            ):
                # The top-level result refers to the requested identity but is
                # contradictory across fields or attempts. Never hide that
                # evidence by selecting a convenient top-level/subdirectory
                # subset.
                candidates = top_level_candidates
                force_no_match = True
            else:
                recovered = self._recover_unrelated_top_level_from_subdirs(
                    job_path=job_path,
                    result_path=result_path,
                    task_id=task_id,
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                    wall_time=wall_time,
                    agent_config=agent_config,
                )
                if recovered is not None:
                    return recovered
                candidates = top_level_candidates
                force_no_match = True
        else:
            candidates = [
                item
                for item in self._load_trial_results_from_subdirs(job_path)
                if isinstance(item, dict)
            ]

        if force_no_match:
            allowed: frozenset[str] | None = frozenset()
        elif candidates:
            allowed = frozenset(
                _stable_attempt_key(item)
                for item in self._matching_trial_results(candidates, task_id)
            )
        else:
            allowed = None

        token = _ALLOWED_ATTEMPT_KEYS.set(allowed)
        try:
            return super().parse_job_dir(
                job_path,
                task_id=task_id,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                wall_time=wall_time,
                agent_config=agent_config,
            )
        finally:
            _ALLOWED_ATTEMPT_KEYS.reset(token)

    def _recover_unrelated_top_level_from_subdirs(
        self,
        *,
        job_path: Path,
        result_path: Path,
        task_id: str,
        returncode: int,
        stdout: str,
        stderr: str,
        wall_time: float,
        agent_config: dict[str, Any] | None,
    ) -> _base.TrialResult | None:
        """Recover exact attempts when a valid top-level result is wholly stale."""

        trial_results = [
            item
            for item in self._load_trial_results_from_subdirs(job_path)
            if isinstance(item, dict)
        ]
        matching = self._matching_trial_results(trial_results, task_id)
        if not matching:
            return None

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

        if len(attempts) == 1:
            result = attempts[0]
        else:
            result = self._aggregate_attempts(
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

        result.metadata.update(
            {
                "attempt_result_source": "trial_subdirs_after_unrelated_top_level",
                "top_level_job_result_present": True,
                "top_level_job_result_valid": True,
                "top_level_job_result_matched_task": False,
                "subdirectory_attempt_recovery": True,
            }
        )
        return result

    def _trial_result_matches_task(
        self,
        result: dict[str, Any],
        task_id: str,
    ) -> bool:
        allowed = _ALLOWED_ATTEMPT_KEYS.get()
        if allowed is not None:
            return _stable_attempt_key(result) in allowed
        return super()._trial_result_matches_task(result, task_id)
