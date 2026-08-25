"""Multi-attempt aggregation constrained by one canonical task identity.

The aggregation and incomplete-job recovery implementations are retained in
:mod:`bench._harbor_issue6_recovery_base`. This facade computes the globally
allowed attempt set once and makes every inherited selection path use it.
"""

from __future__ import annotations

from contextvars import ContextVar
import json
from pathlib import Path
from typing import Any

from bench import _harbor_issue6_recovery_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


_ALLOWED_ATTEMPT_KEYS: ContextVar[frozenset[str] | None] = ContextVar(
    "harness_evolver_allowed_harbor_attempt_keys",
    default=None,
)


def _stable_attempt_key(result: dict[str, Any]) -> str:
    return json.dumps(result, sort_keys=True, separators=(",", ":"), default=str)


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
        candidates: list[dict[str, Any]] = []
        result_path = job_path / "result.json"
        if result_path.exists():
            try:
                top_level = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                top_level = None
            if isinstance(top_level, dict):
                raw_results = top_level.get("trial_results") or []
                if isinstance(raw_results, list):
                    candidates = [
                        item for item in raw_results if isinstance(item, dict)
                    ]
        if not candidates:
            candidates = [
                item
                for item in self._load_trial_results_from_subdirs(job_path)
                if isinstance(item, dict)
            ]

        allowed: frozenset[str] | None
        if candidates:
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

    def _trial_result_matches_task(
        self,
        result: dict[str, Any],
        task_id: str,
    ) -> bool:
        allowed = _ALLOWED_ATTEMPT_KEYS.get()
        if allowed is not None:
            return _stable_attempt_key(result) in allowed
        return super()._trial_result_matches_task(result, task_id)
