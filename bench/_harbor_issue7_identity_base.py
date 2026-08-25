"""Harbor runner with exact task-identity matching and explicit diagnostics.

The inherited Harbor implementation is preserved in
:mod:`bench._harbor_issue7_base`; this compatibility module narrows only the
result/directory identity rules required by issue #7.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import posixpath
import re
from typing import Any

from bench import _harbor_issue7_base as _base

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

_BaseHarborRunner = _base.HarborRunner

_ATTEMPT_SUFFIX = re.compile(
    r"^(?:attempt[-_])?(?:"
    r"\d+|"
    r"[0-9A-Za-z]{7}|"
    r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
    r")$",
    flags=re.IGNORECASE,
)
_IDENTITY_FAILURE_MESSAGE = "Harbor result.json did not contain a matching trial result"


@dataclass
class _TaskIdentity:
    """Task names and task paths kept separate to avoid basename collisions."""

    names: set[str]
    paths: set[str]

    @classmethod
    def empty(cls) -> "_TaskIdentity":
        return cls(names=set(), paths=set())

    def update(self, other: "_TaskIdentity") -> None:
        self.names.update(other.names)
        self.paths.update(other.paths)

    @property
    def has_evidence(self) -> bool:
        return bool(self.names or self.paths)


def _normalise_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    normalized = posixpath.normpath(text)
    return "" if normalized == "." and text not in {".", "./"} else normalized


def _path_tail(value: object) -> str:
    text = _normalise_path(value).rstrip("/")
    return text.rsplit("/", 1)[-1] if text else ""


def _looks_path_like(value: object) -> bool:
    text = str(value or "")
    return "/" in text or "\\" in text


def _add_name(identity: _TaskIdentity, value: object) -> None:
    text = str(value or "").strip()
    if text:
        identity.names.add(text)


def _add_path(identity: _TaskIdentity, value: object) -> None:
    normalized = _normalise_path(value)
    if normalized:
        identity.paths.add(normalized)


def _add_scalar_identity(identity: _TaskIdentity, value: object) -> None:
    if _looks_path_like(value):
        _add_path(identity, value)
    else:
        _add_name(identity, value)


def _task_identity(result: dict[str, Any]) -> _TaskIdentity:
    """Extract explicit task names and paths without substring inference."""

    identity = _TaskIdentity.empty()
    _add_name(identity, result.get("task_name"))
    task_id_value = result.get("task_id")
    if isinstance(task_id_value, dict):
        _add_name(identity, task_id_value.get("name"))
        _add_path(identity, task_id_value.get("path"))
    else:
        _add_scalar_identity(identity, task_id_value)
    return identity


def _configured_task_identity(config: dict[str, Any]) -> _TaskIdentity:
    """Extract task identity from Harbor job/trial configuration shapes."""

    identity = _TaskIdentity.empty()
    _add_name(identity, config.get("task_name"))

    task_id = config.get("task_id")
    if isinstance(task_id, dict):
        _add_name(identity, task_id.get("name"))
        _add_path(identity, task_id.get("path"))
    else:
        _add_scalar_identity(identity, task_id)

    task = config.get("task")
    if isinstance(task, dict):
        _add_name(identity, task.get("name"))
        _add_path(identity, task.get("path"))
        _add_name(identity, task.get("task_name"))
    else:
        _add_scalar_identity(identity, task)

    dataset = config.get("dataset")
    if isinstance(dataset, dict):
        _add_name(identity, dataset.get("include_task_name"))
        _add_scalar_identity(identity, dataset.get("task"))
    return identity


def _identity_matches_requested(identity: _TaskIdentity, requested: object) -> bool:
    """Compare one unambiguous complete identity with the request.

    Structured path evidence is authoritative. Multiple distinct paths are
    contradictory and therefore cannot prove any request. Without path evidence,
    multiple distinct names are likewise ambiguous rather than aliases to guess
    between.
    """

    requested_text = str(requested or "").strip()
    if not requested_text:
        return False

    if identity.paths:
        if len(identity.paths) != 1:
            return False
        observed_path = next(iter(identity.paths))
        if _looks_path_like(requested_text):
            return observed_path == _normalise_path(requested_text)
        return _path_tail(observed_path) == requested_text

    if len(identity.names) != 1:
        return False
    return requested_text == next(iter(identity.names))


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _candidate_task_identity(path: Path) -> _TaskIdentity:
    """Return exact identity evidence recorded inside one fallback trial directory."""

    identity = _TaskIdentity.empty()
    result = _read_json_object(path / "result.json")
    if result is not None:
        identity.update(_task_identity(result))
    config = _read_json_object(path / "config.json")
    if config is not None:
        identity.update(_configured_task_identity(config))
    return identity


def _identity_labels(identity: _TaskIdentity) -> set[str]:
    return {
        *(f"name:{name}" for name in identity.names),
        *(f"path:{path}" for path in identity.paths),
    }


def _observed_task_identities(job_path: Path) -> list[str]:
    """Collect every explicit result/config identity for a failed selection."""

    observed: set[str] = set()
    job_result = _read_json_object(job_path / "result.json")
    if job_result is not None:
        trial_results = job_result.get("trial_results") or []
        if isinstance(trial_results, list):
            for result in trial_results:
                if isinstance(result, dict):
                    observed.update(_identity_labels(_task_identity(result)))

    if job_path.exists():
        try:
            candidates = sorted(
                path
                for path in job_path.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            )
        except OSError:
            candidates = []
        for candidate in candidates:
            identity = _candidate_task_identity(candidate)
            labels = _identity_labels(identity)
            if labels:
                observed.update(labels)
            else:
                observed.add(f"directory:{candidate.name}")
    return sorted(observed)


def _default_trial_task_prefix(requested_name: str) -> str:
    return requested_name[:32].rstrip("_-")


def _trial_directory_matches_task(name: str, requested_name: str) -> bool:
    normalized = _path_tail(name)
    prefixes = {requested_name, _default_trial_task_prefix(requested_name)} - {""}
    if normalized in prefixes:
        return True
    for task_prefix in prefixes:
        prefix = task_prefix + "__"
        if normalized.startswith(prefix) and _ATTEMPT_SUFFIX.fullmatch(
            normalized[len(prefix) :]
        ):
            return True
    return False


def _directory_name_loses_identity(requested_name: str) -> bool:
    return _default_trial_task_prefix(requested_name) != requested_name


def _fallback_candidate_matches(path: Path, requested: str) -> bool:
    requested_name = _path_tail(requested)
    if not requested_name or not _trial_directory_matches_task(path.name, requested_name):
        return False

    recorded = _candidate_task_identity(path)
    if recorded.has_evidence:
        return _identity_matches_requested(recorded, requested)

    if _looks_path_like(requested):
        return False
    return not _directory_name_loses_identity(requested_name)


class HarborRunner(_BaseHarborRunner):
    """HarborRunner variant that requires exact normalized task identity."""

    def _trial_result_matches_task(self, result: dict[str, Any], task_id: str) -> bool:
        return _identity_matches_requested(_task_identity(result), task_id)

    def _select_trial_result(
        self,
        trial_results: list[dict[str, Any]],
        task_id: str,
    ) -> dict[str, Any] | None:
        if not trial_results:
            return None
        matches = [
            result
            for result in trial_results
            if self._trial_result_matches_task(result, task_id)
        ]
        return matches[0] if matches else None

    def _fallback_trial_dir(self, job_path: Path, task_id: str) -> Path | None:
        if not job_path.exists():
            return None
        requested = str(task_id or "").strip()
        if not requested or not _path_tail(requested):
            return None
        candidates = [
            path
            for path in sorted(job_path.iterdir())
            if path.is_dir() and not path.name.startswith(".")
        ]
        matches = [
            path
            for path in candidates
            if _fallback_candidate_matches(path, requested)
        ]
        if len(matches) == 1:
            return matches[0]
        return None

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
        result = super().parse_job_dir(
            job_dir,
            task_id=task_id,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            wall_time=wall_time,
            agent_config=agent_config,
        )
        if not any(_IDENTITY_FAILURE_MESSAGE in error for error in result.error_log):
            return result

        observed = _observed_task_identities(Path(job_dir))
        rendered = ", ".join(observed) if observed else "<none>"
        diagnostic = (
            f"Requested task identity {task_id!r}; observed Harbor identities: "
            f"{rendered}"
        )
        if diagnostic not in result.error_log:
            result.error_log.append(diagnostic)
        result.metadata.update(
            {
                "task_identity_match_failed": True,
                "requested_task_identity": str(task_id),
                "observed_task_identities": observed,
            }
        )
        return result
