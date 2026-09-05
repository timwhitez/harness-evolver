"""Validate the exact finalized Harbor job selected for upload.

Supports legacy inline trial_results and Harbor 0.22 per-trial result files.
This is a TerminalBench single-step, normalized-reward submission contract,
not a replacement for Harbor's general-purpose result models. The score is the
mean of each task's mean reward, with every recorded error attempt scored zero.

Job directories must remain immutable through validation and upload. Anchored
no-follow reads, before/after inventories and a pre-upload recheck detect common
changes; they are NOT an atomic snapshot against an uncooperative writer.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterator

from harness.tools.descriptor_open import open_readonly_checked
from harness.tools.leaderboard_guard import prohibited_command_reason

_MAX_ENTRIES = 50_000
_MAX_TRIALS = 10_000
_MAX_DEPTH = 64
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_READ_BYTES = 256 * 1024 * 1024
_SCORE_TOLERANCE = 0.000050000001  # Scoring.build_summary rounds to four decimals.
_TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".log"}


class EvidenceError(ValueError):
    """The selected job cannot establish a complete, attributable submission."""


def positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def normalized_score(value: object) -> bool:
    # Test the range before conversion: arbitrarily large Python ints are safe.
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and 0 <= value <= 1 and math.isfinite(value))


def component(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and value == value.strip()
            and value not in {".", ".."} and len(value) <= 255
            and not any(c in value for c in "/\\")
            and all(ord(c) >= 32 and ord(c) != 127 for c in value))


def input_reasons(score: object, tasks: object, attempts: object,
                  trigger: object, min_tasks: object, min_attempts: object) -> list[str]:
    reasons = []
    for name, value in (("score", score), ("trigger_score", trigger)):
        if not normalized_score(value):
            reasons.append(f"{name} must be a finite number in [0, 1]")
    for name, value in (("tasks_evaluated", tasks), ("min_tasks_evaluated", min_tasks),
                        ("min_attempts_per_task", min_attempts)):
        if not positive_int(value):
            reasons.append(f"{name} must be a positive integer, not a boolean")
    if attempts is not None and (
        not isinstance(attempts, dict) or not attempts
        or any(not component(k) or not positive_int(v) for k, v in attempts.items())
    ):
        reasons.append("attempts_per_task must map nonempty task names to positive integers")
    return reasons


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


@contextmanager
def _root_fd(root: Path) -> Iterator[int]:
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise EvidenceError("safe submission evidence reads require POSIX no-follow support")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(root.anchor, flags)
    try:
        for part in root.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


class _JobFiles:
    """A bounded inventory and read scope anchored to one authorized root inode."""

    def __init__(self, root: Path, fd: int) -> None:
        self.root, self.fd = root, fd
        self.inventory = self._inventory(fd)
        self.hashes: dict[str, str] = {}
        self.bytes_read = 0

    def _inventory(self, root_fd: int) -> dict[str, tuple[int, ...]]:
        entries = {".": _stamp(os.fstat(root_fd))}

        def walk(fd: int, prefix: str, depth: int) -> None:
            if depth > _MAX_DEPTH:
                raise EvidenceError("Harbor job directory depth exceeds inspection limit")
            children = []
            with os.scandir(fd) as iterator:
                for entry in iterator:
                    if not component(entry.name):
                        raise EvidenceError("Harbor job contains an unsafe path component")
                    path = prefix + entry.name
                    info = entry.stat(follow_symlinks=False)
                    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                        raise EvidenceError(f"Harbor job contains a symlink or special entry: {path}")
                    entries[path] = _stamp(info)
                    if len(entries) > _MAX_ENTRIES:
                        raise EvidenceError("Harbor job entry count exceeds inspection limit")
                    if stat.S_ISDIR(info.st_mode):
                        children.append((path, entry.name))
            for path, name in sorted(children):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                                | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    if _stamp(os.fstat(child)) != entries[path]:
                        raise EvidenceError("Harbor job directory changed during inspection")
                    walk(child, path + "/", depth + 1)
                finally:
                    os.close(child)
        walk(root_fd, "", 0)
        return entries

    def read(self, relative: str) -> bytes:
        parts = relative.split("/")
        if any(not component(part) for part in parts):
            raise EvidenceError("unsafe evidence path")
        expected = self.inventory.get(relative)
        if expected is None:
            raise FileNotFoundError(relative)
        if not stat.S_ISREG(expected[2]):
            raise EvidenceError(f"evidence is not a regular file: {relative}")
        parent = os.dup(self.fd)
        descriptor = None
        try:
            prefix = []
            for part in parts[:-1]:
                prefix.append(part)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                                | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = child
                if _stamp(os.fstat(parent)) != self.inventory.get("/".join(prefix)):
                    raise EvidenceError("Harbor evidence directory changed before reading")
            descriptor, info = open_readonly_checked(parent, parts[-1])
            if _stamp(info) != expected:
                raise EvidenceError(f"Harbor evidence changed before reading: {relative}")
            if info.st_size > _MAX_FILE_BYTES:
                raise EvidenceError(f"evidence exceeds the per-file inspection limit: {relative}")
            payload = bytearray()
            while True:
                chunk = os.read(descriptor, min(65_536, _MAX_FILE_BYTES + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
                self.bytes_read += len(chunk)
                if len(payload) > _MAX_FILE_BYTES or self.bytes_read > _MAX_READ_BYTES:
                    raise EvidenceError("Harbor evidence exceeds the bounded inspection budget")
            if (_stamp(os.fstat(descriptor)) != expected
                    or _stamp(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)) != expected):
                raise EvidenceError(f"Harbor evidence changed while reading: {relative}")
            self.hashes[relative] = hashlib.sha256(payload).hexdigest()
            return bytes(payload)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)

    def fingerprint(self) -> str:
        # Reopen the configured path as well: a renamed root must not substitute
        # an unrelated job while the original directory fd remains readable.
        with _root_fd(self.root) as fresh:
            if self._inventory(fresh) != self.inventory:
                raise EvidenceError("Harbor job changed during evidence inspection")
        manifest = json.dumps({"inventory": self.inventory, "sha256": self.hashes},
                              sort_keys=True, ensure_ascii=True).encode()
        return hashlib.sha256(manifest).hexdigest()


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("JSON evidence contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise EvidenceError("JSON evidence contains a non-finite number")


def _json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError("JSON evidence contains a non-finite number")
    return number


def _document(files: _JobFiles, path: str) -> dict[str, Any]:
    try:
        result = json.loads(files.read(path).decode("utf-8"), object_pairs_hook=_object_pairs,
                            parse_constant=_reject_constant, parse_float=_json_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceError(f"invalid UTF-8 JSON evidence: {path}") from exc
    if not isinstance(result, dict):
        raise EvidenceError(f"JSON evidence must contain an object: {path}")
    return result


@dataclass(frozen=True)
class _Attempt:
    name: str
    task: str
    reward: float
    exception: str | None
    identity: str
    scope: str


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finalized(raw: dict[str, Any], label: str, *, required: bool = False) -> None:
    if "finished_at" not in raw and not required:
        return  # Legacy inline records predate timing fields.
    value = raw.get("finished_at")
    if not isinstance(value, str) or "T" not in value:
        raise EvidenceError(f"{label} is not finalized with an ISO datetime")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError(f"{label} has an invalid finished_at datetime") from exc


def _attempt(raw: object) -> _Attempt:
    if not isinstance(raw, dict):
        raise EvidenceError("trial_results entries must be objects")
    name, task = raw.get("trial_name"), raw.get("task_name")
    if not component(name) or not component(task):
        raise EvidenceError("Harbor trial requires safe nonempty trial_name and task_name")
    if raw.get("step_results") is not None:
        if not isinstance(raw["step_results"], list):
            raise EvidenceError(f"malformed Harbor step_results: {name}")
        if raw["step_results"]:
            raise EvidenceError("multi-step Harbor jobs require a separate submission scoring policy")
    _finalized(raw, f"Harbor trial {name}")
    error = raw.get("exception_info")
    if error is not None and (
        not isinstance(error, dict) or not isinstance(error.get("exception_type"), str)
        or not error["exception_type"].strip()
        or not isinstance(error.get("exception_message"), str)
    ):
        raise EvidenceError(f"malformed Harbor exception_info: {name}")
    verifier = raw.get("verifier_result")
    rewards = None
    if verifier is not None:
        if not isinstance(verifier, dict):
            raise EvidenceError(f"malformed Harbor verifier_result: {name}")
        rewards = verifier.get("rewards")
        if rewards is not None and (not isinstance(rewards, dict) or not rewards
                or any(not isinstance(k, str) or not k or not normalized_score(v)
                       for k, v in rewards.items())):
            raise EvidenceError(f"Harbor rewards must be nonempty finite numbers in [0, 1]: {name}")
    if rewards is None and error is None:
        raise EvidenceError(f"Harbor trial has neither verifier rewards nor a recorded error: {name}")
    reward = 0.0 if error is not None else float(
        rewards["reward"] if "reward" in rewards else math.fsum(rewards.values()) / len(rewards)
    )
    # Same-name tasks from different datasets/config paths must not be folded
    # into one apparent task with enough attempts. These fields are not paths to read.
    config = raw.get("config", {})
    if not isinstance(config, dict) or not isinstance(config.get("task", {}), dict):
        raise EvidenceError(f"malformed Harbor trial config: {name}")
    for field in ("source", "task_checksum"):
        if raw.get(field) is not None and not _nonempty_text(raw[field]):
            raise EvidenceError(f"malformed Harbor {field}: {name}")
    task_path = config.get("task", {}).get("path")
    if task_path is not None and not _nonempty_text(task_path):
        raise EvidenceError(f"malformed Harbor task path: {name}")
    task_id = raw.get("task_id")
    if task_id is not None:
        if not isinstance(task_id, dict) or not task_id:
            raise EvidenceError(f"malformed Harbor task_id: {name}")
        has_path = _nonempty_text(task_id.get("path"))
        has_package = _nonempty_text(task_id.get("org")) and _nonempty_text(task_id.get("name"))
        if not (has_path or has_package) or any(
            value is not None and not _nonempty_text(value) for value in task_id.values()
        ):
            raise EvidenceError(f"malformed Harbor task_id: {name}")
    agent = raw.get("agent_info")
    if agent is not None:
        if (not isinstance(agent, dict) or not _nonempty_text(agent.get("name"))
                or not _nonempty_text(agent.get("version"))):
            raise EvidenceError(f"malformed Harbor agent_info: {name}")
        model = agent.get("model_info")
        if model is not None and (not isinstance(model, dict)
                or not _nonempty_text(model.get("name"))
                or (model.get("provider") is not None and not _nonempty_text(model["provider"]))):
            raise EvidenceError(f"malformed Harbor model_info: {name}")
    scope = {"source": raw.get("source"), "agent_info": agent}
    identity = {**scope, "task_checksum": raw.get("task_checksum"),
                "task_path": task_path, "task_id": task_id}
    return _Attempt(name, task, reward, error["exception_type"] if error else None,
                    json.dumps(identity, sort_keys=True, ensure_ascii=True),
                    json.dumps(scope, sort_keys=True, ensure_ascii=True))


def _attempts(files: _JobFiles, job: dict[str, Any]) -> list[_Attempt]:
    inline = job.get("trial_results", [])
    if not isinstance(inline, list):
        raise EvidenceError("Harbor job trial_results must be a list")
    if len(inline) > _MAX_TRIALS:
        raise EvidenceError("Harbor job exceeds the trial inspection limit")
    by_name: dict[str, _Attempt] = {}
    for raw in inline:
        trial = _attempt(raw)
        if trial.name in by_name:
            raise EvidenceError("duplicate trial_name in Harbor job")
        by_name[trial.name] = trial
    subresults = sorted(p for p in files.inventory if len(p.split("/")) == 2
                        and p.endswith("/result.json"))
    if len(subresults) > _MAX_TRIALS:
        raise EvidenceError("Harbor job exceeds the trial inspection limit")
    disk = {}
    for path in subresults:
        trial = _attempt(_document(files, path))
        if trial.name != path.split("/")[0]:
            raise EvidenceError("Harbor trial_name disagrees with its result directory")
        disk[trial.name] = trial
    if by_name and disk and by_name != disk:
        raise EvidenceError("inline and per-directory Harbor trial evidence disagree")
    trials = list((disk or by_name).values())
    if not trials:
        raise EvidenceError("Harbor job result.json contains no trial_results or per-trial results")
    if not by_name and (not isinstance(job.get("stats"), dict)
                        or "n_total_trials" not in job):
        raise EvidenceError("per-trial layout requires Harbor job stats and n_total_trials")
    _finalized(job, "Harbor job", required=not by_name)
    if not by_name and not ({"n_completed_trials", "n_trials"} & job["stats"].keys()):
        raise EvidenceError("per-trial layout requires an explicit completed-trial count")
    if "n_total_trials" in job:
        total = job["n_total_trials"]
        if not positive_int(total) or total != len(trials):
            raise EvidenceError("Harbor n_total_trials disagrees with actual attempt records")
    if "stats" in job:
        stats = job["stats"]
        if not isinstance(stats, dict):
            raise EvidenceError("Harbor stats must be an object")
        expectations = {
            "n_completed_trials": len(trials), "n_trials": len(trials),
            "n_errored_trials": sum(t.exception is not None for t in trials),
            "n_errors": sum(t.exception is not None for t in trials),
            "n_cancelled_trials": sum(t.exception == "CancelledError" for t in trials),
            "n_running_trials": 0, "n_pending_trials": 0,
        }
        for key, expected in expectations.items():
            if key in stats and (type(stats[key]) is not int or stats[key] != expected):
                raise EvidenceError(f"Harbor {key} disagrees with completed attempt evidence")
    return trials


def _check_atif(files: _JobFiles, trial: _Attempt) -> None:
    # The adapter may also keep a native event log named trajectory.jsonl. Prefer
    # its complete ATIF trajectory.json. Never rescue an invalid preferred file
    # by selecting another candidate. A legacy .jsonl must hold one ATIF object.
    candidates = [f"{trial.name}/{prefix}trajectory.{suffix}"
                  for suffix in ("json", "jsonl") for prefix in ("agent/", "")]
    path = next((p for p in candidates if p in files.inventory), None)
    if path is None:
        raise EvidenceError(f"passing Harbor trial {trial.name} is missing ATIF trajectory")
    payload = _document(files, path)
    if "schema_version" not in payload:
        raise EvidenceError(f"ATIF schema_version is missing: {path}")
    try:
        from harbor.models.trajectories.trajectory import Trajectory
        trajectory = Trajectory.model_validate(payload, strict=True)
    except ImportError as exc:
        raise EvidenceError("Harbor ATIF validator is unavailable; install the declared Harbor dependency") from exc
    except ValueError as exc:
        # Do not echo Pydantic's input values: trajectories may contain secrets.
        raise EvidenceError(f"invalid ATIF trajectory schema or references: {path}") from exc
    # Our Worker exports complete standalone ATIF documents. External references
    # cannot prove complete evidence without a separate resolver; never fetch them.
    def check_refs(document: Any) -> None:
        if document.continued_trajectory_ref is not None:
            raise EvidenceError("ATIF continuation must be consolidated before submission")
        embedded = {s.trajectory_id for s in document.subagent_trajectories or []}
        for step in document.steps:
            if step.observation:
                for result in step.observation.results:
                    for ref in result.subagent_trajectory_ref or []:
                        if ref.trajectory_path is not None or ref.trajectory_id not in embedded:
                            raise EvidenceError("ATIF external or unresolved subagent reference")
        for subagent in document.subagent_trajectories or []:
            check_refs(subagent)
    check_refs(trajectory)


@dataclass(frozen=True)
class JobEvidence:
    score: float
    attempts_per_task: dict[str, int]
    fingerprint: str
    files_inspected: int

    def as_dict(self) -> dict[str, Any]:
        return {"policy": "terminalbench-task-mean-v1", "score": self.score,
                "tasks_evaluated": len(self.attempts_per_task),
                "attempts_per_task": self.attempts_per_task,
                "fingerprint": self.fingerprint, "files_inspected": self.files_inspected}


def inspect_job(job_dir: Path, *, require_atif: bool) -> JobEvidence:
    root = Path(os.path.abspath(job_dir))
    if root == Path(root.anchor):
        raise EvidenceError("filesystem root is not a Harbor job directory")
    try:
        with _root_fd(root) as fd:
            files = _JobFiles(root, fd)
            if "result.json" not in files.inventory:
                raise EvidenceError("Harbor job result.json missing; cannot verify upload integrity")
            job = _document(files, "result.json")
            trials = _attempts(files, job)
            # Keep the original policy scan, but do not silently skip large,
            # unreadable, invalid-UTF8 or symlinked agent artifacts.
            for path, info in sorted(files.inventory.items()):
                parts = path.split("/")
                if "agent" not in {part.lower() for part in parts}:
                    continue
                if {part.lower() for part in parts} & {"tests", "solutions", "solution"}:
                    raise EvidenceError(f"agent artifact contains prohibited benchmark material: {path}")
                if stat.S_ISREG(info[2]) and Path(path).suffix.lower() in _TEXT_SUFFIXES:
                    reason = prohibited_command_reason(files.read(path).decode("utf-8"))
                    if reason:
                        raise EvidenceError(f"agent artifact {path} violates integrity guard: {reason}")
            if len({trial.scope for trial in trials}) != 1:
                raise EvidenceError("ambiguous mixed model/dataset scopes require separate submission jobs")
            groups: dict[str, list[float]] = {}
            identities = {}
            for trial in trials:
                if trial.task in identities and identities[trial.task] != trial.identity:
                    raise EvidenceError(f"ambiguous same-name task/model evidence: {trial.task}")
                identities[trial.task] = trial.identity
                groups.setdefault(trial.task, []).append(trial.reward)
                if require_atif and trial.reward >= 1.0:
                    _check_atif(files, trial)
            score = math.fsum(math.fsum(v) / len(v) for v in groups.values()) / len(groups)
            return JobEvidence(score, {k: len(v) for k, v in sorted(groups.items())},
                               files.fingerprint(), len(files.hashes))
    except EvidenceError:
        raise
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise EvidenceError(f"cannot safely inspect Harbor job evidence ({type(exc).__name__})") from exc


def summary_reasons(evidence: JobEvidence, score: float, tasks: int,
                    attempts: dict[str, int] | None) -> list[str]:
    reasons = []
    if not math.isclose(score, evidence.score, rel_tol=0, abs_tol=_SCORE_TOLERANCE):
        reasons.append("supplied score disagrees with the selected Harbor job")
    if tasks != len(evidence.attempts_per_task):
        reasons.append("tasks_evaluated disagrees with the selected Harbor job")
    if attempts is not None and attempts != evidence.attempts_per_task:
        reasons.append("attempts_per_task disagrees with the selected Harbor job")
    return reasons
