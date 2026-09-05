"""One-shot leaderboard submit gate."""

from __future__ import annotations

import json
import os
import uuid
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.tools.leaderboard_guard import prohibited_command_reason
from hl.submission_storage import prepare_store, valid_campaign_id, write_exclusive_json


@dataclass
class SubmitConfig:
    enabled: bool = False
    trigger_score: float = 0.70
    min_tasks_evaluated: int = 89
    min_attempts_per_task: int = 5
    require_full_regression: bool = True
    require_clean_git: bool = True
    require_no_uncommitted_harness_diff: bool = True
    harbor_upload: bool = True
    visibility: str = "private"
    share_orgs: list[str] = field(default_factory=list)
    share_users: list[str] = field(default_factory=list)
    share_yes: bool = False
    stop_after_submit_attempt: bool = True
    once_per_campaign: bool = True
    require_integrity_scan: bool = True
    require_atif_trajectory: bool = True


@dataclass
class SubmitGateResult:
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    intent_path: str = ""
    result_path: str = ""
    attempted: bool = False
    submitted: bool = False
    returncode: int | None = None
    upload_skipped: bool = False
    terminal: bool = False
    intent_persisted: bool = False
    outcome_unknown: bool = False
    result_persistence_failed: bool = False


class SubmitGate:
    """Validate and perform at-most-once Harbor upload/submit."""

    def __init__(
        self,
        config: SubmitConfig | None = None,
        *,
        submissions_dir: str | Path = "trials/submissions",
        harbor_bin: str = "harbor",
    ) -> None:
        self.config = config or SubmitConfig()
        self.submissions_dir = Path(submissions_dir)
        prepare_store(self.submissions_dir)
        self.harbor_bin = harbor_bin

    def check(
        self,
        *,
        campaign_id: str,
        best_job_dir: str | Path,
        score: float,
        tasks_evaluated: int,
        full_regression_passed: bool,
        attempts_per_task: dict[str, int] | None = None,
    ) -> SubmitGateResult:
        if not valid_campaign_id(campaign_id):
            return SubmitGateResult(eligible=False, reasons=["invalid campaign_id: use 1-128 ASCII letters, digits, dot, underscore or hyphen; start with a letter/digit"])
        reasons: list[str] = []
        best_job = Path(best_job_dir)
        marker = self.submissions_dir / f"{campaign_id}.json"
        intent = self.submissions_dir / f"{campaign_id}.intent.json"
        duplicate_submit = False

        if not self.config.enabled:
            reasons.append("submit.enabled is false")
        if os.path.lexists(marker) and self.config.once_per_campaign:
            duplicate_submit = True
            reasons.append("campaign already has a submit result")
        if os.path.lexists(intent) and self.config.once_per_campaign:
            duplicate_submit = True
            reasons.append("campaign already has a submit intent")
        if score < self.config.trigger_score:
            reasons.append(f"score {score} is below trigger {self.config.trigger_score}")
        if tasks_evaluated < self.config.min_tasks_evaluated:
            reasons.append(
                f"tasks evaluated {tasks_evaluated} is below minimum {self.config.min_tasks_evaluated}"
            )
        if self.config.min_attempts_per_task > 1:
            if attempts_per_task is None:
                reasons.append("attempt evidence missing for leaderboard submit gate")
            else:
                short_tasks = [
                    task
                    for task, attempts in sorted(attempts_per_task.items())
                    if attempts < self.config.min_attempts_per_task
                ]
                if short_tasks:
                    preview = ", ".join(short_tasks[:5])
                    if len(short_tasks) > 5:
                        preview += f", ... (+{len(short_tasks) - 5} more)"
                    reasons.append(
                        "tasks below minimum attempts "
                        f"{self.config.min_attempts_per_task}: {preview}"
                    )
                if len(attempts_per_task) < tasks_evaluated:
                    reasons.append(
                        "attempt evidence task count "
                        f"{len(attempts_per_task)} is below evaluated tasks {tasks_evaluated}"
                    )
        if self.config.require_full_regression and not full_regression_passed:
            reasons.append("full regression did not pass")
        if not best_job.exists():
            reasons.append(f"best job dir not found: {best_job}")
        elif self.config.harbor_upload and self.config.require_integrity_scan:
            reasons.extend(self._integrity_reasons(best_job))
        if self.config.require_clean_git and not self._git_clean():
            reasons.append("git working tree is not clean")
        if self.config.require_no_uncommitted_harness_diff and self._harness_dirty():
            reasons.append("uncommitted harness/config/script diff exists")
        if self.config.harbor_upload and not duplicate_submit and not self._harbor_auth_ok():
            reasons.append("harbor auth status failed")

        command: list[str] = []
        if self.config.harbor_upload:
            command = [self.harbor_bin, "upload", str(best_job)]
            if self.config.visibility == "public":
                command.append("--public")
            else:
                command.append("--private")
            for org in self.config.share_orgs:
                command.extend(["--share-org", org])
            for user in self.config.share_users:
                command.extend(["--share-user", user])
            if self.config.share_yes:
                command.append("--yes")

        return SubmitGateResult(
            eligible=not reasons,
            reasons=reasons,
            command=command,
            intent_path=str(intent),
            result_path=str(marker),
        )

    def submit_once(
        self,
        *,
        campaign_id: str,
        best_job_dir: str | Path,
        score: float,
        tasks_evaluated: int,
        full_regression_passed: bool,
        attempts_per_task: dict[str, int] | None = None,
        dry_run: bool = False,
    ) -> SubmitGateResult:
        gate = self.check(
            campaign_id=campaign_id,
            best_job_dir=best_job_dir,
            score=score,
            tasks_evaluated=tasks_evaluated,
            full_regression_passed=full_regression_passed,
            attempts_per_task=attempts_per_task,
        )
        if not gate.eligible or dry_run:
            return gate

        gate.terminal = self.config.stop_after_submit_attempt
        if not self.config.once_per_campaign:
            attempt_id = f"{campaign_id}.{uuid.uuid4().hex}"
            gate.intent_path = str(self.submissions_dir / f"{attempt_id}.intent.json")
            gate.result_path = str(self.submissions_dir / f"{attempt_id}.json")
        intent = {
            "campaign_id": campaign_id,
            "best_job_dir": str(best_job_dir),
            "score": score,
            "tasks_evaluated": tasks_evaluated,
            "attempts_per_task": attempts_per_task or {},
            "min_attempts_per_task": self.config.min_attempts_per_task,
            "command": gate.command,
            "created_at": datetime.now().isoformat(),
        }
        try:
            write_exclusive_json(Path(gate.intent_path), intent)
        except FileExistsError:
            gate.eligible = False
            gate.reasons.append("campaign already has a submit intent (exclusive claim lost)")
            return gate
        except (OSError, ValueError, TypeError) as exc:
            gate.eligible = False
            gate.reasons.append(f"submit intent could not be durably persisted: {exc}")
            gate.outcome_unknown = os.path.lexists(gate.intent_path)
            return gate
        gate.intent_persisted = True
        gate.attempted = True
        if not self.config.harbor_upload:
            result = {
                **intent,
                "finished_at": datetime.now().isoformat(),
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "submitted": False,
                "upload_skipped": True,
            }
            self._persist_result(gate, result)
            gate.upload_skipped = True
            return gate

        try:
            completed = subprocess.run(gate.command, capture_output=True, text=True)
        except (OSError, subprocess.SubprocessError) as exc:
            # Keep the durable claim. A failed transport/wait does not establish
            # that the external service has not accepted the submission.
            gate.outcome_unknown = True
            gate.reasons.append(f"upload outcome unknown; reconcile intent before retry: {exc}")
            self._persist_result(gate, {**intent, "submitted": False, "outcome_unknown": True})
            return gate
        gate.returncode = completed.returncode
        gate.submitted = completed.returncode == 0
        result: dict[str, Any] = {
            **intent,
            "finished_at": datetime.now().isoformat(),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "submitted": completed.returncode == 0,
            "upload_skipped": False,
        }
        self._persist_result(gate, result)
        return gate

    def _persist_result(self, gate: SubmitGateResult, result: dict[str, Any]) -> None:
        try:
            write_exclusive_json(Path(gate.result_path), result)
        except (OSError, ValueError, TypeError) as exc:
            gate.result_persistence_failed = True
            gate.reasons.append(f"submit result persistence failed; durable intent retained: {exc}")

    def _git_clean(self) -> bool:
        completed = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
        return completed.returncode == 0 and completed.stdout.strip() == ""

    def _harness_dirty(self) -> bool:
        completed = subprocess.run(
            ["git", "status", "--short", "harness", "bench", "hl", "meta", "scripts", "config"],
            capture_output=True,
            text=True,
        )
        return completed.returncode != 0 or bool(completed.stdout.strip())

    def _harbor_auth_ok(self) -> bool:
        try:
            completed = subprocess.run(
                [self.harbor_bin, "auth", "status"],
                capture_output=True,
                text=True,
            )
        except OSError:
            return False
        return completed.returncode == 0

    def _integrity_reasons(self, job_dir: Path) -> list[str]:
        reasons: list[str] = []
        result_path = job_dir / "result.json"
        if not result_path.exists():
            return ["Harbor job result.json missing; cannot verify upload integrity"]
        try:
            job_result = json.loads(result_path.read_text(errors="replace"))
        except json.JSONDecodeError as exc:
            return [f"Harbor job result.json is not valid JSON: {exc}"]
        if not isinstance(job_result, dict):
            return ["Harbor job result.json must contain an object"]

        trial_results = job_result.get("trial_results") or []
        if not isinstance(trial_results, list) or not trial_results:
            reasons.append("Harbor job result.json contains no trial_results")
        for trial in trial_results:
            if not isinstance(trial, dict):
                continue
            trial_name = str(trial.get("trial_name") or "")
            if self.config.require_atif_trajectory and self._trial_passed(trial):
                if not trial_name:
                    reasons.append("passing Harbor trial is missing trial_name")
                elif not self._has_atif_trajectory(job_dir, trial_name):
                    reasons.append(
                        f"passing Harbor trial {trial_name} is missing ATIF trajectory"
                    )

        reasons.extend(self._scan_agent_artifacts(job_dir))
        return reasons

    def _trial_passed(self, trial: dict[str, Any]) -> bool:
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        if isinstance(rewards, dict):
            if "reward" in rewards:
                return self._float_value(rewards.get("reward")) >= 1.0
            numeric = [
                self._float_value(value)
                for value in rewards.values()
                if isinstance(value, int | float | str)
            ]
            numeric = [value for value in numeric if value is not None]
            if numeric:
                return sum(numeric) / len(numeric) >= 1.0
        return False

    def _float_value(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _has_atif_trajectory(self, job_dir: Path, trial_name: str) -> bool:
        trial_dir = job_dir / trial_name
        candidates = [
            trial_dir / "agent" / "trajectory.jsonl",
            trial_dir / "agent" / "trajectory.json",
            trial_dir / "trajectory.jsonl",
            trial_dir / "trajectory.json",
        ]
        return any(path.exists() and path.stat().st_size > 0 for path in candidates)

    def _scan_agent_artifacts(self, job_dir: Path) -> list[str]:
        reasons: list[str] = []
        for path in job_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(job_dir)
            parts = {part.lower() for part in relative.parts}
            if "agent" not in parts:
                continue
            if "tests" in parts or "solutions" in parts or "solution" in parts:
                reasons.append(
                    f"agent artifact contains prohibited benchmark material: {relative}"
                )
                continue
            if path.suffix.lower() not in {".json", ".jsonl", ".txt", ".md", ".log"}:
                continue
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(errors="replace")
            reason = prohibited_command_reason(text)
            if reason:
                reasons.append(f"agent artifact {relative} violates integrity guard: {reason}")
        return reasons
