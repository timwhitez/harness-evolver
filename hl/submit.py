"""One-shot leaderboard submit gate."""

from __future__ import annotations

import os
import uuid
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hl.submission_evidence import (
    EvidenceError, input_reasons, inspect_job, summary_reasons,
)
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
    evidence: dict[str, Any] = field(default_factory=dict)


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
        reasons = input_reasons(
            score, tasks_evaluated, attempts_per_task, self.config.trigger_score,
            self.config.min_tasks_evaluated, self.config.min_attempts_per_task,
        )
        for field in (
            "enabled", "harbor_upload", "require_integrity_scan", "require_atif_trajectory",
            "require_full_regression", "require_clean_git", "require_no_uncommitted_harness_diff",
            "once_per_campaign", "stop_after_submit_attempt", "share_yes",
        ):
            if not isinstance(getattr(self.config, field), bool):
                reasons.append(f"{field} must be a boolean")
        if not valid_campaign_id(campaign_id):
            reasons.append("invalid campaign_id: use one safe ASCII identifier")
        if not isinstance(full_regression_passed, bool):
            reasons.append("full_regression_passed must be a boolean")
        if reasons:
            return SubmitGateResult(eligible=False, reasons=reasons)

        best_job = Path(best_job_dir)
        marker = self.submissions_dir / f"{campaign_id}.json"
        intent = self.submissions_dir / f"{campaign_id}.intent.json"
        if not self.config.enabled:
            reasons.append("submit.enabled is false")
        if os.path.lexists(marker) and self.config.once_per_campaign:
            reasons.append("campaign already has a submit result")
        if os.path.lexists(intent) and self.config.once_per_campaign:
            reasons.append("campaign already has a submit intent")
        if score < self.config.trigger_score:
            reasons.append(f"score {score} is below trigger {self.config.trigger_score}")
        if tasks_evaluated < self.config.min_tasks_evaluated:
            reasons.append(
                f"tasks evaluated {tasks_evaluated} is below minimum {self.config.min_tasks_evaluated}"
            )
        if self.config.require_full_regression and not full_regression_passed:
            reasons.append("full regression did not pass")

        evidence = None
        effective_attempts = attempts_per_task
        if not best_job.is_dir():
            reasons.append(f"best job dir not found or not a directory: {best_job}")
        elif self.config.harbor_upload and self.config.require_integrity_scan:
            try:
                evidence = inspect_job(best_job, require_atif=self.config.require_atif_trajectory)
            except EvidenceError as exc:
                reasons.append(str(exc))
            else:
                reasons.extend(summary_reasons(evidence, score, tasks_evaluated, attempts_per_task))
                effective_attempts = evidence.attempts_per_task
                # Rounded caller summaries may be compared, never used to raise
                # a raw evidence score across the configured submit threshold.
                if evidence.score < self.config.trigger_score:
                    reasons.append("verified Harbor job score is below the submit trigger")

        if effective_attempts is None:
            if self.config.min_attempts_per_task > 1:
                reasons.append("attempt evidence missing for leaderboard submit gate")
        else:
            short_tasks = sorted(task for task, n in effective_attempts.items()
                                 if n < self.config.min_attempts_per_task)
            if short_tasks:
                preview = ", ".join(short_tasks[:5])
                if len(short_tasks) > 5:
                    preview += f", ... (+{len(short_tasks) - 5} more)"
                reasons.append(f"tasks below minimum attempts {self.config.min_attempts_per_task}: {preview}")
            if len(effective_attempts) != tasks_evaluated:
                reasons.append("attempt evidence task count disagrees with evaluated tasks")

        # Do not run external commands for an already-invalid/duplicate request.
        if not reasons:
            if self.config.require_clean_git and not self._git_clean():
                reasons.append("git working tree is not clean")
            if self.config.require_no_uncommitted_harness_diff and self._harness_dirty():
                reasons.append("uncommitted harness/config/script diff exists")
            if self.config.harbor_upload and not self._harbor_auth_ok():
                reasons.append("harbor auth status failed")

        command: list[str] = []
        if self.config.harbor_upload:
            command = [self.harbor_bin, "upload", str(best_job)]
            command.append("--public" if self.config.visibility == "public" else "--private")
            for org in self.config.share_orgs:
                command.extend(["--share-org", org])
            for user in self.config.share_users:
                command.extend(["--share-user", user])
            if self.config.share_yes:
                command.append("--yes")

        return SubmitGateResult(
            eligible=not reasons, reasons=reasons, command=command,
            intent_path=str(intent), result_path=str(marker),
            evidence=evidence.as_dict() if evidence is not None else {},
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
            "evidence": gate.evidence,
            "integrity_scan_enabled": self.config.require_integrity_scan,
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
        if gate.evidence:
            try:
                current = inspect_job(Path(best_job_dir), require_atif=self.config.require_atif_trajectory)
                if current.fingerprint != gate.evidence["fingerprint"]:
                    raise EvidenceError("Harbor job changed after gate validation")
            except EvidenceError as exc:
                gate.eligible = False
                gate.reasons.append(f"upload cancelled before launch: {exc}")
                self._persist_result(gate, {**intent, "attempted": False, "submitted": False,
                                           "evidence_recheck_failed": True})
                return gate
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
        """Compatibility helper; use the same evidence contract as check()."""
        try:
            inspect_job(job_dir, require_atif=self.config.require_atif_trajectory)
        except EvidenceError as exc:
            return [str(exc)]
        return []
