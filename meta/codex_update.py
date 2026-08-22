"""Codex-backed UpdateEngine for outer Heuristic Learning iterations."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from hl.goals import GoalStore
from hl.loop_limits import (
    base_loop_limit_contract,
    disallowed_limit_terminal_reasons,
)
from hl.protocol import UpdateEngine
from hl.types import FeedbackSignal, HarnessPatch, TrialResult
from harness.tools.shell import external_agent_command_reason
from meta import report_contract
from meta.mechanism_coverage import clear_coverage_cache
from meta.packager import CodexWorkPacket, WorkPacketBuilder
from meta.reviewer import PatchReviewer, PatchReviewResult
from meta.update_policy import (
    classify_component_delta,
    merge_validation_commands,
    validation_ladder_for_changed_files,
)


def datetime_now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class CodexRunResult:
    packet_path: str
    events_path: str
    final_message_path: str
    diff_path: str
    exit_code: int
    review: PatchReviewResult
    final_report: dict[str, Any] = field(default_factory=dict)
    record_path: str = ""
    summary_path: str = ""
    validation_results_path: str = ""


def _blocking_reasons(findings: list[Any]) -> list[Any]:
    """Return blocking typed findings; untyped inputs fail closed."""

    return [
        finding
        for finding in findings
        if not isinstance(finding, report_contract.ReportViolation)
        or report_contract.is_blocking_violation(finding)
    ]


class CodexUpdateEngine(UpdateEngine):
    """Run Codex as the meta-coding updater for one bounded harness slice."""

    name = "codex_update"

    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        codex_bin: str = "codex",
        model: str = "gpt-5.4",
        sandbox: str = "workspace-write",
        reasoning_effort: str | None = None,
        provider_name: str | None = None,
        provider_base_url: str | None = None,
        provider_env_key: str | None = None,
        provider_wire_api: str = "responses",
        provider_requires_openai_auth: bool | None = None,
        codex_home: str | Path | None = None,
        codex_config_home: str | Path | None = None,
        timeout_seconds: int | None = None,
        validation_timeout_seconds: int = 600,
        allow_dirty_baseline: bool = True,
        events_dir: str | Path = "trials/diffs",
        dry_run: bool = False,
        env_file: str | Path | None = ".env.local",
        goal_store: GoalStore | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.codex_bin = codex_bin
        self.model = model
        self.sandbox = sandbox
        self.reasoning_effort = reasoning_effort
        self.provider_name = provider_name
        self.provider_base_url = provider_base_url
        self.provider_env_key = provider_env_key
        self.provider_wire_api = provider_wire_api
        self.provider_requires_openai_auth = provider_requires_openai_auth
        self.codex_home = str(codex_home) if codex_home else None
        self.codex_config_home = str(codex_config_home) if codex_config_home else None
        # Compatibility/audit field from older configs. Master, Codex update,
        # diagnostic, context, and Worker loops are not killed by a wall-clock
        # timeout; external cancellation must be explicit so long update runs
        # can finish and persist evidence.
        self.timeout_seconds = timeout_seconds
        # Compatibility/audit field for host validation reporting. Validation
        # commands are allowed to finish naturally; a timeout value must not
        # stop the Codex update sub-agent or reject an otherwise valid patch.
        self.validation_timeout_seconds = validation_timeout_seconds
        self.allow_dirty_baseline = allow_dirty_baseline
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.env_file = Path(env_file) if env_file else None
        self.packet_builder = WorkPacketBuilder(
            repo_root=self.repo_root,
            goal_store=goal_store,
            memory_path=self.events_dir.parent,
        )
        self.reviewer = PatchReviewer(repo_root=self.repo_root)
        self._last_run: CodexRunResult | None = None
        self._accepted_run_stack: list[CodexRunResult] = []

    def analyze_failures(
        self,
        feedback_signals: list[FeedbackSignal],
        trials: list[TrialResult],
    ) -> list[dict[str, Any]]:
        return [
            {
                "trial_id": signal.trial_id,
                "task_id": signal.task_id,
                "status": signal.status.value,
                "score": signal.score,
                "error_summary": signal.error_summary,
                "raw_errors": signal.raw_errors[:5],
            }
            for signal in feedback_signals
        ]

    def suggest_edits(
        self,
        findings: list[dict[str, Any]],
        current_harness: dict[str, Any],
    ) -> list[HarnessPatch]:
        # Codex edits the repo directly in run_update(); the protocol returns
        # metadata patches only after review. Keeping this method side-effect free
        # lets existing tests and HLLoop wiring call it safely.
        return []

    def apply_patch(self, patch: HarnessPatch) -> bool:
        return False

    def rollback_patch(self, patch: HarnessPatch) -> bool:
        return False

    def build_command(
        self,
        *,
        packet_path: str | Path,
        final_message_path: str | Path,
        schema_path: str | Path | None = None,
    ) -> list[str]:
        command = [
            self.codex_bin,
            "exec",
            "--json",
            "--cd",
            str(self.repo_root),
            "--model",
            self.model,
            "--sandbox",
            self.sandbox,
            "--output-last-message",
            str(final_message_path),
        ]
        if self.reasoning_effort and self.reasoning_effort != "none":
            command.extend(
                ["--config", f'model_reasoning_effort="{self.reasoning_effort}"']
            )
        command.extend(self._provider_config_overrides())
        if schema_path is not None:
            command.extend(["--output-schema", str(schema_path)])
        command.append(self._prompt_for_packet(packet_path))
        return command

    def _provider_config_overrides(self) -> list[str]:
        if not self.provider_base_url:
            return []
        provider_name = self.provider_name or "custom"
        fields = {
            "name": provider_name,
            "base_url": self.provider_base_url,
            "wire_api": self.provider_wire_api or "responses",
        }
        if self.provider_env_key:
            fields["env_key"] = self.provider_env_key
        inline_fields = [
            f"{key} = {json.dumps(value)}" for key, value in fields.items()
        ]
        if self.provider_requires_openai_auth is not None:
            inline_fields.append(
                "requires_openai_auth = "
                + ("true" if self.provider_requires_openai_auth else "false")
            )
        return [
            "--config",
            f'model_provider="{provider_name}"',
            "--config",
            f"model_providers.{provider_name}={{ "
            + ", ".join(inline_fields)
            + " }",
        ]

    def run_update(
        self,
        *,
        failures: list[TrialResult],
        current_harness: dict[str, Any],
        allowed_edit_paths: list[str] | None = None,
        regression_contracts: list[str] | None = None,
        required_validation_commands: list[str] | None = None,
    ) -> CodexRunResult:
        packet = self.packet_builder.build(
            failures=failures,
            current_harness=current_harness,
            allowed_edit_paths=allowed_edit_paths,
            regression_contracts=regression_contracts,
            required_validation_commands=required_validation_commands,
        )
        reviewer = self._reviewer_for(allowed_edit_paths)
        baseline_changed_files = reviewer.changed_files()
        baseline_snapshot = (
            self._snapshot_baseline_files(baseline_changed_files)
            if baseline_changed_files and self.allow_dirty_baseline
            else {}
        )
        run_dir = self.events_dir / packet.packet_id
        run_dir.mkdir(parents=True, exist_ok=True)
        packet_path = self.packet_builder.write(packet, run_dir / "codex_update_packet.json")
        schema_path = self._write_schema(packet, run_dir / "codex_report_schema.json")
        self._write_review_context(
            run_dir=run_dir,
            baseline_changed_files=baseline_changed_files,
            baseline_snapshot=baseline_snapshot,
        )
        final_path = run_dir / "final_message.json"
        events_path = run_dir / "codex_events.jsonl"
        diff_path = run_dir / "git.diff"

        skip_decision = (packet.mission_debug.get("evidence_summary") or {}).get(
            "skip_codex_update"
        )
        if isinstance(skip_decision, dict) and skip_decision:
            skip_reason = str(skip_decision.get("reason") or "").strip() or (
                "no uncovered mission candidate is available"
            )
            covered_ids = [
                str(candidate_id)
                for candidate_id in skip_decision.get("covered_candidate_ids") or []
                if str(candidate_id).strip()
            ]
            events_path.write_text("")
            final_report = self._contract_report_defaults(
                status="noop",
                summary="Codex was skipped before execution: " + skip_reason,
                skipped_validation_reason="no diff; " + skip_reason,
                memory_concise="Skipped a covered mission update before codex exec.",
                memory_detailed=(
                    skip_reason
                    + (" Covered candidates: " + ", ".join(covered_ids) if covered_ids else "")
                ),
                loophole_review=[
                    "mission candidates were compared with deterministic Worker policy/test metadata and accepted update memory"
                ],
                loophole_fixes=[
                    "stopped before codex exec instead of requesting a speculative duplicate patch"
                ],
            )
            final_path.write_text(json.dumps(final_report, indent=2))
            diff_path.write_text("")
            review = PatchReviewResult(
                accepted=False,
                changed_files=[],
                violations=[
                    report_contract.violation(
                        "update.skip_all_candidates_covered",
                        skip_reason
                        + (
                            "; covered candidates: " + ", ".join(covered_ids)
                            if covered_ids
                            else ""
                        ),
                    )
                ],
            )
            record_path, summary_path = self._write_update_records(
                run_dir=run_dir,
                packet=packet,
                review=review,
                exit_code=0,
                final_report=final_report,
                diff_path=diff_path,
                validation_results_path=None,
            )
            result = CodexRunResult(
                packet_path=str(packet_path),
                events_path=str(events_path),
                final_message_path=str(final_path),
                diff_path=str(diff_path),
                exit_code=0,
                review=review,
                final_report=final_report,
                record_path=str(record_path),
                summary_path=str(summary_path),
                validation_results_path="",
            )
            self._write_review(run_dir, review, exit_code=0)
            self._last_run = result
            return result

        command = self.build_command(
            packet_path=packet_path,
            final_message_path=final_path,
            schema_path=schema_path,
        )

        if self.dry_run:
            events_path.write_text("")
            final_report = self._contract_report_defaults(
                status="noop",
                summary="dry-run: Codex was not executed",
                skipped_validation_reason="dry-run",
                memory_concise="Dry-run did not execute Codex.",
                memory_detailed=(
                    "Dry-run wrote the work packet and output schema only; "
                    "no Worker or harness files changed."
                ),
            )
            final_path.write_text(json.dumps(final_report, indent=2))
            diff_path.write_text("")
            review = PatchReviewResult(
                accepted=False,
                changed_files=[],
                violations=[report_contract.violation("update.dry_run", "dry-run")],
            )
            record_path, summary_path = self._write_update_records(
                run_dir=run_dir,
                packet=packet,
                review=review,
                exit_code=0,
                final_report=final_report,
                diff_path=diff_path,
                validation_results_path=None,
            )
            result = CodexRunResult(
                packet_path=str(packet_path),
                events_path=str(events_path),
                final_message_path=str(final_path),
                diff_path=str(diff_path),
                exit_code=0,
                review=review,
                final_report=final_report,
                record_path=str(record_path),
                summary_path=str(summary_path),
                validation_results_path="",
            )
            self._last_run = result
            if review.accepted:
                self._accepted_run_stack.append(result)
            return result

        if baseline_changed_files and not self.allow_dirty_baseline:
            events_path.write_text("")
            final_report = self._contract_report_defaults(
                status="rejected",
                summary="Codex was not executed because the baseline worktree is dirty.",
                skipped_validation_reason="baseline worktree is dirty",
                memory_concise="Codex update skipped because the baseline worktree was dirty.",
                memory_detailed=(
                    "The updater refused to start so unrelated local diffs would not be "
                    "mixed with a generated Worker/harness patch."
                ),
                loophole_review=["baseline worktree has uncommitted changes"],
                loophole_fixes=["stop before Codex edits to avoid mixing unrelated diffs"],
                failed_directions_to_avoid=[
                    "Do not run a real Codex update on a dirty baseline unless isolated dirty-baseline mode is explicitly enabled."
                ],
            )
            final_path.write_text(json.dumps(final_report, indent=2))
            diff_path.write_text("")
            review = PatchReviewResult(
                accepted=False,
                changed_files=[],
                violations=[
                    report_contract.violation(
                        "update.dirty_baseline",
                        "baseline worktree has uncommitted changes; "
                        "commit/stash them or rerun with allow_dirty_baseline",
                    )
                ],
            )
            record_path, summary_path = self._write_update_records(
                run_dir=run_dir,
                packet=packet,
                review=review,
                exit_code=0,
                final_report=final_report,
                diff_path=diff_path,
                validation_results_path=None,
            )
            result = CodexRunResult(
                packet_path=str(packet_path),
                events_path=str(events_path),
                final_message_path=str(final_path),
                diff_path=str(diff_path),
                exit_code=0,
                review=review,
                final_report=final_report,
                record_path=str(record_path),
                summary_path=str(summary_path),
                validation_results_path="",
            )
            self._write_review(run_dir, review, exit_code=0)
            self._last_run = result
            return result

        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            env=self._subprocess_env(),
        )
        events_path.write_text(completed.stdout)
        if completed.stderr:
            (run_dir / "codex_stderr.txt").write_text(completed.stderr)

        review, diff_text = self._review_isolated_delta(reviewer, baseline_snapshot)
        diff_path.write_text(diff_text)
        final_report = self._load_final_report(final_path)
        dynamic_ladder = validation_ladder_for_changed_files(
            review.changed_files,
            repo_root=self.repo_root,
        )
        effective_validation_commands = merge_validation_commands(
            required_validation_commands or [],
            dynamic_ladder,
        )
        # Dirty-baseline files that Codex did not actually modify are excluded
        # from the isolated delta; exempt them from report over/under-report
        # gates so an unrelated local edit cannot roll back a valid patch (T3).
        ignored_baseline_files = [
            path
            for path in baseline_changed_files
            if path not in set(review.changed_files)
        ]
        review = self._apply_report_gates(
            review,
            exit_code=completed.returncode,
            final_report=final_report,
            required_validation_commands=required_validation_commands or [],
            exec_output_text=completed.stdout + "\n" + completed.stderr,
            external_research_recommended=(
                packet.external_research_policy.get("status") == "recommended"
            ),
            external_research_policy=packet.external_research_policy,
            host_validation_commands=effective_validation_commands,
            rejected_update_buffer=packet.rejected_update_buffer,
            runner_pivot_policy=packet.runner_pivot_policy,
            change_evaluation_digest=packet.change_evaluation_digest,
            prior_update_lesson_entries=packet.prior_update_lesson_entries,
            failure_pattern_digest=packet.failure_pattern_digest,
            mission_debug=packet.mission_debug,
            ignore_files=ignored_baseline_files,
        )
        validation_results_path: Path | None = None
        if review.accepted and review.changed_files:
            review, validation_results_path = self._run_host_validation_gate(
                review,
                commands=effective_validation_commands,
                run_dir=run_dir,
            )
        if review.changed_files and not review.accepted:
            review = self._rollback_rejected_delta(review, diff_path)
        record_path, summary_path = self._write_update_records(
            run_dir=run_dir,
            packet=packet,
            review=review,
            exit_code=completed.returncode,
            final_report=final_report,
            diff_path=diff_path,
            validation_results_path=validation_results_path,
            validation_ladder=dynamic_ladder,
        )
        result = CodexRunResult(
            packet_path=str(packet_path),
            events_path=str(events_path),
            final_message_path=str(final_path),
            diff_path=str(diff_path),
            exit_code=completed.returncode,
            review=review,
            final_report=final_report,
            record_path=str(record_path),
            summary_path=str(summary_path),
            validation_results_path=str(validation_results_path or ""),
        )
        self._write_review(run_dir, review, exit_code=completed.returncode)
        self._last_run = result
        if review.accepted:
            self._accepted_run_stack.append(result)
        return result

    def _rollback_rejected_delta(
        self,
        review: PatchReviewResult,
        diff_path: Path,
    ) -> PatchReviewResult:
        try:
            has_diff = bool(diff_path.read_text().strip())
        except OSError:
            has_diff = False
        if not has_diff:
            return review
        rolled_back = self.reviewer.rollback(diff_path)
        findings = report_contract.ViolationCollector(review.violations)
        if rolled_back:
            findings.add("update.rollback_applied", "rolled back rejected Codex delta")
        else:
            findings.add("update.rollback_failed", "failed to roll back rejected Codex delta")
        return PatchReviewResult(
            accepted=False,
            changed_files=review.changed_files,
            violations=findings.violations,
        )

    def rollback_last(self) -> bool:
        if self._last_run is None:
            return False
        return self.reviewer.rollback(self._last_run.diff_path)

    def rollback_last_accepted(self) -> bool:
        if not self._accepted_run_stack:
            return False
        run = self._accepted_run_stack.pop()
        return self.reviewer.rollback(run.diff_path)

    def rollback_diff(self, diff_path: str | Path) -> bool:
        return self.reviewer.rollback(diff_path)

    def _write_schema(self, packet: CodexWorkPacket, path: Path) -> Path:
        path.write_text(json.dumps(packet.expected_report_schema, indent=2))
        return path

    def _write_review_context(
        self,
        *,
        run_dir: Path,
        baseline_changed_files: list[str],
        baseline_snapshot: dict[str, bytes | None],
    ) -> Path:
        """Persist non-secret baseline identity for pre-delivery report lint."""

        baseline_files: list[dict[str, Any]] = []
        for path in sorted(baseline_changed_files):
            content = baseline_snapshot.get(path)
            baseline_files.append(
                {
                    "path": path,
                    "exists": content is not None,
                    "sha256": (
                        hashlib.sha256(content).hexdigest()
                        if content is not None
                        else ""
                    ),
                    "size_bytes": len(content) if content is not None else 0,
                }
            )
        context_path = run_dir / "review_context.json"
        context_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "baseline_files": baseline_files,
                    "allow_dirty_baseline": self.allow_dirty_baseline,
                    "actual_changed_files_source": (
                        "current worktree content minus baseline file hashes"
                    ),
                },
                indent=2,
            )
        )
        return context_path

    def _reviewer_for(self, allowed_edit_paths: list[str] | None) -> PatchReviewer:
        if not allowed_edit_paths:
            return self.reviewer
        return PatchReviewer(
            repo_root=self.repo_root,
            allowed_roots=allowed_edit_paths,
        )

    def _timeout_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return str(value)

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.env_file is not None:
            path = (
                self.env_file
                if self.env_file.is_absolute()
                else self.repo_root / self.env_file
            )
            if path.exists():
                for raw_line in path.read_text().splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if not key:
                        continue
                    value = value.strip()
                    if (
                        len(value) >= 2
                        and value[0] == value[-1]
                        and value[0] in {"'", '"'}
                    ):
                        value = value[1:-1]
                    env[key] = value
        if self.codex_home:
            home = self._configured_path(self.codex_home)
            env["HOME"] = home
            env["CODEX_HOME"] = self._configured_path(
                self.codex_config_home or str(Path(home) / ".codex")
            )
        elif self.codex_config_home:
            env["CODEX_HOME"] = self._configured_path(self.codex_config_home)
        return env

    def _configured_path(self, value: str) -> str:
        return str(Path(os.path.expandvars(value)).expanduser())

    def _snapshot_baseline_files(self, paths: list[str]) -> dict[str, bytes | None]:
        snapshot: dict[str, bytes | None] = {}
        for path in paths:
            full_path = self.repo_root / path
            if full_path.is_file():
                snapshot[path] = full_path.read_bytes()
            elif not full_path.exists():
                snapshot[path] = None
        return snapshot

    def _review_isolated_delta(
        self,
        reviewer: PatchReviewer,
        baseline_snapshot: dict[str, bytes | None],
    ) -> tuple[PatchReviewResult, str]:
        changed_files, diff_text, binary_files = self._isolated_delta(baseline_snapshot)
        review = reviewer.review_delta(changed_files, diff_text)
        if binary_files:
            restored_binary_files = self._restore_delta_files(binary_files, baseline_snapshot)
            findings = report_contract.ViolationCollector(review.violations)
            reason = (
                "codex delta contains binary file changes that cannot be "
                "reviewed or rolled back as a text patch: "
                + ", ".join(binary_files)
            )
            findings.add("update.binary_delta", reason)
            if restored_binary_files:
                findings.add(
                    "update.binary_delta",
                    "restored binary delta files to pre-Codex content: "
                    + ", ".join(restored_binary_files),
                )
            review = PatchReviewResult(
                accepted=False,
                changed_files=review.changed_files,
                violations=findings.violations,
            )
        return review, diff_text

    def _isolated_delta(
        self,
        baseline_snapshot: dict[str, bytes | None],
    ) -> tuple[list[str], str, list[str]]:
        after_changed_files = self.reviewer.changed_files()
        ignored_roots = self._ignored_delta_roots()
        candidate_paths = sorted(
            path
            for path in set(baseline_snapshot) | set(after_changed_files)
            if not self._is_ignored_delta_path(path, ignored_roots)
        )
        changed_files: list[str] = []
        diff_parts: list[str] = []
        binary_files: list[str] = []

        for path in candidate_paths:
            before = (
                baseline_snapshot[path]
                if path in baseline_snapshot
                else self._head_file_bytes(path)
            )
            after = self._current_file_bytes(path)
            if before == after:
                continue
            changed_files.append(path)
            diff, binary = self._file_delta_diff(path, before, after)
            if binary:
                binary_files.append(path)
            if diff:
                diff_parts.append(diff)

        return changed_files, "".join(diff_parts), binary_files

    def _ignored_delta_roots(self) -> list[str]:
        try:
            events_dir = self.events_dir
            if events_dir.is_absolute():
                events_dir = events_dir.relative_to(self.repo_root)
            normalized = str(events_dir).replace(os.sep, "/").strip("/")
        except ValueError:
            return []
        return [normalized] if normalized else []

    def _is_ignored_delta_path(self, path: str, ignored_roots: list[str]) -> bool:
        normalized = path.replace(os.sep, "/")
        return any(
            normalized == root or normalized.startswith(root + "/")
            for root in ignored_roots
        )

    def _current_file_bytes(self, path: str) -> bytes | None:
        full_path = self.repo_root / path
        if not full_path.is_file():
            return None
        return full_path.read_bytes()

    def _head_file_bytes(self, path: str) -> bytes | None:
        completed = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=self.repo_root,
            capture_output=True,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout

    def _file_delta_diff(
        self,
        path: str,
        before: bytes | None,
        after: bytes | None,
    ) -> tuple[str, bool]:
        if self._looks_binary(before) or self._looks_binary(after):
            return "", True
        before_text = "" if before is None else before.decode("utf-8")
        after_text = "" if after is None else after.decode("utf-8")
        before_lines = before_text.splitlines(keepends=True)
        after_lines = after_text.splitlines(keepends=True)
        fromfile = "/dev/null" if before is None else f"a/{path}"
        tofile = "/dev/null" if after is None else f"b/{path}"
        return (
            self._format_unified_diff_lines(
                difflib.unified_diff(
                    before_lines,
                    after_lines,
                    fromfile=fromfile,
                    tofile=tofile,
                )
            ),
            False,
        )

    def _format_unified_diff_lines(self, lines: Any) -> str:
        formatted: list[str] = []
        for line in lines:
            if line.endswith("\n"):
                formatted.append(line)
            else:
                formatted.append(line + "\n")
                formatted.append("\\ No newline at end of file\n")
        return "".join(formatted)

    def _looks_binary(self, content: bytes | None) -> bool:
        if content is None:
            return False
        if b"\0" in content:
            return True
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            return True
        return False

    def _restore_delta_files(
        self,
        paths: list[str],
        baseline_snapshot: dict[str, bytes | None],
    ) -> list[str]:
        restored: list[str] = []
        for path in paths:
            before = (
                baseline_snapshot[path]
                if path in baseline_snapshot
                else self._head_file_bytes(path)
            )
            full_path = self.repo_root / path
            if before is None:
                if full_path.is_file():
                    full_path.unlink()
                    restored.append(path)
                continue
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if not full_path.exists() or not full_path.is_file() or full_path.read_bytes() != before:
                full_path.write_bytes(before)
                restored.append(path)
        return restored

    def _write_review(
        self,
        run_dir: Path,
        review: PatchReviewResult,
        *,
        exit_code: int,
    ) -> None:
        (run_dir / "review.json").write_text(
            json.dumps(
                {
                    "accepted": review.accepted,
                    "reasons": review.reasons,
                    "reason_details": review.reason_details,
                    "changed_files": review.changed_files,
                    "exit_code": exit_code,
                },
                indent=2,
            )
        )
        if len(run_dir.parents) >= 2:
            clear_coverage_cache(run_dir.parents[1])

    def _run_host_validation_gate(
        self,
        review: PatchReviewResult,
        *,
        commands: list[str],
        run_dir: Path,
    ) -> tuple[PatchReviewResult, Path]:
        results_path = run_dir / "validation_results.json"
        results: list[dict[str, Any]] = []
        findings = report_contract.ViolationCollector(review.violations)
        if not commands:
            findings.add(
                "validation.host_commands_required",
                "host validation was required for changed files but no commands were supplied"
            )
            results_path.write_text(json.dumps({"commands": results}, indent=2))
            return (
                PatchReviewResult(
                    accepted=False,
                    changed_files=review.changed_files,
                    violations=findings.violations,
                ),
                results_path,
            )

        env = self._subprocess_env()
        for index, command in enumerate(commands, start=1):
            stdout_path = run_dir / f"validation_{index:02d}_stdout.txt"
            stderr_path = run_dir / f"validation_{index:02d}_stderr.txt"
            started_at = time.time()
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                shell=True,
                capture_output=True,
                text=True,
                env=env,
            )
            timed_out = False
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr

            stdout_path.write_text(stdout or "")
            stderr_path.write_text(stderr or "")
            result = {
                "command": command,
                "returncode": returncode,
                "timed_out": timed_out,
                "validation_timeout_seconds_audit_only": self.validation_timeout_seconds,
                "validation_timeout_seconds_stop_condition": False,
                "validation_command_timeout_stop_condition": False,
                "host_validation_timeout_seconds_stop_condition": False,
                "codex_update_sub_agent_stop_condition": False,
                "sub_agent_attempt_count_stop_condition": False,
                "sub_agent_round_limit_stop_condition": False,
                "master_loop_stop_condition": False,
                "worker_loop_stop_condition": False,
                "loop_stop_condition": False,
                "time_round_token_limit_driven": False,
                "duration_seconds": round(time.time() - started_at, 3),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
            results.append(result)
            if returncode != 0:
                findings.add(
                    "validation.host_command_failed",
                    f"host validation command failed ({returncode}): {command}",
                )

        results_path.write_text(json.dumps({"commands": results}, indent=2))
        return (
            PatchReviewResult(
                accepted=not findings.blocking,
                changed_files=review.changed_files,
                violations=findings.violations,
            ),
            results_path,
        )

    def _write_update_records(
        self,
        *,
        run_dir: Path,
        packet: CodexWorkPacket,
        review: PatchReviewResult,
        exit_code: int,
        final_report: dict[str, Any],
        diff_path: Path,
        validation_results_path: Path | None = None,
        validation_ladder: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        memory_record = final_report.get("memory_record")
        if not isinstance(memory_record, dict):
            memory_record = {}
        framework_comparison = final_report.get("framework_comparison")
        if not isinstance(framework_comparison, dict):
            framework_comparison = {}
        prediction = final_report.get("prediction")
        if not isinstance(prediction, dict):
            prediction = {}
        component_delta = classify_component_delta(review.changed_files)
        ladder = validation_ladder or validation_ladder_for_changed_files(
            review.changed_files,
            repo_root=self.repo_root,
        )
        mission_selection = _mission_selection_summary(
            packet.mission_debug,
            final_report,
        )
        decision_input_summaries = _summary_decision_inputs(packet)
        record = {
            "packet_id": packet.packet_id,
            "created_at": packet.created_at.isoformat(),
            "exit_code": exit_code,
            "accepted": review.accepted,
            "review_reasons": review.reasons,
            "strategy_confidence": final_report.get("strategy_confidence", ""),
            "loophole_review": final_report.get("loophole_review", []),
            "loophole_fixes": final_report.get("loophole_fixes", []),
            "changed_files": review.changed_files,
            "diff_path": str(diff_path),
            "status": final_report.get("status", ""),
            "summary": final_report.get("summary", ""),
            "generalization": final_report.get("generalization", {}),
            "cross_round_evidence": final_report.get("cross_round_evidence", {}),
            "prediction": prediction,
            "component_delta": component_delta,
            "memory_record": memory_record,
            "implementation_scope": final_report.get("implementation_scope", {}),
            "framework_comparison": {
                "before_harness": packet.current_harness,
                "reported_before": framework_comparison.get("before", ""),
                "reported_after": framework_comparison.get("after", ""),
                "expected_effect": framework_comparison.get("expected_effect", ""),
                "rollback_trigger": framework_comparison.get("rollback_trigger", ""),
                "reviewed_delta_files": review.changed_files,
            },
            "leaderboard_compliance": final_report.get("leaderboard_compliance", {}),
            "external_research": final_report.get("external_research", {}),
            "prior_update_lessons": packet.prior_update_lessons,
            "prior_update_lesson_entries": packet.prior_update_lesson_entries,
            "update_history": packet.update_history,
            "update_decision_inputs": {
                "analysis_policy_coverage": decision_input_summaries[
                    "analysis_policy_coverage"
                ],
                "policy_recurrence_signals": packet.policy_recurrence_signals,
                "analysis_mechanism_update_classes": decision_input_summaries[
                    "analysis_mechanism_update_classes"
                ],
                "analysis_weakness_signatures": decision_input_summaries[
                    "analysis_weakness_signatures"
                ],
                "self_harness_candidates": decision_input_summaries[
                    "self_harness_improvement_queue"
                ],
                "self_harness_improvement_queue": packet.self_harness_improvement_queue,
                "infrastructure_triage": packet.infrastructure_triage,
                "change_evaluation_digest": packet.change_evaluation_digest,
                "rejected_update_buffer": packet.rejected_update_buffer,
                "runner_pivot_policy": packet.runner_pivot_policy,
                "mission_selection_contract": packet.mission_selection_contract,
                "mission_selection": mission_selection,
                "update_search_policy": packet.update_search_policy,
                "external_research_policy": packet.external_research_policy,
            },
            "campaign_context": {
                "recent_summaries": packet.campaign_context.get("recent_summaries", []),
                "recent_analysis_reports": packet.campaign_context.get(
                    "recent_analysis_reports",
                    [],
                ),
                "policy_recurrence_signals": packet.policy_recurrence_signals,
                "infrastructure_triage": packet.infrastructure_triage,
                "self_harness_improvement_queue": packet.self_harness_improvement_queue,
                "failure_pattern_digest": packet.failure_pattern_digest,
                "same_model_frontier": packet.same_model_frontier,
                "prior_update_lesson_entries": packet.prior_update_lesson_entries,
            },
            "validation_ladder": ladder,
            "loop_limit_contract": self._loop_limit_contract(),
            "change_manifest_path": str(run_dir / "change_manifest.json"),
            "validation_results_path": str(validation_results_path or ""),
        }
        record_path = run_dir / "update_record.json"
        summary_path = run_dir / "update_summary.md"
        manifest_path = run_dir / "change_manifest.json"
        record_path.write_text(json.dumps(record, indent=2))
        manifest_path.write_text(
            json.dumps(
                self._change_manifest(
                    packet=packet,
                    review=review,
                    final_report=final_report,
                    component_delta=component_delta,
                    validation_ladder=ladder,
                    mission_selection=mission_selection,
                ),
                indent=2,
            )
        )
        summary_path.write_text(
            self._update_summary_markdown(
                packet=packet,
                review=review,
                exit_code=exit_code,
                final_report=final_report,
                validation_ladder=ladder,
                mission_selection=mission_selection,
            )
        )
        return record_path, summary_path

    def _change_manifest(
        self,
        *,
        packet: CodexWorkPacket,
        review: PatchReviewResult,
        final_report: dict[str, Any],
        component_delta: dict[str, Any],
        validation_ladder: dict[str, Any],
        mission_selection: dict[str, Any],
    ) -> dict[str, Any]:
        prediction = final_report.get("prediction")
        if not isinstance(prediction, dict):
            prediction = {}
        return {
            "schema_version": 1,
            "packet_id": packet.packet_id,
            "created_at": datetime_now_iso(),
            "accepted": review.accepted,
            "status": final_report.get("status", ""),
            "summary": final_report.get("summary", ""),
            "strategy_confidence": final_report.get("strategy_confidence", ""),
            "loophole_review": final_report.get("loophole_review", []),
            "loophole_fixes": final_report.get("loophole_fixes", []),
            "external_research": final_report.get("external_research", {}),
            "failure_evidence": {
                "failing_tasks": packet.failing_tasks,
                "failure_pattern_digest": packet.failure_pattern_digest,
                "same_model_frontier": packet.same_model_frontier,
                "self_harness_improvement_queue": packet.self_harness_improvement_queue,
            },
            "root_cause": {
                "generalization": final_report.get("generalization", {}),
                "cross_round_evidence": final_report.get("cross_round_evidence", {}),
                "recent_analysis_reports": packet.campaign_context.get(
                    "recent_analysis_reports",
                    [],
                ),
                "policy_recurrence_signals": packet.policy_recurrence_signals,
                "infrastructure_triage": packet.infrastructure_triage,
                "self_harness_improvement_queue": packet.self_harness_improvement_queue,
                "prior_update_lesson_entries": packet.prior_update_lesson_entries,
                "mission_selection": mission_selection,
                "change_evaluation_digest": packet.change_evaluation_digest,
                "rejected_update_buffer": packet.rejected_update_buffer,
                "runner_pivot_policy": packet.runner_pivot_policy,
                "external_research_policy": packet.external_research_policy,
            },
            "memory_record": final_report.get("memory_record", {}),
            "targeted_fix": {
                "changed_files": review.changed_files,
                "implementation_scope": final_report.get("implementation_scope", {}),
                "component_delta": component_delta,
            },
            "prediction": prediction,
            "validation_ladder": validation_ladder,
            "loop_limit_contract": self._loop_limit_contract(),
            "review": {
                "accepted": review.accepted,
                "reasons": review.reasons,
                "changed_files": review.changed_files,
            },
        }

    def _loop_limit_contract(self) -> dict[str, Any]:
        contract = base_loop_limit_contract(
            notes={
                "all_loops": (
                    "Master, Codex update sub-agent, diagnostic/context "
                    "sub-agent, validation/regression sub-agent, mission-debug "
                    "sub-agent, and Worker task loops must not stop because "
                    "time, round, iteration, token, budget, timeout, cooldown, "
                    "K/attempt, max_turns, depth, feature-count, target-task-count, "
                    "validation-contract-count, or context-window values were reached. "
                    "Those values are audit, scheduling, packet-size, recovery, "
                    "or single-operation controls only."
                ),
                "master_loop": (
                    "Codex update artifacts may cite master-loop iteration, task, "
                    "wall-clock, patience, budget, provider, and retry metadata, "
                    "but none of those values stop the master, sub-agent, or "
                    "Worker loops."
                ),
                "validation_regression_sub_agents": (
                    "Validation and regression lanes, snapshot counts, explicit "
                    "selection caps, retry counts, transient cooldowns, host "
                    "validation timeout values, project-test duration, and "
                    "task-concurrency values are audit, selection, retry, or "
                    "throughput metadata only. They may reject or roll back a "
                    "patch on real validation failure, but must not stop master, "
                    "Codex update, diagnostic/context, mission-debug, Worker, or "
                    "validation/regression sub-agent loops because a time, round, "
                    "attempt, snapshot-count, or timeout value was reached."
                ),
            }
        )
        contract["master_loop"].update(
            {
                "infra_retry_unbounded_by_attempt_count": True,
                "allowed_terminal_reasons": [
                    "explicit user marks campaign goal complete/stopped",
                    "explicit one-shot submit terminal action after submit gate handling",
                    "hard validation/regression process error unrelated to loop-limit fields",
                    "explicit user/process cancellation",
                ],
                "disallowed_limit_terminal_reasons": [
                    *disallowed_limit_terminal_reasons(),
                ],
            }
        )
        contract["codex_update_sub_agent"].update(
            {
                "timeout_seconds_audit_only": self.timeout_seconds,
                "timeout_seconds_reference_audit_only": self.timeout_seconds,
                "timeout_seconds": self.timeout_seconds,
                "timeout_seconds_legacy_compatibility_only": self.timeout_seconds,
                "host_validation_timeout_seconds_audit_only": self.validation_timeout_seconds,
                "host_validation_timeout_seconds_reference_audit_only": self.validation_timeout_seconds,
                "host_validation_timeout_seconds": self.validation_timeout_seconds,
                "host_validation_timeout_seconds_legacy_compatibility_only": self.validation_timeout_seconds,
            }
        )
        contract["note"] = (
            "Codex update timeout, host validation timeout, interval, cooldown, "
            "minimum-failure, diagnostic-k, mission-feature, provider-transient, "
            "token, and wall-time fields are audit/progress metadata only; "
            "they must not stop master, Codex update sub-agent, diagnostic/context "
            "sub-agent, or Worker loops."
        )
        return contract

    def _update_summary_markdown(
        self,
        *,
        packet: CodexWorkPacket,
        review: PatchReviewResult,
        exit_code: int,
        final_report: dict[str, Any],
        validation_ladder: dict[str, Any],
        mission_selection: dict[str, Any],
    ) -> str:
        memory_record = final_report.get("memory_record")
        if not isinstance(memory_record, dict):
            memory_record = {}
        framework = final_report.get("framework_comparison")
        if not isinstance(framework, dict):
            framework = {}
        implementation_scope = final_report.get("implementation_scope")
        if not isinstance(implementation_scope, dict):
            implementation_scope = {}
        generalization = final_report.get("generalization")
        if not isinstance(generalization, dict):
            generalization = {}
        cross_round = final_report.get("cross_round_evidence")
        if not isinstance(cross_round, dict):
            cross_round = {}
        prediction = final_report.get("prediction")
        if not isinstance(prediction, dict):
            prediction = {}
        if not isinstance(validation_ladder, dict):
            validation_ladder = {}
        reported_validation_commands = final_report.get("validation_commands")
        if not isinstance(reported_validation_commands, list):
            reported_validation_commands = []
        external_research = final_report.get("external_research")
        if not isinstance(external_research, dict):
            external_research = {}
        research_policy = packet.external_research_policy
        if not isinstance(research_policy, dict):
            research_policy = {}
        loophole_review = final_report.get("loophole_review")
        if not isinstance(loophole_review, list):
            loophole_review = []
        loophole_fixes = final_report.get("loophole_fixes")
        if not isinstance(loophole_fixes, list):
            loophole_fixes = []
        decision_inputs = _summary_decision_inputs(packet)
        concise = str(memory_record.get("concise") or final_report.get("summary") or "").strip()
        detailed = str(memory_record.get("detailed") or "").strip()
        return "\n".join(
            [
                f"# Codex Update {packet.packet_id}",
                "",
                f"- Accepted: {str(review.accepted).lower()}",
                f"- Exit code: {exit_code}",
                "- Changed files: " + (", ".join(review.changed_files) or "none"),
                "- Reasons: " + ("; ".join(review.reasons) or "none"),
                "",
                "## Concise",
                concise or "No concise record supplied.",
                "",
                "## Detailed",
                detailed or "No detailed record supplied.",
                "",
                "## Loophole Review",
                "- Strategy confidence: "
                + str(final_report.get("strategy_confidence") or "not supplied"),
                "- Reviewed risks: "
                + ("; ".join(str(item) for item in loophole_review) or "none"),
                "- Mitigations: "
                + ("; ".join(str(item) for item in loophole_fixes) or "none"),
                "",
                "## Generalization",
                str(generalization.get("problem_class") or "").strip()
                or "No problem class supplied.",
                "",
                "## Cross-Round Evidence",
                "- Used: " + str(cross_round.get("used") or False).lower(),
                "- Summaries: "
                + (
                    ", ".join(cross_round.get("recent_summary_ids") or [])
                    or "none"
                ),
                "- Dominant patterns: "
                + (
                    ", ".join(cross_round.get("dominant_patterns") or [])
                    or "none"
                ),
                "- Selected problem class: "
                + str(cross_round.get("selected_problem_class") or "not supplied"),
                "",
                "## Prediction",
                "- Expected fixed task classes: "
                + (
                    ", ".join(
                        str(item)
                        for item in prediction.get("expected_fixed_task_classes") or []
                    )
                    or "none"
                ),
                "- Risk task classes: "
                + (
                    ", ".join(
                        str(item) for item in prediction.get("risk_task_classes") or []
                    )
                    or "none"
                ),
                "- Expected metric delta: "
                + str(prediction.get("expected_metric_delta") or "not supplied"),
                "- Confidence: " + str(prediction.get("confidence") or "not supplied"),
                "- Falsification window: "
                + str(prediction.get("falsification_window") or "not supplied"),
                "",
                "## Update Decision Inputs",
                "- Analysis candidate classes: "
                + decision_inputs["analysis_candidate_classes"],
                "- Analysis failure buckets: "
                + decision_inputs["analysis_failure_buckets"],
                "- Analysis mechanism update classes: "
                + decision_inputs["analysis_mechanism_update_classes"],
                "- Analysis policy coverage: "
                + decision_inputs["analysis_policy_coverage"],
                "- Policy recurrence signals: "
                + decision_inputs["policy_recurrence_signals"],
                "- Infrastructure triage: "
                + decision_inputs["infrastructure_triage"],
                "- Analysis trajectory evidence: "
                + decision_inputs["analysis_trajectory_evidence"],
                "- Analysis weakness signatures: "
                + decision_inputs["analysis_weakness_signatures"],
                "- Self-Harness candidates: "
                + decision_inputs["self_harness_improvement_queue"],
                "- Change evaluation miss classes: "
                + decision_inputs["change_evaluation_miss_classes"],
                "- Change evaluation risk classes: "
                + decision_inputs["change_evaluation_risk_classes"],
                "- Rejected update packets: "
                + decision_inputs["rejected_update_packets"],
                "- Rejected required mutations: "
                + decision_inputs["rejected_required_mutations"],
                "- Prior update lessons: "
                + decision_inputs["prior_update_lessons"],
                "- Pivot discouraged: " + decision_inputs["pivot_discouraged"],
                "- Pivot layer pressure: " + decision_inputs["pivot_layer_pressure"],
                "- Pivot supported: " + decision_inputs["pivot_supported"],
                "- Search candidate rules: "
                + decision_inputs["search_candidate_rules"],
                "- Mission candidate selected: "
                + str(mission_selection.get("selected_candidate_id") or "none"),
                "- Mission selection enforced: "
                + str(mission_selection.get("enforced") or False).lower(),
                "",
                "## Validation Ladder",
                "- Reported commands: "
                + (
                    ", ".join(str(item) for item in reported_validation_commands)
                    or "none"
                ),
                "- Commands: "
                + (
                    ", ".join(
                        str(item) for item in validation_ladder.get("commands") or []
                    )
                    or "none"
                ),
                "- Policy: "
                + str(validation_ladder.get("policy") or "not supplied"),
                "",
                "## Implementation Scope",
                "- Primary layer: "
                + str(implementation_scope.get("primary_layer") or "not supplied"),
                "- Architectural change considered: "
                + str(implementation_scope.get("architectural_change_considered") or False).lower(),
                "- Structural files changed: "
                + (
                    ", ".join(implementation_scope.get("structural_files_changed") or [])
                    or "none"
                ),
                "- Prompt-only justification: "
                + str(
                    implementation_scope.get("why_prompt_only_is_sufficient")
                    or "not applicable"
                ),
                "",
                "## Framework Comparison",
                "- Before: " + str(framework.get("before") or "not supplied"),
                "- After: " + str(framework.get("after") or "not supplied"),
                "- Expected effect: " + str(framework.get("expected_effect") or "not supplied"),
                "- Rollback trigger: " + str(framework.get("rollback_trigger") or "not supplied"),
                "",
                "## Memory Directions",
                "- Failed directions to avoid: "
                + (
                    "; ".join(
                        str(item)
                        for item in memory_record.get("failed_directions_to_avoid") or []
                    )
                    or "none"
                ),
                "- Supported directions to preserve: "
                + (
                    "; ".join(
                        str(item)
                        for item in memory_record.get("supported_directions_to_preserve")
                        or []
                    )
                    or "none"
                ),
                "",
                "## External Research",
                "- Used: " + str(external_research.get("used") or False).lower(),
                "- Sources: "
                + (
                    ", ".join(str(item) for item in external_research.get("sources") or [])
                    or "none"
                ),
                "- Reason: " + str(external_research.get("reason") or "not supplied"),
                "- Impact: " + str(external_research.get("impact") or "not supplied"),
                "- Policy status: " + str(research_policy.get("status") or "not supplied"),
                "- Policy focus areas: "
                + (
                    "; ".join(
                        str(item)
                        for item in research_policy.get("research_focus_areas") or []
                    )
                    or "none"
                ),
                "- Fetch requirements: "
                + _summary_fetch_requirements(
                    research_policy.get("fetch_requirements")
                ),
                "",
            ]
        )

    def _contract_report_defaults(
        self,
        *,
        status: str,
        summary: str,
        skipped_validation_reason: str,
        memory_concise: str,
        memory_detailed: str,
        loophole_review: list[str] | None = None,
        loophole_fixes: list[str] | None = None,
        failed_directions_to_avoid: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "summary": summary,
            "changed_files": [],
            "validation_commands": [],
            "skipped_validation_reason": skipped_validation_reason,
            "strategy_confidence": "low",
            "loophole_review": loophole_review or [],
            "loophole_fixes": loophole_fixes or [],
            "component_type": "other",
            "implementation_scope": {
                "primary_layer": "other",
                "architectural_change_considered": True,
                "structural_files_changed": [],
                "why_prompt_only_is_sufficient": "No prompt-only patch was produced.",
            },
            "generalization": {
                "problem_class": "not evaluated",
                "applies_to": [],
                "anti_overfit_checks": [],
                "why_not_task_specific": "Codex did not produce a patch.",
            },
            "cross_round_evidence": {
                "used": False,
                "recent_summary_ids": [],
                "dominant_patterns": [],
                "selected_problem_class": "not evaluated",
                "why_this_slice_generalizes": "Codex did not produce a patch.",
            },
            "memory_record": {
                "concise": memory_concise,
                "detailed": memory_detailed,
                "failed_directions_to_avoid": failed_directions_to_avoid or [],
                "supported_directions_to_preserve": [],
            },
            "framework_comparison": {
                "before": "current harness snapshot in packet",
                "after": "unchanged",
                "expected_effect": "none",
                "rollback_trigger": "not applicable",
            },
            "prediction": {
                "expected_fixed_task_classes": [],
                "risk_task_classes": [],
                "expected_metric_delta": 0.0,
                "confidence": "low",
                "falsification_window": "next comparable campaign summary",
            },
            "leaderboard_compliance": {
                "harbor_official_harness_preserved": True,
                "self_owned_worker_preserved": True,
                "benchmark_integrity_preserved": True,
                "timeouts_resources_unchanged": True,
                "submit_gate_preserved": True,
                "official_dataset_preserved": True,
                "five_attempts_per_task_preserved": True,
                "no_prohibited_terminal_bench_access": True,
                "upload_artifacts_trace_preserved": True,
            },
            "external_research": {
                "used": False,
                "sources": [],
                "reason": skipped_validation_reason or "not needed",
                "impact": "",
            },
        }

    def _load_final_report(self, final_path: Path) -> dict[str, Any]:
        if not final_path.exists():
            return {}
        text = final_path.read_text().strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"status": "unparsed", "summary": text[:4000]}

    def _apply_report_gates(
        self,
        review: PatchReviewResult,
        *,
        exit_code: int,
        final_report: dict[str, Any],
        required_validation_commands: list[str],
        exec_output_text: str = "",
        external_research_recommended: bool = False,
        external_research_policy: dict[str, Any] | None = None,
        host_validation_commands: list[str] | None = None,
        rejected_update_buffer: list[dict[str, Any]] | None = None,
        runner_pivot_policy: dict[str, Any] | None = None,
        change_evaluation_digest: dict[str, Any] | None = None,
        prior_update_lesson_entries: list[dict[str, Any]] | None = None,
        failure_pattern_digest: dict[str, Any] | None = None,
        mission_debug: dict[str, Any] | None = None,
        ignore_files: list[str] | None = None,
    ) -> PatchReviewResult:
        findings = report_contract.ViolationCollector(review.violations)
        changed_files = list(review.changed_files)
        context = report_contract.ReportValidationContext(
            changed_files=changed_files,
            ignore_files=set(ignore_files or []),
            required_validation_commands=list(required_validation_commands),
            host_validation_commands=list(host_validation_commands or []),
            rejected_update_buffer=list(rejected_update_buffer or []),
            runner_pivot_policy=dict(runner_pivot_policy or {}),
            change_evaluation_digest=dict(change_evaluation_digest or {}),
            prior_update_lesson_entries=list(prior_update_lesson_entries or []),
            failure_pattern_digest=dict(failure_pattern_digest or {}),
            mission_debug=dict(mission_debug or {}),
            external_research_recommended=external_research_recommended,
            external_research_policy=dict(external_research_policy or {}),
        )

        if exit_code != 0:
            findings.add("codex.exec_failed", f"codex exec exited with {exit_code}")
            failure_reason = self._codex_exec_failure_reason(exec_output_text)
            if failure_reason:
                findings.add("codex.provider_failure", failure_reason)
        nested_sub_agent_hits = self._codex_nested_sub_agent_command_hits(
            exec_output_text
        )
        if nested_sub_agent_hits:
            findings.add(
                "codex.nested_agent",
                "Codex update sub-agent attempted to create a nested sub-agent or "
                "invoke an external coding-agent process: "
                + "; ".join(nested_sub_agent_hits)
            )
        self._validate_registered_report_rules(final_report, context, findings)

        return PatchReviewResult(
            accepted=not findings.blocking,
            changed_files=changed_files,
            violations=findings.violations,
        )

    def _report_validator_bindings(self) -> dict[str, Any]:
        """Map registry bindings to their executable validator path."""

        return {
            "report_presence": self._validate_report_presence,
            "report_status": self._validate_report_status,
            "report_status_changed_files": self._validate_report_status_changed_files,
            "basic_report": self._apply_basic_report_gates,
            "changed_files": self._apply_changed_files_report_gate,
            "validation_shape": self._validate_report_validation_shape,
            "required_validation": self._validate_required_validation,
            "loophole_review": self._apply_loophole_report_gates,
            "generalization_structure": self._validate_generalization_structure,
            "generalization_evidence": self._validate_generalization_evidence,
            "cross_round_structure": self._validate_cross_round_structure,
            "cross_round_patterns": self._validate_cross_round_patterns,
            "cross_round_problem_class": self._validate_cross_round_problem_class,
            "memory_structure": self._validate_memory_structure,
            "memory_failed_directions": self._validate_memory_failed_directions,
            "memory_supported_directions": self._validate_memory_supported_directions,
            "framework_comparison": self._apply_framework_comparison_gates,
            "prediction_structure": self._validate_prediction_structure,
            "prediction_window": self._validate_prediction_window,
            "change_evaluation_misses": self._validate_change_evaluation_misses,
            "change_evaluation_risks": self._validate_change_evaluation_risks,
            "prediction_evidence": self._apply_prediction_evidence_binding_gates,
            "mission_selection": self._validate_mission_selection,
            "mission_scope": self._validate_mission_scope,
            "implementation_scope": self._validate_implementation_scope,
            "implementation_layer": self._validate_implementation_layer,
            "leaderboard_compliance": self._apply_leaderboard_compliance_gates,
            "external_research": self._apply_external_research_gates,
        }

    def _validate_registered_report_rules(
        self,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        bindings = self._report_validator_bindings()
        for rule in report_contract.final_report_rules():
            validator = bindings.get(rule.binding)
            if validator is None:
                findings.add(
                    report_contract.INTERNAL_RULE_ID,
                    f"missing validator binding {rule.binding!r} for rule {rule.id}",
                )
                continue
            validator(rule, final_report, context, findings)

    def validate_report_rule(
        self,
        rule_id: str,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
    ) -> list[report_contract.ReportViolation]:
        """Execute one registered rule, used by consistency fixtures."""

        rule = report_contract.rule_for_id(rule_id)
        findings = report_contract.ViolationCollector()
        if rule is None or rule.scope != report_contract.FINAL_REPORT_SCOPE:
            findings.add(
                report_contract.INTERNAL_RULE_ID,
                f"unknown final-report rule {rule_id!r}",
            )
            return findings.violations
        validator = self._report_validator_bindings().get(rule.binding)
        if validator is None:
            findings.add(
                report_contract.INTERNAL_RULE_ID,
                f"missing validator binding {rule.binding!r} for rule {rule.id}",
            )
            return findings.violations
        validator(rule, final_report, context, findings)
        return findings.violations

    def _missing_required_validation_commands(
        self,
        required_validation_commands: list[str],
        validation_commands: list[Any],
    ) -> list[str]:
        reported = [
            _canonical_validation_command(str(command))
            for command in validation_commands
            if str(command).strip()
        ]
        missing: list[str] = []
        for required in required_validation_commands:
            canonical_required = _canonical_validation_command(required)
            if not any(
                _validation_command_satisfies(candidate, canonical_required)
                for candidate in reported
            ):
                missing.append(required)
        return missing

    def _codex_nested_sub_agent_command_hits(self, events_text: str) -> list[str]:
        hits: list[str] = []
        for command in _codex_event_command_strings(events_text):
            reason = external_agent_command_reason(command)
            if not reason:
                continue
            command_summary = _truncate_summary_text(" ".join(command.split()), 180)
            hits.append(f"{command_summary} ({reason})")
        return list(dict.fromkeys(hits))[:8]

    def _apply_changed_files_report_gate(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not final_report:
            return
        reported_changed_files = final_report.get("changed_files")
        if not isinstance(reported_changed_files, list):
            findings.add(rule.id, "Codex final report changed_files must be a list")
            return
        reported_paths = [str(path) for path in reported_changed_files]
        missing = [path for path in context.changed_files if path not in reported_paths]
        if missing:
            findings.add(
                rule.id,
                "Codex final report changed_files missing changed files: "
                + ", ".join(missing)
            )
        # Files that were already dirty in the baseline worktree (unrelated local
        # edits, untracked scratch files) are excluded from the isolated Codex
        # delta, so a report that still lists them must not be flagged as
        # over-reporting -- Codex legitimately saw them in git status.
        overreported = [
            path
            for path in reported_paths
            if path not in context.changed_files and path not in context.ignore_files
        ]
        if overreported:
            findings.add(
                rule.id,
                "Codex final report changed_files includes files not changed by "
                "the diff: " + ", ".join(overreported)
            )

    def _apply_basic_report_gates(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        _ = context
        if not final_report:
            return
        if not str(final_report.get("summary") or "").strip():
            findings.add(rule.id, "Codex final report summary is required")
        if "skipped_validation_reason" in final_report and not isinstance(
            final_report.get("skipped_validation_reason"),
            str,
        ):
            findings.add(
                rule.id,
                "Codex final report skipped_validation_reason must be a string",
            )

    def _apply_loophole_report_gates(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        if final_report.get("strategy_confidence") not in {"high", "medium", "low"}:
            findings.add(rule.id, "strategy_confidence must be high, medium, or low")
        loophole_review = final_report.get("loophole_review")
        if not isinstance(loophole_review, list) or not [
            item for item in loophole_review if str(item).strip()
        ]:
            findings.add(rule.id, "loophole_review must list at least one reviewed risk")
        loophole_fixes = final_report.get("loophole_fixes")
        if not isinstance(loophole_fixes, list) or not [
            item for item in loophole_fixes if str(item).strip()
        ]:
            findings.add(rule.id, "loophole_fixes must list at least one mitigation")

    def _validate_report_presence(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        _ = context
        if not final_report:
            findings.add(rule.id, "missing Codex final report")

    def _validate_report_status(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        _ = context
        if not final_report:
            return
        status = final_report.get("status")
        if status not in {"edited", "noop", "rejected"}:
            findings.add(rule.id, f"invalid Codex final report status: {status!r}")

    def _validate_report_status_changed_files(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if final_report and context.changed_files and final_report.get("status") != "edited":
            findings.add(
                rule.id,
                "files changed but Codex final report status is not edited",
            )

    def _validate_report_validation_shape(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not final_report:
            return
        validation_commands = final_report.get("validation_commands")
        if not isinstance(validation_commands, list):
            findings.add(rule.id, "Codex final report validation_commands must be a list")
            validation_commands = []
        skipped_reason = str(final_report.get("skipped_validation_reason") or "").strip()
        if (
            context.changed_files
            and not validation_commands
            and not skipped_reason
            and not context.host_validation_commands
        ):
            findings.add(rule.id, "validation commands were skipped without explanation")

    def _validate_required_validation(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not final_report or not context.changed_files:
            return
        validation_commands = final_report.get("validation_commands")
        reported = validation_commands if isinstance(validation_commands, list) else []
        skipped_reason = str(final_report.get("skipped_validation_reason") or "").strip()
        missing_required = self._missing_required_validation_commands(
            context.required_validation_commands,
            reported,
        )
        missing_from_host = self._missing_required_validation_commands(
            missing_required,
            context.host_validation_commands,
        )
        if missing_from_host and not skipped_reason:
            findings.add(
                rule.id,
                "required validation commands missing: " + ", ".join(missing_from_host),
            )

    def _codex_exec_failure_reason(self, output_text: str) -> str:
        lowered = output_text.lower()
        details: list[str] = []
        if "auth_unavailable" in lowered or "no auth available" in lowered:
            details.append("auth_unavailable")
        if "401 unauthorized" in lowered or '"unauthorized"' in lowered:
            details.append("401 Unauthorized")
        if "403 forbidden" in lowered:
            details.append("403 Forbidden")
        if "429" in lowered or "rate limit" in lowered or "too many requests" in lowered:
            details.append("429 Too Many Requests")
        if "500 internal server error" in lowered:
            details.append("500 Internal Server Error")
        if "502 bad gateway" in lowered:
            details.append("502 Bad Gateway")
        if "503 service unavailable" in lowered:
            details.append("503 Service Unavailable")
        if "504 gateway timeout" in lowered:
            details.append("504 Gateway Timeout")
        if not details:
            return ""
        return "codex exec events indicate upstream provider/auth failure: " + ", ".join(
            dict.fromkeys(details)
        )

    def _validate_generalization_structure(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        generalization = final_report.get("generalization")
        if not isinstance(generalization, dict):
            findings.add(rule.id, "missing generalization report")
            return
        if not str(generalization.get("problem_class") or "").strip():
            findings.add(rule.id, "generalization.problem_class is required")
        applies_to = generalization.get("applies_to")
        if not isinstance(applies_to, list) or not [
            item for item in applies_to if str(item).strip()
        ]:
            findings.add(
                rule.id,
                "generalization.applies_to must list reusable task classes",
            )
        checks = generalization.get("anti_overfit_checks")
        if not isinstance(checks, list) or not [
            item for item in checks if str(item).strip()
        ]:
            findings.add(
                rule.id,
                "generalization.anti_overfit_checks must not be empty",
            )
        if not str(generalization.get("why_not_task_specific") or "").strip():
            findings.add(rule.id, "generalization.why_not_task_specific is required")

    def _validate_generalization_evidence(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        generalization = final_report.get("generalization")
        if not isinstance(generalization, dict):
            return
        applies_to = generalization.get("applies_to")
        digest_markers = _failure_pattern_markers(context.failure_pattern_digest)
        pattern_markers = [
            *digest_markers,
            *_mission_candidate_mechanism_markers(context.mission_debug),
        ]
        if digest_markers and not _text_entries_cover_any_marker(
            [generalization.get("problem_class") or "", *(applies_to or [])]
            if isinstance(applies_to, list)
            else [generalization.get("problem_class") or ""],
            pattern_markers,
        ):
            findings.add(
                rule.id,
                "generalization.problem_class or applies_to must reference a concrete "
                "failure_pattern_digest label"
            )

    def _validate_cross_round_structure(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        evidence = final_report.get("cross_round_evidence")
        if not isinstance(evidence, dict):
            findings.add(rule.id, "missing cross_round_evidence")
            return
        if evidence.get("used") is not True:
            findings.add(
                rule.id,
                "cross_round_evidence.used must be true for edited patches",
            )
        summaries = evidence.get("recent_summary_ids")
        if not isinstance(summaries, list) or not [
            item for item in summaries if str(item).strip()
        ]:
            findings.add(
                rule.id,
                "cross_round_evidence.recent_summary_ids must not be empty",
            )
        patterns = evidence.get("dominant_patterns")
        if not isinstance(patterns, list) or not [
            item for item in patterns if str(item).strip()
        ]:
            findings.add(
                rule.id,
                "cross_round_evidence.dominant_patterns must not be empty",
            )
        if not str(evidence.get("selected_problem_class") or "").strip():
            findings.add(
                rule.id,
                "cross_round_evidence.selected_problem_class is required",
            )
        if not str(evidence.get("why_this_slice_generalizes") or "").strip():
            findings.add(
                rule.id,
                "cross_round_evidence.why_this_slice_generalizes is required",
            )

    def _cross_round_marker_state(
        self,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
    ) -> tuple[list[str], list[str], bool]:
        evidence = final_report.get("cross_round_evidence")
        if not isinstance(evidence, dict):
            return [], [], False
        patterns = evidence.get("dominant_patterns")
        pattern_markers = _failure_pattern_markers(context.failure_pattern_digest)
        report_pattern_markers = (
            [str(item) for item in patterns if str(item).strip()]
            if isinstance(patterns, list)
            else []
        )
        covers = bool(pattern_markers) and _text_entries_cover_any_marker(
            report_pattern_markers, pattern_markers
        )
        return pattern_markers, report_pattern_markers, covers

    def _validate_cross_round_patterns(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        pattern_markers, _report_markers, covers = self._cross_round_marker_state(
            final_report, context
        )
        if pattern_markers and not covers:
            findings.add(
                rule.id,
                "cross_round_evidence.dominant_patterns must reference a concrete "
                "failure_pattern_digest label"
            )

    def _validate_cross_round_problem_class(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        evidence = final_report.get("cross_round_evidence")
        if not isinstance(evidence, dict):
            return
        pattern_markers, report_pattern_markers, covers = self._cross_round_marker_state(
            final_report, context
        )
        selected_problem_class = str(
            evidence.get("selected_problem_class") or ""
        ).strip()
        # A concrete mechanism signature is a *more* operable problem class than a
        # generic digest label. Accept coverage from the digest, a mission feature
        # candidate's mechanism, or the report's own dominant_patterns -- but only
        # trust self-declared patterns once they themselves cover the digest, so a
        # report cannot launder an unrelated label through junk dominant_patterns.
        selected_class_markers = [
            *pattern_markers,
            *_mission_candidate_mechanism_markers(context.mission_debug),
        ]
        if covers:
            selected_class_markers.extend(report_pattern_markers)
        if (
            pattern_markers
            and selected_problem_class
            and not _text_entries_cover_any_marker(
                [selected_problem_class],
                selected_class_markers,
            )
        ):
            findings.add(
                rule.id,
                "cross_round_evidence.selected_problem_class must reference a concrete "
                "failure_pattern_digest label"
            )

    def _validate_memory_structure(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        memory_record = final_report.get("memory_record")
        if not isinstance(memory_record, dict):
            findings.add(rule.id, "missing memory_record")
            return
        if not str(memory_record.get("concise") or "").strip():
            findings.add(rule.id, "memory_record.concise is required")
        if not str(memory_record.get("detailed") or "").strip():
            findings.add(rule.id, "memory_record.detailed is required")
        failed_directions = memory_record.get("failed_directions_to_avoid")
        if not isinstance(failed_directions, list):
            findings.add(
                rule.id,
                "memory_record.failed_directions_to_avoid must be a list",
            )
        supported_directions = memory_record.get("supported_directions_to_preserve")
        if not isinstance(supported_directions, list):
            findings.add(
                rule.id,
                "memory_record.supported_directions_to_preserve must be a list"
            )

    def _validate_memory_failed_directions(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        memory_record = final_report.get("memory_record")
        if not isinstance(memory_record, dict):
            return
        failed_directions = memory_record.get("failed_directions_to_avoid")
        if not isinstance(failed_directions, list):
            return
        actionable_rejected_buffer = [
            entry
            for entry in context.rejected_update_buffer
            if not _rejected_buffer_entry_is_non_actionable(entry)
        ]
        if actionable_rejected_buffer and not [
            item for item in failed_directions if str(item).strip()
        ]:
            findings.add(
                rule.id,
                "memory_record.failed_directions_to_avoid must record at least one "
                "rejected or rolled-back direction when rejected_update_buffer is present"
            )
        elif actionable_rejected_buffer and not _failed_directions_cover_rejected_buffer(
            failed_directions,
            actionable_rejected_buffer,
        ):
            findings.add(
                rule.id,
                "memory_record.failed_directions_to_avoid must reference a concrete "
                "mission_candidate_id, packet_id, failure_class, or component_layer "
                "for every rejected_update_buffer entry"
            )
        elif actionable_rejected_buffer and not _failed_directions_cover_loophole_records(
            failed_directions,
            actionable_rejected_buffer,
        ):
            findings.add(
                rule.id,
                "memory_record.failed_directions_to_avoid must reference prior "
                "loophole_review or loophole_fixes evidence for rejected_update_buffer "
                "entries that provide it"
            )
        elif actionable_rejected_buffer and not _failed_directions_cover_required_mutations(
            failed_directions,
            actionable_rejected_buffer,
        ):
            findings.add(
                rule.id,
                "memory_record.failed_directions_to_avoid must reference the "
                "required_mutation guidance for rejected_update_buffer entries "
                "that provide it"
            )
        discouraged_markers = _discouraged_direction_memory_entries(
            context.runner_pivot_policy
        )
        if discouraged_markers and not _failed_directions_cover_rejected_buffer(
            failed_directions,
            discouraged_markers,
        ):
            findings.add(
                rule.id,
                "memory_record.failed_directions_to_avoid must reference each "
                "runner_pivot_policy.discouraged failure_class or component_layer"
            )
        layer_pressure_markers = _layer_pressure_memory_entries(
            context.runner_pivot_policy
        )
        if layer_pressure_markers and not _failed_directions_cover_layer_pressure(
            failed_directions,
            layer_pressure_markers,
        ):
            findings.add(
                rule.id,
                "memory_record.failed_directions_to_avoid must reference each "
                "runner_pivot_policy.layer_pressure component_layer plus a recent "
                "packet_id or failure_class when available"
            )

        prior_lesson_markers = _prior_update_lesson_memory_entries(
            context.prior_update_lesson_entries
        )
        if prior_lesson_markers and not _failed_directions_cover_rejected_buffer(
            failed_directions,
            prior_lesson_markers,
        ):
            findings.add(
                rule.id,
                "memory_record.failed_directions_to_avoid must reference each "
                "prior_update_lesson_entries packet_id, outcome, or mission_candidate_id",
            )

    def _validate_memory_supported_directions(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        memory_record = final_report.get("memory_record")
        if not isinstance(memory_record, dict):
            return
        supported_directions = memory_record.get("supported_directions_to_preserve")
        if not isinstance(supported_directions, list):
            return
        supported_markers = _supported_direction_memory_entries(
            context.runner_pivot_policy
        )
        if supported_markers and not _failed_directions_cover_rejected_buffer(
            supported_directions,
            supported_markers,
        ):
            findings.add(
                rule.id,
                "memory_record.supported_directions_to_preserve must reference each "
                "runner_pivot_policy.supported packet_id, failure_class, or component_layer"
            )

    def _apply_framework_comparison_gates(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        comparison = final_report.get("framework_comparison")
        if not isinstance(comparison, dict):
            findings.add(rule.id, "missing framework_comparison")
            return
        for field_name in ("before", "after", "expected_effect", "rollback_trigger"):
            if not str(comparison.get(field_name) or "").strip():
                findings.add(
                    rule.id,
                    f"framework_comparison.{field_name} is required",
                )

    def _validate_prediction_structure(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        prediction = final_report.get("prediction")
        if not isinstance(prediction, dict):
            findings.add(rule.id, "missing prediction")
            return
        fixed = prediction.get("expected_fixed_task_classes")
        if not isinstance(fixed, list):
            findings.add(
                rule.id,
                "prediction.expected_fixed_task_classes must be a list",
            )
        elif not [item for item in fixed if str(item).strip()]:
            findings.add(
                rule.id,
                "prediction.expected_fixed_task_classes must not be empty",
            )
        risks = prediction.get("risk_task_classes")
        if not isinstance(risks, list):
            findings.add(rule.id, "prediction.risk_task_classes must be a list")
        try:
            float(prediction.get("expected_metric_delta"))
        except (TypeError, ValueError):
            findings.add(rule.id, "prediction.expected_metric_delta must be numeric")
        if prediction.get("confidence") not in {"high", "medium", "low"}:
            findings.add(
                rule.id,
                "prediction.confidence must be high, medium, or low",
            )
        falsification_window = str(prediction.get("falsification_window") or "").strip()
        if not falsification_window:
            findings.add(rule.id, "prediction.falsification_window is required")

    def _validate_prediction_window(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        prediction = final_report.get("prediction")
        if not isinstance(prediction, dict):
            return
        falsification_window = str(prediction.get("falsification_window") or "").strip()
        if falsification_window and not _prediction_window_is_evaluable(falsification_window):
            findings.add(
                rule.id,
                "prediction.falsification_window must name an evaluable next summary, "
                "frontier, regression, or rerun window"
            )

    def _validate_change_evaluation_misses(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not context.change_evaluation_digest:
            return
        memory_record = final_report.get("memory_record")
        if not isinstance(memory_record, dict):
            return
        failed_directions = memory_record.get("failed_directions_to_avoid")
        miss_classes = _change_evaluation_class_markers(
            context.change_evaluation_digest.get("miss_classes"),
            limit=3,
        )
        if miss_classes and not _text_entries_cover_markers(
            failed_directions if isinstance(failed_directions, list) else [],
            miss_classes,
        ):
            findings.add(
                rule.id,
                "memory_record.failed_directions_to_avoid must reference top "
                "change_evaluation_digest.miss_classes"
            )

    def _validate_change_evaluation_risks(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not context.change_evaluation_digest:
            return
        prediction = final_report.get("prediction")
        if not isinstance(prediction, dict):
            return
        risk_task_classes = prediction.get("risk_task_classes")
        risk_classes = _change_evaluation_class_markers(
            context.change_evaluation_digest.get("risk_classes"),
            limit=3,
        )
        if risk_classes and not _text_entries_cover_markers(
            risk_task_classes if isinstance(risk_task_classes, list) else [],
            risk_classes,
        ):
            findings.add(
                rule.id,
                "prediction.risk_task_classes must reference top "
                "change_evaluation_digest.risk_classes"
            )

    def _apply_prediction_evidence_binding_gates(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        prediction = final_report.get("prediction")
        if not isinstance(prediction, dict):
            return
        fixed_classes = prediction.get("expected_fixed_task_classes")
        if not isinstance(fixed_classes, list):
            return
        evidence_markers = _prediction_evidence_markers(
            failure_pattern_digest=context.failure_pattern_digest,
            change_evaluation_digest=context.change_evaluation_digest,
            rejected_update_buffer=context.rejected_update_buffer,
            prior_update_lesson_entries=context.prior_update_lesson_entries,
        )
        evidence_markers.extend(
            _mission_candidate_mechanism_markers(context.mission_debug)
        )
        if evidence_markers and not _text_entries_cover_any_marker(
            fixed_classes,
            evidence_markers,
        ):
            findings.add(
                rule.id,
                "prediction.expected_fixed_task_classes must reference a concrete "
                "label from failure_pattern_digest, change_evaluation_digest, "
                "rejected_update_buffer, or prior_update_lesson_entries"
            )

    def _selected_mission_candidate(
        self,
        mission_debug: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = [
            candidate
            for candidate in _mission_candidates(mission_debug)
            if str(candidate.get("id") or "").startswith("mission-attributed-")
        ]
        selected_id = str(
            (mission_debug.get("evidence_summary") or {}).get("selected_candidate_id")
            or ""
        )
        if selected_id:
            for candidate in candidates:
                if str(candidate.get("id") or "") == selected_id:
                    return candidate
        return candidates[0] if len(candidates) == 1 else {}

    def _validate_mission_selection(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        selected_candidate = self._selected_mission_candidate(context.mission_debug)
        if not selected_candidate:
            attributed = [
                candidate
                for candidate in _mission_candidates(context.mission_debug)
                if str(candidate.get("id") or "").startswith("mission-attributed-")
            ]
            if len(attributed) > 1:
                findings.add(
                    report_contract.INTERNAL_RULE_ID,
                    "mission_debug has multiple attributed candidates without one "
                    "selected_candidate_id",
                )
            return
        report_text = _mission_report_text(final_report)
        if not _matching_mission_candidates([selected_candidate], report_text):
            findings.add(
                rule.id,
                "final report must reference one mission_debug.feature_candidates "
                "id or failure_category",
            )

    def _validate_mission_scope(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        selected_candidate = self._selected_mission_candidate(context.mission_debug)
        if not selected_candidate:
            return
        allowed_roots = [
            str(path).strip().rstrip("/")
            for path in selected_candidate.get("allowed_edit_paths") or []
            if str(path).strip()
        ]
        if not allowed_roots:
            return
        out_of_scope = [
            path
            for path in context.changed_files
            if self._is_structural_change(path)
            and not _path_is_under_allowed_roots(path, allowed_roots)
        ]
        if out_of_scope:
            findings.add(
                rule.id,
                "changed files exceed selected mission candidate allowed_edit_paths: "
                + ", ".join(out_of_scope)
            )

    def _valid_report_layers(self) -> set[str]:
        return {
            "prompt",
            "tool",
            "planning",
            "recovery",
            "context",
            "config",
            "adapter",
            "memory",
            "verification",
            "architecture",
            "orchestration",
            "harbor_integration",
            "other",
        }

    def _validate_implementation_scope(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        scope = final_report.get("implementation_scope")
        if not isinstance(scope, dict):
            findings.add(rule.id, "missing implementation_scope")
            return
        primary_layer = str(scope.get("primary_layer") or "").strip()
        component_type = str(final_report.get("component_type") or "").strip()
        valid_layers = self._valid_report_layers()
        if primary_layer not in valid_layers:
            findings.add(
                rule.id,
                "implementation_scope.primary_layer is invalid or missing",
            )
        if component_type and component_type not in valid_layers:
            findings.add(rule.id, "component_type is invalid")
        considered = scope.get("architectural_change_considered")
        if considered is not True:
            findings.add(
                rule.id,
                "implementation_scope.architectural_change_considered must be true"
            )
        structural_files = scope.get("structural_files_changed")
        if not isinstance(structural_files, list):
            findings.add(
                rule.id,
                "implementation_scope.structural_files_changed must be a list",
            )
            structural_files = []
        structural_file_paths = [str(path) for path in structural_files]
        missing_changed = [
            path
            for path in context.changed_files
            if self._is_structural_change(path) and path not in structural_file_paths
        ]
        if missing_changed:
            findings.add(
                rule.id,
                "implementation_scope.structural_files_changed missing changed "
                "structural files: " + ", ".join(missing_changed)
            )
        overreported = [
            path
            for path in structural_file_paths
            if self._is_structural_change(path)
            and path not in context.changed_files
            and path not in context.ignore_files
        ]
        if overreported:
            findings.add(
                rule.id,
                "implementation_scope.structural_files_changed includes files not "
                "changed by the diff: " + ", ".join(overreported)
            )
        if primary_layer == "prompt":
            justification = str(scope.get("why_prompt_only_is_sufficient") or "").strip()
            if not justification:
                findings.add(
                    rule.id,
                    "implementation_scope.why_prompt_only_is_sufficient is required "
                    "for prompt-layer updates"
                )
            if [path for path in structural_files if self._is_structural_change(path)]:
                findings.add(
                    rule.id,
                    "prompt-layer updates must not report structural file changes"
                )

    def _validate_implementation_layer(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        scope = final_report.get("implementation_scope")
        if not isinstance(scope, dict):
            return
        primary_layer = str(scope.get("primary_layer") or "").strip()
        component_type = str(final_report.get("component_type") or "").strip()
        actual_categories = _report_categories_for_changed_files(context.changed_files)
        reported_categories = {
            category
            for category in (primary_layer, component_type)
            if category in self._valid_report_layers()
        }
        if actual_categories and not (reported_categories & actual_categories):
            findings.add(
                rule.id,
                "implementation_scope.primary_layer or component_type must match "
                "the primary changed-file layer: "
                + ", ".join(sorted(actual_categories)),
            )

    def _is_structural_change(self, path: str) -> bool:
        normalized = path.replace("\\", "/")
        if self._is_validation_or_documentation_change(normalized):
            return False
        structural_roots = (
            "bench/",
            "crates/",
            "harness/",
            "hl/",
            "meta/",
            "scripts/",
            "config/",
        )
        if normalized.startswith(structural_roots):
            return True
        return normalized.endswith(
            (".py", ".rs", ".yaml", ".yml", ".toml", ".json")
        )

    def _is_validation_or_documentation_change(self, normalized_path: str) -> bool:
        validation_roots = (
            "tests/",
            "test/",
            "docs/",
        )
        if normalized_path.startswith(validation_roots):
            return True
        basename = normalized_path.rsplit("/", 1)[-1]
        return basename.startswith("test_") or basename.endswith(("_test.py", ".md", ".rst"))

    def _apply_leaderboard_compliance_gates(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        compliance = final_report.get("leaderboard_compliance")
        if not isinstance(compliance, dict):
            findings.add(rule.id, "missing leaderboard_compliance")
            return
        required_true = [
            "harbor_official_harness_preserved",
            "self_owned_worker_preserved",
            "benchmark_integrity_preserved",
            "timeouts_resources_unchanged",
            "submit_gate_preserved",
            "official_dataset_preserved",
            "five_attempts_per_task_preserved",
            "no_prohibited_terminal_bench_access",
            "upload_artifacts_trace_preserved",
        ]
        for field_name in required_true:
            if compliance.get(field_name) is not True:
                findings.add(
                    rule.id,
                    f"leaderboard_compliance.{field_name} must be true",
                )

    def _apply_external_research_gates(
        self,
        rule: report_contract.ReportContractRule,
        final_report: dict[str, Any],
        context: report_contract.ReportValidationContext,
        findings: report_contract.ViolationCollector,
    ) -> None:
        if not context.changed_files or not final_report:
            return
        research = final_report.get("external_research")
        if not isinstance(research, dict):
            findings.add(rule.id, "missing external_research")
            return
        used = research.get("used")
        if not isinstance(used, bool):
            findings.add(rule.id, "external_research.used must be boolean")
        sources = research.get("sources")
        if not isinstance(sources, list):
            findings.add(rule.id, "external_research.sources must be a list")
            sources = []
        reason = str(research.get("reason") or "").strip()
        impact = str(research.get("impact") or "").strip()
        if used and not [source for source in sources if str(source).strip()]:
            findings.add(
                rule.id,
                "external_research.sources required when research was used",
            )
        if used and not impact:
            findings.add(
                rule.id,
                "external_research.impact required when research was used"
            )
        if used:
            allowed_sources = _external_research_allowed_sources(
                context.external_research_policy
            )
            if allowed_sources and not _external_research_sources_allowed(
                sources,
                allowed_sources,
            ):
                findings.add(
                    rule.id,
                    "external_research.sources must come from packet external_research_policy"
                )
            focus_markers = _external_research_focus_markers(
                context.external_research_policy
            )
            if focus_markers and not _external_research_text_matches_focus(
                reason + "\n" + impact,
                focus_markers,
            ):
                findings.add(
                    rule.id,
                    "external_research.impact must reference a packet research_focus_area"
                )
            fetch_reason = _external_research_fetch_requirement_reason(
                research,
                sources,
                context.external_research_policy,
            )
            if fetch_reason:
                findings.add(rule.id, fetch_reason)
        if context.external_research_recommended and not used and not reason:
            findings.add(
                rule.id,
                "external research was recommended after poor updates but no skip reason was reported"
            )

    def _prompt_for_packet(self, packet_path: str | Path) -> str:
        return (
            "You are the Codex-backed HL updater for HarnessEvolver. "
            "Optimize this repository's self-owned Worker loop or harness; do not solve "
            "the current TerminalBench task directly. Make one bounded improvement slice, "
            "preserve benchmark integrity, do not edit terminal-bench tasks/tests/solutions, "
            "and do not create nested sub-agents: only the master HL orchestrator may create "
            "sub-agents, so this Codex update sub-agent must not run Codex CLI, "
            "OpenAI Codex/openai-codex, Claude, ForgeCode, Factory Droid, Factory/factory, "
            "Droid/droid, Gemini, OpenCode, Aider, Amp, Cursor Agent, "
            "or another external coding-agent process. "
            "Run the required validations when possible. Every edit must target a reusable "
            "failure class or capability improvement rather than a single task id. Before "
            "choosing a patch, review campaign_context and failure_pattern_digest, compare "
            "the trigger failures against the recent multi-round score/status/failure "
            "patterns, and select one generalizable problem class; if the trigger failures "
            "are outliers or the cross-round evidence is too mixed, choose noop/rejected "
            "instead of a speculative edit. Compare the "
            "framework before and after the change, define a rollback trigger, write a "
            "falsifiable prediction with expected fixed task classes, risk task classes, "
            "metric delta, confidence, and falsification window, and write both "
            "concise and detailed memory records. If recent updates are repeatedly poor, use the "
            "read-only Codex/ForgeCode references, harness_reference_contract, or industry "
            "best-practice sources listed in the packet before choosing a new approach. When "
            "using Agentic Harness Engineering, Meta-Harness, TACO, OpenClacky, or Claude Code "
            "large-codebase practices, translate the practice into this repo's current Worker "
            "or harness interfaces: observability, frontier comparison, context compression, "
            "cache-stable tool/skill design, deterministic hooks, and progressive repo context "
            "are acceptable local patterns; copying a reference agent or delegating task solving "
            "is not. Terminal-Bench 2.0 leaderboard "
            "compliance is mandatory: keep Harbor as the official harness path, keep this "
            "repository's Worker as the evaluated custom agent, keep official timeouts/resources "
            "unchanged, keep leaderboard-candidate runs at five attempts per task, block "
            "evaluated-Worker access to the Terminal-Bench website/GitHub/internals, and "
            "preserve submit gates. Do not limit yourself to prompt tuning: if the evidence "
            "points to tool, loop, context, recovery, adapter, verification, configuration, "
            "or architecture defects, make the bounded structural change and document the "
            "implementation scope; justify any prompt-only patch. "
            "Apply self_iteration_contract as a continual-learning discipline: decide what "
            "learnable surface changes, how the update is grounded, when it should be promoted "
            "from episodic evidence into long-term policy, and how regression/replay evidence "
            "protects prior solved capabilities. Your own final-report validation list is an "
            "audit trail; the host updater will still run required validation commands and the "
            "dynamic ladder before accepting changed files. "
            "Before editing, challenge your "
            "own strategy: if the evidence leaves material uncertainty, enumerate likely "
            "loopholes, counterexamples, missing validation, regression risks, and benchmark "
            "integrity risks; then refine the strategy or choose noop/rejected instead of "
            "shipping a speculative patch. Repeat this self-review until the remaining "
            "uncertainty is small enough to justify the bounded slice, and report the "
            "remaining confidence without claiming certainty beyond the evidence. "
            "Do not free-write contract values: after editing, obtain "
            "implementation_scope.primary_layer from report_lint's post-edit "
            "valid_primary_layers, set selected_problem_class / "
            "dominant_patterns / generalization.problem_class from "
            "report_value_budget.selected_problem_class_labels (prefer a concrete mechanism "
            "signature over a generic status label), cover every id in "
            "report_value_budget.rejected_update_buffer_ids_to_cover, and reference exactly "
            "one report_value_budget.attributed_feature_candidate_ids entry. Never write a "
            "TerminalBench task id literal into production code; name the mechanism signature "
            "or failure class instead. report_contract_rules lists which report fields are "
            "advisory ('report') versus blocking ('fatal'); before delivering, self-check "
            "with `python scripts/report_lint.py --packet-dir "
            f"{Path(packet_path).parent} --report <draft-report.json>`; use its exact "
            "isolated changed-file list and valid primary layers, then fix every fatal "
            "finding. Read the "
            "structured packet at "
            f"{packet_path} and return only the requested JSON report."
        )


def _canonical_validation_command(command: str) -> str:
    text = command.strip()
    if not text:
        return ""
    text = re.sub(r"\s+\((passed|failed|skipped|exit code)[^)]*\)\s*$", "", text, flags=re.I)
    text = re.sub(r"\s*#.*$", "", text).strip()
    try:
        tokens = shlex.split(text)
    except ValueError:
        return " ".join(text.split())
    return " ".join(_canonical_validation_tokens(tokens))


def _canonical_validation_tokens(tokens: list[str]) -> list[str]:
    if len(tokens) >= 3 and tokens[0].endswith("python") and tokens[1:3] == ["-m", "pytest"]:
        tokens = ["pytest", *tokens[3:]]
    if tokens and tokens[0].endswith("pytest"):
        tokens = ["pytest", *tokens[1:]]
    if tokens and tokens[0].endswith("python"):
        tokens = ["python", *tokens[1:]]
    return tokens


def _codex_event_command_strings(events_text: str) -> list[str]:
    commands: list[str] = []
    for raw_line in events_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for command in _codex_event_command_values(event):
            command_text = str(command).strip()
            if command_text:
                commands.append(command_text)
    return commands


def _codex_event_command_values(value: Any) -> list[str]:
    commands: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).lower()
            if normalized_key in {"command", "cmd", "argv"}:
                commands.extend(_codex_event_command_texts(nested))
                continue
            commands.extend(_codex_event_command_values(nested))
    elif isinstance(value, list):
        for nested in value:
            commands.extend(_codex_event_command_values(nested))
    return commands


def _codex_event_command_texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if value and all(isinstance(item, (str, int, float)) for item in value):
            return [" ".join(shlex.quote(str(item)) for item in value)]
        return _codex_event_command_values(value)
    if isinstance(value, dict):
        return _codex_event_command_values(value)
    return []


def _validation_command_satisfies(candidate: str, required: str) -> bool:
    if not candidate or not required:
        return False
    if candidate == required:
        return True
    if candidate.startswith(required + " "):
        return True
    return _drop_pytest_verbosity(candidate) == _drop_pytest_verbosity(required)


def _failed_directions_cover_rejected_buffer(
    failed_directions: list[Any],
    rejected_update_buffer: list[dict[str, Any]],
) -> bool:
    actionable_entries = [
        entry
        for entry in rejected_update_buffer
        if not _rejected_buffer_entry_is_non_actionable(entry)
    ]
    if not actionable_entries:
        return True
    direction_text = "\n".join(str(item).lower() for item in failed_directions)
    if not direction_text.strip():
        return False
    layer_counts: dict[str, int] = {}
    for entry in actionable_entries:
        layer = str(entry.get("component_layer") or "").strip().lower()
        if layer:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
    for entry in actionable_entries:
        packet_id = str(entry.get("packet_id") or "")
        failure_class = str(entry.get("failure_class") or "")
        component_layer = str(entry.get("component_layer") or "")
        mission_candidate_id = str(entry.get("mission_candidate_id") or "")
        markers = (
            [mission_candidate_id]
            if mission_candidate_id
            else [packet_id, failure_class]
        )
        layer = component_layer.strip().lower()
        if not mission_candidate_id and layer_counts.get(layer, 0) <= 1:
            markers.append(component_layer)
        entry_covered = False
        for marker in markers:
            marker = marker.strip().lower()
            if len(marker) >= 3 and marker in direction_text:
                entry_covered = True
                break
        if not entry_covered:
            return False
    return True


def _rejected_buffer_entry_is_non_actionable(entry: dict[str, Any]) -> bool:
    return bool(entry.get("superseded_by_current_reviewer")) or entry.get(
        "avoid_repeating"
    ) is False


def _failed_directions_cover_loophole_records(
    failed_directions: list[Any],
    rejected_update_buffer: list[dict[str, Any]],
) -> bool:
    actionable_entries = [
        entry
        for entry in rejected_update_buffer
        if not _rejected_buffer_entry_is_non_actionable(entry)
    ]
    if not actionable_entries:
        return True
    direction_text = "\n".join(str(item).lower() for item in failed_directions)
    if not direction_text.strip():
        return False
    for entry in actionable_entries:
        markers = _loophole_record_markers(entry)
        if not markers:
            continue
        if not any(marker in direction_text for marker in markers):
            return False
    return True


def _failed_directions_cover_required_mutations(
    failed_directions: list[Any],
    rejected_update_buffer: list[dict[str, Any]],
) -> bool:
    actionable_entries = [
        entry
        for entry in rejected_update_buffer
        if not _rejected_buffer_entry_is_non_actionable(entry)
    ]
    if not actionable_entries:
        return True
    direction_text = "\n".join(str(item).lower() for item in failed_directions)
    if not direction_text.strip():
        return False
    for entry in actionable_entries:
        markers = _required_mutation_markers(entry)
        if not markers:
            continue
        if not any(marker in direction_text for marker in markers):
            return False
    return True


def _required_mutation_markers(entry: dict[str, Any]) -> list[str]:
    if _rejected_buffer_entry_is_non_actionable(entry):
        return []
    mutation = str(entry.get("required_mutation") or "").strip().lower()
    if not mutation:
        return []
    specific_marker_candidates = [
        "dirty baseline",
        "git status clean",
        "allow_dirty_baseline",
        "baseline-delta evidence",
        "pre-update regression",
        "stale baseline snapshot",
        "same-model pre-update gate",
        "post-update regression",
        "regressed solved-task class",
        "external_research.impact",
        "external_research.sources",
        "external_research.fetches",
        "external_research_policy",
        "required_user_agent",
        "research_focus_areas",
        "change_evaluation direction",
        "missed evaluation evidence",
        "missed tasks",
        "missed classes",
        "rollback/risk-control check",
    ]
    specific_markers = [
        marker for marker in specific_marker_candidates if marker in mutation
    ]
    if specific_markers:
        return specific_markers
    marker_candidates = [
        "no-diff update",
        "tracked worker/harness change",
        "required validation",
        "skipped_validation_reason",
        "out-of-scope edit",
        "allowed edit roots",
        "review reasons",
        "missing evidence",
        "fresh trajectory",
        "verifier evidence",
        "same-model frontier",
        "regressed tasks",
        "risk control",
        "regression gates",
        "regression risk control",
        "concrete mutation",
        "failure_class/component_layer",
    ]
    markers = [marker for marker in marker_candidates if marker in mutation]
    return markers or [mutation[:80]]


def _loophole_record_markers(entry: dict[str, Any]) -> list[str]:
    markers: list[str] = []
    for key in ("loophole_review", "loophole_fixes"):
        raw_items = entry.get(key)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            marker = str(item).strip().lower()
            if len(marker) >= 3:
                markers.append(marker)
    return markers


def _failed_directions_cover_layer_pressure(
    failed_directions: list[Any],
    layer_pressure_entries: list[dict[str, Any]],
) -> bool:
    direction_text = "\n".join(str(item).lower() for item in failed_directions)
    if not direction_text.strip():
        return False
    for entry in layer_pressure_entries:
        component_layer = str(entry.get("component_layer") or "").strip().lower()
        if not component_layer or component_layer not in direction_text:
            return False
        specific_markers = []
        for key in ("packet_id", "failure_class"):
            specific_markers.extend(
                marker.strip().lower()
                for marker in str(entry.get(key) or "").split()
                if len(marker.strip()) >= 3
            )
        if specific_markers and not any(
            marker in direction_text for marker in specific_markers
        ):
            return False
    return True


# Evidence-binding gates only treat short, referenceable failure-class labels as
# matchable markers. Long free-text narration entries in change_evaluation_digest
# are dropped so the report is not required to reproduce a whole sentence verbatim.
_CHANGE_EVAL_CLASS_MARKER_MIN_LEN = 3
_CHANGE_EVAL_CLASS_MARKER_MAX_LEN = 60
_CHANGE_EVAL_CLASS_SCAN_MULTIPLIER = 4


def _change_evaluation_class_markers(raw_entries: Any, *, limit: int) -> list[str]:
    if not isinstance(raw_entries, list):
        return []
    markers: list[str] = []
    # A "class" marker is only useful as an evidence-binding gate when it is a
    # short, referenceable failure-class label (e.g. "environment_start_timeout").
    # Some change_evaluation_digest entries are long free-text narration carried
    # over from unrelated prior changes; requiring the report to reproduce such a
    # sentence verbatim is unsatisfiable and rolls back otherwise-good patches.
    # Skip over-long entries but keep scanning a bounded window so real ranked
    # short labels are still enforced.
    scan_window = raw_entries[: max(limit * _CHANGE_EVAL_CLASS_SCAN_MULTIPLIER, limit)]
    for raw_entry in scan_window:
        if len(markers) >= limit:
            break
        if not isinstance(raw_entry, dict):
            continue
        marker = str(raw_entry.get("class") or "").strip()
        if _CHANGE_EVAL_CLASS_MARKER_MIN_LEN <= len(marker) <= _CHANGE_EVAL_CLASS_MARKER_MAX_LEN:
            markers.append(marker)
    return markers


def _text_entries_cover_markers(entries: list[Any], markers: list[str]) -> bool:
    text = "\n".join(str(item).lower() for item in entries)
    if not text.strip():
        return False
    for marker in markers:
        marker_text = marker.strip().lower()
        if len(marker_text) >= 3 and marker_text not in text:
            return False
    return True


def _text_entries_cover_any_marker(entries: list[Any], markers: list[str]) -> bool:
    text = "\n".join(str(item).lower() for item in entries)
    if not text.strip():
        return False
    return any(
        marker.strip().lower() in text
        for marker in markers
        if len(marker.strip()) >= 3
    )


def _prediction_evidence_markers(
    *,
    failure_pattern_digest: dict[str, Any],
    change_evaluation_digest: dict[str, Any],
    rejected_update_buffer: list[dict[str, Any]],
    prior_update_lesson_entries: list[dict[str, Any]],
) -> list[str]:
    markers: list[str] = []
    markers.extend(_failure_pattern_markers(failure_pattern_digest))
    markers.extend(
        _change_evaluation_class_markers(
            change_evaluation_digest.get("hit_classes"),
            limit=3,
        )
    )
    markers.extend(
        _change_evaluation_class_markers(
            change_evaluation_digest.get("miss_classes"),
            limit=3,
        )
    )
    markers.extend(
        _rejected_update_buffer_markers(
            rejected_update_buffer,
            limit=3,
        )
    )
    markers.extend(_prior_update_lesson_markers(prior_update_lesson_entries, limit=3))
    deduped: list[str] = []
    seen: set[str] = set()
    for marker in markers:
        normalized = marker.strip().lower()
        if len(normalized) < 3 or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(marker.strip())
    return deduped


def _prior_update_lesson_markers(
    raw_entries: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    if not isinstance(raw_entries, list):
        return []
    markers: list[str] = []
    for raw_entry in raw_entries:
        if len(markers) >= limit:
            break
        if not isinstance(raw_entry, dict):
            continue
        for key in (
            "mission_candidate_id",
            "mission_failure_category",
            "outcome",
            "packet_id",
        ):
            marker = str(raw_entry.get(key) or "").strip()
            if len(marker) >= 3:
                markers.append(marker)
                break
    return markers


def _summary_decision_inputs(packet: CodexWorkPacket) -> dict[str, str]:
    return {
        "analysis_candidate_classes": _summary_analysis_candidate_classes(
            packet.campaign_context.get("recent_analysis_reports")
        ),
        "analysis_failure_buckets": _summary_analysis_failure_buckets(
            packet.campaign_context.get("recent_analysis_reports")
        ),
        "analysis_mechanism_update_classes": _summary_analysis_mechanism_update_classes(
            packet.campaign_context.get("recent_analysis_reports")
        ),
        "analysis_policy_coverage": _summary_analysis_policy_coverage(
            packet.campaign_context.get("recent_analysis_reports")
        ),
        "policy_recurrence_signals": _summary_policy_recurrence_signals(
            packet.policy_recurrence_signals
        ),
        "infrastructure_triage": _summary_infrastructure_triage(
            packet.infrastructure_triage
        ),
        "analysis_trajectory_evidence": _summary_analysis_trajectory_evidence(
            packet.campaign_context.get("recent_analysis_reports")
        ),
        "analysis_weakness_signatures": _summary_analysis_weakness_signatures(
            packet.campaign_context.get("recent_analysis_reports")
        ),
        "self_harness_improvement_queue": _summary_self_harness_improvement_queue(
            packet.self_harness_improvement_queue
        ),
        "change_evaluation_miss_classes": _summary_class_counts(
            packet.change_evaluation_digest.get("miss_classes")
        ),
        "change_evaluation_risk_classes": _summary_class_counts(
            packet.change_evaluation_digest.get("risk_classes")
        ),
        "rejected_update_packets": _summary_rejected_update_packets(
            packet.rejected_update_buffer
        ),
        "rejected_required_mutations": _summary_rejected_required_mutations(
            packet.rejected_update_buffer
        ),
        "prior_update_lessons": _summary_prior_update_lessons(
            packet.prior_update_lesson_entries
        ),
        "pivot_discouraged": _summary_direction_entries(
            packet.runner_pivot_policy.get("discouraged")
        ),
        "pivot_layer_pressure": _summary_layer_pressure_entries(
            packet.runner_pivot_policy.get("layer_pressure")
        ),
        "pivot_supported": _summary_direction_entries(
            packet.runner_pivot_policy.get("supported")
        ),
        "search_candidate_rules": _summary_text_entries(
            packet.update_search_policy.get("candidate_generation_rules"),
            limit=3,
        ),
    }


def _summary_prior_update_lessons(
    raw_entries: Any,
    *,
    limit: int = 4,
) -> str:
    if not isinstance(raw_entries, list):
        return "none"
    entries: list[str] = []
    for raw_entry in raw_entries:
        if len(entries) >= limit:
            break
        if not isinstance(raw_entry, dict):
            continue
        parts = []
        for key in (
            "packet_id",
            "mission_candidate_id",
            "outcome",
            "source",
        ):
            value = str(raw_entry.get(key) or "").strip()
            if value:
                parts.append(value)
        if parts:
            entries.append(" / ".join(parts))
    return "; ".join(entries) or "none"


def _summary_analysis_candidate_classes(
    raw_reports: Any,
    *,
    limit: int = 4,
) -> str:
    if not isinstance(raw_reports, list):
        return "none"
    entries: list[str] = []
    for report in raw_reports:
        if len(entries) >= limit:
            break
        if not isinstance(report, dict):
            continue
        summary_id = str(report.get("summary_id") or "").strip()
        for item in _normalized_summary_analysis_candidate_classes(report):
            if len(entries) >= limit:
                break
            text = str(item).strip()
            if not text:
                continue
            entries.append((summary_id + ": " if summary_id else "") + text)
    return "; ".join(entries) or "none"


def _normalized_summary_analysis_candidate_classes(
    report: dict[str, Any],
) -> list[str]:
    raw_candidates = [
        str(item) for item in report.get("candidate_update_classes") or []
    ]
    failure_buckets = [
        bucket
        for bucket in report.get("failure_buckets") or []
        if isinstance(bucket, dict)
    ]
    bucket_candidates = _summary_analysis_candidate_classes_from_buckets(
        failure_buckets,
    )
    if not raw_candidates or not bucket_candidates:
        return raw_candidates or bucket_candidates
    bucket_categories = {
        str(bucket.get("failure_category") or "").strip()
        for bucket in failure_buckets
        if str(bucket.get("failure_category") or "").strip()
    }
    candidate_categories = {
        _summary_candidate_update_class_category(candidate)
        for candidate in raw_candidates
    }
    candidate_categories.discard("")
    if candidate_categories and candidate_categories.issubset(bucket_categories):
        return raw_candidates
    return bucket_candidates


def _summary_analysis_candidate_classes_from_buckets(
    failure_buckets: list[dict[str, Any]],
) -> list[str]:
    candidates: list[str] = []
    for bucket in failure_buckets:
        category = str(bucket.get("failure_category") or "").strip()
        if not category:
            continue
        components = ", ".join(
            str(item) for item in bucket.get("affected_components") or ["unknown"]
        )
        prefix = "infrastructure " if bucket.get("infrastructure") else ""
        count = bucket.get("count")
        if isinstance(count, int):
            suffix = f" ({count} trial(s))"
        else:
            suffix = ""
        candidates.append(f"{prefix}{category} -> {components}{suffix}")
    return candidates


def _summary_candidate_update_class_category(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    category = text.split("->", 1)[0].strip()
    if category.startswith("infrastructure "):
        category = category[len("infrastructure ") :].strip()
    return category


def _summary_analysis_failure_buckets(
    raw_reports: Any,
    *,
    limit: int = 4,
) -> str:
    if not isinstance(raw_reports, list):
        return "none"
    entries: list[str] = []
    for report in raw_reports:
        if len(entries) >= limit:
            break
        if not isinstance(report, dict):
            continue
        summary_id = str(report.get("summary_id") or "").strip()
        for bucket in report.get("failure_buckets") or []:
            if len(entries) >= limit:
                break
            if not isinstance(bucket, dict):
                continue
            category = str(bucket.get("failure_category") or "").strip()
            if not category:
                continue
            count = bucket.get("count")
            components = ", ".join(
                str(item) for item in bucket.get("affected_components") or []
            )
            parts = [(summary_id + ": " if summary_id else "") + category]
            if isinstance(count, int):
                parts.append(str(count) + " trial(s)")
            if components:
                parts.append(components)
            if bucket.get("infrastructure"):
                parts.append("infrastructure")
            entries.append(" / ".join(parts))
    return "; ".join(entries) or "none"


def _summary_analysis_mechanism_update_classes(
    raw_reports: Any,
    *,
    limit: int = 5,
) -> str:
    if not isinstance(raw_reports, list):
        return "none"
    entries: list[str] = []
    for report in raw_reports:
        if len(entries) >= limit:
            break
        if not isinstance(report, dict):
            continue
        summary_id = str(report.get("summary_id") or "").strip()
        for item in report.get("mechanism_update_classes") or []:
            if len(entries) >= limit:
                break
            text = str(item).strip()
            if not text:
                continue
            entries.append((summary_id + ": " if summary_id else "") + text)
    return "; ".join(entries) or "none"


def _summary_policy_recurrence_signals(
    raw_signals: Any,
    *,
    limit: int = 5,
) -> str:
    if not isinstance(raw_signals, list):
        return "none"
    entries: list[str] = []
    for raw_signal in raw_signals:
        if len(entries) >= limit:
            break
        if not isinstance(raw_signal, dict):
            continue
        summary_id = str(raw_signal.get("summary_id") or "").strip()
        category = str(raw_signal.get("failure_category") or "").strip()
        policy = str(raw_signal.get("policy") or "").strip()
        if not category and not policy:
            continue
        parts = []
        if summary_id:
            parts.append(summary_id)
        if category:
            parts.append(category)
        if policy and policy != category:
            parts.append("policy=" + policy)
        coverage_count = raw_signal.get("policy_coverage_count")
        if isinstance(coverage_count, int) and coverage_count > 0:
            parts.append(f"coverage={coverage_count}")
        trajectory_count = raw_signal.get("trajectory_policy_count")
        if isinstance(trajectory_count, int) and trajectory_count > 0:
            parts.append(f"trajectory={trajectory_count}")
        raw_tasks = raw_signal.get("task_ids")
        if isinstance(raw_tasks, list):
            tasks = [str(item).strip() for item in raw_tasks if str(item).strip()]
            if tasks:
                parts.append("tasks=" + ",".join(tasks[:4]))
        raw_sources = raw_signal.get("evidence_sources")
        if isinstance(raw_sources, list):
            sources = [str(item).strip() for item in raw_sources if str(item).strip()]
            if sources:
                parts.append("sources=" + ",".join(sources[:4]))
        entries.append(" / ".join(parts))
    return "; ".join(entries) or "none"


def _summary_infrastructure_triage(raw_triage: Any) -> str:
    if not isinstance(raw_triage, dict):
        return "none"
    categories = [
        str(item).strip()
        for item in raw_triage.get("infrastructure_categories") or []
        if str(item).strip()
    ]
    if not categories:
        return "none"
    parts = ["infra=" + ",".join(categories[:5])]
    if raw_triage.get("trigger_all_infrastructure") is True:
        parts.append("trigger_all_infrastructure=true")
    trigger_count = raw_triage.get("trigger_infrastructure_count")
    trigger_total = raw_triage.get("trigger_total_failures")
    if isinstance(trigger_count, int) and isinstance(trigger_total, int):
        parts.append(f"trigger={trigger_count}/{trigger_total}")
    layers = [
        str(item).strip()
        for item in raw_triage.get("recommended_layers") or []
        if str(item).strip()
    ]
    if layers:
        parts.append("layers=" + ",".join(layers[:5]))
    warmup_commands: list[str] = []
    for item in raw_triage.get("trigger_items") or []:
        if not isinstance(item, dict):
            continue
        for command in item.get("prebuilt_image_cache_warmup_commands") or []:
            command_text = str(command).strip()
            if command_text and command_text not in warmup_commands:
                warmup_commands.append(command_text)
            if len(warmup_commands) >= 3:
                break
        if len(warmup_commands) >= 3:
            break
    if warmup_commands:
        parts.append("prebuilt_cache_warmup=" + " | ".join(warmup_commands))
    if any(
        isinstance(item, dict) and item.get("network_preflight_recommended") is True
        for item in raw_triage.get("trigger_items") or []
    ):
        parts.append("network_preflight_recommended=true")
    avoid = [
        str(item).strip()
        for item in raw_triage.get("avoid_worker_policy_layers_when_infrastructure_only") or []
        if str(item).strip()
    ]
    if avoid:
        parts.append("avoid_worker_policy_layers=" + ",".join(avoid[:4]))
    recent_items = raw_triage.get("recent_items")
    if isinstance(recent_items, list) and recent_items:
        summaries = []
        for item in recent_items:
            if not isinstance(item, dict):
                continue
            summary_id = str(item.get("summary_id") or "").strip()
            category = str(item.get("failure_category") or "").strip()
            count = item.get("count")
            if summary_id and category:
                suffix = f"={count}" if isinstance(count, int) else ""
                summaries.append(f"{summary_id}:{category}{suffix}")
            if len(summaries) >= 3:
                break
        if summaries:
            parts.append("recent=" + ",".join(summaries))
    guidance = str(raw_triage.get("selection_guidance") or "").strip()
    if guidance and categories:
        parts.append(_truncate_summary_text(guidance, 180))
    return " / ".join(parts)


def _summary_analysis_policy_coverage(
    raw_reports: Any,
    *,
    policy_limit: int = 4,
    uncovered_limit: int = 3,
) -> str:
    if not isinstance(raw_reports, list):
        return "none"
    policy_entries: list[str] = []
    uncovered_entries: list[str] = []
    for report in raw_reports:
        if not isinstance(report, dict):
            continue
        summary_id = str(report.get("summary_id") or "").strip()
        prefix = summary_id + ": " if summary_id else ""
        coverage = report.get("policy_coverage")
        if not isinstance(coverage, dict):
            continue
        for policy in _summary_policy_coverage_items(coverage):
            if len(policy_entries) >= policy_limit:
                break
            if not isinstance(policy, dict):
                continue
            name = str(policy.get("policy") or policy.get("name") or "").strip()
            if not name:
                continue
            count = _summary_policy_count(policy.get("count"))
            if count is not None:
                entry = prefix + f"{name}={count}"
            else:
                entry = prefix + name
            raw_tasks = policy.get("tasks")
            tasks = []
            if isinstance(raw_tasks, list):
                tasks = [str(item).strip() for item in raw_tasks if str(item).strip()][
                    :3
                ]
            if tasks:
                entry += " tasks=" + ",".join(tasks)
            examples = _summary_policy_examples(policy.get("examples"), limit=2)
            if examples:
                entry += " examples=" + " | ".join(examples)
            policy_entries.append(entry)
        for example in coverage.get("uncovered_timeout_examples") or []:
            if len(uncovered_entries) >= uncovered_limit:
                break
            if not isinstance(example, dict):
                continue
            task_id = str(example.get("task_id") or "").strip()
            command = str(example.get("command") or "").strip()
            if not task_id and not command:
                continue
            detail = (task_id + ": " if task_id else "") + _truncate_summary_text(
                command or "timeout without covered policy", 120
            )
            uncovered_entries.append(prefix + "uncovered_timeout " + detail)
        for example in coverage.get("currently_covered_timeout_examples") or []:
            if len(uncovered_entries) >= uncovered_limit:
                break
            if not isinstance(example, dict):
                continue
            task_id = str(example.get("task_id") or "").strip()
            command = str(example.get("command") or "").strip()
            matches = example.get("current_policy_matches")
            match_text = ""
            if isinstance(matches, list):
                match_values = [
                    str(item).strip() for item in matches if str(item).strip()
                ]
                if match_values:
                    match_text = " now=" + ",".join(match_values[:3])
            if not task_id and not command:
                continue
            detail = (task_id + ": " if task_id else "") + _truncate_summary_text(
                command or "historical uncovered timeout now covered", 120
            )
            uncovered_entries.append(
                prefix + "resolved_uncovered_timeout " + detail + match_text
            )
    return "; ".join([*policy_entries, *uncovered_entries]) or "none"


def _summary_policy_coverage_items(coverage: dict[str, Any]) -> list[dict[str, Any]]:
    top_policies = coverage.get("top_policies")
    if isinstance(top_policies, list):
        return [item for item in top_policies if isinstance(item, dict)]
    policies = coverage.get("policies")
    if not isinstance(policies, dict):
        return []
    items: list[dict[str, Any]] = []
    for name, raw_policy in policies.items():
        if not isinstance(raw_policy, dict):
            continue
        item = dict(raw_policy)
        item.setdefault("policy", str(name))
        items.append(item)
    return sorted(
        items,
        key=lambda item: (
            -(_summary_policy_count(item.get("count")) or 0),
            str(item.get("policy") or item.get("name") or ""),
        ),
    )


def _summary_policy_count(raw_count: Any) -> int | None:
    if isinstance(raw_count, bool):
        return None
    if isinstance(raw_count, int):
        return raw_count
    if isinstance(raw_count, str) and raw_count.strip().isdigit():
        return int(raw_count.strip())
    return None


def _summary_policy_examples(raw_examples: Any, *, limit: int) -> list[str]:
    if not isinstance(raw_examples, list):
        return []
    examples: list[str] = []
    for raw_example in raw_examples:
        if len(examples) >= limit:
            break
        if not isinstance(raw_example, dict):
            continue
        task_id = str(raw_example.get("task_id") or "").strip()
        command = str(raw_example.get("command") or "").strip()
        if not task_id and not command:
            continue
        examples.append(
            (task_id + ": " if task_id else "")
            + _truncate_summary_text(command or "policy example", 100)
        )
    return examples


def _summary_analysis_trajectory_evidence(
    raw_reports: Any,
    *,
    limit: int = 6,
) -> str:
    if not isinstance(raw_reports, list):
        return "none"
    entries: list[str] = []
    for report in raw_reports:
        if len(entries) >= limit:
            break
        if not isinstance(report, dict):
            continue
        summary_id = str(report.get("summary_id") or "").strip()
        evidence = report.get("trajectory_evidence")
        if not isinstance(evidence, dict):
            continue
        for task_id, raw_entry in evidence.items():
            if len(entries) >= limit:
                break
            if not isinstance(raw_entry, dict):
                continue
            parts = []
            policy_counts = raw_entry.get("policy_counts")
            if isinstance(policy_counts, dict) and policy_counts:
                top_policies = []
                for name, count in list(policy_counts.items())[:3]:
                    top_policies.append(f"{name}={count}")
                if top_policies:
                    parts.append("policies " + ", ".join(top_policies))
            for label, key in (
                ("timeouts", "timed_out_commands"),
                ("blocked", "blocked_guards"),
                ("deps", "dependency_and_toolchain_evidence"),
                ("progress", "deliverable_progress"),
                ("terminal", "terminal_environment_markers"),
            ):
                items = raw_entry.get(key)
                if not isinstance(items, list) or not items:
                    continue
                commands = []
                for item in items[:2]:
                    if isinstance(item, dict):
                        command = str(item.get("command") or "").strip()
                    else:
                        command = str(item).strip()
                    if command:
                        commands.append(command)
                if commands:
                    parts.append(label + " " + " | ".join(commands))
            if parts:
                prefix = (summary_id + ": " if summary_id else "") + str(task_id)
                entries.append(prefix + " / " + " / ".join(parts))
    return "; ".join(entries) or "none"


def _summary_analysis_weakness_signatures(
    raw_reports: Any,
    *,
    limit: int = 5,
) -> str:
    if not isinstance(raw_reports, list):
        return "none"
    entries: list[str] = []
    for report in raw_reports:
        if len(entries) >= limit:
            break
        if not isinstance(report, dict):
            continue
        summary_id = str(report.get("summary_id") or "").strip()
        for raw_signature in report.get("weakness_signatures") or []:
            if len(entries) >= limit:
                break
            if not isinstance(raw_signature, dict):
                continue
            signature = str(raw_signature.get("signature") or "").strip()
            category = str(raw_signature.get("failure_category") or "").strip()
            if not signature and not category:
                continue
            parts = []
            if summary_id:
                parts.append(summary_id)
            if category:
                parts.append(category)
            if signature:
                parts.append("signature=" + signature)
            count = raw_signature.get("count")
            if isinstance(count, int) and count > 0:
                parts.append(f"count={count}")
            raw_tasks = raw_signature.get("task_ids")
            if isinstance(raw_tasks, list):
                tasks = [str(item).strip() for item in raw_tasks if str(item).strip()]
                if tasks:
                    parts.append("tasks=" + ",".join(tasks[:4]))
            entries.append(" / ".join(parts))
    return "; ".join(entries) or "none"


def _summary_self_harness_improvement_queue(
    raw_queue: Any,
    *,
    limit: int = 5,
) -> str:
    if not isinstance(raw_queue, dict):
        return "none"
    raw_candidates = raw_queue.get("candidates")
    if not isinstance(raw_candidates, list):
        return "none"
    entries: list[str] = []
    for raw_candidate in raw_candidates:
        if len(entries) >= limit:
            break
        if not isinstance(raw_candidate, dict):
            continue
        candidate_id = str(raw_candidate.get("candidate_id") or "").strip()
        label = str(raw_candidate.get("evidence_label") or "").strip()
        category = str(raw_candidate.get("failure_category") or "").strip()
        if not candidate_id and not label and not category:
            continue
        parts = []
        if candidate_id:
            parts.append(candidate_id)
        if category:
            parts.append("category=" + category)
        if label and label != category:
            parts.append("label=" + _truncate_summary_text(label, 160))
        score = raw_candidate.get("score_audit_only")
        if isinstance(score, int):
            parts.append(f"score={score}")
        kind = str(raw_candidate.get("proposal_kind") or "").strip()
        if kind:
            parts.append("kind=" + kind)
        raw_surfaces = raw_candidate.get("recommended_edit_surfaces")
        if isinstance(raw_surfaces, list):
            surfaces = [str(item).strip() for item in raw_surfaces if str(item).strip()]
            if surfaces:
                parts.append("surfaces=" + ",".join(surfaces[:4]))
        raw_tasks = raw_candidate.get("task_ids")
        if isinstance(raw_tasks, list):
            tasks = [str(item).strip() for item in raw_tasks if str(item).strip()]
            if tasks:
                parts.append("tasks=" + ",".join(tasks[:4]))
        entries.append(" / ".join(parts))
    return "; ".join(entries) or "none"


def _summary_class_counts(raw_entries: Any, *, limit: int = 4) -> str:
    if not isinstance(raw_entries, list):
        return "none"
    entries: list[str] = []
    for raw_entry in raw_entries:
        if len(entries) >= limit:
            break
        if not isinstance(raw_entry, dict):
            continue
        label = str(raw_entry.get("class") or "").strip()
        if not label:
            continue
        count = raw_entry.get("count")
        if isinstance(count, int):
            entries.append(f"{label} ({count})")
        else:
            entries.append(label)
    return "; ".join(entries) or "none"


def _summary_rejected_update_packets(
    rejected_update_buffer: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> str:
    entries: list[str] = []
    for entry in rejected_update_buffer:
        if len(entries) >= limit:
            break
        if not isinstance(entry, dict):
            continue
        parts = [str(entry.get("packet_id") or "unknown").strip()]
        failure_class = str(entry.get("failure_class") or "").strip()
        component_layer = str(entry.get("component_layer") or "").strip()
        source = str(entry.get("source") or "").strip()
        if failure_class:
            parts.append(failure_class)
        if component_layer:
            parts.append(component_layer)
        if source:
            parts.append(source)
        entries.append(" / ".join(parts))
    return "; ".join(entries) or "none"


def _summary_rejected_required_mutations(
    rejected_update_buffer: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> str:
    entries: list[str] = []
    for entry in rejected_update_buffer:
        if len(entries) >= limit:
            break
        if not isinstance(entry, dict):
            continue
        mutation = str(entry.get("required_mutation") or "").strip()
        if not mutation:
            continue
        packet_id = str(entry.get("packet_id") or "unknown").strip()
        entries.append(packet_id + ": " + _truncate_summary_text(mutation, 180))
    return "; ".join(entries) or "none"


def _truncate_summary_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _summary_direction_entries(raw_entries: Any, *, limit: int = 4) -> str:
    if not isinstance(raw_entries, list):
        return "none"
    entries: list[str] = []
    for raw_entry in raw_entries:
        if len(entries) >= limit:
            break
        if not isinstance(raw_entry, dict):
            continue
        parts = []
        for key in (
            "packet_id",
            "mission_candidate_id",
            "mission_failure_category",
            "failure_class",
            "component_layer",
        ):
            value = str(raw_entry.get(key) or "").strip()
            if value:
                parts.append(value)
        if parts:
            entries.append(" / ".join(parts))
    return "; ".join(entries) or "none"


def _summary_layer_pressure_entries(raw_entries: Any, *, limit: int = 4) -> str:
    if not isinstance(raw_entries, list):
        return "none"
    entries: list[str] = []
    for raw_entry in raw_entries:
        if len(entries) >= limit:
            break
        if not isinstance(raw_entry, dict):
            continue
        component_layer = str(raw_entry.get("component_layer") or "").strip()
        if not component_layer:
            continue
        failure_classes = _summary_text_entries(
            raw_entry.get("failure_classes"),
            limit=2,
            separator=", ",
        )
        packet_ids = _summary_text_entries(
            raw_entry.get("recent_packet_ids"),
            limit=2,
            separator=", ",
        )
        suffix = []
        if failure_classes != "none":
            suffix.append(failure_classes)
        if packet_ids != "none":
            suffix.append(packet_ids)
        entries.append(
            component_layer + (" (" + "; ".join(suffix) + ")" if suffix else "")
        )
    return "; ".join(entries) or "none"


def _summary_text_entries(
    raw_entries: Any,
    *,
    limit: int,
    separator: str = "; ",
) -> str:
    if not isinstance(raw_entries, list):
        return "none"
    entries = [str(item).strip() for item in raw_entries if str(item).strip()]
    return separator.join(entries[:limit]) or "none"


def _summary_fetch_requirements(raw_entries: Any, *, limit: int = 3) -> str:
    if not isinstance(raw_entries, list):
        return "none"
    entries: list[str] = []
    for raw_entry in raw_entries:
        if len(entries) >= limit:
            break
        if not isinstance(raw_entry, dict):
            continue
        url_prefix = str(raw_entry.get("url_prefix") or "").strip()
        required_user_agent = str(raw_entry.get("required_user_agent") or "").strip()
        failure_signature = str(raw_entry.get("failure_signature") or "").strip()
        parts = []
        if url_prefix:
            parts.append(url_prefix)
        if required_user_agent:
            parts.append("ua=" + _summary_user_agent(required_user_agent))
        if failure_signature:
            parts.append("failure=" + failure_signature)
        if parts:
            entries.append(" / ".join(parts))
    return "; ".join(entries) or "none"


def _summary_user_agent(user_agent: str, *, limit: int = 120) -> str:
    if len(user_agent) <= limit:
        return user_agent
    marker = "MicroMessenger"
    if marker in user_agent:
        marker_index = user_agent.index(marker)
        prefix_start = max(0, marker_index - 40)
        suffix_end = min(len(user_agent), marker_index + 55)
        return user_agent[prefix_start:suffix_end]
    return user_agent[:limit]


def _external_research_allowed_sources(policy: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for key in ("web_sources", "local_read_only_refs"):
        raw_entries = policy.get(key)
        if not isinstance(raw_entries, list):
            continue
        for item in raw_entries:
            source = str(item).strip()
            if source:
                sources.append(source)
    return sources


def _external_research_sources_allowed(
    reported_sources: list[Any],
    allowed_sources: list[str],
) -> bool:
    normalized_allowed = [source.rstrip("/") for source in allowed_sources if source]
    for item in reported_sources:
        source = str(item).strip().rstrip("/")
        if not source:
            continue
        if not any(
            source == allowed
            or source.startswith(allowed + "/")
            for allowed in normalized_allowed
        ):
            return False
    return True


def _external_research_focus_markers(policy: dict[str, Any]) -> list[str]:
    raw_entries = policy.get("research_focus_areas")
    if not isinstance(raw_entries, list):
        return []
    marker_groups = [
        ("action space", "shell/file"),
        ("done", "verifier"),
        ("context", "tool-output"),
        ("provider", "handoff"),
        ("reasoning", "token"),
        ("cache", "validation"),
        ("self-harness", "weakness mining"),
        ("bounded harness proposal", "proposal validation"),
        ("same-model", "self-improvement"),
    ]
    markers: list[str] = []
    for item in raw_entries:
        text = str(item).lower()
        for group in marker_groups:
            if any(marker in text for marker in group):
                markers.extend(group)
    deduped: list[str] = []
    for marker in markers:
        if marker not in deduped:
            deduped.append(marker)
    return deduped


def _external_research_text_matches_focus(text: str, markers: list[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _external_research_fetch_requirement_reason(
    research: dict[str, Any],
    sources: list[Any],
    policy: dict[str, Any],
) -> str:
    requirements = policy.get("fetch_requirements")
    if not isinstance(requirements, list) or not requirements:
        return ""
    fetches = research.get("fetches")
    if not isinstance(fetches, list):
        fetches = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            continue
        url_prefix = str(requirement.get("url_prefix") or "").strip()
        required_header = str(requirement.get("required_header") or "User-Agent").strip()
        required_user_agent = str(requirement.get("required_user_agent") or "").strip()
        if not url_prefix or not required_user_agent:
            continue
        for source in sources:
            source_text = str(source).strip()
            if not source_text.startswith(url_prefix):
                continue
            matching_fetch = _external_research_matching_fetch(fetches, source_text)
            if not matching_fetch:
                return (
                    "external_research.fetches must record fetch_requirements "
                    f"for {url_prefix}"
                )
            headers = matching_fetch.get("headers")
            if not isinstance(headers, dict):
                return (
                    "external_research.fetches headers must include required "
                    f"{required_header} for {url_prefix}"
                )
            reported_header = _case_insensitive_header(headers, required_header)
            if not reported_header:
                return (
                    "external_research.fetches headers must include required "
                    f"{required_header} for {url_prefix}"
                )
            if reported_header.strip() != required_user_agent:
                return (
                    "external_research.fetches User-Agent for mp.weixin.qq.com "
                    "must match the packet required_user_agent"
                )
    return ""


def _external_research_matching_fetch(
    fetches: list[Any],
    source: str,
) -> dict[str, Any] | None:
    normalized_source = source.rstrip("/")
    for fetch in fetches:
        if not isinstance(fetch, dict):
            continue
        fetch_source = str(fetch.get("source") or "").strip().rstrip("/")
        if fetch_source == normalized_source:
            return fetch
    return None


def _case_insensitive_header(headers: dict[Any, Any], header_name: str) -> str:
    target = header_name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return str(value)
    return ""


def _report_categories_for_changed_files(changed_files: list[str]) -> set[str]:
    """Map actual changed-file layers to final-report component categories."""

    return set(report_contract.valid_primary_layers_for_changed_files(changed_files))


def _mission_candidate_mechanism_markers(mission_debug: dict[str, Any]) -> list[str]:
    """Mechanism signatures Codex may legitimately name as a problem class.

    Codex is rewarded for choosing a concrete, operable mechanism signature
    (e.g. ``missing_output_artifact_contract``) over a generic digest label
    (e.g. ``timeout``). Those signatures come from the mission feature
    candidates rather than the failure_pattern_digest, so a report that cites a
    candidate's mechanism must be treated as covering the evidence.
    """

    markers: list[str] = []
    for candidate in _mission_candidates(mission_debug or {}):
        for marker in _mission_candidate_markers(candidate):
            marker_text = marker.strip()
            if len(marker_text) >= 3:
                markers.append(marker_text)
    deduped: list[str] = []
    seen: set[str] = set()
    for marker in markers:
        key = marker.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(marker)
    return deduped


def _failure_pattern_markers(failure_pattern_digest: dict[str, Any]) -> list[str]:
    markers: list[str] = []
    dominant = failure_pattern_digest.get("dominant_pattern")
    if isinstance(dominant, dict):
        category = str(dominant.get("failure_category") or "").strip()
        if len(category) >= 3:
            markers.append(category)
        for component in dominant.get("affected_components") or []:
            component_text = str(component).strip()
            if len(component_text) >= 3:
                markers.append(component_text)
    dominant_mechanism = failure_pattern_digest.get("dominant_mechanism_pattern")
    if isinstance(dominant_mechanism, dict):
        signature = str(dominant_mechanism.get("signature") or "").strip()
        if len(signature) >= 3:
            markers.append(signature)
    dominant_weakness = failure_pattern_digest.get("dominant_weakness_signature")
    if isinstance(dominant_weakness, dict):
        signature = str(dominant_weakness.get("signature") or "").strip()
        if len(signature) >= 3:
            markers.append(signature)
    for pattern in failure_pattern_digest.get("patterns") or []:
        if not isinstance(pattern, dict):
            continue
        category = str(pattern.get("failure_category") or "").strip()
        if len(category) >= 3:
            markers.append(category)
        if len(markers) >= 6:
            break
    for pattern in failure_pattern_digest.get("mechanism_patterns") or []:
        if not isinstance(pattern, dict):
            continue
        signature = str(pattern.get("signature") or "").strip()
        if len(signature) >= 3:
            markers.append(signature)
        if len(markers) >= 10:
            break
    for pattern in failure_pattern_digest.get("weakness_signatures") or []:
        if not isinstance(pattern, dict):
            continue
        signature = str(pattern.get("signature") or "").strip()
        if len(signature) >= 3:
            markers.append(signature)
        if len(markers) >= 12:
            break
    return markers


def _rejected_update_buffer_markers(
    rejected_update_buffer: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    markers: list[str] = []
    for entry in rejected_update_buffer:
        if len(markers) >= limit:
            break
        for key in ("failure_class", "component_layer", "packet_id"):
            marker = str(entry.get(key) or "").strip()
            if len(marker) >= 3:
                markers.append(marker)
                break
    return markers


def _discouraged_direction_memory_entries(
    runner_pivot_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_entries = runner_pivot_policy.get("discouraged")
    if not isinstance(raw_entries, list):
        return []
    entries: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        failure_class = str(raw_entry.get("failure_class") or "").strip()
        component_layer = str(raw_entry.get("component_layer") or "").strip()
        mission_candidate_id = str(
            raw_entry.get("mission_candidate_id") or ""
        ).strip()
        if not failure_class and not component_layer and not mission_candidate_id:
            continue
        if mission_candidate_id:
            entries.append(
                {
                    "packet_id": mission_candidate_id,
                    "failure_class": "",
                    "component_layer": "",
                }
            )
            continue
        entries.append(
            {
                "packet_id": "",
                "failure_class": failure_class,
                "component_layer": component_layer,
            }
        )
    return entries


def _layer_pressure_memory_entries(
    runner_pivot_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_entries = runner_pivot_policy.get("layer_pressure")
    if not isinstance(raw_entries, list):
        return []
    entries: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        component_layer = str(raw_entry.get("component_layer") or "").strip()
        packet_ids = [
            str(packet_id).strip()
            for packet_id in raw_entry.get("recent_packet_ids") or []
            if str(packet_id).strip()
        ]
        if not component_layer and not packet_ids:
            continue
        entries.append(
            {
                "packet_id": " ".join(packet_ids),
                "failure_class": " ".join(
                    str(item).strip()
                    for item in raw_entry.get("failure_classes") or []
                    if str(item).strip()
                ),
                "component_layer": component_layer,
            }
        )
    return entries


def _prior_update_lesson_memory_entries(
    raw_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(raw_entries, list):
        return []
    entries: list[dict[str, Any]] = []
    actionable_outcomes = {
        "prediction_missed",
        "frontier_regression",
        "rollback_applied",
        "rollback_failed",
        "validation_failed",
    }
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        outcome = str(raw_entry.get("outcome") or "").strip()
        if outcome and outcome not in actionable_outcomes:
            continue
        packet_id = str(raw_entry.get("packet_id") or "").strip()
        mission_candidate_id = str(
            raw_entry.get("mission_candidate_id") or ""
        ).strip()
        source = str(raw_entry.get("source") or "").strip()
        if not packet_id and not outcome and not mission_candidate_id:
            continue
        entry = {
            "packet_id": packet_id,
            "failure_class": outcome,
            "component_layer": source,
        }
        if mission_candidate_id:
            entry["mission_candidate_id"] = mission_candidate_id
        entries.append(entry)
    return entries


def _supported_direction_memory_entries(
    runner_pivot_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_entries = runner_pivot_policy.get("supported")
    if not isinstance(raw_entries, list):
        return []
    entries: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        packet_id = str(raw_entry.get("packet_id") or "").strip()
        failure_class = str(raw_entry.get("failure_class") or "").strip()
        component_layer = str(raw_entry.get("component_layer") or "").strip()
        mission_candidate_id = str(
            raw_entry.get("mission_candidate_id") or ""
        ).strip()
        if mission_candidate_id:
            entries.append(
                {
                    "packet_id": mission_candidate_id,
                    "failure_class": "",
                    "component_layer": "",
                }
            )
            continue
        if not packet_id and not failure_class and not component_layer:
            continue
        entries.append(
            {
                "packet_id": packet_id,
                "failure_class": failure_class,
                "component_layer": component_layer,
            }
        )
    return entries


def _prediction_window_is_evaluable(window: str) -> bool:
    lowered = window.lower()
    markers = [
        "next comparable",
        "next summary",
        "campaign summary",
        "frontier",
        "regression",
        "rerun",
        "re-run",
        "held-out",
        "post-update",
        "same-model",
        "next iteration",
        "next round",
    ]
    return any(marker in lowered for marker in markers)


def _mission_candidates(mission_debug: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = mission_debug.get("feature_candidates")
    if not isinstance(raw_candidates, list):
        return []
    candidates = [candidate for candidate in raw_candidates if isinstance(candidate, dict)]
    return candidates


def _mission_selection_summary(
    mission_debug: dict[str, Any],
    final_report: dict[str, Any],
) -> dict[str, Any]:
    candidates = _mission_candidates(mission_debug)
    attributed_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("id") or "").startswith("mission-attributed-")
    ]
    selected_id = str(
        (mission_debug.get("evidence_summary") or {}).get("selected_candidate_id")
        or ""
    )
    selected = next(
        (
            candidate
            for candidate in attributed_candidates
            if str(candidate.get("id") or "") == selected_id
        ),
        {},
    )
    if not selected and len(attributed_candidates) == 1:
        selected = attributed_candidates[0]
    return {
        "enforced": bool(attributed_candidates),
        "available_candidate_ids": [
            str(candidate.get("id") or "") for candidate in candidates if candidate.get("id")
        ],
        "attributed_candidate_ids": [
            str(candidate.get("id") or "")
            for candidate in attributed_candidates
            if candidate.get("id")
        ],
        "selected_candidate_id": str(selected.get("id") or ""),
        "selected_failure_category": str(selected.get("failure_category") or ""),
        "selected_allowed_edit_paths": [
            str(path) for path in selected.get("allowed_edit_paths") or []
        ],
        "selected_target_tasks": [
            str(task_id) for task_id in selected.get("target_tasks") or []
        ],
    }


def _mission_candidate_markers(candidate: dict[str, Any]) -> list[str]:
    markers: list[str] = []
    for key in ("id", "failure_category"):
        value = str(candidate.get(key) or "").strip()
        if len(value) >= 3:
            markers.append(value)
    candidate_id = str(candidate.get("id") or "")
    if candidate_id.startswith("mission-attributed-"):
        markers.append(candidate_id.removeprefix("mission-attributed-").replace("-", "_"))
    return markers


def _matching_mission_candidates(
    candidates: list[dict[str, Any]],
    report_text: str,
) -> list[dict[str, Any]]:
    # Precise-id-first: if the report names a candidate id verbatim, that is the
    # deliberate selection.
    id_matches = [
        candidate
        for candidate in candidates
        if _text_entries_cover_any_marker(
            [report_text],
            [str(candidate.get("id") or "")],
        )
    ]
    if id_matches:
        return id_matches
    # Marker fallback: multiple candidates may share a mechanism signature, so a
    # report that names the mechanism (not a full id) would otherwise match them
    # all and deadlock ("must reference one" vs "matched multiple"). Dedupe to the
    # single most-specific match: the candidate whose longest matched marker is
    # longest wins; ties break deterministically by candidate id.
    marker_matches: list[tuple[int, str, dict[str, Any]]] = []
    for candidate in candidates:
        matched = [
            marker
            for marker in _mission_candidate_markers(candidate)
            if _text_entries_cover_any_marker([report_text], [marker])
        ]
        if matched:
            best_marker_len = max(len(marker) for marker in matched)
            marker_matches.append(
                (best_marker_len, str(candidate.get("id") or ""), candidate)
            )
    if not marker_matches:
        return []
    if len(marker_matches) == 1:
        return [marker_matches[0][2]]
    marker_matches.sort(key=lambda item: (-item[0], item[1]))
    return [marker_matches[0][2]]


def _mission_report_text(final_report: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("summary", "component_type"):
        parts.append(str(final_report.get(key) or ""))
    for key in (
        "cross_round_evidence",
        "prediction",
        "memory_record",
        "implementation_scope",
        "generalization",
    ):
        value = final_report.get(key)
        if isinstance(value, dict):
            parts.append(json.dumps(value, sort_keys=True))
    return "\n".join(parts)


def _path_is_under_allowed_roots(path: str, allowed_roots: list[str]) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    for root in allowed_roots:
        normalized_root = root.replace("\\", "/").strip("/")
        if not normalized_root:
            continue
        if normalized == normalized_root or normalized.startswith(normalized_root + "/"):
            return True
    return False


def _drop_pytest_verbosity(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens or tokens[0] != "pytest":
        return command
    filtered = [token for token in tokens if token not in {"-q", "-v", "-vv"}]
    return " ".join(filtered)
