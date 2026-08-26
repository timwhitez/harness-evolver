"""File-system MemoryStore implementation.

HL memory is explicit, readable, deletable, and refactorable —
not compressed into neural network weights.

Storage layout:
  trials/runs/<trial_id>/       — Full trial results
  trials/summaries/<id>.json    — Aggregated summaries
  trials/regressions/<model-scope>/<task>.json — Golden snapshots (validation contracts)
  trials/diffs/<from>_to_<to>/  — Per-component diffs between trials
"""

from __future__ import annotations

import json
from datetime import datetime
from glob import escape as glob_escape
from pathlib import Path
from typing import Any

from hl.attribution import FailureAttributor
from hl.model_scope import (
    model_config_from_trial,
    model_scope_from_trial,
    model_scope_matches,
    safe_model_scope_name,
)
from hl.protocol import MemoryStore
from hl.types import (
    FeedbackSignal,
    HarnessPatch,
    RegressionSnapshot,
    TrialResult,
    TrialSummary,
)


def _normalize_lesson_body(body: str) -> str:
    """Normalize a component-lesson body for deduplication.

    Collapses whitespace and lowercases so that the same insight logged under a
    different trial id or with trailing whitespace is treated as one entry.
    """
    return " ".join(str(body).split()).lower()


class FileSystemMemory(MemoryStore):
    """File-system based memory store.

    Every trial, summary, regression, and patch is stored as
    plain JSON — inspectable with any text editor, versionable
    in git, and easily diffable.
    """

    def __init__(self, base_path: str = "trials"):
        self.base_path = base_path
        self.runs_dir = Path(base_path) / "runs"
        self.summaries_dir = Path(base_path) / "summaries"
        self.analysis_dir = Path(base_path) / "analysis"
        self.regressions_dir = Path(base_path) / "regressions"
        self.diffs_dir = Path(base_path) / "diffs"
        self.component_lessons_dir = Path(base_path) / "memory" / "component_lessons"
        self.submissions_dir = Path(base_path) / "submissions"
        self.goals_dir = Path(base_path) / "goals"
        # Bounded retention for the itemized lesson playbook. Keeping only the
        # most-recent distinct lessons prevents unbounded append-only growth and
        # the brevity bias that appears when only the file tail is later injected.
        self.component_lesson_max_entries = 40

    def record_trial(self, trial: TrialResult, *, append_scoreboard: bool = True) -> str:
        trial_dir = self.runs_dir / trial.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Write full trial result
        (trial_dir / "result.json").write_text(
            trial.model_dump_json(indent=2)
        )

        # Write trajectory separately for easy parsing
        if trial.trajectory:
            (trial_dir / "trajectory.jsonl").write_text(
                "\n".join(json.dumps(e) for e in trial.trajectory)
            )

        # Write tool call log
        if trial.tool_calls:
            (trial_dir / "tool_calls.jsonl").write_text(
                "\n".join(json.dumps(tc) for tc in trial.tool_calls)
            )

        if trial.harbor_stdout:
            (trial_dir / "harbor_stdout.txt").write_text(trial.harbor_stdout)
        if trial.harbor_stderr:
            (trial_dir / "harbor_stderr.txt").write_text(trial.harbor_stderr)
        if trial.verifier_output:
            (trial_dir / "verifier_output.txt").write_text(trial.verifier_output)

        tool_success_rate = self._tool_success_rate(trial)
        attribution = FailureAttributor().analyze(
            trial,
            tool_success_rate=tool_success_rate,
        )
        feedback = FeedbackSignal(
            trial_id=trial.trial_id,
            task_id=trial.task_id,
            status=trial.status,
            score=trial.score,
            affected_components=attribution.affected_components,
            failure_category=attribution.failure_category,
            component_confidence=attribution.component_confidence,
            error_summary="\n".join(trial.error_log[:3]),
            tool_call_success_rate=tool_success_rate,
            trajectory_length=len(trial.trajectory),
            wall_time_seconds=trial.wall_time_seconds,
            raw_errors=trial.error_log,
        )
        (trial_dir / "feedback.json").write_text(feedback.model_dump_json(indent=2))

        snapshot = {
            "harness_version": trial.harness_version,
            "component_versions": {
                name: version.model_dump(mode="json")
                for name, version in trial.component_versions.items()
            },
            "model_used": trial.model_used,
            "token_usage": trial.token_usage,
            "metadata": trial.metadata,
        }
        (trial_dir / "harness_snapshot.json").write_text(json.dumps(snapshot, indent=2))

        handoff = self._build_handoff(trial, feedback)
        (trial_dir / "handoff.md").write_text(handoff)
        if append_scoreboard:
            self._append_scoreboard(trial)

        return trial.trial_id

    def get_trial(self, trial_id: str) -> TrialResult:
        result_path = self.runs_dir / trial_id / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Trial {trial_id} not found")
        return TrialResult.model_validate_json(result_path.read_text())

    def list_trials(self, task_id: str | None = None) -> list[str]:
        if not self.runs_dir.exists():
            return []
        if task_id:
            prefixed = self._list_trials_by_task_prefix(task_id)
            if prefixed:
                return prefixed
        trials: list[str] = []
        for trial_dir in self.runs_dir.iterdir():
            if trial_dir.is_dir():
                result_path = trial_dir / "result.json"
                if result_path.exists():
                    if task_id:
                        result = TrialResult.model_validate_json(result_path.read_text())
                        if result.task_id == task_id:
                            trials.append(trial_dir.name)
                    else:
                        trials.append(trial_dir.name)
        return sorted(trials)

    def task_historical_analysis_lessons(self, task_id: str) -> list[str]:
        """Return recent task-specific analysis lessons for Worker context.

        These entries are prompt-context compression only. They are not master,
        sub-agent, or Worker loop stop conditions, and they never change how many
        HL iterations, tasks, turns, attempts, or provider calls may run.
        """
        task_id = str(task_id or "").strip()
        if not task_id or not self.analysis_dir.exists():
            return []

        lessons: list[str] = []
        seen: set[str] = set()
        summary_paths = sorted(
            self.analysis_dir.glob("*/*/summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for summary_path in summary_paths:
            try:
                raw_summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw_summary, dict):
                continue
            for lesson in self._task_analysis_lessons_from_summary(
                task_id=task_id,
                summary=raw_summary,
                summary_path=summary_path,
            ):
                key = " ".join(lesson.split())
                if key in seen:
                    continue
                seen.add(key)
                lessons.append(lesson)
                # This caps injected prompt context size only; it is not a loop
                # stop condition and does not bound Worker/master execution.
                if len(lessons) >= 4:
                    return lessons
        return lessons

    def _task_analysis_lessons_from_summary(
        self,
        *,
        task_id: str,
        summary: dict[str, Any],
        summary_path: Path,
    ) -> list[str]:
        evidence_by_task = summary.get("trajectory_evidence")
        if not isinstance(evidence_by_task, dict):
            evidence_by_task = {}
        task_evidence = evidence_by_task.get(task_id)
        if not isinstance(task_evidence, dict):
            task_evidence = {}

        lessons: list[str] = []
        raw_weaknesses = summary.get("weakness_signatures")
        if isinstance(raw_weaknesses, list):
            for raw_entry in raw_weaknesses:
                if not isinstance(raw_entry, dict):
                    continue
                entry_task_ids = [str(item) for item in raw_entry.get("task_ids") or []]
                if task_id not in entry_task_ids:
                    continue
                lessons.append(
                    self._format_task_analysis_lesson(
                        task_id=task_id,
                        summary=summary,
                        summary_path=summary_path,
                        category=str(raw_entry.get("failure_category") or ""),
                        verifier_failure=str(raw_entry.get("verifier_failure") or ""),
                        agent_contribution=str(raw_entry.get("agent_contribution") or ""),
                        reusable_mechanism=str(raw_entry.get("reusable_mechanism") or ""),
                        task_evidence=task_evidence,
                    )
                )

        if lessons:
            return lessons

        raw_buckets = summary.get("failure_buckets")
        if not isinstance(raw_buckets, list):
            return []
        for raw_bucket in raw_buckets:
            if not isinstance(raw_bucket, dict):
                continue
            bucket_task_ids = [str(item) for item in raw_bucket.get("task_ids") or []]
            if task_id not in bucket_task_ids:
                continue
            lessons.append(
                self._format_task_analysis_lesson(
                    task_id=task_id,
                    summary=summary,
                    summary_path=summary_path,
                    category=str(raw_bucket.get("failure_category") or ""),
                    verifier_failure="",
                    agent_contribution=self._analysis_agent_contribution(task_evidence),
                    reusable_mechanism=self._analysis_reusable_mechanism(
                        category=str(raw_bucket.get("failure_category") or ""),
                        bucket=raw_bucket,
                        task_evidence=task_evidence,
                    ),
                    task_evidence=task_evidence,
                )
            )
        return lessons

    def _format_task_analysis_lesson(
        self,
        *,
        task_id: str,
        summary: dict[str, Any],
        summary_path: Path,
        category: str,
        verifier_failure: str,
        agent_contribution: str,
        reusable_mechanism: str,
        task_evidence: dict[str, Any],
    ) -> str:
        summary_id = str(summary.get("summary_id") or summary_path.parent.name)
        parts = [
            f"Historical analysis lesson from {summary_id} for {task_id}: "
            f"failure_category={category or 'unknown'}"
        ]
        if verifier_failure:
            parts.append(f"verifier_failure={self._tail_text(verifier_failure, 240)}")
        if agent_contribution:
            parts.append(
                f"agent_contribution={self._tail_text(agent_contribution, 240)}"
            )
        if reusable_mechanism:
            parts.append(
                f"reusable_mechanism={self._tail_text(reusable_mechanism, 240)}"
            )

        mechanism_details = self._analysis_mechanism_details(task_evidence, category)
        if mechanism_details:
            parts.append("Failure mechanism detail: " + mechanism_details[0])
        policy_summary = self._analysis_policy_summary(task_evidence)
        if policy_summary:
            parts.append("Observed policy recurrence: " + policy_summary)
        dependency_example = self._analysis_dependency_example(task_evidence)
        if dependency_example:
            parts.append(
                "Historical dependency/toolchain expansion before artifact focus: "
                + dependency_example
            )

        parts.append(
            "loop_stop_condition=false; time_round_token_limit_driven=false; "
            "this is not a master, sub-agent, or Worker loop stop condition."
        )
        return "; ".join(part.strip("; ") for part in parts if part)

    def _analysis_mechanism_details(
        self,
        task_evidence: dict[str, Any],
        category: str,
    ) -> list[str]:
        details: list[str] = []
        for raw_item in task_evidence.get("failure_mechanisms") or []:
            if not isinstance(raw_item, dict):
                continue
            name = str(raw_item.get("name") or "").strip()
            description = str(raw_item.get("description") or "").strip()
            evidence = str(raw_item.get("evidence") or "").strip()
            if not description and not evidence:
                continue
            detail = description or evidence
            if name == "structured_csv_table_contract" or category == name:
                detail = "structured_csv_table_contract: " + detail
            if evidence and evidence not in detail:
                detail = f"{detail} Evidence: {evidence}"
            details.append(self._tail_text(detail, 900))
        return details

    def _analysis_agent_contribution(self, task_evidence: dict[str, Any]) -> str:
        policy_counts = task_evidence.get("policy_counts")
        if isinstance(policy_counts, dict):
            ranked: list[tuple[str, int]] = []
            for raw_name, raw_count in policy_counts.items():
                name = str(raw_name).strip()
                if not name or name == "artifact_check_deliverable_progress":
                    continue
                try:
                    count = int(raw_count or 0)
                except (TypeError, ValueError):
                    count = 0
                if count > 0:
                    ranked.append((name, count))
            if ranked:
                name, count = sorted(ranked, key=lambda item: (-item[1], item[0]))[0]
                return f"policy:{name}:{count}"
        return ""

    def _analysis_reusable_mechanism(
        self,
        *,
        category: str,
        bucket: dict[str, Any],
        task_evidence: dict[str, Any],
    ) -> str:
        mechanisms = [
            str(item.get("name") or "").strip()
            for item in task_evidence.get("failure_mechanisms") or []
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
        if mechanisms:
            return "mechanism:" + "+".join(sorted(dict.fromkeys(mechanisms))[:4])
        components = [str(item) for item in bucket.get("affected_components") or [] if item]
        if components:
            return "components:" + "+".join(components[:4])
        return f"category:{category}" if category else ""

    def _analysis_policy_summary(self, task_evidence: dict[str, Any]) -> str:
        policy_counts = task_evidence.get("policy_counts")
        if not isinstance(policy_counts, dict):
            return ""
        ranked: list[tuple[str, int]] = []
        for raw_name, raw_count in policy_counts.items():
            name = str(raw_name).strip()
            if not name:
                continue
            try:
                count = int(raw_count or 0)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                ranked.append((name, count))
        return ", ".join(
            f"{name}={count}"
            for name, count in sorted(ranked, key=lambda item: (-item[1], item[0]))[:4]
        )

    def _analysis_dependency_example(self, task_evidence: dict[str, Any]) -> str:
        for raw_item in task_evidence.get("dependency_and_toolchain_evidence") or []:
            if not isinstance(raw_item, dict):
                continue
            command = str(raw_item.get("command") or "").strip()
            policies = str(raw_item.get("policies") or "").strip()
            if not command:
                continue
            detail = self._tail_text(command, 260)
            if policies:
                detail = f"{detail} policies={self._tail_text(policies, 120)}"
            return detail
        return ""

    def _tail_text(self, value: str, max_chars: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= max_chars:
            return text
        return "..." + text[-max_chars:]

    def _list_trials_by_task_prefix(self, task_id: str) -> list[str]:
        """Fast path for Harbor trial ids, which are written as <task>__<suffix>."""
        pattern = f"{glob_escape(task_id)}__*"
        trials: list[str] = []
        for trial_dir in self.runs_dir.glob(pattern):
            if not trial_dir.is_dir():
                continue
            result_path = trial_dir / "result.json"
            if not result_path.exists():
                continue
            result = TrialResult.model_validate_json(result_path.read_text())
            if result.task_id == task_id:
                trials.append(trial_dir.name)
        return sorted(trials)

    def record_summary(self, summary: TrialSummary) -> str:
        path = self.summaries_dir / f"{summary.summary_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary.model_dump_json(indent=2))
        return summary.summary_id

    def attach_codex_update(
        self,
        trial_id: str,
        *,
        packet_path: str | Path | None = None,
        events_path: str | Path | None = None,
        final_message_path: str | Path | None = None,
        diff_path: str | Path | None = None,
        record_path: str | Path | None = None,
        summary_path: str | Path | None = None,
        review: Any | None = None,
    ) -> None:
        """Attach Codex updater artifacts to a trial memory directory.

        The canonical Codex run remains under ``trials/diffs/``; each affected
        failed trial also gets local copies/pointers so a single trial directory
        is sufficient handoff context for the next HL iteration.
        """
        trial_dir = self.runs_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        copied = {
            "codex_update_packet.json": self._copy_text_artifact(
                packet_path,
                trial_dir / "codex_update_packet.json",
            ),
            "codex_events.jsonl": self._copy_text_artifact(
                events_path,
                trial_dir / "codex_events.jsonl",
            ),
            "codex_final_message.json": self._copy_text_artifact(
                final_message_path,
                trial_dir / "codex_final_message.json",
            ),
            "codex_git.diff": self._copy_text_artifact(
                diff_path,
                trial_dir / "codex_git.diff",
            ),
            "codex_update_record.json": self._copy_text_artifact(
                record_path,
                trial_dir / "codex_update_record.json",
            ),
            "codex_update_summary.md": self._copy_text_artifact(
                summary_path,
                trial_dir / "codex_update_summary.md",
            ),
        }
        manifest = {
            "trial_id": trial_id,
            "sources": {
                "packet_path": str(packet_path) if packet_path else "",
                "events_path": str(events_path) if events_path else "",
                "final_message_path": str(final_message_path) if final_message_path else "",
                "diff_path": str(diff_path) if diff_path else "",
                "record_path": str(record_path) if record_path else "",
                "summary_path": str(summary_path) if summary_path else "",
            },
            "copied": copied,
            "review": self._review_payload(review),
        }
        (trial_dir / "codex_update_manifest.json").write_text(json.dumps(manifest, indent=2))

    def latest_change_manifest(self) -> dict[str, Any]:
        """Return the newest accepted Codex change manifest, if one exists."""
        if not self.diffs_dir.exists():
            return {}
        candidates = sorted(
            self.diffs_dir.glob("codex_packet_*/change_manifest.json"),
            key=lambda path: path.stat().st_mtime,
        )
        for path in reversed(candidates):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and data.get("accepted"):
                data.setdefault("path", str(path))
                return data
        return {}

    def get_latest_summary(self) -> TrialSummary | None:
        if not self.summaries_dir.exists():
            return None
        files = sorted(self.summaries_dir.glob("*.json"), reverse=True)
        for path in files:
            try:
                return TrialSummary.model_validate_json(path.read_text())
            except Exception:
                continue
        return None

    def save_regression(self, task_id: str, snapshot: RegressionSnapshot) -> None:
        path = self._regression_path(task_id, model_scope=snapshot.model_scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(snapshot.model_dump_json(indent=2))

    def mark_regression_stable(
        self,
        task_id: str,
        *,
        source_summary_id: str = "",
        model_scope: str = "",
    ) -> bool:
        snapshot = self.get_regression_snapshot(task_id, model_scope=model_scope)
        if snapshot is None:
            return False
        if source_summary_id and snapshot.source_summary_id != source_summary_id:
            return False
        snapshot.validation_status = "stable"
        snapshot.invalidation_reason = ""
        self.save_regression(task_id, snapshot)
        return True

    def invalidate_regression(
        self,
        task_id: str,
        *,
        source_summary_id: str = "",
        reason: str = "",
        model_scope: str = "",
    ) -> bool:
        snapshot = self.get_regression_snapshot(task_id, model_scope=model_scope)
        if snapshot is None:
            return False
        if source_summary_id and snapshot.source_summary_id != source_summary_id:
            return False
        snapshot.validation_status = "invalidated"
        snapshot.invalidation_reason = reason
        self.save_regression(task_id, snapshot)
        return True

    def record_regression_run(
        self,
        task_id: str,
        result: TrialResult,
        *,
        model_scope: str = "",
    ) -> bool:
        snapshot = self.get_regression_snapshot(
            task_id,
            model_scope=model_scope or model_scope_from_trial(result),
        )
        if snapshot is None:
            return False
        snapshot.regression_runs += 1
        snapshot.last_regression_status = result.status.value
        snapshot.last_regression_at = datetime.now()
        snapshot.last_regression_wall_time_seconds = result.wall_time_seconds
        if self.snapshot_validation_failed(snapshot, result):
            snapshot.regression_failures += 1
        self.save_regression(task_id, snapshot)
        return True

    def record_regression_transient_failure(
        self,
        task_id: str,
        *,
        model_scope: str = "",
        reason: str = "",
        cooldown_seconds: int = 0,
    ) -> bool:
        snapshot = self.get_regression_snapshot(task_id, model_scope=model_scope)
        if snapshot is None:
            return False
        snapshot.regression_transient_failures += 1
        snapshot.regression_cooldown_reason = reason
        if cooldown_seconds > 0:
            from datetime import timedelta

            snapshot.regression_cooldown_until = datetime.now() + timedelta(
                seconds=cooldown_seconds,
            )
        self.save_regression(task_id, snapshot)
        return True

    def check_regression(self, task_id: str, result: TrialResult) -> bool:
        snapshot = self.get_regression_snapshot(
            task_id,
            model_scope=model_scope_from_trial(result),
        )
        if snapshot is None:
            return False  # No regression snapshot = nothing to regress from
        if snapshot.validation_status != "stable":
            return False

        # Harbor may still run the verifier after an agent-side timeout. If the
        # official verifier reward is a full pass, this is completion hygiene
        # evidence, not a solved-task score regression.
        if result.verified and result.score >= 1.0:
            return False

        # Regression if previously-solved task now fails
        if result.status.value not in ("passed",):
            return True

        # Regression if score dropped significantly
        if result.score < 0.5:  # Binary pass/fail for TerminalBench
            return True

        return False

    def snapshot_validation_failed(
        self,
        snapshot: RegressionSnapshot,
        result: TrialResult,
    ) -> bool:
        """Fail closed for the exact snapshot selected by a validation gate.

        Unlike :meth:`check_regression`, this contract also applies to pending
        snapshots. A pending snapshot is not yet historical solved-task memory,
        but its post-update Harbor rerun must still produce a verified full pass
        before the campaign may promote it to stable.
        """

        if result.task_id != snapshot.task_id:
            return True
        if not model_scope_matches(
            snapshot.model_scope,
            model_scope_from_trial(result),
        ):
            return True
        return not (result.verified and result.score >= 1.0)

    def save_patch(self, patch: HarnessPatch) -> str:
        safe_name = patch.component_name.replace("/", "_")
        patch_dir = self.diffs_dir / f"patch_{safe_name}_{patch.after_version}"
        patch_dir.mkdir(parents=True, exist_ok=True)

        (patch_dir / "patch.json").write_text(
            patch.model_dump_json(indent=2)
        )
        diff_path = patch_dir / f"{safe_name}.diff"
        diff_path.write_text(patch.diff)

        return patch_dir.name

    def list_patches(self, component_name: str | None = None) -> list[str]:
        if not self.diffs_dir.exists():
            return []
        patches: list[str] = []
        for patch_dir in self.diffs_dir.iterdir():
            if patch_dir.is_dir():
                patch_path = patch_dir / "patch.json"
                if patch_path.exists():
                    if component_name:
                        patch = HarnessPatch.model_validate_json(patch_path.read_text())
                        if patch.component_name == component_name:
                            patches.append(patch_dir.name)
                    else:
                        patches.append(patch_dir.name)
        return sorted(patches)

    def get_regression_snapshot(
        self,
        task_id: str,
        *,
        model_scope: str = "",
    ) -> RegressionSnapshot | None:
        path = self._regression_path(task_id, model_scope=model_scope)
        if not path.exists():
            if model_scope:
                legacy = self.get_regression_snapshot(task_id)
                if legacy is None:
                    return None
                return (
                    legacy
                    if model_scope_matches(legacy.model_scope, model_scope)
                    else None
                )
            legacy_path = self._legacy_regression_path(task_id)
            if not legacy_path.exists():
                return None
            return self._with_model_scope_from_source_trial(
                RegressionSnapshot.model_validate_json(legacy_path.read_text())
            )
        return self._with_model_scope_from_source_trial(
            RegressionSnapshot.model_validate_json(path.read_text())
        )

    def list_regression_snapshots(
        self,
        *,
        model_scope: str = "",
    ) -> list[RegressionSnapshot]:
        if not self.regressions_dir.exists():
            return []
        snapshots: list[RegressionSnapshot] = []
        for path in sorted(self.regressions_dir.rglob("*.json")):
            try:
                snapshot = RegressionSnapshot.model_validate_json(path.read_text())
            except Exception:
                continue
            snapshot = self._with_model_scope_from_source_trial(snapshot)
            if model_scope_matches(snapshot.model_scope, model_scope):
                snapshots.append(snapshot)
        return snapshots

    def _with_model_scope_from_source_trial(
        self,
        snapshot: RegressionSnapshot,
    ) -> RegressionSnapshot:
        if snapshot.model_scope or not snapshot.source_trial_id:
            return snapshot
        try:
            source_trial = self.get_trial(snapshot.source_trial_id)
        except FileNotFoundError:
            return snapshot
        snapshot.model_scope = model_scope_from_trial(source_trial)
        snapshot.scope_config = model_config_from_trial(source_trial)
        return snapshot

    def _regression_path(self, task_id: str, *, model_scope: str = "") -> Path:
        safe_name = self._safe_task_name(task_id)
        if model_scope:
            return (
                self.regressions_dir
                / safe_model_scope_name(model_scope)
                / f"{safe_name}.json"
            )
        return self._legacy_regression_path(task_id)

    def _legacy_regression_path(self, task_id: str) -> Path:
        return self.regressions_dir / f"{self._safe_task_name(task_id)}.json"

    def _safe_task_name(self, task_id: str) -> str:
        return task_id.replace("/", "_").replace("::", "_")

    def save_component_lesson(
        self,
        component_name: str,
        lesson: str,
        *,
        source_trial_id: str | None = None,
    ) -> Path:
        safe_name = component_name.replace("/", "_")
        path = self.component_lessons_dir / f"{safe_name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = f"## {datetime.now().isoformat(timespec='seconds')}"
        if source_trial_id:
            prefix += f" — {source_trial_id}"
        new_body = lesson.strip()
        # ACE-style itemized playbook: keep each distinct lesson body once and
        # refresh recurrence by moving it to the most-recent position instead of
        # accumulating near-duplicate append-only blocks (which cause context
        # collapse and brevity bias when only the file tail is later injected).
        header = f"# {component_name}"
        existing_blocks = self._parse_component_lesson_blocks(path)
        new_key = _normalize_lesson_body(new_body)
        kept_blocks = [
            block
            for block in existing_blocks
            if _normalize_lesson_body(block["body"]) != new_key
        ]
        kept_blocks.append({"prefix": prefix, "body": new_body})
        # Bounded retention: keep only the most-recent distinct lessons so the
        # playbook does not grow without bound across long campaigns.
        max_entries = getattr(self, "component_lesson_max_entries", 0)
        if max_entries and len(kept_blocks) > max_entries:
            kept_blocks = kept_blocks[-max_entries:]
        rendered = [header]
        for block in kept_blocks:
            rendered.append("")
            rendered.append(block["prefix"])
            rendered.append("")
            rendered.append(block["body"])
        path.write_text("\n".join(rendered).rstrip() + "\n")
        return path

    def _parse_component_lesson_blocks(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        text = path.read_text()
        blocks: list[dict[str, str]] = []
        current_prefix: str | None = None
        current_body: list[str] = []
        for line in text.splitlines():
            if line.startswith("## "):
                if current_prefix is not None:
                    blocks.append(
                        {"prefix": current_prefix, "body": "\n".join(current_body).strip()}
                    )
                current_prefix = line
                current_body = []
            elif current_prefix is not None:
                current_body.append(line)
        if current_prefix is not None:
            blocks.append(
                {"prefix": current_prefix, "body": "\n".join(current_body).strip()}
            )
        return [block for block in blocks if block["body"]]

    def _tool_success_rate(self, trial: TrialResult) -> float:
        if not trial.tool_calls:
            return 1.0
        successes = sum(1 for call in trial.tool_calls if call.get("success"))
        return successes / len(trial.tool_calls)

    def _build_handoff(self, trial: TrialResult, feedback: FeedbackSignal) -> str:
        return "\n".join(
            [
                f"# Trial {trial.trial_id}",
                "",
                f"- Task: {trial.task_id}",
                f"- Status: {trial.status.value}",
                f"- Score: {trial.score}",
                f"- Verified: {trial.verified}",
                f"- Model: {trial.model_used or 'unknown'}",
                f"- Failure category: {feedback.failure_category or 'unknown'}",
                "- Affected components: "
                + (", ".join(feedback.affected_components) or "none"),
                f"- Tool success rate: {feedback.tool_call_success_rate:.2f}",
                f"- Harbor job: {trial.harbor_job_dir or 'not recorded'}",
                "",
                "## Errors",
                "\n".join(f"- {err}" for err in trial.error_log) or "- none",
            ]
        )

    def _append_scoreboard(self, trial: TrialResult) -> None:
        path = self.summaries_dir / "scoreboard.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "timestamp,trial_id,task_id,status,score,verified,model,wall_time_seconds\n"
        if not path.exists():
            path.write_text(header)
        row = ",".join(
            [
                trial.timestamp.isoformat(),
                trial.trial_id,
                trial.task_id,
                trial.status.value,
                str(trial.score),
                str(trial.verified).lower(),
                trial.model_used,
                f"{trial.wall_time_seconds:.3f}",
            ]
        )
        with path.open("a") as f:
            f.write(row + "\n")

    def _copy_text_artifact(self, source: str | Path | None, destination: Path) -> bool:
        if not source:
            destination.write_text("")
            return False
        source_path = Path(source)
        if not source_path.exists():
            destination.write_text("")
            return False
        destination.write_text(source_path.read_text(errors="replace"))
        return True

    def _review_payload(self, review: Any | None) -> dict[str, Any]:
        if review is None:
            return {}
        return {
            "accepted": getattr(review, "accepted", None),
            "reasons": list(getattr(review, "reasons", []) or []),
            "changed_files": list(getattr(review, "changed_files", []) or []),
        }
