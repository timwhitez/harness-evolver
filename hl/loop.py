"""HLLoop — the outer optimization loop.

This is the heart of Heuristic Learning.  The loop:

  1. Run → Execute tasks with current harness
  2. Collect → Extract feedback from results
  3. Analyze → Meta-agent identifies root causes
  4. Edit → Meta-agent edits harness components
  5. Verify → Regression check on previously-solved tasks
  6. Record → Write results, patches, summaries to memory
  7. Repeat → Next iteration

This maps directly to the HL loop from the paper:

  环境反馈 / 测试失败 / 日志异常
  -> coding agent 读 context
  -> 修改 policy / test / memory
  -> 重新运行
  -> 把结果写回 trials 和 summary
  -> 下一轮继续
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.scoring import Scoring
from hl.attribution import FailureAttributor
from hl.system import HeuristicSystem
from hl.coupling import CouplingTracker
from hl.compression import CompressionEngine
from hl.goals import (
    GoalStore,
    is_explicit_goal_completion_reason,
    is_explicit_goal_stop_reason,
)
from hl.memory import FileSystemMemory
from hl.model_scope import model_config_from_trial, model_scope_from_trial
from hl.types import (
    FeedbackSignal,
    RegressionSnapshot,
    TrialResult,
    TrialStatus,
    TrialSummary,
    trial_is_infrastructure_failure,
)


@dataclass
class HLLoop:
    """The Heuristic Learning optimization loop.

    Orchestrates the full cycle: run tasks, collect feedback,
    trigger meta-agent analysis, apply improvements, verify,
    and repeat.
    """

    system: HeuristicSystem = field(default_factory=HeuristicSystem)
    memory: FileSystemMemory = field(default_factory=FileSystemMemory)
    coupling: CouplingTracker = field(default_factory=CouplingTracker)
    goal_store: GoalStore = field(default_factory=GoalStore)
    compression: CompressionEngine = field(default_factory=CompressionEngine)

    # Loop progress policy. These fields are kept as compatibility and
    # reporting/audit inputs only; they are not stop conditions for the master
    # HL loop.
    max_iterations: int | None = None
    min_improvement: float = 0.01
    patience: int | None = None

    # State
    iteration: int = 0
    best_score: float = 0.0
    iterations_without_improvement: int = 0
    history: list[TrialSummary] = field(default_factory=list)
    pending_update_baseline_score: float | None = None
    pending_update_baseline_tasks: set[str] = field(default_factory=set)
    pending_update_baseline_summary_id: str = ""

    # Held-out (D_out) split for the Self-Harness acceptance gate. When
    # holdout_fraction > 0, held-out solved-task snapshots are hidden from the
    # Codex proposer's regression contracts so a held-out regression is a genuine
    # anti-overfit signal rather than something the proposer optimized against.
    regression_holdout_fraction: float = 0.0
    regression_holdout_seed: int = 0

    def __post_init__(self):
        self.system.set_memory(self.memory)

    def run_iteration(
        self,
        tasks: list[str],
        task_instructions: dict[str, str],
        task_contexts: dict[str, dict[str, Any]],
        agent_runner,  # HLAgent or HarborRunner
        on_trial_recorded=None,
        task_concurrency: int = 1,
        update_decision=None,
        pre_update_hook=None,
        stop_after_trial=None,
        required_validation_commands: list[str] | None = None,
    ) -> TrialSummary:
        """Execute one full iteration of the HL loop.

        Returns a TrialSummary for this iteration.
        """
        if task_concurrency < 1:
            raise ValueError("task_concurrency must be positive")
        self.iteration += 1
        summary_id = f"summary_{self.iteration:03d}"

        # ── 1. Run ──
        run_specs: list[tuple[int, str, str, dict[str, Any]]] = []
        for index, task_id in enumerate(tasks):
            instruction = task_instructions.get(task_id, "")
            context = dict(task_contexts.get(task_id, {}))
            previous_errors = self._context_previous_errors(context)

            # Inject task-specific lessons mined from prior campaign analysis.
            # This is context transfer only, not a master/sub-agent/Worker loop
            # stop condition or a time/round/turn/attempt cap.
            if hasattr(self.memory, "task_historical_analysis_lessons"):
                previous_errors.extend(
                    self.memory.task_historical_analysis_lessons(task_id)
                )

            # Inject previous errors into context (HL feedback)
            previous_trials = self.memory.list_trials(task_id)
            if previous_trials:
                last_trial = self.memory.get_trial(previous_trials[-1])
                if last_trial.status != TrialStatus.PASSED:
                    previous_errors.extend(last_trial.error_log)
            if previous_errors:
                context["previous_errors"] = previous_errors
            run_specs.append((index, task_id, instruction, context))

        task_run_started = time.monotonic()
        results = self._run_iteration_tasks(
            run_specs=run_specs,
            agent_runner=agent_runner,
            task_concurrency=task_concurrency,
            on_trial_recorded=on_trial_recorded,
            summary_id=summary_id,
            stop_after_trial=stop_after_trial,
        )
        task_run_wall_time_seconds = max(0.0, time.monotonic() - task_run_started)

        # ── 2. Collect Feedback ──
        feedback_signals: list[FeedbackSignal] = []
        for channel in self.system.feedback_channels.values():
            for trial in results:
                signal = channel.collect(trial)
                feedback_signals.append(signal)
        if not feedback_signals:
            feedback_signals = [self._feedback_from_trial(trial) for trial in results]

        # ── 3. Build Summary ──
        summary = Scoring.build_summary(summary_id, results)
        self.memory.record_summary(summary)
        self.history.append(summary)
        self._rollback_pending_update_if_worse(summary, results)

        # ── 4. Analyze & Edit (if there are failures) ──
        failures = [s for s in feedback_signals if s.status != TrialStatus.PASSED]
        failed_trials = [t for t in results if t.status != TrialStatus.PASSED]
        accepted_update_count = 0
        diagnostic_trials: list[TrialResult] = []
        diagnostic_wall_time_seconds = 0.0
        self._absorb_repeated_failure_lessons(failures)

        if failed_trials and self.system.update_engine:
            if update_decision is not None and not update_decision(
                summary=summary,
                failed_trials=failed_trials,
            ):
                self.memory.save_component_lesson(
                    "codex_update",
                    (
                        "Skipped Codex update for this iteration because the "
                        "campaign update policy requested more evidence before "
                        "editing the harness."
                    ),
                )
            elif hasattr(self.system.update_engine, "run_update"):
                if pre_update_hook is not None:
                    diagnostic_started = time.monotonic()
                    diagnostic_trials = list(
                        pre_update_hook(
                            summary=summary,
                            failed_trials=failed_trials,
                            agent_runner=agent_runner,
                            task_instructions=task_instructions,
                            task_contexts=task_contexts,
                        )
                        or []
                    )
                    diagnostic_wall_time_seconds = max(
                        0.0,
                        time.monotonic() - diagnostic_started,
                    )
                    for diagnostic in diagnostic_trials:
                        diagnostic.metadata.setdefault("diagnostic", True)
                        self.memory.record_trial(diagnostic)
                        if on_trial_recorded is not None:
                            on_trial_recorded(
                                trial=diagnostic,
                                completed_results=list(results),
                                iteration=self.iteration,
                                summary_id=summary_id,
                                diagnostic=True,
                            )
                        self.system.trial_count += 1
                update_result = self.system.update_engine.run_update(
                    failures=[*failed_trials, *diagnostic_trials],
                    current_harness=self.system.summary(),
                    regression_contracts=self._regression_contracts(),
                    required_validation_commands=(
                        required_validation_commands
                        if required_validation_commands is not None
                        else self._required_validation_commands()
                    ),
                )
                self._attach_codex_update_artifacts(failed_trials, update_result)
                if update_result.review.accepted:
                    accepted_update_count += 1
                    self.system.patch_count += 1
                    summary.patches_applied.append("codex_update:accepted")
                    self._track_pending_update_baseline(summary, results)
                    self._save_regression_snapshots(results, summary.summary_id)
                else:
                    if update_result.review.changed_files and hasattr(
                        self.system.update_engine,
                        "rollback_last",
                    ):
                        self.system.update_engine.rollback_last()
                    self.memory.save_component_lesson(
                        "codex_update",
                        "Rejected Codex update: "
                        + "; ".join(update_result.review.reasons),
                    )
            else:
                findings = self.system.update_engine.analyze_failures(
                    feedback_signals=failures,
                    trials=failed_trials,
                )
                self._record_legacy_update_engine_analysis_only(
                    findings=findings,
                    failed_trials=failed_trials,
                )

        # Persist the final summary after update/review decisions so campaign
        # reports and on-disk memory carry the actual patch lineage.
        self.memory.record_summary(summary)

        # ── 5. Check for improvement ──
        if summary.overall_score > self.best_score + self.min_improvement:
            self.best_score = summary.overall_score
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1

        # ── 6. Check coupling complexity ──
        if self.coupling.needs_compression() or self.system.needs_compression():
            # Signal that compression is needed (meta-agent handles this)
            self._flag_compression_needed()

        self.goal_store.update_usage(
            worker_tokens=self._sum_worker_tokens([*results, *diagnostic_trials]),
            harbor_wall_time_seconds=(
                task_run_wall_time_seconds + diagnostic_wall_time_seconds
            ),
            patch_count=accepted_update_count,
            best_score=self.best_score,
        )

        return summary

    def _context_previous_errors(self, context: dict[str, Any]) -> list[str]:
        raw_errors = context.get("previous_errors")
        if raw_errors is None:
            return []
        if isinstance(raw_errors, list):
            return [str(item) for item in raw_errors]
        return [str(raw_errors)]

    def _run_iteration_tasks(
        self,
        *,
        run_specs: list[tuple[int, str, str, dict[str, Any]]],
        agent_runner: Any,
        task_concurrency: int,
        on_trial_recorded: Any,
        summary_id: str,
        stop_after_trial: Any = None,
    ) -> list[TrialResult]:
        if not run_specs:
            return []
        if task_concurrency <= 1 or len(run_specs) == 1:
            results: list[TrialResult] = []
            for _index, task_id, instruction, context in run_specs:
                trial = agent_runner.run(instruction, context)
                self._record_iteration_trial(
                    trial=trial,
                    task_id=task_id,
                    completed_results=results,
                    on_trial_recorded=on_trial_recorded,
                    summary_id=summary_id,
                    stop_after_trial=stop_after_trial,
                )
            return results

        max_workers = min(task_concurrency, len(run_specs))
        ordered: list[TrialResult | None] = [None] * len(run_specs)
        completed_results: list[TrialResult] = []

        def record_done_future(
            future: Any,
            index: int,
            task_id: str,
            *,
            suppress_worker_errors: bool = False,
        ) -> None:
            if ordered[index] is not None:
                return
            try:
                trial = future.result()
            except Exception:
                if suppress_worker_errors:
                    return
                raise
            ordered[index] = trial
            self._record_iteration_trial(
                trial=trial,
                task_id=task_id,
                completed_results=completed_results,
                on_trial_recorded=on_trial_recorded,
                summary_id=summary_id,
                stop_after_trial=stop_after_trial,
            )

        def drain_completed_futures(futures: dict[Any, tuple[int, str]]) -> None:
            for future in list(futures):
                if not future.done():
                    continue
                index, task_id = futures.pop(future)
                record_done_future(future, index, task_id)

        def drain_all_futures(futures: dict[Any, tuple[int, str]]) -> None:
            for future, (index, task_id) in list(futures.items()):
                futures.pop(future, None)
                record_done_future(
                    future,
                    index,
                    task_id,
                    suppress_worker_errors=True,
                )

        next_index = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Any, tuple[int, str]] = {}

            def submit_next() -> None:
                nonlocal next_index
                if next_index >= len(run_specs):
                    return
                index, task_id, instruction, context = run_specs[next_index]
                futures[executor.submit(agent_runner.run, instruction, context)] = (
                    index,
                    task_id,
                )
                next_index += 1

            for _ in range(max_workers):
                submit_next()

            while futures:
                try:
                    done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                except BaseException:
                    # Preserve completed task evidence before propagating the
                    # interrupt/error. ThreadPoolExecutor still lets already
                    # running Harbor jobs finish during shutdown; without this
                    # drain their result files exist but campaign state does not
                    # know about them, making resume skip nothing.
                    drain_all_futures(futures)
                    raise
                for future in done:
                    index, task_id = futures.pop(future)
                    record_done_future(future, index, task_id)

                while (
                    next_index < len(run_specs)
                    and len(futures) < max_workers
                ):
                    submit_next()

        return [trial for trial in ordered if trial is not None]

    def _record_iteration_trial(
        self,
        *,
        trial: TrialResult,
        task_id: str,
        completed_results: list[TrialResult],
        on_trial_recorded: Any,
        summary_id: str,
        stop_after_trial: Any = None,
    ) -> None:
        self.memory.record_trial(trial)
        completed_results.append(trial)
        if on_trial_recorded is not None:
            on_trial_recorded(
                trial=trial,
                completed_results=list(completed_results),
                iteration=self.iteration,
                summary_id=summary_id,
            )
        self.system.trial_count += 1

        if trial.status == TrialStatus.PASSED and trial.verified:
            self.system.record_solved(task_id)
        if stop_after_trial is not None:
            requested_stop = bool(
                stop_after_trial(
                    trial=trial,
                    completed_results=list(completed_results),
                    iteration=self.iteration,
                    summary_id=summary_id,
                )
            )
            if requested_stop:
                self.memory.save_component_lesson(
                    "loop_limit_contract",
                    (
                        "A stop_after_trial hook requested a stop after "
                        f"{trial.trial_id}, but trial-count and per-round stop "
                        "requests are audit-only. The HL master loop continues "
                        "submitting the remaining tasks in the iteration."
                    ),
                    source_trial_id=trial.trial_id,
                )

    def _flag_compression_needed(self) -> None:
        """Flag that the system needs compression.

        From the paper: "An HS that only grows and never compresses
        will eventually become a big ball of mud."
        """
        plan = self.compression.dry_run()
        if plan.apply and self.memory is not None:
            self.memory.save_component_lesson(
                "compression",
                self.compression.build_codex_instruction(plan),
            )

    def _absorb_repeated_failure_lessons(self, failures: list[FeedbackSignal]) -> None:
        grouped: dict[tuple[str, str], list[FeedbackSignal]] = {}
        for signal in failures:
            category = signal.failure_category or "unknown"
            components = signal.affected_components or ["prompts/system"]
            for component in components[:3]:
                grouped.setdefault((category, component), []).append(signal)

        for (category, component), signals in grouped.items():
            if len(signals) < 2:
                continue
            task_ids = ", ".join(sorted({signal.task_id for signal in signals}))
            trial_ids = ", ".join(signal.trial_id for signal in signals)
            lesson = (
                f"Repeated failure category `{category}` was observed across "
                f"{len(signals)} trial(s). Candidate component: `{component}`. "
                f"Tasks: {task_ids}. Trials: {trial_ids}. "
                "Treat this as attribution evidence, not proof; verify with "
                "trajectory slices and regression before editing the harness."
            )
            self.memory.save_component_lesson(
                component,
                lesson,
                source_trial_id=signals[-1].trial_id,
            )

    def _record_legacy_update_engine_analysis_only(
        self,
        *,
        findings: list[dict[str, Any]],
        failed_trials: list[TrialResult],
    ) -> None:
        """Record legacy updater analysis without allowing direct harness edits.

        The production HL updater is Codex-backed and must edit through
        ``run_update`` so packet evidence, diff review, host validation, and
        rollback metadata stay intact. Older analyzer scaffolds may still be
        useful for deterministic tests, but they must not become a parallel
        patching path.
        """

        update_engine = self.system.update_engine
        engine_name = str(getattr(update_engine, "name", "") or type(update_engine).__name__)
        task_ids = sorted({trial.task_id for trial in failed_trials})
        components: list[str] = []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            for component in finding.get("affected_components") or []:
                component_text = str(component).strip()
                if component_text:
                    components.append(component_text)
        component_text = ", ".join(sorted(dict.fromkeys(components))[:8]) or "none"
        self.memory.save_component_lesson(
            "codex_update",
            (
                f"Legacy update engine `{engine_name}` produced analysis for "
                f"{len(failed_trials)} failed trial(s): {', '.join(task_ids) or 'none'}. "
                f"Affected components from analysis: {component_text}. "
                "The analysis was recorded as diagnostic evidence only; direct "
                "suggest_edits/apply_patch mutation is disabled so harness edits "
                "must flow through the Codex-backed run_update packet, diff review, "
                "host validation, and rollback path."
            ),
            source_trial_id=failed_trials[-1].trial_id if failed_trials else None,
        )

    def should_continue(self) -> bool:
        """Return whether a non-limit terminal condition has occurred."""
        goal = self.goal_store.get_goal()
        if (
            goal is not None
            and goal.status == "complete"
            and is_explicit_goal_completion_reason(goal.completion_reason)
        ):
            return False
        if (
            goal is not None
            and goal.status == "stopped"
            and is_explicit_goal_stop_reason(goal.completion_reason)
        ):
            return False
        return True

    def get_progress(self) -> dict[str, Any]:
        """Return current progress for monitoring (Mission Control)."""
        return {
            "iteration": self.iteration,
            "best_score": self.best_score,
            "iterations_without_improvement": self.iterations_without_improvement,
            "total_trials": self.system.trial_count,
            "total_patches": self.system.patch_count,
            "solved_tasks": len(self.system.solved_tasks),
            "coupling": self.coupling.summary(),
            "goal": (
                self.goal_store.get_goal().model_dump(mode="json")
                if self.goal_store.get_goal()
                else None
            ),
            "latest_summary": self.history[-1].model_dump() if self.history else None,
        }

    def _sum_worker_tokens(self, results: list[TrialResult]) -> dict[str, int]:
        total = {"input": 0, "output": 0, "cache": 0}
        for result in results:
            for key in total:
                total[key] += int(result.token_usage.get(key, 0))
        return total

    def _feedback_from_trial(self, trial: TrialResult) -> FeedbackSignal:
        tool_success_rate = self._tool_success_rate(trial)
        attribution = FailureAttributor().analyze(
            trial,
            tool_success_rate=tool_success_rate,
        )
        return FeedbackSignal(
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

    def _tool_success_rate(self, trial: TrialResult) -> float:
        if not trial.tool_calls:
            return 1.0
        successes = sum(1 for call in trial.tool_calls if call.get("success"))
        return successes / len(trial.tool_calls)

    def _save_regression_snapshots(
        self,
        results: list[TrialResult],
        summary_id: str,
    ) -> None:
        for trial in results:
            if (
                trial.status != TrialStatus.PASSED
                or not trial.verified
                or trial.score < 1.0
            ):
                continue
            snapshot = RegressionSnapshot(
                task_id=trial.task_id,
                harness_version=trial.harness_version or self.system.version,
                model_scope=model_scope_from_trial(trial),
                scope_config=model_config_from_trial(trial),
                component_hashes={
                    name: version.content_hash
                    for name, version in trial.component_versions.items()
                },
                verification_output=trial.verifier_output,
                required_assertions=[
                    f"Task {trial.task_id} must pass Harbor verification with score >= 1.0",
                    f"Source trial: {trial.trial_id}",
                    f"Source summary: {summary_id}",
                    "Snapshot remains pending until post-update regression succeeds.",
                ],
                source_trial_id=trial.trial_id,
                source_summary_id=summary_id,
                validation_status="pending",
            )
            self.memory.save_regression(trial.task_id, snapshot)

    def _track_pending_update_baseline(
        self,
        summary: TrialSummary,
        results: list[TrialResult],
    ) -> None:
        self.pending_update_baseline_score = summary.overall_score
        self.pending_update_baseline_tasks = {result.task_id for result in results}
        self.pending_update_baseline_summary_id = summary.summary_id

    def _clear_pending_update_baseline(self) -> None:
        self.pending_update_baseline_score = None
        self.pending_update_baseline_tasks = set()
        self.pending_update_baseline_summary_id = ""

    def _rollback_pending_update_if_worse(
        self,
        summary: TrialSummary,
        results: list[TrialResult],
    ) -> None:
        baseline_score = self.pending_update_baseline_score
        if baseline_score is None:
            return
        current_tasks = {result.task_id for result in results}
        if current_tasks != self.pending_update_baseline_tasks:
            self.memory.save_component_lesson(
                "codex_update",
                (
                    "Skipped automatic score-regression rollback because the "
                    "post-update task set was not comparable to baseline summary "
                    f"{self.pending_update_baseline_summary_id}."
                ),
            )
            self._clear_pending_update_baseline()
            return
        if summary.overall_score + self.min_improvement >= baseline_score:
            self._clear_pending_update_baseline()
            return
        update_engine = self.system.update_engine
        if update_engine is None or not hasattr(update_engine, "rollback_last"):
            self._clear_pending_update_baseline()
            return
        if hasattr(update_engine, "rollback_last_accepted"):
            rolled_back = bool(update_engine.rollback_last_accepted())
        else:
            rolled_back = bool(update_engine.rollback_last())
        summary.patches_applied.append("codex_update:rolled_back_worse_score")
        if rolled_back and self.system.patch_count > 0:
            self.system.patch_count -= 1
        self.memory.save_component_lesson(
            "codex_update",
            (
                "Rolled back previous Codex update because the next comparable "
                f"summary score dropped from {baseline_score:.4f} "
                f"({self.pending_update_baseline_summary_id}) to "
                f"{summary.overall_score:.4f} ({summary.summary_id}). "
                f"Rollback applied: {str(rolled_back).lower()}."
            ),
        )
        self._clear_pending_update_baseline()

    def _regression_contracts(self) -> list[str]:
        regressions_dir = getattr(self.memory, "regressions_dir", None)
        if regressions_dir is None:
            return []
        paths = sorted(regressions_dir.rglob("*.json"))
        if self.regression_holdout_fraction <= 0:
            return [str(path) for path in paths]
        # Hide held-out (D_out) snapshots from the proposer so a held-out
        # regression remains a genuine anti-overfit signal.
        from hl.regression_split import is_holdout_task

        contracts: list[str] = []
        for path in paths:
            task_id = self._regression_snapshot_task_id(path)
            if task_id and is_holdout_task(
                task_id,
                fraction=self.regression_holdout_fraction,
                seed=self.regression_holdout_seed,
            ):
                continue
            contracts.append(str(path))
        return contracts

    @staticmethod
    def _regression_snapshot_task_id(path: Any) -> str:
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return ""
        return str(data.get("task_id") or "")

    def _required_validation_commands(self) -> list[str]:
        return [
            "pytest tests/ -v",
            "python scripts/regression_check.py --dry-run",
        ]

    def _attach_codex_update_artifacts(
        self,
        failed_trials: list[TrialResult],
        update_result: Any,
    ) -> None:
        if not hasattr(self.memory, "attach_codex_update"):
            return
        for trial in failed_trials:
            self.memory.attach_codex_update(
                trial.trial_id,
                packet_path=getattr(update_result, "packet_path", None),
                events_path=getattr(update_result, "events_path", None),
                final_message_path=getattr(update_result, "final_message_path", None),
                diff_path=getattr(update_result, "diff_path", None),
                record_path=getattr(update_result, "record_path", None),
                summary_path=getattr(update_result, "summary_path", None),
                review=getattr(update_result, "review", None),
            )


def _trial_is_infrastructure_failure(trial: TrialResult) -> bool:
    return trial_is_infrastructure_failure(trial)
