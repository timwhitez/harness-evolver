"""Tests for the HL optimization loop."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

import hl.loop as hl_loop_module
from hl.loop import HLLoop
from hl.memory import FileSystemMemory
from hl.goals import GoalStore
from hl.types import (
    RegressionSnapshot,
    TaskDifficulty,
    TaskDomain,
    TrialResult,
    TrialStatus,
)
from meta.reviewer import PatchReviewResult


class TestHLLoop:
    def test_initial_state(self):
        loop = HLLoop()
        assert loop.iteration == 0
        assert loop.best_score == 0.0
        assert loop.iterations_without_improvement == 0
        assert loop.max_iterations is None
        assert loop.patience is None
        assert len(loop.history) == 0

    def test_should_continue_at_start(self):
        loop = HLLoop()
        assert loop.should_continue() is True

    def test_should_continue_past_legacy_max_iterations(self):
        loop = HLLoop(max_iterations=3)
        loop.iteration = 3
        assert loop.should_continue() is True

    def test_should_continue_past_patience_plateau(self):
        loop = HLLoop(patience=3)
        loop.iterations_without_improvement = 3
        assert loop.should_continue() is True

    def test_none_patience_disables_plateau_stop(self):
        loop = HLLoop(patience=None)
        loop.iterations_without_improvement = 100
        assert loop.should_continue() is True

    def test_should_continue_after_per_iteration_token_budget_overrun(self, tmp_path):
        store = GoalStore(tmp_path / "goal.json")
        store.create_goal("reach score", token_budget=1)
        store.update_usage(worker_tokens={"input": 1})
        loop = HLLoop(goal_store=store)
        assert loop.should_continue() is True
        assert store.get_goal().status == "active"
        assert store.get_goal().usage.token_budget_overruns == 1
        assert store.get_goal().usage.token_budget_observations == 1

    def test_should_continue_when_campaign_token_budget_exhausted(self, tmp_path):
        store = GoalStore(tmp_path / "goal.json")
        store.create_goal("reach score", token_budget=1, token_budget_scope="campaign")
        store.update_usage(worker_tokens={"input": 1})
        loop = HLLoop(goal_store=store)
        assert loop.should_continue() is True

    def test_budget_reference_crossing_does_not_truncate_master_iteration(self, tmp_path):
        goal_store = GoalStore(tmp_path / "goal.json")
        goal_store.create_goal(
            "keep running",
            token_budget=1,
            token_budget_scope="campaign",
            wall_time_budget_seconds=1,
        )
        goal_store.update_usage(
            worker_tokens={"input": 1},
            harbor_wall_time_seconds=1,
        )
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=goal_store,
            max_iterations=1,
            patience=1,
        )
        runner = CountingRunner()

        summary = loop.run_iteration(
            tasks=["task-a", "task-b", "task-c"],
            task_instructions={
                "task-a": "do task",
                "task-b": "do task",
                "task-c": "do task",
            },
            task_contexts={
                "task-a": {"task_id": "task-a"},
                "task-b": {"task_id": "task-b"},
                "task-c": {"task_id": "task-c"},
            },
            agent_runner=runner,
            stop_after_trial=lambda **_kwargs: True,
        )

        assert loop.should_continue() is True
        assert runner.started == ["task-a", "task-b", "task-c"]
        assert summary.trial_ids == ["trial-task-a", "trial-task-b", "trial-task-c"]
        goal = goal_store.get_goal()
        assert goal is not None
        assert goal.status == "active"
        assert "audit observation only" in goal.completion_reason
        assert "loops remain active" in goal.completion_reason

    def test_should_continue_after_legacy_limit_stopped_goal(self, tmp_path):
        goal_path = tmp_path / "goal.json"
        goal_path.write_text(
            '{"objective":"reach score","status":"stopped",'
            '"completion_reason":"iteration round limit exhausted"}'
        )
        loop = HLLoop(goal_store=GoalStore(goal_path))

        assert loop.should_continue() is True
        assert loop.goal_store.get_goal().status == "active"

    def test_should_continue_after_legacy_limit_complete_goal(self, tmp_path):
        goal_path = tmp_path / "goal.json"
        goal_path.write_text(
            '{"objective":"reach score","status":"complete",'
            '"completion_reason":"token budget and timeout exhausted",'
            '"completed_at":"2026-06-20T00:00:00"}'
        )
        loop = HLLoop(goal_store=GoalStore(goal_path))

        assert loop.should_continue() is True
        goal = loop.goal_store.get_goal()
        assert goal is not None
        assert goal.status == "active"
        assert goal.completed_at is None
        assert "Legacy complete status" in goal.completion_reason

    def test_should_continue_after_any_owner_limit_reason_goal(self, tmp_path):
        limit_reasons = (
            "master loop reached max_iterations",
            "sub-agent attempt cap reached",
            "Codex update timeout_seconds exhausted",
            "diagnostic target K reached",
            "context depth window reached",
            "validation regression snapshot count cap reached",
            "mission max_features reached",
            "Worker max_turns reached",
            "token budget exhausted before score target",
        )
        for index, reason in enumerate(limit_reasons):
            goal_path = tmp_path / f"goal-{index}.json"
            goal_path.write_text(
                json.dumps(
                    {
                        "objective": "reach score",
                        "status": "stopped",
                        "completion_reason": reason,
                    }
                )
            )
            loop = HLLoop(goal_store=GoalStore(goal_path))

            assert loop.should_continue() is True, reason
            goal = loop.goal_store.get_goal()
            assert goal is not None
            assert goal.status == "active"
            assert "Legacy stopped status" in goal.completion_reason

    def test_should_continue_after_non_explicit_complete_goal(self, tmp_path):
        goal_path = tmp_path / "goal.json"
        goal_path.write_text(
            '{"objective":"reach score","status":"complete",'
            '"completion_reason":"scheduler marked campaign complete",'
            '"completed_at":"2026-06-20T00:00:00"}'
        )
        loop = HLLoop(goal_store=GoalStore(goal_path))

        assert loop.should_continue() is True
        goal = loop.goal_store.get_goal()
        assert goal is not None
        assert goal.status == "active"
        assert goal.completed_at is None
        assert "Non-explicit complete status" in goal.completion_reason

    def test_should_continue_rejects_non_explicit_complete_status_directly(self):
        loop = HLLoop()
        loop.goal_store = SimpleNamespace(
            get_goal=lambda: SimpleNamespace(
                status="complete",
                completion_reason="scheduler marked campaign complete",
            )
        )

        assert loop.should_continue() is True

    def test_should_continue_preserves_explicit_complete_goal(self, tmp_path):
        store = GoalStore(tmp_path / "goal.json")
        store.create_goal("reach score")
        store.update_goal(
            "complete",
            reason="explicit user marks campaign goal complete",
        )
        loop = HLLoop(goal_store=store)

        assert loop.should_continue() is False

    def test_should_continue_preserves_explicit_stopped_goal(self, tmp_path):
        goal_path = tmp_path / "goal.json"
        goal_path.write_text(
            '{"objective":"reach score","status":"stopped",'
            '"completion_reason":"explicit user stopped campaign"}'
        )
        loop = HLLoop(goal_store=GoalStore(goal_path))

        assert loop.should_continue() is False

    def test_should_continue_normalizes_non_explicit_stopped_goal(self, tmp_path):
        goal_path = tmp_path / "goal.json"
        goal_path.write_text(
            '{"objective":"reach score","status":"stopped",'
            '"completion_reason":"auto halted by scheduler"}'
        )
        loop = HLLoop(goal_store=GoalStore(goal_path))

        assert loop.should_continue() is True
        goal = loop.goal_store.get_goal()
        assert goal is not None
        assert goal.status == "active"
        assert "Non-explicit stopped status" in goal.completion_reason

    def test_get_progress_empty(self):
        loop = HLLoop()
        progress = loop.get_progress()
        assert progress["iteration"] == 0
        assert progress["best_score"] == 0.0
        assert progress["solved_tasks"] == 0

    def test_system_wiring(self):
        loop = HLLoop()
        assert loop.system.memory is not None
        assert isinstance(loop.system.memory, FileSystemMemory)

    def test_codex_style_update_engine_runs_on_failed_trial(self, tmp_path):
        update_engine = FakeCodexUpdateEngine(accepted=True)
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=GoalStore(tmp_path / "goal.json"),
        )
        loop.system.set_update_engine(update_engine)

        summary = loop.run_iteration(
            tasks=["task-a"],
            task_instructions={"task-a": "do task"},
            task_contexts={"task-a": {}},
            agent_runner=FakeRunner(status=TrialStatus.FAILED),
        )

        assert update_engine.called is True
        assert summary.patches_applied == ["codex_update:accepted"]
        assert loop.system.patch_count == 1
        summary_path = tmp_path / "trials" / "summaries" / "summary_001.json"
        assert "codex_update:accepted" in summary_path.read_text()

    def test_codex_style_update_engine_uses_call_site_validation_commands(self, tmp_path):
        update_engine = FakeCodexUpdateEngine(accepted=True)
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=GoalStore(tmp_path / "goal.json"),
        )
        loop.system.set_update_engine(update_engine)

        commands = [
            "pytest tests/test_models_and_worker_policy.py -q",
            (
                "python scripts/regression_check.py --memory-path trials "
                "--worker-role worker_deepseek --snapshot-status stable"
            ),
        ]

        loop.run_iteration(
            tasks=["task-a"],
            task_instructions={"task-a": "do task"},
            task_contexts={"task-a": {}},
            agent_runner=FakeRunner(status=TrialStatus.FAILED),
            required_validation_commands=commands,
        )

        assert update_engine.required_validation_commands == commands

    def test_codex_style_update_engine_runs_on_infrastructure_failure(self, tmp_path):
        update_engine = FakeCodexUpdateEngine(accepted=False)
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=GoalStore(tmp_path / "goal.json"),
        )
        loop.system.set_update_engine(update_engine)

        loop.run_iteration(
            tasks=["task-infra"],
            task_instructions={"task-infra": "do task"},
            task_contexts={"task-infra": {"task_id": "task-infra"}},
            agent_runner=InfrastructureFailureRunner(),
        )

        assert update_engine.called is True
        assert [trial.task_id for trial in update_engine.failures] == ["task-infra"]
        trial = update_engine.failures[0]
        assert trial.metadata["score_exclusion_reason"] == "infrastructure_error"

    def test_rejected_codex_style_update_rolls_back(self, tmp_path):
        update_engine = FakeCodexUpdateEngine(accepted=False)
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=GoalStore(tmp_path / "goal.json"),
        )
        loop.system.set_update_engine(update_engine)

        summary = loop.run_iteration(
            tasks=["task-a"],
            task_instructions={"task-a": "do task"},
            task_contexts={"task-a": {}},
            agent_runner=FakeRunner(status=TrialStatus.FAILED),
        )

        assert update_engine.rollback_called is True
        assert summary.patches_applied == []
        assert loop.system.patch_count == 0

    def test_legacy_update_engine_records_analysis_without_patching(self, tmp_path):
        update_engine = FakeLegacyUpdateEngine()
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        loop = HLLoop(
            memory=memory,
            goal_store=GoalStore(tmp_path / "goal.json"),
        )
        loop.system.set_update_engine(update_engine)

        summary = loop.run_iteration(
            tasks=["task-a"],
            task_instructions={"task-a": "do task"},
            task_contexts={"task-a": {}},
            agent_runner=FakeRunner(status=TrialStatus.FAILED),
        )

        assert update_engine.analyze_called is True
        assert update_engine.suggest_called is False
        assert update_engine.apply_called is False
        assert summary.patches_applied == []
        assert loop.system.patch_count == 0
        lesson_path = memory.component_lessons_dir / "codex_update.md"
        lesson = lesson_path.read_text()
        assert "Legacy update engine `legacy_scaffold` produced analysis" in lesson
        assert "diagnostic evidence only" in lesson
        assert "Codex-backed run_update" in lesson

    def test_next_comparable_score_regression_rolls_back_previous_codex_update(self, tmp_path):
        update_engine = FakeCodexUpdateEngine(accepted=True)
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        loop = HLLoop(
            memory=memory,
            goal_store=GoalStore(tmp_path / "goal.json"),
            min_improvement=0.01,
        )
        loop.system.set_update_engine(update_engine)

        runner = RoundScoreRunner(
            [
                {
                    "task-a": (TrialStatus.PASSED, 1.0),
                    "task-b": (TrialStatus.FAILED, 0.0),
                },
                {
                    "task-a": (TrialStatus.FAILED, 0.0),
                    "task-b": (TrialStatus.FAILED, 0.0),
                },
            ]
        )

        first = loop.run_iteration(
            tasks=["task-a", "task-b"],
            task_instructions={"task-a": "do task", "task-b": "do task"},
            task_contexts={"task-a": {"task_id": "task-a"}, "task-b": {"task_id": "task-b"}},
            agent_runner=runner,
        )
        second = loop.run_iteration(
            tasks=["task-a", "task-b"],
            task_instructions={"task-a": "do task", "task-b": "do task"},
            task_contexts={"task-a": {"task_id": "task-a"}, "task-b": {"task_id": "task-b"}},
            agent_runner=runner,
        )

        assert first.overall_score == 0.5
        assert second.overall_score == 0.0
        assert update_engine.rollback_called is True
        assert "codex_update:rolled_back_worse_score" in second.patches_applied
        lesson = (memory.component_lessons_dir / "codex_update.md").read_text()
        assert "score dropped from 0.5000" in lesson

    def test_repeated_failures_write_component_lesson(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        loop = HLLoop(
            memory=memory,
            goal_store=GoalStore(tmp_path / "goal.json"),
        )

        loop.run_iteration(
            tasks=["task-a", "task-b"],
            task_instructions={"task-a": "do task", "task-b": "do task"},
            task_contexts={"task-a": {"task_id": "task-a"}, "task-b": {"task_id": "task-b"}},
            agent_runner=TaskAwareFailingRunner(),
        )

        lesson_path = memory.component_lessons_dir / "tools_shell.md"
        assert lesson_path.exists()
        lesson = lesson_path.read_text()
        assert "Repeated failure category `dependency_issue`" in lesson
        assert "task-a, task-b" in lesson
        assert "trajectory slices and regression" in lesson

    def test_summary_tracks_domain_and_difficulty_scores(self, tmp_path):
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=GoalStore(tmp_path / "goal.json"),
        )

        summary = loop.run_iteration(
            tasks=["security-task", "database-task"],
            task_instructions={"security-task": "do task", "database-task": "do task"},
            task_contexts={
                "security-task": {"task_id": "security-task"},
                "database-task": {"task_id": "database-task"},
            },
            agent_runner=MixedScoreRunner(),
        )

        assert summary.overall_score == 0.5
        assert summary.per_domain_scores == {"database": 0.0, "security": 1.0}
        assert summary.per_difficulty_scores == {"easy": 1.0, "hard": 0.0}

    def test_round_task_concurrency_overlaps_tasks_and_keeps_summary_order(self, tmp_path):
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=GoalStore(tmp_path / "goal.json"),
        )
        runner = BlockingRunner()
        recorded = []

        summary = loop.run_iteration(
            tasks=["task-a", "task-b"],
            task_instructions={"task-a": "do task", "task-b": "do task"},
            task_contexts={"task-a": {"task_id": "task-a"}, "task-b": {"task_id": "task-b"}},
            agent_runner=runner,
            task_concurrency=2,
            on_trial_recorded=lambda **kwargs: recorded.append(
                [trial.trial_id for trial in kwargs["completed_results"]]
            ),
        )

        assert runner.max_active == 2
        assert summary.trial_ids == ["trial-task-a", "trial-task-b"]
        assert sorted(recorded[-1]) == ["trial-task-a", "trial-task-b"]

    def test_parallel_keyboard_interrupt_records_finished_futures(
        self, tmp_path, monkeypatch
    ):
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=GoalStore(tmp_path / "goal.json"),
        )
        recorded = []

        def interrupted_wait(futures, return_when=None):
            raise KeyboardInterrupt

        monkeypatch.setattr(hl_loop_module, "wait", interrupted_wait)

        with pytest.raises(KeyboardInterrupt):
            loop.run_iteration(
                tasks=["task-a", "task-b"],
                task_instructions={"task-a": "do task", "task-b": "do task"},
                task_contexts={
                    "task-a": {"task_id": "task-a"},
                    "task-b": {"task_id": "task-b"},
                },
                agent_runner=InflatedWallTimeRunner(),
                task_concurrency=2,
                on_trial_recorded=lambda **kwargs: recorded.append(
                    kwargs["trial"].trial_id
                ),
            )

        assert sorted(recorded) == ["trial-task-a", "trial-task-b"]
        assert loop.system.trial_count == 2
        assert (tmp_path / "trials" / "runs" / "trial-task-a" / "result.json").exists()
        assert (tmp_path / "trials" / "runs" / "trial-task-b" / "result.json").exists()

    def test_parallel_goal_budget_counts_elapsed_wall_time(self, tmp_path):
        goal_store = GoalStore(tmp_path / "goal.json")
        goal_store.create_goal("keep running", wall_time_budget_seconds=5)
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=goal_store,
        )

        loop.run_iteration(
            tasks=["task-a", "task-b"],
            task_instructions={"task-a": "do task", "task-b": "do task"},
            task_contexts={"task-a": {"task_id": "task-a"}, "task-b": {"task_id": "task-b"}},
            agent_runner=InflatedWallTimeRunner(),
            task_concurrency=2,
        )

        goal = goal_store.get_goal()
        assert goal is not None
        assert goal.status == "active"
        assert goal.usage.harbor_wall_time_seconds < 5

    def test_stop_after_trial_is_audit_only_and_does_not_cap_parallel_submissions(self, tmp_path):
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=GoalStore(tmp_path / "goal.json"),
        )
        runner = StopAfterTrialRunner()

        summary = loop.run_iteration(
            tasks=["task-a", "task-b", "task-c", "task-d"],
            task_instructions={
                task_id: "do task"
                for task_id in ["task-a", "task-b", "task-c", "task-d"]
            },
            task_contexts={
                task_id: {"task_id": task_id}
                for task_id in ["task-a", "task-b", "task-c", "task-d"]
            },
            agent_runner=runner,
            task_concurrency=2,
            stop_after_trial=lambda **kwargs: kwargs["trial"].task_id == "task-a",
        )

        assert set(runner.started) == {"task-a", "task-b", "task-c", "task-d"}
        assert set(summary.trial_ids) == {
            "trial-task-a",
            "trial-task-b",
            "trial-task-c",
            "trial-task-d",
        }
        lesson = (
            tmp_path
            / "trials"
            / "memory"
            / "component_lessons"
            / "loop_limit_contract.md"
        ).read_text()
        assert "stop_after_trial hook requested a stop" in lesson
        assert "audit-only" in lesson
        assert "continues submitting the remaining tasks" in lesson

    def test_legacy_master_loop_limits_do_not_cap_iterations_or_tasks(self, tmp_path):
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=GoalStore(tmp_path / "goal.json"),
            max_iterations=1,
            patience=1,
        )
        loop.iterations_without_improvement = 99
        runner = CountingRunner()

        first = loop.run_iteration(
            tasks=["task-a", "task-b"],
            task_instructions={"task-a": "do task", "task-b": "do task"},
            task_contexts={
                "task-a": {"task_id": "task-a"},
                "task-b": {"task_id": "task-b"},
            },
            agent_runner=runner,
            stop_after_trial=lambda **_kwargs: True,
        )
        second = loop.run_iteration(
            tasks=["task-c", "task-d"],
            task_instructions={"task-c": "do task", "task-d": "do task"},
            task_contexts={
                "task-c": {"task_id": "task-c"},
                "task-d": {"task_id": "task-d"},
            },
            agent_runner=runner,
            stop_after_trial=lambda **_kwargs: True,
        )

        assert loop.should_continue() is True
        assert loop.iteration == 2
        assert runner.started == ["task-a", "task-b", "task-c", "task-d"]
        assert first.trial_ids == ["trial-task-a", "trial-task-b"]
        assert second.trial_ids == ["trial-task-c", "trial-task-d"]
        lesson = (
            tmp_path
            / "trials"
            / "memory"
            / "component_lessons"
            / "loop_limit_contract.md"
        ).read_text()
        assert lesson.count("stop_after_trial hook requested a stop") == 4
        assert "trial-count and per-round stop requests are audit-only" in lesson

    def test_master_and_sub_agent_limit_fields_do_not_stop_future_iterations(self, tmp_path):
        goal_store = GoalStore(tmp_path / "goal.json")
        goal_store.create_goal(
            "keep optimizing",
            token_budget=1,
            token_budget_scope="campaign",
            wall_time_budget_seconds=1,
        )
        goal_store.update_usage(
            worker_tokens={"input": 2},
            harbor_wall_time_seconds=2,
        )
        loop = HLLoop(
            memory=FileSystemMemory(base_path=str(tmp_path / "trials")),
            goal_store=goal_store,
            max_iterations=1,
            patience=1,
        )
        loop.iteration = 99
        loop.iterations_without_improvement = 99
        runner = CountingRunner()

        first = loop.run_iteration(
            tasks=["task-a", "task-b"],
            task_instructions={"task-a": "do task", "task-b": "do task"},
            task_contexts={
                "task-a": {"task_id": "task-a"},
                "task-b": {"task_id": "task-b"},
            },
            agent_runner=runner,
            stop_after_trial=lambda **_kwargs: True,
        )
        second = loop.run_iteration(
            tasks=["task-c", "task-d"],
            task_instructions={"task-c": "do task", "task-d": "do task"},
            task_contexts={
                "task-c": {"task_id": "task-c"},
                "task-d": {"task_id": "task-d"},
            },
            agent_runner=runner,
            stop_after_trial=lambda **_kwargs: True,
        )

        assert loop.should_continue() is True
        assert first.trial_ids == ["trial-task-a", "trial-task-b"]
        assert second.trial_ids == ["trial-task-c", "trial-task-d"]
        assert runner.started == ["task-a", "task-b", "task-c", "task-d"]
        goal = goal_store.get_goal()
        assert goal is not None
        assert goal.status == "active"
        assert "loops remain active" in goal.completion_reason

    def test_run_iteration_injects_task_historical_analysis_lessons(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        analysis_dir = (
            tmp_path
            / "trials"
            / "analysis"
            / "campaign-a"
            / "summary_010"
        )
        analysis_dir.mkdir(parents=True)
        (analysis_dir / "summary.json").write_text(
            json.dumps(
                {
                    "summary_id": "summary_010",
                    "failure_buckets": [
                        {
                            "failure_category": "structured_csv_table_contract",
                            "count": 1,
                            "task_ids": ["sam-cell-seg"],
                            "affected_components": ["tools/shell", "verification/checks"],
                        }
                    ],
                    "trajectory_evidence": {
                        "sam-cell-seg": {
                            "failure_mechanisms": [
                                {
                                    "name": "structured_csv_table_contract",
                                    "description": (
                                        "Verifier loads demo_metadata.csv via "
                                        "pd.read_csv(args.csv_path) as a table and "
                                        "checks keyed row content; repair must preserve "
                                        "columns rgb_path, csv_path, area, bbox, width, "
                                        "height in exact order, row count, key column "
                                        "identity, blank-vs-nonblank cells, numeric/text "
                                        "dtype and formatting."
                                    ),
                                    "evidence": "df = pd.read_csv(args.csv_path)",
                                }
                            ],
                            "policy_counts": {
                                "package_manager_timeout_cap": 53,
                                "structured_csv_table_contract": 1,
                            },
                            "dependency_and_toolchain_evidence": [
                                {
                                    "command": "pip install numpy pandas torch torchvision opencv-python",
                                    "policies": "package_manager_timeout_cap",
                                }
                            ],
                        }
                    },
                }
            )
        )
        runner = CapturingRunner()
        loop = HLLoop(
            memory=memory,
            goal_store=GoalStore(tmp_path / "goal.json"),
        )

        loop.run_iteration(
            tasks=["sam-cell-seg"],
            task_instructions={"sam-cell-seg": "repair demo metadata csv"},
            task_contexts={
                "sam-cell-seg": {
                    "task_id": "sam-cell-seg",
                    "previous_errors": ["caller supplied context"],
                }
            },
            agent_runner=runner,
        )

        previous_errors = runner.contexts[0]["previous_errors"]
        assert previous_errors[0] == "caller supplied context"
        lesson = next(
            item
            for item in previous_errors
            if "Historical analysis lesson from summary_010" in item
        )
        assert "structured_csv_table_contract" in lesson
        assert "pd.read_csv(args.csv_path)" in lesson
        assert "demo_metadata.csv" in lesson
        assert "columns rgb_path, csv_path, area, bbox, width, height" in lesson
        assert "package_manager_timeout_cap=53" in lesson
        assert "pip install numpy pandas torch torchvision opencv-python" in lesson
        assert "loop_stop_condition=false" in lesson
        assert "time_round_token_limit_driven=false" in lesson
        assert "not a master, sub-agent, or Worker loop stop condition" in lesson

    def test_regression_contracts_include_model_scoped_snapshots(self, tmp_path):
        memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
        memory.save_regression(
            "legacy-task",
            RegressionSnapshot(
                task_id="legacy-task",
                harness_version="v1",
                source_trial_id="trial-legacy",
            ),
        )
        memory.save_regression(
            "scoped-task",
            RegressionSnapshot(
                task_id="scoped-task",
                harness_version="v1",
                model_scope="provider=openai_compatible;base_url_host=api.deepseek.com;model=deepseek-v4-flash",
                source_trial_id="trial-scoped",
            ),
        )
        loop = HLLoop(memory=memory, goal_store=GoalStore(tmp_path / "goal.json"))

        contracts = set(loop._regression_contracts())

        assert str(memory.regressions_dir / "legacy-task.json") in contracts
        assert any(contract.endswith("/scoped-task.json") for contract in contracts)


class FakeRunner:
    def __init__(self, status: TrialStatus):
        self.status = status

    def run(self, instruction, context):
        return TrialResult(
            trial_id="trial-a",
            task_id="task-a",
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=self.status,
            score=0.0,
            verified=True,
            error_log=["failed"],
        )


class InfrastructureFailureRunner:
    def run(self, instruction, context):
        return TrialResult(
            trial_id="trial-infra",
            task_id=context["task_id"],
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.TIMEOUT,
            score=0.0,
            verified=False,
            error_log=["Environment start timed out after 600.0 seconds"],
            metadata={
                "timeout_phase": "environment_start",
                "infra_error_detected": True,
                "score_exclusion_reason": "infrastructure_error",
            },
        )


class TaskAwareFailingRunner:
    def run(self, instruction, context):
        task_id = context["task_id"]
        return TrialResult(
            trial_id=f"trial-{task_id}",
            task_id=task_id,
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.FAILED,
            score=0.0,
            verified=True,
            error_log=["bash: rg: command not found"],
            tool_calls=[
                {"tool": "bash", "success": False, "error": "command not found"}
            ],
        )


class MixedScoreRunner:
    def run(self, instruction, context):
        task_id = context["task_id"]
        if task_id == "security-task":
            return TrialResult(
                trial_id="trial-security",
                task_id=task_id,
                task_domain=TaskDomain.SECURITY,
                task_difficulty=TaskDifficulty.EASY,
                status=TrialStatus.PASSED,
                score=1.0,
                verified=True,
            )
        return TrialResult(
            trial_id="trial-database",
            task_id=task_id,
            task_domain=TaskDomain.DATABASE,
            task_difficulty=TaskDifficulty.HARD,
            status=TrialStatus.FAILED,
            score=0.0,
            verified=True,
            error_log=["failed"],
        )


class BlockingRunner:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(2)

    def run(self, instruction, context):
        task_id = context["task_id"]
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=2)
            if task_id == "task-a":
                time.sleep(0.05)
            return TrialResult(
                trial_id=f"trial-{task_id}",
                task_id=task_id,
                task_domain=TaskDomain.SOFTWARE_ENGINEERING,
                task_difficulty=TaskDifficulty.EASY,
                status=TrialStatus.PASSED,
                score=1.0,
                verified=True,
            )
        finally:
            with self.lock:
                self.active -= 1


class InflatedWallTimeRunner:
    def run(self, instruction, context):
        task_id = context["task_id"]
        time.sleep(0.05)
        return TrialResult(
            trial_id=f"trial-{task_id}",
            task_id=task_id,
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
            score=1.0,
            verified=True,
            wall_time_seconds=10.0,
        )


class StopAfterTrialRunner:
    def __init__(self):
        self.started: list[str] = []
        self.lock = threading.Lock()

    def run(self, instruction, context):
        task_id = context["task_id"]
        with self.lock:
            self.started.append(task_id)
        if task_id == "task-b":
            time.sleep(0.05)
        status = TrialStatus.FAILED if task_id == "task-a" else TrialStatus.PASSED
        return TrialResult(
            trial_id=f"trial-{task_id}",
            task_id=task_id,
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=status,
            score=1.0 if status == TrialStatus.PASSED else 0.0,
            verified=True,
            error_log=[] if status == TrialStatus.PASSED else ["provider billing error"],
        )


class CountingRunner:
    def __init__(self):
        self.started: list[str] = []

    def run(self, instruction, context):
        task_id = context["task_id"]
        self.started.append(task_id)
        return TrialResult(
            trial_id=f"trial-{task_id}",
            task_id=task_id,
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
            score=1.0,
            verified=True,
        )


class CapturingRunner:
    def __init__(self):
        self.contexts: list[dict] = []

    def run(self, instruction, context):
        self.contexts.append(dict(context))
        task_id = context["task_id"]
        return TrialResult(
            trial_id=f"trial-{task_id}",
            task_id=task_id,
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=TrialStatus.PASSED,
            score=1.0,
            verified=True,
        )


class RoundScoreRunner:
    def __init__(self, rounds):
        self.rounds = rounds
        self.index = 0

    def run(self, instruction, context):
        round_index = min(self.index // 2, len(self.rounds) - 1)
        task_id = context["task_id"]
        status, score = self.rounds[round_index][task_id]
        self.index += 1
        return TrialResult(
            trial_id=f"trial-{round_index}-{task_id}",
            task_id=task_id,
            task_domain=TaskDomain.SOFTWARE_ENGINEERING,
            task_difficulty=TaskDifficulty.EASY,
            status=status,
            score=score,
            verified=True,
            error_log=[] if status == TrialStatus.PASSED else ["failed"],
        )


class FakeCodexUpdateEngine:
    name = "fake_codex"

    def __init__(self, accepted: bool):
        self.accepted = accepted
        self.called = False
        self.rollback_called = False
        self.failures = []

    def run_update(self, **kwargs):
        self.called = True
        self.failures = list(kwargs.get("failures") or [])
        self.required_validation_commands = list(
            kwargs.get("required_validation_commands") or []
        )
        return SimpleNamespace(
            review=PatchReviewResult(
                accepted=self.accepted,
                reasons=[] if self.accepted else ["rejected"],
                changed_files=["bench/agent.py"],
            )
        )

    def rollback_last(self):
        self.rollback_called = True
        return True


class FakeLegacyUpdateEngine:
    name = "legacy_scaffold"

    def __init__(self):
        self.analyze_called = False
        self.suggest_called = False
        self.apply_called = False

    def analyze_failures(self, feedback_signals, trials):
        self.analyze_called = True
        return [
            {
                "task_id": trials[0].task_id,
                "affected_components": ["harness/recovery", "bench/agent"],
            }
        ]

    def suggest_edits(self, findings, current_harness):
        self.suggest_called = True
        raise AssertionError("legacy update engine must not suggest direct edits")

    def apply_patch(self, patch):
        self.apply_called = True
        raise AssertionError("legacy update engine must not apply direct patches")

    def rollback_patch(self, patch):
        return False
