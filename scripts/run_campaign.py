#!/usr/bin/env python3
"""Run a minimal HL campaign iteration through Harbor and HLLoop."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_trial import (  # noqa: E402
    _network_preflight_argv,
    add_docker_resource_args,
    docker_resource_forward_args,
    validate_docker_concurrency_budget,
)
from harness.tools.shell import (  # noqa: E402
    background_package_command_reason,
    broad_proc_scan_command_reason,
    broad_root_find_command_reason,
    large_toolchain_install_command_reason,
    manual_deb_dependency_chase_reason,
    manual_dependency_download_reason,
    package_manager_timeout_cap,
    shell_semantic_failure_kind,
    staged_dependency_script_reason,
)
from hl.failure_mechanisms import (  # noqa: E402
    DEPENDENCY_LOOP_BASE_REPLACEMENT_NEUTRAL_MECHANISM_NAMES,
    DEPENDENCY_PIVOT_MECHANISM_NAMES,
    PRIMARY_VERIFIER_CONTRACT_MECHANISM_NAMES,
    TERMINAL_ENVIRONMENT_UNAVAILABLE_AFTER_DEPENDENCY_LOOP_MECHANISM,
    affected_components_for_failure_mechanism,
    dependency_loop_failure_category_for_trial,
    dependency_loop_mechanism_for_failure_category,
    failure_mechanisms_replace_base_components,
    failure_mechanisms_for_trial,
)
from hl.loop_limits import (  # noqa: E402
    all_loop_non_terminal_event_metadata,
    normalize_legacy_limit_driven_skip_event,
)
from hl.guard_convergence import (  # noqa: E402
    fixed_eval_audit,
    fixed_eval_task_ids,
)
from hl.types import trial_is_infrastructure_failure  # noqa: E402


@dataclass
class RegressionRunResult:
    returncode: int
    failed_tasks: list[str]
    stdout: str = ""
    stderr: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the HL campaign loop. An explicitly requested --iterations "
            "value is a campaign completion target; timeout, budget, patience, "
            "cooldown, K/attempt, and max-turn fields remain request/audit "
            "controls and do not bound Codex update, diagnostic/context, or "
            "Worker task loops."
        )
    )
    parser.add_argument("--path", default="terminal-bench-tasks/terminal-bench")
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="TerminalBench task to run; repeatable",
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task ids to run")
    parser.add_argument("--task-file", default=None, help="File containing one task id per line")
    parser.add_argument(
        "--task-selection",
        choices=["random", "full", "ids"],
        default=None,
        help="Campaign task selection mode; defaults to config/trials.yaml",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=None,
        help=(
            "Number of tasks sampled for --task-selection random. This selects "
            "the campaign task pool or per-round slice; it is not a master-loop "
            "round limit."
        ),
    )
    parser.add_argument(
        "--random-seed",
        default=None,
        help="Seed for random task selection; defaults to the campaign id",
    )
    parser.add_argument(
        "--task-index",
        action="append",
        default=None,
        help="1-based TerminalBench task index in stable full-catalog order; repeatable",
    )
    parser.add_argument(
        "--task-indices",
        default=None,
        help="Comma-separated 1-based task indices or ranges, for example 1,3,10-12",
    )
    parser.add_argument(
        "--task-set",
        choices=["smoke", "domain-balanced", "hard-focus", "full"],
        default=None,
        help="Select tasks from the local TerminalBench catalog",
    )
    parser.add_argument(
        "--guard-convergence-eval",
        action="store_true",
        help=(
            "Use config/trials.yaml guard_convergence.fixed_eval as the fixed "
            "score-safety set for guard reduction work."
        ),
    )
    parser.add_argument("--domain", action="append", default=None)
    parser.add_argument("--difficulty", action="append", default=None)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help=(
            "Compatibility audit field for older catalog caps. It is reported "
            "but does not cap task pools or master, sub-agent, or Worker loops. "
            "Use explicit --task/--tasks/--task-file/--task-index selectors "
            "when a fixed subset is intended."
        ),
    )
    parser.add_argument(
        "--run-task-cap",
        type=int,
        default=None,
        help=(
            "Compatibility audit field for older per-round task slices. It is "
            "reported but does not cap master, sub-agent, or Worker loops."
        ),
    )
    parser.add_argument(
        "--round-task-concurrency",
        type=int,
        default=None,
        help=(
            "Run up to this many Harbor tasks in parallel within one HL round. "
            "Rounds, Codex updates, and regression checks remain serial."
        ),
    )
    rotation = parser.add_mutually_exclusive_group()
    rotation.add_argument(
        "--rotate-tasks-per-iteration",
        dest="rotate_tasks_per_iteration",
        action="store_true",
        default=None,
        help=(
            "For catalog campaigns, rotate the evaluated task slice each "
            "iteration instead of reusing the first sample."
        ),
    )
    rotation.add_argument(
        "--no-rotate-tasks-per-iteration",
        dest="rotate_tasks_per_iteration",
        action="store_false",
        help="Reuse the same task slice every iteration.",
    )
    parser.add_argument("--agent", default="hl-worker")
    parser.add_argument("--worker-role", default=None, help="Model role to use from config roles")
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--reasoning-max-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", default=None)
    parser.add_argument("--llm-timeout-seconds", type=int, default=None)
    parser.add_argument("--tool-timeout-seconds", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument(
        "--max-turns-audit",
        dest="max_turns_audit",
        type=int,
        default=None,
        help=(
            "Audit/progress reference for Worker heuristics. This is not a "
            "turn-count stop condition."
        ),
    )
    parser.add_argument(
        "--max-turns",
        dest="max_turns_audit",
        type=int,
        default=None,
        help=(
            "Deprecated alias for --max-turns-audit; retained for old local "
            "scripts and never used as a Worker loop limit."
        ),
    )
    parser.add_argument(
        "--n-attempts",
        type=int,
        default=None,
        help=(
            "Harbor attempts per task. Use 5 for leaderboard-candidate "
            "Terminal-Bench 2.0 evaluations."
        ),
    )
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--trials-config", default="config/trials.yaml")
    parser.add_argument("--jobs-dir", default="jobs")
    parser.add_argument("--memory-path", default="trials")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "Audit-only compatibility field for historical plateau tracking. "
            "No-improvement plateaus never stop the HL loop."
        ),
    )
    parser.add_argument(
        "--disable-patience",
        action="store_true",
        help=(
            "Audit-only compatibility flag; no-improvement plateaus never stop "
            "the HL loop."
        ),
    )
    parser.add_argument(
        "--goal-path",
        default=None,
        help=(
            "Persistent campaign goal path. Defaults to "
            "<memory-path>/goals/<campaign-id>.json so separate campaigns do not "
            "share stale budget state."
        ),
    )
    parser.add_argument(
        "--goal-token-budget",
        type=int,
        default=None,
        help=(
            "Override the HL goal token budget. By default this is a per-iteration "
            "budget window for audit/progress, not a loop stop."
        ),
    )
    parser.add_argument(
        "--goal-token-budget-scope",
        choices=["iteration", "campaign"],
        default=None,
        help="Interpret the goal token budget per HL iteration or across the whole campaign.",
    )
    parser.add_argument(
        "--goal-wall-time-budget-seconds",
        type=int,
        default=None,
        help=(
            "Record a cumulative HL campaign wall-time budget in seconds for "
            "audit/progress. Use 0 to clear it; it is not a stop condition."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "Audit-only host Harbor wait reference in seconds; defaults to "
            "execution.timeout_per_task and does not stop master, sub-agent, "
            "Harbor host-wait, or Worker loops."
        ),
    )
    parser.add_argument(
        "--skip-network-preflight",
        action="store_true",
        help="Skip Docker/apt network checks before launching Harbor",
    )
    parser.add_argument(
        "--network-preflight-mode",
        choices=["quick", "full"],
        default=None,
    )
    parser.add_argument("--network-preflight-timeout", type=int, default=None)
    add_docker_resource_args(parser)
    force_build = parser.add_mutually_exclusive_group()
    force_build.add_argument("--force-build", dest="force_build", action="store_true", default=None)
    force_build.add_argument("--no-force-build", dest="force_build", action="store_false")
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=None,
        help="Disabled: Terminal-Bench 2.0 leaderboard runs must keep task timeouts unchanged.",
    )
    parser.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=None,
        help="Disabled: Terminal-Bench 2.0 leaderboard runs must keep task timeouts unchanged.",
    )
    parser.add_argument(
        "--verifier-timeout-multiplier",
        type=float,
        default=None,
        help="Disabled: Terminal-Bench 2.0 leaderboard runs must keep task timeouts unchanged.",
    )
    parser.add_argument(
        "--agent-setup-timeout-multiplier",
        type=float,
        default=None,
        help="Disabled: Terminal-Bench 2.0 leaderboard runs must keep task timeouts unchanged.",
    )
    parser.add_argument(
        "--environment-build-timeout-multiplier",
        type=float,
        default=None,
        help="Disabled: Terminal-Bench 2.0 leaderboard runs must keep task timeouts unchanged.",
    )
    parser.add_argument("--mounts-json", default=None)
    parser.add_argument("--verifier-env", action="append", default=None)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--no-network-hardened-environment",
        dest="network_hardened_environment",
        action="store_false",
        default=None,
        help="Do not inject the apt mirror Harbor environment wrapper",
    )
    parser.add_argument(
        "--regression-lane",
        choices=["auto", "none", "smoke", "standard", "full"],
        default="auto",
        help="Regression lane to run around Codex updates; auto runs only with --codex-update",
    )
    parser.add_argument("--skip-pre-regression", action="store_true")
    parser.add_argument("--skip-post-regression", action="store_true")
    parser.add_argument("--regression-task-concurrency", type=int, default=None)
    parser.add_argument(
        "--regression-selection-policy",
        choices=["stable-order", "adaptive"],
        default=None,
    )
    baseline_retry = parser.add_mutually_exclusive_group()
    baseline_retry.add_argument(
        "--retry-baseline-pre-regression",
        dest="retry_baseline_pre_regression",
        action="store_true",
        default=None,
        help=(
            "After a pre-regression failure with no active Codex update, rerun "
            "each failed snapshot before quarantine. Disabled by default because "
            "long/unstable solved snapshots can stall learning campaigns."
        ),
    )
    baseline_retry.add_argument(
        "--no-retry-baseline-pre-regression",
        dest="retry_baseline_pre_regression",
        action="store_false",
        help=(
            "Quarantine failed baseline snapshots immediately after the completed "
            "pre-regression gate attributes them."
        ),
    )
    parser.add_argument(
        "--codex-update-interval",
        type=int,
        default=None,
        help=(
            "Audit/reporting field for older configs. Codex update sub-agent "
            "execution is not skipped by this interval."
        ),
    )
    parser.add_argument(
        "--codex-update-min-failures",
        type=int,
        default=None,
        help=(
            "Evidence-strength audit field. A smaller failed-trial count is "
            "recorded but does not block Codex update sub-agent execution."
        ),
    )
    parser.add_argument(
        "--regression-holdout-fraction",
        type=float,
        default=None,
        help=(
            "Target held-out (D_out) share of solved-task snapshots for the "
            "Self-Harness acceptance gate. Overrides regression.holdout_fraction "
            "in trials config. 0 disables the held-out gate."
        ),
    )
    parser.add_argument(
        "--regression-holdout-seed",
        type=int,
        default=None,
        help="Deterministic seed for the held-in/held-out partition.",
    )
    parser.add_argument(
        "--codex-update-cooldown-after-rollback",
        type=int,
        default=None,
        help=(
            "Rollback cooldown audit field. It is recorded in campaign state "
            "but does not stop master or Codex update sub-agent loops."
        ),
    )
    parser.add_argument(
        "--partial-pass-diagnostic-k",
        type=int,
        default=None,
        help=(
            "Audit/reporting target for partial-pass diagnostics. It no "
            "longer controls how many diagnostic sub-agent attempts run, and "
            "is not a master, sub-agent, or Worker loop stop condition."
        ),
    )
    parser.add_argument("--codex-update", action="store_true")
    parser.add_argument("--codex-dry-run", action="store_true")
    parser.add_argument(
        "--stability-run",
        action="store_true",
        help=(
            "Label repeated evaluation as a stability measurement. This flag "
            "does not enable, cap, or bound iterations; --iterations remains "
            "audit-only with or without --codex-update."
        ),
    )
    parser.add_argument("--codex-bin", default=None)
    parser.add_argument("--codex-model", default=None)
    parser.add_argument(
        "--codex-sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default=None,
    )
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument(
        "--codex-timeout-seconds",
        type=int,
        default=None,
        help=(
            "Compatibility/audit field for older configs. Codex update "
            "sub-agent runs are not killed by this value."
        ),
    )
    dirty_baseline = parser.add_mutually_exclusive_group()
    dirty_baseline.add_argument(
        "--codex-allow-dirty-baseline",
        dest="codex_allow_dirty_baseline",
        action="store_true",
        default=None,
        help=(
            "Allow real Codex updates when the repository already has uncommitted "
            "changes. Enabled by default using isolated-delta review."
        ),
    )
    dirty_baseline.add_argument(
        "--no-codex-allow-dirty-baseline",
        dest="codex_allow_dirty_baseline",
        action="store_false",
        help=(
            "Reject Codex updates when the repository already has uncommitted "
            "changes."
        ),
    )
    parser.add_argument("--campaign-id", default=os.environ.get("HL_CAMPAIGN_ID", "local"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted campaign-id by skipping tasks already recorded in its checkpoint",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "Requested campaign rounds. A positive value stops the campaign "
            "after that many persisted round summaries, including summaries "
            "restored by --resume."
        ),
    )
    parser.add_argument(
        "--submit-check",
        action="store_true",
        help="Evaluate submit gate after the campaign without uploading",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Attempt one-shot submit if config and gates allow it",
    )
    parser.add_argument("--submit-trigger-score", type=float, default=None)
    parser.add_argument("--submit-min-tasks-evaluated", type=int, default=None)
    parser.add_argument("--submit-min-attempts-per-task", type=int, default=None)
    parser.add_argument("--submit-visibility", choices=["private", "public"], default=None)
    parser.add_argument("--submit-share-org", action="append", default=None)
    parser.add_argument("--submit-share-user", action="append", default=None)
    parser.add_argument(
        "--submit-share-yes",
        action="store_true",
        help="Pass Harbor upload --yes for non-interactive share confirmation",
    )
    parser.add_argument(
        "--mission-debug",
        action="store_true",
        help="Write a Factory Missions-style debug packet after campaign evidence is summarized",
    )
    parser.add_argument(
        "--mission-debug-output",
        default=None,
        help="Optional mission debug JSON output path",
    )
    parser.add_argument(
        "--mission-debug-max-features",
        type=int,
        default=6,
        help=(
            "Compatibility/audit field for older mission-debug configs. Mission "
            "debug emits all feature candidates; this value is not a stop condition."
        ),
    )
    parser.add_argument(
        "--best-job-dir",
        default=None,
        help="Best Harbor job directory to use for dry-run submit checks",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from bench.harbor import HarborRunner
    from hl.frontier import (
        frontier_path,
        frontier_summary,
        load_frontier,
        update_frontier,
        write_frontier,
    )
    from hl.goals import GoalStore
    from hl.loop import HLLoop
    from hl.memory import FileSystemMemory
    from hl.model_scope import model_scope_from_agent_config
    from meta.codex_update import CodexUpdateEngine
    from scripts.run_trial import (
        _apply_execution_defaults,
        _require_worker_api_key,
        _run_network_preflight,
        resolve_agent_config,
    )

    trials_config = _load_trials_config(Path(args.trials_config))
    _apply_execution_defaults(args, parser)
    iteration_limit = _iteration_limit(args, trials_config, parser)
    round_task_concurrency = _round_task_concurrency(args, trials_config, parser)
    docker_concurrency_budget = validate_docker_concurrency_budget(
        concurrency=round_task_concurrency,
        args=args,
        parser=parser,
        source="execution.round_task_concurrency",
    )
    rate_limit_concurrency = _rate_limit_concurrency_policy(trials_config, parser)
    provider_fail_fast = _provider_fail_fast_policy(trials_config, parser)
    campaign_tasks = _resolve_tasks(
        args,
        trials_config,
        parser,
        dataset_path=args.path,
    )
    _validate_campaign_mode(args, iteration_limit, parser)
    network_preflight_plan = _network_preflight_plan(args)
    memory_path = Path(args.memory_path)
    memory = FileSystemMemory(base_path=str(memory_path))
    campaign_state = _load_or_create_campaign_state(
        memory_path=memory_path,
        campaign_id=args.campaign_id,
        tasks=campaign_tasks,
        resume=args.resume,
        memory=memory,
        parser=parser,
        write=not args.dry_run,
    )
    round_task_concurrency = _initialize_round_task_concurrency(
        campaign_state,
        configured_concurrency=round_task_concurrency,
        policy=rate_limit_concurrency,
    )
    campaign_state["docker_concurrency_budget"] = docker_concurrency_budget
    task_rotation = _task_rotation_plan(
        args,
        trials_config,
        parser,
    )
    _ensure_task_rotation_state(campaign_state, campaign_tasks, task_rotation)
    task_rotation["task_costs"] = _task_duration_costs(
        memory,
        campaign_tasks,
        campaign_state=campaign_state,
    )
    all_pending_tasks = _pending_campaign_tasks(campaign_tasks, campaign_state)
    tasks = _iteration_tasks(
        campaign_tasks=campaign_tasks,
        fixed_tasks=_cap_pending_tasks(
            _fixed_iteration_task_slice(campaign_tasks, all_pending_tasks),
            args.run_task_cap,
            parser,
        ),
        campaign_state=campaign_state,
        rotation_plan=task_rotation,
        parser=parser,
    )
    if not args.dry_run:
        _write_campaign_state(memory_path, args.campaign_id, campaign_state)
    goal_path = _goal_store_path(args, memory_path)
    goal_store = GoalStore(goal_path)
    goal_plan = _goal_plan(goal_store, trials_config, args, parser)
    agent_config = resolve_agent_config(args, parser)
    agent_config.setdefault("goal_path", str(goal_path))
    agent_config.setdefault("memory_path", str(memory_path))
    agent_model_scope = model_scope_from_agent_config(agent_config)
    frontier_file = frontier_path(memory_path, args.campaign_id, agent_model_scope)
    current_frontier = load_frontier(frontier_file)
    codex_config = _resolve_codex_config(args, parser)
    harbor = HarborRunner(
        dataset_path=args.path if not args.dataset else None,
        dataset_name=args.dataset,
        jobs_dir=Path(args.jobs_dir),
        output_dir=memory_path / "runs",
    )
    commands = [
        harbor.build_command(task_id, agent_config, jobs_dir=args.jobs_dir)
        for task_id in tasks
    ]
    regression_lane = _resolve_regression_lane(args, trials_config)
    regression_plan = _regression_plan(
        args,
        memory_path,
        regression_lane,
        trials_config,
        parser,
    )
    update_policy = _codex_update_policy(args, trials_config, parser)

    if args.dry_run:
        submit_plan = None
        if args.submit_check or args.submit:
            fallback_job_dir = commands[0].job_dir if commands else Path(args.jobs_dir)
            submit_plan = _submit_plan_for_dry_run(
                args=args,
                trials_config=trials_config,
                memory_path=memory_path,
                fallback_job_dir=fallback_job_dir,
                tasks_evaluated=len(campaign_tasks),
            )
        print(
            json.dumps(
                {
                    "task": tasks[0] if tasks else None,
                    "tasks": campaign_tasks,
                    "pending_tasks": tasks,
                    "all_pending_tasks": all_pending_tasks,
                    "completed_tasks": _completed_campaign_task_ids(campaign_state),
                    "resume": args.resume,
                    "iterations_requested_audit_only": iteration_limit,
                    "iterations_requested_audit": iteration_limit,
                    "iterations_stop_condition": bool(
                        iteration_limit is not None and iteration_limit > 0
                    ),
                    "loop_limit_contract": _loop_limit_contract(
                        iteration_limit=iteration_limit,
                        args=args,
                        trials_config=trials_config,
                        codex_config=codex_config,
                        goal_plan=goal_plan,
                        update_policy=update_policy,
                        requested_iterations_stop=bool(
                            iteration_limit is not None and iteration_limit > 0
                        ),
                    ),
                    "harbor_command": commands[0].shell_command() if commands else "",
                    "harbor_commands": [command.shell_command() for command in commands],
                    "job_config": commands[0].config if commands else None,
                    "job_configs": [command.config for command in commands],
                    "memory_path": str(memory_path),
                    "goal_path": str(goal_path),
                    "task_timeout_seconds_audit_only": args.timeout,
                    "task_timeout_seconds": args.timeout,
                    "task_timeout_seconds_stop_condition": False,
                    "round_task_concurrency": round_task_concurrency,
                    "docker_concurrency_budget": docker_concurrency_budget,
                    "rate_limit_concurrency": rate_limit_concurrency,
                    "provider_fail_fast": provider_fail_fast,
                    "campaign_mode": _campaign_mode(args, iteration_limit),
                    "codex_update": args.codex_update,
                    "codex_config": codex_config,
                    "codex_update_policy": update_policy,
                    "goal": goal_plan,
                    "regression": regression_plan,
                    "network_preflight": network_preflight_plan,
                    "task_rotation": task_rotation,
                    "guard_convergence": fixed_eval_audit(trials_config),
                    "same_model_frontier": frontier_summary(current_frontier),
                    "same_model_frontier_path": str(frontier_file),
                    "submit": submit_plan,
                    "mission_debug": _mission_debug_plan(args, memory_path),
                },
                indent=2,
            )
        )
        return 0

    _require_worker_api_key(agent_config, parser)

    _ensure_goal_from_config(goal_store, trials_config, args, parser)

    if not tasks:
        parser.error(
            "campaign task selection produced no runnable tasks; choose at least "
            "one task or enable task rotation. An exhausted prior task slice is "
            "not a master-loop stop condition."
        )

    loop = HLLoop(memory=memory, goal_store=goal_store)
    _configure_loop(loop, trials_config, iteration_limit, args, parser)
    # Hide the held-out (D_out) split from the Codex proposer's regression
    # contracts so a held-out regression is a genuine anti-overfit signal.
    loop.regression_holdout_fraction = float(
        regression_plan.get("holdout_fraction") or 0.0
    )
    loop.regression_holdout_seed = int(regression_plan.get("holdout_seed") or 0)
    if args.resume:
        loop.iteration = _resume_loop_iteration_index(campaign_state)

    if not args.skip_network_preflight:
        preflight_result = _run_network_preflight(args, blocking=False)
        _record_network_preflight_event(
            campaign_state,
            result=preflight_result,
            timeout_seconds=args.network_preflight_timeout,
        )
        _write_campaign_state(memory_path, args.campaign_id, campaign_state)

    def on_trial_recorded(**kwargs: Any) -> None:
        trial = kwargs["trial"]
        _record_campaign_trial(
            campaign_state,
            trial,
            iteration=int(kwargs.get("iteration") or 0),
            summary_id=str(kwargs.get("summary_id") or ""),
        )
        _write_campaign_state(memory_path, args.campaign_id, campaign_state)
        checkpoint_report = _build_campaign_report_from_state(
            campaign_id=args.campaign_id,
            tasks=campaign_tasks,
            iteration_limit=iteration_limit,
            campaign_state=campaign_state,
            summaries=summaries,
            memory=memory,
            memory_path=memory_path,
            regression_plan=regression_plan,
            submit_results=submit_results,
            codex_update=args.codex_update,
            trials_config=trials_config,
            update_policy=update_policy,
            round_task_concurrency=round_task_concurrency,
            task_rotation=task_rotation,
            stopped_reason=f"checkpoint after task {trial.task_id}",
            checkpoint=True,
        )
        _write_campaign_checkpoint_report(memory_path, args.campaign_id, checkpoint_report)

    def stop_after_trial(**kwargs: Any) -> bool:
        _maybe_record_provider_fail_fast(
            campaign_state,
            trials=[kwargs["trial"]],
            policy=provider_fail_fast,
            iteration=int(kwargs.get("iteration") or 0),
            summary_id=str(kwargs.get("summary_id") or ""),
        )
        _write_campaign_state(memory_path, args.campaign_id, campaign_state)
        return False

    if args.codex_update:
        loop.system.set_update_engine(
            CodexUpdateEngine(
                repo_root=Path("."),
                codex_bin=codex_config["codex_bin"],
                model=codex_config["model"],
                sandbox=codex_config["sandbox"],
                reasoning_effort=codex_config.get("reasoning_effort"),
                provider_name=codex_config.get("provider_name"),
                provider_base_url=codex_config.get("provider_base_url"),
                provider_env_key=codex_config.get("provider_env_key"),
                provider_wire_api=codex_config.get("provider_wire_api", "responses"),
                provider_requires_openai_auth=codex_config.get(
                    "provider_requires_openai_auth"
                ),
                codex_home=codex_config.get("codex_home"),
                codex_config_home=codex_config.get("codex_config_home"),
                timeout_seconds=codex_config.get("timeout_seconds"),
                allow_dirty_baseline=codex_config.get("allow_dirty_baseline", False),
                events_dir=memory_path / "diffs",
                dry_run=args.codex_dry_run,
                env_file=args.env_file or ".env.local",
                goal_store=goal_store,
            )
        )

    summaries = []
    submit_results = []
    stopped_reason = ""
    codex_update_cooldown_audit = int(update_policy.get("cooldown_after_rollback") or 0)
    runner = HarborLoopRunner(
        harbor=harbor,
        agent_config=agent_config,
        timeout_audit=args.timeout,
        jobs_dir=args.jobs_dir,
    )
    last_codex_update_summary: Any | None = _last_accepted_codex_update_summary(
        campaign_state
    )
    while True:
        if _explicit_requested_iteration_target_complete(
            goal_store,
            iteration_limit=iteration_limit,
            completed_iterations=len(campaign_state.get("summaries") or []),
        ):
            stopped_reason = "explicit local target completion after requested iterations"
            break
        if not loop.should_continue():
            stopped_reason = _campaign_goal_terminal_reason(loop)
            break

        all_pending_tasks = _pending_campaign_tasks(campaign_tasks, campaign_state)
        tasks = _iteration_tasks(
            campaign_tasks=campaign_tasks,
            fixed_tasks=_cap_pending_tasks(
                _fixed_iteration_task_slice(campaign_tasks, all_pending_tasks),
                args.run_task_cap,
                parser,
            ),
            campaign_state=campaign_state,
            rotation_plan=task_rotation,
            parser=parser,
        )
        if not tasks:
            _record_task_epoch_rollover_event(
                campaign_state,
                iteration=loop.iteration + 1,
                reason=(
                    "campaign task selection produced no runnable tasks; "
                    "recording an epoch rollover and rebuilding the selected "
                    "task pool because pool exhaustion is not a master-loop "
                    "stop condition"
                ),
            )
            _write_campaign_state(memory_path, args.campaign_id, campaign_state)
            _ensure_task_rotation_state(campaign_state, campaign_tasks, task_rotation)
            all_pending_tasks = _pending_campaign_tasks(campaign_tasks, campaign_state)
            tasks = _iteration_tasks(
                campaign_tasks=campaign_tasks,
                fixed_tasks=_cap_pending_tasks(
                    _fixed_iteration_task_slice(campaign_tasks, all_pending_tasks),
                    args.run_task_cap,
                    parser,
                ),
                campaign_state=campaign_state,
                rotation_plan=task_rotation,
                parser=parser,
            )
            if not tasks:
                parser.error(
                    "campaign task selection produced no runnable tasks after "
                    "epoch rollover bookkeeping; configure at least one task"
                )

        pre_regression_enabled = _pre_regression_should_run(
            regression_plan,
            iteration=loop.iteration + 1,
            last_codex_update_summary=last_codex_update_summary,
            submit_requested=bool(args.submit or args.submit_check),
            update_policy=update_policy,
        )
        if pre_regression_enabled:
            baseline_pre_regression = last_codex_update_summary is None and not (
                args.submit or args.submit_check
            )
            if baseline_pre_regression:
                _quarantine_known_failed_baseline_snapshots(
                    memory,
                    campaign_state,
                    memory_path,
                    args.campaign_id,
                    model_scope=agent_model_scope,
                    reason=(
                        "stable regression snapshot already has a failed last "
                        "regression run in the current model scope before any "
                        "active Codex update"
                    ),
                )
            pre_result = _run_regression(regression_plan["pre"]["argv"])
            if pre_result.returncode != 0:
                recovered = False
                if last_codex_update_summary is not None:
                    recovered = _recover_from_codex_validation_failure(
                        loop,
                        memory,
                        campaign_state,
                        memory_path,
                        args.campaign_id,
                        last_codex_update_summary,
                        status=pre_result.returncode,
                        failure_marker="codex_update:rolled_back_pre_regression_gate",
                        reason=(
                            "pre-update regression failed with exit code "
                            f"{pre_result.returncode}; accepted Codex diff "
                            "failed a stable regression gate before the next "
                            "iteration"
                        ),
                    )
                    if recovered:
                        last_codex_update_summary = None
                        _record_codex_update_event(
                            campaign_state,
                            action="audit",
                            iteration=loop.iteration + 1,
                            reason=(
                                "rollback cooldown is recorded for audit only and "
                                "does not block master or Codex update sub-agent "
                                f"execution; configured cooldown={codex_update_cooldown_audit}"
                            ),
                        )
                elif not (args.submit or args.submit_check):
                    recovered = _recover_from_baseline_pre_regression_failure(
                        memory,
                        campaign_state,
                        memory_path,
                        args.campaign_id,
                        regression_plan["pre"]["argv"],
                        pre_result,
                        model_scope=agent_model_scope,
                        transient_cooldown_seconds=int(
                            regression_plan.get("transient_cooldown_seconds") or 0
                        ),
                        retry_failed_tasks=bool(
                            regression_plan.get("retry_baseline_failures", False)
                        ),
                        reason=(
                            "pre-update regression failed with exit code "
                            f"{pre_result.returncode}; no accepted Codex diff "
                            "was active, so failed stable snapshots were treated "
                            "as baseline instability instead of a Codex regression"
                        ),
                    )
                if not recovered:
                    return pre_result.returncode

        codex_update_enabled = _codex_update_should_run(
            args,
            update_policy,
            next_iteration=loop.iteration + 1,
            cooldown_audit=codex_update_cooldown_audit,
        )
        update_engine = loop.system.update_engine
        update_run_before = (
            getattr(update_engine, "_last_run", None)
            if codex_update_enabled
            else None
        )
        pre_iteration_packet_id = _active_codex_packet_id(campaign_state)
        if not codex_update_enabled:
            loop.system.update_engine = None

        summary = loop.run_iteration(
            tasks=tasks,
            task_instructions={task_id: "" for task_id in tasks},
            task_contexts={task_id: {"task_id": task_id} for task_id in tasks},
            agent_runner=runner,
            on_trial_recorded=on_trial_recorded,
            task_concurrency=round_task_concurrency,
            update_decision=_codex_update_decision(
                update_policy,
                campaign_state=campaign_state,
                memory_path=memory_path,
                campaign_id=args.campaign_id,
                provider_fail_fast_policy=provider_fail_fast,
            ),
            pre_update_hook=_partial_pass_diagnostic_hook(
                update_policy,
                campaign_state=campaign_state,
                memory=memory,
                memory_path=memory_path,
                campaign_id=args.campaign_id,
            ),
            stop_after_trial=stop_after_trial,
            required_validation_commands=_codex_host_validation_commands(
                regression_plan
            ),
        )
        loop.system.update_engine = update_engine
        update_run_after = (
            getattr(update_engine, "_last_run", None)
            if codex_update_enabled
            else None
        )
        codex_update_diff_path = _accepted_codex_update_diff_path(loop, summary)
        summary_trials = _load_summary_trials(memory, summary)
        round_task_concurrency = _maybe_reduce_round_task_concurrency(
            campaign_state,
            trials=summary_trials,
            current_concurrency=round_task_concurrency,
            policy=rate_limit_concurrency,
            iteration=loop.iteration,
            summary_id=summary.summary_id,
        )
        frontier_before = copy.deepcopy(current_frontier)
        current_frontier = update_frontier(
            current_frontier,
            trials=summary_trials,
            campaign_id=args.campaign_id,
            model_scope=agent_model_scope,
            summary_id=summary.summary_id,
            active_packet_id=pre_iteration_packet_id,
        )
        write_frontier(frontier_file, current_frontier)
        _record_frontier_event(
            campaign_state,
            path=frontier_file,
            model_scope=agent_model_scope,
            summary_id=summary.summary_id,
            frontier=current_frontier,
        )
        frontier_regression = _frontier_regression_gate(
            loop=loop,
            memory=memory,
            campaign_state=campaign_state,
            memory_path=memory_path,
            campaign_id=args.campaign_id,
            summary=summary,
            frontier=current_frontier,
            packet_id=pre_iteration_packet_id,
        )
        if frontier_regression.get("rollback_applied"):
            last_codex_update_summary = None
            _record_codex_update_event(
                campaign_state,
                action="audit",
                iteration=loop.iteration,
                reason=(
                    "frontier rollback recorded configured cooldown as audit "
                    "metadata only; Codex update sub-agent execution is not "
                    f"blocked by cooldown={codex_update_cooldown_audit}"
                ),
            )
        change_evaluation = _evaluate_pending_change_manifest(
            loop=loop,
            memory=memory,
            campaign_state=campaign_state,
            memory_path=memory_path,
            campaign_id=args.campaign_id,
            summary=summary,
            trials=summary_trials,
            frontier_before=frontier_before,
            update_policy=update_policy,
        )
        if change_evaluation and change_evaluation.get("rollback_applied"):
            last_codex_update_summary = None
            _record_codex_update_event(
                campaign_state,
                action="audit",
                iteration=loop.iteration,
                reason=(
                    "change-evaluation rollback recorded configured cooldown "
                    "as audit metadata only; Codex update sub-agent execution "
                    f"is not blocked by cooldown={codex_update_cooldown_audit}"
                ),
            )
        analysis_paths = _write_iteration_analysis_report(
            memory_path=memory_path,
            campaign_id=args.campaign_id,
            summary=summary,
            trials=summary_trials,
            campaign_state=campaign_state,
        )
        if analysis_paths:
            campaign_state.setdefault("analysis_reports", []).append(analysis_paths)
        summaries.append(summary)
        _record_campaign_summary(
            campaign_state,
            summary,
            codex_update_diff_path=codex_update_diff_path,
        )
        if update_run_after is not None and update_run_after is not update_run_before:
            _record_codex_update_run_event(
                campaign_state,
                iteration=loop.iteration,
                summary=summary,
                run=update_run_after,
            )
            provider_failure = _codex_update_provider_failure(update_run_after)
            if provider_failure["reason"]:
                _record_codex_update_event(
                    campaign_state,
                    action="audit",
                    iteration=loop.iteration,
                    reason=(
                        "Codex update sub-agent provider/API failure was recorded "
                        "as evidence but is not a master-loop stop condition: "
                        f"{provider_failure['reason']}"
                    ),
                )
        _advance_task_rotation(campaign_state, task_rotation)
        _write_campaign_state(memory_path, args.campaign_id, campaign_state)

        post_regression_passed = False
        if regression_plan["post"]["enabled"] and summary.patches_applied:
            post_result = _run_regression(regression_plan["post"]["argv"])
            if post_result.returncode != 0:
                recovered = _recover_from_codex_validation_failure(
                    loop,
                    memory,
                    campaign_state,
                    memory_path,
                    args.campaign_id,
                    summary,
                    status=post_result.returncode,
                    failure_marker="codex_update:rolled_back_regression_gate",
                    reason=(
                        "post-update regression failed with exit code "
                        f"{post_result.returncode}; accepted Codex diff is "
                        "worse than the pre-update validation contract"
                    ),
                )
                if recovered:
                    last_codex_update_summary = None
                    _record_codex_update_event(
                        campaign_state,
                        action="audit",
                        iteration=loop.iteration,
                        reason=(
                            "post-regression rollback recorded configured "
                            "cooldown as audit metadata only; Codex update "
                            "sub-agent execution is not blocked by "
                            f"cooldown={codex_update_cooldown_audit}"
                        ),
                    )
                    _write_campaign_state(memory_path, args.campaign_id, campaign_state)
                    continue
                return post_result.returncode
            pending_status = _validate_pending_regression_snapshots(
                regression_plan["post"]["argv"],
                memory,
                summary,
            )
            if pending_status != 0:
                recovered = _recover_from_codex_validation_failure(
                    loop,
                    memory,
                    campaign_state,
                    memory_path,
                    args.campaign_id,
                    summary,
                    status=pending_status,
                    failure_marker="codex_update:rolled_back_pending_regression_gate",
                    reason=(
                        "pending post-update regression failed with exit code "
                        f"{pending_status}; newly captured regression snapshot "
                        "did not survive the accepted Codex diff"
                    ),
                )
                if recovered:
                    last_codex_update_summary = None
                    _record_codex_update_event(
                        campaign_state,
                        action="audit",
                        iteration=loop.iteration,
                        reason=(
                            "pending-regression rollback recorded configured "
                            "cooldown as audit metadata only; Codex update "
                            "sub-agent execution is not blocked by "
                            f"cooldown={codex_update_cooldown_audit}"
                        ),
                    )
                    _write_campaign_state(memory_path, args.campaign_id, campaign_state)
                    continue
                return pending_status
            _mark_pending_regression_snapshots_stable(memory, summary)
            if regression_plan["holdout"]["enabled"]:
                holdout_result = _run_regression(regression_plan["holdout"]["argv"])
                if holdout_result.returncode != 0:
                    recovered = _recover_from_codex_validation_failure(
                        loop,
                        memory,
                        campaign_state,
                        memory_path,
                        args.campaign_id,
                        summary,
                        status=holdout_result.returncode,
                        failure_marker="codex_update:rolled_back_holdout_regression_gate",
                        reason=(
                            "held-out regression failed with exit code "
                            f"{holdout_result.returncode}; accepted Codex diff "
                            "introduced a regression on the held-out (D_out) split "
                            "that was hidden from the proposer, indicating "
                            "overfitting to the shown tasks"
                        ),
                    )
                    if recovered:
                        last_codex_update_summary = None
                        _record_codex_update_event(
                            campaign_state,
                            action="audit",
                            iteration=loop.iteration,
                            reason=(
                                "held-out regression rollback recorded configured "
                                "cooldown as audit metadata only; Codex update "
                                "sub-agent execution is not blocked by "
                                f"cooldown={codex_update_cooldown_audit}"
                            ),
                        )
                        _write_campaign_state(memory_path, args.campaign_id, campaign_state)
                        continue
                    return holdout_result.returncode
            post_regression_passed = regression_lane == "full"
        if _summary_has_accepted_codex_update(summary):
            last_codex_update_summary = _accepted_codex_update_summary(
                summary,
                diff_path=codex_update_diff_path,
            )

        if args.submit_check or args.submit:
            submit_result = _submit_after_summary(
                args=args,
                trials_config=trials_config,
                memory=memory,
                memory_path=memory_path,
                summary=summary,
                full_regression_passed=post_regression_passed,
            )
            submit_results.append(submit_result)
            terminal_submit_failure = (
                submit_result.terminal
                and submit_result.attempted
                and not submit_result.submitted
                and not submit_result.upload_skipped
            )
            if args.submit and (not submit_result.eligible or terminal_submit_failure):
                report = _build_campaign_report_from_state(
                    campaign_id=args.campaign_id,
                    tasks=campaign_tasks,
                    iteration_limit=iteration_limit,
                    campaign_state=campaign_state,
                    summaries=summaries,
                    memory=memory,
                    memory_path=memory_path,
                    regression_plan=regression_plan,
                    submit_results=submit_results,
                    codex_update=args.codex_update,
                    trials_config=trials_config,
                    update_policy=update_policy,
                    round_task_concurrency=round_task_concurrency,
                    task_rotation=task_rotation,
                    stopped_reason="submit gate failed",
                    checkpoint=False,
                )
                report_path = _write_campaign_report(memory_path, args.campaign_id, report)
                mission_debug = _write_mission_debug_packet(
                    args=args,
                    report=report,
                    report_path=report_path,
                    memory_path=memory_path,
                )
                if mission_debug:
                    report["mission_debug"] = mission_debug
                    _write_campaign_report(memory_path, args.campaign_id, report)
                print(json.dumps(report, indent=2))
                return 1
            if submit_result.terminal:
                stopped_reason = "submit terminal action"
                break

        if _guard_convergence_fixed_eval_artifact_complete(
            args=args,
            campaign_state=campaign_state,
            campaign_tasks=campaign_tasks,
        ):
            stopped_reason = "guard-convergence fixed evaluation artifact complete"
            break
        if _explicit_requested_iteration_target_complete(
            goal_store,
            iteration_limit=iteration_limit,
            completed_iterations=len(campaign_state.get("summaries") or []),
        ):
            stopped_reason = "explicit local target completion after requested iterations"
            break

    report = _build_campaign_report(
        campaign_id=args.campaign_id,
        tasks=campaign_tasks,
        iteration_limit=iteration_limit,
        summaries=summaries,
        memory=memory,
        memory_path=memory_path,
        regression_plan=regression_plan,
        submit_results=submit_results,
        codex_update=args.codex_update,
        trials_config=trials_config,
        update_policy=update_policy,
        round_task_concurrency=round_task_concurrency,
        stopped_reason=stopped_reason,
    )
    if campaign_state.get("completed"):
        report = _build_campaign_report_from_state(
            campaign_id=args.campaign_id,
            tasks=campaign_tasks,
            iteration_limit=iteration_limit,
            campaign_state=campaign_state,
            summaries=summaries,
            memory=memory,
            memory_path=memory_path,
            regression_plan=regression_plan,
            submit_results=submit_results,
            codex_update=args.codex_update,
            trials_config=trials_config,
            update_policy=update_policy,
            round_task_concurrency=round_task_concurrency,
            task_rotation=task_rotation,
            stopped_reason=stopped_reason,
            checkpoint=False,
        )
    report_path = _write_campaign_report(memory_path, args.campaign_id, report)
    mission_debug = _write_mission_debug_packet(
        args=args,
        report=report,
        report_path=report_path,
        memory_path=memory_path,
    )
    if mission_debug:
        report["mission_debug"] = mission_debug
        _write_campaign_report(memory_path, args.campaign_id, report)
    print(json.dumps(report, indent=2))
    return 0


class HarborLoopRunner:
    """Adapter from HLLoop's runner shape to HarborRunner.run_task."""

    def __init__(
        self,
        *,
        harbor: Any,
        agent_config: dict[str, object],
        timeout_audit: int | None,
        jobs_dir: str,
    ) -> None:
        self.harbor = harbor
        self.agent_config = agent_config
        self.timeout_audit = timeout_audit
        self.jobs_dir = jobs_dir

    def run(self, instruction: str, context: dict[str, Any]):
        return self.harbor.run_task(
            task_id=str(context["task_id"]),
            agent_config=self.agent_config,
            timeout_audit=self.timeout_audit,
            jobs_dir=self.jobs_dir,
        )


def _new_campaign_state(campaign_id: str, tasks: list[str]) -> dict[str, Any]:
    now = datetime.now().isoformat()
    return {
        "campaign_id": campaign_id,
        "tasks": list(tasks),
        "completed": [],
        "summaries": [],
        "last_accepted_codex_update": None,
        "codex_validation_failures": [],
        "codex_update_events": [],
        "regression_gate_events": [],
        "change_evaluations": [],
        "partial_pass_diagnostics": [],
        "task_epoch_rollovers": [],
        "same_model_frontier": {},
        "analysis_reports": [],
        "failure_class_attempts": [],
        "frontier_regression_events": [],
        "concurrency_events": [],
        "provider_fail_fast_events": [],
        "round_task_concurrency": {},
        "docker_concurrency_budget": {},
        "created_at": now,
        "updated_at": now,
    }


def _load_or_create_campaign_state(
    *,
    memory_path: Path,
    campaign_id: str,
    tasks: list[str],
    resume: bool,
    memory: Any,
    parser: argparse.ArgumentParser,
    write: bool = True,
) -> dict[str, Any]:
    state_path = _campaign_state_path(memory_path, campaign_id)
    if resume and state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError as exc:
            parser.error(f"Could not parse campaign checkpoint {state_path}: {exc}")
        if list(state.get("tasks") or []) != list(tasks):
            parser.error(
                "--resume campaign task list does not match checkpoint. "
                f"Checkpoint: {state_path}"
            )
        state = _prune_missing_campaign_trials(state, memory)
    else:
        state = _new_campaign_state(campaign_id, tasks)
    if write:
        _write_campaign_state(memory_path, campaign_id, state)
    return state


def _prune_missing_campaign_trials(state: dict[str, Any], memory: Any) -> dict[str, Any]:
    completed = []
    for entry in state.get("completed") or []:
        trial_id = str(entry.get("trial_id") or "")
        if not trial_id:
            continue
        try:
            memory.get_trial(trial_id)
        except FileNotFoundError:
            continue
        completed.append(entry)
    state["completed"] = completed
    state["updated_at"] = datetime.now().isoformat()
    return state


def _pending_campaign_tasks(
    tasks: list[str],
    campaign_state: dict[str, Any],
) -> list[str]:
    completed = set(_completed_campaign_task_ids(campaign_state))
    return [task for task in tasks if task not in completed]


def _fixed_iteration_task_slice(
    campaign_tasks: list[str],
    pending_tasks: list[str],
) -> list[str]:
    """Return the next fixed-mode task slice without using pool exhaustion.

    Historical ``completed`` entries are progress memory, not a master-loop stop
    condition. Once every selected task has at least one recorded trial, fixed
    mode starts a new evaluation epoch over the configured task list instead of
    returning an empty slice that would stop the campaign.
    """
    if pending_tasks:
        return list(pending_tasks)
    return list(campaign_tasks)


def _iteration_tasks(
    *,
    campaign_tasks: list[str],
    fixed_tasks: list[str],
    campaign_state: dict[str, Any],
    rotation_plan: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> list[str]:
    if not rotation_plan.get("enabled"):
        return list(fixed_tasks)
    if not campaign_tasks:
        return []
    state = _ensure_task_rotation_state(campaign_state, campaign_tasks, rotation_plan)
    resumed_current = _resume_current_rotation_tasks(campaign_state, state)
    if resumed_current:
        return resumed_current
    order = _rotation_order(state, campaign_tasks, rotation_plan)
    cursor = int(state.get("cursor") or 0)
    selected: list[str] = []
    local_cursor = cursor
    local_cycle = int(state.get("cycle") or 0)
    local_order = list(order)
    seen: set[str] = set()
    target_count = _rotation_batch_target_count(campaign_tasks, rotation_plan)
    while len(selected) < target_count:
        if local_cursor >= len(local_order):
            local_cycle += 1
            local_order = _rotated_task_order(campaign_tasks, rotation_plan, local_cycle)
            local_cursor = 0
        task_id = local_order[local_cursor]
        local_cursor += 1
        if task_id in seen:
            continue
        selected.append(task_id)
        seen.add(task_id)
    state["current"] = selected
    state["next_cursor"] = local_cursor
    state["next_cycle"] = local_cycle
    state["next_order"] = local_order
    campaign_state["updated_at"] = datetime.now().isoformat()
    return selected


def _resume_current_rotation_tasks(
    campaign_state: dict[str, Any],
    rotation_state: dict[str, Any],
) -> list[str]:
    current = rotation_state.get("current")
    if not isinstance(current, list) or not current:
        return []
    if "next_cursor" not in rotation_state:
        return []
    completed = _current_rotation_completed_task_ids(campaign_state)
    remaining = [str(task_id) for task_id in current if str(task_id) not in completed]
    if remaining:
        rotation_state["current"] = remaining
        campaign_state["updated_at"] = datetime.now().isoformat()
    else:
        rotation_state["cursor"] = int(
            rotation_state.get("next_cursor") or rotation_state.get("cursor") or 0
        )
        rotation_state["cycle"] = int(
            rotation_state.get("next_cycle") or rotation_state.get("cycle") or 0
        )
        next_order = rotation_state.get("next_order")
        if isinstance(next_order, list) and next_order:
            rotation_state["order"] = [str(task_id) for task_id in next_order]
        rotation_state.pop("next_cursor", None)
        rotation_state.pop("next_cycle", None)
        rotation_state.pop("next_order", None)
        rotation_state["current"] = []
        rotation_state["updated_at"] = datetime.now().isoformat()
        campaign_state["updated_at"] = datetime.now().isoformat()
    return remaining


def _advance_task_rotation(
    campaign_state: dict[str, Any],
    rotation_plan: dict[str, Any],
) -> None:
    if not rotation_plan.get("enabled"):
        return
    state = campaign_state.get("task_rotation")
    if not isinstance(state, dict):
        return
    state["cursor"] = int(state.get("next_cursor") or state.get("cursor") or 0)
    state["cycle"] = int(state.get("next_cycle") or state.get("cycle") or 0)
    next_order = state.get("next_order")
    if isinstance(next_order, list) and next_order:
        state["order"] = [str(task_id) for task_id in next_order]
    state.pop("next_cursor", None)
    state.pop("next_cycle", None)
    state.pop("next_order", None)
    state["updated_at"] = datetime.now().isoformat()
    campaign_state["updated_at"] = datetime.now().isoformat()


def _resume_loop_iteration_index(campaign_state: dict[str, Any]) -> int:
    """Return the HLLoop iteration counter to use before the next resume run.

    ``HLLoop.run_iteration`` increments before creating ``summary_NNN``. If a
    prior process stopped mid-iteration, completed task records can reference a
    summary that is not yet present in ``campaign_state["summaries"]``. In that
    case resume must reuse that active summary id instead of starting again at
    ``summary_001`` or skipping to the following iteration.
    """

    completed_iterations: list[int] = []
    for entry in campaign_state.get("completed") or []:
        if not isinstance(entry, dict):
            continue
        try:
            iteration = int(entry.get("iteration") or 0)
        except (TypeError, ValueError):
            continue
        if iteration > 0:
            completed_iterations.append(iteration)
    recorded_summary_count = len(_campaign_state_summaries(campaign_state))
    max_completed_iteration = max(completed_iterations, default=0)
    if max_completed_iteration > recorded_summary_count:
        return max_completed_iteration - 1
    return recorded_summary_count


def _pending_current_rotation_tasks(
    campaign_state: dict[str, Any],
    rotation_plan: dict[str, Any],
) -> list[str]:
    if not rotation_plan.get("enabled"):
        return []
    state = campaign_state.get("task_rotation")
    if not isinstance(state, dict) or "next_cursor" not in state:
        return []
    current = state.get("current")
    if not isinstance(current, list) or not current:
        return []
    completed = _current_rotation_completed_task_ids(campaign_state)
    return [str(task_id) for task_id in current if str(task_id) not in completed]


def _current_rotation_completed_task_ids(campaign_state: dict[str, Any]) -> set[str]:
    recorded_summary_ids = {
        str(entry.get("summary_id") or "")
        for entry in campaign_state.get("summaries") or []
        if isinstance(entry, dict)
    }
    recorded_trial_ids = {
        str(trial_id)
        for entry in campaign_state.get("summaries") or []
        if isinstance(entry, dict)
        for trial_id in entry.get("trial_ids") or []
        if str(trial_id)
    }
    active_completed: set[str] = set()
    for entry in campaign_state.get("completed") or []:
        if not isinstance(entry, dict):
            continue
        summary_id = str(entry.get("summary_id") or "")
        task_id = str(entry.get("task_id") or "")
        trial_id = str(entry.get("trial_id") or "")
        if not task_id:
            continue
        if trial_id and recorded_trial_ids and trial_id not in recorded_trial_ids:
            active_completed.add(task_id)
        elif summary_id and summary_id not in recorded_summary_ids:
            active_completed.add(task_id)
    return active_completed


def _checkpoint_pending_tasks(
    *,
    tasks: list[str],
    campaign_state: dict[str, Any],
    rotation_plan: dict[str, Any] | None = None,
) -> list[str]:
    rotation_pending = _pending_current_rotation_tasks(
        campaign_state,
        rotation_plan or {},
    )
    if rotation_pending:
        return rotation_pending
    completed_tasks = set(_completed_campaign_task_ids(campaign_state))
    return [task for task in tasks if task not in completed_tasks]


def _checkpoint_task_counts(
    *,
    tasks: list[str],
    campaign_state: dict[str, Any],
    rotation_plan: dict[str, Any] | None = None,
) -> tuple[int, int]:
    rotation_plan = rotation_plan or {}
    state = campaign_state.get("task_rotation")
    if rotation_plan.get("enabled") and isinstance(state, dict) and "next_cursor" in state:
        current = [str(task_id) for task_id in state.get("current") or []]
        if current:
            active_completed = _current_rotation_completed_task_ids(campaign_state)
            pending = [task_id for task_id in current if task_id not in active_completed]
            return len(current) - len(pending), len(pending)
    completed_tasks = set(_completed_campaign_task_ids(campaign_state))
    pending = [task for task in tasks if task not in completed_tasks]
    return len(completed_tasks), len(pending)


def _ensure_task_rotation_state(
    campaign_state: dict[str, Any],
    campaign_tasks: list[str],
    rotation_plan: dict[str, Any],
) -> dict[str, Any]:
    if not rotation_plan.get("enabled"):
        campaign_state.pop("task_rotation", None)
        return {}
    state = campaign_state.get("task_rotation")
    desired_order = _rotated_task_order(campaign_tasks, rotation_plan, cycle=0)
    if not isinstance(state, dict) or set(state.get("pool") or []) != set(
        campaign_tasks
    ):
        state = {
            "enabled": True,
            "mode": "per_iteration_without_replacement",
            "batch_size": _rotation_batch_target_count(campaign_tasks, rotation_plan),
            "seed": rotation_plan.get("seed", ""),
            "cycle": 0,
            "cursor": 0,
            "pool": list(campaign_tasks),
            "order": desired_order,
            "current": [],
            "created_at": datetime.now().isoformat(),
        }
        campaign_state["task_rotation"] = state
    else:
        state["enabled"] = True
        state["batch_size"] = _rotation_batch_target_count(campaign_tasks, rotation_plan)
        state["seed"] = rotation_plan.get("seed", "")
    state["updated_at"] = datetime.now().isoformat()
    return state


def _rotation_order(
    state: dict[str, Any],
    campaign_tasks: list[str],
    rotation_plan: dict[str, Any],
) -> list[str]:
    order = state.get("order")
    if isinstance(order, list) and set(order) == set(campaign_tasks):
        return [str(task_id) for task_id in order]
    cycle = int(state.get("cycle") or 0)
    refreshed = _rotated_task_order(campaign_tasks, rotation_plan, cycle)
    state["order"] = refreshed
    state["cursor"] = 0
    return refreshed


def _rotated_task_order(
    campaign_tasks: list[str],
    rotation_plan: dict[str, Any],
    cycle: int,
) -> list[str]:
    from random import Random

    order = list(dict.fromkeys(campaign_tasks))
    if not order:
        return []
    seed = f"{rotation_plan.get('seed', '')}:cycle:{cycle}"
    Random(seed).shuffle(order)
    if rotation_plan.get("balance_by_duration"):
        order = _duration_balanced_order(
            order,
            batch_size=_rotation_batch_target_count(order, rotation_plan),
            task_costs=rotation_plan.get("task_costs", {}),
        )
    return order


def _rotation_batch_target_count(
    campaign_tasks: list[str],
    rotation_plan: dict[str, Any],
) -> int:
    if not campaign_tasks:
        return 0
    try:
        configured = int(rotation_plan.get("batch_size") or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured <= 0:
        return len(campaign_tasks)
    return min(configured, len(campaign_tasks))


def _duration_balanced_order(
    task_ids: list[str],
    *,
    batch_size: int,
    task_costs: dict[str, Any],
) -> list[str]:
    if batch_size <= 1 or len(task_ids) <= batch_size:
        return task_ids
    ranked = sorted(
        list(enumerate(task_ids)),
        key=lambda item: (-float(task_costs.get(item[1], 0.0) or 0.0), item[0]),
    )
    split = (len(ranked) + 1) // 2
    heavy = [task_id for _index, task_id in ranked[:split]]
    light = [task_id for _index, task_id in reversed(ranked[split:])]
    max_heavy_per_batch = max(1, batch_size // 3)
    balanced: list[str] = []
    while heavy or light:
        batch: list[str] = []
        while heavy and len(batch) < max_heavy_per_batch:
            batch.append(heavy.pop(0))
        while light and len(batch) < batch_size:
            batch.append(light.pop(0))
        while heavy and len(batch) < batch_size:
            batch.append(heavy.pop(0))
        balanced.extend(batch)
    return balanced


def _cap_pending_tasks(
    pending_tasks: list[str],
    run_task_cap: int | None,
    parser: argparse.ArgumentParser,
) -> list[str]:
    _ = run_task_cap, parser
    return pending_tasks


def _completed_campaign_task_ids(campaign_state: dict[str, Any]) -> list[str]:
    seen: dict[str, None] = {}
    for entry in campaign_state.get("completed") or []:
        task_id = str(entry.get("task_id") or "")
        if task_id:
            seen.setdefault(task_id, None)
    return list(seen)


def _task_duration_costs(
    memory: Any,
    task_ids: list[str],
    *,
    campaign_state: dict[str, Any] | None = None,
) -> dict[str, float]:
    task_set = set(task_ids)
    existing_costs = {}
    if isinstance(campaign_state, dict):
        rotation_state = campaign_state.get("task_rotation")
        if isinstance(rotation_state, dict):
            raw_costs = rotation_state.get("task_costs")
            if isinstance(raw_costs, dict):
                for task_id, value in raw_costs.items():
                    task_key = str(task_id)
                    duration = _float_or_none(value)
                    if task_key in task_set and duration is not None:
                        existing_costs[task_key] = duration
    if task_set and task_set <= set(existing_costs):
        return existing_costs
    durations_by_task: dict[str, list[float]] = {task_id: [] for task_id in task_ids}
    for task_id, duration in existing_costs.items():
        durations_by_task.setdefault(task_id, []).append(duration)
    if isinstance(campaign_state, dict):
        for entry in reversed(campaign_state.get("completed") or []):
            if not isinstance(entry, dict):
                continue
            task_id = str(entry.get("task_id") or "")
            if task_id not in task_set:
                continue
            bucket = durations_by_task[task_id]
            if len(bucket) >= 5:
                continue
            duration = _float_or_none(entry.get("wall_time_seconds"))
            if duration is not None and duration > 0:
                bucket.append(duration)
            if all(len(values) >= 5 for values in durations_by_task.values()):
                break
        campaign_costs = {
            task_id: sum(values) / len(values)
            for task_id, values in durations_by_task.items()
            if values
        }
        if campaign_costs:
            return campaign_costs
    runs_dir = getattr(memory, "runs_dir", None)
    if runs_dir is None or not Path(runs_dir).exists():
        return existing_costs
    scanned = 0
    max_scanned = max(200, len(task_ids) * 8)
    for result_path in sorted(Path(runs_dir).glob("*/result.json"), reverse=True):
        scanned += 1
        try:
            data = json.loads(result_path.read_text(errors="replace"))
        except Exception:
            continue
        task_id = str(data.get("task_id") or "")
        if task_id not in task_set:
            continue
        bucket = durations_by_task[task_id]
        if len(bucket) >= 5:
            continue
        try:
            duration = float(data.get("wall_time_seconds") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            bucket.append(duration)
        if all(len(values) >= 5 for values in durations_by_task.values()):
            break
        if scanned >= max_scanned and any(values for values in durations_by_task.values()):
            break
    return {
        task_id: sum(values) / len(values)
        for task_id, values in durations_by_task.items()
        if values
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_campaign_trial(
    campaign_state: dict[str, Any],
    trial: Any,
    *,
    iteration: int,
    summary_id: str,
) -> None:
    completed = campaign_state.setdefault("completed", [])
    for entry in completed:
        if entry.get("trial_id") == trial.trial_id:
            return
    completed.append(
        {
            "task_id": trial.task_id,
            "trial_id": trial.trial_id,
            "iteration": iteration,
            "summary_id": summary_id,
            "status": trial.status.value,
            "score": trial.score,
            "verified": trial.verified,
            "wall_time_seconds": float(getattr(trial, "wall_time_seconds", 0.0) or 0.0),
            "diagnostic": bool((getattr(trial, "metadata", {}) or {}).get("diagnostic")),
            "completed_at": datetime.now().isoformat(),
        }
    )
    campaign_state["updated_at"] = datetime.now().isoformat()


def _record_campaign_summary(
    campaign_state: dict[str, Any],
    summary: Any,
    *,
    codex_update_diff_path: str = "",
) -> None:
    summaries = campaign_state.setdefault("summaries", [])
    new_trial_ids = list(summary.trial_ids)
    for entry in summaries:
        if entry.get("summary_id") == summary.summary_id and entry.get("trial_ids") == new_trial_ids:
            entry["patches_applied"] = list(summary.patches_applied)
            entry["overall_score"] = summary.overall_score
            if codex_update_diff_path:
                entry["codex_update_diff_path"] = str(codex_update_diff_path)
                entry["codex_update_packet_id"] = _codex_packet_id_from_diff_path(
                    codex_update_diff_path
                )
            entry["recorded_at"] = datetime.now().isoformat()
            _record_last_accepted_codex_update(
                campaign_state,
                summary,
                diff_path=codex_update_diff_path,
            )
            campaign_state["updated_at"] = datetime.now().isoformat()
            return
    summaries.append(
        {
            "summary_id": summary.summary_id,
            "state_summary_id": f"{summary.summary_id}:{len(summaries) + 1}",
            "trial_ids": new_trial_ids,
            "patches_applied": list(summary.patches_applied),
            "overall_score": summary.overall_score,
            "codex_update_diff_path": str(codex_update_diff_path),
            "codex_update_packet_id": _codex_packet_id_from_diff_path(
                codex_update_diff_path
            ),
            "recorded_at": datetime.now().isoformat(),
        }
    )
    _record_last_accepted_codex_update(
        campaign_state,
        summary,
        diff_path=codex_update_diff_path,
    )
    campaign_state["updated_at"] = datetime.now().isoformat()


def _guard_convergence_fixed_eval_artifact_complete(
    *,
    args: argparse.Namespace,
    campaign_state: dict[str, Any],
    campaign_tasks: list[str],
) -> bool:
    if not getattr(args, "guard_convergence_eval", False):
        return False
    expected_tasks = {str(task_id) for task_id in campaign_tasks if str(task_id)}
    if not expected_tasks:
        return False
    completed_by_trial = {
        str(entry.get("trial_id") or ""): entry
        for entry in campaign_state.get("completed") or []
        if isinstance(entry, dict) and str(entry.get("trial_id") or "")
    }
    for summary in reversed(_campaign_state_summaries(campaign_state)):
        trial_ids = {
            str(trial_id)
            for trial_id in summary.get("trial_ids") or []
            if str(trial_id)
        }
        if not trial_ids:
            continue
        summary_tasks = {
            str(completed_by_trial[trial_id].get("task_id") or "")
            for trial_id in trial_ids
            if trial_id in completed_by_trial
            and not bool(completed_by_trial[trial_id].get("diagnostic"))
        }
        if expected_tasks <= summary_tasks:
            return True
    return False


def _record_task_epoch_rollover_event(
    campaign_state: dict[str, Any],
    *,
    iteration: int,
    reason: str,
) -> None:
    events = campaign_state.setdefault("task_epoch_rollovers", [])
    events.append(
        {
            "iteration": int(iteration),
            "reason": reason,
            "task_pool_exhausted_stop_condition": False,
            "task_pool_epoch_rollover_stop_condition": False,
            "fixed_task_epoch_rollover_stop_condition": False,
            "time_round_token_limit_driven": False,
            "recorded_at": datetime.now().isoformat(),
        }
    )
    campaign_state["updated_at"] = datetime.now().isoformat()


def _record_last_accepted_codex_update(
    campaign_state: dict[str, Any],
    summary: Any,
    *,
    diff_path: str = "",
) -> None:
    if not _summary_has_accepted_codex_update(summary):
        return
    existing = campaign_state.get("last_accepted_codex_update")
    if not diff_path and isinstance(existing, dict):
        diff_path = str(existing.get("diff_path") or "")
    packet_id = _codex_packet_id_from_diff_path(diff_path)
    campaign_state["last_accepted_codex_update"] = {
        "packet_id": packet_id,
        "summary_id": str(summary.summary_id),
        "trial_ids": list(summary.trial_ids),
        "patches_applied": list(summary.patches_applied),
        "overall_score": float(getattr(summary, "overall_score", 0.0) or 0.0),
        "diff_path": str(diff_path),
    }


def _active_codex_packet_id(campaign_state: dict[str, Any]) -> str:
    entry = campaign_state.get("last_accepted_codex_update")
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("packet_id") or "")


def _clear_last_accepted_codex_update(campaign_state: dict[str, Any]) -> None:
    campaign_state["last_accepted_codex_update"] = None
    campaign_state["updated_at"] = datetime.now().isoformat()


def _accepted_codex_update_summary_for_packet(
    campaign_state: dict[str, Any],
    summary: Any,
    *,
    packet_id: str,
) -> Any | None:
    if _summary_has_accepted_codex_update(summary):
        summary_packet_id = str(
            getattr(summary, "codex_update_packet_id", "")
            or _codex_packet_id_from_diff_path(
                str(getattr(summary, "codex_update_diff_path", "") or "")
            )
        )
        if not packet_id or not summary_packet_id or summary_packet_id == packet_id:
            return summary

    accepted = _last_accepted_codex_update_summary(campaign_state)
    if accepted is None:
        return None
    accepted_packet_id = str(getattr(accepted, "codex_update_packet_id", "") or "")
    if packet_id and accepted_packet_id and accepted_packet_id != packet_id:
        return None
    return accepted


def _codex_update_packet_id_for_summary(
    campaign_state: dict[str, Any],
    summary: Any,
) -> str:
    packet_id = str(
        getattr(summary, "codex_update_packet_id", "")
        or _codex_packet_id_from_diff_path(
            str(getattr(summary, "codex_update_diff_path", "") or "")
        )
    )
    if packet_id:
        return packet_id
    summary_id = str(getattr(summary, "summary_id", "") or "")
    if not summary_id:
        return ""
    summary_trial_ids = [
        str(trial_id) for trial_id in getattr(summary, "trial_ids", []) or []
    ]
    for entry in reversed(campaign_state.get("summaries") or []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("summary_id") or "") != summary_id:
            continue
        entry_trial_ids = [str(trial_id) for trial_id in entry.get("trial_ids") or []]
        if summary_trial_ids and entry_trial_ids != summary_trial_ids:
            continue
        packet_id = str(entry.get("codex_update_packet_id") or "")
        if packet_id:
            return packet_id
    for event in reversed(campaign_state.get("codex_update_events") or []):
        if not isinstance(event, dict):
            continue
        if str(event.get("summary_id") or "") != summary_id:
            continue
        packet_id = str(event.get("packet_id") or "")
        if packet_id:
            return packet_id
    return ""


def _record_frontier_event(
    campaign_state: dict[str, Any],
    *,
    path: Path,
    model_scope: str,
    summary_id: str,
    frontier: dict[str, Any],
) -> None:
    campaign_state["same_model_frontier"] = {
        "path": str(path),
        "model_scope": model_scope,
        "summary_id": summary_id,
        "aggregate": frontier.get("aggregate", {}),
        "updated_at": datetime.now().isoformat(),
    }
    campaign_state["updated_at"] = datetime.now().isoformat()


def _evaluate_pending_change_manifest(
    *,
    loop: Any,
    memory: Any,
    campaign_state: dict[str, Any],
    memory_path: Path,
    campaign_id: str,
    summary: Any,
    trials: list[Any],
    frontier_before: dict[str, Any],
    update_policy: dict[str, int],
) -> dict[str, Any]:
    from hl.frontier import evaluate_change_manifest

    manifest = _pending_change_manifest(campaign_state, memory)
    if not manifest:
        return {}
    packet_id = str(manifest.get("packet_id") or "")
    if not packet_id:
        return {}
    already_evaluated = {
        str(entry.get("packet_id") or "")
        for entry in campaign_state.get("change_evaluations") or []
        if isinstance(entry, dict)
    }
    if packet_id in already_evaluated:
        return {}

    evaluation = evaluate_change_manifest(
        manifest=manifest,
        trials=trials,
        frontier_before=frontier_before,
        summary_id=str(getattr(summary, "summary_id", "") or ""),
    )
    active_update = campaign_state.get("last_accepted_codex_update")
    if isinstance(active_update, dict):
        for key in (
            "mission_candidate_id",
            "mission_failure_category",
            "mission_selection",
        ):
            value = active_update.get(key)
            if value:
                evaluation[key] = value
    run_dir = _change_manifest_run_dir(manifest)
    if run_dir is not None:
        evaluation_path = run_dir / "change_evaluation.json"
        evaluation_path.write_text(json.dumps(evaluation, indent=2))
        evaluation["path"] = str(evaluation_path)
    evaluations = campaign_state.setdefault("change_evaluations", [])
    evaluations.append(evaluation)
    _attach_next_eval_result_to_failure_attempts(
        campaign_state,
        packet_id=packet_id,
        outcome=str(evaluation.get("outcome") or ""),
    )

    if evaluation.get("rollback_recommended"):
        reason = (
            "change prediction misses exceeded hits for packet "
            f"{packet_id}: hits={evaluation.get('hit_count', 0)} "
            f"misses={evaluation.get('miss_count', 0)}"
        )
        if hasattr(memory, "save_component_lesson"):
            memory.save_component_lesson(
                "codex_update",
                _codex_update_outcome_lesson(
                    "change_evaluation",
                    packet_id=packet_id,
                    outcome=str(evaluation.get("outcome") or ""),
                    reason=reason,
                    mission_candidate_id=str(
                        evaluation.get("mission_candidate_id") or ""
                    ),
                    mission_failure_category=str(
                        evaluation.get("mission_failure_category") or ""
                    ),
                    summary_id=str(getattr(summary, "summary_id", "") or ""),
                    rollback_applied=False,
                ),
            )
        rolled_back = False
        rollback_summary = _accepted_codex_update_summary_for_packet(
            campaign_state,
            summary,
            packet_id=packet_id,
        )
        if rollback_summary is not None:
            rolled_back = _rollback_codex_update_after_failed_validation(
                loop,
                memory,
                rollback_summary,
                reason=reason,
            )
        evaluation["rollback_applied"] = bool(rolled_back)
        rollback_cooldown_audit = (
            int(update_policy.get("cooldown_after_rollback") or 0)
            if rolled_back
            else 0
        )
        evaluation["rollback_cooldown_iterations"] = rollback_cooldown_audit
        evaluation["rollback_cooldown_iterations_audit_only"] = rollback_cooldown_audit
        evaluation["rollback_cooldown_stop_condition"] = False
        evaluation["sub_agent_cooldown_stop_condition"] = False
        if rolled_back:
            if _summary_has_accepted_codex_update(summary):
                _mark_codex_update_rolled_back(
                    summary,
                    "codex_update:rolled_back_prediction_miss",
                )
            _clear_last_accepted_codex_update(campaign_state)

    campaign_state["updated_at"] = datetime.now().isoformat()
    _write_campaign_state(memory_path, campaign_id, campaign_state)
    return evaluation


def _frontier_regression_gate(
    *,
    loop: Any,
    memory: Any,
    campaign_state: dict[str, Any],
    memory_path: Path,
    campaign_id: str,
    summary: Any,
    frontier: dict[str, Any],
    packet_id: str,
) -> dict[str, Any]:
    if not packet_id:
        return {}
    tasks = frontier.get("tasks") if isinstance(frontier, dict) else {}
    if not isinstance(tasks, dict):
        return {}
    regressed_tasks = sorted(
        task_id
        for task_id, entry in tasks.items()
        if packet_id in (entry.get("regressed_after_packet") or [])
    )
    if not regressed_tasks:
        return {}
    event = {
        "packet_id": packet_id,
        "summary_id": str(getattr(summary, "summary_id", "") or ""),
        "regressed_tasks": regressed_tasks,
        "regression_count": len(regressed_tasks),
        "recorded_at": datetime.now().isoformat(),
        "rollback_applied": False,
    }
    active_update = campaign_state.get("last_accepted_codex_update")
    if isinstance(active_update, dict):
        for key in (
            "mission_candidate_id",
            "mission_failure_category",
            "mission_selection",
        ):
            value = active_update.get(key)
            if value:
                event[key] = value
    events = campaign_state.setdefault("frontier_regression_events", [])
    events.append(event)
    reason = (
        "same-model per-task frontier regression after packet "
        f"{packet_id}: {', '.join(regressed_tasks)}"
    )
    if hasattr(memory, "save_component_lesson"):
        memory.save_component_lesson(
            "codex_update",
            _codex_update_outcome_lesson(
                "frontier_regression",
                packet_id=packet_id,
                outcome="frontier_regression",
                reason=reason,
                mission_candidate_id=str(event.get("mission_candidate_id") or ""),
                mission_failure_category=str(
                    event.get("mission_failure_category") or ""
                ),
                summary_id=str(getattr(summary, "summary_id", "") or ""),
                rollback_applied=False,
                regressed_tasks=regressed_tasks,
            ),
        )
    rollback_summary = _accepted_codex_update_summary_for_packet(
        campaign_state,
        summary,
        packet_id=packet_id,
    )
    if rollback_summary is not None:
        event["rollback_applied"] = _rollback_codex_update_after_failed_validation(
            loop,
            memory,
            rollback_summary,
            reason=reason,
        )
        if event["rollback_applied"]:
            if _summary_has_accepted_codex_update(summary):
                _mark_codex_update_rolled_back(
                    summary,
                    "codex_update:rolled_back_frontier_regression",
                )
            _clear_last_accepted_codex_update(campaign_state)
    _attach_next_eval_result_to_failure_attempts(
        campaign_state,
        packet_id=packet_id,
        outcome="frontier_regression",
    )
    campaign_state["updated_at"] = datetime.now().isoformat()
    _write_campaign_state(memory_path, campaign_id, campaign_state)
    return event


def _codex_update_outcome_lesson(
    source: str,
    *,
    packet_id: str,
    outcome: str,
    reason: str,
    mission_candidate_id: str = "",
    mission_failure_category: str = "",
    summary_id: str = "",
    rollback_applied: bool = False,
    regressed_tasks: list[str] | None = None,
) -> str:
    lines = [
        "Codex update outcome evidence.",
        f"source: {source}",
        f"packet_id: {packet_id or 'unknown'}",
        f"outcome: {outcome or 'unknown'}",
        f"summary_id: {summary_id or 'unknown'}",
        f"rollback_applied: {str(bool(rollback_applied)).lower()}",
    ]
    if mission_candidate_id:
        lines.append(f"mission_candidate_id: {mission_candidate_id}")
    if mission_failure_category:
        lines.append(f"mission_failure_category: {mission_failure_category}")
    if regressed_tasks:
        lines.append("regressed_tasks: " + ", ".join(str(item) for item in regressed_tasks))
    lines.append("reason: " + reason)
    return "\n".join(lines)


def _attach_next_eval_result_to_failure_attempts(
    campaign_state: dict[str, Any],
    *,
    packet_id: str,
    outcome: str,
) -> None:
    attempts = campaign_state.get("failure_class_attempts")
    if not isinstance(attempts, list):
        return
    for entry in reversed(attempts):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("packet_id") or "") != packet_id:
            continue
        if _should_replace_next_eval_result(
            str(entry.get("next_eval_result") or ""),
            outcome,
        ):
            entry["next_eval_result"] = outcome
        break


def _should_replace_next_eval_result(current: str, incoming: str) -> bool:
    if not current:
        return bool(incoming)
    if not incoming:
        return False
    priority = {
        "": 0,
        "insufficient_prediction": 1,
        "insufficient_evidence": 1,
        "mixed": 2,
        "prediction_supported": 3,
        "prediction_missed": 4,
        "rollback_applied": 5,
        "validation_failed": 5,
        "frontier_regression": 5,
    }
    return priority.get(incoming, 2) > priority.get(current, 2)


def _pending_change_manifest(
    campaign_state: dict[str, Any],
    memory: Any,
) -> dict[str, Any]:
    entry = campaign_state.get("last_accepted_codex_update")
    manifest: dict[str, Any] = {}
    if isinstance(entry, dict):
        packet_id = str(entry.get("packet_id") or "")
        diff_path = str(entry.get("diff_path") or "")
        if packet_id and diff_path:
            path = Path(diff_path).parent / "change_manifest.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    data = {}
                if isinstance(data, dict):
                    manifest = data
                    manifest.setdefault("path", str(path))
    if manifest:
        return manifest
    return {}


def _change_manifest_run_dir(manifest: dict[str, Any]) -> Path | None:
    manifest_path = str(manifest.get("path") or "")
    if manifest_path:
        return Path(manifest_path).parent
    packet_id = str(manifest.get("packet_id") or "")
    if not packet_id:
        return None
    return None


def _campaign_state_path(memory_path: Path, campaign_id: str) -> Path:
    return memory_path / "summaries" / f"{_safe_campaign_id(campaign_id)}_campaign_state.json"


def _goal_store_path(args: argparse.Namespace, memory_path: Path) -> Path:
    if args.goal_path:
        return Path(args.goal_path)
    return memory_path / "goals" / f"{_safe_campaign_id(args.campaign_id)}.json"


def _write_campaign_state(
    memory_path: Path,
    campaign_id: str,
    campaign_state: dict[str, Any],
) -> Path:
    path = _campaign_state_path(memory_path, campaign_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(campaign_state, indent=2))
    return path


def _write_campaign_checkpoint_report(
    memory_path: Path,
    campaign_id: str,
    report: dict[str, Any],
) -> Path:
    summaries_dir = memory_path / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    path = summaries_dir / f"{_safe_campaign_id(campaign_id)}_campaign.checkpoint.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def _load_trials_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    return data if isinstance(data, dict) else {}


def _iteration_limit(
    args: argparse.Namespace,
    trials_config: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> int | None:
    """Return the requested campaign-round completion target.

    A positive value is enforced against persisted round summaries so resumed
    campaigns cannot silently run additional rounds beyond the requested total.
    """
    configured = trials_config.get("hl_loop", {}).get("max_iterations")
    limit = args.iterations if args.iterations is not None else configured
    if limit is None:
        return None
    return int(limit)


def _validate_campaign_mode(
    args: argparse.Namespace,
    iteration_limit: int | None,
    parser: argparse.ArgumentParser,
) -> None:
    """Keep historical campaign-mode checks from becoming loop limits."""
    _ = args, iteration_limit, parser
    return


def _campaign_mode(args: argparse.Namespace, iteration_limit: int | None) -> str:
    if args.codex_update:
        return "hl_update"
    if args.stability_run:
        return "stability_measurement"
    return "evaluation"


def _explicit_requested_iteration_target_complete(
    goal_store: Any,
    *,
    iteration_limit: int | None,
    completed_iterations: int,
) -> bool:
    if iteration_limit is None or iteration_limit <= 0:
        return False
    if completed_iterations < iteration_limit:
        return False
    goal = goal_store.get_goal() if goal_store is not None else None
    objective = str(getattr(goal, "objective", "") or "").lower()
    if goal is not None and (
        "requested iterations" in objective or "standard task" in objective
    ):
        goal_store.update_goal(
            "complete",
            reason="explicit local target completion after requested iterations",
        )
    return True


def _campaign_goal_terminal_reason(loop: Any) -> str:
    goal_store = getattr(loop, "goal_store", None)
    goal = goal_store.get_goal() if goal_store is not None else None
    if goal is None:
        return "explicit campaign goal terminal action"
    status = str(getattr(goal, "status", "") or "")
    reason = str(getattr(goal, "completion_reason", "") or "").strip()
    if status == "complete":
        return reason or "explicit campaign goal complete"
    if status == "stopped":
        return reason or "explicit user stopped campaign"
    return "explicit campaign goal terminal action"


def _round_task_concurrency(
    args: argparse.Namespace,
    trials_config: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> int:
    raw = args.round_task_concurrency
    if raw is None:
        execution = trials_config.get("execution", {})
        if isinstance(execution, dict):
            raw = execution.get("round_task_concurrency", 1)
    try:
        value = int(raw if raw is not None else 1)
    except (TypeError, ValueError):
        parser.error("execution.round_task_concurrency must be a positive integer")
    if value <= 0:
        parser.error("--round-task-concurrency must be positive")
    return value


def _rate_limit_concurrency_policy(
    trials_config: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    execution = trials_config.get("execution", {})
    raw = execution.get("rate_limit_concurrency", {}) if isinstance(execution, dict) else {}
    if raw is None:
        raw = {}
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    if not isinstance(raw, dict):
        parser.error("execution.rate_limit_concurrency must be a mapping or boolean")

    enabled = bool(raw.get("enabled", True))
    first_fallback = _positive_int_config(
        raw.get("first_fallback", 3),
        "execution.rate_limit_concurrency.first_fallback",
        parser,
    )
    minimum = _positive_int_config(
        raw.get("min", 1),
        "execution.rate_limit_concurrency.min",
        parser,
    )
    restore_after_clean_iterations = _nonnegative_int_config(
        raw.get("restore_after_clean_iterations", 1),
        "execution.rate_limit_concurrency.restore_after_clean_iterations",
        parser,
    )
    if first_fallback < minimum:
        parser.error(
            "execution.rate_limit_concurrency.first_fallback must be >= "
            "execution.rate_limit_concurrency.min"
        )
    return {
        "enabled": enabled,
        "first_fallback": first_fallback,
        "first_fallback_audit_only": first_fallback,
        "first_fallback_stop_condition": False,
        "min": minimum,
        "min_audit_only": minimum,
        "min_stop_condition": False,
        "restore_after_clean_iterations": restore_after_clean_iterations,
        "restore_after_clean_iterations_audit_only": restore_after_clean_iterations,
        "restore_after_clean_iterations_stop_condition": False,
        "controls_round_task_concurrency": False,
        "concurrency_backoff_stop_condition": False,
        "concurrency_restore_stop_condition": False,
        "rate_limit_concurrency_stop_condition": False,
    }


def _provider_fail_fast_policy(
    trials_config: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    execution = trials_config.get("execution", {})
    raw = execution.get("provider_fail_fast", {}) if isinstance(execution, dict) else {}
    if raw is None:
        raw = {}
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    if not isinstance(raw, dict):
        parser.error("execution.provider_fail_fast must be a mapping or boolean")
    return {
        "enabled": bool(raw.get("enabled", True)),
        "billing_quota": bool(raw.get("billing_quota", True)),
    }


def _positive_int_config(
    value: Any,
    name: str,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parser.error(f"{name} must be a positive integer")
    if parsed <= 0:
        parser.error(f"{name} must be a positive integer")
    return parsed


def _nonnegative_int_config(
    value: Any,
    name: str,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parser.error(f"{name} must be a non-negative integer")
    if parsed < 0:
        parser.error(f"{name} must be a non-negative integer")
    return parsed


def _initialize_round_task_concurrency(
    campaign_state: dict[str, Any],
    *,
    configured_concurrency: int,
    policy: dict[str, Any],
) -> int:
    existing = campaign_state.get("round_task_concurrency")
    current = configured_concurrency

    campaign_state["round_task_concurrency"] = {
        "configured": configured_concurrency,
        "current": current,
        "rate_limit_policy": dict(policy),
        "clean_iterations_after_backoff_audit_only": _existing_clean_iterations_after_backoff(
            existing
        ),
        "restore_after_clean_iterations": int(
            policy.get("restore_after_clean_iterations", 1)
        ),
        "restore_after_clean_iterations_audit_only": int(
            policy.get("restore_after_clean_iterations", 1)
        ),
        "rate_limit_concurrency_audit_only": True,
        "rate_limit_concurrency_stop_condition": False,
        "rate_limit_concurrency_controls_current": False,
        "rate_limit_concurrency_backoff_stop_condition": False,
        "rate_limit_concurrency_restore_wait_stop_condition": False,
        "concurrency_backoff_stop_condition": False,
        "concurrency_restore_stop_condition": False,
        "updated_at": datetime.now().isoformat(),
    }
    campaign_state["updated_at"] = datetime.now().isoformat()
    return current


def _maybe_reduce_round_task_concurrency(
    campaign_state: dict[str, Any],
    *,
    trials: list[Any],
    current_concurrency: int,
    policy: dict[str, Any],
    iteration: int,
    summary_id: str,
) -> int:
    if not policy.get("enabled", True):
        return current_concurrency

    rate_limited_tasks = [
        str(getattr(trial, "task_id", ""))
        for trial in trials
        if _trial_has_rate_limit_error(trial)
    ]
    rate_limited_tasks = [task_id for task_id in dict.fromkeys(rate_limited_tasks) if task_id]
    if not rate_limited_tasks:
        _record_rate_limit_concurrency_audit_state(
            campaign_state,
            current_concurrency=current_concurrency,
            policy=policy,
            clean_iteration=True,
        )
        return current_concurrency

    fallback_reference = _next_rate_limit_concurrency(
        current_concurrency,
        first_fallback=int(policy.get("first_fallback") or 3),
        minimum=int(policy.get("min") or 1),
    )
    event = {
        "event": "rate_limit_concurrency_audit",
        "iteration": iteration,
        "summary_id": summary_id,
        "old_concurrency": current_concurrency,
        "new_concurrency": current_concurrency,
        "fallback_reference_audit_only": fallback_reference,
        "rate_limit_concurrency_audit_only": True,
        "rate_limit_concurrency_stop_condition": False,
        "rate_limit_concurrency_controls_current": False,
        "rate_limit_concurrency_backoff_stop_condition": False,
        "rate_limit_concurrency_restore_wait_stop_condition": False,
        "concurrency_backoff_stop_condition": False,
        "concurrency_restore_stop_condition": False,
        "clean_iteration_count_stop_condition": False,
        "restore_after_clean_iterations_stop_condition": False,
        **_non_terminal_loop_event_metadata(),
        "rate_limited_tasks": rate_limited_tasks,
        "reason": (
            "rate-limit evidence recorded without changing round task concurrency; "
            "fallback/min/clean-iteration values are audit-only and do not stop, "
            "delay, truncate, or throttle master, sub-agent, or Worker loops"
        ),
        "created_at": datetime.now().isoformat(),
    }
    campaign_state.setdefault("concurrency_events", []).append(event)
    _record_rate_limit_concurrency_audit_state(
        campaign_state,
        current_concurrency=current_concurrency,
        policy=policy,
        clean_iteration=False,
    )
    campaign_state["updated_at"] = datetime.now().isoformat()
    return current_concurrency


def _record_rate_limit_concurrency_audit_state(
    campaign_state: dict[str, Any],
    *,
    current_concurrency: int,
    policy: dict[str, Any],
    clean_iteration: bool,
) -> None:
    existing = campaign_state.get("round_task_concurrency")
    configured = _round_task_configured_concurrency(existing, current_concurrency)
    clean_iterations = _existing_clean_iterations_after_backoff(existing)
    if clean_iteration:
        clean_iterations += 1
    else:
        clean_iterations = 0
    restore_after = int(policy.get("restore_after_clean_iterations", 1))
    campaign_state["round_task_concurrency"] = {
        "configured": configured,
        "current": current_concurrency,
        "rate_limit_policy": dict(policy),
        "clean_iterations_after_backoff_audit_only": clean_iterations,
        "restore_after_clean_iterations": restore_after,
        "restore_after_clean_iterations_audit_only": restore_after,
        "rate_limit_concurrency_audit_only": True,
        "rate_limit_concurrency_stop_condition": False,
        "rate_limit_concurrency_controls_current": False,
        "rate_limit_concurrency_backoff_stop_condition": False,
        "rate_limit_concurrency_restore_wait_stop_condition": False,
        "concurrency_backoff_stop_condition": False,
        "concurrency_restore_stop_condition": False,
        "clean_iteration_count_stop_condition": False,
        "restore_after_clean_iterations_stop_condition": False,
        "updated_at": datetime.now().isoformat(),
    }


def _maybe_restore_round_task_concurrency(
    campaign_state: dict[str, Any],
    *,
    current_concurrency: int,
    configured_concurrency: int,
    clean_iterations_after_backoff: int,
    restore_after_clean_iterations: int,
    policy: dict[str, Any],
    iteration: int,
    summary_id: str,
) -> int:
    current = max(1, int(current_concurrency))
    configured = max(1, int(configured_concurrency))
    restore_after = max(0, int(restore_after_clean_iterations))
    clean_iterations = max(0, int(clean_iterations_after_backoff)) + 1
    event = {
        "event": "rate_limit_concurrency_restore_audit",
        "iteration": iteration,
        "summary_id": summary_id,
        "old_concurrency": current,
        "new_concurrency": current,
        "configured_concurrency_audit_only": configured,
        "clean_iterations_after_backoff_audit_only": clean_iterations,
        "restore_after_clean_iterations": restore_after,
        "restore_after_clean_iterations_audit_only": restore_after,
        "rate_limit_concurrency_audit_only": True,
        "rate_limit_concurrency_stop_condition": False,
        "rate_limit_concurrency_controls_current": False,
        "rate_limit_concurrency_backoff_stop_condition": False,
        "rate_limit_concurrency_restore_wait_stop_condition": False,
        "concurrency_backoff_stop_condition": False,
        "concurrency_restore_stop_condition": False,
        "clean_iteration_count_stop_condition": False,
        "restore_after_clean_iterations_stop_condition": False,
        **_non_terminal_loop_event_metadata(),
        "reason": (
            "clean rate-limit iterations were recorded without restoring or "
            "changing round task concurrency; clean-iteration and restore-after "
            "counts are audit-only"
        ),
        "created_at": datetime.now().isoformat(),
    }
    campaign_state.setdefault("concurrency_events", []).append(event)
    _write_round_task_concurrency_state(
        campaign_state,
        configured_concurrency=configured,
        current_concurrency=current,
        policy=policy,
        clean_iterations_after_backoff=clean_iterations,
        restore_after_clean_iterations=restore_after,
    )
    return current


def _write_round_task_concurrency_state(
    campaign_state: dict[str, Any],
    *,
    configured_concurrency: int,
    current_concurrency: int,
    policy: dict[str, Any],
    clean_iterations_after_backoff: int,
    restore_after_clean_iterations: int,
) -> None:
    campaign_state["round_task_concurrency"] = {
        "configured": configured_concurrency,
        "current": current_concurrency,
        "rate_limit_policy": dict(policy),
        "clean_iterations_after_backoff_audit_only": max(
            0,
            int(clean_iterations_after_backoff),
        ),
        "restore_after_clean_iterations": max(0, int(restore_after_clean_iterations)),
        "restore_after_clean_iterations_audit_only": max(
            0,
            int(restore_after_clean_iterations),
        ),
        "rate_limit_concurrency_audit_only": True,
        "rate_limit_concurrency_stop_condition": False,
        "rate_limit_concurrency_controls_current": False,
        "rate_limit_concurrency_backoff_stop_condition": False,
        "rate_limit_concurrency_restore_wait_stop_condition": False,
        "concurrency_backoff_stop_condition": False,
        "concurrency_restore_stop_condition": False,
        "clean_iteration_count_stop_condition": False,
        "restore_after_clean_iterations_stop_condition": False,
        "updated_at": datetime.now().isoformat(),
    }
    campaign_state["updated_at"] = datetime.now().isoformat()


def _round_task_configured_concurrency(
    state: Any,
    fallback: int,
) -> int:
    configured = state.get("configured") if isinstance(state, dict) else fallback
    try:
        parsed = int(configured)
    except (TypeError, ValueError):
        parsed = fallback
    return max(1, parsed)


def _existing_clean_iterations_after_backoff(state: Any) -> int:
    if not isinstance(state, dict):
        return 0
    try:
        parsed = int(
            state.get("clean_iterations_after_backoff_audit_only")
            or state.get("clean_iterations_after_backoff")
            or 0
        )
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _next_rate_limit_concurrency(
    current_concurrency: int,
    *,
    first_fallback: int,
    minimum: int,
) -> int:
    current = max(1, int(current_concurrency))
    floor = max(1, int(minimum))
    if current <= floor:
        return current
    if current > first_fallback:
        return max(first_fallback, floor)
    return max(current - 1, floor)


def _trial_has_rate_limit_error(trial: Any) -> bool:
    if _trial_has_provider_billing_quota_error(trial):
        return False
    text = _trial_provider_error_text(trial).lower()
    if not text:
        return False
    if _contains_http_status_code(text, 429) and any(
        marker in text
        for marker in (
            "rate",
            "limit",
            "rate limit",
            "rate limiting",
            "too many",
            "too many requests",
            "tpm",
            "rpm",
            "tokens per minute",
            "requests per minute",
            "quota",
            "resource exhausted",
        )
    ):
        return True
    return any(
        marker in text
        for marker in (
            "too many requests",
            "rate limit",
            "rate-limit",
            "ratelimit",
            "rate_limit",
            "quota exceeded",
            "resource exhausted",
        )
    )


def _maybe_record_provider_fail_fast(
    campaign_state: dict[str, Any],
    *,
    trials: list[Any],
    policy: dict[str, Any],
    iteration: int,
    summary_id: str,
) -> str:
    if not _provider_fail_fast_enabled(policy):
        return ""

    existing_trial_ids = _provider_fail_fast_trial_ids(campaign_state)
    hits: list[dict[str, Any]] = []
    for trial in trials:
        reason = _trial_provider_billing_quota_error_reason(trial)
        if not reason:
            continue
        trial_id = str(getattr(trial, "trial_id", "") or "")
        if trial_id and trial_id in existing_trial_ids:
            continue
        hits.append(
            {
                "task_id": str(getattr(trial, "task_id", "") or ""),
                "trial_id": trial_id,
                "status": str(
                    getattr(getattr(trial, "status", ""), "value", "")
                    or getattr(trial, "status", "")
                    or ""
                ),
                "reason": reason,
            }
        )
    if not hits:
        return ""

    tasks = [hit["task_id"] for hit in hits if hit.get("task_id")]
    event = {
        "event": "provider_billing_quota_fail_fast",
        "action": "audit",
        "iteration": int(iteration),
        "summary_id": str(summary_id),
        "tasks": list(dict.fromkeys(tasks)),
        "trials": hits,
        "reason": (
            "provider billing/quota failure detected; record account-state "
            "evidence without stopping master, sub-agent, or Worker loops"
        ),
        "provider_fail_fast_audit_only": True,
        "provider_fail_fast_stop_condition": False,
        **_non_terminal_loop_event_metadata(),
        "policy": dict(policy),
        "created_at": datetime.now().isoformat(),
    }
    campaign_state.setdefault("provider_fail_fast_events", []).append(event)
    campaign_state["updated_at"] = datetime.now().isoformat()
    return _provider_fail_fast_event_reason(event)


def _provider_fail_fast_enabled(policy: dict[str, Any]) -> bool:
    return bool(policy.get("enabled", True)) and bool(policy.get("billing_quota", True))


def _provider_fail_fast_trial_ids(campaign_state: dict[str, Any]) -> set[str]:
    trial_ids: set[str] = set()
    for event in campaign_state.get("provider_fail_fast_events") or []:
        if not isinstance(event, dict):
            continue
        for hit in event.get("trials") or []:
            if not isinstance(hit, dict):
                continue
            trial_id = str(hit.get("trial_id") or "")
            if trial_id:
                trial_ids.add(trial_id)
    return trial_ids


def _provider_fail_fast_stop_reason(campaign_state: dict[str, Any]) -> str:
    _ = campaign_state
    return ""


def _provider_fail_fast_event_reason(event: dict[str, Any]) -> str:
    tasks = [str(task) for task in event.get("tasks") or [] if str(task)]
    task_text = ", ".join(tasks[:5]) if tasks else "unknown task"
    if len(tasks) > 5:
        task_text += f", +{len(tasks) - 5} more"
    summary_id = str(event.get("summary_id") or "")
    suffix = f" at {summary_id}" if summary_id else ""
    return (
        "provider billing/quota fail-fast audit recorded"
        f"{suffix} for {task_text}; loop stop condition is false"
    )


def _trial_has_provider_billing_quota_error(trial: Any) -> bool:
    return bool(_trial_provider_billing_quota_error_reason(trial))


def _trial_provider_billing_quota_error_reason(trial: Any) -> str:
    return _provider_billing_quota_error_reason_from_text(
        _trial_provider_error_text(trial)
    )


def _provider_billing_quota_error_reason_from_text(text: str) -> str:
    raw = (text or "").lower()
    if not raw:
        return ""
    normalized = " ".join(raw.replace("_", " ").replace("-", " ").split())
    provider_present = any(
        marker in normalized
        for marker in (
            "deepseek",
            "lite llm",
            "litellm",
            "openai",
            "openrouter",
            "anthropic",
            "gemini",
            "siliconflow",
            "llm",
            "provider",
            "api error",
            "apierror",
        )
    )
    if any(
        phrase in normalized
        for phrase in (
            "insufficient balance",
            "balance is not enough",
            "balance not enough",
            "not enough balance",
            "no balance",
            "account balance",
            "balance insufficient",
        )
    ):
        return "provider balance exhausted"
    if any(
        phrase in normalized
        for phrase in (
            "insufficient credit",
            "insufficient credits",
            "out of credit",
            "out of credits",
            "credit exhausted",
            "credits exhausted",
        )
    ):
        return "provider credits exhausted"
    if any(
        phrase in normalized
        for phrase in (
            "payment required",
            "billing hard limit",
            "billing quota",
            "billing details",
            "plan and billing",
        )
    ):
        return "provider billing quota exhausted"
    if any(
        phrase in normalized
        for phrase in (
            "insufficient quota",
            "quota exhausted",
            "quota is exhausted",
            "user quota is exhausted",
            "current quota",
            "exceeded your current quota",
        )
    ):
        return "provider quota exhausted"
    if ("budgetexceeded" in raw or any(
        phrase in normalized
        for phrase in (
            "budget exceeded",
            "budget exhausted",
            "litellm budget",
            "max budget",
        )
    )) and provider_present:
        return "provider budget exhausted"
    if _contains_http_status_code(raw, 402) and (
        provider_present or "payment" in normalized or "billing" in normalized
    ):
        return "provider payment required"
    if "quota exceeded" in normalized and any(
        marker in normalized
        for marker in ("billing", "credit", "credits", "balance", "hard limit")
    ):
        return "provider quota exhausted"
    if "resource exhausted" in normalized and any(
        marker in normalized
        for marker in ("quota", "billing", "credit", "credits", "balance")
    ):
        return "provider quota exhausted"
    if provider_present and "insufficient" in normalized and any(
        marker in normalized for marker in ("quota", "billing", "credit", "balance")
    ):
        return "provider quota exhausted"
    return ""


def _trial_provider_error_text(trial: Any) -> str:
    """Return only provider/LLM-facing error text for quota/rate-limit policy.

    Full trial JSON includes verifier logs, task output, and Worker goal/todo
    tool output. Those surfaces can legitimately contain phrases such as
    "budget exhausted" without indicating provider billing state.
    """

    parts: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                add(item)
            return
        text = str(value)
        if text:
            parts.append(text)

    def add_if_provider_error_surface(value: Any) -> None:
        for item in _flatten_provider_error_values(value):
            if _looks_like_provider_error_surface(item):
                parts.append(item)

    add_if_provider_error_surface(getattr(trial, "error_log", []) or [])
    add_if_provider_error_surface(getattr(trial, "harbor_stdout", "") or "")
    add_if_provider_error_surface(getattr(trial, "harbor_stderr", "") or "")

    metadata = getattr(trial, "metadata", None)
    if isinstance(metadata, dict):
        for key in (
            "agent_exception_type",
            "agent_exception_message",
            "exception_type",
            "exception_message",
            "agent_error_log",
            "llm_error",
            "llm_error_type",
            "provider_error",
            "provider_error_type",
        ):
            add(metadata.get(key))

    add(_trial_harbor_agent_error_text(trial))
    return "\n".join(parts)


def _trial_harbor_agent_error_text(trial: Any) -> str:
    parts: list[str] = []
    seen_paths: set[Path] = set()
    for attr in ("harbor_trial_dir", "harbor_job_dir"):
        value = str(getattr(trial, attr, "") or "")
        if not value:
            continue
        result_path = Path(value) / "result.json"
        if result_path in seen_paths or not result_path.exists():
            continue
        seen_paths.add(result_path)
        try:
            payload = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for selected in _harbor_result_trial_payloads(payload):
            agent_result = selected.get("agent_result") if isinstance(selected, dict) else {}
            agent_metadata = (
                agent_result.get("metadata")
                if isinstance(agent_result, dict)
                else {}
            )
            if isinstance(agent_metadata, dict):
                for key in (
                    "error_log",
                    "llm_error",
                    "llm_error_type",
                    "provider_error",
                    "provider_error_type",
                ):
                    for item in _flatten_provider_error_values(agent_metadata.get(key)):
                        parts.append(item)
            exception = selected.get("exception_info") if isinstance(selected, dict) else {}
            if isinstance(exception, dict):
                for key in ("exception_type", "exception_message"):
                    for item in _flatten_provider_error_values(exception.get(key)):
                        parts.append(item)
    return "\n".join(parts)


def _harbor_result_trial_payloads(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("agent_result"), dict):
        return [payload]
    trials = payload.get("trial_results")
    if isinstance(trials, list):
        return [trial for trial in trials if isinstance(trial, dict)]
    return []


def _flatten_provider_error_values(value: Any) -> list[str]:
    values: list[str] = []
    if value is None:
        return values
    if isinstance(value, (list, tuple, set)):
        for item in value:
            values.extend(_flatten_provider_error_values(item))
        return values
    if isinstance(value, dict):
        for item in value.values():
            values.extend(_flatten_provider_error_values(item))
        return values
    text = str(value)
    if text:
        values.append(text)
    return values


def _looks_like_provider_error_surface(text: str) -> bool:
    raw = (text or "").lower()
    if not raw:
        return False
    normalized = " ".join(raw.replace("_", " ").replace("-", " ").split())
    if any(
        marker in normalized
        for marker in (
            "litellm",
            "lite llm",
            "llm call failed",
            "openai exception",
            "apierror",
            "api error",
            "deepseek",
            "siliconflow",
            "openrouter",
            "anthropic",
            "gemini",
            "provider",
        )
    ):
        return True
    if _contains_http_status_code(raw, 429) and any(
        marker in normalized
        for marker in (
            "rate",
            "rate limit",
            "rate limiting",
            "too many requests",
            "tpm",
            "rpm",
            "tokens per minute",
            "requests per minute",
            "quota exceeded",
            "resource exhausted",
        )
    ):
        return True
    return any(
        phrase in normalized
        for phrase in (
            "too many requests",
            "rate limit",
            "rate limiting",
            "ratelimit",
            "tpm limit",
            "rpm limit",
            "tokens per minute",
            "requests per minute",
            "insufficient balance",
            "balance is not enough",
            "insufficient quota",
            "payment required",
        )
    )


def _contains_http_status_code(text: str, code: int) -> bool:
    return re.search(rf"(?<!\d){int(code)}(?!\d)", text or "") is not None


def _trial_search_text(trial: Any) -> str:
    try:
        payload = trial.model_dump(mode="json")
    except AttributeError:
        payload = getattr(trial, "__dict__", trial)
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        return str(payload)


def _configure_loop(
    loop: Any,
    trials_config: dict[str, Any],
    iteration_limit: int | None,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    loop_config = trials_config.get("hl_loop", {})
    # These are compatibility/progress fields only. The master loop must not
    # stop because of requested iterations or plateau patience, and historical
    # patience inputs must not mutate loop runtime policy.
    loop.max_iterations = None
    loop.patience = None
    _loop_patience_audit_value(args, loop_config, parser)
    if isinstance(loop_config.get("min_improvement"), int | float):
        loop.min_improvement = float(loop_config["min_improvement"])


def _loop_patience_audit_value(
    args: argparse.Namespace,
    loop_config: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> int | None:
    if args.disable_patience:
        return None
    if args.patience is not None:
        return int(args.patience) if args.patience > 0 else None
    if "patience" not in loop_config:
        return None
    configured = loop_config.get("patience")
    if configured is None or configured is False:
        return None
    if isinstance(configured, int):
        return configured if configured > 0 else None
    parser.error("hl_loop.patience must be an integer, 0, false, or null")
    return None


def _resolve_codex_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    from harness.config import ModelsConfig
    from scripts.run_trial import _select_models_config

    model = args.codex_model
    sandbox = args.codex_sandbox
    reasoning_effort = args.codex_reasoning_effort
    timeout_seconds = args.codex_timeout_seconds
    provider_name = None
    provider_base_url = None
    provider_env_key = None
    provider_wire_api = "responses"
    provider_requires_openai_auth = None
    models_path = _select_models_config(args.models_config, parser)
    if models_path is not None:
        models = ModelsConfig.from_yaml(models_path)
        if "orchestrator" in models.roles:
            role = models.get_role("orchestrator")
            model = model or role.model
            sandbox = sandbox or role.sandbox
            reasoning_effort = reasoning_effort or role.reasoning.effort
            timeout_seconds = timeout_seconds or role.timeout_seconds
            provider_base_url = role.base_url
            provider_env_key = role.api_key_env
            provider_name = str(role.extra.get("codex_provider") or "custom")
            provider_wire_api = str(role.extra.get("codex_wire_api") or "responses")
            codex_home = role.extra.get("codex_home")
            codex_config_home = role.extra.get("codex_config_home")
            if "codex_requires_openai_auth" in role.extra:
                provider_requires_openai_auth = bool(
                    role.extra["codex_requires_openai_auth"]
                )
        else:
            codex_home = None
            codex_config_home = None
    else:
        codex_home = None
        codex_config_home = None

    return {
        "codex_bin": args.codex_bin or "codex",
        "model": model or "gpt-5.4",
        "sandbox": sandbox or "workspace-write",
        "reasoning_effort": reasoning_effort or "xhigh",
        "provider_name": provider_name,
        "provider_base_url": provider_base_url,
        "provider_env_key": provider_env_key,
        "provider_wire_api": provider_wire_api,
        "provider_requires_openai_auth": provider_requires_openai_auth,
        "codex_home": str(codex_home) if codex_home else None,
        "codex_config_home": str(codex_config_home) if codex_config_home else None,
        "timeout_seconds": timeout_seconds,
        "allow_dirty_baseline": (
            True
            if args.codex_allow_dirty_baseline is None
            else bool(args.codex_allow_dirty_baseline)
        ),
    }


def _goal_plan(
    goal_store: Any,
    trials_config: dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, Any] | None:
    existing = goal_store.get_goal()
    if existing is not None:
        goal = existing.model_dump(mode="json")
        changed = _apply_goal_cli_overrides(goal, args, parser)
        return {"source": "existing+cli" if changed else "existing", "goal": goal}
    goal = _goal_config(trials_config, args, parser)
    if not goal:
        return None
    return {"source": "config+cli" if _has_goal_cli_overrides(args) else "config", "goal": goal}


def _ensure_goal_from_config(
    goal_store: Any,
    trials_config: dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
):
    existing = goal_store.get_goal()
    if existing is not None:
        if _has_goal_cli_overrides(args):
            wall_time_overridden, wall_time_budget = _validated_goal_wall_time_budget_override(
                args,
                parser,
            )
            budget_updates: dict[str, Any] = {}
            if args.goal_token_budget is not None:
                budget_updates["token_budget"] = _validated_goal_token_budget(args, parser)
            if args.goal_token_budget_scope is not None:
                budget_updates["token_budget_scope"] = args.goal_token_budget_scope
            if wall_time_overridden:
                budget_updates["wall_time_budget_seconds"] = wall_time_budget
            return goal_store.update_budget(
                **budget_updates,
            )
        return existing
    raw = _goal_config(trials_config, args, parser)
    if not raw:
        return None
    return goal_store.create_goal(
        str(raw["objective"]),
        score_target=raw.get("score_target"),
        guard_convergence_score_floor=raw.get("guard_convergence_score_floor"),
        guard_budget_baseline_total=raw.get("guard_budget_baseline_total"),
        guard_budget_target_total=raw.get("guard_budget_target_total"),
        guard_budget_reduction_target_fraction=raw.get(
            "guard_budget_reduction_target_fraction"
        ),
        convergence_plateau_rounds=raw.get("convergence_plateau_rounds"),
        convergence_min_score_delta=raw.get("convergence_min_score_delta"),
        token_budget=raw.get("token_budget"),
        token_budget_scope=raw.get("token_budget_scope", "iteration"),
        wall_time_budget_seconds=raw.get("wall_time_budget_seconds"),
        submit_when_reached=bool(raw.get("submit_when_reached", False)),
    )


def _goal_config(
    trials_config: dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, Any] | None:
    raw = trials_config.get("goal")
    if not isinstance(raw, dict) or not raw.get("objective"):
        return None
    goal = dict(raw)
    if "token_budget_scope" not in goal:
        goal["token_budget_scope"] = "iteration"
    _validate_goal_budget_config(goal, parser)
    _apply_guard_convergence_goal_defaults(goal, trials_config)
    _apply_goal_cli_overrides(goal, args, parser)
    return goal


def _apply_guard_convergence_goal_defaults(
    goal: dict[str, Any],
    trials_config: dict[str, Any],
) -> None:
    guard_eval = fixed_eval_audit(trials_config)
    if guard_eval.get("minimum_accept_score") is not None:
        goal.setdefault(
            "guard_convergence_score_floor",
            guard_eval["minimum_accept_score"],
        )
    raw_guard = trials_config.get("guard_convergence", {})
    guard_budget = raw_guard.get("guard_budget", {}) if isinstance(raw_guard, dict) else {}
    if isinstance(guard_budget, dict):
        baseline_counts = guard_budget.get("baseline_counts", {})
        if (
            isinstance(baseline_counts, dict)
            and baseline_counts.get("total_guard_surface") is not None
        ):
            goal.setdefault(
                "guard_budget_baseline_total",
                baseline_counts.get("total_guard_surface"),
            )
        if guard_budget.get("target_total_guard_surface") is not None:
            goal.setdefault(
                "guard_budget_target_total",
                guard_budget.get("target_total_guard_surface"),
            )
        if guard_budget.get("target_reduction_fraction") is not None:
            goal.setdefault(
                "guard_budget_reduction_target_fraction",
                guard_budget.get("target_reduction_fraction"),
            )
    convergence = raw_guard.get("convergence", {}) if isinstance(raw_guard, dict) else {}
    if isinstance(convergence, dict):
        if convergence.get("plateau_rounds") is not None:
            goal.setdefault(
                "convergence_plateau_rounds",
                convergence.get("plateau_rounds"),
            )
        if convergence.get("min_score_delta") is not None:
            goal.setdefault(
                "convergence_min_score_delta",
                convergence.get("min_score_delta"),
            )


def _has_goal_cli_overrides(args: argparse.Namespace) -> bool:
    return (
        args.goal_token_budget is not None
        or args.goal_token_budget_scope is not None
        or args.goal_wall_time_budget_seconds is not None
    )


def _apply_goal_cli_overrides(
    goal: dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> bool:
    changed = False
    if args.goal_token_budget is not None:
        goal["token_budget"] = _validated_goal_token_budget(args, parser)
        changed = True
    if args.goal_token_budget_scope is not None:
        goal["token_budget_scope"] = args.goal_token_budget_scope
        changed = True
    wall_time_overridden, wall_time_budget = _validated_goal_wall_time_budget_override(
        args,
        parser,
    )
    if wall_time_overridden:
        goal["wall_time_budget_seconds"] = wall_time_budget
        changed = True
    return changed


def _validated_goal_token_budget(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int | None:
    if args.goal_token_budget is None:
        return None
    budget = int(args.goal_token_budget)
    return budget if budget > 0 else None


def _validated_goal_wall_time_budget_override(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[bool, int | None]:
    if args.goal_wall_time_budget_seconds is None:
        return False, None
    if args.goal_wall_time_budget_seconds <= 0:
        return True, None
    return True, int(args.goal_wall_time_budget_seconds)


def _normalize_goal_budget_status(goal: dict[str, Any]) -> None:
    if goal.get("status") != "budget_exhausted":
        return
    goal["status"] = "active"
    goal["legacy_status"] = "budget_exhausted"
    goal["budget_exhaustion_audit_only"] = True
    goal["completion_reason"] = (
        "Legacy budget_exhausted status was normalized to active because "
        "budget, token, and wall-time fields are audit metadata only."
    )


def _validate_goal_budget_config(
    goal: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> None:
    token_budget = goal.get("token_budget")
    if token_budget is not None:
        try:
            normalized_budget = int(token_budget)
        except (TypeError, ValueError):
            parser.error("goal.token_budget must be an integer")
        goal["token_budget"] = normalized_budget if normalized_budget > 0 else None
    token_budget_scope = goal.get("token_budget_scope")
    if token_budget_scope not in {"iteration", "campaign"}:
        parser.error("goal.token_budget_scope must be iteration or campaign")
    wall_time_budget = goal.get("wall_time_budget_seconds")
    if wall_time_budget is not None:
        try:
            normalized_wall_time_budget = int(wall_time_budget)
        except (TypeError, ValueError):
            parser.error("goal.wall_time_budget_seconds must be an integer or null")
        goal["wall_time_budget_seconds"] = (
            normalized_wall_time_budget if normalized_wall_time_budget > 0 else None
        )


def _resolve_tasks(
    args: argparse.Namespace,
    trials_config: dict[str, Any],
    parser: argparse.ArgumentParser,
    *,
    dataset_path: str,
) -> list[str]:
    tasks: list[str] = []
    cli_index_specs: list[str] = []
    if getattr(args, "guard_convergence_eval", False):
        if _has_explicit_task_selection(args) or getattr(args, "task_set", None):
            parser.error(
                "--guard-convergence-eval cannot be combined with explicit task selectors"
            )
        guard_tasks = fixed_eval_task_ids(trials_config)
        if not guard_tasks:
            parser.error(
                "--guard-convergence-eval requires guard_convergence.fixed_eval.tasks"
            )
        return guard_tasks
    if getattr(args, "task", None):
        tasks.extend(args.task)
    if getattr(args, "tasks", None):
        tasks.extend(item.strip() for item in args.tasks.split(","))
    if getattr(args, "task_file", None):
        task_file = Path(args.task_file)
        if not task_file.exists():
            parser.error(f"--task-file path does not exist: {task_file}")
        for line in task_file.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                tasks.append(stripped)
    if getattr(args, "task_index", None):
        cli_index_specs.extend(str(item) for item in args.task_index)
    if getattr(args, "task_indices", None):
        cli_index_specs.append(str(args.task_indices))

    explicit_task_selection = bool(tasks or cli_index_specs)
    if (
        getattr(args, "task_selection", None) in {"random", "full"}
        and explicit_task_selection
    ):
        parser.error(
            "--task-selection random/full cannot be combined with explicit "
            "--task/--tasks/--task-file/--task-index/--task-indices"
        )

    tasks_config = _tasks_config(trials_config)
    selection_config = _task_selection_config(trials_config)
    task_selection = _configured_task_selection_mode(
        args,
        selection_config,
        explicit_task_selection=explicit_task_selection,
        parser=parser,
    )
    _ = _rotate_tasks_per_iteration(
        args,
        selection_config,
        explicit_task_selection=explicit_task_selection,
        task_selection=task_selection,
    )

    domains = args.domain or _list_config(trials_config, "domains")
    difficulties = args.difficulty or _list_config(trials_config, "difficulties")

    catalog = None
    if cli_index_specs or args.task_set or task_selection in {"random", "full", "ids"}:
        catalog = _load_task_catalog(dataset_path, parser)

    if cli_index_specs:
        try:
            indices = _parse_task_index_specs(cli_index_specs)
            selected_by_index = catalog.select_by_indices(
                indices,
                domains=domains,
                difficulties=difficulties,
            )
        except (IndexError, ValueError) as exc:
            parser.error(str(exc))
        if not selected_by_index:
            parser.error("Task index selection selected no tasks")
        tasks.extend(selected_by_index)

    task_set = args.task_set
    if task_set:
        try:
            selected = catalog.select_curriculum(
                str(task_set),
                domains=domains,
                difficulties=difficulties,
                max_tasks=_curriculum_cap(trials_config, str(task_set)),
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        if not selected:
            parser.error(f"Task set {task_set!r} selected no tasks")
        tasks.extend(selected)

    if task_selection and not tasks:
        if task_selection == "full":
            selected = catalog.select_curriculum(
                "full",
                domains=domains,
                difficulties=difficulties,
                max_tasks=None,
            )
        elif task_selection == "random":
            _ = _random_selection_count(
                args,
                trials_config,
                selection_config,
                parser,
            )
            random_seed = _random_selection_seed(
                args,
                selection_config,
                args.campaign_id,
            )
            selection_count = _filtered_catalog_size(
                catalog,
                domains,
                difficulties,
                parser,
            )
            try:
                selected = catalog.select_random(
                    count=selection_count,
                    seed=random_seed,
                    domains=domains,
                    difficulties=difficulties,
                )
            except ValueError as exc:
                parser.error(str(exc))
        else:
            selected = _configured_selection_ids(selection_config)
            index_specs = _configured_selection_index_specs(selection_config)
            if index_specs:
                try:
                    selected.extend(
                        catalog.select_by_indices(
                            _parse_task_index_specs(index_specs),
                            domains=domains,
                            difficulties=difficulties,
                        )
                    )
                except (IndexError, ValueError) as exc:
                    parser.error(str(exc))
            if not selected:
                parser.error(
                    "tasks.selection.mode=ids requires tasks.selection.ids "
                    "or tasks.selection.indices"
                )
        if not selected:
            parser.error(f"Task selection {task_selection!r} selected no tasks")
        tasks.extend(selected)

    if not tasks:
        env_task = os.environ.get("HL_TASK")
        if env_task:
            tasks.append(env_task)
        else:
            configured_task_set = tasks_config.get("task_set")
            if isinstance(configured_task_set, str) and configured_task_set:
                try:
                    catalog = catalog or _load_task_catalog(dataset_path, parser)
                    selected = catalog.select_curriculum(
                        configured_task_set,
                        domains=domains,
                        difficulties=difficulties,
                        max_tasks=_curriculum_cap(trials_config, configured_task_set),
                    )
                except (FileNotFoundError, ValueError) as exc:
                    parser.error(str(exc))
                if not selected:
                    parser.error(f"Task set {configured_task_set!r} selected no tasks")
                tasks.extend(selected)
            if not tasks:
                configured = (
                    tasks_config.get("task_ids")
                    or tasks_config.get("include")
                    or tasks_config.get("smoke")
                )
                if isinstance(configured, str):
                    tasks.extend(item.strip() for item in configured.split(","))
                elif isinstance(configured, list):
                    tasks.extend(str(item).strip() for item in configured)

    tasks = [task for task in tasks if task]
    if not tasks:
        tasks = ["vulnerable-secret"]

    deduped = list(dict.fromkeys(tasks))
    return deduped


def _tasks_config(trials_config: dict[str, Any]) -> dict[str, Any]:
    raw = trials_config.get("tasks", {})
    return raw if isinstance(raw, dict) else {}


def _task_selection_config(trials_config: dict[str, Any]) -> dict[str, Any]:
    raw = _tasks_config(trials_config).get("selection", {})
    return raw if isinstance(raw, dict) else {}


def _configured_task_selection_mode(
    args: argparse.Namespace,
    selection_config: dict[str, Any],
    *,
    explicit_task_selection: bool,
    parser: argparse.ArgumentParser,
) -> str | None:
    task_selection = getattr(args, "task_selection", None)
    if (
        task_selection is None
        and not explicit_task_selection
        and not getattr(args, "task_set", None)
    ):
        configured_mode = selection_config.get("mode")
        if isinstance(configured_mode, str) and configured_mode:
            task_selection = configured_mode.replace("_", "-").lower()
            if task_selection == "specified":
                task_selection = "ids"
            if task_selection not in {"random", "full", "ids"}:
                parser.error(
                    "tasks.selection.mode must be one of random, full, ids, or specified"
                )
    return task_selection


def _rotate_tasks_per_iteration(
    args: argparse.Namespace,
    selection_config: dict[str, Any],
    *,
    explicit_task_selection: bool,
    task_selection: str | None,
) -> bool:
    cli_value = getattr(args, "rotate_tasks_per_iteration", None)
    if cli_value is not None:
        return bool(cli_value)
    if getattr(args, "guard_convergence_eval", False):
        return False
    if explicit_task_selection or getattr(args, "task_set", None):
        return False
    configured = selection_config.get("rotate_each_iteration")
    if isinstance(configured, bool):
        return configured
    if configured is not None:
        enabled = str(configured).strip().lower() in {"1", "true", "yes", "on"}
        return enabled
    return task_selection == "random" and not explicit_task_selection


def _task_rotation_plan(
    args: argparse.Namespace,
    trials_config: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    selection_config = _task_selection_config(trials_config)
    explicit_task_selection = _has_explicit_task_selection(args)
    if getattr(args, "guard_convergence_eval", False):
        fixed_eval = fixed_eval_audit(trials_config)
        return {
            "enabled": False,
            "mode": "fixed_guard_convergence_eval",
            "batch_size": 0,
            "batch_size_audit_only": fixed_eval.get("task_count"),
            "batch_size_controls_task_pool": False,
            "batch_size_controls_batch_size": False,
            "fixed_eval": fixed_eval,
            "fixed_eval_score_acceptance_gate": fixed_eval.get("minimum_accept_score"),
            "fixed_eval_score_regression_stop_condition": False,
            "scope": "guard_convergence_fixed_eval",
            "reason": (
                "fixed guard-reduction safety set; every guard deletion must keep "
                "this set's score at or above minimum_accept_score"
            ),
        }
    task_selection = _configured_task_selection_mode(
        args,
        selection_config,
        explicit_task_selection=explicit_task_selection,
        parser=parser,
    )
    enabled = _rotate_tasks_per_iteration(
        args,
        selection_config,
        explicit_task_selection=explicit_task_selection,
        task_selection=task_selection,
    )
    random_count = _random_selection_count(
        args,
        trials_config,
        selection_config,
        parser,
    )
    batch_size = (
        random_count
        if enabled and task_selection == "random" and random_count is not None and random_count > 0
        else 0
    )
    rotation_pool_count = _rotation_pool_count_audit(trials_config, selection_config)
    run_task_cap = getattr(args, "run_task_cap", None)
    max_tasks_audit = getattr(args, "max_tasks", None)
    max_tasks_per_trial_audit = _tasks_config(trials_config).get("max_tasks_per_trial")
    return {
        "enabled": enabled,
        "mode": "per_iteration_without_replacement" if enabled else "fixed",
        "batch_size": batch_size,
        "batch_size_audit_only": random_count,
        "batch_size_controls_task_pool": False,
        "batch_size_controls_batch_size": bool(batch_size),
        "max_tasks_audit_only": max_tasks_audit,
        "max_tasks_per_trial_audit_only": max_tasks_per_trial_audit,
        "max_tasks_stop_condition": False,
        "max_tasks_controls_task_pool": False,
        "max_tasks_controls_batch_size": False,
        "max_tasks_per_trial_stop_condition": False,
        "max_tasks_per_trial_controls_task_pool": False,
        "max_tasks_per_trial_controls_batch_size": False,
        "run_task_cap_audit_only": run_task_cap,
        "run_task_cap_stop_condition": False,
        "run_task_cap_controls_task_pool": False,
        "run_task_cap_controls_batch_size": False,
        "random_count_audit_only": random_count,
        "random_count_stop_condition": False,
        "random_count_controls_task_pool": False,
        "random_count_controls_batch_size": bool(batch_size),
        "rotation_pool_count_audit_only": rotation_pool_count,
        "rotation_pool_count_stop_condition": False,
        "rotation_pool_count_controls_task_pool": False,
        "rotation_pool_count_controls_batch_size": False,
        "seed": _random_selection_seed(args, selection_config, args.campaign_id),
        "balance_by_duration": bool(selection_config.get("balance_by_duration", False)),
        "scope": (
            "catalog"
            if task_selection == "random" and not explicit_task_selection
            else "explicit"
        ),
        "reason": _task_rotation_reason(enabled=enabled, batch_size=batch_size),
    }


def _task_rotation_reason(*, enabled: bool, batch_size: int | None) -> str:
    if not enabled:
        return "fixed full task slice"
    if batch_size and batch_size > 0:
        return (
            "rotate the full task-pool order in per-iteration task slices; "
            "batch size controls the current slice, not the campaign pool or "
            "master-loop completion"
        )
    return "rotate the full task-pool order without imposing a task-count cap"


def _has_explicit_task_selection(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "task", None)
        or getattr(args, "tasks", None)
        or getattr(args, "task_file", None)
        or getattr(args, "task_index", None)
        or getattr(args, "task_indices", None)
    )


def _load_task_catalog(dataset_path: str, parser: argparse.ArgumentParser):
    from bench.tasks import TaskCatalog

    try:
        return TaskCatalog.from_terminal_bench_path(dataset_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))


def _random_selection_count(
    args: argparse.Namespace,
    trials_config: dict[str, Any],
    selection_config: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> int | None:
    configured = selection_config.get("random_count")
    if not isinstance(configured, int):
        configured = _tasks_config(trials_config).get("random_count")
    if args.random_count is not None:
        count = args.random_count
    else:
        count = configured
    if count is None:
        count = 5
    if count <= 0:
        return None
    return int(count)


def _filtered_catalog_size(
    catalog: Any,
    domains: list[str] | None,
    difficulties: list[str] | None,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        return len(
            catalog.select_curriculum(
                "full",
                domains=domains,
                difficulties=difficulties,
                max_tasks=None,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


def _random_selection_seed(
    args: argparse.Namespace,
    selection_config: dict[str, Any],
    campaign_id: str,
) -> str | int:
    if args.random_seed is not None:
        return args.random_seed
    configured = selection_config.get("random_seed")
    if configured is not None:
        return str(configured)
    return f"campaign:{campaign_id}"


def _rotation_pool_size(
    args: argparse.Namespace,
    trials_config: dict[str, Any],
    selection_config: dict[str, Any],
    catalog: Any,
    domains: list[str] | None,
    difficulties: list[str] | None,
    parser: argparse.ArgumentParser,
) -> int:
    # Historical rotation_pool_count/max_tasks values are audit metadata. They
    # must not shrink the master campaign pool or any sub-agent evidence set.
    _ = getattr(args, "max_tasks", None), _rotation_pool_count_audit(
        trials_config,
        selection_config,
    )
    try:
        return len(
            catalog.select_curriculum(
                "full",
                domains=domains,
                difficulties=difficulties,
                max_tasks=None,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


def _rotation_pool_count_audit(
    trials_config: dict[str, Any],
    selection_config: dict[str, Any],
) -> int | None:
    configured = selection_config.get("rotation_pool_count")
    if not isinstance(configured, int):
        configured = _tasks_config(trials_config).get("rotation_pool_count")
    return configured if isinstance(configured, int) else None


def _configured_selection_ids(selection_config: dict[str, Any]) -> list[str]:
    raw = selection_config.get("ids") or selection_config.get("task_ids")
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _configured_selection_index_specs(selection_config: dict[str, Any]) -> list[str]:
    raw = selection_config.get("indices") or selection_config.get("task_indices")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [str(raw)]


def _parse_task_index_specs(specs: list[str]) -> list[int]:
    indices: list[int] = []
    for spec in specs:
        for part in str(spec).split(","):
            chunk = part.strip()
            if not chunk:
                continue
            if "-" in chunk:
                start_text, end_text = chunk.split("-", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                if start > end:
                    raise ValueError(f"Invalid descending task index range: {chunk}")
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(chunk))
    if not indices:
        raise ValueError("No task indices were provided")
    return indices


def _list_config(trials_config: dict[str, Any], key: str) -> list[str]:
    raw = _tasks_config(trials_config).get(key)
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    if isinstance(raw, str) and raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return []


def _curriculum_cap(trials_config: dict[str, Any], task_set: str) -> int | None:
    normalized = task_set.replace("_", "-").lower()
    if normalized == "full":
        return None
    # Curriculum names are ordering/selection policies only. Historical
    # smoke/domain caps are audit metadata at the campaign layer and must not
    # shrink master, sub-agent, or Worker loops.
    return None


def _resolve_regression_lane(
    args: argparse.Namespace,
    trials_config: dict[str, Any],
) -> str:
    if args.regression_lane != "auto":
        return args.regression_lane
    regression_config = trials_config.get("regression", {})
    if regression_config.get("enabled") is False:
        return "none"
    if not args.codex_update:
        return "none"
    return str(regression_config.get("default_lane") or "smoke")


def _worker_forward_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    optional_args = [
        ("--agent", args.agent),
        ("--worker-role", args.worker_role),
        ("--model", args.model),
        ("--provider", args.provider),
        ("--base-url", args.base_url),
        ("--api-key-env", args.api_key_env),
        ("--reasoning-effort", args.reasoning_effort),
        ("--reasoning-max-tokens", args.reasoning_max_tokens),
        ("--max-output-tokens", args.max_output_tokens),
        ("--llm-timeout-seconds", args.llm_timeout_seconds),
        ("--tool-timeout-seconds", args.tool_timeout_seconds),
        ("--max-retries", args.max_retries),
        ("--max-turns-audit", getattr(args, "max_turns_audit", None)),
        ("--n-attempts", args.n_attempts),
        ("--patience", args.patience),
        ("--models-config", args.models_config),
        ("--trials-config", args.trials_config),
        ("--env-file", args.env_file),
        ("--timeout", args.timeout),
        ("--mounts-json", args.mounts_json),
        ("--network-preflight-mode", args.network_preflight_mode),
        ("--network-preflight-timeout", args.network_preflight_timeout),
    ]
    for flag, value in optional_args:
        if value is not None:
            forwarded.extend([flag, str(value)])
    if args.force_build:
        forwarded.append("--force-build")
    elif args.force_build is False:
        forwarded.append("--no-force-build")
    if args.verifier_env:
        for item in args.verifier_env:
            forwarded.extend(["--verifier-env", item])
    if args.yes:
        forwarded.append("--yes")
    if args.skip_network_preflight:
        forwarded.append("--skip-network-preflight")
    if args.disable_patience:
        forwarded.append("--disable-patience")
    if args.stability_run:
        forwarded.append("--stability-run")
    if args.network_hardened_environment is False:
        forwarded.append("--no-network-hardened-environment")
    forwarded.extend(docker_resource_forward_args(args))
    return forwarded


def _regression_plan(
    args: argparse.Namespace,
    memory_path: Path,
    regression_lane: str,
    trials_config: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    enabled = regression_lane != "none"
    regression_config = trials_config.get("regression", {})
    if not isinstance(regression_config, dict):
        regression_config = {}
    task_concurrency = _positive_config_int(
        args.regression_task_concurrency,
        regression_config.get("task_concurrency"),
        default=1,
        name="regression.task_concurrency",
        parser=parser,
    )
    selection_policy = (
        args.regression_selection_policy
        or str(regression_config.get("selection_policy") or "stable-order")
    )
    if selection_policy not in {"stable-order", "adaptive"}:
        parser.error("regression.selection_policy must be stable-order or adaptive")
    retry_baseline_failures = _config_bool(
        getattr(args, "retry_baseline_pre_regression", None),
        regression_config.get("retry_baseline_failures"),
        default=False,
        name="regression.retry_baseline_failures",
        parser=parser,
    )
    base = [
        sys.executable,
        "scripts/regression_check.py",
        "--memory-path",
        str(memory_path),
        "--path",
        args.path,
        "--jobs-dir",
        args.jobs_dir,
    ]
    if enabled:
        base.extend(
            [
                "--lane",
                regression_lane,
                "--selection-policy",
                selection_policy,
                "--task-concurrency",
                str(task_concurrency),
            ]
        )
    base.extend(_worker_forward_args(args))
    pre_enabled = enabled and not args.skip_pre_regression
    post_enabled = enabled and not args.skip_post_regression

    holdout_fraction = _resolve_holdout_fraction(args, regression_config, parser)
    holdout_seed = _resolve_holdout_seed(args, regression_config)
    holdout_enabled = post_enabled and holdout_fraction > 0
    held_in_argv: list[str] = []
    held_out_argv: list[str] = []
    if holdout_enabled:
        holdout_common = [
            "--holdout-fraction",
            str(holdout_fraction),
            "--holdout-seed",
            str(holdout_seed),
        ]
        held_in_argv = [*base, "--holdout-mode", "held_in", *holdout_common]
        held_out_argv = [*base, "--holdout-mode", "held_out", *holdout_common]

    return {
        "lane": regression_lane,
        "selection_policy": selection_policy,
        "task_concurrency": task_concurrency,
        "transient_cooldown_seconds": int(
            regression_config.get("transient_cooldown_seconds") or 0
        ),
        "retry_baseline_failures": retry_baseline_failures,
        "holdout_fraction": holdout_fraction,
        "holdout_seed": holdout_seed,
        "pre": {
            "enabled": pre_enabled,
            "argv": base if pre_enabled else [],
            "command": _shell_join(base) if pre_enabled else "",
        },
        "post": {
            "enabled": post_enabled,
            # When a held-out split is active, the post (held-in) gate runs only
            # the snapshots the proposer was shown; the held-out gate runs the
            # hidden split. Otherwise post runs the full solved set.
            "argv": (held_in_argv if holdout_enabled else base) if post_enabled else [],
            "command": (
                _shell_join(held_in_argv if holdout_enabled else base)
                if post_enabled
                else ""
            ),
        },
        "holdout": {
            "enabled": holdout_enabled,
            "argv": held_out_argv if holdout_enabled else [],
            "command": _shell_join(held_out_argv) if holdout_enabled else "",
        },
    }


def _resolve_holdout_fraction(
    args: argparse.Namespace,
    regression_config: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> float:
    raw = getattr(args, "regression_holdout_fraction", None)
    if raw is None:
        raw = regression_config.get("holdout_fraction", 0.0)
    try:
        fraction = float(raw)
    except (TypeError, ValueError):
        parser.error("regression.holdout_fraction must be a number in [0, 1]")
    if fraction < 0 or fraction > 1:
        parser.error("regression.holdout_fraction must be in [0, 1]")
    return fraction


def _resolve_holdout_seed(
    args: argparse.Namespace,
    regression_config: dict[str, Any],
) -> int:
    raw = getattr(args, "regression_holdout_seed", None)
    if raw is None:
        raw = regression_config.get("holdout_seed", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _codex_host_validation_commands(regression_plan: dict[str, Any]) -> list[str]:
    """Return campaign-scoped commands for Codex host validation.

    The updater still adds its changed-file validation ladder. These commands
    bind acceptance to the active campaign's memory path, model/role, jobs dir,
    and regression lane, so a patch cannot pass on a generic dry-run while the
    same-model solved-task gate would fail immediately afterward.
    """
    commands = ["pytest tests/ -v"]
    post = regression_plan.get("post") if isinstance(regression_plan, dict) else None
    if isinstance(post, dict) and post.get("enabled") and post.get("argv"):
        argv = [str(item) for item in post.get("argv") or []]
        if "--snapshot-status" not in argv:
            argv = [*argv, "--snapshot-status", "stable"]
        commands.append(_shell_join(argv))
    return commands


def _run_regression(argv: list[str]) -> RegressionRunResult:
    process = subprocess.Popen(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name != "nt"),
    )
    try:
        stdout, stderr = process.communicate()
        returncode = process.returncode
    except BaseException:
        _terminate_process_tree(process)
        raise
    completed = subprocess.CompletedProcess(
        argv,
        returncode,
        stdout or "",
        stderr or "",
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return RegressionRunResult(
        returncode=completed.returncode,
        failed_tasks=_parse_regression_failed_tasks(completed.stdout),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()


def _positive_config_int(
    cli_value: int | None,
    config_value: Any,
    *,
    default: int,
    name: str,
    parser: argparse.ArgumentParser,
) -> int:
    raw = cli_value if cli_value is not None else config_value
    if raw is None:
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        parser.error(f"{name} must be a positive integer")
    if value <= 0:
        parser.error(f"{name} must be a positive integer")
    return value


def _audit_config_int(
    cli_value: int | None,
    config_value: Any,
    *,
    default: int,
    name: str,
    parser: argparse.ArgumentParser,
) -> int:
    """Return a legacy limit/config value that is now audit metadata only."""
    raw = cli_value if cli_value is not None else config_value
    if raw is None:
        raw = default
    try:
        return int(raw)
    except (TypeError, ValueError):
        parser.error(f"{name} must be an integer audit value")
    return default


def _config_bool(
    cli_value: bool | None,
    config_value: Any,
    *,
    default: bool,
    name: str,
    parser: argparse.ArgumentParser,
) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if config_value is None:
        return bool(default)
    if isinstance(config_value, bool):
        return config_value
    normalized = str(config_value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    parser.error(f"{name} must be a boolean")
    return bool(default)


def _nonnegative_config_int(
    cli_value: int | None,
    config_value: Any,
    *,
    default: int,
    name: str,
    parser: argparse.ArgumentParser,
) -> int:
    raw = cli_value if cli_value is not None else config_value
    if raw is None:
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        parser.error(f"{name} must be a non-negative integer")
    if value < 0:
        parser.error(f"{name} must be a non-negative integer")
    return value


def _parse_regression_failed_tasks(stdout: str) -> list[str]:
    failed: list[str] = []
    in_failures = False
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line == "Regressions detected:":
            in_failures = True
            continue
        if in_failures:
            if line.startswith("- "):
                failed.append(line[2:].strip())
                continue
            if line:
                break
    return failed


def _codex_update_policy(
    args: argparse.Namespace,
    trials_config: dict[str, Any],
    parser: argparse.ArgumentParser,
) -> dict[str, int]:
    config = trials_config.get("codex_update", {})
    if not isinstance(config, dict):
        config = {}
    return {
        "interval": _audit_config_int(
            args.codex_update_interval,
            config.get("interval"),
            default=2,
            name="codex_update.interval",
            parser=parser,
        ),
        "min_failures": _audit_config_int(
            args.codex_update_min_failures,
            config.get("min_failures"),
            default=2,
            name="codex_update.min_failures",
            parser=parser,
        ),
        "cooldown_after_rollback": _audit_config_int(
            args.codex_update_cooldown_after_rollback,
            config.get("cooldown_after_rollback"),
            default=2,
            name="codex_update.cooldown_after_rollback",
            parser=parser,
        ),
        "partial_pass_diagnostic_k": _audit_config_int(
            args.partial_pass_diagnostic_k,
            config.get("partial_pass_diagnostic_k"),
            default=2,
            name="codex_update.partial_pass_diagnostic_k",
            parser=parser,
        ),
    }


def _codex_update_should_run(
    args: argparse.Namespace,
    update_policy: dict[str, int],
    *,
    next_iteration: int,
    cooldown_audit: int,
) -> bool:
    _ = update_policy, next_iteration, cooldown_audit
    if not args.codex_update:
        return False
    return True


def _codex_update_decision(
    update_policy: dict[str, Any],
    *,
    campaign_state: dict[str, Any] | None = None,
    memory_path: Path | None = None,
    campaign_id: str = "",
    provider_fail_fast_policy: dict[str, Any] | None = None,
):
    def decide(*, summary: Any, failed_trials: list[Any]) -> bool:
        try:
            iteration = int(str(summary.summary_id).rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            iteration = 0
        if (
            provider_fail_fast_policy is not None
            and _provider_fail_fast_enabled(provider_fail_fast_policy)
            and any(
                _trial_has_provider_billing_quota_error(trial)
                for trial in failed_trials
            )
        ):
            if campaign_state is not None:
                _maybe_record_provider_fail_fast(
                    campaign_state,
                    trials=failed_trials,
                    policy=provider_fail_fast_policy,
                    iteration=iteration,
                    summary_id=str(getattr(summary, "summary_id", "") or ""),
                )
                _record_codex_update_event(
                    campaign_state,
                    action="audit",
                    iteration=iteration,
                    reason=(
                        "provider billing/quota fail-fast audit recorded; "
                        "this account-state signal is not a Codex update "
                        "sub-agent stop condition"
                    ),
                )
                if memory_path is not None and campaign_id:
                    _write_campaign_state(memory_path, campaign_id, campaign_state)

        actionable_failed_trials = [
            trial for trial in failed_trials if not _trial_is_infrastructure_failure(trial)
        ]
        if not actionable_failed_trials and failed_trials:
            if campaign_state is not None:
                _record_codex_update_event(
                    campaign_state,
                    action="audit",
                    iteration=iteration,
                    reason=(
                        "all failed trials were infrastructure/environment failures; "
                        "recording attribution context while still allowing the "
                        "Codex update sub-agent to run because infrastructure, "
                        "Harbor, and environment evidence can require harness "
                        "policy fixes and must not become a sub-agent stop condition"
                    ),
                )
                if memory_path is not None and campaign_id:
                    _write_campaign_state(memory_path, campaign_id, campaign_state)

        min_failures = int(update_policy.get("min_failures", 1))
        if (
            min_failures > 0
            and len(actionable_failed_trials) < min_failures
            and campaign_state is not None
        ):
            _record_codex_update_event(
                campaign_state,
                action="audit",
                iteration=iteration,
                reason=(
                    f"only {len(actionable_failed_trials)} actionable failed trial(s) "
                    f"after excluding {len(failed_trials) - len(actionable_failed_trials)} "
                    "infrastructure failure(s); "
                    f"codex_update.min_failures is {min_failures}, but this "
                    "minimum is an audit/evidence-strength field and does not "
                    "block Codex update sub-agent execution"
                ),
            )
            if memory_path is not None and campaign_id:
                _write_campaign_state(memory_path, campaign_id, campaign_state)
        return True

    return decide


def _partial_pass_diagnostic_hook(
    update_policy: dict[str, int],
    *,
    campaign_state: dict[str, Any],
    memory: Any,
    memory_path: Path,
    campaign_id: str,
):
    diagnostic_sample_target_audit = int(update_policy.get("partial_pass_diagnostic_k", 1))

    def hook(
        *,
        summary: Any,
        failed_trials: list[Any],
        agent_runner: Any,
        task_instructions: dict[str, str],
        task_contexts: dict[str, dict[str, Any]],
    ) -> list[Any]:
        candidates = _partial_pass_candidates(memory, failed_trials)
        if not candidates:
            return []
        diagnostics: list[Any] = []
        for task_id in candidates:
            context = dict(task_contexts.get(task_id, {}))
            context["task_id"] = task_id
            trial = agent_runner.run(task_instructions.get(task_id, ""), context)
            metadata = getattr(trial, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                trial.metadata = metadata
            metadata.update(
                {
                    "diagnostic": True,
                    "diagnostic_type": "partial_pass_k",
                    "diagnostic_sample_reason": "historical pass/fail evidence",
                    "diagnostic_sample_target": diagnostic_sample_target_audit,
                    "diagnostic_sample_target_audit_only": diagnostic_sample_target_audit,
                    "diagnostic_sample_target_stop_condition": False,
                    "diagnostic_attempt_index_stop_condition": False,
                    "diagnostic_target_k": diagnostic_sample_target_audit,
                    "diagnostic_target_k_audit_only": diagnostic_sample_target_audit,
                    "diagnostic_target_k_stop_condition": False,
                    "partial_pass_diagnostic_k_stop_condition": False,
                    "diagnostic_attempt_count_stop_condition": False,
                    "diagnostic_attempt_count_controlled_by_target_k": False,
                    "diagnostic_round_limit_stop_condition": False,
                    "sub_agent_attempt_count_stop_condition": False,
                    "source_summary_id": str(getattr(summary, "summary_id", "") or ""),
                }
            )
            diagnostics.append(trial)
        _record_partial_pass_diagnostic(
            campaign_state,
            summary=summary,
            source_failed_trials=failed_trials,
            diagnostics=diagnostics,
            diagnostic_sample_target=diagnostic_sample_target_audit,
        )
        _write_campaign_state(memory_path, campaign_id, campaign_state)
        return diagnostics

    return hook


def _partial_pass_candidates(memory: Any, failed_trials: list[Any]) -> list[str]:
    candidates: list[str] = []
    for trial in failed_trials:
        task_id = str(getattr(trial, "task_id", "") or "")
        if not task_id:
            continue
        if _task_has_pass_and_fail_history(memory, task_id):
            candidates.append(task_id)
    return list(dict.fromkeys(candidates))


def _task_has_pass_and_fail_history(memory: Any, task_id: str) -> bool:
    if not hasattr(memory, "list_trials") or not hasattr(memory, "get_trial"):
        return False
    has_pass = False
    has_fail = False
    try:
        trial_ids = memory.list_trials(task_id)
    except Exception:
        return False
    for trial_id in trial_ids:
        try:
            trial = memory.get_trial(trial_id)
        except Exception:
            continue
        score = float(getattr(trial, "score", 0.0) or 0.0)
        status = str(getattr(getattr(trial, "status", None), "value", "") or "")
        if score >= 1.0 or status == "passed":
            has_pass = True
        elif status in {"failed", "timeout", "error", "cancelled"} or score < 1.0:
            has_fail = True
        if has_pass and has_fail:
            return True
    return False


def _record_partial_pass_diagnostic(
    campaign_state: dict[str, Any],
    *,
    summary: Any,
    source_failed_trials: list[Any],
    diagnostics: list[Any],
    diagnostic_sample_target: int,
) -> None:
    if not diagnostics:
        return
    events = campaign_state.setdefault("partial_pass_diagnostics", [])
    events.append(
        {
            "summary_id": str(getattr(summary, "summary_id", "") or ""),
            "target_k": int(diagnostic_sample_target),
            "target_k_audit_only": int(diagnostic_sample_target),
            "diagnostic_sample_target": int(diagnostic_sample_target),
            "diagnostic_sample_target_audit_only": int(diagnostic_sample_target),
            "diagnostic_sample_target_stop_condition": False,
            "target_k_stop_condition": False,
            "partial_pass_diagnostic_k_stop_condition": False,
            "diagnostic_attempt_count_stop_condition": False,
            "diagnostic_attempt_count_controlled_by_target_k": False,
            "diagnostic_round_limit_stop_condition": False,
            "sub_agent_attempt_count_stop_condition": False,
            "source_failed_trials": [
                str(getattr(trial, "trial_id", "") or "") for trial in source_failed_trials
            ],
            "diagnostic_trials": [
                {
                    "trial_id": str(getattr(trial, "trial_id", "") or ""),
                    "task_id": str(getattr(trial, "task_id", "") or ""),
                    "status": str(getattr(getattr(trial, "status", None), "value", "") or ""),
                    "score": float(getattr(trial, "score", 0.0) or 0.0),
                    "verified": bool(getattr(trial, "verified", False)),
                }
                for trial in diagnostics
            ],
            "recorded_at": datetime.now().isoformat(),
        }
    )
    campaign_state["updated_at"] = datetime.now().isoformat()


def _write_iteration_analysis_report(
    *,
    memory_path: Path,
    campaign_id: str,
    summary: Any,
    trials: list[Any],
    campaign_state: dict[str, Any],
) -> dict[str, str]:
    if not trials:
        return {}
    analysis_dir = (
        memory_path
        / "analysis"
        / _safe_campaign_id(campaign_id)
        / str(getattr(summary, "summary_id", "") or "summary")
    )
    detail_dir = analysis_dir / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)
    overview_path = analysis_dir / "overview.md"
    summary_path = analysis_dir / "summary.json"

    policy_coverage = _analysis_policy_coverage(trials)
    detail_paths: dict[str, str] = {}
    trajectory_evidence: dict[str, Any] = {}
    for trial in trials:
        detail_path = detail_dir / f"{_safe_campaign_id(str(trial.task_id))}.md"
        evidence = _analysis_trajectory_evidence(trial)
        detail_path.write_text(_analysis_detail_markdown(trial, evidence=evidence))
        detail_paths[str(trial.task_id)] = str(detail_path)
        trajectory_evidence[str(trial.task_id)] = evidence

    failure_buckets = _analysis_failure_buckets_with_mechanisms(
        _analysis_failure_buckets(trials),
        trials,
    )
    candidate_update_classes = _candidate_update_classes(failure_buckets)
    mechanism_update_entries = _analysis_mechanism_update_entries(
        failure_buckets=failure_buckets,
        trajectory_evidence=trajectory_evidence,
    )
    mechanism_update_classes = _format_analysis_mechanism_update_classes(
        mechanism_update_entries
    )
    weakness_signatures = _analysis_weakness_signatures(
        trials,
        failure_buckets=failure_buckets,
        trajectory_evidence=trajectory_evidence,
    )
    infra_failure_count = sum(1 for trial in trials if _trial_is_infrastructure_failure(trial))
    terminal_environment_signal_count = sum(
        1 for trial in trials if _trial_has_terminal_environment_signal(trial)
    )
    overview_path.write_text(
        "\n".join(
            [
                f"# Analysis {campaign_id} {getattr(summary, 'summary_id', '')}",
                "",
                f"- Overall score: {float(getattr(summary, 'overall_score', 0.0) or 0.0):.4f}",
                f"- Trials: {len(trials)}",
                f"- Passed: {sum(1 for trial in trials if getattr(getattr(trial, 'status', None), 'value', '') == 'passed')}",
                f"- Failed/timeout/error: {sum(1 for trial in trials if getattr(getattr(trial, 'status', None), 'value', '') != 'passed')}",
                f"- Infrastructure failures included as Codex update evidence: {infra_failure_count}",
                f"- Terminal environment unavailable signals: {terminal_environment_signal_count}",
                "",
                "## Failure Buckets",
                _markdown_table(
                    ["category", "count", "infrastructure", "tasks", "components", "timeout_phases"],
                    [
                        [
                            item["failure_category"],
                            str(item["count"]),
                            "yes" if item.get("infrastructure") else "no",
                            ", ".join(item["task_ids"]),
                            ", ".join(item["affected_components"]),
                            ", ".join(item["timeout_phases"]),
                        ]
                        for item in failure_buckets
                    ],
                ),
                "",
                "## Candidate Update Classes",
                "\n".join(f"- {item}" for item in candidate_update_classes) or "- none",
                "",
                "## Mechanism Update Classes",
                "\n".join(f"- {item}" for item in mechanism_update_classes) or "- none",
                "",
                "## Weakness Signatures",
                _analysis_weakness_signatures_markdown(weakness_signatures),
                "",
                "## Policy Coverage",
                _analysis_policy_coverage_markdown(policy_coverage),
                "",
                "## Traceability",
                "\n".join(
                    f"- {trial.task_id}: detail/{_safe_campaign_id(trial.task_id)}.md"
                    for trial in trials
                ),
                "",
            ]
        )
    )

    analysis_summary = {
        "campaign_id": str(campaign_id),
        "summary_id": str(getattr(summary, "summary_id", "") or ""),
        "overall_score": float(getattr(summary, "overall_score", 0.0) or 0.0),
        "trial_count": len(trials),
        "passed_count": sum(
            1
            for trial in trials
            if getattr(getattr(trial, "status", None), "value", "") == "passed"
        ),
        "failed_count": sum(
            1
            for trial in trials
            if getattr(getattr(trial, "status", None), "value", "") != "passed"
        ),
        "infrastructure_failure_count": infra_failure_count,
        "terminal_environment_signal_count": terminal_environment_signal_count,
        "failure_buckets": failure_buckets,
        "candidate_update_classes": candidate_update_classes,
        "mechanism_update_entries": mechanism_update_entries,
        "mechanism_update_classes": mechanism_update_classes,
        "weakness_signatures": weakness_signatures,
        "policy_coverage": policy_coverage,
        "trajectory_evidence": trajectory_evidence,
        "detail_paths": detail_paths,
        "overview_path": str(overview_path),
        "recorded_at": datetime.now().isoformat(),
    }
    summary_path.write_text(json.dumps(analysis_summary, indent=2))

    return {
        "summary_id": str(getattr(summary, "summary_id", "") or ""),
        "overview_path": str(overview_path),
        "summary_path": str(summary_path),
        "detail_dir": str(detail_dir),
        "candidate_update_classes": candidate_update_classes,
        "mechanism_update_entries": mechanism_update_entries,
        "mechanism_update_classes": mechanism_update_classes,
        "recorded_at": datetime.now().isoformat(),
    }


def _analysis_failure_buckets(trials: list[Any]) -> list[dict[str, Any]]:
    from hl.attribution import FailureAttributor

    buckets: dict[str, dict[str, Any]] = {}
    for trial in trials:
        status = str(getattr(getattr(trial, "status", None), "value", "") or "")
        if status == "passed":
            continue
        try:
            attribution = FailureAttributor().analyze(trial)
        except Exception:
            attribution = SimpleNamespace(
                failure_category=status or "unknown",
                affected_components=[],
            )
        category = _analysis_effective_failure_category(
            trial,
            attribution_category=str(
                getattr(attribution, "failure_category", "") or status or "unknown"
            ),
        )
        infra_phase_components = _analysis_infrastructure_phase_components(trial)
        mechanism_components = set()
        replace_base_components = False
        if not infra_phase_components:
            mechanism_components, replace_base_components = (
                _analysis_failure_mechanism_components(
                    trial,
                    failure_category=category,
                )
            )
        infrastructure = _trial_is_infrastructure_failure(trial) or _analysis_category_is_infrastructure(
            category
        )
        bucket = buckets.setdefault(
            category,
            {
                "failure_category": category,
                "count": 0,
                "infrastructure": infrastructure,
                "task_ids": set(),
                "affected_components": set(),
                "timeout_phases": set(),
            },
        )
        bucket["count"] += 1
        bucket["infrastructure"] = bool(bucket.get("infrastructure")) or infrastructure
        bucket["task_ids"].add(str(getattr(trial, "task_id", "") or ""))
        category_infra_components = _analysis_infrastructure_category_components(category)
        if infra_phase_components:
            bucket["affected_components"].update(infra_phase_components)
        elif category_infra_components:
            bucket["affected_components"].update(category_infra_components)
        elif replace_base_components:
            bucket["affected_components"].update(mechanism_components)
        else:
            for component in getattr(attribution, "affected_components", []) or []:
                bucket["affected_components"].add(str(component))
            bucket["affected_components"].update(mechanism_components)
        metadata = getattr(trial, "metadata", {}) or {}
        timeout_phase = str(metadata.get("timeout_phase") or "")
        if timeout_phase:
            bucket["timeout_phases"].add(timeout_phase)
    result = []
    for bucket in buckets.values():
        result.append(
            {
                "failure_category": bucket["failure_category"],
                "count": int(bucket["count"]),
                "infrastructure": bool(bucket.get("infrastructure")),
                "task_ids": sorted(bucket["task_ids"]),
                "affected_components": sorted(bucket["affected_components"]),
                "timeout_phases": sorted(bucket["timeout_phases"]),
            }
        )
    result.sort(key=lambda item: (-int(item["count"]), item["failure_category"]))
    return result


def _analysis_failure_buckets_with_mechanisms(
    failure_buckets: list[dict[str, Any]],
    trials: list[Any],
) -> list[dict[str, Any]]:
    trial_by_task = {
        str(getattr(trial, "task_id", "") or ""): trial for trial in trials
    }
    enriched: list[dict[str, Any]] = []
    for bucket in failure_buckets:
        entry = dict(bucket)
        if bool(entry.get("infrastructure")):
            entry.pop("failure_mechanisms", None)
            entry.pop("failure_mechanism_count_stop_condition", None)
            enriched.append(entry)
            continue
        mechanisms: list[str] = []
        category = str(entry.get("failure_category") or "")
        for task_id in entry.get("task_ids") or []:
            trial = trial_by_task.get(str(task_id))
            if trial is None:
                continue
            for mechanism in _analysis_bucket_failure_mechanisms(
                trial,
                category,
            ):
                if mechanism not in mechanisms:
                    mechanisms.append(mechanism)
        if mechanisms:
            entry["failure_mechanisms"] = sorted(mechanisms)
            entry["failure_mechanism_count_stop_condition"] = False
        enriched.append(entry)
    return enriched


def _analysis_failure_buckets_with_evidence_mechanisms(
    failure_buckets: list[dict[str, Any]],
    trajectory_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for bucket in failure_buckets:
        entry = dict(bucket)
        if bool(entry.get("infrastructure")):
            entry.pop("failure_mechanisms", None)
            entry.pop("failure_mechanism_count_stop_condition", None)
            enriched.append(entry)
            continue
        if entry.get("failure_mechanisms"):
            enriched.append(entry)
            continue
        mechanisms: list[str] = []
        category_mechanism = dependency_loop_mechanism_for_failure_category(
            str(entry.get("failure_category") or "")
        )
        if category_mechanism:
            mechanisms.append(category_mechanism)
        for task_id in entry.get("task_ids") or []:
            evidence = trajectory_evidence.get(str(task_id))
            if not isinstance(evidence, dict):
                continue
            for mechanism in evidence.get("failure_mechanisms") or []:
                if not isinstance(mechanism, dict):
                    continue
                name = str(mechanism.get("name") or "").strip()
                if name and name not in mechanisms:
                    mechanisms.append(name)
        if mechanisms:
            entry["failure_mechanisms"] = sorted(mechanisms)
            entry["failure_mechanism_count_stop_condition"] = False
        enriched.append(entry)
    return enriched


def _analysis_bucket_failure_mechanisms(
    trial: Any,
    failure_category: str,
) -> list[str]:
    if _analysis_infrastructure_phase_category(trial) or _analysis_category_is_infrastructure(
        failure_category
    ):
        return []
    names: list[str] = []
    category_mechanism = dependency_loop_mechanism_for_failure_category(
        failure_category
    )
    if category_mechanism:
        names.append(category_mechanism)
    for mechanism in failure_mechanisms_for_trial(trial):
        if mechanism.name not in names:
            names.append(mechanism.name)
    return names


def _analysis_effective_failure_category(
    trial: Any,
    *,
    attribution_category: str,
) -> str:
    infra_phase_category = _analysis_infrastructure_phase_category(trial)
    if infra_phase_category:
        return infra_phase_category
    evidence = _analysis_trajectory_evidence(trial)
    has_terminal_environment_signal = _trial_has_terminal_environment_signal(trial)
    failure_mechanism_names = [
        str(item.get("name") or "")
        for item in evidence.get("failure_mechanisms") or []
        if isinstance(item, dict) and str(item.get("name") or "")
    ]
    if (
        has_terminal_environment_signal
        and _analysis_has_dependency_or_toolchain_evidence(evidence)
        and _analysis_terminal_environment_should_override_strong_mechanism(evidence)
    ):
        return "terminal_environment_unavailable_after_dependency_loop"
    mechanism_category = _analysis_strong_failure_mechanism_category(evidence)
    if mechanism_category is not None:
        return mechanism_category
    category_from_trial = dependency_loop_failure_category_for_trial(
        trial,
        failure_mechanism_names,
    )
    if category_from_trial:
        return category_from_trial
    if not has_terminal_environment_signal:
        if (
            _analysis_has_dependency_loop_without_deliverable_progress(evidence)
            and _analysis_dependency_loop_can_override_attribution(
                attribution_category
            )
        ):
            return "dependency_loop_without_deliverable_progress"
        return attribution_category
    if _analysis_has_dependency_or_toolchain_evidence(evidence):
        return "terminal_environment_unavailable_after_dependency_loop"
    return "terminal_environment_unavailable"


_ANALYSIS_INFRASTRUCTURE_PHASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "environment_start": (
        "environment_start_timeout",
        ("bench/harbor", "bench/network_environment"),
    ),
    "environment_build": (
        "environment_build_timeout",
        ("bench/harbor", "bench/network_environment"),
    ),
    "verifier_runtime_prepare": (
        "verifier_runtime_prepare_timeout",
        ("bench/harbor", "bench/network_environment"),
    ),
    "harbor_process": (
        "harbor_process_timeout",
        ("bench/harbor", "orchestration/run_campaign"),
    ),
    "harbor_cancelled": (
        "harbor_process_timeout",
        ("bench/harbor", "orchestration/run_campaign"),
    ),
}

_ANALYSIS_INFRASTRUCTURE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "terminal_environment_unavailable": ("bench/harbor", "bench/network_environment"),
    "terminal_environment_unavailable_after_dependency_loop": (
        "bench/harbor",
        "bench/network_environment",
    ),
}


def _analysis_category_is_infrastructure(category: str) -> bool:
    return str(category) in _ANALYSIS_INFRASTRUCTURE_CATEGORIES


def _analysis_infrastructure_category_components(category: str) -> set[str]:
    return set(_ANALYSIS_INFRASTRUCTURE_CATEGORIES.get(str(category), ()))


def _analysis_infrastructure_phase_category(trial: Any) -> str:
    metadata = getattr(trial, "metadata", {}) or {}
    timeout_phase = str(metadata.get("timeout_phase") or "")
    entry = _ANALYSIS_INFRASTRUCTURE_PHASES.get(timeout_phase)
    if entry is None:
        return ""
    return entry[0]


def _analysis_infrastructure_phase_components(trial: Any) -> set[str]:
    metadata = getattr(trial, "metadata", {}) or {}
    timeout_phase = str(metadata.get("timeout_phase") or "")
    entry = _ANALYSIS_INFRASTRUCTURE_PHASES.get(timeout_phase)
    if entry is None:
        return set()
    return set(entry[1])


def _analysis_terminal_environment_should_override_strong_mechanism(
    evidence: dict[str, Any],
) -> bool:
    mechanism_names = [
        str(mechanism.get("name") or "")
        for mechanism in evidence.get("failure_mechanisms") or []
        if isinstance(mechanism, dict)
    ]
    non_environment_mechanisms = [
        name
        for name in mechanism_names
        if name != "terminal_environment_unavailable_after_dependency_loop_mechanism"
    ]
    if set(non_environment_mechanisms) & PRIMARY_VERIFIER_CONTRACT_MECHANISM_NAMES:
        return False
    if len(set(non_environment_mechanisms)) > 1:
        return True
    dependency_evidence_count = len(
        evidence.get("dependency_and_toolchain_evidence") or []
    )
    blocked_guard_count = len(evidence.get("blocked_guards") or [])
    timed_out_count = len(evidence.get("timed_out_commands") or [])
    return dependency_evidence_count >= 4 or (
        blocked_guard_count >= 2 and timed_out_count >= 4
    )


def _analysis_dependency_loop_can_override_attribution(
    attribution_category: str,
) -> bool:
    return attribution_category not in _ANALYSIS_STRONG_ATTRIBUTION_CATEGORIES


def _analysis_has_dependency_loop_without_deliverable_progress(
    evidence: dict[str, Any],
) -> bool:
    if not _analysis_has_dependency_or_toolchain_evidence(evidence):
        return False
    return not bool(evidence.get("deliverable_progress") or [])


def _analysis_has_dependency_or_toolchain_evidence(evidence: dict[str, Any]) -> bool:
    policy_counts = evidence.get("policy_counts") or {}
    dependency_count = len(evidence.get("dependency_and_toolchain_evidence") or [])
    if dependency_count:
        return True
    return any(
        int(policy_counts.get(policy, 0) or 0) > 0
        for policy in _ANALYSIS_DEPENDENCY_TOOLCHAIN_POLICIES
    )


def _analysis_strong_failure_mechanism_category(
    evidence: dict[str, Any],
) -> str | None:
    mechanism_names = {
        str(mechanism.get("name") or "")
        for mechanism in evidence.get("failure_mechanisms") or []
        if isinstance(mechanism, dict)
    }
    if (
        "terminal_environment_unavailable_after_dependency_loop_mechanism"
        in mechanism_names
        and "cython_extension_optional_import_pivot_mechanism" in mechanism_names
    ):
        return "terminal_environment_unavailable_after_dependency_loop"
    for preferred_name in (
        "literal_output_file_content_contract",
        "tokenized_output_file_contract",
    ):
        if preferred_name in mechanism_names:
            return preferred_name
    for mechanism in evidence.get("failure_mechanisms") or []:
        if not isinstance(mechanism, dict):
            continue
        name = str(mechanism.get("name") or "")
        if name in PRIMARY_VERIFIER_CONTRACT_MECHANISM_NAMES:
            return name
    for mechanism in evidence.get("failure_mechanisms") or []:
        if not isinstance(mechanism, dict):
            continue
        name = str(mechanism.get("name") or "")
        if name in _ANALYSIS_STRONG_FAILURE_MECHANISM_CATEGORIES:
            return name
    return None


def _analysis_failure_mechanism_components(
    trial: Any,
    *,
    failure_category: str = "",
) -> tuple[set[str], bool]:
    components: set[str] = set()
    mechanisms = failure_mechanisms_for_trial(trial)
    mechanism_names = [mechanism.name for mechanism in mechanisms]
    category_mechanism = dependency_loop_mechanism_for_failure_category(
        failure_category
    )
    if category_mechanism and category_mechanism not in mechanism_names:
        mechanism_names.insert(0, category_mechanism)
    if (
        TERMINAL_ENVIRONMENT_UNAVAILABLE_AFTER_DEPENDENCY_LOOP_MECHANISM
        in mechanism_names
        and failure_category
        not in _ANALYSIS_INFRASTRUCTURE_CATEGORIES
    ):
        mechanism_names = [
            name
            for name in mechanism_names
            if name != TERMINAL_ENVIRONMENT_UNAVAILABLE_AFTER_DEPENDENCY_LOOP_MECHANISM
        ]
    replace_base_components = failure_mechanisms_replace_base_components(mechanism_names)
    mechanism_names_by_object = {mechanism.name for mechanism in mechanisms}
    for mechanism_name in mechanism_names:
        if mechanism_name in mechanism_names_by_object:
            continue
        components.update(affected_components_for_failure_mechanism(mechanism_name))
    for mechanism in mechanisms:
        if (
            mechanism.name
            == TERMINAL_ENVIRONMENT_UNAVAILABLE_AFTER_DEPENDENCY_LOOP_MECHANISM
            and failure_category
            not in _ANALYSIS_INFRASTRUCTURE_CATEGORIES
        ):
            continue
        if (
            replace_base_components
            and mechanism.name in DEPENDENCY_LOOP_BASE_REPLACEMENT_NEUTRAL_MECHANISM_NAMES
        ):
            continue
        components.update(affected_components_for_failure_mechanism(mechanism.name))
        components.update(_ANALYSIS_FAILURE_MECHANISM_COMPONENTS.get(mechanism.name, ()))
    return components, replace_base_components


def _candidate_update_classes(failure_buckets: list[dict[str, Any]]) -> list[str]:
    candidates = []
    for bucket in failure_buckets[:5]:
        components = ", ".join(bucket.get("affected_components") or ["unknown"])
        prefix = "infrastructure " if bucket.get("infrastructure") else ""
        candidates.append(
            f"{prefix}{bucket['failure_category']} -> {components} "
            f"({bucket['count']} trial(s))"
        )
    return candidates


def _analysis_mechanism_update_classes(
    *,
    failure_buckets: list[dict[str, Any]],
    trajectory_evidence: dict[str, Any],
) -> list[str]:
    return _format_analysis_mechanism_update_classes(
        _analysis_mechanism_update_entries(
            failure_buckets=failure_buckets,
            trajectory_evidence=trajectory_evidence,
        )
    )


def _analysis_mechanism_update_entries(
    *,
    failure_buckets: list[dict[str, Any]],
    trajectory_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for bucket in failure_buckets:
        if bool(bucket.get("infrastructure")):
            continue
        category = str(bucket.get("failure_category") or "").strip()
        if not category:
            continue
        for task_id in bucket.get("task_ids") or []:
            task_text = str(task_id).strip()
            if not task_text:
                continue
            evidence = trajectory_evidence.get(task_text)
            if not isinstance(evidence, dict):
                continue
            for mechanism in evidence.get("failure_mechanisms") or []:
                if not isinstance(mechanism, dict):
                    continue
                mechanism_name = str(mechanism.get("name") or "").strip()
                if not mechanism_name:
                    continue
                key = (category, mechanism_name)
                entry = grouped.setdefault(
                    key,
                    {
                        "failure_category": category,
                        "mechanism": mechanism_name,
                        "task_ids": set(),
                        "affected_components": set(),
                    },
                )
                entry["task_ids"].add(task_text)
                entry["affected_components"].update(
                    affected_components_for_failure_mechanism(mechanism_name)
                )
                entry["affected_components"].update(
                    _ANALYSIS_FAILURE_MECHANISM_COMPONENTS.get(mechanism_name, ())
                )

    entries: list[dict[str, Any]] = []
    for entry in grouped.values():
        components = sorted(entry["affected_components"])
        entries.append(
            {
                "failure_category": entry["failure_category"],
                "mechanism": entry["mechanism"],
                "count": len(entry["task_ids"]),
                "task_ids": sorted(entry["task_ids"]),
                "affected_components": components or ["unknown"],
            }
        )
    entries.sort(
        key=lambda item: (
            -int(item["count"]),
            str(item["failure_category"]),
            str(item["mechanism"]),
        )
    )
    return entries[:8]


def _format_analysis_mechanism_update_classes(
    entries: list[dict[str, Any]],
) -> list[str]:
    return [
        f"{entry['failure_category']} / {entry['mechanism']} -> "
        f"{', '.join(entry.get('affected_components') or ['unknown'])} "
        f"({entry['count']} trial(s))"
        for entry in entries[:8]
    ]


def _analysis_weakness_signatures(
    trials: list[Any],
    *,
    failure_buckets: list[dict[str, Any]],
    trajectory_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build deterministic Self-Harness weakness-mining signatures."""

    trial_by_task = {str(getattr(trial, "task_id", "") or ""): trial for trial in trials}
    grouped: dict[str, dict[str, Any]] = {}
    for bucket in failure_buckets:
        category = str(bucket.get("failure_category") or "").strip()
        if not category:
            continue
        task_ids = [str(task_id) for task_id in bucket.get("task_ids") or []]
        count_increment = int(bucket.get("count") or 0) if len(task_ids) <= 1 else 1
        for task_id in task_ids or [""]:
            evidence = trajectory_evidence.get(task_id)
            evidence_items = [evidence] if isinstance(evidence, dict) else []
            mechanisms = _analysis_weakness_mechanisms(evidence_items)
            trial = trial_by_task.get(task_id)
            verifier_failure = _analysis_weakness_verifier_failure(
                category,
                bucket=bucket,
                trials=[trial] if trial is not None else [],
            )
            agent_contribution = _analysis_weakness_agent_contribution(
                evidence_items,
                bucket=bucket,
            )
            reusable_mechanism = _analysis_weakness_reusable_mechanism(
                category,
                mechanisms=mechanisms,
                bucket=bucket,
            )
            signature = _analysis_weakness_signature_label(
                verifier_failure=verifier_failure,
                agent_contribution=agent_contribution,
                reusable_mechanism=reusable_mechanism,
            )
            entry = grouped.setdefault(
                signature,
                {
                    "signature": signature,
                    "verifier_failure": verifier_failure,
                    "agent_contribution": agent_contribution,
                    "reusable_mechanism": reusable_mechanism,
                    "failure_category": category,
                    "count": 0,
                    "task_ids": set(),
                    "affected_components": set(),
                    "timeout_phases": set(),
                    "failure_mechanisms": set(),
                    "evidence_sources": set(),
                    "purpose": (
                        "Self-Harness weakness mining: cluster only failures "
                        "that share the same verifier failure, agent behavior "
                        "contribution, and reusable mechanism before proposing "
                        "a bounded harness update."
                    ),
                    "loop_stop_condition": False,
                    "time_round_token_limit_driven": False,
                },
            )
            entry["count"] = int(entry.get("count") or 0) + max(count_increment, 1)
            if task_id:
                entry["task_ids"].add(task_id)
            for component in bucket.get("affected_components") or []:
                component_text = str(component)
                if component_text:
                    entry["affected_components"].add(component_text)
            for phase in bucket.get("timeout_phases") or []:
                phase_text = str(phase)
                if phase_text:
                    entry["timeout_phases"].add(phase_text)
            for mechanism in mechanisms:
                entry["failure_mechanisms"].add(mechanism)
            for source in _analysis_weakness_evidence_sources(evidence_items):
                entry["evidence_sources"].add(source)
            for source in _analysis_weakness_infrastructure_evidence_sources(bucket):
                entry["evidence_sources"].add(source)

    signatures: list[dict[str, Any]] = []
    for entry in grouped.values():
        signatures.append(
            {
                "signature": entry["signature"],
                "verifier_failure": entry["verifier_failure"],
                "agent_contribution": entry["agent_contribution"],
                "reusable_mechanism": entry["reusable_mechanism"],
                "failure_category": entry["failure_category"],
                "count": int(entry.get("count") or 0),
                "task_ids": sorted(entry["task_ids"])[:12],
                "affected_components": sorted(entry["affected_components"])[:12],
                "timeout_phases": sorted(entry["timeout_phases"])[:8],
                "failure_mechanisms": sorted(entry["failure_mechanisms"])[:12],
                "evidence_sources": sorted(entry["evidence_sources"]),
                "purpose": entry["purpose"],
                "loop_stop_condition": False,
                "time_round_token_limit_driven": False,
            }
        )
    signatures.sort(
        key=lambda item: (
            -int(item.get("count") or 0),
            str(item.get("signature") or ""),
        )
    )
    return signatures


def _analysis_weakness_mechanisms(evidence_items: list[Any]) -> list[str]:
    names: list[str] = []
    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            continue
        for item in evidence.get("failure_mechanisms") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
    return sorted(dict.fromkeys(names))


def _analysis_failure_mechanisms_for_trial(trial: Any) -> list[dict[str, str]]:
    """Return mechanisms that should guide analysis for this trial.

    Verifier/runtime infrastructure phase failures can contain stale Worker
    todo, expected-artifact, or hidden-verifier text that looks like a semantic
    deliverable contract. Do not let those verifier-contract mechanisms become
    Self-Harness update directions for infrastructure buckets; the infra phase
    and dependency/toolchain evidence remain available separately.
    """

    if _analysis_infrastructure_phase_category(trial):
        return []
    return [mechanism.as_dict() for mechanism in failure_mechanisms_for_trial(trial)]


def _analysis_weakness_verifier_failure(
    category: str,
    *,
    bucket: dict[str, Any],
    trials: list[Any],
) -> str:
    if bucket.get("infrastructure"):
        phases = [str(phase) for phase in bucket.get("timeout_phases") or [] if phase]
        if phases:
            return "infra_timeout_phase:" + "+".join(sorted(dict.fromkeys(phases)))
        return "infrastructure_failure:" + category
    if any(bool(getattr(trial, "verified", False)) for trial in trials):
        return "verifier_assertion:" + category
    status_labels = sorted(
        dict.fromkeys(
            str(getattr(getattr(trial, "status", None), "value", "") or "unknown")
            for trial in trials
        )
    )
    if status_labels:
        return "unverified_status:" + "+".join(status_labels)
    return "failure_category:" + category


def _analysis_weakness_agent_contribution(
    evidence_items: list[Any],
    *,
    bucket: dict[str, Any] | None = None,
) -> str:
    policy_counts: dict[str, int] = {}
    source_counts = {
        "dependency_loop": 0,
        "blocked_guard": 0,
        "timed_out_command": 0,
        "deliverable_progress": 0,
        "terminal_environment": 0,
        "semantic_mechanism": 0,
    }
    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            continue
        for name, count in (evidence.get("policy_counts") or {}).items():
            try:
                numeric = int(count or 0)
            except (TypeError, ValueError):
                numeric = 0
            if numeric > 0 and str(name) not in {
                "artifact_check_deliverable_progress",
            }:
                policy_counts[str(name)] = policy_counts.get(str(name), 0) + numeric
        source_counts["dependency_loop"] += len(
            evidence.get("dependency_and_toolchain_evidence") or []
        )
        source_counts["blocked_guard"] += len(evidence.get("blocked_guards") or [])
        source_counts["timed_out_command"] += len(
            evidence.get("timed_out_commands") or []
        )
        source_counts["deliverable_progress"] += len(
            evidence.get("deliverable_progress") or []
        )
        source_counts["terminal_environment"] += len(
            evidence.get("terminal_environment_markers") or []
        )
        source_counts["semantic_mechanism"] += len(
            evidence.get("failure_mechanisms") or []
        )
    if policy_counts:
        dominant_policy, dominant_count = sorted(
            policy_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        return f"policy:{dominant_policy}:{dominant_count}"
    for label in (
        "dependency_loop",
        "blocked_guard",
        "timed_out_command",
        "semantic_mechanism",
        "terminal_environment",
        "deliverable_progress",
    ):
        if source_counts[label] > 0:
            return f"evidence:{label}:{source_counts[label]}"
    if bucket and bool(bucket.get("infrastructure")):
        phases = [
            str(phase)
            for phase in bucket.get("timeout_phases") or []
            if str(phase).strip()
        ]
        count = int(bucket.get("count") or 0)
        if phases:
            return (
                "infrastructure:timeout_phase:"
                + "+".join(sorted(dict.fromkeys(phases)))
                + f":{count}"
            )
        category = str(bucket.get("failure_category") or "infrastructure_failure")
        return f"infrastructure:{category}:{count}"
    return "agent_behavior:unclassified"


def _analysis_weakness_reusable_mechanism(
    category: str,
    *,
    mechanisms: list[str],
    bucket: dict[str, Any],
) -> str:
    if mechanisms:
        return "mechanism:" + "+".join(mechanisms[:4])
    components = [
        str(component) for component in bucket.get("affected_components") or [] if component
    ]
    if components:
        return "components:" + "+".join(components[:4])
    return "category:" + category


def _analysis_weakness_signature_label(
    *,
    verifier_failure: str,
    agent_contribution: str,
    reusable_mechanism: str,
) -> str:
    return (
        f"verifier={verifier_failure}|agent={agent_contribution}|"
        f"mechanism={reusable_mechanism}"
    )


def _analysis_weakness_evidence_sources(evidence_items: list[Any]) -> list[str]:
    sources: set[str] = set()
    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            continue
        if evidence.get("failure_mechanisms"):
            sources.add("failure_mechanisms")
        if evidence.get("policy_counts"):
            sources.add("policy_counts")
        if evidence.get("timed_out_commands"):
            sources.add("timed_out_commands")
        if evidence.get("blocked_guards"):
            sources.add("blocked_guards")
        if evidence.get("dependency_and_toolchain_evidence"):
            sources.add("dependency_and_toolchain_evidence")
        if evidence.get("deliverable_progress"):
            sources.add("deliverable_progress")
        if evidence.get("terminal_environment_markers"):
            sources.add("terminal_environment_markers")
    return sorted(sources)


def _analysis_weakness_infrastructure_evidence_sources(
    bucket: dict[str, Any],
) -> list[str]:
    if not bool(bucket.get("infrastructure")):
        return []
    phases = [
        str(phase)
        for phase in bucket.get("timeout_phases") or []
        if str(phase).strip()
    ]
    if phases:
        return [
            "infrastructure_timeout_phase:" + phase
            for phase in sorted(dict.fromkeys(phases))
        ]
    category = str(bucket.get("failure_category") or "infrastructure_failure")
    return ["infrastructure_failure:" + category]


def _analysis_weakness_signatures_markdown(signatures: list[dict[str, Any]]) -> str:
    if not signatures:
        return "- none"
    lines: list[str] = []
    for item in signatures[:8]:
        lines.append(
            f"- {item.get('failure_category', '')}: {item.get('signature', '')} "
            f"({int(item.get('count') or 0)} trial(s))"
        )
    return "\n".join(lines)


_ANALYSIS_DEPENDENCY_TOOLCHAIN_POLICIES = {
    "background_package_command_guard",
    "package_manager_timeout_cap",
    "package_manager_semantic_failure",
    "manual_dependency_download_guard",
    "manual_deb_dependency_chase_guard",
    "large_toolchain_install_guard",
    "large_toolchain_install_plan",
    "staged_dependency_script_guard",
    "manual_dependency_download_timeout_phase",
    "package_cache_search_timeout_phase",
    "repeated_dependency_timeout_path_guard",
    "repeated_dependency_failure_path_guard",
    "cross_arch_build_timeout_phase",
    "repeated_cross_arch_timeout_path_guard",
    "build_compile_timeout_phase",
    "tool_binary_download_timeout_phase",
}


_ANALYSIS_STRONG_ATTRIBUTION_CATEGORIES = {
    "agent_timeout_with_verifier_mismatch",
    "harness_bug",
    "missing_artifact",
    "post_completion_agent_exception",
    "verifier_environment_error",
    "verifier_mismatch",
    "verifier_runtime_prepare_timeout",
    "verifier_timeout",
}


_ANALYSIS_STRONG_FAILURE_MECHANISM_CATEGORIES = {
    "missing_output_artifact_contract",
} | set(PRIMARY_VERIFIER_CONTRACT_MECHANISM_NAMES) | (
    set(DEPENDENCY_PIVOT_MECHANISM_NAMES) - {"ml_cv_heavy_import_pivot_mechanism"}
)


_ANALYSIS_FAILURE_MECHANISM_COMPONENTS = {
    "dependency_loop_without_deliverable_progress_mechanism": set(
        affected_components_for_failure_mechanism(
            "dependency_loop_without_deliverable_progress_mechanism"
        )
    ),
    "terminal_environment_unavailable_after_dependency_loop_mechanism": set(
        affected_components_for_failure_mechanism(
            "terminal_environment_unavailable_after_dependency_loop_mechanism"
        )
    ),
    "adaptive_rejection_sampler_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "arithmetic_reference_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "async_cancellation_cleanup_contract": {
        "bench/agent",
        "crates/hl-worker-core",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "caffe_cifar10_artifact_contract": {
        "bench/agent",
        "crates/hl-worker-core",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "corewar_warrior_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "dataset_shard_generalization_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "deliverable_size_cap_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "dna_assembly_primer_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "dna_insert_primer_pair_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "generated_script_structure_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "git_sanitization_scope_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "gpt2_codegolf_text_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "html_filter_alert_bypass_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "html_filter_blocks_xss_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "image_similarity_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "literal_output_file_content_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "missing_output_artifact_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "model_extraction_matrix_contract": {
        "bench/agent",
        "crates/hl-worker-core",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "pytorch_distributed_parallelism_contract": {
        "bench/agent",
        "crates/hl-worker-core",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "tokenized_output_file_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "native_crash_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "numeric_interval_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "regex_replacement_backreference_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "single_file_deliverable_directory_contract": {
        "bench/agent",
        "crates/hl-worker-core",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "sparql_result_set_aggregation_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "spectral_peak_fit_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "state_transition_set_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "stan_dependency_stack_pivot_mechanism": set(
        affected_components_for_failure_mechanism("stan_dependency_stack_pivot_mechanism")
    ),
    "structured_csv_table_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "structured_output_schema_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "text_output_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "token_substitution_contract": {
        "bench/agent",
        "harness/tools/verify",
        "verification/checks",
    },
    "vm_service_readiness_contract": {
        "bench/agent",
        "harness/tools/verify",
        "recovery/patterns",
        "verification/checks",
    },
    "fasttext_artifact_pivot_mechanism": set(
        affected_components_for_failure_mechanism("fasttext_artifact_pivot_mechanism")
    ),
    "cross_arch_toolchain_pivot_mechanism": set(
        affected_components_for_failure_mechanism("cross_arch_toolchain_pivot_mechanism")
    ),
    "ml_cv_heavy_import_pivot_mechanism": set(
        affected_components_for_failure_mechanism("ml_cv_heavy_import_pivot_mechanism")
    ),
    "numpy_eigensolver_dependency_pivot_mechanism": set(
        affected_components_for_failure_mechanism(
            "numpy_eigensolver_dependency_pivot_mechanism"
        )
    ),
}


def _analysis_policy_coverage(trials: list[Any]) -> dict[str, Any]:
    policies = {
        "background_package_command_guard": {
            "description": "blocks package-manager commands launched in background or detached shells",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "broad_root_find_guard": {
            "description": "blocks unbounded recursive find / probes at the single-operation evidence window level, not as a loop stop condition",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "broad_proc_scan_guard": {
            "description": "blocks broad /proc process glob and PID-loop scans",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "verifier_artifact_search_guard": {
            "description": "blocks hidden verifier log and verifier dependency-cache probes",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "package_manager_timeout_cap": {
            "description": "caps foreground package-manager commands so one install path cannot consume a turn",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "package_manager_semantic_failure": {
            "description": "detects package-manager failures hidden behind successful shell pipelines",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "network_probe_tool_missing": {
            "description": "detects missing ping/curl/wget/nc reachability probes hidden behind successful shell pipelines",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "manual_dependency_download_guard": {
            "description": "blocks hand-written PyPI/CRAN/Debian/Conda/GitHub package archive downloads before execution",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "manual_deb_dependency_chase_guard": {
            "description": "blocks local .deb compiler/R/Stan dependency-chasing before dpkg state is contaminated",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "large_toolchain_install_guard": {
            "description": "blocks explicit large compiler and cross-toolchain package installs before they destabilize task containers",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "large_toolchain_install_plan": {
            "description": "detects apt output plans that expand compiler/toolchain installs into hundreds of MB",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "staged_dependency_script_guard": {
            "description": "blocks scripts that hide package download/install loops behind write/edit tools",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "manual_dependency_download_timeout_phase": {
            "description": "classifies hand-written PyPI/Debian/Conda/GitHub package download timeouts",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "build_compile_timeout_phase": {
            "description": "classifies full build/compile timeouts for lower-parallelism recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "environment_inventory_timeout_phase": {
            "description": "classifies broad tool/path inventory timeouts for targeted environment probes",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "directory_listing_timeout_phase": {
            "description": "classifies broad directory listing timeouts for path-targeted inspection recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "log_file_read_timeout_phase": {
            "description": "classifies large log-file read timeouts for sliced log inspection recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "local_validation_timeout_phase": {
            "description": "classifies generated local validation script timeouts for smaller-fixture recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "package_cache_search_timeout_phase": {
            "description": "classifies broad /tmp package-cache artifact searches after dependency timeouts",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_dependency_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated dependency setup/download timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_dependency_failure_path_guard": {
            "description": "counts Worker-side blocks after repeated dependency setup/download failures hidden behind non-timeout shell output",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "scratch_artifact_search_timeout_phase": {
            "description": "classifies broad /tmp scratch artifact searches for targeted workspace recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "git_history_search_timeout_phase": {
            "description": "classifies full-history git secret search or filter-branch rewrites for targeted history repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_git_history_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated full-history git secret/history search or rewrite timeouts",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "generated_solver_search_timeout_phase": {
            "description": "classifies generated rule/solver search timeouts for incremental fixture-level validation",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "generated_exploration_script_timeout_phase": {
            "description": "classifies Worker-generated open-ended exploration script timeouts for sample-first or direct-deliverable recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_generated_exploration_script_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated open-ended generated exploration script timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_generated_solver_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated generated rule/solver search timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "password_cracking_search_timeout_phase": {
            "description": "classifies full John/hashcat wordlist or incremental cracking timeouts for hash-format and bounded-candidate recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_password_cracking_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated full John/hashcat cracking timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "model_extraction_probe_timeout_phase": {
            "description": "classifies repeated model-extraction scripts and dense forward probes for batched-query checkpoint recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_model_extraction_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated model-extraction script or dense forward-sweep timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "database_query_validation_timeout_phase": {
            "description": "classifies full SQL query/output validation timeouts for plan-first and sliced-query recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_database_query_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated full SQL query/output validation or database recovery timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "service_inventory_probe_timeout_phase": {
            "description": "classifies broad service/socket/config probes for endpoint-focused service recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_service_inventory_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated broad service/socket/config inventory timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "background_process_wait_timeout_phase": {
            "description": "classifies sleep-loop, ps/pgrep, wait, and sleep-then-log polling timeouts around background jobs",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_background_process_wait_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated background process wait or log polling timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "cross_arch_build_timeout_phase": {
            "description": "classifies cross-architecture build script/toolchain timeouts for targeted ABI and object-level recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "emulator_validation_timeout_phase": {
            "description": "classifies emulator/ELF/VM validation timeouts for loader, short-cycle, and progress-logged recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_cross_arch_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated cross-architecture build/emulator timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "image_render_validation_timeout_phase": {
            "description": "classifies image render and full-pixel/SSIM/PPM validation timeouts for header, tiny-render, and sampled-pixel recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_image_render_validation_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated full render or full-pixel/SSIM/cosine image validation timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "tool_binary_download_timeout_phase": {
            "description": "classifies external tool binary/archive download timeouts for installed-tool or cached-artifact recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "external_media_metadata_timeout_phase": {
            "description": "classifies remote video/page metadata probe timeouts for local-media recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "remote_dataset_fetch_timeout_phase": {
            "description": "classifies remote dataset API/cache-discovery timeouts for local shard or sampled streaming recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_remote_dataset_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated remote dataset API/cache-discovery timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "media_batch_processing_timeout_phase": {
            "description": "classifies full media/frame/OCR batch-processing timeouts for sampled incremental recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_media_batch_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated media/frame/OCR batch-processing timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "vm_service_readiness_timeout_phase": {
            "description": "classifies QEMU/Alpine SSH or telnet readiness timeouts separately from numeric simulation validation",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "simulation_validation_timeout_phase": {
            "description": "classifies simulator and physics-evaluation timeouts for smaller-horizon recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_simulation_validation_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated full simulation, VM, boot-wait, or physics validation timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "statistical_eval_validation_timeout_phase": {
            "description": "classifies statistical sampling/eval script timeouts for deterministic sliced validation",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_statistical_eval_validation_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated full statistical sampling/eval validation timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "network_probe_timeout_phase": {
            "description": "classifies generic internet reachability probe timeouts for cached/local-artifact recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "long_compute_timeout_phase": {
            "description": "classifies full training, sampling, and model compilation timeouts for escalation",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "benchmark_validation_timeout_phase": {
            "description": "classifies full benchmark/performance validation timeouts for smaller-probe recovery",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "repeated_compute_timeout_path_guard": {
            "description": "counts Worker-side blocks after repeated long-compute or benchmark timeout paths",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "artifact_check_deliverable_progress": {
            "description": "counts successful expected-artifact inspection as deliverable progress",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "regex_replacement_backreference_contract": {
            "description": "detects verifier-grounded Python re.sub replacement backreference failures for group-count contract repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "git_sanitization_scope_contract": {
            "description": "detects verifier-grounded sanitize-git-repo diff-scope failures where only CONTAMINATED_PATHS may change against the baseline commit",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "dna_insert_primer_pair_contract": {
            "description": "detects verifier-grounded DNA insert primer-pair failures for primers.fasta overlap and Tm contract repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "dna_assembly_primer_contract": {
            "description": "detects verifier-grounded Golden Gate/BsaI assembly primer failures for primers.fasta clamp, overhang, binding, and fragment assembly repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "gpt2_codegolf_text_contract": {
            "description": "detects verifier-grounded GPT2 codegolf prompt/output failures for fixed compile, size, and continuation-text contract repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "html_filter_alert_bypass_contract": {
            "description": "detects verifier-grounded /app/out.html bypass failures where post-filter headless browser alert must still fire",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "html_filter_blocks_xss_contract": {
            "description": "detects verifier-grounded /app/filter.py sanitizer failures where filtered attack-vector batches must not trigger browser alerts",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "adaptive_rejection_sampler_contract": {
            "description": "detects verifier-grounded /app/ars.R API/bounds/sample-statistics failures for adaptive rejection sampler repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "deliverable_size_cap_contract": {
            "description": "detects verifier-grounded deliverable file size-cap failures for behavior-preserving shrinking and exact size-check repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "structured_csv_table_contract": {
            "description": "detects verifier-grounded CSV table schema/content failures for keyed rows, required columns, totals, blanks, and numeric-cell repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "structured_output_schema_contract": {
            "description": "detects verifier-grounded structured output schema failures for parseable TOML/structured files and exact required numeric field names",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "numeric_interval_contract": {
            "description": "detects verifier-grounded numeric interval failures for inclusive frame/range outputs and min/max tuple semantics",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "native_crash_contract": {
            "description": "detects verifier-grounded native binary SIGSEGV/core-dump failures for exact command reproduction and bounds/EOF/allocation repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "state_transition_set_contract": {
            "description": "detects verifier-grounded legal state-transition membership failures such as Python-chess move/state mismatches",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "stan_dependency_stack_pivot_mechanism": {
            "description": "detects repeated RStan/PyStan/httpstan/compiler dependency-stack loops that should pivot to visible Stan/R/PyStan deliverables and static checks",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "text_output_contract": {
            "description": "detects verifier-grounded stdout UTF-8 decoding failures where deliverables emit binary bytes instead of valid text",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "image_similarity_contract": {
            "description": "detects verifier-grounded image/render similarity failures for preserving dimensions, camera, geometry, lighting, sampling, color scale, and output path",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "token_substitution_contract": {
            "description": "detects verifier-grounded synonym/token substitution failures where edits must stay within synonyms.txt families and preserve punctuation/token count",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "literal_output_file_content_contract": {
            "description": "detects verifier-grounded literal output-file content failures where Path(...).read_text() must equal visible expected_output",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "missing_output_artifact_contract": {
            "description": "detects verifier-grounded missing output artifact failures where exact /app, /tmp, or /jail deliverable paths must be created before broad rewrites",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "caffe_cifar10_artifact_contract": {
            "description": "detects verifier-grounded Caffe/CIFAR-10 CPU-only build and quick-model artifact failures where caffe.bin, solver config, and cifar10_quick_iter_500.caffemodel must be checked before broad dependency expansion",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "model_extraction_matrix_contract": {
            "description": "detects verifier-grounded ReLU/logits model-extraction failures where stolen_A1.npy exists but rows do not match the expected matrix up to scale",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "tokenized_output_file_contract": {
            "description": "detects verifier-grounded tokenized output-file failures where Path(...).read_text().strip().split() must contain visible expected whitespace-separated tokens",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "spectral_peak_fit_contract": {
            "description": "detects verifier-grounded spectral/Raman peak fitting mismatches for x0, gamma, amplitude, and offset contract repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "sparql_result_set_aggregation_contract": {
            "description": "detects verifier-grounded SPARQL/RDFLib result-set mismatches where multi-value fields such as countries must be aggregated",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "dataset_shard_generalization_contract": {
            "description": "detects verifier-grounded C4/HuggingFace dataset shard failures where solutions must generalize beyond the visible shard",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "generated_script_structure_contract": {
            "description": "detects verifier-grounded generated script structure failures for required commands, macro/register definitions, executions, and save/exit forms",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "arithmetic_reference_contract": {
            "description": "detects verifier-grounded arithmetic simulator mismatches for integer isqrt/floor semantics, modulo 2^32 wrapping, stdout formatting, and boundary cases",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "vm_service_readiness_contract": {
            "description": "detects verifier-grounded QEMU VM service readiness failures for host port forwarding, guest service startup, credentials, and exact verifier command repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "corewar_warrior_contract": {
            "description": "detects verifier-grounded Core War Redcode warrior failures for /app/my_warrior.red validity, pmars opponent thresholds, and iterative strategy repair",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "fasttext_artifact_pivot_mechanism": {
            "description": "detects repeated fastText/C++/package dependency loops that should pivot to local data conversion and explicit model-artifact evidence",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "cross_arch_toolchain_pivot_mechanism": {
            "description": "detects repeated MIPS/cross-architecture toolchain loops that should pivot to installed binutils, static ELF checks, and the target binary",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
        "numpy_eigensolver_dependency_pivot_mechanism": {
            "description": "detects repeated SciPy/compiler/package dependency loops in NumPy eigensolver tasks that should pivot to local NumPy complex-dtype and residual checks",
            "count": 0,
            "tasks": set(),
            "examples": [],
        },
    }
    uncovered_timeout_examples: list[dict[str, str]] = []
    for trial in trials:
        for mechanism in _analysis_failure_mechanisms_for_trial(trial):
            name = mechanism.get("name", "")
            if name in policies:
                _record_policy_example(
                    policies[name],
                    {
                        "task_id": mechanism.get("task_id", ""),
                        "command": mechanism.get("evidence", ""),
                    },
                )
        for event in _trial_policy_events(trial):
            matches = _policy_matches_for_event(event)
            if matches:
                for policy in matches:
                    _record_policy_example(policies[policy], event)
            elif event["timed_out"] and len(uncovered_timeout_examples) < 8:
                uncovered_timeout_examples.append(_policy_example(event))
    return {
        "policies": {
            name: {
                "description": data["description"],
                "count": data["count"],
                "tasks": sorted(data["tasks"]),
                "examples": data["examples"],
            }
            for name, data in policies.items()
        },
        "uncovered_timeout_examples": uncovered_timeout_examples,
    }


def _trial_policy_events(trial: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    expected_artifacts = _trial_expected_artifacts(trial)
    raw_events = [
        *(getattr(trial, "trajectory", []) or []),
        *(getattr(trial, "tool_calls", []) or []),
    ]
    for raw in raw_events:
        command = _event_command(raw)
        content = _event_policy_content(raw)
        file_path = _event_file_path(raw)
        tool = str(raw.get("tool") or raw.get("function") or raw.get("name") or "")
        if not command and content:
            target = f" {file_path}" if file_path else ""
            command = f"{tool or 'file'}{target}"
        if not command:
            continue
        output = "\n".join(
            str(raw.get(key) or "") for key in ("output", "stdout", "stderr", "error")
        )
        events.append(
            {
                "task_id": str(getattr(trial, "task_id", "") or ""),
                "tool": tool,
                "command": command,
                "file_path": file_path,
                "content": content,
                "output": output,
                "success": raw.get("success"),
                "timed_out": _event_timed_out(raw, output),
                "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
                "artifacts": [str(item) for item in (getattr(trial, "artifacts", []) or [])],
                "expected_artifacts": expected_artifacts,
            }
        )
    return events


def _event_arguments(event: dict[str, Any]) -> dict[str, Any]:
    arguments = event.get("arguments") or event.get("args")
    return arguments if isinstance(arguments, dict) else {}


def _event_command(event: dict[str, Any]) -> str:
    candidates = [
        event.get("command"),
        event.get("cmd"),
        event.get("input"),
    ]
    arguments = _event_arguments(event)
    candidates.extend([arguments.get("command"), arguments.get("cmd")])
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _event_file_path(event: dict[str, Any]) -> str:
    arguments = _event_arguments(event)
    for candidate in [event.get("file_path"), arguments.get("file_path"), arguments.get("path")]:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _event_policy_content(event: dict[str, Any]) -> str:
    arguments = _event_arguments(event)
    for candidate in [
        event.get("content"),
        event.get("new_string"),
        arguments.get("content"),
        arguments.get("new_string"),
    ]:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return ""


def _event_timed_out(event: dict[str, Any], output: str) -> bool:
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    if bool(event.get("timeout") or event.get("timed_out") or metadata.get("timed_out")):
        return True
    lowered = " ".join(
        str(value or "").lower()
        for value in [
            event.get("status"),
            event.get("error"),
            output,
            metadata.get("error"),
        ]
    )
    return "timeout" in lowered or "timed out" in lowered


def _trial_expected_artifacts(trial: Any) -> list[str]:
    metadata = getattr(trial, "metadata", {}) or {}
    targets: list[str] = []
    _extend_artifact_targets(
        targets,
        metadata.get("expected_artifacts") or metadata.get("expected_artifact") or [],
    )
    for raw in [
        *(getattr(trial, "trajectory", []) or []),
        *(getattr(trial, "tool_calls", []) or []),
    ]:
        if not isinstance(raw, dict):
            continue
        for key in (
            "expected_artifacts",
            "expected_artifact",
            "touched_deliverable_paths",
            "untouched_deliverable_paths",
            "deliverable_paths",
            "deliverable_path",
        ):
            _extend_artifact_targets(targets, raw.get(key))
        raw_metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        for key in (
            "expected_artifacts",
            "expected_artifact",
            "touched_deliverable_paths",
            "untouched_deliverable_paths",
            "deliverable_paths",
            "deliverable_path",
        ):
            _extend_artifact_targets(targets, raw_metadata.get(key))
    deduped: list[str] = []
    seen: set[str] = set()
    for target in targets:
        normalized = str(target).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _extend_artifact_targets(targets: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            targets.append(value.strip())
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _extend_artifact_targets(targets, item)


@dataclass(frozen=True)
class _AnalysisTimeoutPhasePolicy:
    name: str
    matcher: str
    fallback_only: bool = False


def _analysis_timeout_phase_policy_matches(
    command: str,
    existing_matches: list[str],
    policies: tuple[_AnalysisTimeoutPhasePolicy, ...] | None = None,
) -> list[str]:
    matches: list[str] = []
    for policy in policies or _ANALYSIS_TIMEOUT_PHASE_POLICIES:
        if policy.fallback_only and (existing_matches or matches):
            continue
        matcher = globals().get(policy.matcher)
        if callable(matcher) and matcher(command):
            matches.append(policy.name)
    return matches


_ANALYSIS_TIMEOUT_PHASE_POLICIES: tuple[_AnalysisTimeoutPhasePolicy, ...] = (
    _AnalysisTimeoutPhasePolicy(
        "manual_dependency_download_timeout_phase",
        "_analysis_command_matches_manual_dependency_download",
    ),
    _AnalysisTimeoutPhasePolicy(
        "build_compile_timeout_phase",
        "_analysis_command_matches_build_compile",
    ),
    _AnalysisTimeoutPhasePolicy(
        "environment_inventory_timeout_phase",
        "_analysis_command_matches_environment_inventory",
    ),
    _AnalysisTimeoutPhasePolicy(
        "local_validation_timeout_phase",
        "_analysis_command_matches_local_validation",
    ),
    _AnalysisTimeoutPhasePolicy(
        "package_cache_search_timeout_phase",
        "_analysis_command_matches_package_cache_search",
    ),
    _AnalysisTimeoutPhasePolicy(
        "scratch_artifact_search_timeout_phase",
        "_analysis_command_matches_scratch_artifact_search",
    ),
    _AnalysisTimeoutPhasePolicy(
        "git_history_search_timeout_phase",
        "_analysis_command_matches_git_history_search",
    ),
    _AnalysisTimeoutPhasePolicy(
        "generated_solver_search_timeout_phase",
        "_analysis_command_matches_generated_solver_search",
    ),
    _AnalysisTimeoutPhasePolicy(
        "generated_exploration_script_timeout_phase",
        "_analysis_command_matches_generated_exploration_script",
    ),
    _AnalysisTimeoutPhasePolicy(
        "password_cracking_search_timeout_phase",
        "_analysis_command_matches_password_cracking_search",
    ),
    _AnalysisTimeoutPhasePolicy(
        "model_extraction_probe_timeout_phase",
        "_analysis_command_matches_model_extraction_probe",
    ),
    _AnalysisTimeoutPhasePolicy(
        "database_query_validation_timeout_phase",
        "_analysis_command_matches_database_query_validation",
    ),
    _AnalysisTimeoutPhasePolicy(
        "cross_arch_build_timeout_phase",
        "_analysis_command_matches_cross_arch_build",
    ),
    _AnalysisTimeoutPhasePolicy(
        "emulator_validation_timeout_phase",
        "_analysis_command_matches_emulator_validation",
    ),
    _AnalysisTimeoutPhasePolicy(
        "image_render_validation_timeout_phase",
        "_analysis_command_matches_image_render_validation",
    ),
    _AnalysisTimeoutPhasePolicy(
        "service_inventory_probe_timeout_phase",
        "_analysis_command_matches_service_inventory_probe",
    ),
    _AnalysisTimeoutPhasePolicy(
        "tool_binary_download_timeout_phase",
        "_analysis_command_matches_tool_binary_download",
    ),
    _AnalysisTimeoutPhasePolicy(
        "external_media_metadata_timeout_phase",
        "_analysis_command_matches_external_media_metadata_probe",
    ),
    _AnalysisTimeoutPhasePolicy(
        "remote_dataset_fetch_timeout_phase",
        "_analysis_command_matches_remote_dataset_fetch",
    ),
    _AnalysisTimeoutPhasePolicy(
        "network_probe_timeout_phase",
        "_analysis_command_matches_network_probe",
    ),
    _AnalysisTimeoutPhasePolicy(
        "media_batch_processing_timeout_phase",
        "_analysis_command_matches_media_batch_processing",
    ),
    _AnalysisTimeoutPhasePolicy(
        "vm_service_readiness_timeout_phase",
        "_analysis_command_matches_vm_service_readiness",
    ),
    _AnalysisTimeoutPhasePolicy(
        "simulation_validation_timeout_phase",
        "_analysis_command_matches_simulation_validation",
    ),
    _AnalysisTimeoutPhasePolicy(
        "statistical_eval_validation_timeout_phase",
        "_analysis_command_matches_statistical_eval_validation",
    ),
    _AnalysisTimeoutPhasePolicy(
        "log_file_read_timeout_phase",
        "_analysis_command_matches_large_log_file_read",
        fallback_only=True,
    ),
    _AnalysisTimeoutPhasePolicy(
        "long_compute_timeout_phase",
        "_analysis_command_matches_long_compute",
    ),
    _AnalysisTimeoutPhasePolicy(
        "benchmark_validation_timeout_phase",
        "_analysis_command_matches_benchmark_validation",
    ),
    _AnalysisTimeoutPhasePolicy(
        "directory_listing_timeout_phase",
        "_analysis_command_matches_directory_listing",
        fallback_only=True,
    ),
    _AnalysisTimeoutPhasePolicy(
        "background_process_wait_timeout_phase",
        "_analysis_command_matches_background_process_wait",
        fallback_only=True,
    ),
)


def _policy_matches_for_event(event: dict[str, Any]) -> list[str]:
    command = event["command"]
    output = event["output"]
    matches: list[str] = []
    if background_package_command_reason(command):
        matches.append("background_package_command_guard")
    if broad_root_find_command_reason(command):
        matches.append("broad_root_find_guard")
    if broad_proc_scan_command_reason(command):
        matches.append("broad_proc_scan_guard")
    if _analysis_command_matches_verifier_artifact_search(command):
        matches.append("verifier_artifact_search_guard")
    large_toolchain_command = large_toolchain_install_command_reason(command)
    manual_deb_chase_command = manual_deb_dependency_chase_reason(command)
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    returncode = metadata.get("exit_code")
    try:
        returncode = int(returncode) if returncode is not None else None
    except (TypeError, ValueError):
        returncode = None
    semantic_failure = shell_semantic_failure_kind(
        output,
        command=command,
        returncode=returncode,
    )
    repeated_dependency_timeout_guard = "Blocked repeated dependency timeout path" in output
    repeated_dependency_failure_guard = "Blocked repeated dependency failure path" in output
    repeated_git_history_timeout_guard = "Blocked repeated git-history timeout path" in output
    repeated_cross_arch_timeout_guard = (
        "Blocked repeated cross-architecture timeout path" in output
    )
    repeated_compute_timeout_guard = "Blocked repeated compute timeout path" in output
    repeated_media_batch_timeout_guard = (
        "Blocked repeated media batch timeout path" in output
    )
    repeated_remote_dataset_timeout_guard = (
        "Blocked repeated remote dataset timeout path" in output
    )
    repeated_model_extraction_timeout_guard = (
        "Blocked repeated model extraction timeout path" in output
    )
    repeated_database_query_timeout_guard = (
        "Blocked repeated database query timeout path" in output
    )
    repeated_generated_solver_timeout_guard = (
        "Blocked repeated generated solver timeout path" in output
    )
    repeated_generated_exploration_script_timeout_guard = (
        "Blocked repeated generated exploration script timeout path" in output
    )
    repeated_password_cracking_timeout_guard = (
        "Blocked repeated password cracking timeout path" in output
    )
    repeated_service_inventory_timeout_guard = (
        "Blocked repeated service inventory timeout path" in output
    )
    repeated_background_process_wait_timeout_guard = (
        "Blocked repeated background process wait timeout path" in output
    )
    repeated_simulation_validation_timeout_guard = (
        "Blocked repeated simulation validation timeout path" in output
    )
    repeated_statistical_eval_validation_timeout_guard = (
        "Blocked repeated statistical eval validation timeout path" in output
    )
    repeated_image_render_validation_timeout_guard = (
        "Blocked repeated image render validation timeout path" in output
    )
    timeout_phase_candidate = (
        event["timed_out"]
        and not repeated_dependency_timeout_guard
        and not repeated_dependency_failure_guard
        and not repeated_git_history_timeout_guard
        and not repeated_cross_arch_timeout_guard
        and not repeated_compute_timeout_guard
        and not repeated_media_batch_timeout_guard
        and not repeated_remote_dataset_timeout_guard
        and not repeated_model_extraction_timeout_guard
        and not repeated_database_query_timeout_guard
        and not repeated_generated_solver_timeout_guard
        and not repeated_generated_exploration_script_timeout_guard
        and not repeated_password_cracking_timeout_guard
        and not repeated_service_inventory_timeout_guard
        and not repeated_background_process_wait_timeout_guard
        and not repeated_simulation_validation_timeout_guard
        and not repeated_statistical_eval_validation_timeout_guard
        and not repeated_image_render_validation_timeout_guard
    )
    capped_timeout, note = package_manager_timeout_cap(command, 120.0)
    if (
        note
        and capped_timeout < 120.0
        and not large_toolchain_command
        and not manual_deb_chase_command
        and not repeated_dependency_timeout_guard
        and not repeated_dependency_failure_guard
    ):
        matches.append("package_manager_timeout_cap")
    if semantic_failure == "package_manager_failure":
        matches.append("package_manager_semantic_failure")
    if semantic_failure == "network_probe_tool_missing":
        matches.append("network_probe_tool_missing")
    if semantic_failure == "large_toolchain_install_plan":
        matches.append("large_toolchain_install_plan")
    if manual_dependency_download_reason(command):
        matches.append("manual_dependency_download_guard")
    if manual_deb_chase_command:
        matches.append("manual_deb_dependency_chase_guard")
    if large_toolchain_command:
        matches.append("large_toolchain_install_guard")
    if staged_dependency_script_reason(event.get("file_path", ""), event.get("content", "")):
        matches.append("staged_dependency_script_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[0:5],
            )
        )
    if repeated_dependency_timeout_guard:
        matches.append("repeated_dependency_timeout_path_guard")
    if repeated_dependency_failure_guard:
        matches.append("repeated_dependency_failure_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[5:7],
            )
        )
    if repeated_git_history_timeout_guard:
        matches.append("repeated_git_history_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[7:9],
            )
        )
    if repeated_generated_exploration_script_timeout_guard:
        matches.append("repeated_generated_exploration_script_timeout_path_guard")
    if repeated_generated_solver_timeout_guard:
        matches.append("repeated_generated_solver_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[9:10],
            )
        )
    if repeated_password_cracking_timeout_guard:
        matches.append("repeated_password_cracking_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[10:11],
            )
        )
    if repeated_model_extraction_timeout_guard:
        matches.append("repeated_model_extraction_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[11:12],
            )
        )
    if repeated_database_query_timeout_guard:
        matches.append("repeated_database_query_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[12:14],
            )
        )
    if repeated_cross_arch_timeout_guard:
        matches.append("repeated_cross_arch_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[14:15],
            )
        )
    if repeated_image_render_validation_timeout_guard:
        matches.append("repeated_image_render_validation_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[15:16],
            )
        )
    if repeated_service_inventory_timeout_guard:
        matches.append("repeated_service_inventory_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[16:19],
            )
        )
    if repeated_remote_dataset_timeout_guard:
        matches.append("repeated_remote_dataset_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[19:21],
            )
        )
    if repeated_media_batch_timeout_guard:
        matches.append("repeated_media_batch_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[21:23],
            )
        )
    if repeated_simulation_validation_timeout_guard:
        matches.append("repeated_simulation_validation_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[23:24],
            )
        )
    if repeated_statistical_eval_validation_timeout_guard:
        matches.append("repeated_statistical_eval_validation_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[24:27],
            )
        )
    if repeated_compute_timeout_guard:
        matches.append("repeated_compute_timeout_path_guard")
    if timeout_phase_candidate:
        matches.extend(
            _analysis_timeout_phase_policy_matches(
                command,
                matches,
                _ANALYSIS_TIMEOUT_PHASE_POLICIES[27:29],
            )
        )
    if repeated_background_process_wait_timeout_guard:
        matches.append("repeated_background_process_wait_timeout_path_guard")
    if _event_is_artifact_progress(event):
        matches.append("artifact_check_deliverable_progress")
    return matches


def _analysis_command_matches_manual_dependency_download(command: str) -> bool:
    return manual_dependency_download_reason(command) is not None


def _analysis_command_matches_package_cache_search(command: str) -> bool:
    lowered = command.lower()
    return "find /tmp" in lowered and any(
        marker in lowered
        for marker in [".whl", "*.whl", ".tar.gz", "*.tar.gz", ".deb", "pip-unpack", "pip-download"]
    )


def _analysis_command_matches_scratch_artifact_search(command: str) -> bool:
    lowered = command.lower()
    searches_tmp = any(
        marker in lowered for marker in ["find /tmp", "ls -la /tmp", "ls /tmp", "/tmp/rtmp"]
    )
    if not searches_tmp:
        return False
    scratch_artifact_marker = any(
        marker in lowered
        for marker in [
            "*.db",
            "*.db*",
            "*.wal",
            "*wal*",
            ".wal",
            ".db",
            "*.csv",
            "rtmp",
            "*.lock",
            "-name \"lock\"",
            "-name 'lock'",
            "*.pt",
            "*.pth",
            "checkpoint",
            "model",
            "/tmp/c4",
        ]
    )
    broad_tmp_listing = (
        "find /tmp -type d" in lowered
        or ("find /tmp" in lowered and "-exec ls" in lowered)
        or ("ls -la /tmp" in lowered and "*" in lowered)
    )
    return scratch_artifact_marker or broad_tmp_listing


def _analysis_command_matches_git_history_search(command: str) -> bool:
    lowered = command.lower()
    full_history_search = "git grep" in lowered and any(
        marker in lowered
        for marker in ["git rev-list --all", "$(git rev-list", "`git rev-list"]
    )
    commit_loop_secret_search = (
        "git log --all" in lowered
        and any(marker in lowered for marker in ["while", "for "])
        and any(
            marker in lowered
            for marker in ["token", "secret", "akia", "ghp_", "hf_"]
        )
    )
    tree_filter_rewrite = "git filter-branch" in lowered and any(
        marker in lowered
        for marker in ["--tree-filter", "find . -type f", "grep -rl", "sed -i"]
    )
    broad_object_secret_check = "git rev-list --objects" in lowered and any(
        marker in lowered for marker in ["secret", "token", "akia", "ghp_", "hf_"]
    )
    return (
        full_history_search
        or commit_loop_secret_search
        or tree_filter_rewrite
        or broad_object_secret_check
    )


def _analysis_command_matches_verifier_artifact_search(command: str) -> bool:
    lowered = command.lower()
    return any(
        re.search(rf"(?<![\w./-]){re.escape(marker)}(?:/|(?![\w./-]))", lowered)
        for marker in ("/logs/verifier", "/tmp/hl-verifier-cache")
    )


def _analysis_command_matches_network_probe(command: str) -> bool:
    lowered = command.lower()
    _, package_note = package_manager_timeout_cap(command, 120.0)
    if (
        package_note
        or _analysis_command_matches_manual_dependency_download(command)
        or _analysis_command_matches_tool_binary_download(command)
    ):
        return False
    has_short_timeout = any(
        marker in lowered for marker in ["timeout 5", "timeout 10", "--connect-timeout", "timeout="]
    )
    url_probe = any(
        marker in lowered for marker in ["urlopen(", "requests.get", "curl ", "wget "]
    ) and any(marker in lowered for marker in ["http://", "https://"])
    generic_host_probe = any(
        marker in lowered
        for marker in [
            "google.com",
            "huggingface.co",
            "ping -c",
            "nc -z",
            "wget --spider",
            "curl -i",
            "curl -s -i",
            "curl --head",
        ]
    )
    return (has_short_timeout and url_probe) or generic_host_probe


def _analysis_command_matches_remote_dataset_fetch(command: str) -> bool:
    lowered = command.lower()
    dataset_marker = any(
        marker in lowered
        for marker in [
            "huggingface",
            "hfapi",
            "load_dataset",
            "dataset_info",
            "/datasets/",
            "*.arrow",
            "*.parquet",
            "*.jsonl",
        ]
    )
    if not dataset_marker:
        return False
    return any(
        marker in lowered
        for marker in [
            "load_dataset",
            "dataset_info",
            "huggingface_hub",
            "from datasets import",
            "hfapi",
            "requests_ca_bundle",
            "curl_ca_bundle",
            "find /",
            "find /tmp",
            "find /app",
            "*.arrow",
            "*.parquet",
            "*.jsonl",
        ]
    )


def _analysis_command_matches_generated_solver_search(command: str) -> bool:
    lowered = command.lower()
    generated_rule_artifact = any(
        marker in lowered
        for marker in [
            "re.json",
            "rules.json",
            "generated_rules",
            "generated solver",
            "generated_solver",
            "gen_simple.py",
            "re.sub",
            "regex",
        ]
    )
    generator_or_validator = any(
        marker in lowered
        for marker in [
            "gen_simple.py",
            "gen_regex",
            "gen_rules",
            "generate_rules",
            "check.py",
            "run_solution",
            "json.load(open('/app/re.json'))",
            'json.load(open("/app/re.json"))',
        ]
    )
    executable_script = any(
        marker in lowered for marker in ["python ", "python3 ", "python -c", "python3 -c", "cat >"]
    )
    return generated_rule_artifact and generator_or_validator and executable_script


def _analysis_command_matches_generated_exploration_script(command: str) -> bool:
    lowered = command.lower()
    if _analysis_command_matches_local_validation(command):
        return False
    if _analysis_command_matches_generated_solver_search(command):
        return False
    if _analysis_command_matches_model_extraction_probe(command):
        return False
    if _analysis_command_matches_database_query_validation(command):
        return False
    if not _command_invokes_python(lowered):
        return False
    if _analysis_command_matches_small_generated_exploration_script(command):
        return False
    script_markers = [
        "explore.py",
        "analyze_data.py",
        "analysis_data.py",
        "inspect_data.py",
        "probe_data.py",
        "download_data.py",
        "count_tokens.py",
        "count_data.py",
        "token_count.py",
        "sample_data.py",
    ]
    return any(marker in lowered for marker in script_markers)


def _analysis_command_matches_small_generated_exploration_script(command: str) -> bool:
    lowered = command.lower()
    tokens = _shell_tokens_lower(command)
    small_markers = [
        "tiny",
        "smoke",
        "small sample",
        "--limit",
        "--max-rows",
        "--rows",
        "--nrows",
        "--head",
        "--sample",
        "sample-size",
        "sample_size",
        "head -",
    ]
    return any(marker in lowered for marker in small_markers) or _command_has_short_timeout(
        tokens, 10
    )


def _analysis_command_matches_password_cracking_search(command: str) -> bool:
    lowered = command.lower()
    uses_cracker = any(
        marker in lowered
        for marker in ["john/run/john", "/john ", "./john", " john ", "hashcat"]
    )
    if not uses_cracker:
        return False
    return any(
        marker in lowered
        for marker in [
            "--wordlist",
            "--incremental",
            "--mask",
            "--max-run-time",
            "--show",
            "potfile",
            "password.lst",
            "small_wordlist",
            "7z_hash",
            "secrets_hash",
            "clean_hash",
        ]
    )


def _analysis_command_matches_model_extraction_probe(command: str) -> bool:
    lowered = command.lower()
    executable_script = any(
        marker in lowered
        for marker in [
            "python ",
            "python3 ",
            "python -c",
            "python3 -c",
            "python3 <<",
            "cat >",
        ]
    )
    if not executable_script:
        return False
    direct_extraction_script = any(
        marker in lowered
        for marker in [
            "steal.py",
            "steal_a1",
            "extract_a1",
            "stolen_a1",
            "test_extract",
            "debug_gradient",
            "test_binary_boundary",
        ]
    )
    forward_module_probe = any(
        marker in lowered
        for marker in ["from forward import", "import forward", "fwd_module.forward"]
    ) and any(
        marker in lowered
        for marker in ["np.", "numpy", "linspace", "num_points", "breakpoints", "sweep_1d"]
    )
    relu_logit_probe = any(marker in lowered for marker in ["relu", "logits"]) and any(
        marker in lowered for marker in ["forward", "true_a1", "stolen_a1", "steal"]
    )
    return direct_extraction_script or forward_module_probe or relu_logit_probe


def _analysis_command_matches_database_query_validation(command: str) -> bool:
    lowered = command.lower()
    if not _command_uses_database_cli(lowered):
        return False
    if _analysis_command_matches_small_database_probe(lowered):
        return False
    query_text = any(
        marker in lowered
        for marker in [
            "select ",
            "with ",
            "join ",
            "group by",
            "order by",
            "cat my-sql-query.sql",
            "cat sol.sql",
            "< /app/my-sql-query.sql",
            ".read",
            "explain query plan",
        ]
    )
    full_validation_marker = any(
        marker in lowered
        for marker in [
            "time ",
            "timeout ",
            "> /tmp/",
            ".output /tmp/",
            "wc -l",
            "head -",
            "original_output",
            "out_orig",
            "final_output",
            "my_final_output",
        ]
    )
    return query_text and full_validation_marker or _analysis_command_matches_database_recovery_dump(
        command
    )


def _command_uses_database_cli(lowered: str) -> bool:
    return any(
        marker in lowered for marker in ["sqlite3 ", " duckdb ", " psql ", " mysql "]
    )


def _analysis_command_matches_database_recovery_dump(command: str) -> bool:
    lowered = command.lower()
    if not _command_uses_database_cli(lowered):
        return False
    if _analysis_command_matches_small_database_probe(lowered):
        return False
    recovery_marker = any(
        marker in lowered
        for marker in [
            ".recover",
            ".dump",
            ".backup",
            ".clone",
            "pragma wal_checkpoint",
            "wal_checkpoint",
            "-readonly",
        ]
    )
    full_output_marker = any(
        marker in lowered
        for marker in [
            "> /tmp/",
            "> /app/",
            "full.sql",
            "dump.sql",
            "recovered.db",
            "recovered.json",
            "sqlite_master",
            "wc -l",
            "time ",
            "timeout ",
        ]
    )
    return recovery_marker and full_output_marker


def _analysis_command_matches_small_database_probe(lowered: str) -> bool:
    schema_or_plan_probe = any(
        marker in lowered
        for marker in [
            "explain query plan",
            ".schema",
            ".tables",
            "pragma table_info",
            "pragma index_list",
            "pragma index_info",
            "pragma database_list",
            "select name from sqlite_master",
            "select sql from sqlite_master",
        ]
    )
    artifact_shape_probe = any(
        marker in lowered
        for marker in [
            "test -s /app/recovered.json",
            "test -f /app/recovered.json",
            "json.tool /app/recovered.json",
            "open('/app/recovered.json')",
            'open("/app/recovered.json")',
            "stat /app/recovered.json",
            "wc -c /app/recovered.json",
        ]
    ) or ("head -" in lowered and "/app/recovered.json" in lowered)
    limit_match = re.search(r"\blimit\s+([0-9]+)(?![0-9])", lowered)
    return schema_or_plan_probe or artifact_shape_probe or bool(
        limit_match and int(limit_match.group(1)) <= 100
    )


def _analysis_command_matches_service_inventory_probe(command: str) -> bool:
    lowered = command.lower()
    service_marker = any(
        marker in lowered
        for marker in [
            "nginx",
            "mailman",
            " ssh ",
            "sshpass",
            "stricthostkeychecking",
            "passwordauthentication",
            "websockify",
            "novnc",
            "http://localhost",
            "localhost:",
            "127.0.0.1",
            "/proc/net/tcp",
            "ss -tln",
            "netstat",
            "/etc/postfix",
            "master.cf",
        ]
    ) or lowered.startswith("ssh ")
    if not service_marker:
        return False
    return any(
        marker in lowered
        for marker in [
            "/etc/nginx",
            "nginx.conf",
            "sites-enabled",
            "conf.d",
            "websockify --help",
            "novnc_proxy",
            "/usr/share/novnc",
            "/proc/net/tcp",
            "ss -tln",
            "netstat",
            "urlopen('http://localhost",
            'urlopen("http://localhost',
            "curl http://localhost",
            "curl -s http://localhost",
            "wget http://localhost",
            "wget -q http://localhost",
            "wget -qo- http://localhost",
            "wget -q -o - http://localhost",
            "wget -q -o- http://localhost",
            "wget http://127.0.0.1",
            "wget -q http://127.0.0.1",
            "wget -qo- http://127.0.0.1",
            "wget -q -o - http://127.0.0.1",
            "wget -q -o- http://127.0.0.1",
            "mailman --run-as-root conf",
            "mailman conf",
            "stricthostkeychecking",
            "passwordauthentication",
            "sshpass",
            "/etc/postfix",
            "master.cf",
        ]
    ) or (
        (" ssh " in lowered or lowered.startswith("ssh "))
        and any(
            marker in lowered
            for marker in ["localhost", "127.0.0.1", "timeout ", "-p "]
        )
    )


def _analysis_command_matches_background_process_wait(command: str) -> bool:
    lowered = command.lower()
    tokens = _shell_tokens_lower(command)
    if _analysis_command_matches_small_background_process_probe(lowered, tokens):
        return False
    process_table_poll = _analysis_command_matches_process_table_poll(lowered, tokens)
    background_operator = _command_has_background_operator(command)
    if _command_invokes_python(lowered) and not process_table_poll and not background_operator:
        return False
    has_shell_sleep = _command_has_shell_sleep(tokens, lowered)
    sleep_loop = has_shell_sleep and any(
        marker in lowered
        for marker in ["while", "for ", "$(seq", "kill -0", "wait "]
    )
    sleep_log_poll = has_shell_sleep and _command_tails_or_cats_log(lowered)
    _, package_note = package_manager_timeout_cap(command, 120.0)
    if package_note and not process_table_poll:
        return False
    blind_wait = (
        lowered.strip() == "wait"
        or lowered.strip().startswith("wait;")
        or ("wait" in tokens and "echo" in lowered)
    )
    background_launch_then_wait = background_operator and (
        has_shell_sleep
        or "wait" in tokens
        or process_table_poll
        or _command_tails_or_cats_log(lowered)
    )
    return sleep_loop or sleep_log_poll or process_table_poll or blind_wait or background_launch_then_wait


def _analysis_command_matches_small_background_process_probe(
    lowered: str,
    tokens: list[str],
) -> bool:
    explicit_small_probe = any(
        marker in lowered
        for marker in [
            "one concrete pid",
            "one pid",
            "known pid",
            "single pid",
            "specific pid",
            "one bounded process line",
            "bounded process line",
            "small log slice",
            "existing log slice",
            "explicit output artifact",
        ]
    )
    bounded_text_probe = _command_has_head_or_tail_limit_at_most(tokens, 80)
    single_pid_probe = (
        "/proc/" in lowered
        and "/proc/*" not in lowered
        and "/proc/[0-9]" not in lowered
        and "$(ls /proc" not in lowered
        and bounded_text_probe
    )
    return explicit_small_probe or single_pid_probe or (
        _command_has_short_timeout(tokens, 10)
        and "while" not in lowered
        and "for " not in lowered
    )


def _analysis_command_matches_process_table_poll(lowered: str, tokens: list[str]) -> bool:
    process_tool = any(token in {"ps", "pgrep", "pidof"} for token in tokens)
    grep_process = any(
        marker in lowered
        for marker in ["ps aux", "ps -ef", "ps -eo", "pgrep"]
    ) or ("grep -q" in lowered and "ps " in lowered)
    status_probe = "kill -0" in lowered or "fuser " in lowered
    return (process_tool and grep_process) or status_probe


def _command_invokes_python(lowered: str) -> bool:
    return any(
        marker in lowered
        for marker in ["python ", "python3 ", "python -c", "python3 -c", "python <<", "python3 <<"]
    )


def _command_has_shell_sleep(tokens: list[str], lowered: str) -> bool:
    return (
        "sleep" in tokens
        or "; sleep" in lowered
        or "&& sleep" in lowered
        or " do sleep" in lowered
    )


def _command_tails_or_cats_log(lowered: str) -> bool:
    reads_text = any(marker in lowered for marker in ["cat ", "tail ", "head "])
    log_or_scratch = any(
        marker in lowered
        for marker in [".log", "/tmp/", "/var/log", "_out", "_err"]
    )
    return reads_text and log_or_scratch


def _command_has_background_operator(command: str) -> bool:
    for index, char in enumerate(command):
        if char != "&":
            continue
        prev = command[index - 1] if index > 0 else ""
        next_char = command[index + 1] if index + 1 < len(command) else ""
        if prev in {"&", ">"} or next_char in {"&", ">"} or next_char.isdigit():
            continue
        return True
    return False


def _command_has_short_timeout(tokens: list[str], max_seconds: int) -> bool:
    for index, token in enumerate(tokens):
        if token != "timeout":
            continue
        for candidate in tokens[index + 1 : index + 5]:
            if candidate.startswith("-"):
                continue
            value = candidate.rstrip("s")
            try:
                return float(value) <= max_seconds
            except ValueError:
                return False
    return False


def _command_has_head_or_tail_limit_at_most(tokens: list[str], max_value: int) -> bool:
    for first, second in zip(tokens, tokens[1:]):
        if first not in {"head", "tail"}:
            continue
        try:
            if float(second.lstrip("-")) <= max_value:
                return True
        except ValueError:
            continue
    return False


def _analysis_command_matches_cross_arch_build(command: str) -> bool:
    lowered = command.lower()
    target_marker = any(
        marker in lowered
        for marker in [
            "build_mips",
            "doomgeneric_mips",
            "mips-linux",
            "mipsel-linux",
            "arm-linux-gnueabi",
            "aarch64-linux",
            "riscv64-linux",
            "powerpc-linux",
            "--target=mips",
            "--target mips",
            "target=mips",
            "cross-compile",
            "cross compile",
            "cmake_toolchain_file",
        ]
    )
    if not target_marker:
        return False
    return any(
        marker in lowered
        for marker in [
            "build",
            "make",
            "cmake",
            "gcc",
            "g++",
            "clang",
            "ld ",
            "ar ",
            "bash ",
            "sh ",
            "cargo build",
        ]
    )


def _analysis_command_matches_emulator_validation(command: str) -> bool:
    lowered = command.lower()
    emulator_marker = any(
        marker in lowered
        for marker in [
            "loadelf",
            "load_elf",
            "new cpu",
            "cpu(",
            "cpu.run",
            "cpu.step",
            "vm.js",
            "qemu-mips",
            "qemu-arm",
            "qemu-aarch64",
            "qemu-riscv",
            "qemu-ppc",
        ]
    )
    if not emulator_marker:
        return False
    return any(
        marker in lowered
        for marker in [
            "doomgeneric_mips",
            ".elf",
            "loadelf",
            "entrypoint",
            "pc",
            "register",
            "memory",
            "cycle",
            "step",
            "run",
        ]
    )


def _analysis_command_matches_image_render_validation(command: str) -> bool:
    lowered = command.lower()
    media_batch_markers = [
        "tesseract",
        "pytesseract",
        "ffmpeg",
        "cv2.videocapture",
        "extract_frames",
        "extract_commands.py",
        "extract_moves",
        "ocr_batch.py",
        "ocr_small.py",
        "frame_*.png",
        "frame_*.jpg",
        "frame_%04d",
        "keyframe_",
        "frames_crop",
        "frames/",
        "frames_all",
        "for f in frame",
        "for f in frames",
        "for i in $(seq",
        "video.mp4",
        ".mkv",
        ".webm",
        "ocr",
    ]
    if any(marker in lowered for marker in media_batch_markers):
        return False
    ppm_or_tga_artifact = any(
        marker in lowered
        for marker in [
            ".ppm",
            "reconstructed.ppm",
            "image.ppm",
            "image2.ppm",
            "original.ppm",
            "illum1.tga",
            ".tga",
        ]
    )
    render_or_ray_marker = any(
        marker in lowered
        for marker in [
            "povray",
            "path tracing",
            "path-tracing",
            "ray tracing",
            "raytrac",
            "render",
            "rays",
            "samples per pixel",
            "chroot /jail",
            "./image",
            "/image",
            "/reverse",
            "+w640",
            "+h480",
            "+ft",
            "+a0.1",
        ]
    )
    image_compare_marker = any(
        marker in lowered
        for marker in [
            "ssim",
            "image similarity",
            "cosine similarity",
            "np.array",
            "pil.image",
            "from pil import image",
            "image.open",
            "getdata()",
            "full-frame",
            "full pixel",
        ]
    ) or ("numpy" in lowered and "flatten" in lowered) or (
        "pixel" in lowered and "compare" in lowered
    )
    png_render_artifact = ".png" in lowered and any(
        marker in lowered
        for marker in [
            "povray",
            "path tracing",
            "path-tracing",
            "ray tracing",
            "raytrac",
            "samples per pixel",
            "ssim",
            "image similarity",
            "cosine similarity",
            "full-frame",
            "full pixel",
        ]
    )
    render_artifact = ppm_or_tga_artifact or png_render_artifact
    ppm_header_or_dimension_check = any(
        marker in lowered for marker in ["p6", "pnmfile", "identify "]
    ) or (render_artifact and any(marker in lowered for marker in ["file ", "head -", "xxd"]))
    render_runner = any(
        marker in lowered
        for marker in [
            "chroot /jail /image",
            "chroot /jail_orig /mystery",
            "chroot /jail_clean /reverse",
            "./image >",
            "./reverse >",
            "/usr/local/bin/povray",
            " povray ",
        ]
    )
    return (
        render_artifact
        and (render_or_ray_marker or image_compare_marker or ppm_header_or_dimension_check)
    ) or (render_runner and (render_or_ray_marker or image_compare_marker))


def _analysis_command_matches_environment_inventory(command: str) -> bool:
    lowered = command.lower()
    lists_system_bins = any(
        marker in lowered
        for marker in [
            "ls /usr/bin",
            "ls -la /usr/bin",
            "find /usr/bin",
            "compgen -a command",
            "compgen -a",
            "compgen -c",
            "compgen -A command".lower(),
        ]
    )
    multi_tool_probe = any(marker in lowered for marker in ["which ", "command -v "]) and len(
        lowered.split()
    ) >= 10 and any(
        marker in lowered
        for marker in [" sleep ", " timeout ", " chmod ", " basename "]
    )
    root_header_or_binary_find = "find / " in lowered and any(
        marker in lowered
        for marker in [
            "stdio.h",
            "python*",
            "pip*",
            "ggml-*",
            "whisper*",
            "*.bin",
            "*.pt",
            "*.pth",
        ]
    )
    return lists_system_bins or multi_tool_probe or root_header_or_binary_find


def _analysis_command_matches_directory_listing(command: str) -> bool:
    lowered = command.lower()
    tokens = _analysis_shell_tokens(lowered)
    if not tokens:
        return False
    if (
        "| head" in lowered
        or "| tail" in lowered
        or "--max-depth" in lowered
        or "-maxdepth" in lowered
    ):
        return False
    if _command_targets_specific_file(tokens):
        return False
    has_listing_tool = any(token in {"ls", "du", "tree", "find"} for token in tokens)
    if not has_listing_tool:
        return False
    broad_listing_flag = any(
        token in {"-r", "-la", "-al", "-lra", "-ral"} for token in tokens
    ) or any(token.startswith("--recursive") for token in tokens)
    explicit_directory = any(
        token.endswith("/")
        or token in {".", "..", "/", "/app", "/tmp", "/var", "/usr", "/home"}
        or token.startswith(("/app/", "/tmp/", "/var/", "/usr/", "/home/"))
        for token in tokens
    )
    find_directory_walk = "find" in tokens and any(
        token in {"/", "/app", "/tmp", "/var", "/usr", "/home", "."}
        or token.startswith(("/app/", "/tmp/", "/var/", "/usr/", "/home/"))
        for token in tokens
    )
    plain_directory_ls = "ls" in tokens and explicit_directory
    directory_du = "du" in tokens and explicit_directory
    tree_directory_walk = "tree" in tokens and explicit_directory
    return (
        plain_directory_ls
        or directory_du
        or tree_directory_walk
        or (broad_listing_flag and explicit_directory)
        or find_directory_walk
    )


def _analysis_shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _command_targets_specific_file(tokens: list[str]) -> bool:
    file_suffixes = (
        ".c",
        ".cc",
        ".cpp",
        ".csv",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".ml",
        ".py",
        ".rs",
        ".txt",
        ".yaml",
        ".yml",
    )
    return any(token.lower().endswith(file_suffixes) for token in tokens)


def _analysis_command_matches_tool_binary_download(command: str) -> bool:
    lowered = command.lower()
    _, package_note = package_manager_timeout_cap(command, 120.0)
    if package_note:
        return False
    network_fetch = any(
        marker in lowered
        for marker in [
            "curl ",
            "wget ",
            "urllib.request",
            "urlretrieve",
            "requests.get",
            "yt-dlp/releases",
            "github.com/ytdl-org",
            "github.com/yt-dlp",
            "ffmpeg-builds",
            "johnvansickle.com/ffmpeg",
        ]
    )
    binary_archive = any(
        marker in lowered
        for marker in [
            "ffmpeg",
            "yt-dlp",
            "youtube-dl",
            ".tar.xz",
            ".tgz",
            ".zip",
            "/usr/local/bin/",
        ]
    )
    return network_fetch and binary_archive


def _analysis_command_matches_external_media_metadata_probe(command: str) -> bool:
    lowered = command.lower()
    uses_media_fetcher = "yt-dlp" in lowered or "youtube-dl" in lowered
    metadata_only = any(
        marker in lowered
        for marker in [
            "--print",
            "--dump-json",
            "--dump-single-json",
            "--get-title",
            "--get-description",
            "--skip-download",
            "--write-info-json",
            " -j ",
        ]
    )
    external_url = any(
        marker in lowered for marker in ["http://", "https://", "youtube.com", "youtu.be"]
    )
    return uses_media_fetcher and metadata_only and external_url


def _analysis_command_matches_media_batch_processing(command: str) -> bool:
    lowered = command.lower()
    media_or_ocr_tool = any(
        marker in lowered
        for marker in [
            "tesseract",
            "ffmpeg",
            "pytesseract",
            "extract_frames",
            "extract_commands.py",
            "extract_moves",
            "ocr_batch.py",
            "ocr_small.py",
            "cv2.videocapture",
            "image.open",
            "ocr",
        ]
    )
    batch_scope = any(
        marker in lowered
        for marker in [
            "frame_*.png",
            "frame_*.jpg",
            "frame_%04d",
            "keyframe_",
            "f_*.jpg",
            "f2/f_",
            "f3/f_",
            "scene_*.png",
            "/app/scenes",
            "frames_crop",
            "frames/",
            "frames_all",
            "frames2",
            "frames3",
            "frames5",
            "frame_$(",
            "for f in frame",
            "for f in frames",
            "for i in $(seq",
            "os.listdir(frames",
            "os.listdir('.')",
            "range(1, 191)",
            "glob.glob",
            "processpoolexecutor",
            "extract_moves",
            "extract_commands.py",
            "ocr_batch.py",
            "ocr_small.py",
        ]
    )
    frame_sampling_or_copy = any(
        marker in lowered
        for marker in [
            "frames_crop_sample",
            "roughly 1 fps",
            "every 30th",
            "grep -op",
            "% 30",
            "%30",
            "mod 30",
            "cp \"$f\"",
            "cp $f",
            "sample",
        ]
    ) and any(
        marker in lowered for marker in ["for f in", "frame_*."]
    )
    scene_scan = "ffmpeg" in lowered and any(
        marker in lowered
        for marker in ["select='gt(scene", "select=gt(scene", "showinfo", "pts_time", "keyframe_"]
    )
    inline_video_ocr = any(
        marker in lowered for marker in ["pytesseract", "tesseract", "cv2.videocapture"]
    ) and any(
        marker in lowered
        for marker in [
            "frame",
            "frames",
            "scene",
            "video",
            "zcbvypbhrfa",
            "speedrun",
            ".mp4",
            ".mkv",
            ".webm",
        ]
    )
    return scene_scan or inline_video_ocr or (batch_scope and (media_or_ocr_tool or frame_sampling_or_copy))


def _analysis_command_matches_simulation_validation(command: str) -> bool:
    lowered = command.lower()
    direct_simulator = any(
        marker in lowered
        for marker in [
            "mujoco",
            "mjdata",
            "total_sim_time",
            "simulate_model",
            "m.opt.timestep",
            "./sim",
        ]
    )
    eval_script = "eval.py" in lowered and any(
        marker in lowered for marker in ["sim", "model", "timestep", "mujoco"]
    )
    sim_reference_probe = "./sim" in lowered and any(
        marker in lowered for marker in ["subprocess", "for ", "timeout ", "cd /app", "./sim "]
    )
    return direct_simulator or eval_script or sim_reference_probe


def _analysis_command_matches_vm_service_readiness(command: str) -> bool:
    lowered = command.lower()
    qemu_or_alpine_marker = any(
        marker in lowered
        for marker in [
            "qemu-system",
            "alpine_boot",
            "vm_ready.flag",
            "expect_output.log",
            "alpine_output.log",
            "expect_test",
            "/tmp/boot",
            "vmlinuz",
            "initramfs",
            "combined-initrd",
        ]
    ) or ("isoinfo" in lowered and "alpine.iso" in lowered)
    boot_wait_loop = any(marker in lowered for marker in ["sleep ", "for i in $(seq"]) and (
        "vm_ready" in lowered
        or "alpine_boot" in lowered
        or ("qemu" in lowered and "boot" in lowered)
        or ("alpine" in lowered and "boot" in lowered)
    )
    boot_log_poll = (
        any(marker in lowered for marker in ["cat ", "tail "])
        and "/tmp/" in lowered
        and (
            "alpine_boot" in lowered
            or "expect_output.log" in lowered
            or "alpine_output.log" in lowered
            or ("qemu" in lowered and "boot" in lowered)
        )
    )
    verifier_connection = any(
        marker in lowered for marker in ["sshpass", "root@localhost", "telnet"]
    ) and any(
        marker in lowered for marker in ["uname -r", "kernel version", "2222", "6665"]
    )
    return qemu_or_alpine_marker or boot_wait_loop or boot_log_poll or verifier_connection


def _analysis_command_matches_statistical_eval_validation(command: str) -> bool:
    lowered = command.lower()
    runs_r_sampling = any(marker in lowered for marker in ["rscript", "r --"]) and any(
        marker in lowered
        for marker in ["ars.r", "ars(", "dnorm", "test()", "test(n", "sapply(", "sample"]
    )
    full_eval_script = any(
        marker in lowered for marker in ["python3 eval.py", "python eval.py", "./eval.py"]
    ) and "explain query plan" not in lowered
    return runs_r_sampling or full_eval_script


def _analysis_command_matches_build_compile(command: str) -> bool:
    tokens = _shell_tokens_lower(command)
    return any(
        token in {
            "make",
            "cmake",
            "ninja",
            "cargo",
            "gcc",
            "g++",
            "clang",
            "clang++",
            "rustc",
            "javac",
            "mvn",
            "gradle",
        }
        for token in tokens
    )


def _analysis_command_matches_local_validation(command: str) -> bool:
    lowered = command.lower()
    if "pytest" in lowered or "py.test" in lowered:
        return True
    validation_script = any(
        marker in lowered for marker in ["test_", "final_test", "validate", "check_"]
    )
    return validation_script and any(
        marker in lowered
        for marker in ["perl ", "python ", "python3 ", "ruby ", "node ", "cat >"]
    )


def _shell_tokens_lower(command: str) -> list[str]:
    try:
        return [Path(token).name.lower() for token in shlex.split(command, posix=True)]
    except ValueError:
        return [Path(token).name.lower() for token in command.split()]


def _analysis_command_matches_benchmark_validation(command: str) -> bool:
    lowered = command.lower()
    markers = [
        "benchmark.py",
        "bench.py",
        "run_benchmark",
        "benchmark --",
        "performance benchmark",
    ]
    return any(marker in lowered for marker in markers)


def _analysis_command_matches_long_compute(command: str) -> bool:
    lowered = command.lower()
    markers = [
        "fasttext supervised",
        "fasttext test",
        "fasttext predict",
        "train_supervised",
        "pystan_analysis.py",
        "stan_model",
        "sampling(",
        "rscript analysis.r",
        "r -f analysis.r",
        "r --file=analysis.r",
        "python3 train",
        "python train",
        "minimal_train.py",
        "quick_train.py",
        "train_final.py",
        "optimized_packer.py",
        "decompress.py",
        "./attack",
    ]
    large_dataset_probe = any(
        marker in lowered for marker in ["pyarrow.parquet", "pq.read_table", "load_dataset("]
    ) and any(
        marker in lowered
        for marker in [
            "train-00000",
            "len(ds)",
            "features",
            "label distribution",
            "first 50000",
            "first 100000",
        ]
    )
    large_sample_file_op = any(
        marker in lowered
        for marker in ["head -20000", "head -50000", "head -100000", "head -200000"]
    ) and any(marker in lowered for marker in ["train.txt", ".parquet"])
    full_input_batch_edit = (
        "input.csv" in lowered
        and any(marker in lowered for marker in ["1m rows", "full input", "test_output.csv"])
        and ("vim -nu" in lowered or "vim " in lowered)
    )
    return (
        any(marker in lowered for marker in markers)
        or large_dataset_probe
        or large_sample_file_op
        or full_input_batch_edit
    )


def _analysis_command_matches_large_log_file_read(command: str) -> bool:
    lowered = command.lower()
    bounded_task_log_slice = (
        (
            lowered.startswith("head ")
            or lowered.startswith("tail ")
            or " head " in lowered
            or " tail " in lowered
        )
        and "/app/logs/" in lowered
        and ".log" in lowered
    )
    if bounded_task_log_slice:
        return True
    if "cat " not in lowered or ".log" not in lowered:
        return False
    if any(marker in lowered for marker in ["| head", "| tail", "sed -n", "grep ", "rg "]):
        return False
    return any(token.endswith(".log") for token in _analysis_shell_tokens(lowered))


def _event_is_artifact_progress(event: dict[str, Any]) -> bool:
    if event.get("success") is False or event["timed_out"]:
        return False
    if _event_writes_deliverable_path(event):
        return True
    command = event["command"]
    first_token = ""
    try:
        tokens = shlex.split(command, posix=True)
        first_token = Path(tokens[0]).name if tokens else ""
    except ValueError:
        first_token = command.split(maxsplit=1)[0] if command.split() else ""
    if first_token not in {
        "ls",
        "stat",
        "file",
        "wc",
        "head",
        "tail",
        "cat",
        "du",
        "test",
        "[",
    }:
        return False
    targets = [*event.get("expected_artifacts", []), *event.get("artifacts", [])]
    if not targets:
        return False
    if _artifact_probe_output_shows_missing_target(event, targets):
        return False
    haystack = f"{command}\n{event['output']}"
    for target in targets:
        name = Path(str(target)).name
        if target and (str(target) in haystack or (name and name in haystack)):
            return True
    return False


def _artifact_probe_output_shows_missing_target(
    event: dict[str, Any],
    targets: list[Any],
) -> bool:
    output = str(event.get("output") or "").lower()
    if not output:
        return False
    missing_markers = (
        "no such file or directory",
        "cannot access",
        "does not exist",
        "not found",
        "missing",
    )
    if not any(marker in output for marker in missing_markers):
        return False
    return _text_mentions_any_artifact_target(output, targets)


def _event_writes_deliverable_path(event: dict[str, Any]) -> bool:
    tool = str(event.get("tool") or "").rsplit(".", 1)[-1].lower()
    if tool not in {"write", "edit", "file_write", "file_edit"}:
        return False
    file_path = str(event.get("file_path") or "").strip()
    if not file_path:
        return False
    targets = [*event.get("expected_artifacts", []), *event.get("artifacts", [])]
    if targets and _text_mentions_any_artifact_target(file_path, targets):
        return True
    if targets:
        return False
    path = Path(file_path)
    return path.is_absolute() and len(path.parts) >= 3 and path.parts[1] == "app"


def _text_mentions_any_artifact_target(text: str, targets: list[Any]) -> bool:
    lowered = str(text).lower()
    for target in targets:
        cleaned = str(target).strip()
        if not cleaned:
            continue
        name = Path(cleaned).name
        if cleaned.lower() in lowered or (name and name.lower() in lowered):
            return True
    return False


def _record_policy_example(policy: dict[str, Any], event: dict[str, Any]) -> None:
    policy["count"] += 1
    policy["tasks"].add(event["task_id"])
    if len(policy["examples"]) < 5:
        policy["examples"].append(_policy_example(event))


def _policy_example(event: dict[str, Any]) -> dict[str, str]:
    return {
        "task_id": event["task_id"],
        "command": _truncate_one_line(event["command"], 180),
    }


def _truncate_one_line(text: str, max_chars: int) -> str:
    compact = " ".join(_redact_analysis_text(str(text)).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _redact_analysis_text(text: str) -> str:
    redacted = str(text)
    redacted = re.sub(
        r"(?i)\b(authorization\s*[:=]\s*)(?:bearer|basic)?\s*[A-Za-z0-9._~+/=-]{8,}",
        r"\1[REDACTED_SECRET]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)\b(cookie\s*[:=]\s*)[^\n|]{8,}",
        r"\1[REDACTED_SECRET]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd|pwd)\s*[:=]\s*([^\s'\"&|;]+)",
        r"\1=[REDACTED_SECRET]",
        redacted,
    )
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED_SECRET]", redacted)
    redacted = re.sub(r"\bghp_[A-Za-z0-9_]{20,}\b", "[REDACTED_SECRET]", redacted)
    redacted = re.sub(r"\bgithub_pat_[A-Za-z0-9_]+\b", "[REDACTED_SECRET]", redacted)
    redacted = re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "[REDACTED_SECRET]", redacted)
    redacted = re.sub(r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED_SECRET]", redacted)
    return redacted


def _analysis_policy_coverage_markdown(policy_coverage: dict[str, Any]) -> str:
    policies = policy_coverage.get("policies", {})
    lines = [
        _markdown_table(
            ["policy", "count", "tasks", "example"],
            [
                [
                    name,
                    str(data.get("count", 0)),
                    ", ".join(data.get("tasks") or []),
                    (data.get("examples") or [{}])[0].get("command", ""),
                ]
                for name, data in policies.items()
                if int(data.get("count", 0) or 0) > 0
            ],
        )
    ]
    uncovered = policy_coverage.get("uncovered_timeout_examples") or []
    lines.extend(["", "Uncovered timeout examples:"])
    lines.extend(
        f"- {item.get('task_id', '')}: `{item.get('command', '')}`" for item in uncovered
    )
    if not uncovered:
        lines.append("- none")
    return "\n".join(lines)


def _analysis_trajectory_evidence(trial: Any) -> dict[str, Any]:
    events = _trial_policy_events(trial)
    recent_commands = [_analysis_event_summary(event) for event in events[-8:]]
    timed_out_commands: list[dict[str, str]] = []
    blocked_guards: list[dict[str, str]] = []
    dependency_and_toolchain: list[dict[str, str]] = []
    deliverable_progress: list[dict[str, str]] = []
    terminal_environment_markers: list[dict[str, str]] = []
    failure_mechanisms = _analysis_failure_mechanisms_for_trial(trial)
    policy_counts: dict[str, int] = {}
    for mechanism in failure_mechanisms:
        name = mechanism.get("name", "")
        if name:
            policy_counts[name] = policy_counts.get(name, 0) + 1

    for event in events:
        matches = _policy_matches_for_event(event)
        for policy in matches:
            policy_counts[policy] = policy_counts.get(policy, 0) + 1
        summary = _analysis_event_summary(event)
        if event["timed_out"] and len(timed_out_commands) < 8:
            timed_out_commands.append(summary)
        if _analysis_event_has_blocked_guard(event) and len(blocked_guards) < 8:
            blocked_guards.append(summary | {"guards": ", ".join(matches)})
        if (
            _analysis_event_is_dependency_or_toolchain(event, matches)
            and len(dependency_and_toolchain) < 8
        ):
            dependency_and_toolchain.append(summary | {"policies": ", ".join(matches)})
        if (
            "artifact_check_deliverable_progress" in matches
            and len(deliverable_progress) < 5
        ):
            deliverable_progress.append(summary)
        if (
            _analysis_event_has_terminal_environment_marker(event)
            and len(terminal_environment_markers) < 5
        ):
            terminal_environment_markers.append(summary)

    high_signal_policy_counts = {
        key: policy_counts[key]
        for key in sorted(policy_counts)
        if policy_counts[key] > 0
    }
    return {
        "recent_commands": recent_commands,
        "timed_out_commands": timed_out_commands,
        "blocked_guards": blocked_guards,
        "dependency_and_toolchain_evidence": dependency_and_toolchain,
        "deliverable_progress": deliverable_progress,
        "terminal_environment_markers": terminal_environment_markers,
        "failure_mechanisms": failure_mechanisms,
        "policy_counts": high_signal_policy_counts,
    }


def _analysis_event_summary(event: dict[str, Any]) -> dict[str, str]:
    output = _redact_analysis_text(event.get("output", ""))
    return {
        "tool": _truncate_one_line(event.get("tool", "") or "tool", 40),
        "command": _truncate_one_line(event.get("command", ""), 220),
        "timed_out": "yes" if event.get("timed_out") else "no",
        "success": str(event.get("success")),
        "output_tail": _truncate_one_line(output[-600:], 220),
    }


def _analysis_event_has_blocked_guard(event: dict[str, Any]) -> bool:
    lowered = str(event.get("output") or "").lower()
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    blocked_by = str(metadata.get("blocked_by") or "")
    return (
        bool(blocked_by)
        or "blocked repeated" in lowered
        or "worker shell policy blocked" in lowered
    )


def _analysis_event_is_dependency_or_toolchain(
    event: dict[str, Any],
    matches: list[str],
) -> bool:
    command = event.get("command", "")
    if any(policy in _ANALYSIS_DEPENDENCY_TOOLCHAIN_POLICIES for policy in matches):
        return True
    lowered = str(command).lower()
    return any(
        marker in lowered
        for marker in [
            "apt-get",
            "apt ",
            "pip install",
            "dpkg",
            "r cmd install",
            "cran.r-project",
            "pypi.org",
            "files.pythonhosted.org",
            "conda install",
            "cargo install",
            "toolchain",
            "build-essential",
            "gcc-mips",
            "g++-mips",
            "fasttext",
            "rstan",
            "httpstan",
        ]
    )


def _analysis_event_has_terminal_environment_marker(event: dict[str, Any]) -> bool:
    text = f"{event.get('command', '')}\n{event.get('output', '')}".lower()
    return _text_has_terminal_environment_marker(text)


def _trial_has_terminal_environment_signal(trial: Any) -> bool:
    metadata = getattr(trial, "metadata", {}) or {}
    if bool(metadata.get("terminal_environment_unavailable")):
        return True
    text_parts = [
        str(metadata.get("terminal_environment_marker") or ""),
        str(metadata.get("terminal_environment_reason") or ""),
        "\n".join(str(line) for line in (getattr(trial, "error_log", []) or [])),
        str(getattr(trial, "verifier_output", "") or ""),
    ]
    if _text_has_terminal_environment_marker("\n".join(text_parts).lower()):
        return True
    return any(
        _analysis_event_has_terminal_environment_marker(event)
        for event in _trial_policy_events(trial)
    )


def _text_has_terminal_environment_marker(text: str) -> bool:
    return any(
        marker in text
        for marker in [
            "terminal_environment_unavailable",
            "service \"main\" is not running",
            "broken pipe",
            "environment unavailable",
            "container is not running",
            "cannot exec in a stopped state",
        ]
    )


def _analysis_trajectory_evidence_markdown(evidence: dict[str, Any]) -> str:
    sections = [
        ("Failure Mechanisms", evidence.get("failure_mechanisms") or []),
        ("Recent Commands", evidence.get("recent_commands") or []),
        ("Timed-Out Commands", evidence.get("timed_out_commands") or []),
        ("Blocked Guard Evidence", evidence.get("blocked_guards") or []),
        (
            "Dependency And Toolchain Evidence",
            evidence.get("dependency_and_toolchain_evidence") or [],
        ),
        ("Deliverable Progress", evidence.get("deliverable_progress") or []),
        (
            "Terminal Environment Markers",
            evidence.get("terminal_environment_markers") or [],
        ),
    ]
    lines: list[str] = []
    policy_counts = evidence.get("policy_counts") or {}
    lines.extend(["## Policy Counts", ""])
    if policy_counts:
        for policy, count in policy_counts.items():
            lines.append(f"- {policy}: {count}")
    else:
        lines.append("- none")
    for title, items in sections:
        lines.extend(["", f"## {title}", ""])
        if not items:
            lines.append("- none")
            continue
        for item in items:
            suffix_parts = []
            if item.get("timed_out"):
                suffix_parts.append(f"timeout={item['timed_out']}")
            if item.get("success"):
                suffix_parts.append(f"success={item['success']}")
            if item.get("guards"):
                suffix_parts.append(f"guards={item['guards']}")
            if item.get("policies"):
                suffix_parts.append(f"policies={item['policies']}")
            if item.get("name"):
                suffix_parts.append(f"mechanism={item['name']}")
            suffix = " (" + "; ".join(suffix_parts) + ")" if suffix_parts else ""
            primary = item.get("command") or item.get("evidence") or item.get("description") or ""
            lines.append(f"- `{primary}`{suffix}")
            if item.get("output_tail"):
                lines.append(f"  - output: {item['output_tail']}")
            if item.get("description") and item.get("evidence"):
                lines.append(f"  - detail: {item['description']}")
    return "\n".join(lines)


def _trial_is_infrastructure_failure(trial: Any) -> bool:
    return trial_is_infrastructure_failure(trial)


def _analysis_detail_markdown(
    trial: Any,
    *,
    evidence: dict[str, Any] | None = None,
) -> str:
    metadata = getattr(trial, "metadata", {}) or {}
    evidence = evidence if evidence is not None else _analysis_trajectory_evidence(trial)
    return "\n".join(
        [
            f"# Task {getattr(trial, 'task_id', '')}",
            "",
            f"- Trial: {getattr(trial, 'trial_id', '')}",
            f"- Status: {getattr(getattr(trial, 'status', None), 'value', '')}",
            f"- Score: {float(getattr(trial, 'score', 0.0) or 0.0):.4f}",
            f"- Verified: {bool(getattr(trial, 'verified', False))}",
            f"- Timeout phase: {metadata.get('timeout_phase', '')}",
            f"- Harbor job: {getattr(trial, 'harbor_job_dir', '') or 'not recorded'}",
            "",
            _analysis_trajectory_evidence_markdown(evidence),
            "",
            "## Error Tail",
            "\n".join(
                f"- {_redact_analysis_text(str(line))}"
                for line in (getattr(trial, "error_log", []) or [])[-8:]
            )
            or "- none",
            "",
            "## Verifier Tail",
            _tail_text(
                _redact_analysis_text(str(getattr(trial, "verifier_output", "") or "")),
                4000,
            )
            or "none",
            "",
        ]
    )


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "none"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(_escape_md_cell(value) for value in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _escape_md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _tail_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _pre_regression_should_run(
    regression_plan: dict[str, Any],
    *,
    iteration: int,
    last_codex_update_summary: Any | None,
    submit_requested: bool,
    update_policy: dict[str, int],
) -> bool:
    if not regression_plan["pre"]["enabled"]:
        return False
    if submit_requested or last_codex_update_summary is not None:
        return True
    _ = iteration, update_policy
    return True


def _validate_pending_regression_snapshots(
    regression_argv: list[str],
    memory: Any,
    summary: Any,
) -> int:
    for argv in _pending_regression_validation_commands(
        regression_argv,
        memory,
        summary,
    ):
        result = _run_regression(argv)
        if result.returncode != 0:
            return result.returncode
    return 0


def _pending_regression_validation_commands(
    regression_argv: list[str],
    memory: Any,
    summary: Any,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for task_id in _pending_regression_task_ids(memory, summary):
        commands.append(
            [
                *regression_argv,
                "--task",
                task_id,
                "--snapshot-status",
                "pending",
            ]
        )
    return commands


def _pending_regression_task_ids(memory: Any, summary: Any) -> list[str]:
    if not hasattr(memory, "get_regression_snapshot"):
        return []
    task_ids: list[str] = []
    model_scopes = _summary_passed_task_model_scopes(memory, summary)
    for task_id in _summary_passed_task_ids(memory, summary):
        snapshot = _get_regression_snapshot(
            memory,
            task_id,
            model_scope=model_scopes.get(task_id, ""),
        )
        if snapshot is None:
            continue
        if snapshot.validation_status != "pending":
            continue
        if snapshot.source_summary_id != summary.summary_id:
            continue
        task_ids.append(task_id)
    return task_ids


def _summary_has_accepted_codex_update(summary: Any) -> bool:
    patches = getattr(summary, "patches_applied", []) or []
    return "codex_update:accepted" in patches


def _last_accepted_codex_update_summary(campaign_state: dict[str, Any]) -> Any | None:
    entry = campaign_state.get("last_accepted_codex_update")
    if not isinstance(entry, dict):
        return None
    summary_id = str(entry.get("summary_id") or "")
    trial_ids = [str(trial_id) for trial_id in entry.get("trial_ids") or []]
    if not summary_id or not trial_ids:
        return None
    return type(
        "CampaignAcceptedCodexUpdateSummary",
        (),
        {
            "summary_id": summary_id,
            "trial_ids": trial_ids,
            "patches_applied": [
                str(patch) for patch in entry.get("patches_applied") or []
            ],
            "overall_score": float(entry.get("overall_score") or 0.0),
            "codex_update_diff_path": str(entry.get("diff_path") or ""),
            "codex_update_packet_id": str(entry.get("packet_id") or ""),
        },
    )()


def _accepted_codex_update_summary(summary: Any, *, diff_path: str = "") -> Any:
    return type(
        "CampaignAcceptedCodexUpdateSummary",
        (),
        {
            "summary_id": str(summary.summary_id),
            "trial_ids": [str(trial_id) for trial_id in summary.trial_ids],
            "patches_applied": [str(patch) for patch in summary.patches_applied],
            "overall_score": float(getattr(summary, "overall_score", 0.0) or 0.0),
            "codex_update_diff_path": str(diff_path),
            "codex_update_packet_id": _codex_packet_id_from_diff_path(diff_path),
        },
    )()


def _codex_packet_id_from_diff_path(diff_path: str) -> str:
    if not diff_path:
        return ""
    path = Path(diff_path)
    parent = path.parent.name
    return parent if parent.startswith("codex_packet_") else ""


def _accepted_codex_update_diff_path(loop: Any, summary: Any) -> str:
    if not _summary_has_accepted_codex_update(summary):
        return ""
    update_engine = getattr(getattr(loop, "system", None), "update_engine", None)
    stack = getattr(update_engine, "_accepted_run_stack", None)
    if stack:
        last_run = stack[-1]
        return str(getattr(last_run, "diff_path", "") or "")
    last_run = getattr(update_engine, "_last_run", None)
    if last_run is not None and getattr(getattr(last_run, "review", None), "accepted", False):
        return str(getattr(last_run, "diff_path", "") or "")
    return ""


def _rollback_codex_update_after_failed_validation(
    loop: Any,
    memory: Any,
    summary: Any | None = None,
    *,
    reason: str,
) -> bool:
    update_engine = getattr(getattr(loop, "system", None), "update_engine", None)
    if update_engine is None:
        return False
    diff_path = (
        str(getattr(summary, "codex_update_diff_path", "") or "")
        if summary is not None
        else ""
    )
    if hasattr(update_engine, "rollback_last_accepted"):
        rolled_back = bool(update_engine.rollback_last_accepted())
        if not rolled_back and diff_path and hasattr(update_engine, "rollback_diff"):
            rolled_back = bool(update_engine.rollback_diff(diff_path))
    elif diff_path and hasattr(update_engine, "rollback_diff"):
        rolled_back = bool(update_engine.rollback_diff(diff_path))
    elif hasattr(update_engine, "rollback_last"):
        rolled_back = bool(update_engine.rollback_last())
    else:
        return False
    if hasattr(memory, "save_component_lesson"):
        summary_id = str(getattr(summary, "summary_id", "") or "")
        packet_id = (
            str(getattr(summary, "codex_update_packet_id", "") or "")
            if summary is not None
            else ""
        )
        memory.save_component_lesson(
            "codex_update",
            _codex_update_outcome_lesson(
                "post_update_rollback",
                packet_id=packet_id,
                outcome="rollback_applied" if rolled_back else "rollback_failed",
                reason=reason,
                summary_id=summary_id,
                rollback_applied=rolled_back,
            ),
        )
    if rolled_back:
        system = getattr(loop, "system", None)
        patch_count = (
            int(getattr(system, "patch_count", 0) or 0)
            if system is not None
            else 0
        )
        if system is not None and patch_count > 0:
            system.patch_count = patch_count - 1
        if hasattr(loop, "_clear_pending_update_baseline"):
            loop._clear_pending_update_baseline()
    return rolled_back


def _recover_from_codex_validation_failure(
    loop: Any,
    memory: Any,
    campaign_state: dict[str, Any],
    memory_path: Path,
    campaign_id: str,
    summary: Any,
    *,
    status: int,
    failure_marker: str,
    reason: str,
) -> bool:
    """Rollback a bad accepted Codex update and keep the campaign alive."""
    _invalidate_pending_regression_snapshots(
        memory,
        summary,
        reason=reason,
        include_stable=True,
    )
    rolled_back = _rollback_codex_update_after_failed_validation(
        loop,
        memory,
        summary,
        reason=reason,
    )
    if rolled_back:
        _mark_codex_update_rolled_back(summary, failure_marker)
        if hasattr(memory, "record_summary") and hasattr(summary, "model_dump_json"):
            memory.record_summary(summary)
        _record_campaign_summary(campaign_state, summary)
        _clear_last_accepted_codex_update(campaign_state)
    _record_codex_validation_failure(
        campaign_state,
        summary,
        status=status,
        reason=reason,
        rolled_back=rolled_back,
        failure_marker=failure_marker,
    )
    packet_id = _codex_update_packet_id_for_summary(campaign_state, summary)
    if packet_id:
        _attach_next_eval_result_to_failure_attempts(
            campaign_state,
            packet_id=packet_id,
            outcome="validation_failed",
        )
    _write_campaign_state(memory_path, campaign_id, campaign_state)
    return rolled_back


def _mark_codex_update_rolled_back(summary: Any, marker: str) -> None:
    patches = [
        str(patch)
        for patch in (getattr(summary, "patches_applied", []) or [])
        if str(patch) != "codex_update:accepted"
    ]
    if marker not in patches:
        patches.append(marker)
    summary.patches_applied = patches


def _record_codex_validation_failure(
    campaign_state: dict[str, Any],
    summary: Any,
    *,
    status: int,
    reason: str,
    rolled_back: bool,
    failure_marker: str,
) -> None:
    failures = campaign_state.setdefault("codex_validation_failures", [])
    packet_id = _codex_update_packet_id_for_summary(campaign_state, summary)
    failures.append(
        {
            "packet_id": packet_id,
            "summary_id": str(getattr(summary, "summary_id", "") or ""),
            "trial_ids": [
                str(trial_id)
                for trial_id in getattr(summary, "trial_ids", []) or []
            ],
            "exit_code": int(status),
            "reason": reason,
            "rolled_back": bool(rolled_back),
            "failure_marker": failure_marker,
            "recorded_at": datetime.now().isoformat(),
        }
    )
    campaign_state["updated_at"] = datetime.now().isoformat()


def _recover_from_baseline_pre_regression_failure(
    memory: Any,
    campaign_state: dict[str, Any],
    memory_path: Path,
    campaign_id: str,
    regression_argv: list[str],
    pre_result: RegressionRunResult,
    *,
    model_scope: str,
    transient_cooldown_seconds: int = 0,
    retry_failed_tasks: bool = False,
    reason: str,
) -> bool:
    """Quarantine unstable baseline snapshots so long campaigns keep running."""
    if not pre_result.failed_tasks:
        _record_regression_gate_event(
            campaign_state,
            phase="pre",
            action="unparsed_failure",
            tasks=[],
            exit_code=pre_result.returncode,
            reason=reason,
            model_scope=model_scope,
        )
        _write_campaign_state(memory_path, campaign_id, campaign_state)
        return False

    retry_passed: list[str] = []
    quarantined: list[str] = []
    for task_id in pre_result.failed_tasks:
        retry_exit_code = pre_result.returncode
        quarantine_reason = (
            f"{reason}; baseline failure was already attributed by the "
            "completed pre-regression gate, so the stale or unstable snapshot "
            "was quarantined without a same-task retry"
        )
        if retry_failed_tasks:
            retry = _run_regression(
                [
                    *regression_argv,
                    "--task",
                    task_id,
                ]
            )
            if retry.returncode == 0:
                retry_passed.append(task_id)
                continue
            retry_exit_code = retry.returncode
            quarantine_reason = (
                f"{reason}; targeted retry also failed with exit code "
                f"{retry.returncode}"
            )
        if _quarantine_regression_snapshot(
            memory,
            task_id,
            model_scope=model_scope,
            reason=quarantine_reason,
        ):
            quarantined.append(task_id)
        else:
            _record_regression_gate_event(
                campaign_state,
                phase="pre",
                action="quarantine_failed",
                tasks=[task_id],
                exit_code=retry_exit_code,
                reason=reason,
                model_scope=model_scope,
            )
            _write_campaign_state(memory_path, campaign_id, campaign_state)
            return False

    if retry_passed:
        for task_id in retry_passed:
            if hasattr(memory, "record_regression_transient_failure"):
                memory.record_regression_transient_failure(
                    task_id,
                    model_scope=model_scope,
                    reason=reason,
                    cooldown_seconds=transient_cooldown_seconds,
                )
        _record_regression_gate_event(
            campaign_state,
            phase="pre",
            action="transient_retry_passed",
            tasks=retry_passed,
            exit_code=pre_result.returncode,
            reason=reason,
            model_scope=model_scope,
        )
        if hasattr(memory, "save_component_lesson"):
            memory.save_component_lesson(
                "regression_gate",
                (
                    "Pre-regression failed on stable snapshot(s), but targeted "
                    "retry passed; treating as transient baseline instability. "
                    f"Tasks: {', '.join(retry_passed)}. Reason: {reason}."
                ),
            )
    if quarantined:
        _record_regression_gate_event(
            campaign_state,
            phase="pre",
            action="quarantined_unstable_snapshot",
            tasks=quarantined,
            exit_code=pre_result.returncode,
            reason=reason,
            model_scope=model_scope,
        )
        if hasattr(memory, "save_component_lesson"):
            memory.save_component_lesson(
                "regression_gate",
                (
                    "Pre-regression stable snapshot(s) failed without an active "
                    "accepted Codex update to roll back, and targeted retry also "
                    "failed. The snapshot(s) were invalidated so the long HL "
                    "campaign can continue collecting evidence instead of being "
                    "stopped by a stale or flaky baseline. Tasks: "
                    f"{', '.join(quarantined)}. Reason: {reason}."
                ),
            )
    _write_campaign_state(memory_path, campaign_id, campaign_state)
    return True


def _quarantine_regression_snapshot(
    memory: Any,
    task_id: str,
    *,
    model_scope: str,
    reason: str,
) -> bool:
    if not hasattr(memory, "invalidate_regression"):
        return False
    return bool(
        memory.invalidate_regression(
            task_id,
            reason=reason,
            model_scope=model_scope,
        )
    )


def _quarantine_known_failed_baseline_snapshots(
    memory: Any,
    campaign_state: dict[str, Any],
    memory_path: Path,
    campaign_id: str,
    *,
    model_scope: str,
    reason: str,
) -> list[str]:
    if not hasattr(memory, "list_regression_snapshots"):
        return []

    quarantined: list[str] = []
    for snapshot in memory.list_regression_snapshots(model_scope=model_scope):
        if getattr(snapshot, "validation_status", "") != "stable":
            continue
        failures = int(getattr(snapshot, "regression_failures", 0) or 0)
        last_status = str(getattr(snapshot, "last_regression_status", "") or "")
        if failures <= 0 or last_status in {"", "passed"}:
            continue
        task_id = str(getattr(snapshot, "task_id", "") or "")
        if not task_id:
            continue
        if _quarantine_regression_snapshot(
            memory,
            task_id,
            model_scope=model_scope,
            reason=reason,
        ):
            quarantined.append(task_id)

    if not quarantined:
        return []

    _record_regression_gate_event(
        campaign_state,
        phase="pre",
        action="quarantined_known_failed_baseline_snapshot",
        tasks=quarantined,
        exit_code=0,
        reason=reason,
        model_scope=model_scope,
    )
    if hasattr(memory, "save_component_lesson"):
        memory.save_component_lesson(
            "regression_gate",
            (
                "Stable regression snapshot(s) in the current model scope "
                "already recorded a failed last regression run before any "
                "active accepted Codex update. They were invalidated before "
                "the next baseline gate so campaign restarts do not rerun the "
                "same known stale/flaky snapshot. Tasks: "
                f"{', '.join(quarantined)}. Reason: {reason}."
            ),
        )
    _write_campaign_state(memory_path, campaign_id, campaign_state)
    return quarantined


def _record_regression_gate_event(
    campaign_state: dict[str, Any],
    *,
    phase: str,
    action: str,
    tasks: list[str],
    exit_code: int,
    reason: str,
    model_scope: str,
) -> None:
    events = campaign_state.setdefault("regression_gate_events", [])
    events.append(
        {
            "phase": phase,
            "action": action,
            "tasks": [str(task) for task in tasks],
            "exit_code": int(exit_code),
            "reason": reason,
            "model_scope": model_scope,
            **_non_terminal_loop_event_metadata(),
            "recorded_at": datetime.now().isoformat(),
        }
    )
    campaign_state["updated_at"] = datetime.now().isoformat()


def _record_network_preflight_event(
    campaign_state: dict[str, Any],
    *,
    result: Any,
    timeout_seconds: int | None,
) -> dict[str, Any]:
    """Record network preflight output without making it a loop stop."""
    returncode = int(getattr(result, "returncode", 0) or 0)
    command = [str(item) for item in (getattr(result, "command", None) or [])]
    stdout = str(getattr(result, "stdout", "") or "")
    stderr = str(getattr(result, "stderr", "") or "")
    event = {
        "returncode": returncode,
        "ok": returncode == 0,
        "command": command,
        "timeout_seconds_audit_only": timeout_seconds,
        "network_preflight_failure_stop_condition": False,
        "network_preflight_timeout_stop_condition": False,
        "diagnostic_sub_agent_stop_condition": False,
        "diagnostic_sub_agent_timeout_stop_condition": False,
        "codex_update_sub_agent_stop_condition": False,
        "validation_regression_sub_agent_stop_condition": False,
        "worker_loop_stop_condition": False,
        "master_loop_stop_condition": False,
        "loop_stop_condition": False,
        "time_round_token_limit_driven": False,
        "stdout_tail": _tail_text(stdout, 4000),
        "stderr_tail": _tail_text(stderr, 4000),
        "recorded_at": datetime.now().isoformat(),
    }
    events = campaign_state.setdefault("network_preflight_events", [])
    events.append(event)
    if returncode != 0:
        _record_codex_update_event(
            campaign_state,
            action="audit",
            iteration=int(campaign_state.get("next_iteration", 0) or 0),
            reason=(
                "network preflight returned a non-zero diagnostic result; "
                "this is infrastructure evidence for Harbor/network recovery "
                "and is not a master, diagnostic sub-agent, Codex update "
                "sub-agent, or Worker loop stop condition"
            ),
        )
    campaign_state["updated_at"] = datetime.now().isoformat()
    return event


def _record_codex_update_event(
    campaign_state: dict[str, Any],
    *,
    action: str,
    iteration: int,
    reason: str,
) -> None:
    events = campaign_state.setdefault("codex_update_events", [])
    events.append(
        {
            "action": action,
            "iteration": int(iteration),
            "reason": reason,
            **_non_terminal_loop_event_metadata(),
            "recorded_at": datetime.now().isoformat(),
        }
    )
    campaign_state["updated_at"] = datetime.now().isoformat()


def _non_terminal_loop_event_metadata() -> dict[str, bool]:
    return all_loop_non_terminal_event_metadata()


def _record_codex_update_run_event(
    campaign_state: dict[str, Any],
    *,
    iteration: int,
    summary: Any,
    run: Any,
) -> None:
    review = getattr(run, "review", None)
    accepted = bool(getattr(review, "accepted", False))
    reasons = [str(reason) for reason in (getattr(review, "reasons", []) or [])]
    changed_files = [
        str(path) for path in (getattr(review, "changed_files", []) or [])
    ]
    final_report = getattr(run, "final_report", {}) or {}
    report_status = str(
        final_report.get("status", "") if isinstance(final_report, dict) else ""
    ).strip()
    decision_summary = str(
        final_report.get("summary", "") if isinstance(final_report, dict) else ""
    ).strip()
    action = "accepted" if accepted else "rejected"
    if not accepted and not changed_files and report_status == "noop":
        action = "noop"
    reason = decision_summary or "; ".join(reasons)
    if not reason:
        reason = f"{action} Codex update"
    implementation_scope = (
        final_report.get("implementation_scope")
        if isinstance(final_report, dict)
        else {}
    )
    if not isinstance(implementation_scope, dict):
        implementation_scope = {}
    generalization = (
        final_report.get("generalization")
        if isinstance(final_report, dict)
        else {}
    )
    if not isinstance(generalization, dict):
        generalization = {}
    component_layer = str(
        implementation_scope.get("primary_layer")
        or (final_report.get("component_type") if isinstance(final_report, dict) else "")
        or ""
    )
    failure_class = str(
        generalization.get("problem_class")
        or (
            final_report.get("summary", "")
            if isinstance(final_report, dict)
            else ""
        )
        or ""
    )
    packet_id = _codex_packet_id_from_run(run)
    mission_selection = _codex_update_mission_selection_from_run(run)
    events = campaign_state.setdefault("codex_update_events", [])
    event = {
        "action": action,
        "iteration": int(iteration),
        "summary_id": str(getattr(summary, "summary_id", "") or ""),
        "packet_id": packet_id,
        "exit_code": int(getattr(run, "exit_code", 0) or 0),
        "changed_files": changed_files,
        "reasons": reasons,
        "reason": reason,
        "review_reason": "; ".join(reasons),
        "report_status": report_status,
        "decision_summary": decision_summary,
        "failure_class": failure_class,
        "component_layer": component_layer,
        **_non_terminal_loop_event_metadata(),
        "recorded_at": datetime.now().isoformat(),
    }
    if mission_selection:
        event["mission_selection"] = mission_selection
        selected_candidate = str(
            mission_selection.get("selected_candidate_id") or ""
        ).strip()
        selected_category = str(
            mission_selection.get("selected_failure_category") or ""
        ).strip()
        if selected_candidate:
            event["mission_candidate_id"] = selected_candidate
        if selected_category:
            event["mission_failure_category"] = selected_category
    events.append(event)
    attempts = campaign_state.setdefault("failure_class_attempts", [])
    attempt = {
        "failure_class": failure_class,
        "component_layer": component_layer,
        "packet_id": packet_id,
        "accepted": accepted,
        "action": action,
        "next_eval_result": "",
        "summary_id": str(getattr(summary, "summary_id", "") or ""),
        "attempt_index_audit_only": len(attempts) + 1,
        **_non_terminal_loop_event_metadata(),
        "recorded_at": datetime.now().isoformat(),
    }
    if mission_selection:
        attempt["mission_selection"] = mission_selection
        selected_candidate = str(
            mission_selection.get("selected_candidate_id") or ""
        ).strip()
        selected_category = str(
            mission_selection.get("selected_failure_category") or ""
        ).strip()
        if selected_candidate:
            attempt["mission_candidate_id"] = selected_candidate
        if selected_category:
            attempt["mission_failure_category"] = selected_category
    attempts.append(attempt)
    campaign_state["failure_class_attempts"] = attempts
    if accepted and isinstance(campaign_state.get("last_accepted_codex_update"), dict):
        accepted_entry = campaign_state["last_accepted_codex_update"]
        accepted_entry["failure_class"] = failure_class
        accepted_entry["component_layer"] = component_layer
        if mission_selection:
            accepted_entry["mission_selection"] = mission_selection
            selected_candidate = str(
                mission_selection.get("selected_candidate_id") or ""
            ).strip()
            selected_category = str(
                mission_selection.get("selected_failure_category") or ""
            ).strip()
            if selected_candidate:
                accepted_entry["mission_candidate_id"] = selected_candidate
            if selected_category:
                accepted_entry["mission_failure_category"] = selected_category
    campaign_state["updated_at"] = datetime.now().isoformat()


def _codex_update_mission_selection_from_run(run: Any) -> dict[str, Any]:
    record_path = str(getattr(run, "record_path", "") or "").strip()
    if not record_path:
        return {}
    try:
        record = json.loads(Path(record_path).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(record, dict):
        return {}
    decision_inputs = record.get("update_decision_inputs")
    if not isinstance(decision_inputs, dict):
        return {}
    mission_selection = decision_inputs.get("mission_selection")
    if not isinstance(mission_selection, dict):
        return {}
    selected_candidate_id = str(
        mission_selection.get("selected_candidate_id") or ""
    ).strip()
    if bool(mission_selection.get("enforced")) and not selected_candidate_id:
        return {}
    return {
        "enforced": bool(mission_selection.get("enforced")),
        "selected_candidate_id": selected_candidate_id,
        "selected_failure_category": str(
            mission_selection.get("selected_failure_category") or ""
        ),
        "selected_allowed_edit_paths": [
            str(path)
            for path in mission_selection.get("selected_allowed_edit_paths") or []
        ],
        "selected_target_tasks": [
            str(task_id)
            for task_id in mission_selection.get("selected_target_tasks") or []
        ],
    }


def _codex_update_api_failure_reason(run: Any) -> str:
    return _codex_update_provider_failure(run)["reason"]


def _codex_update_provider_failure(run: Any) -> dict[str, Any]:
    review = getattr(run, "review", None)
    if bool(getattr(review, "accepted", False)):
        return {
            "reason": "",
            "terminal": False,
            "transient": False,
            "loop_stop_condition": False,
        }
    reasons = [str(reason) for reason in (getattr(review, "reasons", []) or [])]
    api_reasons = [
        reason for reason in reasons if _is_codex_update_api_failure_reason(reason)
    ]
    if not api_reasons:
        return {
            "reason": "",
            "terminal": False,
            "transient": False,
            "loop_stop_condition": False,
        }
    reason = "; ".join(api_reasons)
    transient = all(_is_transient_codex_update_provider_failure(item) for item in api_reasons)
    return {
        "reason": reason,
        "terminal": False,
        "transient": transient,
        "terminal_audit_only": not transient,
        "loop_stop_condition": False,
    }


def _is_codex_update_api_failure_reason(reason: str) -> bool:
    lowered = reason.lower()
    markers = (
        "upstream provider/auth failure",
        "auth_unavailable",
        "no auth available",
        "401 unauthorized",
        "403 forbidden",
        "429",
        "rate limit",
        "too many requests",
        "api key",
        "authentication",
        "unauthorized",
        "500 internal server error",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
        "service unavailable",
        "gateway timeout",
    )
    return any(marker in lowered for marker in markers)


def _is_transient_codex_update_provider_failure(reason: str) -> bool:
    lowered = reason.lower()
    terminal_markers = (
        "auth_unavailable",
        "no auth available",
        "401 unauthorized",
        "403 forbidden",
        "api key",
        "authentication",
        "unauthorized",
    )
    if any(marker in lowered for marker in terminal_markers):
        return False
    transient_markers = (
        "429",
        "rate limit",
        "too many requests",
        "500 internal server error",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
        "service unavailable",
        "gateway timeout",
    )
    return any(marker in lowered for marker in transient_markers)


def _codex_packet_id_from_run(run: Any) -> str:
    for field_name in ("diff_path", "packet_path", "events_path", "summary_path"):
        value = str(getattr(run, field_name, "") or "")
        packet_id = _codex_packet_id_from_diff_path(value)
        if packet_id:
            return packet_id
        parent = Path(value).parent.name if value else ""
        if parent.startswith("codex_packet_"):
            return parent
    return ""


def _mark_pending_regression_snapshots_stable(memory: Any, summary: Any) -> None:
    if not hasattr(memory, "mark_regression_stable"):
        return
    model_scopes = _summary_passed_task_model_scopes(memory, summary)
    for task_id in _summary_passed_task_ids(memory, summary):
        memory.mark_regression_stable(
            task_id,
            source_summary_id=summary.summary_id,
            model_scope=model_scopes.get(task_id, ""),
        )


def _invalidate_pending_regression_snapshots(
    memory: Any,
    summary: Any,
    *,
    reason: str,
    include_stable: bool = False,
) -> None:
    if not hasattr(memory, "invalidate_regression"):
        return
    model_scopes = _summary_passed_task_model_scopes(memory, summary)
    for task_id in _summary_passed_task_ids(memory, summary):
        if not include_stable:
            snapshot = _get_regression_snapshot(
                memory,
                task_id,
                model_scope=model_scopes.get(task_id, ""),
            )
            if snapshot is not None and snapshot.validation_status != "pending":
                continue
        memory.invalidate_regression(
            task_id,
            source_summary_id=summary.summary_id,
            reason=reason,
            model_scope=model_scopes.get(task_id, ""),
        )


def _summary_passed_task_ids(memory: Any, summary: Any) -> list[str]:
    task_ids: list[str] = []
    for trial_id in getattr(summary, "trial_ids", []) or []:
        try:
            trial = memory.get_trial(trial_id)
        except FileNotFoundError:
            continue
        if (
            getattr(trial, "status", None) is not None
            and getattr(trial.status, "value", "") == "passed"
            and getattr(trial, "verified", False)
            and float(getattr(trial, "score", 0.0) or 0.0) >= 1.0
        ):
            task_ids.append(str(trial.task_id))
    return task_ids


def _summary_passed_task_model_scopes(memory: Any, summary: Any) -> dict[str, str]:
    from hl.model_scope import model_scope_from_trial

    scopes: dict[str, str] = {}
    for trial_id in getattr(summary, "trial_ids", []) or []:
        try:
            trial = memory.get_trial(trial_id)
        except FileNotFoundError:
            continue
        if (
            getattr(getattr(trial, "status", None), "value", None) != "passed"
            or not getattr(trial, "verified", False)
            or float(getattr(trial, "score", 0.0) or 0.0) < 1.0
        ):
            continue
        scopes[str(trial.task_id)] = model_scope_from_trial(trial)
    return scopes


def _get_regression_snapshot(
    memory: Any,
    task_id: str,
    *,
    model_scope: str = "",
) -> Any:
    try:
        return memory.get_regression_snapshot(task_id, model_scope=model_scope)
    except TypeError:
        return memory.get_regression_snapshot(task_id)


def _submit_config(trials_config: dict[str, Any]):
    from hl.submit import SubmitConfig

    raw = trials_config.get("submit", {})
    allowed = set(SubmitConfig.__dataclass_fields__)
    return SubmitConfig(**{key: value for key, value in raw.items() if key in allowed})


def _resolve_submit_config(args: argparse.Namespace, trials_config: dict[str, Any]):
    config = _submit_config(trials_config)
    if args.submit_trigger_score is not None:
        config.trigger_score = args.submit_trigger_score
    if args.submit_min_tasks_evaluated is not None:
        config.min_tasks_evaluated = args.submit_min_tasks_evaluated
    if args.submit_min_attempts_per_task is not None:
        config.min_attempts_per_task = args.submit_min_attempts_per_task
    if args.submit_visibility:
        config.visibility = args.submit_visibility
    if args.submit_share_org is not None:
        config.share_orgs = args.submit_share_org
    if args.submit_share_user is not None:
        config.share_users = args.submit_share_user
    if args.submit_share_yes:
        config.share_yes = True
    return config


def _submit_plan_for_dry_run(
    *,
    args: argparse.Namespace,
    trials_config: dict[str, Any],
    memory_path: Path,
    fallback_job_dir: Path,
    tasks_evaluated: int,
) -> dict[str, Any]:
    from hl.submit import SubmitGate

    gate = SubmitGate(
        _resolve_submit_config(args, trials_config),
        submissions_dir=memory_path / "submissions",
    )
    result = gate.check(
        campaign_id=args.campaign_id,
        best_job_dir=args.best_job_dir or fallback_job_dir,
        score=0.0,
        tasks_evaluated=tasks_evaluated,
        full_regression_passed=False,
    )
    return {
        "eligible": result.eligible,
        "reasons": result.reasons,
        "command": _shell_join(result.command),
        "intent_path": result.intent_path,
        "result_path": result.result_path,
    }


def _submit_after_summary(
    *,
    args: argparse.Namespace,
    trials_config: dict[str, Any],
    memory: Any,
    memory_path: Path,
    summary: Any,
    full_regression_passed: bool,
):
    from hl.submit import SubmitGate

    gate = SubmitGate(
        _resolve_submit_config(args, trials_config),
        submissions_dir=memory_path / "submissions",
    )
    trials = [memory.get_trial(trial_id) for trial_id in summary.trial_ids]
    best_trial = max(trials, key=lambda trial: trial.score)
    attempts_per_task = _attempts_per_task_from_trials(trials)
    return gate.submit_once(
        campaign_id=args.campaign_id,
        best_job_dir=args.best_job_dir or best_trial.harbor_job_dir,
        score=summary.overall_score,
        tasks_evaluated=summary.total_tasks,
        full_regression_passed=full_regression_passed,
        attempts_per_task=attempts_per_task,
        dry_run=not args.submit,
    )


def _attempts_per_task_from_trials(trials: list[Any]) -> dict[str, int]:
    attempts: dict[str, int] = {}
    for trial in trials:
        task_id = str(getattr(trial, "task_id", "") or "")
        if not task_id:
            continue
        metadata = getattr(trial, "metadata", {}) or {}
        observed = metadata.get("attempts_observed_for_task")
        if observed is None:
            observed = metadata.get("harbor_trial_result_count")
        try:
            count = int(observed) if observed is not None else 1
        except (TypeError, ValueError):
            count = 1
        attempts[task_id] = attempts.get(task_id, 0) + max(1, count)
    return attempts


def _mission_debug_plan(args: argparse.Namespace, memory_path: Path) -> dict[str, Any]:
    output_path = (
        Path(args.mission_debug_output)
        if args.mission_debug_output
        else memory_path
        / "summaries"
        / f"{_safe_campaign_id(args.campaign_id)}_mission_debug.json"
    )
    return {
        "enabled": bool(args.mission_debug),
        "output": str(output_path),
        "max_features": args.mission_debug_max_features,
        "max_features_audit_only": args.mission_debug_max_features,
        "max_features_stop_condition": False,
        "feature_count_stop_condition": False,
        "target_task_count_stop_condition": False,
        "validation_contract_count_stop_condition": False,
        "time_and_round_limits_stop_condition": False,
        "time_limit_stop_condition": False,
        "round_limit_stop_condition": False,
        "attempt_count_stop_condition": False,
        "note": (
            "Mission debug max-features, feature count, target-task count, "
            "validation-contract count, time, round, and attempt values are "
            "audit/reference fields only; they do not stop mission-debug or "
            "diagnostic sub-agent execution and do not truncate the candidate list."
        ),
    }


def _network_preflight_plan(args: argparse.Namespace) -> dict[str, Any]:
    mode = args.network_preflight_mode or "quick"
    timeout = args.network_preflight_timeout or 120
    argv = _network_preflight_argv(args)
    return {
        "enabled": not bool(args.skip_network_preflight),
        "mode": mode,
        "timeout_seconds": timeout,
        "docker_resource_args": docker_resource_forward_args(args),
        "timeout_seconds_audit_only": timeout,
        "network_preflight_failure_stop_condition": False,
        "network_preflight_timeout_stop_condition": False,
        "diagnostic_sub_agent_stop_condition": False,
        "diagnostic_sub_agent_timeout_stop_condition": False,
        "codex_update_sub_agent_stop_condition": False,
        "validation_regression_sub_agent_stop_condition": False,
        "worker_loop_stop_condition": False,
        "master_loop_stop_condition": False,
        "loop_stop_condition": False,
        "time_round_token_limit_driven": False,
        "command": _shell_join(argv),
        "dry_run_behavior": (
            "reported only; checks are not executed and timeout is an audit "
            "reference, not a master/sub-agent/Worker loop stop condition"
        ),
    }


def _write_mission_debug_packet(
    *,
    args: argparse.Namespace,
    report: dict[str, Any],
    report_path: Path,
    memory_path: Path,
) -> dict[str, Any] | None:
    if not args.mission_debug:
        return None

    from meta.missions import MissionPlanner

    packet = MissionPlanner().from_campaign_summary(
        report,
        source_path=str(report_path),
        max_features=args.mission_debug_max_features,
    )
    output_path = (
        Path(args.mission_debug_output)
        if args.mission_debug_output
        else memory_path
        / "summaries"
        / f"{_safe_campaign_id(args.campaign_id)}_mission_debug.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(packet.model_dump_json(indent=2))
    return {
        "path": str(output_path),
        "mission_id": packet.mission_id,
        "max_features": args.mission_debug_max_features,
        "max_features_audit_only": args.mission_debug_max_features,
        "max_features_stop_condition": False,
        "feature_candidates": [feature.id for feature in packet.feature_candidates],
        "validation_contracts": [contract.id for contract in packet.validation_contracts],
    }


def _build_campaign_report(
    *,
    campaign_id: str,
    tasks: list[str],
    iteration_limit: int | None,
    summaries: list[Any],
    memory: Any,
    memory_path: Path,
    regression_plan: dict[str, Any],
    submit_results: list[Any],
    codex_update: bool,
    trials_config: dict[str, Any] | None = None,
    update_policy: dict[str, Any] | None = None,
    round_task_concurrency: int = 1,
    stopped_reason: str = "",
    include_task_results: bool = True,
) -> dict[str, Any]:
    score_history = []
    patch_lineage = []
    previous_score: float | None = None
    best_score = -1.0
    best_summary_id = ""
    all_trials = []

    for index, summary in enumerate(summaries, start=1):
        score = float(summary.overall_score)
        delta = 0.0 if previous_score is None else round(score - previous_score, 4)
        score_history.append(
            {
                "iteration": index,
                "summary_id": summary.summary_id,
                "score": score,
                "delta_from_previous": delta,
                "passed": summary.passed,
                "failed": summary.failed,
                "timeout": summary.timeout,
                "error": summary.error,
            }
        )
        patch_lineage.append(
            {
                "iteration": index,
                "summary_id": summary.summary_id,
                "patches_applied": list(summary.patches_applied),
                "score_after_patch": score,
                "delta_from_previous": delta,
            }
        )
        if score > best_score:
            best_score = score
            best_summary_id = summary.summary_id
        previous_score = score
        all_trials.extend(_load_summary_trials(memory, summary))

    return {
        "campaign_id": campaign_id,
        "iterations_requested_audit": iteration_limit,
        "iterations_completed": len(summaries),
        "loop_limit_contract": _loop_limit_contract(
            iteration_limit=iteration_limit,
            trials_config=trials_config,
            update_policy=update_policy,
            requested_iterations_stop=bool(
                iteration_limit is not None and iteration_limit > 0
            ),
        ),
        "stopped_reason": stopped_reason,
        "codex_update": codex_update,
        "round_task_concurrency": round_task_concurrency,
        "tasks": tasks,
        "score_history": score_history,
        "best": {
            "score": 0.0 if best_score < 0 else best_score,
            "summary_id": best_summary_id,
        },
        "summaries": [summary.model_dump(mode="json") for summary in summaries],
        "task_results": (
            [_trial_report(trial) for trial in all_trials]
            if include_task_results
            else []
        ),
        "task_results_deep_analysis_skipped": not include_task_results,
        "domain_metrics": _aggregate_by(all_trials, "task_domain"),
        "difficulty_metrics": _aggregate_by(all_trials, "task_difficulty"),
        "task_type_metrics": _aggregate_task_types(all_trials),
        "efficiency_metrics": _aggregate_efficiency(all_trials),
        "patch_lineage": patch_lineage,
        "regression": regression_plan,
        "submit": [result.__dict__ for result in submit_results],
        "reproducibility": {
            "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
            "git_dirty": bool(_git_output(["git", "status", "--short"])),
            "memory_path": str(memory_path),
            "trials_config": "config/trials.yaml",
            "models_config_priority": ["config/local.yaml", "config/models.yaml"],
        },
    }


def _loop_limit_contract(
    *,
    iteration_limit: int | None,
    args: argparse.Namespace | None = None,
    trials_config: dict[str, Any] | None = None,
    codex_config: dict[str, Any] | None = None,
    goal_plan: dict[str, Any] | None = None,
    update_policy: dict[str, Any] | None = None,
    requested_iterations_stop: bool | None = None,
) -> dict[str, Any]:
    from hl.loop_limits import (
        base_loop_limit_contract,
        disallowed_limit_terminal_reasons,
    )

    goal = goal_plan.get("goal") if isinstance(goal_plan, dict) else None
    if not isinstance(goal, dict):
        goal = {}
    if not isinstance(update_policy, dict):
        update_policy = {}
    tasks_config = _tasks_config(trials_config or {})
    requested_patience = getattr(args, "patience", None) if args is not None else None
    requested_iterations_audit = (
        int(iteration_limit) if iteration_limit is not None else None
    )
    if requested_iterations_stop is None:
        explicit_cli_iterations = (
            getattr(args, "iterations", None) if args is not None else None
        )
        requested_iterations_stop = bool(
            explicit_cli_iterations is not None and explicit_cli_iterations > 0
        )
    requested_iterations_stop_condition = bool(
        requested_iterations_stop
        and requested_iterations_audit is not None
        and requested_iterations_audit > 0
    )
    selection_config = _task_selection_config(trials_config or {})
    requested_random_count = (
        getattr(args, "random_count", None) if args is not None else None
    )
    random_count_audit = (
        requested_random_count
        if requested_random_count is not None
        else selection_config.get("random_count", tasks_config.get("random_count"))
    )
    codex_timeout = (
        codex_config.get("timeout_seconds") if isinstance(codex_config, dict) else None
    )
    contract = base_loop_limit_contract(
        notes={
            "all_loops": (
                "Master, Codex update sub-agent, diagnostic/context sub-agent, "
                "validation/regression sub-agent, mission-debug sub-agent, and "
                "Worker task loops must not stop because time, round, iteration, "
                "token, budget, timeout, cooldown, K/attempt, max_turns, depth, "
                "feature-count, target-task-count, validation-contract-count, or "
                "context-window values were reached. Those values are audit, "
                "scheduling, packet-size, recovery, or single-operation controls only."
            ),
            "master_loop": (
                "A positive requested-iterations value is an explicit campaign "
                "completion target evaluated against persisted round summaries, "
                "including summaries restored by --resume. It does not bound a "
                "Worker, Codex update, diagnostic, context, or validation loop. "
                "Task-selection controls, random-count, batch-size, legacy "
                "run-task-cap audit values, rate-limit concurrency restore waits, "
                "and infrastructure retry counts are audit, ordering, recovery, "
                "or throughput metadata. They must not stop, skip, or truncate "
                "master, sub-agent, or Worker loops. Task-pool epoch rollover is also progress "
                "bookkeeping only, not completion. They must not stop master, "
                "sub-agent, or Worker loops."
            ),
            "codex_update_sub_agent": (
                "Codex update timeout, interval, cooldown, minimum-failure, "
                "provider failure, token, and wall-time values are audit or "
                "evidence-strength metadata only. This does not stop the Codex "
                "update sub-agent, master loop, diagnostic/context sub-agents, "
                "or Worker loop."
            ),
            "diagnostic_sub_agents": (
                "Partial-pass K, diagnostic attempt indexes, mission feature "
                "counts, wall-clock references, and token budgets size evidence "
                "packets or label reruns only. They must not stop diagnostic "
                "sub-agent execution, the master loop, Codex update sub-agent, "
                "context sub-agents, or Worker loop."
            ),
            "context_sub_agents": (
                "Sub-agent depth, returned-summary token counts, and context "
                "window thresholds are isolation/compaction hints only. They "
                "must not stop master, sub-agent, or Worker loops."
            ),
            "validation_regression_sub_agents": (
                "Regression lanes, snapshot counts, explicit --cap values, retry "
                "counts, transient cooldowns, validation timeout references, "
                "project-test command duration, and task-concurrency values are "
                "audit, selection, retry, or throughput metadata. They must not "
                "stop validation/regression sub-agent, master, Codex update, "
                "diagnostic/context, mission-debug, or Worker loops. A failed "
                "validation gate may reject or roll back a patch, but not because "
                "a time, round, attempt, snapshot-count, or timeout value was reached."
            ),
            "mission_debug_sub_agent": (
                "Mission-debug max-features, candidate count, target-task count, "
                "validation-contract count, elapsed time, round count, and attempt "
                "count are audit or packet-size metadata only. They must not stop "
                "mission-debug, master, Codex update, diagnostic/context, or Worker loops."
            ),
            "worker_task_loop": (
                "Per-tool and regression timeouts bound individual operations; "
                "network-preflight timeouts are diagnostics recorded before the "
                "campaign continues; Harbor host wait references are audit-only; "
                "retry thresholds, subtask caps, checkpoint caps, empty-response "
                "recovery thresholds, compaction thresholds, tool-output truncation, "
                "and reasoning budgets are recovery/audit signals. None of these "
                "are master, sub-agent, or Worker loop stop conditions."
            ),
        }
    )
    contract["master_loop"].update(
        {
            "requested_iterations_audit_only": requested_iterations_audit,
            "requested_iterations_audit": requested_iterations_audit,
            "requested_iterations_stop_condition": (
                requested_iterations_stop_condition
            ),
            "requested_iterations_completion_condition": (
                requested_iterations_stop_condition
            ),
            "plateau_patience_audit_only": requested_patience,
            "infra_retry_unbounded_by_attempt_count": True,
            "max_tasks_audit_only": (
                getattr(args, "max_tasks", None) if args is not None else None
            ),
            "max_tasks_per_trial_audit_only": tasks_config.get(
                "max_tasks_per_trial"
            ),
            "max_tasks_controls_task_pool": False,
            "max_tasks_controls_batch_size": False,
            "max_tasks_per_trial_controls_task_pool": False,
            "max_tasks_per_trial_controls_batch_size": False,
            "run_task_cap_audit_only": (
                getattr(args, "run_task_cap", None) if args is not None else None
            ),
            "run_task_cap_controls_task_pool": False,
            "run_task_cap_controls_batch_size": False,
            "random_count_audit_only": random_count_audit,
            "random_count_controls_task_pool": False,
            "random_count_controls_batch_size": True,
            "batch_size_controls_task_pool": False,
            "batch_size_controls_batch_size": True,
            "task_pool_exhausted_is_loop_limit": False,
            "task_pool_exhausted_meaning": (
                "selected task set history has been evaluated at least once; "
                "fixed mode starts another evaluation epoch and rotation mode "
                "moves to another cycle instead of treating this as a time, "
                "round, token, patience, budget, or master-loop stop condition"
            ),
            "allowed_terminal_reasons": [
                "explicit user marks campaign goal complete/stopped",
                *(
                    ["explicit requested campaign rounds are fully persisted"]
                    if requested_iterations_stop_condition
                    else []
                ),
                "explicit one-shot submit terminal action after submit gate handling",
                "hard validation/regression process error unrelated to loop-limit fields",
                "explicit user/process cancellation",
            ],
            "disallowed_limit_terminal_reasons": [
                *disallowed_limit_terminal_reasons(),
            ],
        }
    )
    if requested_iterations_stop_condition:
        contract["master_loop"]["requested_iterations"] = requested_iterations_audit
    contract["codex_update_sub_agent"].update(
        {
            "interval_audit_only": update_policy.get("interval"),
            "cooldown_after_rollback_audit_only": update_policy.get(
                "cooldown_after_rollback"
            ),
            "min_failures_audit_only": update_policy.get("min_failures"),
            "timeout_seconds_audit_only": codex_timeout,
            "timeout_seconds_reference_audit_only": codex_timeout,
            "timeout_seconds": codex_timeout,
            "timeout_seconds_legacy_compatibility_only": codex_timeout,
        }
    )
    contract["diagnostic_sub_agents"].update(
        {
            "partial_pass_diagnostic_k_audit_only": update_policy.get(
                "partial_pass_diagnostic_k"
            ),
            "diagnostic_target_k_audit_only": update_policy.get(
                "partial_pass_diagnostic_k"
            ),
        }
    )
    contract["mission_debug_sub_agent"].update(
        {
            "max_features_audit_only": (
                getattr(args, "mission_debug_max_features", None)
                if args is not None
                else None
            ),
        }
    )
    contract["worker_task_loop"].update(
        {
            "max_turns_audit_only": (
                getattr(args, "max_turns_audit", None) if args is not None else None
            ),
            "max_turns_audit": (
                getattr(args, "max_turns_audit", None) if args is not None else None
            ),
            "llm_timeout_seconds_audit_only": (
                getattr(args, "llm_timeout_seconds", None) if args is not None else None
            ),
            "llm_timeout_seconds_reference_audit_only": (
                getattr(args, "llm_timeout_seconds", None) if args is not None else None
            ),
            "llm_timeout_seconds": (
                getattr(args, "llm_timeout_seconds", None) if args is not None else None
            ),
            "llm_timeout_seconds_legacy_compatibility_only": (
                getattr(args, "llm_timeout_seconds", None) if args is not None else None
            ),
            "tool_timeout_seconds_audit_only": (
                getattr(args, "tool_timeout_seconds", None) if args is not None else None
            ),
            "tool_timeout_seconds": (
                getattr(args, "tool_timeout_seconds", None) if args is not None else None
            ),
            "tool_timeout_seconds_legacy_compatibility_only": (
                getattr(args, "tool_timeout_seconds", None) if args is not None else None
            ),
        }
    )
    contract["goal_budgets"].update(
        {
            "token_budget": goal.get("token_budget"),
            "token_budget_audit_only": goal.get("token_budget"),
            "token_budget_scope": goal.get("token_budget_scope"),
            "wall_time_budget_seconds": goal.get("wall_time_budget_seconds"),
            "wall_time_budget_seconds_audit_only": goal.get("wall_time_budget_seconds"),
        }
    )
    return contract


def _build_campaign_report_from_state(
    *,
    campaign_id: str,
    tasks: list[str],
    iteration_limit: int | None,
    campaign_state: dict[str, Any],
    summaries: list[Any],
    memory: Any,
    memory_path: Path,
    regression_plan: dict[str, Any],
    submit_results: list[Any],
    codex_update: bool,
    trials_config: dict[str, Any] | None = None,
    update_policy: dict[str, Any] | None = None,
    round_task_concurrency: int = 1,
    task_rotation: dict[str, Any] | None = None,
    stopped_reason: str,
    checkpoint: bool,
) -> dict[str, Any]:
    from bench.scoring import Scoring

    trials = _load_campaign_state_trials(memory, campaign_state)
    summaries_for_report = list(summaries)
    if trials:
        aggregate = Scoring.build_summary(
            f"{_safe_campaign_id(campaign_id)}_aggregate",
            trials,
        )
        aggregate.patches_applied = _campaign_state_patches(campaign_state, summaries)
        summaries_for_report = [aggregate]

    report = _build_campaign_report(
        campaign_id=campaign_id,
        tasks=tasks,
        iteration_limit=iteration_limit,
        summaries=summaries_for_report,
        memory=memory,
        memory_path=memory_path,
        regression_plan=regression_plan,
        submit_results=submit_results,
        codex_update=codex_update,
        trials_config=trials_config,
        update_policy=update_policy,
        round_task_concurrency=round_task_concurrency,
        stopped_reason=stopped_reason,
        include_task_results=not checkpoint,
    )
    state_summaries = _campaign_state_summaries(campaign_state)
    if state_summaries:
        report["iterations_completed"] = len(state_summaries)
        report["score_history"] = _score_history_from_state_summaries(state_summaries)
        report["patch_lineage"] = _patch_lineage_from_state_summaries(state_summaries)
    codex_update_events = _normalized_codex_update_events(campaign_state)
    analysis_reports = _normalized_state_analysis_reports(campaign_state)
    last_accepted_codex_update = campaign_state.get("last_accepted_codex_update")
    if isinstance(last_accepted_codex_update, dict):
        last_accepted_codex_update = copy.deepcopy(last_accepted_codex_update)
    else:
        last_accepted_codex_update = None
    checkpoint_tasks_completed, checkpoint_tasks_pending = _checkpoint_task_counts(
        tasks=tasks,
        campaign_state=campaign_state,
        rotation_plan=task_rotation,
    )
    report["checkpoint"] = checkpoint
    report["tasks_completed"] = checkpoint_tasks_completed
    report["tasks_pending"] = checkpoint_tasks_pending
    report["tasks_pending_list"] = _checkpoint_pending_tasks(
        tasks=tasks,
        campaign_state=campaign_state,
        rotation_plan=task_rotation,
    )
    if checkpoint:
        report["task_results"] = _campaign_state_checkpoint_task_results(
            campaign_state
        )
        report["task_results_deep_analysis_skipped"] = True
    report["campaign_state_path"] = str(_campaign_state_path(memory_path, campaign_id))
    report["campaign_state"] = {
        "campaign_id": campaign_state.get("campaign_id"),
        "tasks": campaign_state.get("tasks") or list(tasks),
        "created_at": campaign_state.get("created_at", ""),
        "updated_at": campaign_state.get("updated_at", ""),
        "completed": campaign_state.get("completed", []),
        "summaries": campaign_state.get("summaries", []),
        "last_accepted_codex_update": last_accepted_codex_update,
        "codex_validation_failures": campaign_state.get(
            "codex_validation_failures",
            [],
        ),
        "codex_update_events": codex_update_events,
        "regression_gate_events": campaign_state.get("regression_gate_events", []),
        "change_evaluations": campaign_state.get("change_evaluations", []),
        "partial_pass_diagnostics": campaign_state.get("partial_pass_diagnostics", []),
        "task_epoch_rollovers": campaign_state.get("task_epoch_rollovers", []),
        "analysis_reports": analysis_reports,
        "failure_class_attempts": campaign_state.get("failure_class_attempts", []),
        "frontier_regression_events": campaign_state.get(
            "frontier_regression_events",
            [],
        ),
        "concurrency_events": campaign_state.get("concurrency_events", []),
        "provider_fail_fast_events": campaign_state.get(
            "provider_fail_fast_events",
            [],
        ),
        "round_task_concurrency": campaign_state.get("round_task_concurrency", {}),
        "docker_concurrency_budget": campaign_state.get("docker_concurrency_budget", {}),
        "same_model_frontier": campaign_state.get("same_model_frontier", {}),
        "task_rotation": campaign_state.get("task_rotation", {}),
    }
    report["codex_validation_failures"] = campaign_state.get(
        "codex_validation_failures",
        [],
    )
    report["last_accepted_codex_update"] = last_accepted_codex_update
    report["regression_gate_events"] = campaign_state.get(
        "regression_gate_events",
        [],
    )
    report["codex_update_events"] = codex_update_events
    report["change_evaluations"] = campaign_state.get("change_evaluations", [])
    report["partial_pass_diagnostics"] = campaign_state.get(
        "partial_pass_diagnostics",
        [],
    )
    report["analysis_reports"] = analysis_reports
    _attach_campaign_analysis_digest(report, analysis_reports)
    _attach_campaign_analysis_digest(report["campaign_state"], analysis_reports)
    report["failure_class_attempts"] = campaign_state.get(
        "failure_class_attempts",
        [],
    )
    report["frontier_regression_events"] = campaign_state.get(
        "frontier_regression_events",
        [],
    )
    report["concurrency_events"] = campaign_state.get("concurrency_events", [])
    report["provider_fail_fast_events"] = campaign_state.get(
        "provider_fail_fast_events",
        [],
    )
    report["round_task_concurrency_state"] = campaign_state.get(
        "round_task_concurrency",
        {},
    )
    report["docker_concurrency_budget"] = campaign_state.get(
        "docker_concurrency_budget",
        {},
    )
    report["same_model_frontier"] = campaign_state.get("same_model_frontier", {})
    if summaries:
        report["iteration_summaries"] = [
            summary.model_dump(mode="json") for summary in summaries
        ]
    return report


def _normalized_codex_update_events(
    campaign_state: dict[str, Any],
) -> list[dict[str, Any]]:
    events = campaign_state.get("codex_update_events")
    if not isinstance(events, list):
        return []
    return [
        normalize_legacy_limit_driven_skip_event(event)
        for event in events
        if isinstance(event, dict)
    ]


def _normalized_state_analysis_reports(
    campaign_state: dict[str, Any],
) -> list[dict[str, Any]]:
    reports = campaign_state.get("analysis_reports")
    if not isinstance(reports, list):
        return []
    return [
        _normalized_state_analysis_report(report)
        for report in reports
        if isinstance(report, dict)
    ]


def _attach_campaign_analysis_digest(
    report: dict[str, Any],
    analysis_reports: list[dict[str, Any]],
) -> None:
    digest = _campaign_analysis_digest_from_reports(analysis_reports)
    if not digest:
        return
    report["failure_buckets"] = digest["failure_buckets"]
    report["weakness_signatures"] = digest["weakness_signatures"]
    report["candidate_update_classes"] = digest["candidate_update_classes"]
    report["mechanism_update_classes"] = digest["mechanism_update_classes"]
    report["analysis_digest_source"] = digest["analysis_digest_source"]
    report["analysis_digest_summary_id"] = digest["analysis_digest_summary_id"]
    report["analysis_digest_stop_condition"] = False


def _campaign_analysis_digest_from_reports(
    analysis_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    for analysis_report in reversed(analysis_reports):
        if not isinstance(analysis_report, dict):
            continue
        failure_buckets = analysis_report.get("failure_buckets")
        weakness_signatures = analysis_report.get("weakness_signatures")
        candidate_update_classes = analysis_report.get("candidate_update_classes")
        mechanism_update_classes = analysis_report.get("mechanism_update_classes")
        if not (
            isinstance(failure_buckets, list)
            and failure_buckets
            and isinstance(candidate_update_classes, list)
            and candidate_update_classes
        ):
            continue
        if not isinstance(weakness_signatures, list):
            weakness_signatures = []
        if not isinstance(mechanism_update_classes, list):
            mechanism_update_classes = []
        source = str(
            analysis_report.get("summary_path")
            or _summary_path_for_state_analysis_report(analysis_report)
            or analysis_report.get("overview_path")
            or ""
        )
        return {
            "failure_buckets": failure_buckets,
            "weakness_signatures": weakness_signatures,
            "candidate_update_classes": candidate_update_classes,
            "mechanism_update_classes": mechanism_update_classes,
            "analysis_digest_source": source,
            "analysis_digest_summary_id": str(analysis_report.get("summary_id") or ""),
        }
    return {}


def _normalized_state_analysis_report(report: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(report)
    summary = _analysis_summary_for_state_report(report)
    if not summary:
        return normalized
    failure_buckets = summary.get("failure_buckets")
    if not isinstance(failure_buckets, list) or not failure_buckets:
        return normalized
    failure_buckets = _normalize_analysis_failure_buckets_from_timeout_phases(
        failure_buckets
    )
    trajectory_evidence = summary.get("trajectory_evidence")
    if not isinstance(trajectory_evidence, dict):
        trajectory_evidence = {}
    failure_buckets = _analysis_failure_buckets_with_evidence_mechanisms(
        failure_buckets,
        trajectory_evidence,
    )
    bucket_candidates = _candidate_update_classes(failure_buckets)
    normalized["failure_buckets"] = failure_buckets
    normalized.setdefault(
        "summary_path",
        str(_summary_path_for_state_analysis_report(report)),
    )

    raw_candidates = [
        str(item) for item in normalized.get("candidate_update_classes") or []
    ]
    if bucket_candidates:
        bucket_categories = {
            str(bucket.get("failure_category") or "")
            for bucket in failure_buckets
            if isinstance(bucket, dict)
        }
        candidate_categories = {
            _candidate_update_class_category(candidate)
            for candidate in raw_candidates
        }
        candidate_categories.discard("")
        if not (
            candidate_categories
            and candidate_categories.issubset(bucket_categories)
        ):
            if raw_candidates != bucket_candidates:
                normalized["raw_candidate_update_classes"] = raw_candidates
                normalized["candidate_update_classes_normalized_from"] = (
                    "summary_json_failure_buckets"
                )
            normalized["candidate_update_classes"] = bucket_candidates

    mechanism_update_classes = summary.get("mechanism_update_classes")
    if isinstance(mechanism_update_classes, list):
        normalized["mechanism_update_classes"] = [
            str(item) for item in mechanism_update_classes if str(item).strip()
        ][:8]

    raw_weakness_signatures = normalized.get("weakness_signatures")
    if not isinstance(raw_weakness_signatures, list):
        raw_weakness_signatures = summary.get("weakness_signatures")
    existing_signatures = _state_analysis_weakness_signature_entries(
        raw_weakness_signatures
    )
    synthesized_signatures = _analysis_weakness_signatures(
        [],
        failure_buckets=failure_buckets,
        trajectory_evidence=trajectory_evidence,
    )
    weakness_signatures = _normalized_weakness_signatures_from_failure_buckets(
        existing=existing_signatures,
        synthesized=synthesized_signatures,
        failure_buckets=failure_buckets,
    )
    if weakness_signatures:
        if existing_signatures and weakness_signatures != existing_signatures:
            normalized["raw_weakness_signatures"] = existing_signatures
            normalized["weakness_signatures_normalized_from"] = (
                "summary_json_failure_buckets"
            )
        normalized["weakness_signatures"] = weakness_signatures
    return normalized


def _state_analysis_weakness_signature_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    entries: list[dict[str, Any]] = []
    for raw_entry in value[:12]:
        if not isinstance(raw_entry, dict):
            continue
        signature = _tail_text(str(raw_entry.get("signature") or ""), 600)
        category = _tail_text(str(raw_entry.get("failure_category") or ""), 160)
        if not signature and not category:
            continue
        entry = {
            "signature": signature,
            "verifier_failure": _tail_text(
                str(raw_entry.get("verifier_failure") or ""),
                240,
            ),
            "agent_contribution": _tail_text(
                str(raw_entry.get("agent_contribution") or ""),
                240,
            ),
            "reusable_mechanism": _tail_text(
                str(raw_entry.get("reusable_mechanism") or ""),
                240,
            ),
            "failure_category": category,
            "count": int(raw_entry.get("count") or 0),
            "task_ids": [str(item) for item in raw_entry.get("task_ids") or []][:12],
            "affected_components": [
                str(item) for item in raw_entry.get("affected_components") or []
            ][:12],
            "timeout_phases": [
                str(item) for item in raw_entry.get("timeout_phases") or []
            ][:8],
            "failure_mechanisms": [
                str(item) for item in raw_entry.get("failure_mechanisms") or []
            ][:12],
            "evidence_sources": [
                str(item) for item in raw_entry.get("evidence_sources") or []
            ][:12],
            "loop_stop_condition": bool(raw_entry.get("loop_stop_condition", False)),
            "time_round_token_limit_driven": bool(
                raw_entry.get("time_round_token_limit_driven", False)
            ),
        }
        if raw_entry.get("purpose"):
            entry["purpose"] = _tail_text(str(raw_entry.get("purpose") or ""), 300)
        if raw_entry.get("synthesized_from_legacy_analysis"):
            entry["synthesized_from_legacy_analysis"] = True
        entries.append(entry)
    entries.sort(
        key=lambda item: (
            -int(item.get("count") or 0),
            str(item.get("signature") or ""),
        )
    )
    return entries


def _normalized_weakness_signatures_from_failure_buckets(
    *,
    existing: list[dict[str, Any]],
    synthesized: list[dict[str, Any]],
    failure_buckets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not existing:
        return synthesized
    if not synthesized:
        return existing

    bucket_categories = {
        str(bucket.get("failure_category") or "")
        for bucket in failure_buckets
        if str(bucket.get("failure_category") or "")
    }
    raw_category_map: dict[str, set[str]] = {}
    for bucket in failure_buckets:
        category = str(bucket.get("failure_category") or "")
        if not category:
            continue
        for raw_category in bucket.get("raw_failure_categories") or []:
            raw_text = str(raw_category)
            if raw_text:
                raw_category_map.setdefault(raw_text, set()).add(category)

    kept: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for entry in existing:
        if _weakness_signature_needs_bucket_resynthesis(
            entry,
            bucket_categories=bucket_categories,
            raw_bucket_categories=set(raw_category_map),
        ):
            stale.append(entry)
        else:
            kept.append(entry)
    if not stale:
        return existing

    stale_categories = {
        str(entry.get("failure_category") or "")
        for entry in stale
        if str(entry.get("failure_category") or "")
    }
    replacement_categories = set(stale_categories)
    for category in stale_categories:
        replacement_categories.update(raw_category_map.get(category, set()))
    stale_task_ids = {
        str(task_id)
        for entry in stale
        for task_id in entry.get("task_ids") or []
        if str(task_id)
    }
    stale_timeout_phases = {
        str(phase)
        for entry in stale
        for phase in entry.get("timeout_phases") or []
        if str(phase)
    }
    kept_signatures = {str(entry.get("signature") or "") for entry in kept}
    replacements = [
        entry
        for entry in synthesized
        if str(entry.get("signature") or "") not in kept_signatures
        and _synthesized_weakness_matches_stale_entry(
            entry,
            replacement_categories=replacement_categories,
            stale_task_ids=stale_task_ids,
            stale_timeout_phases=stale_timeout_phases,
        )
    ]
    if not replacements and not kept:
        replacements = synthesized
    normalized = [*kept, *replacements]
    normalized.sort(
        key=lambda item: (
            -int(item.get("count") or 0),
            str(item.get("signature") or ""),
        )
    )
    return normalized


def _weakness_signature_needs_bucket_resynthesis(
    entry: dict[str, Any],
    *,
    bucket_categories: set[str],
    raw_bucket_categories: set[str],
) -> bool:
    contribution = str(entry.get("agent_contribution") or "")
    signature = str(entry.get("signature") or "")
    if contribution == "agent_behavior:unclassified":
        return True
    if "agent=agent_behavior:unclassified" in signature:
        return True
    category = str(entry.get("failure_category") or "")
    return bool(
        category
        and category not in bucket_categories
        and category in raw_bucket_categories
    )


def _synthesized_weakness_matches_stale_entry(
    entry: dict[str, Any],
    *,
    replacement_categories: set[str],
    stale_task_ids: set[str],
    stale_timeout_phases: set[str],
) -> bool:
    category = str(entry.get("failure_category") or "")
    if category and category in replacement_categories:
        return True
    task_ids = {str(item) for item in entry.get("task_ids") or [] if str(item)}
    if stale_task_ids and task_ids & stale_task_ids:
        return True
    timeout_phases = {
        str(item) for item in entry.get("timeout_phases") or [] if str(item)
    }
    return bool(stale_timeout_phases and timeout_phases & stale_timeout_phases)


def _normalize_analysis_failure_buckets_from_timeout_phases(
    failure_buckets: list[Any],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, bool], dict[str, Any]] = {}
    order: list[tuple[str, bool]] = []
    for raw_bucket in failure_buckets:
        if not isinstance(raw_bucket, dict):
            continue
        bucket = _normalize_analysis_failure_bucket_from_timeout_phase(raw_bucket)
        key = (
            str(bucket.get("failure_category") or ""),
            bool(bucket.get("infrastructure")),
        )
        if key not in merged:
            merged[key] = bucket
            order.append(key)
            continue
        target = merged[key]
        target["count"] = int(target.get("count") or 0) + int(
            bucket.get("count") or 0
        )
        target["infrastructure"] = bool(
            target.get("infrastructure") or bucket.get("infrastructure")
        )
        for field in (
            "task_ids",
            "affected_components",
            "timeout_phases",
            "failure_mechanisms",
            "raw_failure_categories",
            "normalized_from_timeout_phases",
        ):
            _merge_analysis_bucket_list_field(target, bucket, field)
    return [merged[key] for key in order]


def _normalize_analysis_failure_bucket_from_timeout_phase(
    raw_bucket: dict[str, Any],
) -> dict[str, Any]:
    bucket = {
        "failure_category": str(raw_bucket.get("failure_category") or ""),
        "count": int(raw_bucket.get("count") or 0),
        "infrastructure": bool(raw_bucket.get("infrastructure")),
        "task_ids": [str(item) for item in raw_bucket.get("task_ids") or []],
        "affected_components": [
            str(item) for item in raw_bucket.get("affected_components") or []
        ],
        "timeout_phases": [str(item) for item in raw_bucket.get("timeout_phases") or []],
    }
    failure_mechanisms = [
        str(item) for item in raw_bucket.get("failure_mechanisms") or []
    ]
    if failure_mechanisms:
        bucket["failure_mechanisms"] = failure_mechanisms
        bucket["failure_mechanism_count_stop_condition"] = False
    for timeout_phase in bucket["timeout_phases"]:
        entry = _ANALYSIS_INFRASTRUCTURE_PHASES.get(timeout_phase)
        if entry is None:
            continue
        category, components = entry
        original_category = str(bucket.get("failure_category") or "")
        original_components = [
            str(component) for component in bucket.get("affected_components") or []
        ]
        original_infrastructure = bool(bucket.get("infrastructure"))
        canonical_components = list(components)
        changed = (
            original_category != category
            or not original_infrastructure
            or original_components != canonical_components
        )
        bucket["failure_category"] = category
        bucket["infrastructure"] = True
        bucket["affected_components"] = canonical_components
        if original_category and original_category != category:
            bucket["raw_failure_categories"] = [original_category]
        if changed:
            bucket["normalized_from_timeout_phases"] = [timeout_phase]
        break
    return bucket


def _merge_analysis_bucket_list_field(
    target: dict[str, Any],
    source: dict[str, Any],
    field: str,
) -> None:
    values = [str(item) for item in target.get(field) or []]
    seen = set(values)
    for item in source.get(field) or []:
        text = str(item)
        if text and text not in seen:
            values.append(text)
            seen.add(text)
    if values:
        target[field] = values


def _analysis_summary_for_state_report(report: dict[str, Any]) -> dict[str, Any]:
    summary_path = _summary_path_for_state_analysis_report(report)
    if not summary_path.exists() or not summary_path.is_file():
        return {}
    try:
        data = json.loads(summary_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _summary_path_for_state_analysis_report(report: dict[str, Any]) -> Path:
    raw_summary_path = str(report.get("summary_path") or "").strip()
    if raw_summary_path:
        return Path(raw_summary_path)
    raw_overview_path = str(report.get("overview_path") or "").strip()
    if raw_overview_path:
        return Path(raw_overview_path).with_name("summary.json")
    return Path("")


def _candidate_update_class_category(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    left = text.split("->", 1)[0].strip()
    if left.startswith("infrastructure "):
        left = left[len("infrastructure ") :].strip()
    return left


def _campaign_state_summaries(campaign_state: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for entry in campaign_state.get("summaries") or []:
        if isinstance(entry, dict) and entry.get("summary_id"):
            summaries.append(entry)
    return summaries


def _score_history_from_state_summaries(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    history = []
    previous_score: float | None = None
    for index, summary in enumerate(summaries, start=1):
        score = round(float(summary.get("overall_score") or 0.0), 4)
        delta = 0.0 if previous_score is None else round(score - previous_score, 4)
        history.append(
            {
                "iteration": index,
                "summary_id": str(summary.get("summary_id")),
                "score": score,
                "delta_from_previous": delta,
                "patches_applied": list(summary.get("patches_applied") or []),
            }
        )
        previous_score = score
    return history


def _patch_lineage_from_state_summaries(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "iteration": index,
            "summary_id": str(summary.get("summary_id")),
            "patches_applied": list(summary.get("patches_applied") or []),
            "score_after_patch": round(float(summary.get("overall_score") or 0.0), 4),
            "delta_from_previous": (
                0.0
                if index == 1
                else round(
                    float(summary.get("overall_score") or 0.0)
                    - float(summaries[index - 2].get("overall_score") or 0.0),
                    4,
                )
            ),
        }
        for index, summary in enumerate(summaries, start=1)
    ]


def _load_campaign_state_trials(memory: Any, campaign_state: dict[str, Any]) -> list[Any]:
    trials = []
    for entry in campaign_state.get("completed") or []:
        trial_id = str(entry.get("trial_id") or "")
        if not trial_id:
            continue
        try:
            trials.append(memory.get_trial(trial_id))
        except FileNotFoundError:
            continue
    return trials


def _campaign_state_checkpoint_task_results(
    campaign_state: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for entry in campaign_state.get("completed") or []:
        if not isinstance(entry, dict):
            continue
        trial_id = str(entry.get("trial_id") or "")
        task_id = str(entry.get("task_id") or "")
        if not trial_id or not task_id:
            continue
        results.append(
            {
                "trial_id": trial_id,
                "task_id": task_id,
                "status": str(entry.get("status") or ""),
                "score": float(entry.get("score") or 0.0),
                "verified": bool(entry.get("verified")),
                "wall_time_seconds": round(
                    float(entry.get("wall_time_seconds") or 0.0), 3
                ),
                "summary_id": str(entry.get("summary_id") or ""),
                "iteration": int(entry.get("iteration") or 0),
                "diagnostic": bool(entry.get("diagnostic")),
                "checkpoint_lightweight": True,
            }
        )
    return results


def _campaign_state_patches(
    campaign_state: dict[str, Any],
    summaries: list[Any],
) -> list[str]:
    patches: list[str] = []
    for entry in campaign_state.get("summaries") or []:
        for patch in entry.get("patches_applied") or []:
            if patch not in patches:
                patches.append(str(patch))
    for summary in summaries:
        for patch in summary.patches_applied:
            if patch not in patches:
                patches.append(str(patch))
    return patches


def _load_summary_trials(memory: Any, summary: Any) -> list[Any]:
    trials = []
    for trial_id in summary.trial_ids:
        try:
            trials.append(memory.get_trial(trial_id))
        except FileNotFoundError:
            continue
    return trials


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _trial_report(trial: Any) -> dict[str, Any]:
    from hl.attribution import FailureAttributor

    task_metadata = trial.metadata.get("task_metadata") or {}
    metadata = trial.metadata if isinstance(trial.metadata, dict) else {}
    try:
        attribution = FailureAttributor().analyze(trial)
    except Exception:
        attribution = SimpleNamespace(
            failure_category=str(getattr(getattr(trial, "status", None), "value", "") or "unknown"),
            affected_components=[],
            component_confidence={},
            evidence=[],
        )
    raw_attribution_category = str(getattr(attribution, "failure_category", "") or "")
    base_failure_category = _trial_report_base_failure_category(
        trial,
        raw_attribution_category,
    )
    enhanced_failure_category = _analysis_effective_failure_category(
        trial,
        attribution_category=base_failure_category
        or str(getattr(getattr(trial, "status", None), "value", "") or "unknown"),
    )
    infra_phase_components = _analysis_infrastructure_phase_components(trial)
    category_infra_components = _analysis_infrastructure_category_components(
        enhanced_failure_category
    )
    if infra_phase_components:
        affected_components = sorted(infra_phase_components)
    elif category_infra_components:
        affected_components = sorted(category_infra_components)
    else:
        failure_mechanism_components, replace_base_components = (
            _analysis_failure_mechanism_components(
                trial,
                failure_category=enhanced_failure_category,
            )
        )
        attribution_components = [
            str(component)
            for component in getattr(attribution, "affected_components", []) or []
            if str(component)
        ]
        if failure_mechanism_components and replace_base_components:
            affected_components = sorted(failure_mechanism_components)
        elif failure_mechanism_components:
            affected_components = sorted(
                {
                    str(component)
                    for component in [
                        *attribution_components,
                        *failure_mechanism_components,
                    ]
                    if str(component)
                }
            )
        else:
            affected_components = _dedupe_preserve_order(attribution_components)
    failure_mechanisms = [
        mechanism.name for mechanism in failure_mechanisms_for_trial(trial)
    ]
    category_mechanism = dependency_loop_mechanism_for_failure_category(
        enhanced_failure_category
    )
    if category_mechanism and category_mechanism not in failure_mechanisms:
        failure_mechanisms.insert(0, category_mechanism)
    report = {
        "trial_id": trial.trial_id,
        "task_id": trial.task_id,
        "domain": _value_key(trial.task_domain),
        "difficulty": _value_key(trial.task_difficulty),
        "task_type": task_metadata.get("task_type", ""),
        "tags": task_metadata.get("tags", []),
        "status": trial.status.value,
        "score": trial.score,
        "verified": trial.verified,
        "model": trial.model_used,
        "wall_time_seconds": round(float(trial.wall_time_seconds), 3),
        "token_usage": trial.token_usage,
        "trial_metrics": trial.metadata.get("trial_metrics") or {},
        "harbor_job_dir": trial.harbor_job_dir,
    }
    report.update(
        {
            "failure_category": enhanced_failure_category,
            "base_failure_category": base_failure_category,
            "analysis_failure_category": enhanced_failure_category,
            "failure_mechanisms": failure_mechanisms,
            "affected_components": affected_components,
            "component_confidence": dict(
                getattr(attribution, "component_confidence", {}) or {}
            ),
            "timeout_phase": str(metadata.get("timeout_phase") or ""),
            "infra_error_detected": bool(metadata.get("infra_error_detected")),
            "verifier_infra_error": bool(metadata.get("verifier_infra_error")),
            "score_exclusion_reason": str(metadata.get("score_exclusion_reason") or ""),
            "environment_start_attribution_hint": str(
                metadata.get("environment_start_attribution_hint") or ""
            ),
            "docker_image_validation_failed": bool(
                metadata.get("docker_image_validation_failed")
            ),
            "prebuilt_image_cache_miss_detected": bool(
                metadata.get("prebuilt_image_cache_miss_detected")
            ),
            "prebuilt_image_cache_warmup_commands": [
                str(command)
                for command in metadata.get("prebuilt_image_cache_warmup_commands")
                or []
            ][:3],
            "network_preflight_recommended": bool(
                metadata.get("network_preflight_recommended")
            ),
            "heavy_dockerfile_install_detected": bool(
                metadata.get("heavy_dockerfile_install_detected")
            ),
        }
    )
    return report


def _trial_report_base_failure_category(
    trial: Any,
    attribution_category: str,
) -> str:
    timeout_phase = str(getattr(trial, "metadata", {}).get("timeout_phase") or "")
    status = getattr(getattr(trial, "status", None), "value", "")
    if timeout_phase == "agent_execution" and status == "timeout":
        return "agent_execution_timeout"
    return attribution_category


def _aggregate_task_types(trials: list[Any]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for trial in trials:
        metadata = trial.metadata.get("task_metadata") or {}
        task_type = str(metadata.get("task_type") or "unknown")
        bucket = buckets.setdefault(
            task_type,
            {
                "tasks": 0,
                "passed": 0,
                "score_sum": 0.0,
                "wall_time_seconds": 0.0,
                "token_usage": {"input": 0, "cache": 0, "output": 0},
                "trial_metrics": _empty_metric_totals(),
            },
        )
        bucket["tasks"] += 1
        bucket["passed"] += 1 if trial.status.value == "passed" else 0
        bucket["score_sum"] += float(trial.score)
        bucket["wall_time_seconds"] += float(trial.wall_time_seconds)
        for token_key in bucket["token_usage"]:
            bucket["token_usage"][token_key] += int(trial.token_usage.get(token_key, 0))
        _add_trial_metrics(bucket["trial_metrics"], trial)
    return {
        task_type: {
            "tasks": value["tasks"],
            "passed": value["passed"],
            "score": round(value["score_sum"] / value["tasks"], 4),
            "wall_time_seconds": round(value["wall_time_seconds"], 3),
            "token_usage": value["token_usage"],
            "trial_metrics": _finalize_metric_totals(value["trial_metrics"], value["tasks"]),
        }
        for task_type, value in sorted(buckets.items())
    }


def _aggregate_by(trials: list[Any], attr: str) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for trial in trials:
        key = _value_key(getattr(trial, attr))
        bucket = buckets.setdefault(
            key,
            {
                "tasks": 0,
                "passed": 0,
                "score_sum": 0.0,
                "wall_time_seconds": 0.0,
                "token_usage": {"input": 0, "cache": 0, "output": 0},
                "trial_metrics": _empty_metric_totals(),
            },
        )
        bucket["tasks"] += 1
        bucket["passed"] += 1 if trial.status.value == "passed" else 0
        bucket["score_sum"] += float(trial.score)
        bucket["wall_time_seconds"] += float(trial.wall_time_seconds)
        for token_key in bucket["token_usage"]:
            bucket["token_usage"][token_key] += int(trial.token_usage.get(token_key, 0))
        _add_trial_metrics(bucket["trial_metrics"], trial)

    return {
        key: {
            "tasks": value["tasks"],
            "passed": value["passed"],
            "score": round(value["score_sum"] / value["tasks"], 4),
            "wall_time_seconds": round(value["wall_time_seconds"], 3),
            "token_usage": value["token_usage"],
            "trial_metrics": _finalize_metric_totals(value["trial_metrics"], value["tasks"]),
        }
        for key, value in sorted(buckets.items())
    }


def _aggregate_efficiency(trials: list[Any]) -> dict[str, Any]:
    totals = _empty_metric_totals()
    token_usage = {"input": 0, "cache": 0, "output": 0}
    wall_time = 0.0
    for trial in trials:
        wall_time += float(getattr(trial, "wall_time_seconds", 0.0) or 0.0)
        for key in token_usage:
            token_usage[key] += int(getattr(trial, "token_usage", {}).get(key, 0))
        _add_trial_metrics(totals, trial)
    return {
        "tasks": len(trials),
        "wall_time_seconds": round(wall_time, 3),
        "token_usage": token_usage,
        "trial_metrics": _finalize_metric_totals(totals, len(trials)),
    }


def _empty_metric_totals() -> dict[str, float]:
    return {
        "cost_usd": 0.0,
        "n_turns": 0.0,
        "n_api_calls": 0.0,
        "api_error_count": 0.0,
        "provider_latency_ms": 0.0,
        "cache_hit_ratio": 0.0,
    }


def _add_trial_metrics(totals: dict[str, float], trial: Any) -> None:
    metrics = getattr(trial, "metadata", {}).get("trial_metrics") or {}
    if not isinstance(metrics, dict):
        return
    for key in totals:
        try:
            totals[key] += float(metrics.get(key) or 0.0)
        except (TypeError, ValueError):
            continue


def _finalize_metric_totals(totals: dict[str, float], count: int) -> dict[str, float]:
    count = max(1, int(count or 0))
    return {
        "cost_usd": round(totals["cost_usd"], 6),
        "n_turns": int(totals["n_turns"]),
        "n_api_calls": int(totals["n_api_calls"]),
        "api_error_count": int(totals["api_error_count"]),
        "provider_latency_ms_mean": round(totals["provider_latency_ms"] / count, 3),
        "cache_hit_ratio_mean": round(totals["cache_hit_ratio"] / count, 4),
    }


def _write_campaign_report(
    memory_path: Path,
    campaign_id: str,
    report: dict[str, Any],
) -> Path:
    summaries_dir = memory_path / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    path = summaries_dir / f"{_safe_campaign_id(campaign_id)}_campaign.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def _safe_campaign_id(campaign_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in campaign_id)


def _value_key(value: Any) -> str:
    return str(getattr(value, "value", value))


def _git_output(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


if __name__ == "__main__":
    raise SystemExit(main())
