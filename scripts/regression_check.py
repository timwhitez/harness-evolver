#!/usr/bin/env python3
"""Run solved-task regression checks through Harbor."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_trial import (  # noqa: E402
    add_docker_resource_args,
    validate_docker_concurrency_budget,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check solved-task regressions")
    parser.add_argument("--task", help="Check a specific task only")
    parser.add_argument("--memory-path", default="trials")
    parser.add_argument("--path", default="terminal-bench-tasks/terminal-bench")
    parser.add_argument("--jobs-dir", default="jobs")
    parser.add_argument("--lane", choices=["smoke", "standard", "full"], default="smoke")
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help=(
            "Audit/reference field for older regression configs. Every matching "
            "same-model solved-task snapshot is checked; explicit caps do not "
            "impose count limits, and lane names do not impose count limits."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=["stable-order", "adaptive"],
        default="stable-order",
        help=(
            "Regression snapshot selection strategy. adaptive is a deterministic "
            "audit label only; cooldown, failure counts, and runtime history do "
            "not skip or deprioritize snapshots."
        ),
    )
    parser.add_argument(
        "--task-concurrency",
        type=int,
        default=1,
        help="Host-side Harbor task concurrency within this regression gate.",
    )
    parser.add_argument("--project-test-command", default="pytest tests/ -v")
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
        help="Audit/progress reference only; not a regression Worker loop limit.",
    )
    parser.add_argument(
        "--max-turns",
        dest="max_turns_audit",
        type=int,
        default=None,
        help="Deprecated alias for --max-turns-audit.",
    )
    parser.add_argument(
        "--n-attempts",
        type=int,
        default=None,
        help=(
            "Harbor attempts per task. Accepted for run_campaign parity and "
            "forwarded to regression Harbor jobs."
        ),
    )
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--trials-config", default="config/trials.yaml")
    parser.add_argument("--env-file", default=None)
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "Audit-only host Harbor wait reference in seconds; defaults to "
            "execution.timeout_per_task and does not stop regression Harbor runs"
        ),
    )
    parser.add_argument(
        "--skip-network-preflight",
        action="store_true",
        help="Accepted for campaign command parity; regression runs do not preflight per task",
    )
    parser.add_argument("--network-preflight-mode", choices=["quick", "full"], default=None)
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
    parser.add_argument(
        "--mounts-json",
        default=None,
        help="JSON array of Docker Compose service volume mounts for Harbor",
    )
    parser.add_argument(
        "--verifier-env",
        action="append",
        default=None,
        help="Verifier environment variable in KEY=VALUE format; repeatable",
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--no-network-hardened-environment",
        dest="network_hardened_environment",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--record-from-trial",
        default=None,
        help="Create/update a regression snapshot from a verified passed trial id",
    )
    parser.add_argument(
        "--snapshot-status",
        choices=["stable", "pending", "invalidated", "all"],
        default="stable",
        help=(
            "Which regression snapshots to execute. Campaign pre/post checks "
            "default to stable; Codex update validation uses pending before "
            "promoting a newly solved task to stable."
        ),
    )
    parser.add_argument(
        "--holdout-mode",
        choices=["all", "held_in", "held_out"],
        default="all",
        help=(
            "Held-in/held-out split for Self-Harness acceptance. 'held_in' runs "
            "only snapshots shown to the Codex proposer; 'held_out' runs only the "
            "hidden split to detect collateral regressions; 'all' runs both."
        ),
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Target held-out share in [0,1]; 0 disables the held-out split.",
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=0,
        help="Deterministic seed for the held-in/held-out partition.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from bench.harbor import HarborRunner
    from hl.memory import FileSystemMemory
    from hl.model_scope import model_scope_from_agent_config
    from hl.types import RegressionSnapshot
    from scripts.run_trial import (
        _apply_execution_defaults,
        _require_worker_api_key,
        resolve_agent_config,
    )

    memory_path = Path(args.memory_path)
    memory = FileSystemMemory(base_path=str(memory_path))
    _apply_execution_defaults(args, parser)
    docker_concurrency_budget = validate_docker_concurrency_budget(
        concurrency=args.task_concurrency,
        args=args,
        parser=parser,
        source="--task-concurrency",
    )

    if args.record_from_trial:
        snapshot = record_snapshot_from_trial(memory, args.record_from_trial)
        print(
            json.dumps(
                {
                    "recorded": True,
                    "task_id": snapshot.task_id,
                    "harness_version": snapshot.harness_version,
                    "required_assertions": snapshot.required_assertions,
                },
                indent=2,
            )
        )
        return 0

    agent_config = resolve_agent_config(args, parser)
    model_scope = model_scope_from_agent_config(agent_config)
    all_snapshots = memory.list_regression_snapshots(model_scope=model_scope)
    if args.task:
        all_snapshots = [
            snapshot for snapshot in all_snapshots if snapshot.task_id == args.task
        ]
    snapshots = [
        snapshot
        for snapshot in all_snapshots
        if args.snapshot_status == "all"
        or snapshot.validation_status == args.snapshot_status
    ]
    if args.holdout_mode != "all" and args.holdout_fraction > 0:
        from hl.regression_split import is_holdout_task

        want_holdout = args.holdout_mode == "held_out"
        snapshots = [
            snapshot
            for snapshot in snapshots
            if is_holdout_task(
                snapshot.task_id,
                fraction=args.holdout_fraction,
                seed=args.holdout_seed,
            )
            == want_holdout
        ]
    skipped_snapshots = [
        snapshot for snapshot in all_snapshots if snapshot not in snapshots
    ]
    snapshots = _select_regression_snapshots(snapshots, args)

    if not snapshots:
        if args.dry_run and skipped_snapshots:
            print(
                json.dumps(
                    {
                        "lane": args.lane,
                        "snapshot_status": args.snapshot_status,
                        "model_scope": model_scope,
                        "cap_audit_only": args.cap,
                        "regression_selection_cap_stop_condition": False,
                        "regression_snapshot_count_stop_condition": False,
                        "task_concurrency": args.task_concurrency,
                        "docker_concurrency_budget": docker_concurrency_budget,
                        "commands": [],
                        "skipped_snapshots": [
                            {
                                "task_id": snapshot.task_id,
                                "validation_status": snapshot.validation_status,
                                "invalidation_reason": snapshot.invalidation_reason,
                                "model_scope": snapshot.model_scope,
                            }
                            for snapshot in skipped_snapshots
                        ],
                        "project_test_command": (
                            args.project_test_command if args.lane == "full" else None
                        ),
                    },
                    indent=2,
                )
            )
            return 0
        print("No regression snapshots found.")
        return 0

    runner = HarborRunner(dataset_path=args.path, jobs_dir=Path(args.jobs_dir))
    run_stamp = int(time.time())
    commands = [
        runner.build_command(
            snapshot.task_id,
            agent_config,
            job_name=_regression_job_name(snapshot.task_id, run_stamp),
            jobs_dir=args.jobs_dir,
        )
        for snapshot in snapshots
    ]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "lane": args.lane,
                    "snapshot_status": args.snapshot_status,
                    "model_scope": model_scope,
                    "commands": [cmd.shell_command() for cmd in commands],
                    "selection_policy": args.selection_policy,
                    "cap_audit_only": args.cap,
                    "regression_selection_cap_stop_condition": False,
                    "regression_snapshot_count_stop_condition": False,
                    "task_concurrency": args.task_concurrency,
                    "docker_concurrency_budget": docker_concurrency_budget,
                    "selected_snapshots": [
                        {
                            "task_id": snapshot.task_id,
                            "regression_runs": snapshot.regression_runs,
                            "regression_failures": snapshot.regression_failures,
                            "regression_transient_failures": (
                                snapshot.regression_transient_failures
                            ),
                            "last_regression_wall_time_seconds": (
                                snapshot.last_regression_wall_time_seconds
                            ),
                            "regression_cooldown_until": (
                                snapshot.regression_cooldown_until.isoformat()
                                if snapshot.regression_cooldown_until
                                else None
                            ),
                        }
                        for snapshot in snapshots
                    ],
                    "skipped_snapshots": [
                        {
                            "task_id": snapshot.task_id,
                            "validation_status": snapshot.validation_status,
                            "invalidation_reason": snapshot.invalidation_reason,
                            "model_scope": snapshot.model_scope,
                        }
                        for snapshot in skipped_snapshots
                    ],
                    "project_test_command": (
                        args.project_test_command if args.lane == "full" else None
                    ),
                },
                indent=2,
            )
        )
        return 0

    _require_worker_api_key(agent_config, parser)

    failed: list[str] = []
    for snapshot, result in _run_regression_snapshots(
        snapshots=snapshots,
        runner=runner,
        agent_config=agent_config,
        timeout_audit=args.timeout,
        jobs_dir=args.jobs_dir,
        run_stamp=run_stamp,
        task_concurrency=args.task_concurrency,
    ):
        memory.record_trial(result)
        if hasattr(memory, "record_regression_run"):
            memory.record_regression_run(
                snapshot.task_id,
                result,
                model_scope=model_scope,
            )
        if memory.check_regression(snapshot.task_id, result):
            failed.append(snapshot.task_id)

    if failed:
        print("Regressions detected:")
        for task_id in failed:
            print(f"- {task_id}")
        return 1

    if args.lane == "full":
        completed = subprocess.run(shlex.split(args.project_test_command))
        if completed.returncode != 0:
            print(f"Project test command failed: {args.project_test_command}")
            return completed.returncode

    print(f"Regression lane {args.lane} passed for {len(snapshots)} snapshot(s).")
    return 0


def record_snapshot_from_trial(
    memory: "FileSystemMemory",
    trial_id: str,
) -> "RegressionSnapshot":
    from hl.types import RegressionSnapshot, TrialStatus
    from hl.model_scope import model_config_from_trial, model_scope_from_trial

    trial = memory.get_trial(trial_id)
    if not trial.verified or trial.status != TrialStatus.PASSED or trial.score < 1.0:
        raise SystemExit(
            "Refusing to create regression snapshot from non-passing or unverified "
            f"trial {trial_id!r}: status={trial.status.value}, "
            f"score={trial.score}, verified={trial.verified}"
        )

    snapshot = RegressionSnapshot(
        task_id=trial.task_id,
        harness_version=trial.harness_version,
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
        ],
        source_trial_id=trial.trial_id,
        validation_status="stable",
    )
    memory.save_regression(trial.task_id, snapshot)
    return snapshot


def _select_regression_snapshots(
    snapshots: list["RegressionSnapshot"],
    args: argparse.Namespace,
) -> list["RegressionSnapshot"]:
    if args.selection_policy == "adaptive":
        snapshots = sorted(snapshots, key=_adaptive_snapshot_rank)
    _ = args.cap
    return snapshots


def _adaptive_snapshot_rank(snapshot: "RegressionSnapshot") -> tuple[Any, ...]:
    return (
        snapshot.task_id,
    )


def _run_regression_snapshots(
    *,
    snapshots: list["RegressionSnapshot"],
    runner: Any,
    agent_config: dict[str, object],
    timeout_audit: int | None,
    jobs_dir: str,
    run_stamp: int,
    task_concurrency: int,
) -> list[tuple["RegressionSnapshot", Any]]:
    if task_concurrency <= 0:
        raise SystemExit("--task-concurrency must be positive")
    if task_concurrency <= 1 or len(snapshots) <= 1:
        return [
            (
                snapshot,
                runner.run_task(
                    task_id=snapshot.task_id,
                    agent_config=agent_config,
                    timeout_audit=timeout_audit,
                    job_name=_regression_job_name(snapshot.task_id, run_stamp),
                    jobs_dir=jobs_dir,
                ),
            )
            for snapshot in snapshots
        ]

    results: list[tuple["RegressionSnapshot", Any] | None] = [None] * len(snapshots)
    max_workers = min(task_concurrency, len(snapshots))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                runner.run_task,
                task_id=snapshot.task_id,
                agent_config=agent_config,
                timeout_audit=timeout_audit,
                job_name=_regression_job_name(snapshot.task_id, run_stamp),
                jobs_dir=jobs_dir,
            ): (index, snapshot)
            for index, snapshot in enumerate(snapshots)
        }
        for future in as_completed(futures):
            index, snapshot = futures[future]
            results[index] = (snapshot, future.result())
    return [item for item in results if item is not None]


def _regression_job_name(task_id: str, run_stamp: int) -> str:
    safe_task = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task_id)
    return f"regression_{run_stamp}_{safe_task[:48]}"


if __name__ == "__main__":
    raise SystemExit(main())
