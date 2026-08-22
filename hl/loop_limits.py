"""Shared loop-limit contract fields.

The values here are reporting and audit metadata. They must not be used as
runtime predicates for stopping master, sub-agent, mission-debug, context, or
Worker loops, including time, round, attempt, timeout, cap, count, token, or
budget fields.
"""

from __future__ import annotations

from typing import Literal, Mapping


LoopLimitScope = Literal[
    "all_loops",
    "master_loop",
    "codex_update_sub_agent",
    "diagnostic_sub_agents",
    "context_sub_agents",
    "validation_regression_sub_agents",
    "mission_debug_sub_agent",
    "worker_task_loop",
    "goal_budgets",
]


LOOP_LIMIT_SCOPES: tuple[LoopLimitScope, ...] = (
    "all_loops",
    "master_loop",
    "codex_update_sub_agent",
    "diagnostic_sub_agents",
    "context_sub_agents",
    "validation_regression_sub_agents",
    "mission_debug_sub_agent",
    "worker_task_loop",
    "goal_budgets",
)


LOOP_OWNER_SCOPES: tuple[LoopLimitScope, ...] = (
    "master_loop",
    "codex_update_sub_agent",
    "diagnostic_sub_agents",
    "context_sub_agents",
    "validation_regression_sub_agents",
    "mission_debug_sub_agent",
    "worker_task_loop",
)


LOOP_OWNER_NO_LIMIT_FLAGS: dict[str, bool] = {
    "loop_has_no_time_limit": True,
    "loop_has_no_round_limit": True,
    "loop_has_no_iteration_limit": True,
    "loop_has_no_time_or_round_limit": True,
    "loop_has_no_time_round_or_attempt_limit": True,
    "loop_has_no_timeout_limit": True,
    "loop_has_no_turn_limit": True,
    "loop_has_no_max_turns_limit": True,
    "loop_has_no_time_round_turn_or_attempt_limit": True,
    "loop_has_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limit": True,
    "loop_has_no_time_round_attempt_timeout_cap_count_token_or_budget_limit": True,
    "loop_owner_has_no_time_limit": True,
    "loop_owner_has_no_round_limit": True,
    "loop_owner_has_no_iteration_limit": True,
    "loop_owner_has_no_time_or_round_limit": True,
    "loop_owner_has_no_attempt_limit": True,
    "loop_owner_has_no_time_round_or_attempt_limit": True,
    "loop_owner_has_no_wall_clock_limit": True,
    "loop_owner_has_no_timeout_limit": True,
    "loop_owner_has_no_turn_limit": True,
    "loop_owner_has_no_max_turns_limit": True,
    "loop_owner_has_no_time_round_turn_or_attempt_limit": True,
    "loop_owner_has_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limit": True,
    "loop_owner_has_no_cap_limit": True,
    "loop_owner_has_no_count_limit": True,
    "loop_owner_has_no_token_limit": True,
    "loop_owner_has_no_budget_limit": True,
    "loop_owner_has_no_time_round_attempt_timeout_cap_count_token_or_budget_limit": True,
    "loop_owner_has_no_limit_driven_completion": True,
    "loop_owner_has_no_limit_driven_stop": True,
    "loop_owner_has_no_limit_driven_skip": True,
    "loop_owner_has_no_limit_driven_truncation": True,
    "loop_owner_has_no_limit_driven_kill": True,
    "loop_owner_has_no_limit_driven_cancellation": True,
    "loop_owner_has_no_retry_threshold_stop": True,
    "loop_owner_has_no_direct_replay_threshold_stop": True,
    "loop_owner_has_no_retry_threshold_retry_denial": True,
    "loop_owner_has_no_direct_replay_threshold_retry_denial": True,
    "time_limit_allowed": False,
    "round_limit_allowed": False,
    "iteration_limit_allowed": False,
    "wall_clock_limit_allowed": False,
    "time_or_round_limit_allowed": False,
    "attempt_limit_allowed": False,
    "time_round_or_attempt_limit_allowed": False,
    "timeout_limit_allowed": False,
    "cap_limit_allowed": False,
    "count_limit_allowed": False,
    "token_limit_allowed": False,
    "budget_limit_allowed": False,
}


LOOP_OWNER_POLICY_FORBIDDEN_TERMINAL_PREDICATES: tuple[str, ...] = (
    "time",
    "elapsed",
    "wall_clock",
    "wall_time",
    "round",
    "iteration",
    "turn",
    "attempt",
    "retry",
    "timeout",
    "operation_timeout",
    "validation_timeout",
    "cap",
    "count",
    "token",
    "budget",
    "max_turns",
    "deadline",
    "cooldown",
    "interval",
    "min_failures",
    "k",
    "diagnostic_k",
    "depth",
    "context_depth",
    "context_window",
    "feature_count",
    "candidate_count",
    "target_task_count",
    "validation_contract_count",
    "snapshot_count",
)


LOOP_OWNER_POLICY_FORBIDDEN_LIMIT_DRIVEN_ACTIONS: tuple[str, ...] = (
    "complete",
    "stop",
    "skip",
    "truncate",
    "kill",
    "cancel",
)


LOOP_OWNER_POLICY_AUDIT_ONLY_FIELDS: tuple[str, ...] = (
    "max_iterations",
    "iterations",
    "iteration_limit",
    "rounds",
    "max_rounds",
    "round_limit",
    "patience",
    "elapsed_seconds",
    "duration_seconds",
    "wall_time_seconds",
    "deadline",
    "wall_clock_deadline",
    "timeout_seconds",
    "operation_timeout_seconds",
    "validation_timeout_seconds",
    "attempt_count",
    "max_attempts",
    "n_attempts",
    "retry_count",
    "max_retries",
    "max_turns",
    "max_turns_audit",
    "interval",
    "codex_update_interval",
    "min_failures",
    "cooldown_after_rollback",
    "cooldown_remaining",
    "diagnostic_target_k",
    "partial_pass_diagnostic_k",
    "diagnostic_attempt_index",
    "diagnostic_attempt_count",
    "max_features",
    "feature_count",
    "candidate_count",
    "target_task_count",
    "validation_contract_count",
    "max_tasks",
    "run_task_cap",
    "random_count",
    "rotation_pool_count",
    "snapshot_count",
    "regression_snapshot_count",
    "regression_cap",
    "context_depth",
    "max_sub_agent_depth",
    "sub_agent_summary_max_tokens",
    "token_budget",
    "wall_time_budget_seconds",
)


SUB_AGENT_CREATION_POLICY: dict[str, object] = {
    "allowed_creator_owner": "master_loop",
    "master_loop_may_create_sub_agents": True,
    "only_master_loop_may_create_sub_agents": True,
    "sub_agent_owners_may_create_sub_agents": False,
    "worker_task_loop_may_create_sub_agents": False,
    "nested_sub_agent_creation_allowed": False,
    "sub_agents_may_spawn_nested_sub_agents": False,
    "max_sub_agent_nesting_depth": 1,
    "max_sub_agent_creation_depth": 1,
    "creation_depth_policy": "master_loop -> sub_agent only",
    "codex_update_sub_agent_may_create_sub_agents": False,
    "diagnostic_sub_agents_may_create_sub_agents": False,
    "context_sub_agents_may_create_sub_agents": False,
    "validation_regression_sub_agents_may_create_sub_agents": False,
    "mission_debug_sub_agent_may_create_sub_agents": False,
    "sub_agent_creation_permission_guard": True,
    "sub_agent_creation_loop_stop_condition": False,
    "nested_sub_agent_creation_stop_condition": False,
    "blocked_nested_agent_commands": [
        "aider",
        "amp",
        "codex",
        "claude",
        "cursor-agent",
        "droid",
        "forgecode",
        "factory",
        "factory droid",
        "external coding agent CLI",
        "gemini",
        "openai-codex",
        "opencode",
    ],
    "note": (
        "Only the master HL orchestrator may create sub-agents. A running "
        "sub-agent, including the Codex update sub-agent, diagnostic/context "
        "sub-agent, validation/regression sub-agent, or mission-debug "
        "sub-agent, must not start another agent or delegate to another "
        "agent CLI. Rejecting such a command is an operation policy guard, "
        "not a loop-limit stop condition."
    ),
}


def sub_agent_creation_policy() -> dict[str, object]:
    policy = dict(SUB_AGENT_CREATION_POLICY)
    policy["owner_creation_permissions"] = {
        "master_loop": True,
        "codex_update_sub_agent": False,
        "diagnostic_sub_agents": False,
        "context_sub_agents": False,
        "validation_regression_sub_agents": False,
        "mission_debug_sub_agent": False,
        "worker_task_loop": False,
    }
    return policy


UNBOUNDED_LOOP_FLAGS: dict[str, bool] = {
    "all_loops_have_no_time_limits": True,
    "all_loops_have_no_round_limits": True,
    "all_agent_loops_have_no_time_or_round_limits": True,
    "all_sub_agent_loops_have_no_time_or_round_limits": True,
    "all_sub_agent_loops_have_no_time_limits": True,
    "all_sub_agent_loops_have_no_round_limits": True,
    "all_sub_agent_loops_unbounded_by_time": True,
    "all_sub_agent_loops_unbounded_by_round": True,
    "all_sub_agent_loops_unbounded_by_time_and_round": True,
    "all_sub_agent_loops_unbounded_by_time_round_and_attempt": True,
    "all_sub_agent_loops_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    "all_sub_agent_loop_owners_have_no_time_limits": True,
    "all_sub_agent_loop_owners_have_no_round_limits": True,
    "all_sub_agent_loop_owners_have_no_time_or_round_limits": True,
    "all_sub_agent_loop_owners_have_no_attempt_limits": True,
    "all_sub_agent_loop_owners_have_no_time_round_or_attempt_limits": True,
    "all_sub_agent_loop_owners_have_no_timeout_limits": True,
    "all_sub_agent_loop_owners_have_no_cap_limits": True,
    "all_sub_agent_loop_owners_have_no_count_limits": True,
    "all_sub_agent_loop_owners_have_no_token_limits": True,
    "all_sub_agent_loop_owners_have_no_budget_limits": True,
    "all_sub_agent_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits": True,
    "master_sub_agent_and_worker_loops_have_no_time_or_round_limits": True,
    "master_sub_agent_and_worker_loops_have_no_time_limits": True,
    "master_sub_agent_and_worker_loops_have_no_round_limits": True,
    "master_sub_agent_and_worker_loops_have_no_attempt_limits": True,
    "master_sub_agent_and_worker_loops_have_no_time_round_or_attempt_limits": True,
    "master_and_all_sub_agent_and_worker_loops_have_no_time_limits": True,
    "master_and_all_sub_agent_and_worker_loops_have_no_round_limits": True,
    "master_and_all_sub_agent_and_worker_loops_have_no_time_or_round_limits": True,
    "all_agent_loops_unbounded_by_time_and_round": True,
    "all_agent_loops_unbounded_by_time": True,
    "all_agent_loops_unbounded_by_round": True,
    "all_agent_loops_unbounded_by_time_round_turn_attempt_and_budget": True,
    "all_agent_loops_unbounded_by_time_round_turn_attempt_cap_and_budget": True,
    "all_agent_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_and_budget": True,
    "all_agent_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget": True,
    "master_and_sub_agent_loops_have_no_time_limits": True,
    "master_and_sub_agent_loops_have_no_round_limits": True,
    "master_and_sub_agent_loops_have_no_time_or_round_limits": True,
    "master_and_sub_agent_loops_unbounded_by_time": True,
    "master_and_sub_agent_loops_unbounded_by_round": True,
    "master_and_sub_agent_loops_unbounded_by_time_and_round": True,
    "master_and_sub_agent_loops_unbounded_by_time_round_and_attempt": True,
    "master_and_sub_agent_loops_unbounded_by_time_round_attempt_and_cap": True,
    "master_and_sub_agent_loops_unbounded_by_time_round_attempt_timeout_cap_count_and_budget": True,
    "master_and_sub_agent_loops_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    "master_and_all_sub_agent_loops_have_no_time_limits": True,
    "master_and_all_sub_agent_loops_have_no_round_limits": True,
    "master_and_all_sub_agent_loops_have_no_time_or_round_limits": True,
    "master_and_all_sub_agent_loops_unbounded_by_time": True,
    "master_and_all_sub_agent_loops_unbounded_by_round": True,
    "master_and_all_sub_agent_loops_unbounded_by_time_and_round": True,
    "master_and_all_sub_agent_loops_unbounded_by_time_round_and_attempt": True,
    "master_and_all_sub_agent_loops_unbounded_by_time_round_attempt_and_cap": True,
    "master_and_all_sub_agent_loops_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    "master_and_all_sub_agent_and_worker_loops_unbounded_by_time": True,
    "master_and_all_sub_agent_and_worker_loops_unbounded_by_round": True,
    "master_and_all_sub_agent_and_worker_loops_unbounded_by_time_and_round": True,
    "master_and_all_sub_agent_and_worker_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget": True,
    "master_sub_agent_and_worker_loops_unbounded_by_time": True,
    "master_sub_agent_and_worker_loops_unbounded_by_round": True,
    "master_sub_agent_and_worker_loops_unbounded_by_time_and_round": True,
    "master_sub_agent_and_worker_loops_unbounded_by_time_round_and_attempt": True,
    "master_sub_agent_and_worker_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget": True,
    "master_loop_unbounded_by_time_and_round": True,
    "master_loop_unbounded_by_time": True,
    "master_loop_unbounded_by_round": True,
    "master_loop_unbounded_by_time_round_and_attempt": True,
    "master_loop_unbounded_by_time_round_attempt_and_cap": True,
    "master_loop_unbounded_by_time_round_attempt_timeout_cap_count_and_budget": True,
    "master_loop_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    "codex_update_sub_agent_unbounded_by_time_and_round": True,
    "codex_update_sub_agent_unbounded_by_time": True,
    "codex_update_sub_agent_unbounded_by_round": True,
    "codex_update_sub_agent_unbounded_by_time_round_and_attempt": True,
    "codex_update_sub_agent_unbounded_by_time_round_attempt_and_cap": True,
    "codex_update_sub_agent_unbounded_by_time_round_attempt_timeout_cap_count_and_budget": True,
    "codex_update_sub_agent_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    "diagnostic_sub_agents_unbounded_by_time_and_round": True,
    "diagnostic_sub_agents_unbounded_by_time": True,
    "diagnostic_sub_agents_unbounded_by_round": True,
    "diagnostic_sub_agents_unbounded_by_time_round_and_attempt": True,
    "diagnostic_sub_agents_unbounded_by_time_round_attempt_and_cap": True,
    "diagnostic_sub_agents_unbounded_by_time_round_attempt_timeout_cap_count_and_budget": True,
    "diagnostic_sub_agents_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    "context_sub_agents_unbounded_by_time_and_round": True,
    "context_sub_agents_unbounded_by_time": True,
    "context_sub_agents_unbounded_by_round": True,
    "context_sub_agents_unbounded_by_time_round_and_attempt": True,
    "context_sub_agents_unbounded_by_time_round_attempt_and_cap": True,
    "context_sub_agents_unbounded_by_time_round_attempt_timeout_cap_count_and_budget": True,
    "context_sub_agents_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    "validation_regression_sub_agents_unbounded_by_time_and_round": True,
    "validation_regression_sub_agents_unbounded_by_time": True,
    "validation_regression_sub_agents_unbounded_by_round": True,
    "validation_regression_sub_agents_unbounded_by_time_round_and_attempt": True,
    "validation_regression_sub_agents_unbounded_by_time_round_attempt_and_cap": True,
    "validation_regression_sub_agents_unbounded_by_time_round_attempt_timeout_cap_count_and_budget": True,
    "validation_regression_sub_agents_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    "mission_debug_sub_agent_unbounded_by_time_and_round": True,
    "mission_debug_sub_agent_unbounded_by_time": True,
    "mission_debug_sub_agent_unbounded_by_round": True,
    "mission_debug_sub_agent_unbounded_by_time_round_and_attempt": True,
    "mission_debug_sub_agent_unbounded_by_time_round_attempt_and_cap": True,
    "mission_debug_sub_agent_unbounded_by_time_round_attempt_and_feature_count": True,
    "mission_debug_sub_agent_unbounded_by_time_round_attempt_timeout_cap_count_and_budget": True,
    "mission_debug_sub_agent_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    "worker_task_loop_unbounded_by_time_and_round": True,
    "worker_task_loop_unbounded_by_time": True,
    "worker_task_loop_unbounded_by_round": True,
    "worker_task_loop_unbounded_by_time_round_turn_and_attempt": True,
    "worker_task_loop_unbounded_by_time_round_turn_attempt_and_cap": True,
    "worker_task_loop_unbounded_by_time_round_turn_attempt_timeout_cap_count_and_budget": True,
    "worker_task_loop_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget": True,
    "all_loop_owners_have_no_time_limits": True,
    "all_loop_owners_have_no_round_limits": True,
    "all_loop_owners_have_no_time_or_round_limits": True,
    "all_loop_owners_have_no_attempt_limits": True,
    "all_loop_owners_have_no_time_round_or_attempt_limits": True,
    "all_loop_owners_have_no_timeout_limits": True,
    "all_loop_owners_have_no_turn_limits": True,
    "all_loop_owners_have_no_max_turns_limits": True,
    "all_loop_owners_have_no_time_round_turn_or_attempt_limits": True,
    "all_loop_owners_have_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limits": True,
    "all_loop_owners_have_no_cap_limits": True,
    "all_loop_owners_have_no_count_limits": True,
    "all_loop_owners_have_no_token_limits": True,
    "all_loop_owners_have_no_budget_limits": True,
    "all_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_time_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_round_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_time_or_round_limits": True,
    "master_and_every_sub_agent_loop_owner_has_no_time_limit": True,
    "master_and_every_sub_agent_loop_owner_has_no_round_limit": True,
    "master_and_every_sub_agent_loop_owner_has_no_time_or_round_limit": True,
    "master_and_all_sub_agent_loop_owners_have_no_timeout_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_turn_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_max_turns_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_time_round_turn_or_attempt_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_cap_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_count_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_token_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_budget_limits": True,
    "master_and_all_sub_agent_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_time_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_round_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_time_or_round_limits": True,
    "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_limits": True,
    "master_and_all_sub_agent_and_worker_loop_owners_have_no_round_limits": True,
    "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_or_round_limits": True,
    "master_every_sub_agent_and_worker_loop_owner_has_no_time_limit": True,
    "master_every_sub_agent_and_worker_loop_owner_has_no_round_limit": True,
    "master_every_sub_agent_and_worker_loop_owner_has_no_time_or_round_limit": True,
    "master_sub_agent_and_worker_loop_owners_have_no_timeout_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_turn_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_max_turns_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_time_round_turn_or_attempt_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_cap_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_count_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_token_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_budget_limits": True,
    "master_sub_agent_and_worker_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits": True,
    "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits": True,
}


SCOPE_UNBOUNDED_LOOP_FLAGS: dict[LoopLimitScope, dict[str, bool]] = {
    "all_loops": UNBOUNDED_LOOP_FLAGS,
    "master_loop": {
        "no_time_or_round_limits": True,
        "no_time_limit": True,
        "no_round_limit": True,
        "unbounded_by_time": True,
        "unbounded_by_round": True,
        "unbounded_by_time_and_round": True,
        "unbounded_by_time_round_and_attempt": True,
        "unbounded_by_time_round_attempt_and_cap": True,
        "unbounded_by_time_round_attempt_cap_count_and_budget": True,
        "unbounded_by_time_round_attempt_timeout_cap_count_and_budget": True,
        "unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    },
    "codex_update_sub_agent": {
        "no_time_or_round_limits": True,
        "no_time_limit": True,
        "no_round_limit": True,
        "unbounded_by_time": True,
        "unbounded_by_round": True,
        "unbounded_by_time_and_round": True,
        "unbounded_by_time_round_and_attempt": True,
        "unbounded_by_budget_attempt_and_cooldown": True,
        "unbounded_by_time_round_attempt_and_cap": True,
        "unbounded_by_time_round_attempt_timeout_cap_count_and_budget": True,
        "unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    },
    "diagnostic_sub_agents": {
        "no_time_or_round_limits": True,
        "no_time_limit": True,
        "no_round_limit": True,
        "unbounded_by_time": True,
        "unbounded_by_round": True,
        "unbounded_by_time_and_round": True,
        "unbounded_by_time_round_and_attempt": True,
        "unbounded_by_time_round_attempt_and_k": True,
        "unbounded_by_time_round_attempt_and_cap": True,
        "unbounded_by_time_round_attempt_timeout_cap_count_and_budget": True,
        "unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    },
    "context_sub_agents": {
        "no_time_or_round_limits": True,
        "no_time_limit": True,
        "no_round_limit": True,
        "unbounded_by_time": True,
        "unbounded_by_round": True,
        "unbounded_by_time_and_round": True,
        "unbounded_by_time_round_and_attempt": True,
        "unbounded_by_depth_and_tokens": True,
        "unbounded_by_time_round_attempt_depth_token_cap_count_and_budget": True,
        "unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    },
    "validation_regression_sub_agents": {
        "no_time_or_round_limits": True,
        "no_time_limit": True,
        "no_round_limit": True,
        "unbounded_by_time": True,
        "unbounded_by_round": True,
        "unbounded_by_time_and_round": True,
        "unbounded_by_time_round_and_attempt": True,
        "unbounded_by_snapshot_count_and_lane_cap": True,
        "unbounded_by_time_round_attempt_snapshot_count_lane_cap_timeout_and_budget": True,
        "unbounded_by_time_round_attempt_snapshot_count_lane_cap_timeout_token_and_budget": True,
        "unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    },
    "mission_debug_sub_agent": {
        "no_time_or_round_limits": True,
        "no_time_limit": True,
        "no_round_limit": True,
        "unbounded_by_time": True,
        "unbounded_by_round": True,
        "unbounded_by_time_and_round": True,
        "unbounded_by_time_round_and_attempt": True,
        "unbounded_by_time_round_attempt_and_cap": True,
        "unbounded_by_time_round_attempt_and_feature_count": True,
        "unbounded_by_time_round_attempt_feature_count_cap_and_budget": True,
        "unbounded_by_time_round_attempt_timeout_feature_count_cap_count_token_and_budget": True,
        "unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget": True,
    },
    "worker_task_loop": {
        "no_time_or_round_limits": True,
        "no_time_limit": True,
        "no_round_limit": True,
        "unbounded_by_time": True,
        "unbounded_by_round": True,
        "unbounded_by_time_and_round": True,
        "unbounded_by_time_round_turn_and_attempt": True,
        "unbounded_by_time_round_turn_attempt_and_cap": True,
        "unbounded_by_time_round_turn_attempt_timeout_cap_count_and_budget": True,
        "unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget": True,
    },
    "goal_budgets": {
        "no_time_or_round_limits": True,
        "no_time_limit": True,
        "no_round_limit": True,
        "unbounded_by_time": True,
        "unbounded_by_round": True,
        "unbounded_by_time_round_token_and_budget": True,
        "unbounded_by_time_round_timeout_cap_count_token_and_budget": True,
    },
}


COMMON_NON_TERMINAL_LIMIT_FLAGS: dict[str, bool] = {
    "time_and_round_limits_stop_condition": False,
    "time_round_limit_stop_condition": False,
    "time_or_round_limit_stop_condition": False,
    "agent_time_limit_stop_condition": False,
    "agent_round_limit_stop_condition": False,
    "master_time_round_limit_stop_condition": False,
    "master_time_limit_stop_condition": False,
    "master_round_limit_stop_condition": False,
    "sub_agent_time_round_limit_stop_condition": False,
    "sub_agent_time_limit_stop_condition": False,
    "sub_agent_round_limit_stop_condition": False,
    "worker_time_round_limit_stop_condition": False,
    "worker_time_limit_stop_condition": False,
    "worker_round_limit_stop_condition": False,
    "iteration_limit_stop_condition": False,
    "iteration_count_stop_condition": False,
    "round_stop_condition": False,
    "round_limit_stop_condition": False,
    "round_count_stop_condition": False,
    "time_stop_condition": False,
    "time_limit_stop_condition": False,
    "wall_clock_deadline_stop_condition": False,
    "deadline_stop_condition": False,
    "timeout_stop_condition": False,
    "timeout_limit_stop_condition": False,
    "timeout_count_stop_condition": False,
    "timeout_seconds_stop_condition": False,
    "operation_timeout_stop_condition": False,
    "process_timeout_stop_condition": False,
    "token_count_stop_condition": False,
    "token_limit_stop_condition": False,
    "token_budget_stop_condition": False,
    "budget_stop_condition": False,
    "budget_limit_stop_condition": False,
    "time_budget_stop_condition": False,
    "wall_time_budget_stop_condition": False,
    "budget_exhaustion_stop_condition": False,
    "cooldown_stop_condition": False,
    "attempt_count_stop_condition": False,
    "attempt_limit_stop_condition": False,
    "time_round_attempt_limit_stop_condition": False,
    "time_round_or_attempt_limit_stop_condition": False,
    "retry_count_stop_condition": False,
    "retry_limit_stop_condition": False,
    "cap_stop_condition": False,
    "cap_limit_stop_condition": False,
    "count_limit_stop_condition": False,
    "turn_count_stop_condition": False,
    "limit_driven_completion_stop_condition": False,
    "limit_driven_stop_condition": False,
    "limit_driven_skip_stop_condition": False,
    "limit_driven_truncation_stop_condition": False,
    "limit_driven_kill_stop_condition": False,
    "limit_driven_cancellation_stop_condition": False,
    "retry_threshold_loop_stop_condition": False,
    "direct_replay_threshold_stop_condition": False,
}


SCOPE_NON_TERMINAL_LIMIT_FLAGS: dict[LoopLimitScope, dict[str, bool]] = {
    "all_loops": {
        "max_turns_stop_condition": False,
        "context_depth_stop_condition": False,
        "context_token_stop_condition": False,
        "diagnostic_k_stop_condition": False,
        "mission_debug_max_features_stop_condition": False,
        "mission_feature_count_stop_condition": False,
        "mission_target_task_count_stop_condition": False,
        "mission_validation_contract_count_stop_condition": False,
        "feature_count_stop_condition": False,
        "candidate_count_stop_condition": False,
        "component_count_stop_condition": False,
        "affected_components_count_stop_condition": False,
        "target_task_count_stop_condition": False,
        "validation_contract_count_stop_condition": False,
        "rounds_requested_stop_condition": False,
        "max_rounds_stop_condition": False,
        "max_iterations_stop_condition": False,
        "max_attempts_stop_condition": False,
        "max_time_stop_condition": False,
        "network_preflight_failure_stop_condition": False,
        "network_preflight_timeout_stop_condition": False,
        "time_limit_allowed": False,
        "round_limit_allowed": False,
        "time_or_round_limit_allowed": False,
        "attempt_limit_allowed": False,
        "time_round_or_attempt_limit_allowed": False,
        "timeout_limit_allowed": False,
        "cap_limit_allowed": False,
        "count_limit_allowed": False,
        "token_limit_allowed": False,
        "budget_limit_allowed": False,
    },
    "master_loop": {
        "requested_iterations_default_stop_condition": False,
        "requested_iterations_stop_condition": False,
        "task_selection_cap_stop_condition": False,
        "run_task_cap_stop_condition": False,
        "max_tasks_stop_condition": False,
        "max_tasks_per_trial_stop_condition": False,
        "task_count_stop_condition": False,
        "per_round_task_slice_stop_condition": False,
        "random_count_stop_condition": False,
        "random_task_count_stop_condition": False,
        "task_pool_exhausted_stop_condition": False,
        "task_pool_epoch_rollover_stop_condition": False,
        "fixed_task_epoch_rollover_stop_condition": False,
        "rotation_cycle_stop_condition": False,
        "rotation_pool_cycle_stop_condition": False,
        "wall_time_stop_condition": False,
        "plateau_patience_stop_condition": False,
        "rate_limit_concurrency_backoff_stop_condition": False,
        "rate_limit_concurrency_restore_wait_stop_condition": False,
        "infra_retry_attempt_count_stop_condition": False,
        "infra_retries_stop_condition": False,
        "infra_retry_reference_stop_condition": False,
        "infra_retry_loop_stop_condition": False,
        "time_round_token_limit_stop_condition": False,
        "provider_fail_fast_stop_condition": False,
        "provider_billing_quota_stop_condition": False,
        "provider_auth_failure_stop_condition": False,
        "network_preflight_failure_stop_condition": False,
        "network_preflight_timeout_stop_condition": False,
        "stop_after_trial_hook_stop_condition": False,
        "trial_record_hook_stop_condition": False,
    },
    "codex_update_sub_agent": {
        "timeout_seconds_stop_condition": False,
        "host_validation_timeout_seconds_stop_condition": False,
        "validation_timeout_seconds_stop_condition": False,
        "validation_command_timeout_stop_condition": False,
        "round_stop_condition": False,
        "sub_agent_round_limit_stop_condition": False,
        "interval_stop_condition": False,
        "min_failures_stop_condition": False,
        "partial_pass_diagnostic_k_stop_condition": False,
        "diagnostic_attempt_count_stop_condition": False,
        "diagnostic_round_limit_stop_condition": False,
        "sub_agent_attempt_count_stop_condition": False,
        "sub_agent_attempt_limit_stop_condition": False,
        "update_attempt_history_truncation_stop_condition": False,
        "mission_debug_max_features_stop_condition": False,
        "provider_transient_failure_stop_condition": False,
        "provider_auth_failure_stop_condition": False,
        "provider_billing_quota_stop_condition": False,
    },
    "diagnostic_sub_agents": {
        "partial_pass_diagnostic_k_stop_condition": False,
        "diagnostic_target_k_stop_condition": False,
        "diagnostic_attempt_count_stop_condition": False,
        "diagnostic_attempt_index_stop_condition": False,
        "diagnostic_round_limit_stop_condition": False,
        "sub_agent_attempt_count_stop_condition": False,
        "sub_agent_attempt_limit_stop_condition": False,
        "sub_agent_round_limit_stop_condition": False,
        "mission_debug_max_features_stop_condition": False,
        "timeout_seconds_stop_condition": False,
        "network_preflight_failure_stop_condition": False,
        "network_preflight_timeout_stop_condition": False,
    },
    "context_sub_agents": {
        "depth_stop_condition": False,
        "summary_token_stop_condition": False,
        "context_token_stop_condition": False,
    },
    "validation_regression_sub_agents": {
        "regression_lane_cap_stop_condition": False,
        "regression_cap_stop_condition": False,
        "regression_lane_stop_condition": False,
        "snapshot_count_stop_condition": False,
        "regression_snapshot_count_stop_condition": False,
        "regression_selection_cap_stop_condition": False,
        "regression_retry_count_stop_condition": False,
        "regression_failed_task_retry_stop_condition": False,
        "regression_transient_cooldown_stop_condition": False,
        "validation_timeout_stop_condition": False,
        "project_test_timeout_stop_condition": False,
        "snapshot_status_count_stop_condition": False,
        "task_concurrency_stop_condition": False,
    },
    "mission_debug_sub_agent": {
        "max_features_stop_condition": False,
        "feature_count_stop_condition": False,
        "candidate_count_stop_condition": False,
        "target_task_count_stop_condition": False,
        "validation_contract_count_stop_condition": False,
        "timeout_seconds_stop_condition": False,
        "sub_agent_attempt_count_stop_condition": False,
        "sub_agent_attempt_limit_stop_condition": False,
        "sub_agent_round_limit_stop_condition": False,
    },
    "worker_task_loop": {
        "max_turns_stop_condition": False,
        "turn_limit_stop_condition": False,
        "turn_count_stop_condition": False,
        "llm_timeout_seconds_stop_condition": False,
        "model_request_timeout_stop_condition": False,
        "provider_request_timeout_stop_condition": False,
        "tool_timeout_seconds_stop_condition": False,
        "operation_timeout_stop_condition": False,
        "direct_retry_threshold_stop_condition": False,
        "subtask_plan_cap_stop_condition": False,
        "reasoning_budget_stop_condition": False,
        "checkpoint_cap_stop_condition": False,
        "checkpoint_count_stop_condition": False,
        "checkpoint_cooldown_stop_condition": False,
        "timeout_escalation_count_stop_condition": False,
        "timeout_phase_count_stop_condition": False,
        "empty_response_recovery_limit_stop_condition": False,
        "empty_response_recovery_threshold_stop_condition": False,
        "compaction_threshold_stop_condition": False,
        "tool_output_truncation_stop_condition": False,
    },
    "goal_budgets": {
        "time_round_token_budget_stop_condition": False,
    },
}


DISALLOWED_LIMIT_TERMINAL_REASONS: tuple[str, ...] = (
    "iteration count reached",
    "round count reached",
    "master round limit reached",
    "sub-agent round limit reached",
    "worker round limit reached",
    "wall-clock deadline reached",
    "master time limit reached",
    "sub-agent time limit reached",
    "worker time limit reached",
    "deadline reached",
    "timeout reached",
    "operation timeout reached",
    "token budget exhausted",
    "budget exhausted",
    "attempt count reached",
    "retry count reached",
    "cap reached",
    "count limit reached",
    "max_turns reached",
    "context depth/window reached",
    "rollback cooldown active",
    "diagnostic K reached",
)


LIMIT_TERMINAL_REASON_MARKERS: tuple[str, ...] = (
    "iteration",
    "round",
    "wall clock",
    "wall-clock",
    "wall_time",
    "wall time",
    "time budget",
    "time limit",
    "deadline",
    "timeout",
    "token",
    "budget",
    "attempt",
    "retry",
    "cap",
    "count",
    "max turns",
    "max_turns",
    "max iterations",
    "max_iterations",
    "max rounds",
    "max_rounds",
    "max features",
    "max_features",
    "feature count",
    "candidate count",
    "target task count",
    "validation contract count",
    "context depth",
    "context window",
    "context token",
    "cooldown",
    "diagnostic k",
    "diagnostic target",
    "snapshot count",
    "lane cap",
    "turn limit",
    "turn count",
    "loop limit",
)


DEFAULT_LOOP_LIMIT_NOTES: dict[LoopLimitScope, str] = {
    "all_loops": (
        "Master, Codex update sub-agent, diagnostic/context sub-agent, "
        "validation/regression sub-agent, mission-debug sub-agent, and "
        "Worker task loops have no time or round limits. Time, round, "
        "iteration, token, budget, timeout, cooldown, K/attempt, max_turns, "
        "depth, feature-count, target-task-count, validation-contract-count, "
        "and context-window values are audit, scheduling, packet-size, "
        "recovery, or single-operation controls only."
    ),
    "master_loop": (
        "Master-loop progress metadata must not become a time, round, "
        "attempt, timeout, cap, count, token, or budget stop condition."
    ),
    "codex_update_sub_agent": (
        "Codex update sub-agent metadata must not become a time, round, "
        "attempt, timeout, cap, count, token, budget, cooldown, or "
        "provider-state stop condition."
    ),
    "diagnostic_sub_agents": (
        "Diagnostic K, attempt indexes, elapsed time, timeout, cap, count, "
        "token budgets, and mission feature counts are evidence labels only "
        "and must not stop diagnostic sub-agent execution."
    ),
    "context_sub_agents": (
        "Context depth, summary tokens, context windows, elapsed time, timeout, "
        "cap, count, budget, and round counts are isolation/compaction metadata "
        "only and must not stop context sub-agent loops."
    ),
    "validation_regression_sub_agents": (
        "Validation/regression metadata may reject unsafe evidence, but time, "
        "round, attempt, snapshot-count, timeout, cap, count, token, and budget "
        "values must not stop validation/regression sub-agent loops."
    ),
    "mission_debug_sub_agent": (
        "Mission-debug feature, candidate, task, validation-contract, elapsed "
        "time, round, attempt, timeout, cap, count, token, and budget values "
        "are audit or packet-size metadata only."
    ),
    "worker_task_loop": (
        "Worker task loops may use single-operation guards and recovery "
        "checkpoints, but must not stop because time, round, turn, attempt, "
        "timeout, cap, count, token, budget, max_turns, checkpoint, compaction, "
        "or reasoning-budget references were reached."
    ),
    "goal_budgets": (
        "Goal budget fields are audit/progress metadata only and must not "
        "stop master, sub-agent, or Worker loops."
    ),
}


LEGACY_LIMIT_DRIVEN_SKIP_REASON_MARKERS: tuple[str, ...] = (
    "codex update interval",
    "collecting evidence only",
    "cooldown",
    "min failures",
    "min failure",
    "minimum failures",
    "failure threshold",
    "iteration",
    "round",
    "time limit",
    "wall clock",
    "wall-clock",
    "deadline",
    "timeout",
    "token budget",
    "budget",
    "attempt",
    "retry count",
    "attempt count",
    "attempt limit",
    "cap limit",
    "count limit",
    "max iterations",
    "max rounds",
    "max features",
    "max turns",
    "max_turns",
)


def unbounded_loop_flags() -> dict[str, bool]:
    return dict(UNBOUNDED_LOOP_FLAGS)


def unbounded_scope_flags(scope: LoopLimitScope) -> dict[str, bool]:
    return dict(SCOPE_UNBOUNDED_LOOP_FLAGS[scope])


def non_terminal_limit_flags(scope: LoopLimitScope | None = None) -> dict[str, bool]:
    flags = dict(COMMON_NON_TERMINAL_LIMIT_FLAGS)
    if scope is not None:
        flags.update(SCOPE_NON_TERMINAL_LIMIT_FLAGS[scope])
    return flags


def all_loop_non_terminal_event_metadata() -> dict[str, bool]:
    """Return event metadata proving limit-like fields are audit-only."""
    metadata = {
        "loop_stop_condition": False,
        "time_round_token_limit_driven": False,
    }
    for scope in LOOP_LIMIT_SCOPES:
        metadata.update(non_terminal_limit_flags(scope))
    return metadata


def is_legacy_limit_driven_skip_event(event: Mapping[str, object]) -> bool:
    """Detect old skip records created by limit-like Codex update metadata."""
    if str(event.get("action") or "") != "skipped":
        return False
    reason = normalize_limit_reason_text(event.get("reason"))
    if not reason:
        return False
    return any(marker in reason for marker in LEGACY_LIMIT_DRIVEN_SKIP_REASON_MARKERS)


def normalize_legacy_limit_driven_skip_event(
    event: Mapping[str, object],
) -> dict[str, object]:
    """Rewrite legacy limit-driven `skipped` records as audit-only evidence.

    Historical campaign states may contain Codex update events where interval,
    cooldown, min-failure, round, or attempt metadata was recorded as
    `action: skipped`. Current policy treats these fields only as audit or
    scheduling evidence, so packet/report consumers should see a non-terminal
    audit event while the raw legacy action stays available for traceability.
    """
    normalized: dict[str, object] = dict(event)
    if not is_legacy_limit_driven_skip_event(event):
        return normalized
    raw_action = str(event.get("action") or "")
    raw_reason = str(event.get("reason") or "")
    normalized.update(all_loop_non_terminal_event_metadata())
    normalized.update(
        {
            "action": "audit",
            "raw_action": raw_action,
            "raw_reason": raw_reason,
            "legacy_skip_action_normalized_from": raw_action,
            "legacy_skip_reason_normalized_from": raw_reason,
            "legacy_limit_driven_skip_normalized": True,
            "limit_driven_skip_audit_only": True,
            "limit_driven_skip_stop_condition": False,
            "master_loop_stop_condition": False,
            "sub_agent_loop_stop_condition": False,
            "codex_update_sub_agent_stop_condition": False,
            "diagnostic_sub_agent_stop_condition": False,
            "context_sub_agent_stop_condition": False,
            "validation_regression_sub_agent_stop_condition": False,
            "mission_debug_sub_agent_stop_condition": False,
            "worker_loop_stop_condition": False,
            "reason": (
                "legacy limit-driven skipped Codex update event normalized to "
                "audit-only evidence; interval, cooldown, min_failures, time, "
                "round, attempt, cap, count, token, and budget metadata must "
                "not skip master, sub-agent, validation/regression, "
                "mission-debug, context, or Worker loops"
            ),
        }
    )
    return normalized


def disallowed_limit_terminal_reasons() -> list[str]:
    return list(DISALLOWED_LIMIT_TERMINAL_REASONS)


def loop_owner_policy() -> dict[str, object]:
    """Return the centralized owner-level loop contract.

    This compact policy is easier to audit than the many compatibility flag
    aliases: every runtime owner is named once, and every limit-like field is
    declared audit/recovery metadata instead of a terminal predicate.
    """
    owners = tuple(LOOP_OWNER_SCOPES)
    runtime_owners = list(owners)
    return {
        "owners": list(owners),
        "runtime_loop_owners": runtime_owners,
        "owner_count": len(owners),
        "runtime_loop_owner_count": len(runtime_owners),
        "master_owner": "master_loop",
        "sub_agent_creation_policy": sub_agent_creation_policy(),
        "sub_agent_owners": [
            owner
            for owner in owners
            if owner.endswith("sub_agent") or owner.endswith("sub_agents")
        ],
        "worker_owner": "worker_task_loop",
        "every_runtime_loop_owner_has_no_time_limit": True,
        "every_runtime_loop_owner_has_no_round_limit": True,
        "every_runtime_loop_owner_has_no_time_or_round_limit": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_time_limits": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_round_limits": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_time_or_round_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_time_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_round_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_time_or_round_limits": True,
        "every_runtime_loop_owner_has_no_turn_limit": True,
        "every_runtime_loop_owner_has_no_max_turns_limit": True,
        "every_runtime_loop_owner_has_no_attempt_limit": True,
        "every_runtime_loop_owner_has_no_timeout_limit": True,
        "every_runtime_loop_owner_has_no_cap_limit": True,
        "every_runtime_loop_owner_has_no_count_limit": True,
        "every_runtime_loop_owner_has_no_token_limit": True,
        "every_runtime_loop_owner_has_no_budget_limit": True,
        "every_runtime_loop_owner_has_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limit": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_turn_limits": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_max_turns_limits": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_attempt_limits": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_timeout_limits": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_cap_limits": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_count_limits": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_token_limits": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_budget_limits": True,
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_turn_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_max_turns_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_attempt_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_timeout_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_cap_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_count_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_token_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_budget_limits": True,
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limits": True,
        "no_owner_has_time_limit": True,
        "no_owner_has_round_limit": True,
        "no_owner_has_time_or_round_limit": True,
        "no_owner_has_terminal_limit_predicate": True,
        "no_owner_has_limit_driven_completion": True,
        "no_owner_has_limit_driven_stop": True,
        "no_owner_has_limit_driven_skip": True,
        "no_owner_has_limit_driven_truncation": True,
        "no_owner_has_limit_driven_kill": True,
        "no_owner_has_limit_driven_cancellation": True,
        "no_owner_has_retry_threshold_stop": True,
        "no_owner_has_direct_replay_threshold_stop": True,
        "no_owner_has_retry_threshold_retry_denial": True,
        "no_owner_has_direct_replay_threshold_retry_denial": True,
        "time_round_limit_stop_condition": False,
        "time_or_round_limit_stop_condition": False,
        "terminal_limit_predicate_stop_condition": False,
        "limit_driven_completion_stop_condition": False,
        "limit_driven_stop_condition": False,
        "limit_driven_skip_stop_condition": False,
        "limit_driven_truncation_stop_condition": False,
        "limit_driven_kill_stop_condition": False,
        "limit_driven_cancellation_stop_condition": False,
        "retry_threshold_loop_stop_condition": False,
        "direct_replay_threshold_stop_condition": False,
        "retry_threshold_denies_retry": False,
        "direct_replay_threshold_denies_retry": False,
        "forbidden_terminal_predicates": list(
            LOOP_OWNER_POLICY_FORBIDDEN_TERMINAL_PREDICATES
        ),
        "forbidden_limit_driven_actions": list(
            LOOP_OWNER_POLICY_FORBIDDEN_LIMIT_DRIVEN_ACTIONS
        ),
        "audit_only_fields": list(LOOP_OWNER_POLICY_AUDIT_ONLY_FIELDS),
        "note": (
            "Every master loop, every sub-agent loop, and the Worker loop owner "
            "is unbounded by time, round, turn, attempt, cooldown, K, count, "
            "token, and budget metadata. Limit-like values may describe audit, "
            "scheduling, packet sizing, recovery, or single-operation policy, "
            "but must not become completion, stop, skip, truncate, or kill "
            "predicates."
        ),
    }


def normalize_limit_reason_text(reason: object) -> str:
    """Normalize free-form terminal reason text for contract checks."""
    return " ".join(str(reason or "").lower().replace("_", " ").split())


def is_limit_terminal_reason(reason: object) -> bool:
    """Return true when a terminal reason is based on loop-limit metadata.

    This is intentionally broader than historical goal parsing. It covers
    master, sub-agent, validation/regression, mission-debug, context, and
    Worker loop fields so new compatibility values cannot become terminal
    stop/completion reasons by accident.
    """
    normalized = normalize_limit_reason_text(reason)
    if not normalized:
        return False
    for marker in LIMIT_TERMINAL_REASON_MARKERS:
        if marker.replace("_", " ") in normalized:
            return True
    return any(
        normalize_limit_reason_text(disallowed).replace(" reached", "") in normalized
        for disallowed in DISALLOWED_LIMIT_TERMINAL_REASONS
    )


def base_loop_limit_contract(
    *,
    notes: Mapping[LoopLimitScope, str] | None = None,
) -> dict[str, dict[str, object]]:
    """Build the shared no-time/no-round contract for all agent loops.

    Callers may attach audit values or override notes, but the unbounded and
    non-terminal limit flags should come from this single source so master,
    sub-agent, mission-debug, validation/regression, and Worker reports cannot
    drift into different loop-limit semantics.
    """
    contract: dict[str, dict[str, object]] = {
        "all_loops": {
            **unbounded_loop_flags(),
            **non_terminal_limit_flags("all_loops"),
        }
    }
    owner_policy = loop_owner_policy()
    contract["all_loops"]["loop_owner_policy"] = owner_policy
    contract["all_loops"]["sub_agent_creation_policy"] = sub_agent_creation_policy()
    for scope in LOOP_LIMIT_SCOPES:
        if scope == "all_loops":
            continue
        loop_owner_flags = (
            LOOP_OWNER_NO_LIMIT_FLAGS if scope in LOOP_OWNER_SCOPES else {}
        )
        contract[scope] = {
            **unbounded_scope_flags(scope),
            **non_terminal_limit_flags(scope),
            **loop_owner_flags,
        }
        if scope in LOOP_OWNER_SCOPES:
            creation_policy = sub_agent_creation_policy()
            contract[scope]["loop_owner_policy"] = {
                "owner": scope,
                "runtime_loop_owner": True,
                "listed_in_owner_policy": scope in owner_policy["owners"],
                "listed_in_runtime_loop_owners": scope
                in owner_policy["runtime_loop_owners"],
                "sub_agent_creation_policy": creation_policy,
                "may_create_sub_agents": bool(
                    creation_policy["owner_creation_permissions"].get(scope)
                ),
                "nested_sub_agent_creation_allowed": False,
                "only_master_loop_may_create_sub_agents": True,
                "runtime_owner_has_no_time_limit": True,
                "runtime_owner_has_no_round_limit": True,
                "runtime_owner_has_no_time_or_round_limit": True,
                "master_sub_agent_worker_owner_family_has_no_time_limit": True,
                "master_sub_agent_worker_owner_family_has_no_round_limit": True,
                "master_sub_agent_worker_owner_family_has_no_time_or_round_limit": True,
                "no_time_limit": True,
                "no_round_limit": True,
                "no_time_or_round_limit": True,
                "no_terminal_limit_predicate": True,
                "no_limit_driven_completion": True,
                "no_limit_driven_stop": True,
                "no_limit_driven_skip": True,
                "no_limit_driven_truncation": True,
                "no_limit_driven_kill": True,
                "no_limit_driven_cancellation": True,
                "no_retry_threshold_stop": True,
                "no_direct_replay_threshold_stop": True,
                "no_retry_threshold_retry_denial": True,
                "no_direct_replay_threshold_retry_denial": True,
                "time_round_limit_stop_condition": False,
                "time_or_round_limit_stop_condition": False,
                "terminal_limit_predicate_stop_condition": False,
                "limit_driven_completion_stop_condition": False,
                "limit_driven_stop_condition": False,
                "limit_driven_skip_stop_condition": False,
                "limit_driven_truncation_stop_condition": False,
                "limit_driven_kill_stop_condition": False,
                "limit_driven_cancellation_stop_condition": False,
                "retry_threshold_loop_stop_condition": False,
                "direct_replay_threshold_stop_condition": False,
                "retry_threshold_denies_retry": False,
                "direct_replay_threshold_denies_retry": False,
                "forbidden_terminal_predicates": owner_policy[
                    "forbidden_terminal_predicates"
                ],
                "forbidden_limit_driven_actions": owner_policy[
                    "forbidden_limit_driven_actions"
                ],
                "audit_only_fields": owner_policy["audit_only_fields"],
            }

    scope_notes = dict(DEFAULT_LOOP_LIMIT_NOTES)
    if notes:
        scope_notes.update(notes)
    for scope, note in scope_notes.items():
        contract[scope]["note"] = note
    return contract
