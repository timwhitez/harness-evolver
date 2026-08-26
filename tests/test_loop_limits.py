from types import SimpleNamespace
from pathlib import Path

from hl.loop_limits import (
    disallowed_limit_terminal_reasons,
    is_limit_terminal_reason,
    loop_owner_policy,
    non_terminal_limit_flags,
    sub_agent_creation_policy,
    unbounded_loop_flags,
    unbounded_scope_flags,
)
from harness.context import (
    CompactionStrategy,
    ContextIsolation,
    ContextManager,
    TrajectoryPack,
)
from hl.loop import HLLoop
from meta.codex_update import CodexUpdateEngine
from meta.missions import MissionPlanner
from scripts.run_campaign import (
    _codex_update_should_run,
    _loop_limit_contract,
)


LOOP_LIMIT_SCOPES = (
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


def _assert_no_stop_conditions(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.endswith("_stop_condition"):
                assert nested is False, f"{key} must remain audit-only"
            _assert_no_stop_conditions(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_stop_conditions(nested)


def _assert_no_time_or_round_limits(contract: dict) -> None:
    assert contract["all_loops"]["all_loops_have_no_time_limits"] is True
    assert contract["all_loops"]["all_loops_have_no_round_limits"] is True
    assert contract["all_loops"]["all_agent_loops_have_no_time_or_round_limits"] is True
    assert contract["all_loops"]["all_sub_agent_loops_have_no_time_or_round_limits"] is True
    assert contract["all_loops"]["all_sub_agent_loops_have_no_time_limits"] is True
    assert contract["all_loops"]["all_sub_agent_loops_have_no_round_limits"] is True
    assert contract["all_loops"]["all_sub_agent_loop_owners_have_no_time_limits"] is True
    assert contract["all_loops"]["all_sub_agent_loop_owners_have_no_round_limits"] is True
    assert (
        contract["all_loops"]["all_sub_agent_loop_owners_have_no_time_or_round_limits"]
        is True
    )
    assert contract["all_loops"]["all_loop_owners_have_no_time_or_round_limits"] is True
    assert (
        contract["all_loops"][
            "all_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_and_all_sub_agent_loop_owners_have_no_time_or_round_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_sub_agent_and_worker_loop_owners_have_no_time_or_round_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_and_all_sub_agent_and_worker_loop_owners_have_no_round_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_or_round_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_sub_agent_and_worker_loops_have_no_time_or_round_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_sub_agent_and_worker_loops_have_no_time_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_sub_agent_and_worker_loops_have_no_round_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_and_all_sub_agent_and_worker_loops_have_no_time_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_and_all_sub_agent_and_worker_loops_have_no_round_limits"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_and_all_sub_agent_and_worker_loops_have_no_time_or_round_limits"
        ]
        is True
    )
    for scope in LOOP_LIMIT_SCOPES:
        scope_contract = contract[scope]
        if scope != "all_loops":
            assert scope_contract["no_time_or_round_limits"] is True, scope
            assert scope_contract["no_time_limit"] is True, scope
            assert scope_contract["no_round_limit"] is True, scope
            assert scope_contract["unbounded_by_time"] is True, scope
            assert scope_contract["unbounded_by_round"] is True, scope
            if scope != "goal_budgets":
                assert scope_contract["loop_owner_has_no_time_limit"] is True, scope
                assert scope_contract["loop_owner_has_no_round_limit"] is True, scope
                assert (
                    scope_contract["loop_owner_has_no_time_or_round_limit"] is True
                ), scope
                assert scope_contract["loop_owner_has_no_timeout_limit"] is True, scope
                assert scope_contract["loop_owner_has_no_cap_limit"] is True, scope
                assert scope_contract["loop_owner_has_no_count_limit"] is True, scope
                assert scope_contract["loop_owner_has_no_token_limit"] is True, scope
                assert scope_contract["loop_owner_has_no_budget_limit"] is True, scope
                assert (
                    scope_contract[
                        "loop_owner_has_no_time_round_attempt_timeout_cap_count_token_or_budget_limit"
                    ]
                    is True
                ), scope
        for key in (
            "time_and_round_limits_stop_condition",
            "time_round_limit_stop_condition",
            "time_or_round_limit_stop_condition",
            "agent_time_limit_stop_condition",
            "agent_round_limit_stop_condition",
            "master_time_round_limit_stop_condition",
            "master_time_limit_stop_condition",
            "master_round_limit_stop_condition",
            "sub_agent_time_round_limit_stop_condition",
            "sub_agent_time_limit_stop_condition",
            "sub_agent_round_limit_stop_condition",
            "worker_time_round_limit_stop_condition",
            "worker_time_limit_stop_condition",
            "worker_round_limit_stop_condition",
            "time_limit_stop_condition",
            "round_limit_stop_condition",
        ):
            if key in scope_contract:
                assert scope_contract[key] is False, f"{scope}.{key}"


def _assert_master_only_sub_agent_creation_policy(contract: dict) -> None:
    policy = contract["all_loops"]["sub_agent_creation_policy"]
    assert policy["allowed_creator_owner"] == "master_loop"
    assert policy["master_loop_may_create_sub_agents"] is True
    assert policy["only_master_loop_may_create_sub_agents"] is True
    assert policy["sub_agent_owners_may_create_sub_agents"] is False
    assert policy["worker_task_loop_may_create_sub_agents"] is False
    assert policy["nested_sub_agent_creation_allowed"] is False
    assert policy["sub_agents_may_spawn_nested_sub_agents"] is False
    assert policy["max_sub_agent_nesting_depth"] == 1
    assert policy["owner_creation_permissions"]["master_loop"] is True
    for owner in (
        "codex_update_sub_agent",
        "diagnostic_sub_agents",
        "context_sub_agents",
        "validation_regression_sub_agents",
        "mission_debug_sub_agent",
        "worker_task_loop",
    ):
        assert policy["owner_creation_permissions"][owner] is False
        owner_policy = contract[owner]["loop_owner_policy"]
        assert owner_policy["may_create_sub_agents"] is False
        assert owner_policy["nested_sub_agent_creation_allowed"] is False
        assert owner_policy["only_master_loop_may_create_sub_agents"] is True
    master_policy = contract["master_loop"]["loop_owner_policy"]
    assert master_policy["may_create_sub_agents"] is True

def _campaign_loop_limit_contract() -> dict:
    return _loop_limit_contract(
        iteration_limit=3,
        args=SimpleNamespace(
            max_turns_audit=7,
            llm_timeout_seconds=11,
            tool_timeout_seconds=13,
            mission_debug_max_features=2,
        ),
        codex_config={"timeout_seconds": 17},
        update_policy={
            "interval": 19,
            "min_failures": 23,
            "cooldown_after_rollback": 29,
            "partial_pass_diagnostic_k": 31,
        },
        goal_plan={
            "goal": {
                "token_budget": 37,
                "token_budget_scope": "campaign",
                "wall_time_budget_seconds": 41,
            }
        },
    )


def test_shared_loop_limit_helpers_are_unbounded_and_non_terminal() -> None:
    for key, value in unbounded_loop_flags().items():
        assert value is True, f"{key} must stay explicitly unbounded"

    assert unbounded_loop_flags()["all_agent_loops_have_no_time_or_round_limits"] is True
    assert unbounded_loop_flags()["all_sub_agent_loops_have_no_time_or_round_limits"] is True
    assert unbounded_loop_flags()["all_sub_agent_loop_owners_have_no_time_limits"] is True
    assert unbounded_loop_flags()["all_sub_agent_loop_owners_have_no_round_limits"] is True
    assert (
        unbounded_loop_flags()["all_sub_agent_loop_owners_have_no_time_or_round_limits"]
        is True
    )
    assert unbounded_loop_flags()["all_loop_owners_have_no_time_or_round_limits"] is True
    assert (
        unbounded_loop_flags()[
            "all_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits"
        ]
        is True
    )
    for key in (
        "master_and_sub_agent_loops_have_no_time_limits",
        "master_and_sub_agent_loops_have_no_round_limits",
        "master_and_sub_agent_loops_have_no_time_or_round_limits",
        "master_and_sub_agent_loops_unbounded_by_time",
        "master_and_sub_agent_loops_unbounded_by_round",
        "master_and_sub_agent_loops_unbounded_by_time_and_round",
        "all_agent_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget",
        "master_and_sub_agent_loops_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget",
        "master_sub_agent_and_worker_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget",
        "master_and_all_sub_agent_and_worker_loops_have_no_time_limits",
        "master_and_all_sub_agent_and_worker_loops_have_no_round_limits",
        "master_and_all_sub_agent_and_worker_loops_have_no_time_or_round_limits",
        "master_and_all_sub_agent_and_worker_loops_unbounded_by_time",
        "master_and_all_sub_agent_and_worker_loops_unbounded_by_round",
        "master_and_all_sub_agent_and_worker_loops_unbounded_by_time_and_round",
        "master_and_all_sub_agent_and_worker_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget",
    ):
        assert unbounded_loop_flags()[key] is True
    assert (
        unbounded_loop_flags()[
            "master_and_all_sub_agent_loop_owners_have_no_time_or_round_limits"
        ]
        is True
    )
    assert (
        unbounded_loop_flags()[
            "master_sub_agent_and_worker_loop_owners_have_no_time_or_round_limits"
        ]
        is True
    )
    assert (
        unbounded_loop_flags()[
            "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_or_round_limits"
        ]
        is True
    )

    for scope in LOOP_LIMIT_SCOPES:
        flags = non_terminal_limit_flags(scope)
        assert flags
        scoped_unbounded = unbounded_scope_flags(scope)
        if scope != "all_loops":
            assert scoped_unbounded["no_time_or_round_limits"] is True
            assert scoped_unbounded["no_time_limit"] is True
            assert scoped_unbounded["no_round_limit"] is True
        for key, value in flags.items():
            assert key.endswith("_stop_condition") or key.endswith("_limit_allowed"), key
            assert value is False, f"{scope}.{key} must stay audit-only"

    assert non_terminal_limit_flags("codex_update_sub_agent")[
        "host_validation_timeout_seconds_stop_condition"
    ] is False
    assert non_terminal_limit_flags("diagnostic_sub_agents")[
        "diagnostic_target_k_stop_condition"
    ] is False
    assert non_terminal_limit_flags("mission_debug_sub_agent")[
        "max_features_stop_condition"
    ] is False
    assert non_terminal_limit_flags("validation_regression_sub_agents")[
        "validation_timeout_stop_condition"
    ] is False
    assert non_terminal_limit_flags("worker_task_loop")[
        "checkpoint_cooldown_stop_condition"
    ] is False
    assert non_terminal_limit_flags("goal_budgets")[
        "time_round_token_budget_stop_condition"
    ] is False

    disallowed = "\n".join(disallowed_limit_terminal_reasons())
    for term in (
        "iteration",
        "round",
        "master round",
        "sub-agent round",
        "worker round",
        "wall-clock",
        "master time",
        "sub-agent time",
        "worker time",
        "deadline",
        "timeout",
        "token budget",
        "budget exhausted",
        "attempt",
        "retry",
        "cap",
        "count limit",
        "max_turns",
        "context depth",
        "cooldown",
        "diagnostic K",
    ):
        assert term in disallowed


def test_limit_terminal_reason_classifier_covers_all_loop_owners() -> None:
    for reason in disallowed_limit_terminal_reasons():
        assert is_limit_terminal_reason(reason), reason

    for reason in (
        "master loop reached max_iterations",
        "sub-agent attempt cap reached",
        "Codex update timeout_seconds exhausted",
        "diagnostic target K reached",
        "context depth window reached",
        "validation regression snapshot count cap reached",
        "mission max_features reached",
        "Worker max_turns reached",
        "token budget exhausted before score target",
        "rate-limit cooldown active",
    ):
        assert is_limit_terminal_reason(reason), reason

    for reason in (
        "explicit user stopped campaign",
        "operator complete after submit terminal action",
        "hard verifier contract failed",
    ):
        assert not is_limit_terminal_reason(reason), reason


def test_trajectory_pack_keeps_full_context_despite_legacy_max_fields() -> None:
    pack = TrajectoryPack(max_events=2, max_output_chars=3)
    long_output = "abcdefghijklmnopqrstuvwxyz"
    events = [
        {"type": "tool_call", "output": long_output, "index": index}
        for index in range(5)
    ]

    selected = pack.select(events)
    metadata = pack.audit_metadata(events)

    assert selected == events
    assert len(selected) == 5
    assert selected[0] is not events[0]
    assert all(event.get("type") != "omitted" for event in selected)
    assert selected[0]["output"] == long_output
    assert metadata["max_events_audit_only"] == 2
    assert metadata["max_output_chars_audit_only"] == 3
    for key, value in metadata.items():
        if key.endswith("_stop_condition"):
            assert value is False, key
    assert metadata["time_round_token_limit_driven"] is False


def test_context_package_exports_all_context_managers() -> None:
    assert ContextManager.__name__ == "ContextManager"
    assert CompactionStrategy.__name__ == "CompactionStrategy"
    assert ContextIsolation.__name__ == "ContextIsolation"
    assert TrajectoryPack.__name__ == "TrajectoryPack"


def test_all_loop_limit_report_entrypoints_share_non_terminal_contract(
    tmp_path,
) -> None:
    contracts = {
        "campaign": _campaign_loop_limit_contract(),
        "mission_debug": MissionPlanner()._loop_limit_contract(max_features=0),
        "codex_update": CodexUpdateEngine(
            repo_root=tmp_path,
            events_dir=tmp_path / "diffs",
            timeout_seconds=5,
            validation_timeout_seconds=7,
        )._loop_limit_contract(),
    }

    for name, contract in contracts.items():
        _assert_no_stop_conditions(contract)
        for key, expected in unbounded_loop_flags().items():
            assert contract["all_loops"][key] is expected, f"{name}.{key}"
        for scope in LOOP_LIMIT_SCOPES:
            for key, expected in non_terminal_limit_flags(scope).items():
                assert contract[scope][key] is expected, f"{name}.{scope}.{key}"

        _assert_no_time_or_round_limits(contract)
        _assert_master_only_sub_agent_creation_policy(contract)

        assert (
            contract["all_loops"][
                "all_agent_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_and_budget"
            ]
            is True
        )
        assert contract["all_loops"]["deadline_stop_condition"] is False
        assert contract["all_loops"]["agent_time_limit_stop_condition"] is False
        assert contract["all_loops"]["agent_round_limit_stop_condition"] is False
        assert contract["all_loops"]["master_time_limit_stop_condition"] is False
        assert contract["all_loops"]["master_round_limit_stop_condition"] is False
        assert contract["all_loops"]["sub_agent_time_limit_stop_condition"] is False
        assert contract["all_loops"]["sub_agent_round_limit_stop_condition"] is False
        assert contract["all_loops"]["worker_time_limit_stop_condition"] is False
        assert contract["all_loops"]["worker_round_limit_stop_condition"] is False
        assert contract["all_loops"]["timeout_seconds_stop_condition"] is False
        assert contract["all_loops"]["process_timeout_stop_condition"] is False
        assert contract["all_loops"]["retry_count_stop_condition"] is False
        assert contract["all_loops"]["cap_stop_condition"] is False
        assert contract["all_loops"]["count_limit_stop_condition"] is False
        assert contract["all_loops"]["turn_count_stop_condition"] is False
        assert contract["master_loop"]["run_task_cap_stop_condition"] is False
        assert contract["master_loop"]["max_tasks_stop_condition"] is False
        assert contract["validation_regression_sub_agents"][
            "regression_failed_task_retry_stop_condition"
        ] is False
        assert contract["worker_task_loop"]["turn_limit_stop_condition"] is False
        assert contract["worker_task_loop"][
            "provider_request_timeout_stop_condition"
        ] is False


def test_sub_agent_creation_policy_allows_only_master_owner() -> None:
    policy = sub_agent_creation_policy()

    assert policy["allowed_creator_owner"] == "master_loop"
    assert policy["master_loop_may_create_sub_agents"] is True
    assert policy["only_master_loop_may_create_sub_agents"] is True
    assert policy["nested_sub_agent_creation_allowed"] is False
    assert policy["sub_agents_may_spawn_nested_sub_agents"] is False
    assert policy["max_sub_agent_nesting_depth"] == 1
    assert policy["sub_agent_creation_loop_stop_condition"] is False
    assert policy["nested_sub_agent_creation_stop_condition"] is False
    assert policy["owner_creation_permissions"]["master_loop"] is True
    assert policy["owner_creation_permissions"]["codex_update_sub_agent"] is False
    assert policy["owner_creation_permissions"]["worker_task_loop"] is False
    assert "codex" in policy["blocked_nested_agent_commands"]
    assert "openai-codex" in policy["blocked_nested_agent_commands"]
    assert "gemini" in policy["blocked_nested_agent_commands"]
    assert "opencode" in policy["blocked_nested_agent_commands"]
    assert "aider" in policy["blocked_nested_agent_commands"]
    assert "cursor-agent" in policy["blocked_nested_agent_commands"]
    assert "factory" in policy["blocked_nested_agent_commands"]
    assert "droid" in policy["blocked_nested_agent_commands"]


def test_every_loop_scope_rejects_time_round_attempt_budget_and_cap_stops() -> None:
    contract = _campaign_loop_limit_contract()
    expected_limit_flags = (
        "time_and_round_limits_stop_condition",
        "time_or_round_limit_stop_condition",
        "time_limit_stop_condition",
        "round_limit_stop_condition",
        "agent_time_limit_stop_condition",
        "agent_round_limit_stop_condition",
        "master_time_limit_stop_condition",
        "master_round_limit_stop_condition",
        "sub_agent_time_limit_stop_condition",
        "sub_agent_round_limit_stop_condition",
        "worker_time_limit_stop_condition",
        "worker_round_limit_stop_condition",
        "attempt_count_stop_condition",
        "attempt_limit_stop_condition",
        "timeout_seconds_stop_condition",
        "timeout_limit_stop_condition",
        "token_limit_stop_condition",
        "token_budget_stop_condition",
        "token_count_stop_condition",
        "budget_stop_condition",
        "budget_limit_stop_condition",
        "budget_exhaustion_stop_condition",
        "cap_stop_condition",
        "cap_limit_stop_condition",
        "count_limit_stop_condition",
    )

    for scope_name in LOOP_LIMIT_SCOPES:
        scope = contract[scope_name]
        _assert_no_stop_conditions(scope)
        unbounded = [
            key
            for key, value in scope.items()
            if key.startswith("unbounded_by_") and value is True
        ]
        if scope_name != "all_loops":
            assert unbounded, f"{scope_name} must expose explicit unbounded flags"
        for flag in expected_limit_flags:
            if flag in scope:
                assert scope[flag] is False, f"{scope_name}.{flag} must be non-terminal"

    assert (
        contract["all_loops"][
            "all_agent_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_and_budget"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "all_agent_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "master_sub_agent_and_worker_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget"
        ]
        is True
    )
    for key in (
        "master_and_sub_agent_loops_have_no_time_limits",
        "master_and_sub_agent_loops_have_no_round_limits",
        "master_and_sub_agent_loops_have_no_time_or_round_limits",
        "master_and_sub_agent_loops_unbounded_by_time",
        "master_and_sub_agent_loops_unbounded_by_round",
        "master_and_sub_agent_loops_unbounded_by_time_and_round",
        "master_and_all_sub_agent_loops_have_no_time_limits",
        "master_and_all_sub_agent_loops_have_no_round_limits",
        "master_and_all_sub_agent_loops_have_no_time_or_round_limits",
        "master_and_all_sub_agent_loops_unbounded_by_time",
        "master_and_all_sub_agent_loops_unbounded_by_round",
        "master_and_all_sub_agent_loops_unbounded_by_time_and_round",
        "master_and_all_sub_agent_loops_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget",
    ):
        assert contract["all_loops"][key] is True
    assert (
        contract["validation_regression_sub_agents"][
            "unbounded_by_time_round_attempt_snapshot_count_lane_cap_timeout_and_budget"
        ]
        is True
    )
    assert (
        contract["validation_regression_sub_agents"][
            "unbounded_by_time_round_attempt_snapshot_count_lane_cap_timeout_token_and_budget"
        ]
        is True
    )
    assert (
        contract["worker_task_loop"][
            "unbounded_by_time_round_turn_attempt_timeout_cap_count_and_budget"
        ]
        is True
    )
    assert (
        contract["worker_task_loop"][
            "unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget"
        ]
        is True
    )


def test_every_loop_owner_is_unbounded_by_time_round_and_counting_metadata() -> None:
    contract = _campaign_loop_limit_contract()
    owner_scopes = (
        "master_loop",
        "codex_update_sub_agent",
        "diagnostic_sub_agents",
        "context_sub_agents",
        "validation_regression_sub_agents",
        "mission_debug_sub_agent",
        "worker_task_loop",
    )

    for scope_name in owner_scopes:
        scope = contract[scope_name]
        for key in (
            "loop_owner_has_no_time_limit",
            "loop_owner_has_no_round_limit",
            "loop_owner_has_no_attempt_limit",
            "loop_owner_has_no_timeout_limit",
            "loop_owner_has_no_cap_limit",
            "loop_owner_has_no_count_limit",
            "loop_owner_has_no_token_limit",
            "loop_owner_has_no_budget_limit",
            "loop_owner_has_no_time_round_attempt_timeout_cap_count_token_or_budget_limit",
        ):
            assert scope[key] is True, f"{scope_name}.{key}"
        for key in (
            "time_limit_allowed",
            "round_limit_allowed",
            "attempt_limit_allowed",
            "timeout_limit_allowed",
            "cap_limit_allowed",
            "count_limit_allowed",
            "token_limit_allowed",
            "budget_limit_allowed",
        ):
            assert scope[key] is False, f"{scope_name}.{key}"

    all_loops = contract["all_loops"]
    for key in (
        "all_sub_agent_loop_owners_have_no_time_limits",
        "all_sub_agent_loop_owners_have_no_round_limits",
        "all_sub_agent_loop_owners_have_no_time_or_round_limits",
        "master_and_all_sub_agent_loops_have_no_time_limits",
        "master_and_all_sub_agent_loops_have_no_round_limits",
        "master_and_all_sub_agent_loops_have_no_time_or_round_limits",
        "master_and_all_sub_agent_loops_unbounded_by_time",
        "master_and_all_sub_agent_loops_unbounded_by_round",
        "master_and_all_sub_agent_loops_unbounded_by_time_and_round",
        "master_and_all_sub_agent_loops_unbounded_by_time_round_attempt_timeout_cap_count_token_and_budget",
        "master_and_all_sub_agent_and_worker_loops_have_no_time_limits",
        "master_and_all_sub_agent_and_worker_loops_have_no_round_limits",
        "master_and_all_sub_agent_and_worker_loops_have_no_time_or_round_limits",
        "master_and_all_sub_agent_and_worker_loops_unbounded_by_time",
        "master_and_all_sub_agent_and_worker_loops_unbounded_by_round",
        "master_and_all_sub_agent_and_worker_loops_unbounded_by_time_and_round",
        "master_and_all_sub_agent_and_worker_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget",
    ):
        assert all_loops[key] is True
    assert (
        all_loops[
            "master_and_all_sub_agent_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits"
        ]
        is True
    )
    assert (
        all_loops[
            "master_sub_agent_and_worker_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits"
        ]
        is True
    )
    assert (
        all_loops[
            "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits"
        ]
        is True
    )


def test_master_sub_agent_worker_loop_owners_have_no_runtime_limits(tmp_path) -> None:
    contracts = {
        "campaign": _campaign_loop_limit_contract(),
        "codex_update": CodexUpdateEngine(
            repo_root=tmp_path,
            events_dir=tmp_path / "diffs",
            timeout_seconds=1,
            validation_timeout_seconds=1,
        )._loop_limit_contract(),
        "mission_debug": MissionPlanner()._loop_limit_contract(max_features=1),
    }
    owner_scopes = (
        "master_loop",
        "codex_update_sub_agent",
        "diagnostic_sub_agents",
        "context_sub_agents",
        "validation_regression_sub_agents",
        "mission_debug_sub_agent",
        "worker_task_loop",
    )
    forbidden_stop_flags = (
        "time_limit_stop_condition",
        "round_limit_stop_condition",
        "attempt_limit_stop_condition",
        "timeout_seconds_stop_condition",
        "token_budget_stop_condition",
        "budget_exhaustion_stop_condition",
        "max_turns_stop_condition",
        "cap_stop_condition",
        "count_limit_stop_condition",
    )

    for contract_name, contract in contracts.items():
        all_loops = contract["all_loops"]
        for key in (
            "all_sub_agent_loop_owners_have_no_time_limits",
            "all_sub_agent_loop_owners_have_no_round_limits",
            "all_sub_agent_loop_owners_have_no_time_or_round_limits",
            "master_and_all_sub_agent_loop_owners_have_no_time_limits",
            "master_and_all_sub_agent_loop_owners_have_no_round_limits",
            "master_and_all_sub_agent_loop_owners_have_no_time_or_round_limits",
            "master_and_every_sub_agent_loop_owner_has_no_time_limit",
            "master_and_every_sub_agent_loop_owner_has_no_round_limit",
            "master_and_every_sub_agent_loop_owner_has_no_time_or_round_limit",
            "master_sub_agent_and_worker_loop_owners_have_no_time_limits",
            "master_sub_agent_and_worker_loop_owners_have_no_round_limits",
            "master_sub_agent_and_worker_loop_owners_have_no_time_or_round_limits",
            "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_limits",
            "master_and_all_sub_agent_and_worker_loop_owners_have_no_round_limits",
            "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_or_round_limits",
            "master_every_sub_agent_and_worker_loop_owner_has_no_time_limit",
            "master_every_sub_agent_and_worker_loop_owner_has_no_round_limit",
            "master_every_sub_agent_and_worker_loop_owner_has_no_time_or_round_limit",
            "master_sub_agent_and_worker_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits",
            "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits",
        ):
            assert all_loops[key] is True, f"{contract_name}.{key}"

        for scope_name in owner_scopes:
            scope = contract[scope_name]
            for key in (
                "loop_owner_has_no_time_limit",
                "loop_owner_has_no_round_limit",
                "loop_owner_has_no_attempt_limit",
                "loop_owner_has_no_timeout_limit",
                "loop_owner_has_no_cap_limit",
                "loop_owner_has_no_count_limit",
                "loop_owner_has_no_token_limit",
                "loop_owner_has_no_budget_limit",
                "loop_owner_has_no_time_round_attempt_timeout_cap_count_token_or_budget_limit",
            ):
                assert scope[key] is True, f"{contract_name}.{scope_name}.{key}"
            for key in (
                "time_limit_allowed",
                "round_limit_allowed",
                "attempt_limit_allowed",
                "timeout_limit_allowed",
                "cap_limit_allowed",
                "count_limit_allowed",
                "token_limit_allowed",
                "budget_limit_allowed",
            ):
                assert scope[key] is False, f"{contract_name}.{scope_name}.{key}"
            for key in forbidden_stop_flags:
                if key in scope:
                    assert scope[key] is False, f"{contract_name}.{scope_name}.{key}"

        assert contract["goal_budgets"]["token_budget_stop_condition"] is False
        assert contract["goal_budgets"]["wall_time_budget_stop_condition"] is False
        assert contract["goal_budgets"]["time_round_token_budget_stop_condition"] is False


def test_loop_owner_policy_explicitly_covers_master_sub_agents_and_worker(tmp_path) -> None:
    expected_owners = [
        "master_loop",
        "codex_update_sub_agent",
        "diagnostic_sub_agents",
        "context_sub_agents",
        "validation_regression_sub_agents",
        "mission_debug_sub_agent",
        "worker_task_loop",
    ]
    expected_predicates = {
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
    }
    expected_audit_fields = {
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
    }
    expected_forbidden_actions = {
        "complete",
        "stop",
        "skip",
        "truncate",
        "kill",
        "cancel",
    }

    direct_policy = loop_owner_policy()
    assert direct_policy["owners"] == expected_owners
    assert direct_policy["runtime_loop_owners"] == expected_owners
    assert direct_policy["runtime_loop_owner_count"] == len(expected_owners)
    assert direct_policy["sub_agent_owners"] == expected_owners[1:6]
    assert direct_policy["sub_agent_creation_policy"]["allowed_creator_owner"] == "master_loop"
    assert direct_policy["sub_agent_creation_policy"]["nested_sub_agent_creation_allowed"] is False
    assert direct_policy["sub_agent_creation_policy"]["owner_creation_permissions"]["master_loop"] is True
    assert direct_policy["sub_agent_creation_policy"]["owner_creation_permissions"]["codex_update_sub_agent"] is False
    assert direct_policy["every_runtime_loop_owner_has_no_time_limit"] is True
    assert direct_policy["every_runtime_loop_owner_has_no_round_limit"] is True
    assert direct_policy["every_runtime_loop_owner_has_no_time_or_round_limit"] is True
    assert (
        direct_policy[
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_time_limits"
        ]
        is True
    )
    assert (
        direct_policy[
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_round_limits"
        ]
        is True
    )
    assert (
        direct_policy[
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_time_or_round_limits"
        ]
        is True
    )
    assert (
        direct_policy["master_all_sub_agent_and_worker_runtime_loop_owners_have_no_time_limits"]
        is True
    )
    assert (
        direct_policy["master_all_sub_agent_and_worker_runtime_loop_owners_have_no_round_limits"]
        is True
    )
    assert (
        direct_policy[
            "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_time_or_round_limits"
        ]
        is True
    )
    for key in (
        "every_runtime_loop_owner_has_no_turn_limit",
        "every_runtime_loop_owner_has_no_max_turns_limit",
        "every_runtime_loop_owner_has_no_attempt_limit",
        "every_runtime_loop_owner_has_no_timeout_limit",
        "every_runtime_loop_owner_has_no_cap_limit",
        "every_runtime_loop_owner_has_no_count_limit",
        "every_runtime_loop_owner_has_no_token_limit",
        "every_runtime_loop_owner_has_no_budget_limit",
        "every_runtime_loop_owner_has_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limit",
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_turn_limits",
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_max_turns_limits",
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_attempt_limits",
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_timeout_limits",
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_cap_limits",
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_count_limits",
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_token_limits",
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_budget_limits",
        "master_sub_agent_and_worker_runtime_loop_owners_have_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limits",
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_turn_limits",
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_max_turns_limits",
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_attempt_limits",
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_timeout_limits",
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_cap_limits",
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_count_limits",
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_token_limits",
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_budget_limits",
        "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limits",
    ):
        assert direct_policy[key] is True, key
    assert direct_policy["no_owner_has_time_limit"] is True
    assert direct_policy["no_owner_has_round_limit"] is True
    assert direct_policy["no_owner_has_terminal_limit_predicate"] is True
    assert direct_policy["no_owner_has_limit_driven_completion"] is True
    assert direct_policy["no_owner_has_limit_driven_stop"] is True
    assert direct_policy["no_owner_has_limit_driven_skip"] is True
    assert direct_policy["no_owner_has_limit_driven_truncation"] is True
    assert direct_policy["no_owner_has_limit_driven_kill"] is True
    assert direct_policy["no_owner_has_limit_driven_cancellation"] is True
    assert direct_policy["no_owner_has_retry_threshold_stop"] is True
    assert direct_policy["no_owner_has_direct_replay_threshold_stop"] is True
    assert direct_policy["no_owner_has_retry_threshold_retry_denial"] is True
    assert direct_policy["no_owner_has_direct_replay_threshold_retry_denial"] is True
    assert direct_policy["terminal_limit_predicate_stop_condition"] is False
    assert direct_policy["limit_driven_completion_stop_condition"] is False
    assert direct_policy["limit_driven_stop_condition"] is False
    assert direct_policy["limit_driven_skip_stop_condition"] is False
    assert direct_policy["limit_driven_truncation_stop_condition"] is False
    assert direct_policy["limit_driven_kill_stop_condition"] is False
    assert direct_policy["limit_driven_cancellation_stop_condition"] is False
    assert direct_policy["retry_threshold_loop_stop_condition"] is False
    assert direct_policy["direct_replay_threshold_stop_condition"] is False
    assert direct_policy["retry_threshold_denies_retry"] is False
    assert direct_policy["direct_replay_threshold_denies_retry"] is False
    assert set(direct_policy["forbidden_terminal_predicates"]) == expected_predicates
    assert (
        set(direct_policy["forbidden_limit_driven_actions"])
        == expected_forbidden_actions
    )
    assert set(direct_policy["audit_only_fields"]) == expected_audit_fields

    contracts = {
        "campaign": _campaign_loop_limit_contract(),
        "codex_update": CodexUpdateEngine(
            repo_root=tmp_path,
            events_dir=tmp_path / "diffs",
            timeout_seconds=1,
            validation_timeout_seconds=1,
        )._loop_limit_contract(),
        "mission_debug": MissionPlanner()._loop_limit_contract(max_features=1),
    }

    for contract_name, contract in contracts.items():
        _assert_master_only_sub_agent_creation_policy(contract)
        policy = contract["all_loops"]["loop_owner_policy"]
        assert policy["owners"] == expected_owners, contract_name
        assert policy["runtime_loop_owners"] == expected_owners, contract_name
        assert policy["runtime_loop_owner_count"] == len(expected_owners), contract_name
        assert policy["sub_agent_owners"] == expected_owners[1:6], contract_name
        assert policy["master_owner"] == "master_loop", contract_name
        assert policy["worker_owner"] == "worker_task_loop", contract_name
        assert policy["every_runtime_loop_owner_has_no_time_limit"] is True, contract_name
        assert policy["every_runtime_loop_owner_has_no_round_limit"] is True, contract_name
        assert (
            policy["every_runtime_loop_owner_has_no_time_or_round_limit"] is True
        ), contract_name
        assert (
            policy[
                "master_sub_agent_and_worker_runtime_loop_owners_have_no_time_limits"
            ]
            is True
        ), contract_name
        assert (
            policy[
                "master_sub_agent_and_worker_runtime_loop_owners_have_no_round_limits"
            ]
            is True
        ), contract_name
        assert (
            policy[
                "master_sub_agent_and_worker_runtime_loop_owners_have_no_time_or_round_limits"
            ]
            is True
        ), contract_name
        assert (
            policy["master_all_sub_agent_and_worker_runtime_loop_owners_have_no_time_limits"]
            is True
        ), contract_name
        assert (
            policy["master_all_sub_agent_and_worker_runtime_loop_owners_have_no_round_limits"]
            is True
        ), contract_name
        assert (
            policy[
                "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_time_or_round_limits"
            ]
            is True
        ), contract_name
        for key in (
            "every_runtime_loop_owner_has_no_turn_limit",
            "every_runtime_loop_owner_has_no_max_turns_limit",
            "every_runtime_loop_owner_has_no_attempt_limit",
            "every_runtime_loop_owner_has_no_timeout_limit",
            "every_runtime_loop_owner_has_no_cap_limit",
            "every_runtime_loop_owner_has_no_count_limit",
            "every_runtime_loop_owner_has_no_token_limit",
            "every_runtime_loop_owner_has_no_budget_limit",
            "every_runtime_loop_owner_has_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limit",
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_turn_limits",
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_max_turns_limits",
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_attempt_limits",
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_timeout_limits",
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_cap_limits",
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_count_limits",
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_token_limits",
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_budget_limits",
            "master_sub_agent_and_worker_runtime_loop_owners_have_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limits",
            "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_turn_limits",
            "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_max_turns_limits",
            "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_attempt_limits",
            "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_timeout_limits",
            "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_cap_limits",
            "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_count_limits",
            "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_token_limits",
            "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_budget_limits",
            "master_all_sub_agent_and_worker_runtime_loop_owners_have_no_time_round_turn_attempt_timeout_cap_count_token_or_budget_limits",
        ):
            assert policy[key] is True, f"{contract_name}.{key}"
        assert policy["no_owner_has_time_limit"] is True, contract_name
        assert policy["no_owner_has_round_limit"] is True, contract_name
        assert policy["no_owner_has_time_or_round_limit"] is True, contract_name
        assert policy["no_owner_has_terminal_limit_predicate"] is True, contract_name
        assert policy["no_owner_has_limit_driven_completion"] is True, contract_name
        assert policy["no_owner_has_limit_driven_stop"] is True, contract_name
        assert policy["no_owner_has_limit_driven_skip"] is True, contract_name
        assert policy["no_owner_has_limit_driven_truncation"] is True, contract_name
        assert policy["no_owner_has_limit_driven_kill"] is True, contract_name
        assert policy["no_owner_has_limit_driven_cancellation"] is True, contract_name
        assert policy["no_owner_has_retry_threshold_stop"] is True, contract_name
        assert policy["no_owner_has_direct_replay_threshold_stop"] is True, contract_name
        assert policy["no_owner_has_retry_threshold_retry_denial"] is True, contract_name
        assert (
            policy["no_owner_has_direct_replay_threshold_retry_denial"] is True
        ), contract_name
        assert policy["time_round_limit_stop_condition"] is False, contract_name
        assert policy["time_or_round_limit_stop_condition"] is False, contract_name
        assert policy["terminal_limit_predicate_stop_condition"] is False, contract_name
        assert policy["limit_driven_completion_stop_condition"] is False, contract_name
        assert policy["limit_driven_stop_condition"] is False, contract_name
        assert policy["limit_driven_skip_stop_condition"] is False, contract_name
        assert policy["limit_driven_truncation_stop_condition"] is False, contract_name
        assert policy["limit_driven_kill_stop_condition"] is False, contract_name
        assert policy["limit_driven_cancellation_stop_condition"] is False, contract_name
        assert policy["retry_threshold_loop_stop_condition"] is False, contract_name
        assert policy["direct_replay_threshold_stop_condition"] is False, contract_name
        assert policy["retry_threshold_denies_retry"] is False, contract_name
        assert policy["direct_replay_threshold_denies_retry"] is False, contract_name
        assert set(policy["forbidden_terminal_predicates"]) == expected_predicates
        assert set(policy["forbidden_limit_driven_actions"]) == expected_forbidden_actions
        assert set(policy["audit_only_fields"]) == expected_audit_fields

        for owner in expected_owners:
            owner_policy = contract[owner]["loop_owner_policy"]
            assert owner_policy["owner"] == owner, f"{contract_name}.{owner}"
            assert owner_policy["runtime_loop_owner"] is True
            assert owner_policy["listed_in_owner_policy"] is True
            assert owner_policy["listed_in_runtime_loop_owners"] is True
            assert owner_policy["runtime_owner_has_no_time_limit"] is True
            assert owner_policy["runtime_owner_has_no_round_limit"] is True
            assert owner_policy["runtime_owner_has_no_time_or_round_limit"] is True
            assert owner_policy["master_sub_agent_worker_owner_family_has_no_time_limit"] is True
            assert owner_policy["master_sub_agent_worker_owner_family_has_no_round_limit"] is True
            assert (
                owner_policy["master_sub_agent_worker_owner_family_has_no_time_or_round_limit"]
                is True
            )
            assert owner_policy["no_time_limit"] is True
            assert owner_policy["no_round_limit"] is True
            assert owner_policy["no_time_or_round_limit"] is True
            assert owner_policy["no_terminal_limit_predicate"] is True
            assert owner_policy["no_limit_driven_completion"] is True
            assert owner_policy["no_limit_driven_stop"] is True
            assert owner_policy["no_limit_driven_skip"] is True
            assert owner_policy["no_limit_driven_truncation"] is True
            assert owner_policy["no_limit_driven_kill"] is True
            assert owner_policy["no_limit_driven_cancellation"] is True
            assert owner_policy["no_retry_threshold_stop"] is True
            assert owner_policy["no_direct_replay_threshold_stop"] is True
            assert owner_policy["no_retry_threshold_retry_denial"] is True
            assert owner_policy["no_direct_replay_threshold_retry_denial"] is True
            assert owner_policy["time_round_limit_stop_condition"] is False
            assert owner_policy["time_or_round_limit_stop_condition"] is False
            assert owner_policy["terminal_limit_predicate_stop_condition"] is False
            assert owner_policy["limit_driven_completion_stop_condition"] is False
            assert owner_policy["limit_driven_stop_condition"] is False
            assert owner_policy["limit_driven_skip_stop_condition"] is False
            assert owner_policy["limit_driven_truncation_stop_condition"] is False
            assert owner_policy["limit_driven_kill_stop_condition"] is False
            assert owner_policy["limit_driven_cancellation_stop_condition"] is False
            assert owner_policy["retry_threshold_loop_stop_condition"] is False
            assert owner_policy["direct_replay_threshold_stop_condition"] is False
            assert owner_policy["retry_threshold_denies_retry"] is False
            assert owner_policy["direct_replay_threshold_denies_retry"] is False
            assert set(owner_policy["forbidden_terminal_predicates"]) == expected_predicates
            assert (
                set(owner_policy["forbidden_limit_driven_actions"])
                == expected_forbidden_actions
            )
            assert set(owner_policy["audit_only_fields"]) == expected_audit_fields


def test_loop_owner_sources_do_not_reintroduce_limit_control_flow() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    campaign_source = (repo_root / "scripts" / "run_campaign.py").read_text()
    hl_loop_source = (repo_root / "hl" / "loop.py").read_text()
    codex_source = (repo_root / "meta" / "codex_update.py").read_text()
    packager_source = (repo_root / "meta" / "packager.py").read_text()
    missions_source = (repo_root / "meta" / "missions.py").read_text()
    regression_source = (repo_root / "scripts" / "regression_check.py").read_text()
    agent_source = "\n".join(
        (repo_root / path).read_text()
        for path in ("bench/agent.py", "bench/_agent_bridge.py")
    )
    harbor_source = "\n".join(
        (repo_root / path).read_text()
        for path in (
            "bench/harbor.py",
            "bench/_harbor_issue5_logic.py",
            "bench/_harbor_issue9_base.py",
        )
    )
    harbor_adapter_source = "\n".join(
        (repo_root / path).read_text()
        for path in (
            "bench/harbor_adapter.py",
            "bench/_harbor_adapter_issue16_base.py",
        )
    )
    retry_source = "\n".join(
        (repo_root / path).read_text()
        for path in (
            "harness/recovery/retry.py",
            "harness/recovery/_retry_issue20_base.py",
        )
    )
    task_catalog_source = (repo_root / "bench" / "tasks.py").read_text()
    context_source = "\n".join(
        [
            (repo_root / "harness" / "context" / "isolation.py").read_text(),
            (repo_root / "harness" / "context" / "trajectory_pack.py").read_text(),
        ]
    )
    worker_source = (
        repo_root / "crates" / "hl-worker-core" / "src" / "main.rs"
    ).read_text()

    worker_loop = worker_source[
        worker_source.index("fn run_worker"):
        worker_source.index("fn initialize_worker_messages")
    ]
    campaign_master_loop = campaign_source[
        campaign_source.index("    while True:"):
        campaign_source.index("    report = _build_campaign_report", 1)
    ]
    task_selection_source = campaign_source[
        campaign_source.index("def _resolve_tasks"):
        campaign_source.index("def _resolve_regression_lane")
    ]
    codex_run_source = codex_source
    mission_candidate_source = missions_source
    regression_run_source = regression_source

    forbidden_by_scope = {
        "campaign_master_loop": (
            "while loop.iteration <",
            "range(iteration_limit)",
            "loop.iteration >= iteration_limit",
            "loop.iteration == iteration_limit",
            "if not codex_update_enabled:\n            break",
            "if not codex_update_enabled:\n            return",
            "cooldown_remaining",
            "timeout=args.timeout",
            "future.result(timeout=",
            "process.communicate(timeout=",
        ),
        "task_selection": (
            "task_ids[:max_tasks]",
            "deduped = deduped[:max_tasks]",
            "pending_tasks[:run_task_cap]",
            "batch_size = run_task_cap",
            "return selected[:",
        ),
        "hl_loop": (
            "self.iteration >= self.max_iterations",
            "self.iterations_without_improvement >= self.patience",
            "future.result(timeout=",
            "if should_stop:\n                    break",
        ),
        "codex_update_sub_agent": (
            "timeout=self.timeout_seconds",
            "timeout=self.validation_timeout_seconds",
            "process.communicate(timeout=",
            "future.result(timeout=",
            "time.time() - started_at > self.timeout_seconds",
        ),
        "codex_work_packet_failure_evidence": (
            "trial.trajectory[:",
            "trial.trajectory[-",
            "trial.verifier_output[:",
            "trial.error_log[:",
            "if len(trial.trajectory) <=",
            '"type": "omitted", "count": len(trial.trajectory)',
            "if call.get(\"success\") is False\n                    ][:",
            "call.get(\"success\") is False][:",
        ),
        "diagnostic_context_sub_agents": (
            "candidates[:max_features",
            "feature_candidates[:max_features",
            "return tasks[:limit]",
            "most_common(limit)",
            "if self.max_sub_agent_depth is not None",
            "events[:",
            "events[-",
            "value[:",
        ),
        "mission_debug_sub_agent": (
            "candidates[:max_features",
            "feature_candidates[:max_features",
            "return tasks[:limit]",
            "most_common(limit)",
        ),
        "validation_regression_sub_agent": (
            "process.communicate(timeout=",
            "future.result(timeout=",
            "as_completed(futures, timeout=",
            "wait(futures, timeout=",
            "timeout=args.timeout",
            "return snapshots[:cap]",
            "snapshots[:cap]",
        ),
        "worker_python": (
            "process.wait(timeout=",
            '"max_turns": self.max_turns',
        ),
        "worker_rust_loop": (
            "while state.turn_count <",
            "state.turn_count >= state.max_turns",
            "state.turn_count >= state.max_turns_audit",
            "state.turn_count == state.max_turns",
            "state.turn_count == state.max_turns_audit",
            "round_limit",
            "timeout_seconds",
            "deadline",
        ),
        "harbor_host": (
            "process.communicate(timeout=timeout)",
            "while attempt_index <",
            "attempt_index >= infra_retry_reference",
            "time.sleep(retry_delay_seconds)",
        ),
        "harbor_adapter": (
            "future.result(timeout=",
            "process.communicate(timeout=",
        ),
        "retry_strategy": (
            "loop_stop_condition: bool = True",
            "master_loop_stop_condition: bool = True",
            "sub_agent_loop_stop_condition: bool = True",
            "worker_loop_stop_condition: bool = True",
            "attempt_count_stop_condition: bool = True",
            "retry_limit_stop_condition: bool = True",
            "direct_replay_threshold_stop_condition: bool = True",
            "retry_threshold_loop_stop_condition: bool = True",
            "retry_threshold_denies_retry: bool = True",
            "owning_loop_continues: bool = False",
            "This is not a loop stop condition",
        ),
        "task_catalog": (
            "task_ids[:max_tasks]",
            "len(selected) < limit",
            "def _cap(",
        ),
    }
    sources = {
        "campaign_master_loop": campaign_master_loop,
        "task_selection": task_selection_source,
        "hl_loop": hl_loop_source,
        "codex_update_sub_agent": codex_run_source,
        "codex_work_packet_failure_evidence": packager_source,
        "diagnostic_context_sub_agents": missions_source + "\n" + context_source,
        "mission_debug_sub_agent": mission_candidate_source,
        "validation_regression_sub_agent": regression_run_source,
        "worker_python": agent_source,
        "worker_rust_loop": worker_loop,
        "harbor_host": harbor_source,
        "harbor_adapter": harbor_adapter_source,
        "retry_strategy": retry_source,
        "task_catalog": task_catalog_source,
    }

    for scope, snippets in forbidden_by_scope.items():
        source = sources[scope]
        for snippet in snippets:
            assert snippet not in source, f"{scope} reintroduced {snippet!r}"

    assert "while True:" in campaign_master_loop
    assert "loop.max_iterations = None" in campaign_source
    assert "loop.patience = None" in campaign_source
    assert "loop {" in worker_loop
    assert "done_tool_requested" in worker_loop
    assert "completion_gate_passed" in worker_loop
    assert "This is not a master, sub-agent, or Worker loop stop " in retry_source
    assert "condition; continue solving" in retry_source
    assert "continue the owning loop" in retry_source


def test_master_and_sub_agent_limit_fields_are_runtime_noops() -> None:
    loop = HLLoop()
    loop.max_iterations = 1
    loop.patience = 1
    loop.iteration = 10_000
    loop.iterations_without_improvement = 10_000

    assert loop.should_continue() is True

    policy = {
        "interval": 99,
        "min_failures": 99,
        "cooldown_after_rollback": 99,
        "partial_pass_diagnostic_k": 99,
    }
    args = SimpleNamespace(codex_update=True)

    for next_iteration in (0, 1, 2, 99, 10_000):
        assert (
            _codex_update_should_run(
                args,
                policy,
                next_iteration=next_iteration,
                cooldown_audit=99,
            )
            is True
        )

    disabled_args = SimpleNamespace(codex_update=False)
    assert (
        _codex_update_should_run(
            disabled_args,
            policy,
            next_iteration=10_000,
            cooldown_audit=99,
        )
        is False
    )


def test_master_sub_agent_and_worker_loops_ignore_extreme_limit_metadata(tmp_path) -> None:
    from scripts.regression_check import _select_regression_snapshots

    extreme_args = SimpleNamespace(
        patience=0,
        max_tasks=0,
        run_task_cap=0,
        random_count=0,
        max_turns_audit=0,
        llm_timeout_seconds=0,
        tool_timeout_seconds=0,
        mission_debug_max_features=0,
    )
    extreme_update_policy = {
        "interval": 10**9,
        "min_failures": 10**9,
        "cooldown_after_rollback": 10**9,
        "partial_pass_diagnostic_k": 0,
    }
    contracts = {
        "campaign": _loop_limit_contract(
            iteration_limit=0,
            args=extreme_args,
            trials_config={
                "tasks": {"max_tasks_per_trial": 0, "random_count": 0},
            },
            codex_config={"timeout_seconds": 0},
            update_policy=extreme_update_policy,
            goal_plan={
                "goal": {
                    "token_budget": 0,
                    "token_budget_scope": "campaign",
                    "wall_time_budget_seconds": 0,
                }
            },
        ),
        "codex_update": CodexUpdateEngine(
            repo_root=tmp_path,
            events_dir=tmp_path / "diffs",
            timeout_seconds=0,
            validation_timeout_seconds=0,
        )._loop_limit_contract(),
        "mission_debug": MissionPlanner()._loop_limit_contract(max_features=0),
    }
    owner_scopes = (
        "master_loop",
        "codex_update_sub_agent",
        "diagnostic_sub_agents",
        "context_sub_agents",
        "validation_regression_sub_agents",
        "mission_debug_sub_agent",
        "worker_task_loop",
    )
    owner_no_limit_keys = (
        "loop_owner_has_no_time_limit",
        "loop_owner_has_no_round_limit",
        "loop_owner_has_no_attempt_limit",
        "loop_owner_has_no_timeout_limit",
        "loop_owner_has_no_cap_limit",
        "loop_owner_has_no_count_limit",
        "loop_owner_has_no_token_limit",
        "loop_owner_has_no_budget_limit",
        "loop_owner_has_no_limit_driven_completion",
        "loop_owner_has_no_limit_driven_stop",
        "loop_owner_has_no_limit_driven_skip",
        "loop_owner_has_no_limit_driven_truncation",
        "loop_owner_has_no_time_round_attempt_timeout_cap_count_token_or_budget_limit",
    )
    owner_limit_allowed_keys = (
        "time_limit_allowed",
        "round_limit_allowed",
        "attempt_limit_allowed",
        "timeout_limit_allowed",
        "cap_limit_allowed",
        "count_limit_allowed",
        "token_limit_allowed",
        "budget_limit_allowed",
    )

    for contract_name, contract in contracts.items():
        _assert_no_stop_conditions(contract)
        assert (
            contract["all_loops"][
                "master_and_all_sub_agent_and_worker_loops_unbounded_by_time_round_turn_attempt_timeout_cap_count_token_and_budget"
            ]
            is True
        ), contract_name
        assert (
            contract["all_loops"][
                "master_and_all_sub_agent_and_worker_loop_owners_have_no_time_round_attempt_timeout_cap_count_token_or_budget_limits"
            ]
            is True
        ), contract_name
        for scope_name in owner_scopes:
            scope = contract[scope_name]
            for key in owner_no_limit_keys:
                assert scope[key] is True, f"{contract_name}.{scope_name}.{key}"
            for key in owner_limit_allowed_keys:
                assert scope[key] is False, f"{contract_name}.{scope_name}.{key}"

    loop = HLLoop()
    loop.max_iterations = 0
    loop.patience = 0
    loop.iteration = 10**9
    loop.iterations_without_improvement = 10**9
    assert loop.should_continue() is True

    assert _codex_update_should_run(
        SimpleNamespace(codex_update=True),
        extreme_update_policy,
        next_iteration=10**9,
        cooldown_audit=10**9,
    ) is True

    snapshots = [SimpleNamespace(task_id=f"task-{index}") for index in range(5)]
    assert _select_regression_snapshots(
        snapshots,
        SimpleNamespace(selection_policy="stable-order", cap=0),
    ) == snapshots
