import json
import subprocess
import sys
import tomllib
from pathlib import Path

from hl.loop_limits import unbounded_scope_flags
from hl.types import TaskDifficulty, TaskDomain, TrialResult, TrialStatus
from meta.missions import MissionPlanner
from meta.packager import WorkPacketBuilder


def campaign_summary():
    return {
        "campaign_id": "mission-smoke",
        "task_results": [
            {
                "trial_id": "fix-git__ok",
                "task_id": "fix-git",
                "domain": "software_engineering",
                "difficulty": "easy",
                "status": "passed",
                "score": 1.0,
                "verified": True,
            },
            {
                "trial_id": "build-big__timeout",
                "task_id": "build-big",
                "domain": "software_engineering",
                "difficulty": "hard",
                "status": "timeout",
                "score": 0.0,
                "verified": False,
            },
            {
                "trial_id": "env-broken__error",
                "task_id": "env-broken",
                "domain": "system_administration",
                "difficulty": "medium",
                "status": "error",
                "score": 0.0,
                "verified": False,
            },
            {
                "trial_id": "verifier-fail__failed",
                "task_id": "verifier-fail",
                "domain": "security",
                "difficulty": "medium",
                "status": "failed",
                "score": 0.0,
                "verified": True,
            },
        ],
        "score_history": [
            {
                "score": 0.25,
                "passed": 1,
                "failed": 1,
                "timeout": 1,
                "error": 1,
            }
        ],
        "patch_lineage": [{"iteration": 1}],
        "reproducibility": {"git_commit": "abc"},
        "tasks_completed": 4,
        "tasks_pending": 0,
    }


def _assert_no_stop_conditions(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.endswith("_stop_condition"):
                assert nested is False, f"{key} must remain audit-only"
            _assert_no_stop_conditions(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_stop_conditions(nested)


def _assert_shared_unbounded_flags(contract):
    for scope_name in (
        "all_loops",
        "master_loop",
        "codex_update_sub_agent",
        "diagnostic_sub_agents",
        "context_sub_agents",
        "validation_regression_sub_agents",
        "mission_debug_sub_agent",
        "worker_task_loop",
        "goal_budgets",
    ):
        scope = contract[scope_name]
        for key, expected in unbounded_scope_flags(scope_name).items():
            assert scope[key] is expected, f"{scope_name}.{key} drifted"


def _assert_no_time_or_round_limits(contract):
    assert contract["all_loops"]["all_agent_loops_have_no_time_or_round_limits"] is True
    assert contract["all_loops"]["all_sub_agent_loops_have_no_time_or_round_limits"] is True
    assert contract["all_loops"]["all_sub_agent_loops_unbounded_by_time"] is True
    assert contract["all_loops"]["all_sub_agent_loops_unbounded_by_round"] is True
    assert (
        contract["all_loops"][
            "all_sub_agent_loops_unbounded_by_time_and_round"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "all_sub_agent_loops_unbounded_by_time_round_and_attempt"
        ]
        is True
    )
    assert contract["all_loops"]["all_loop_owners_have_no_time_or_round_limits"] is True
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
    for scope_name in (
        "master_loop",
        "codex_update_sub_agent",
        "diagnostic_sub_agents",
        "context_sub_agents",
        "validation_regression_sub_agents",
        "mission_debug_sub_agent",
        "worker_task_loop",
        "goal_budgets",
    ):
        scope = contract[scope_name]
        assert scope["no_time_or_round_limits"] is True, scope_name
        assert scope["no_time_limit"] is True, scope_name
        assert scope["no_round_limit"] is True, scope_name
        if scope_name != "goal_budgets":
            assert scope["loop_has_no_time_limit"] is True, scope_name
            assert scope["loop_has_no_round_limit"] is True, scope_name
            assert scope["loop_has_no_time_or_round_limit"] is True, scope_name
            assert scope["loop_owner_has_no_time_limit"] is True, scope_name
            assert scope["loop_owner_has_no_round_limit"] is True, scope_name
            assert scope["loop_owner_has_no_time_or_round_limit"] is True, scope_name
            assert scope["time_limit_allowed"] is False, scope_name
            assert scope["round_limit_allowed"] is False, scope_name
            assert scope["time_or_round_limit_allowed"] is False, scope_name
        for key in (
            "time_and_round_limits_stop_condition",
            "time_round_limit_stop_condition",
            "time_limit_stop_condition",
            "round_limit_stop_condition",
        ):
            if key in scope:
                assert scope[key] is False, f"{scope_name}.{key}"


def test_mission_planner_creates_validation_first_debug_packet():
    packet = MissionPlanner().from_campaign_summary(
        campaign_summary(),
        source_path="trials/summaries/mission-smoke_campaign.json",
    )

    contract_ids = [contract.id for contract in packet.validation_contracts]
    feature_ids = [feature.id for feature in packet.feature_candidates]

    assert contract_ids[0] == "contract-readiness-audit"
    assert "contract-unit-suite" in contract_ids
    assert "contract-solved-task-protection" in contract_ids
    assert "mission-harbor-error-attribution" in feature_ids
    assert "mission-timeout-recovery-policy" in feature_ids
    assert "mission-verified-failure-learning" in feature_ids
    verified_failure = next(
        feature
        for feature in packet.feature_candidates
        if feature.id == "mission-verified-failure-learning"
    )
    assert "crates/hl-worker-core/src/main.rs" in verified_failure.affected_components
    assert "crates" in verified_failure.allowed_edit_paths
    assert "bench/agent.py" not in verified_failure.affected_components
    assert "harness/prompts/task.py" not in verified_failure.affected_components
    assert any("terminal-bench-tasks" in action for action in packet.blocked_actions)
    assert any(control["name"] == "adjust_worker_role" for control in packet.external_loop_controls)
    assert packet.evidence_summary["task_results"] == 4
    assert packet.evidence_summary["has_reproducibility"] is True


def test_mission_debug_max_features_is_audit_only_and_does_not_truncate_candidates():
    packet = MissionPlanner().from_campaign_summary(campaign_summary(), max_features=1)

    feature_ids = [feature.id for feature in packet.feature_candidates]

    assert len(feature_ids) > 1
    assert "mission-harbor-error-attribution" in feature_ids
    assert "mission-timeout-recovery-policy" in feature_ids
    assert "mission-verified-failure-learning" in feature_ids
    assert "mission-regression-contract-hardening" in feature_ids
    assert packet.evidence_summary["max_features_audit_only"] == 1
    assert packet.evidence_summary["max_features_stop_condition"] is False
    assert packet.loop_limit_contract["mission_debug_sub_agent"]["max_features_audit_only"] == 1
    assert packet.loop_limit_contract["mission_debug_sub_agent"]["max_features_stop_condition"] is False
    _assert_no_stop_conditions(packet.model_dump())


def test_mission_debug_packet_carries_loop_limit_contract():
    packet = MissionPlanner().from_campaign_summary(campaign_summary(), max_features=0)

    contract = packet.loop_limit_contract
    _assert_no_stop_conditions(contract)
    _assert_shared_unbounded_flags(contract)
    _assert_no_time_or_round_limits(contract)
    assert contract["all_loops"]["master_loop_unbounded_by_time_and_round"] is True
    assert contract["all_loops"]["master_loop_unbounded_by_time_round_and_attempt"] is True
    assert (
        contract["all_loops"]["mission_debug_sub_agent_unbounded_by_time_and_round"]
        is True
    )
    assert (
        contract["all_loops"][
            "mission_debug_sub_agent_unbounded_by_time_round_and_attempt"
        ]
        is True
    )
    assert (
        contract["all_loops"][
            "mission_debug_sub_agent_unbounded_by_time_round_attempt_and_feature_count"
        ]
        is True
    )
    assert contract["all_loops"]["codex_update_sub_agent_unbounded_by_time_and_round"] is True
    assert contract["all_loops"]["diagnostic_sub_agents_unbounded_by_time_and_round"] is True
    assert contract["all_loops"]["context_sub_agents_unbounded_by_time_and_round"] is True
    assert (
        contract["all_loops"][
            "validation_regression_sub_agents_unbounded_by_time_round_and_attempt"
        ]
        is True
    )
    assert contract["all_loops"]["worker_task_loop_unbounded_by_time_and_round"] is True
    assert contract["master_loop"]["unbounded_by_time_round_and_attempt"] is True
    assert contract["master_loop"]["iteration_limit_stop_condition"] is False
    assert contract["master_loop"]["budget_exhaustion_stop_condition"] is False
    assert contract["codex_update_sub_agent"]["unbounded_by_budget_attempt_and_cooldown"] is True
    assert contract["codex_update_sub_agent"]["interval_stop_condition"] is False
    assert contract["codex_update_sub_agent"]["cooldown_stop_condition"] is False
    assert contract["codex_update_sub_agent"]["sub_agent_attempt_limit_stop_condition"] is False
    assert contract["diagnostic_sub_agents"]["unbounded_by_time_round_attempt_and_k"] is True
    assert contract["diagnostic_sub_agents"]["diagnostic_target_k_stop_condition"] is False
    assert contract["diagnostic_sub_agents"]["sub_agent_attempt_count_stop_condition"] is False
    assert contract["context_sub_agents"]["unbounded_by_depth_and_tokens"] is True
    assert contract["context_sub_agents"]["depth_stop_condition"] is False
    validation = contract["validation_regression_sub_agents"]
    assert validation["unbounded_by_time_round_and_attempt"] is True
    assert validation["regression_snapshot_count_stop_condition"] is False
    assert validation["validation_timeout_stop_condition"] is False
    assert contract["worker_task_loop"]["unbounded_by_time_round_turn_and_attempt"] is True
    assert contract["worker_task_loop"]["max_turns_stop_condition"] is False
    assert contract["worker_task_loop"]["timeout_phase_count_stop_condition"] is False
    assert contract["goal_budgets"]["token_budget_stop_condition"] is False
    assert (
        contract["mission_debug_sub_agent"][
            "unbounded_by_time_round_attempt_and_feature_count"
        ]
        is True
    )
    assert (
        contract["mission_debug_sub_agent"]["unbounded_by_time_round_and_attempt"]
        is True
    )
    assert contract["mission_debug_sub_agent"]["max_features_audit_only"] == 0
    assert contract["mission_debug_sub_agent"]["max_features_stop_condition"] is False
    assert contract["mission_debug_sub_agent"]["attempt_limit_stop_condition"] is False
    assert (
        contract["mission_debug_sub_agent"]["sub_agent_attempt_limit_stop_condition"]
        is False
    )
    assert "must not stop master" in contract["all_loops"]["note"]
    assert any("elapsed time" in note for note in packet.architecture_notes)
    assert any(
        "feature-count stops" in control["description"]
        for control in packet.external_loop_controls
    )


def test_mission_debug_solved_task_contract_does_not_limit_target_tasks():
    summary = campaign_summary()
    summary["task_results"] = [
        {
            "trial_id": f"passed-{index}",
            "task_id": f"passed-task-{index:02d}",
            "domain": "software_engineering",
            "difficulty": "medium",
            "status": "passed",
            "score": 1.0,
            "verified": True,
        }
        for index in range(7)
    ]

    packet = MissionPlanner().from_campaign_summary(summary, max_features=1)

    solved_contract = next(
        contract
        for contract in packet.validation_contracts
        if contract.id == "contract-solved-task-protection"
    )
    regression_candidate = next(
        feature
        for feature in packet.feature_candidates
        if feature.id == "mission-regression-contract-hardening"
    )
    assert regression_candidate.target_tasks == [
        f"passed-task-{index:02d}" for index in range(7)
    ]
    assert "every observed passed task" in solved_contract.description
    assert "All observed solved-task regressions" in solved_contract.pass_condition
    assert "target-task" in solved_contract.pass_condition
    for index in range(7):
        assert f"passed-task-{index:02d}" in solved_contract.commands[0]
    assert solved_contract.commands[0].count("--task") == 7
    assert packet.loop_limit_contract["mission_debug_sub_agent"][
        "target_task_count_stop_condition"
    ] is False
    assert packet.loop_limit_contract["mission_debug_sub_agent"][
        "validation_contract_count_stop_condition"
    ] is False


def test_mission_planner_prefers_attributed_failure_buckets():
    summary = campaign_summary()
    summary["task_results"][1].update(
        {
            "failure_category": "environment_start_timeout",
            "affected_components": ["bench/harbor", "bench/network_environment"],
            "timeout_phase": "environment_start",
            "infra_error_detected": True,
            "score_exclusion_reason": "infrastructure_error",
        }
    )
    summary["task_results"][2].update(
        {
            "failure_category": "harbor_environment_error",
            "affected_components": ["bench/harbor", "bench/network_environment"],
            "infra_error_detected": True,
            "score_exclusion_reason": "infrastructure_error",
        }
    )
    summary["task_results"][3].update(
        {
            "failure_category": "verifier_mismatch",
            "affected_components": ["verification/checks", "harness/tools/verify"],
        }
    )

    packet = MissionPlanner().from_campaign_summary(summary)

    feature_ids = [feature.id for feature in packet.feature_candidates]
    assert feature_ids[:3] == [
        "mission-attributed-environment-start-timeout",
        "mission-attributed-harbor-environment-error",
        "mission-attributed-verifier-mismatch",
    ]
    first = packet.feature_candidates[0]
    assert first.target_tasks == ["build-big"]
    assert first.affected_components == ["bench/harbor", "bench/network_environment"]
    assert first.allowed_edit_paths == ["bench", "tests"]
    assert first.priority == "P1"
    verifier = packet.feature_candidates[2]
    assert verifier.allowed_edit_paths == ["harness", "tests"]
    assert "status-only" in verifier.rationale


def test_mission_planner_maps_logical_components_to_edit_roots():
    summary = campaign_summary()
    summary["task_results"] = [
        {
            "trial_id": "entrypoint-miss",
            "task_id": "task-a",
            "domain": "software_engineering",
            "difficulty": "easy",
            "status": "failed",
            "score": 0.0,
            "verified": True,
            "failure_category": "entrypoint_miss",
            "affected_components": ["entrypoint/semantic", "tools/file_read"],
        }
    ]

    packet = MissionPlanner().from_campaign_summary(summary)

    candidate = packet.feature_candidates[0]
    assert candidate.id == "mission-attributed-entrypoint-miss"
    assert candidate.allowed_edit_paths == ["bench", "harness", "crates", "tests"]


def test_mission_planner_splits_attributed_candidates_by_failure_mechanism():
    summary = campaign_summary()
    summary["task_results"] = [
        {
            "trial_id": "cython-timeout",
            "task_id": "build-cython-ext",
            "domain": "software_engineering",
            "difficulty": "hard",
            "status": "timeout",
            "score": 0.0,
            "verified": True,
            "failure_category": "terminal_environment_unavailable_after_dependency_loop",
            "affected_components": [
                "bench/agent",
                "harness/tools/shell",
                "recovery/patterns",
            ],
            "failure_mechanisms": [
                {"name": "cython_extension_optional_import_pivot_mechanism"}
            ],
        },
        {
            "trial_id": "artifact-timeout",
            "task_id": "db-wal-recovery",
            "domain": "software_engineering",
            "difficulty": "hard",
            "status": "timeout",
            "score": 0.0,
            "verified": True,
            "failure_category": "terminal_environment_unavailable_after_dependency_loop",
            "affected_components": [
                "bench/agent",
                "harness/tools/verify",
                "verification/checks",
            ],
            "failure_mechanisms": [
                {"name": "missing_output_artifact_contract"}
            ],
        },
    ]

    packet = MissionPlanner().from_campaign_summary(summary, max_features=1)

    feature_ids = [feature.id for feature in packet.feature_candidates]
    assert feature_ids == [
        "mission-attributed-terminal-environment-unavailable-after-dependency-loop-cython-extension-optional-import-pivot-mechanism",
        "mission-attributed-terminal-environment-unavailable-after-dependency-loop-missing-output-artifact-contract",
    ]
    cython = packet.feature_candidates[0]
    assert cython.target_tasks == ["build-cython-ext"]
    assert "harness/tools/shell" in cython.affected_components
    assert "specific root mechanism" in cython.rationale
    artifact = packet.feature_candidates[1]
    assert artifact.target_tasks == ["db-wal-recovery"]
    assert "harness/tools/verify" in artifact.affected_components
    assert artifact.allowed_edit_paths == ["bench", "harness", "tests"]
    assert packet.evidence_summary["max_features_stop_condition"] is False
    _assert_no_stop_conditions(packet.model_dump())


def test_mission_planner_treats_max_features_as_audit_only():
    packet = MissionPlanner().from_campaign_summary(campaign_summary(), max_features=1)

    feature_ids = [feature.id for feature in packet.feature_candidates]
    assert feature_ids == [
        "mission-harbor-error-attribution",
        "mission-timeout-recovery-policy",
        "mission-verified-failure-learning",
        "mission-regression-contract-hardening",
    ]
    assert packet.evidence_summary["max_features_audit_only"] == 1
    assert packet.evidence_summary["max_features_stop_condition"] is False


def test_mission_planner_does_not_cap_candidate_target_tasks_by_k_or_feature_limit():
    summary = campaign_summary()
    timeout_tasks = [
        {
            "trial_id": f"timeout-{index}",
            "task_id": f"timeout-task-{index:02d}",
            "domain": "software_engineering",
            "difficulty": "medium",
            "status": "timeout",
            "score": 0.0,
            "verified": False,
        }
        for index in range(24)
    ]
    summary["task_results"] = [summary["task_results"][0], *timeout_tasks]

    packet = MissionPlanner().from_campaign_summary(summary, max_features=1)

    timeout_candidate = next(
        feature
        for feature in packet.feature_candidates
        if feature.id == "mission-timeout-recovery-policy"
    )
    assert timeout_candidate.target_tasks == [
        f"timeout-task-{index:02d}" for index in range(24)
    ]
    assert packet.evidence_summary["max_features_audit_only"] == 1
    assert packet.evidence_summary["max_features_stop_condition"] is False


def test_mission_task_status_helper_treats_limit_as_audit_only():
    summary = campaign_summary()
    summary["task_results"] = [
        {
            "trial_id": f"timeout-{index}",
            "task_id": f"timeout-task-{index:02d}",
            "domain": "software_engineering",
            "difficulty": "medium",
            "status": "timeout",
            "score": 0.0,
            "verified": False,
        }
        for index in range(5)
    ]

    tasks = MissionPlanner()._tasks_with_status(
        summary["task_results"],
        {"timeout"},
        limit=1,
    )

    assert tasks == [f"timeout-task-{index:02d}" for index in range(5)]


def test_mission_top_value_summary_treats_limit_as_audit_only():
    summary = campaign_summary()
    summary["task_results"] = [
        {
            "trial_id": f"domain-{index}",
            "task_id": f"domain-task-{index:02d}",
            "domain": f"domain_{index:02d}",
            "difficulty": f"difficulty_{index:02d}",
            "status": "failed",
            "score": 0.0,
            "verified": True,
        }
        for index in range(8)
    ]

    packet = MissionPlanner().from_campaign_summary(summary, max_features=1)

    assert list(packet.evidence_summary["top_domains"]) == [
        f"domain_{index:02d}" for index in range(8)
    ]
    assert list(packet.evidence_summary["top_difficulties"]) == [
        f"difficulty_{index:02d}" for index in range(8)
    ]
    assert packet.evidence_summary["max_features_stop_condition"] is False


def test_mission_planner_does_not_cap_attributed_bucket_target_tasks():
    summary = campaign_summary()
    summary["task_results"] = [
        {
            "trial_id": f"verifier-{index}",
            "task_id": f"verifier-task-{index:02d}",
            "domain": "security",
            "difficulty": "medium",
            "status": "failed",
            "score": 0.0,
            "verified": True,
            "failure_category": "verifier_mismatch",
            "affected_components": ["crates/hl-worker-core/src/main.rs"],
        }
        for index in range(21)
    ]

    packet = MissionPlanner().from_campaign_summary(summary, max_features=1)

    candidate = packet.feature_candidates[0]
    assert candidate.id == "mission-attributed-verifier-mismatch"
    assert candidate.target_tasks == [
        f"verifier-task-{index:02d}" for index in range(21)
    ]
    assert packet.evidence_summary["max_features_stop_condition"] is False


def test_work_packet_embeds_mission_debug_without_benchmark_delegate():
    trial = TrialResult(
        trial_id="t-timeout",
        task_id="task-timeout",
        task_domain=TaskDomain.SOFTWARE_ENGINEERING,
        task_difficulty=TaskDifficulty.MEDIUM,
        status=TrialStatus.TIMEOUT,
        score=0.0,
        verified=False,
    )

    packet = WorkPacketBuilder().build(
        failures=[trial],
        current_harness={"version": "x"},
    )

    mission = packet.mission_debug
    assert mission["feature_candidates"][0]["id"] == "mission-timeout-recovery-policy"
    assert any("Do not delegate benchmark task execution" in action for action in mission["blocked_actions"])
    assert packet.sub_agent_creation_policy["allowed_creator_owner"] == "master_loop"
    assert packet.sub_agent_creation_policy["nested_sub_agent_creation_allowed"] is False
    assert packet.sub_agent_creation_policy["owner_creation_permissions"]["codex_update_sub_agent"] is False
    assert any("Do not create nested sub-agents" in action for action in packet.leaderboard_compliance_contract["must_preserve"])
    assert "terminal-bench-tasks" in packet.forbidden_paths


def test_mission_debug_cli_writes_json_packet(tmp_path):
    summary_path = tmp_path / "campaign.json"
    output_path = tmp_path / "mission.json"
    summary_path.write_text(json.dumps(campaign_summary()))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/mission_debug.py",
            "--campaign-summary",
            str(summary_path),
            "--output",
            str(output_path),
            "--max-features",
            "2",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    written = json.loads(output_path.read_text())
    assert payload["mission_id"] == written["mission_id"]
    assert len(payload["feature_candidates"]) == 4
    assert payload["evidence_summary"]["max_features_audit_only"] == 2
    assert payload["evidence_summary"]["max_features_stop_condition"] is False
    assert payload["loop_limit_contract"]["mission_debug_sub_agent"][
        "max_features_audit_only"
    ] == 2
    assert payload["loop_limit_contract"]["mission_debug_sub_agent"][
        "max_features_stop_condition"
    ] is False
    assert payload["source"] == str(summary_path)


def test_mission_debug_cli_treats_non_positive_feature_limit_as_audit(tmp_path):
    summary_path = tmp_path / "campaign.json"
    summary_path.write_text(json.dumps(campaign_summary()))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/mission_debug.py",
            "--campaign-summary",
            str(summary_path),
            "--max-features",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert len(payload["feature_candidates"]) == 4
    assert payload["evidence_summary"]["max_features_audit_only"] == 0
    assert payload["evidence_summary"]["max_features_stop_condition"] is False


def test_mission_debug_console_entrypoint_writes_json_packet(tmp_path):
    summary_path = tmp_path / "campaign.json"
    output_path = tmp_path / "mission.json"
    summary_path.write_text(json.dumps(campaign_summary()))

    from harness_evolver.cli import mission_debug

    original_argv = sys.argv
    try:
        sys.argv = [
            "harness-evolver-mission-debug",
            "--campaign-summary",
            str(summary_path),
            "--output",
            str(output_path),
            "--max-features",
            "1",
        ]
        assert mission_debug() == 0
    finally:
        sys.argv = original_argv

    payload = json.loads(output_path.read_text())
    assert payload["source"] == str(summary_path)
    assert len(payload["feature_candidates"]) == 4
    assert payload["evidence_summary"]["max_features_audit_only"] == 1


def test_module_cli_dispatches_mission_debug(tmp_path):
    summary_path = tmp_path / "campaign.json"
    output_path = tmp_path / "mission.json"
    summary_path.write_text(json.dumps(campaign_summary()))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "harness_evolver.cli",
            "mission-debug",
            "--campaign-summary",
            str(summary_path),
            "--output",
            str(output_path),
            "--max-features",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    written = json.loads(output_path.read_text())
    assert payload["mission_id"] == written["mission_id"]
    assert payload["source"] == str(summary_path)
    assert len(payload["feature_candidates"]) == 4
    assert payload["evidence_summary"]["max_features_audit_only"] == 1


def test_pyproject_registers_mission_debug_entrypoint():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    assert (
        pyproject["project"]["scripts"]["harness-evolver-mission-debug"]
        == "harness_evolver.cli:mission_debug"
    )
