import pytest

from harness.recovery.retry import RetryStrategy


def test_retry_strategy_records_failures_with_strategy_switch_threshold_and_backoff():
    strategy = RetryStrategy(max_retries=3, base_delay_seconds=0.5, backoff_multiplier=3)

    first = strategy.record_failure(
        tool_name="bash",
        args={"command": "pytest"},
        error="exit code: 1",
    )
    second = strategy.record_failure(
        tool_name="bash",
        args={"command": "pytest"},
        error="exit code: 1",
    )
    third = strategy.record_failure(
        tool_name="bash",
        args={"command": "pytest"},
        error="exit code: 1",
    )
    exhausted = strategy.record_failure(
        tool_name="bash",
        args={"command": "pytest"},
        error="exit code: 1",
    )

    assert first.should_retry is True
    assert first.retry_attempt == 1
    assert first.delay_seconds == pytest.approx(0.5)
    assert first.refresh_context is True

    assert second.should_retry is True
    assert second.delay_seconds == pytest.approx(1.5)
    assert third.should_retry is True
    assert third.delay_seconds == pytest.approx(4.5)

    assert exhausted.should_retry is True
    assert exhausted.retry_attempt == 4
    assert exhausted.delay_seconds == pytest.approx(13.5)
    assert exhausted.refresh_context is True
    assert exhausted.requires_strategy_change is True
    assert exhausted.same_call_replay_allowed is False
    assert exhausted.retry_threshold_denies_retry is False
    assert "Direct retry threshold observed" in exhausted.reason
    assert "does not make should_retry false" in exhausted.reason
    assert "not a master, sub-agent, or Worker loop stop condition" in exhausted.reason
    assert "continue solving" in exhausted.reason
    for decision in (first, second, third, exhausted):
        assert decision.retry_threshold_denies_retry is False
        assert decision.loop_stop_condition is False
        assert decision.master_loop_stop_condition is False
        assert decision.sub_agent_loop_stop_condition is False
        assert decision.worker_loop_stop_condition is False
        assert decision.attempt_count_stop_condition is False
        assert decision.retry_limit_stop_condition is False
        assert decision.direct_replay_threshold_stop_condition is False
        assert decision.retry_threshold_loop_stop_condition is False
        assert decision.time_limit_stop_condition is False
        assert decision.round_limit_stop_condition is False
        assert decision.time_round_token_limit_driven is False
        assert decision.owning_loop_continues is True


def test_retry_strategy_render_names_master_sub_agent_and_worker_loops():
    rendered = RetryStrategy(max_retries=1).render({"observed_failures": 2})

    assert "master, sub-agent, or Worker loops" in rendered
    assert "not an attempt-count" in rendered
    assert "continue the owning loop" in rendered
    assert "does not deny retry" in rendered


def test_retry_strategy_caps_backoff_delay():
    strategy = RetryStrategy(
        max_retries=5,
        base_delay_seconds=2,
        backoff_multiplier=4,
        max_delay_seconds=10,
    )

    assert strategy.delay_for_attempt(1) == pytest.approx(2)
    assert strategy.delay_for_attempt(2) == pytest.approx(8)
    assert strategy.delay_for_attempt(3) == pytest.approx(10)


def test_retry_strategy_counts_only_matching_failed_history_items():
    strategy = RetryStrategy(max_retries=2)
    history = [
        {
            "tool": "bash",
            "args": {"command": "pytest"},
            "success": False,
            "error": "exit code: 1",
        },
        {
            "tool": "bash",
            "args": {"command": "pytest -q"},
            "success": False,
            "error": "exit code: 1",
        },
        {
            "tool": "bash",
            "args": {"command": "pytest"},
            "success": True,
            "error": "",
        },
        {
            "tool": "bash",
            "command": " pytest ",
            "success": False,
            "error": "exit code: 1",
        },
    ]

    decision = strategy.decision_from_history(
        tool_name="bash",
        args={"command": "pytest"},
        error="exit code: 1",
        history=history,
    )

    assert decision.should_retry is True
    assert decision.retry_attempt == 2
    assert decision.observed_failures == 2
    assert decision.same_call_replay_allowed is True


def test_retry_strategy_threshold_never_denies_loop_retry() -> None:
    strategy = RetryStrategy(max_retries=1, max_delay_seconds=3)

    first = strategy.decision_for_observed_failures(1)
    threshold = strategy.decision_for_observed_failures(99)

    assert first.should_retry is True
    assert first.requires_strategy_change is False
    assert first.same_call_replay_allowed is True

    assert threshold.should_retry is True
    assert threshold.retry_attempt == 99
    assert threshold.delay_seconds == pytest.approx(3)
    assert threshold.requires_strategy_change is True
    assert threshold.same_call_replay_allowed is False
    assert threshold.retry_threshold_denies_retry is False
    assert threshold.retry_limit_stop_condition is False
    assert threshold.direct_replay_threshold_stop_condition is False
    assert threshold.retry_threshold_loop_stop_condition is False
    assert threshold.owning_loop_continues is True


def test_retry_strategy_validation_rejects_invalid_policy_values():
    strategy = RetryStrategy(
        max_retries=-1,
        base_delay_seconds=-0.1,
        backoff_multiplier=0.5,
        max_delay_seconds=-1,
    )

    errors = strategy.validate()

    assert "max_retries must be >= 0" in errors
    assert any(
        error.startswith("base_delay_seconds must") and ">= 0" in error
        for error in errors
    )
    assert any(
        error.startswith("backoff_multiplier must") and ">= 1" in error
        for error in errors
    )
    assert any(
        error.startswith("max_delay_seconds must") and ">= 0" in error
        for error in errors
    )
