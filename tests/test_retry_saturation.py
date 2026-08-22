from __future__ import annotations

import math
import sys

import pytest

from harness.recovery.retry import RetryStrategy


def test_capped_backoff_saturates_without_overflow() -> None:
    strategy = RetryStrategy(
        base_delay_seconds=1.0,
        backoff_multiplier=2.0,
        max_delay_seconds=60.0,
    )

    assert strategy.delay_for_attempt(1) == 1.0
    assert strategy.delay_for_attempt(6) == 32.0
    assert strategy.delay_for_attempt(7) == 60.0
    assert strategy.delay_for_attempt(2_000) == 60.0


def test_uncapped_backoff_saturates_at_largest_finite_float() -> None:
    strategy = RetryStrategy(
        base_delay_seconds=1.0,
        backoff_multiplier=2.0,
        max_delay_seconds=None,
    )

    delay = strategy.delay_for_attempt(2_000)

    assert delay == sys.float_info.max
    assert math.isfinite(delay)


def test_zero_base_and_unit_multiplier_keep_exact_semantics() -> None:
    assert RetryStrategy(base_delay_seconds=0.0).delay_for_attempt(100_000) == 0.0
    assert (
        RetryStrategy(
            base_delay_seconds=3.5,
            backoff_multiplier=1.0,
            max_delay_seconds=2.0,
        ).delay_for_attempt(100_000)
        == 2.0
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_delay_seconds", float("nan")),
        ("base_delay_seconds", float("inf")),
        ("backoff_multiplier", float("nan")),
        ("backoff_multiplier", float("inf")),
        ("max_delay_seconds", float("nan")),
        ("max_delay_seconds", float("inf")),
    ],
)
def test_validation_rejects_non_finite_parameters(field: str, value: float) -> None:
    strategy = RetryStrategy(**{field: value})

    assert strategy.validate()
    with pytest.raises(ValueError):
        strategy.delay_for_attempt(1)


def test_retry_attempt_must_be_positive() -> None:
    with pytest.raises(ValueError, match="retry_attempt must be >= 1"):
        RetryStrategy().delay_for_attempt(0)
