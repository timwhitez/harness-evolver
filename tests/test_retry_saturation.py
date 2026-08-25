from __future__ import annotations

from decimal import Decimal
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


def test_arbitrarily_large_attempt_returns_cap_without_decimal_overflow() -> None:
    strategy = RetryStrategy(
        base_delay_seconds=math.nextafter(0.0, 1.0),
        backoff_multiplier=math.nextafter(1.0, math.inf),
        max_delay_seconds=60.0,
    )

    assert strategy.delay_for_attempt(10**1_000) == 60.0


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
    ("base", "multiplier", "exponent"),
    [
        (0.1, 2.0, 0),
        (0.3, 1.2, 10),
        (0.3, 2.0, 8),
        (1.0, 2.0, 8),
        (2.0, 3.0, 2),
    ],
)
def test_cap_one_ulp_above_direct_result_never_clamps_early(
    base: float,
    multiplier: float,
    exponent: int,
) -> None:
    uncapped = base * (multiplier**exponent)
    cap = math.nextafter(uncapped, math.inf)
    strategy = RetryStrategy(
        base_delay_seconds=base,
        backoff_multiplier=multiplier,
        max_delay_seconds=cap,
    )

    assert uncapped < cap
    assert strategy.delay_for_attempt(exponent + 1) == uncapped


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("base_delay_seconds", 10**100_000, id="huge-integer"),
        pytest.param(
            "backoff_multiplier",
            Decimal("1e100000"),
            id="huge-decimal",
        ),
        pytest.param("max_delay_seconds", object(), id="non-numeric-object"),
        pytest.param("base_delay_seconds", True, id="boolean"),
    ],
)
def test_numeric_conversion_failures_are_reported_not_leaked(
    field: str,
    value: object,
) -> None:
    strategy = RetryStrategy(**{field: value})

    errors = strategy.validate()

    assert errors
    assert any(field in error for error in errors)
    with pytest.raises(ValueError, match=field):
        strategy.delay_for_attempt(1)


@pytest.mark.parametrize("attempt", [True, 1.0, Decimal("1")])
def test_retry_attempt_requires_a_non_boolean_integer(attempt: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RetryStrategy().delay_for_attempt(attempt)  # type: ignore[arg-type]


def test_retry_attempt_must_be_positive() -> None:
    with pytest.raises(ValueError, match="retry_attempt must be >= 1"):
        RetryStrategy().delay_for_attempt(0)
