from __future__ import annotations

import math

from harness.recovery.retry import RetryStrategy


def test_subnormal_base_saturates_after_growth_factor_overflows() -> None:
    strategy = RetryStrategy(
        base_delay_seconds=math.nextafter(0.0, 1.0),
        backoff_multiplier=2.0,
        max_delay_seconds=60.0,
    )

    # cap / base and 2**1999 both overflow in binary64. The final mathematical
    # product is above the cap, so the bounded high-precision path returns it.
    assert strategy.delay_for_attempt(2_000) == 60.0


def test_overflowed_growth_factor_can_still_produce_a_value_below_cap() -> None:
    smallest = math.nextafter(0.0, 1.0)  # exactly 2**-1074
    strategy = RetryStrategy(
        base_delay_seconds=smallest,
        backoff_multiplier=2.0,
        max_delay_seconds=2.0,
    )

    # 2**1074 cannot be represented as a float, but
    # 2**-1074 * 2**1074 is exactly 1 and must not be saturated to 2.
    assert strategy.delay_for_attempt(1_075) == 1.0
