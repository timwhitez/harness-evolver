from __future__ import annotations

import math

from harness.recovery.retry import RetryStrategy


def test_subnormal_base_saturates_before_growth_factor_overflows() -> None:
    strategy = RetryStrategy(
        base_delay_seconds=math.nextafter(0.0, 1.0),
        backoff_multiplier=2.0,
        max_delay_seconds=60.0,
    )

    # cap / base is infinite in binary64, and 2**1999 overflows. The backoff
    # implementation must compare in log space and return the configured cap.
    assert strategy.delay_for_attempt(2_000) == 60.0
