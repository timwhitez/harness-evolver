"""Retry strategy with overflow-safe saturating exponentiation."""

from __future__ import annotations

from decimal import Decimal, localcontext
import math
import sys
from typing import Any

from harness.recovery import _retry_issue20_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


_DECIMAL_SATURATION_PRECISION = 100


def _delay_for_attempt(self: Any, retry_attempt: int) -> float:
    """Return finite exponential backoff without overflow or early clamping.

    Ordinary representable powers use Python's direct binary64 arithmetic, so a
    cap that is merely the next float above the uncapped result does not get
    selected early because of a rounded logarithmic threshold. Only when the
    growth factor itself overflows do we switch to bounded high-precision
    Decimal arithmetic for the saturation comparison.
    """

    if retry_attempt < 1:
        raise ValueError("retry_attempt must be >= 1")

    base = float(self.base_delay_seconds)
    multiplier = float(self.backoff_multiplier)
    configured_cap = self.max_delay_seconds
    cap = float(configured_cap) if configured_cap is not None else sys.float_info.max

    if not math.isfinite(base) or base < 0:
        raise ValueError("base_delay_seconds must be finite and >= 0")
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError("backoff_multiplier must be finite and >= 1")
    if not math.isfinite(cap) or cap < 0:
        raise ValueError("max_delay_seconds must be finite and >= 0 when set")

    if base == 0 or cap == 0:
        return 0.0
    if multiplier == 1:
        return min(base, cap)
    if base >= cap:
        return cap

    exponent = retry_attempt - 1

    # Preserve the exact ordinary binary64 path whenever the growth factor is
    # representable. This also handles exponent zero without any log-rounding
    # ambiguity and lets normal min() semantics choose the cap only when the
    # computed delay actually reaches/exceeds it.
    try:
        growth = multiplier**exponent
    except OverflowError:
        growth = math.inf
    if math.isfinite(growth):
        delay = base * growth
        return cap if not math.isfinite(delay) else min(delay, cap)

    # The factor overflowed before multiplication. A very small base can still
    # make the final product finite (for example the smallest subnormal base
    # times 2**1999). Compute the comparison in Decimal instead of using a
    # rounded logarithmic threshold that can clamp one attempt too early.
    with localcontext() as context:
        context.prec = _DECIMAL_SATURATION_PRECISION
        decimal_delay = Decimal.from_float(base) * (
            Decimal.from_float(multiplier) ** exponent
        )
        decimal_cap = Decimal.from_float(cap)
        if decimal_delay >= decimal_cap:
            return cap
        return min(float(decimal_delay), cap)


_base.RetryStrategy.delay_for_attempt = _delay_for_attempt
RetryStrategy = _base.RetryStrategy
