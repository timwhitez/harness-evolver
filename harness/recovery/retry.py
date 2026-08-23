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
    """Return finite exponential backoff without overflow or early clamping."""

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

    # Preserve ordinary binary64 behavior whenever the growth factor is
    # representable. This prevents logarithmic threshold rounding from
    # selecting a cap that is merely the next float above the actual result.
    try:
        growth = multiplier**exponent
    except OverflowError:
        growth = math.inf
    if math.isfinite(growth):
        delay = base * growth
        return cap if not math.isfinite(delay) else min(delay, cap)

    # The factor overflowed before multiplication. First prove obviously
    # saturated attempts using a conservative logarithmic threshold. The
    # integer/float comparison does not require converting an arbitrarily large
    # Python integer to float. A margin measured in threshold ULPs ensures this
    # fast path can never clamp at the rounded boundary.
    log_multiplier = math.log(multiplier)
    threshold = (math.log(cap) - math.log(base)) / log_multiplier
    threshold_margin = max(2.0, 8.0 * math.ulp(threshold))
    if exponent > threshold + threshold_margin:
        return cap

    # Near or below the boundary, the complete product's magnitude is bounded
    # by cap/base (at most about 10**632 for finite binary64 inputs), even when
    # the exponent itself is very large because multiplier is close to one.
    # Decimal exponentiation by an integer is logarithmic in the exponent and
    # avoids materializing the already-overflowed binary64 growth factor.
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
