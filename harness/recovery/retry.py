"""Retry strategy with overflow-safe logarithmic saturation."""

from __future__ import annotations

import math
import sys
from typing import Any

from harness.recovery import _retry_issue20_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


def _delay_for_attempt(self: Any, retry_attempt: int) -> float:
    """Return finite exponential backoff without constructing unsafe intermediates."""

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
    log_multiplier = math.log(multiplier)
    log_base = math.log(base)
    log_cap = math.log(cap)
    saturation_exponent = (log_cap - log_base) / log_multiplier
    if exponent >= saturation_exponent:
        return cap

    # At this point the mathematical result is strictly below the cap. Prefer
    # direct multiplication when the growth factor itself is representable so
    # ordinary powers retain their exact floating-point behavior. For a tiny
    # base with a very large exponent, evaluate in log space instead.
    log_growth = exponent * log_multiplier
    if log_growth <= math.log(sys.float_info.max):
        try:
            return min(base * (multiplier**exponent), cap)
        except OverflowError:
            pass
    return min(math.exp(log_base + log_growth), cap)


_base.RetryStrategy.delay_for_attempt = _delay_for_attempt
RetryStrategy = _base.RetryStrategy
