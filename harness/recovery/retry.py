"""Retry strategy with overflow-safe saturating exponentiation."""

from __future__ import annotations

from decimal import Decimal, localcontext
import math
import sys
from typing import Any

from harness.recovery import _retry_issue20_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, _value)


_DECIMAL_SATURATION_PRECISION = 100


def _coerce_finite_float(
    value: object,
    *,
    field: str,
    minimum: float,
) -> float:
    """Return a finite float or raise the field's stable validation error.

    ``float()`` itself can raise ``OverflowError`` for very large integers and
    Decimals.  Public retry-policy validation must convert those cases into a
    deterministic ``ValueError`` rather than leaking an implementation-specific
    numeric exception.
    """

    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not a boolean")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number >= {minimum:g}") from exc
    if not math.isfinite(converted) or converted < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum:g}")
    return converted


def _numeric_validation_error(
    value: object,
    *,
    field: str,
    minimum: float,
) -> str | None:
    try:
        _coerce_finite_float(value, field=field, minimum=minimum)
    except ValueError as exc:
        return str(exc)
    return None


def _delay_for_attempt(self: Any, retry_attempt: int) -> float:
    """Return finite exponential backoff without overflow or early clamping."""

    if isinstance(retry_attempt, bool) or not isinstance(retry_attempt, int):
        raise ValueError("retry_attempt must be a positive integer")
    if retry_attempt < 1:
        raise ValueError("retry_attempt must be >= 1")

    base = _coerce_finite_float(
        self.base_delay_seconds,
        field="base_delay_seconds",
        minimum=0.0,
    )
    multiplier = _coerce_finite_float(
        self.backoff_multiplier,
        field="backoff_multiplier",
        minimum=1.0,
    )
    configured_cap = self.max_delay_seconds
    cap = (
        _coerce_finite_float(
            configured_cap,
            field="max_delay_seconds",
            minimum=0.0,
        )
        if configured_cap is not None
        else sys.float_info.max
    )

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


def _validate(self: Any) -> list[str]:
    """Validate every public numeric field without raising conversion errors."""

    errors: list[str] = []
    max_retries = self.max_retries
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        errors.append("max_retries must be an integer >= 0")
    elif max_retries < 0:
        errors.append("max_retries must be >= 0")

    for field, minimum in (
        ("base_delay_seconds", 0.0),
        ("backoff_multiplier", 1.0),
    ):
        error = _numeric_validation_error(
            getattr(self, field),
            field=field,
            minimum=minimum,
        )
        if error:
            errors.append(error)

    if self.max_delay_seconds is not None:
        error = _numeric_validation_error(
            self.max_delay_seconds,
            field="max_delay_seconds",
            minimum=0.0,
        )
        if error:
            errors.append(error)
    return errors


_base.RetryStrategy.delay_for_attempt = _delay_for_attempt
_base.RetryStrategy.validate = _validate
RetryStrategy = _base.RetryStrategy
