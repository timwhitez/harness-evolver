"""Trial CLI with None-aware precedence and validated effective request values.

The issue-12 implementation is retained in
:mod:`scripts._run_trial_issue12_base`; this facade preserves every existing
command/helper while validating the values produced by the precedence merge.
"""

from __future__ import annotations

import math
from typing import Any

from scripts import _run_trial_issue12_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_BASE_RESOLVE_AGENT_CONFIG = _base.resolve_agent_config


def _integer_value(
    value: object,
    *,
    field: str,
    parser: Any,
) -> int:
    """Parse one effective integer without accepting bool or non-finite floats."""

    if isinstance(value, bool):
        parser.error(f"{field} must be an integer, not a boolean")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            parser.error(f"{field} must be a finite integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parser.error(f"{field} must be an integer")
    return parsed


def _validate_effective_model_request_values(
    config: dict[str, object],
    parser: Any,
) -> None:
    """Reject invalid falsy values instead of silently replacing them.

    ``None`` means absent. Zero remains a valid explicit value only for fields
    whose existing schema permits zero: retry count and reasoning-token budget.
    Provider timeout and maximum output length must be strictly positive.
    """

    reasoning_tokens = config.get("reasoning_max_tokens")
    if reasoning_tokens is not None:
        parsed = _integer_value(
            reasoning_tokens,
            field="reasoning_max_tokens",
            parser=parser,
        )
        if parsed < 0:
            parser.error("reasoning_max_tokens must be non-negative")
        config["reasoning_max_tokens"] = parsed

    retries = config.get("max_retries")
    if retries is not None:
        parsed = _integer_value(retries, field="max_retries", parser=parser)
        if parsed < 0:
            parser.error("max_retries must be non-negative")
        config["max_retries"] = parsed

    timeout_seconds = config.get("timeout_seconds")
    if timeout_seconds is not None:
        parsed = _integer_value(
            timeout_seconds,
            field="timeout_seconds",
            parser=parser,
        )
        if parsed <= 0:
            parser.error("timeout_seconds must be positive")
        config["timeout_seconds"] = parsed

    output_tokens = config.get("max_output_tokens")
    if output_tokens is not None:
        parsed = _integer_value(
            output_tokens,
            field="max_output_tokens",
            parser=parser,
        )
        if parsed <= 0:
            parser.error("max_output_tokens must be positive")
        # Harbor transports this optional agent kwarg as text; retain that
        # established representation after validating the numeric contract.
        config["max_output_tokens"] = str(parsed)


def resolve_agent_config(args: Any, parser: Any) -> dict[str, object]:
    config = _BASE_RESOLVE_AGENT_CONFIG(args, parser)
    _validate_effective_model_request_values(config, parser)
    return config


_base.resolve_agent_config = resolve_agent_config
main = _base.main


if __name__ == "__main__":
    raise SystemExit(main())
