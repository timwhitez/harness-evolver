"""Retry strategy - deterministic repeated-failure recovery helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from harness.recovery.base import ErrorRecovery


@dataclass(frozen=True)
class RetryDecision:
    """Decision returned after observing a failing tool call."""

    should_retry: bool
    retry_attempt: int
    observed_failures: int
    delay_seconds: float
    refresh_context: bool
    reason: str
    requires_strategy_change: bool = False
    same_call_replay_allowed: bool = True
    failure_signature: str = ""
    loop_stop_condition: bool = False
    master_loop_stop_condition: bool = False
    sub_agent_loop_stop_condition: bool = False
    worker_loop_stop_condition: bool = False
    attempt_count_stop_condition: bool = False
    retry_limit_stop_condition: bool = False
    direct_replay_threshold_stop_condition: bool = False
    retry_threshold_loop_stop_condition: bool = False
    retry_threshold_denies_retry: bool = False
    time_limit_stop_condition: bool = False
    round_limit_stop_condition: bool = False
    time_round_token_limit_driven: bool = False
    owning_loop_continues: bool = True


@dataclass
class RetryStrategy(ErrorRecovery):
    name: str = "retry_strategy"
    version: str = "0.1.0"

    # Historical name kept for config compatibility. This is a strategy-change
    # signal for one identical failed tool call, not a retry denial and not a
    # Worker, sub-agent, or master loop stop.
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float | None = 60.0
    reset_context_on_retry: bool = True
    failure_counts: dict[str, int] = field(default_factory=dict)

    def record_failure(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any] | str | None = None,
        error: str = "",
    ) -> RetryDecision:
        """Record one failed tool call and return the next retry decision.

        The method is intentionally deterministic: it computes a stable failure
        signature, increments an in-memory count for that exact failure, and
        returns the backoff delay the Worker should apply before retrying.
        """

        signature = self.failure_signature(tool_name=tool_name, args=args, error=error)
        observed_failures = self.failure_counts.get(signature, 0) + 1
        self.failure_counts[signature] = observed_failures
        return self.decision_for_observed_failures(
            observed_failures,
            failure_signature=signature,
        )

    def decision_from_history(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any] | str | None = None,
        error: str = "",
        history: Iterable[Mapping[str, Any]],
    ) -> RetryDecision:
        """Return a retry decision from existing trajectory/tool-call history.

        Only failed calls with the same tool, normalized arguments, and error
        count toward the direct-replay threshold. Successful calls and
        different command arguments do not consume this failure signal.
        """

        signature = self.failure_signature(tool_name=tool_name, args=args, error=error)
        observed_failures = 0
        for item in history:
            if item.get("success") is True:
                continue
            if self.failure_signature(
                tool_name=str(item.get("tool", "")),
                args=self._history_args(item),
                error=str(item.get("error", "")),
            ) == signature:
                observed_failures += 1
        return self.decision_for_observed_failures(
            observed_failures,
            failure_signature=signature,
        )

    def decision_for_observed_failures(
        self,
        observed_failures: int,
        *,
        failure_signature: str = "",
    ) -> RetryDecision:
        """Map a repeated failure count to an explicit retry/backoff decision."""

        if observed_failures < 1:
            return RetryDecision(
                should_retry=False,
                retry_attempt=0,
                observed_failures=observed_failures,
                delay_seconds=0.0,
                refresh_context=False,
                requires_strategy_change=False,
                same_call_replay_allowed=False,
                reason="No failed attempt has been observed; do not retry.",
                failure_signature=failure_signature,
            )

        if observed_failures <= self.max_retries:
            delay_seconds = self.delay_for_attempt(observed_failures)
            context_note = (
                "refresh context before retrying"
                if self.reset_context_on_retry
                else "reuse existing context"
            )
            return RetryDecision(
                should_retry=True,
                retry_attempt=observed_failures,
                observed_failures=observed_failures,
                delay_seconds=delay_seconds,
                refresh_context=self.reset_context_on_retry,
                requires_strategy_change=False,
                reason=(
                    f"Direct retry attempt {observed_failures}/{self.max_retries}; "
                    f"wait {delay_seconds:g}s and {context_note}. This is "
                    "single-operation recovery metadata, not a Worker, "
                    "sub-agent, or master loop stop condition."
                ),
                failure_signature=failure_signature,
            )

        return RetryDecision(
            should_retry=True,
            retry_attempt=observed_failures,
            observed_failures=observed_failures,
            delay_seconds=self.delay_for_attempt(observed_failures),
            refresh_context=True,
            requires_strategy_change=True,
            same_call_replay_allowed=False,
            reason=(
                f"Direct retry threshold observed after {observed_failures} repeated "
                "failures. This threshold requires a strategy change and does not "
                "make should_retry false. This is not a master, sub-agent, or Worker loop stop "
                "condition; continue solving by refreshing "
                "context, changing arguments, inspecting state, or using a "
                "different tool before any further attempt. The threshold is "
                "recovery metadata, not an attempt-count, time, round, token, "
                "budget, cap, timeout, or max_turns stop."
            ),
            failure_signature=failure_signature,
        )

    def delay_for_attempt(self, retry_attempt: int) -> float:
        """Return a finite, saturating exponential backoff delay.

        A configured ``max_delay_seconds`` is applied before exponentiation can
        overflow. In uncapped mode the delay saturates at the largest finite
        Python float, preserving the method's finite-float return contract.
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
        saturation_exponent = math.log(cap / base) / math.log(multiplier)
        if exponent >= saturation_exponent:
            return cap

        # The logarithmic saturation check above guarantees that this
        # intermediate remains within the selected finite cap.
        return min(float(base * (multiplier**exponent)), cap)

    def failure_signature(
        self,
        *,
        tool_name: str,
        args: Mapping[str, Any] | str | None = None,
        error: str = "",
    ) -> str:
        """Build a stable signature for repeated-failure accounting."""

        normalized_args = self._normalise_args(args)
        normalized_error = " ".join(str(error).split())
        return f"{tool_name}:{normalized_args}:{normalized_error}"

    def reset(self, failure_signature: str | None = None) -> None:
        """Clear all retry counts, or only the count for one failure signature."""

        if failure_signature is None:
            self.failure_counts.clear()
        else:
            self.failure_counts.pop(failure_signature, None)

    def render(self, context: dict[str, Any]) -> str:
        lines = [
            "## Retry Policy",
            "",
            "- Direct-replay strategy-change signal per identical failed tool call: "
            f"{self.max_retries}",
            f"- Base delay seconds: {self.base_delay_seconds:g}",
            f"- Backoff multiplier: {self.backoff_multiplier:g}",
            f"- Max delay seconds: {self.max_delay_seconds:g}"
            if self.max_delay_seconds is not None
            else "- Max delay seconds: unlimited",
            f"- Refresh context before retry: {self.reset_context_on_retry}",
            "- The threshold requests a strategy change but does not deny retry "
            "or stop master, sub-agent, or Worker loops; inspect state or "
            "switch strategy and continue the owning loop.",
        ]
        if context.get("observed_failures"):
            decision = self.decision_for_observed_failures(int(context["observed_failures"]))
            lines.extend(["", f"Current decision: {decision.reason}"])
        return "\n".join(lines)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.max_retries < 0:
            errors.append("max_retries must be >= 0")
        if not math.isfinite(self.base_delay_seconds) or self.base_delay_seconds < 0:
            errors.append("base_delay_seconds must be finite and >= 0")
        if not math.isfinite(self.backoff_multiplier) or self.backoff_multiplier < 1:
            errors.append("backoff_multiplier must be finite and >= 1")
        if self.max_delay_seconds is not None and (
            not math.isfinite(self.max_delay_seconds) or self.max_delay_seconds < 0
        ):
            errors.append("max_delay_seconds must be finite and >= 0 when set")
        return errors

    def raw_content(self) -> str:
        return self.render({})

    def _history_args(self, item: Mapping[str, Any]) -> Mapping[str, Any] | str | None:
        args = item.get("args")
        if args is not None:
            return args if isinstance(args, Mapping | str) else str(args)
        command = item.get("command")
        if command is not None:
            return {"command": command}
        return None

    def _normalise_args(self, args: Mapping[str, Any] | str | None) -> str:
        if args is None:
            return "{}"
        if isinstance(args, str):
            return self._normalise_command(args)

        normalized: dict[str, Any] = {}
        for key, value in args.items():
            if key == "command":
                normalized[key] = self._normalise_command(str(value))
            else:
                normalized[key] = value
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)

    def _normalise_command(self, command: str) -> str:
        return "\n".join(line.rstrip() for line in command.strip().splitlines())
