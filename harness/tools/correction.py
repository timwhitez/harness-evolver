"""Tool-call correction helpers for malformed or repeatedly failing calls."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolFailureTracker:
    """Track repeated tool failures and produce model-facing recovery hints."""

    failures: dict[str, int] = field(default_factory=dict)

    def record(self, *, tool_name: str, error: str) -> str | None:
        key = f"{tool_name}:{error[:160]}"
        self.failures[key] = self.failures.get(key, 0) + 1
        count = self.failures[key]
        if count < 2:
            return None
        if "timed out" in error.lower() or "timeout" in error.lower():
            return (
                f"The tool call {tool_name!r} has timed out {count} times with "
                f"similar evidence: {error}. Do not retry the same long command "
                "unchanged. Break the work into smaller checks, add progress output, "
                "reduce the search/build scope, or choose a faster implementation path. "
                "This is strategy-recovery evidence, not a master, sub-agent, "
                "or Worker loop stop condition."
            )
        return (
            f"The tool call {tool_name!r} has failed {count} times with the same "
            f"error: {error}. Re-check the path, arguments, permissions, and task "
            "state before retrying; switch strategy if the precondition is false. "
            "This is strategy-recovery evidence, not a master, sub-agent, or "
            "Worker loop stop condition."
        )
