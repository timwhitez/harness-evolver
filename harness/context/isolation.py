"""Context isolation — master-created sub-agent window boundaries.

Whatever a sub-agent consumes in tokens never pollutes the parent context.
Only a summary returns. Context isolation does not authorize nested agents:
only the master orchestrator may create sub-agents, and sub-agents may not
create another sub-agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.context.base import ContextManager


@dataclass
class ContextIsolation(ContextManager):
    name: str = "context_isolation"
    version: str = "0.1.0"

    max_sub_agent_depth: int | None = 1
    sub_agent_summary_max_tokens: int = 2000
    isolate_tools: bool = True
    isolate_memory: bool = True

    def render(self, context: dict[str, object]) -> str:
        depth = (
            "invalid-null" if self.max_sub_agent_depth is None else self.max_sub_agent_depth
        )
        return "\n".join(
            [
                "## Context Isolation",
                "",
                f"- Sub-agent depth audit reference: {depth}",
                "- Sub-agent creation policy: master-created sub-agents only; nested sub-agent creation is prohibited.",
                f"- Returned summary token audit reference: {self.sub_agent_summary_max_tokens}",
                f"- Isolate tools: {self.isolate_tools}",
                f"- Isolate memory: {self.isolate_memory}",
                "- Depth, summary tokens, context token windows, elapsed time, and "
                "round counts are isolation or compaction hints only; they never "
                "stop master, diagnostic/context sub-agent, Codex update sub-agent, "
                "or Worker loops. The creation permission is separate: only the master orchestrator may create sub-agents.",
            ]
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        depth = self.max_sub_agent_depth
        if depth is None:
            errors.append(
                "max_sub_agent_depth must be explicit and <= 1 because nested sub-agent creation is prohibited"
            )
        elif depth > 1:
            errors.append(
                "max_sub_agent_depth must be <= 1 because nested sub-agent creation is prohibited"
            )
        elif depth < 0:
            errors.append("max_sub_agent_depth must be >= 0")
        if self.sub_agent_summary_max_tokens < 0:
            errors.append("sub_agent_summary_max_tokens must be >= 0")
        return errors

    def raw_content(self) -> str:
        return self.render({})
