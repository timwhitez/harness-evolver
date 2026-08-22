"""Base tool definitions.

ToolDef is the base for all tools.  Each tool has:
  - A name and description (rendered into the agent context)
  - A JSON Schema (for LLM tool-calling)
  - An execute method (for deterministic execution)
  - A version for tracking edits

The design follows the "minimal tool" principle from Factory Droid:
simpler tools → higher LLM accuracy.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSchema:
    """JSON Schema for a tool's parameters (OpenAI function-calling format)."""

    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "",
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: str
    error: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def policy_guard_metadata(blocked_by: str, **extra: Any) -> dict[str, Any]:
    """Metadata for recovery-oriented policy guards.

    Tool policy guards block a single unsafe or low-yield operation. They are
    evidence for the next Worker step, not a master/sub-agent/Worker loop stop.
    """

    metadata: dict[str, Any] = {
        "blocked_by": blocked_by,
        "policy_guard_stop_condition": False,
        "operation_guard_stop_condition": False,
        "loop_stop_condition": False,
    }
    metadata.update(extra)
    return metadata


def operation_timeout_metadata(
    *,
    timeout_seconds: float | int,
    requested_timeout_seconds: float | int | None = None,
    timeout_capped: bool = False,
    elapsed_ms: float | int | None = None,
    stdout: str = "",
    stderr: str = "",
    telemetry_source: str = "tool",
    **extra: Any,
) -> dict[str, Any]:
    """Metadata for one timed-out tool operation.

    A tool timeout is evidence for recovery and update selection. It must not
    become a master/sub-agent/Worker loop stop condition.
    """

    metadata: dict[str, Any] = {
        "timed_out": True,
        "timeout_seconds": timeout_seconds,
        "requested_timeout_seconds": requested_timeout_seconds,
        "timeout_capped": timeout_capped,
        "operation_timeout_stop_condition": False,
        "timeout_seconds_stop_condition": False,
        "loop_stop_condition": False,
        "tool_timeout_telemetry": True,
        "tool_timeout_telemetry_source": telemetry_source,
        "tool_timeout_telemetry_stop_condition": False,
        "timeout_telemetry_stop_condition": False,
        "stdout_len": len(stdout),
        "stderr_len": len(stderr),
        "stdout_tail": _tail_text(stdout),
        "stderr_tail": _tail_text(stderr),
        "partial_output_available": bool(stdout or stderr),
    }
    if elapsed_ms is not None:
        metadata["elapsed_ms"] = elapsed_ms
        metadata["elapsed_seconds"] = round(float(elapsed_ms) / 1000.0, 3)
    metadata.update(extra)
    return metadata


def _tail_text(value: str, *, max_chars: int = 2000) -> str:
    if len(value) <= max_chars:
        return value
    return value[-max_chars:]


class ToolDef(ABC):
    """Abstract base class for all tools.

    Each tool is a Policy in HL terms — it can be versioned,
    rendered into the agent context, and independently edited.
    """

    name: str
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = ""

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Return the JSON Schema for this tool's parameters."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters."""
        ...

    def render(self, context: dict[str, Any]) -> str:
        """Render tool description for the agent context window."""
        schema = self.get_schema()
        return f"- **{self.name}**: {self.description}"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("tool name is empty")
        if not self.description:
            errors.append(f"[{self.name}] tool description is empty")
        return errors

    def raw_content(self) -> str:
        return self.description

    def _timed_execute(self, **kwargs: Any) -> ToolResult:
        """Execute with timing."""
        start = time.time()
        try:
            result = self.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
        result.duration_ms = (time.time() - start) * 1000
        return result
