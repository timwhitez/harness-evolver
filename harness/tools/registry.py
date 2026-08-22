"""ToolRegistry — register, validate, and resolve tools.

Tools are registered by name and can be resolved at runtime.
The registry provides the tool list to the LLM for function calling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.tools.base import ToolDef, ToolResult, ToolSchema


@dataclass
class ToolRegistry:
    """Central registry for all tools available to the agent.

    Tools are registered with their schemas.  The registry
    can render all tool descriptions for the agent context
    and dispatch tool calls to the correct handler.
    """

    _tools: dict[str, ToolDef] = field(default_factory=dict)

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in OpenAI function-calling format."""
        schemas = []
        for name, tool in self._tools.items():
            schema = tool.get_schema().to_openai()
            schema["function"]["name"] = name
            schemas.append(schema)
        return schemas

    def render_descriptions(self) -> str:
        """Render all tool descriptions for the agent context."""
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- **{name}**: {tool.description}")
        return "\n".join(lines)

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Dispatch a tool call to the correct handler."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}. Available: {', '.join(self.list_tools())}",
            )
        return tool._timed_execute(**kwargs)

    def validate_all(self) -> list[str]:
        errors: list[str] = []
        for tool in self._tools.values():
            errors.extend(tool.validate())
        return errors

    def __len__(self) -> int:
        return len(self._tools)
