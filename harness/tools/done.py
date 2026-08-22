"""Explicit task-completion tool for the Worker.

Completion is an explicit model action, not a phrase detected in assistant
prose. The Worker declares it believes the task is ready for Harbor/verifier
evaluation by calling this tool. The Rust Worker core still gates the actual
finalization on pending todos and any configured local verification command,
and Harbor/verifier remains the only benchmark pass signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.tools.base import ToolDef, ToolResult, ToolSchema


@dataclass
class DoneTool(ToolDef):
    name: str = "done"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = (
        "Declare that the task is complete and ready for Harbor verification. "
        "Call this only after the required work is implemented and any relevant "
        "local checks pass. Do not announce completion in plain text; the run "
        "only finalizes when you call this tool. This does not decide "
        "benchmark pass/fail and is rejected while todos remain pending or a "
        "configured local verification command still fails."
    )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "Optional short summary of what was completed and "
                            "which local checks were run."
                        ),
                    }
                },
            },
        )

    def execute(self, summary: str = "", **_: Any) -> ToolResult:
        message = "Completion requested; Harbor/verifier decides pass/fail."
        if summary.strip():
            message = f"{message} Summary: {summary.strip()}"
        return ToolResult(success=True, output=message, metadata={"completion_requested": True})
