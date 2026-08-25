"""FileWrite tool — atomic whole-file write.

Creates or overwrites a file atomically.  Used for creating new files
or completely rewriting existing ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.tools.base import ToolDef, ToolResult, ToolSchema, policy_guard_metadata
from harness.tools.host_memory_guard import (
    host_memory_access_reason,
    host_memory_block_metadata,
    host_memory_blocked_error,
)
from harness.tools.leaderboard_guard import prohibited_path_reason
from harness.tools.shell import (
    deliverable_size_cap_write_reason,
    staged_dependency_script_reason,
)


@dataclass
class FileWriteTool(ToolDef):
    name: str = "write"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = (
        "Write a file to the filesystem. Overwrites existing files. "
        "Use this for creating new files or completely rewriting existing ones. "
        "For editing existing files, prefer the 'edit' tool."
    )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to write the file to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write.",
                    },
                },
                "required": ["file_path", "content"],
            },
        )

    def execute(self, file_path: str, content: str, **kwargs: Any) -> ToolResult:
        path = Path(file_path)
        prohibited_reason = prohibited_path_reason(file_path, operation="write")
        if prohibited_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Leaderboard integrity guard blocked write: {prohibited_reason}.",
                metadata=policy_guard_metadata("leaderboard_integrity_guard"),
            )
        if host_memory_access_reason(file_path):
            return ToolResult(
                success=False,
                output="",
                error=host_memory_blocked_error(file_path),
                metadata=host_memory_block_metadata(),
            )
        staged_dependency_reason = staged_dependency_script_reason(file_path, content)
        if staged_dependency_reason:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Worker file policy blocked write: "
                    f"{staged_dependency_reason}. Keep dependency recovery visible in "
                    "one bounded foreground shell command, or pivot to an installed, "
                    "cached, or dependency-free path."
                ),
                metadata=policy_guard_metadata("staged_dependency_script_guard"),
            )
        size_cap_reason = deliverable_size_cap_write_reason(file_path, content)
        if size_cap_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked write: {size_cap_reason}",
                metadata=policy_guard_metadata(
                    "deliverable_size_cap_write_guard",
                    path=file_path,
                    content_bytes=len(content.encode("utf-8")),
                    limit_bytes=5000,
                ),
            )

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(content)
            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} chars to {file_path}",
                metadata={"chars_written": len(content), "lines": content.count("\n") + 1},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
