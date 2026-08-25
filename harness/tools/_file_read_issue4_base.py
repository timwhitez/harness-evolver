"""FileRead tool — read file contents with line numbering.

Prefer this over `cat` in shell — it provides structured output
with line numbers for easier reference in edits.
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


@dataclass
class FileReadTool(ToolDef):
    name: str = "read"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = (
        "Read a file from the filesystem. Returns content with line numbers. "
        "Use this instead of 'cat' for reading file contents. "
        "Supports offset and limit for reading large files. "
        "Can also read images (PNG, JPG) and PDF files."
    )
    max_lines: int = 2000

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-based).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Maximum number of lines to read (default: {self.max_lines}).",
                    },
                },
                "required": ["file_path"],
            },
        )

    def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        limit = limit or self.max_lines
        path = Path(file_path)
        prohibited_reason = prohibited_path_reason(file_path, operation="read")
        if prohibited_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Leaderboard integrity guard blocked read: {prohibited_reason}.",
                metadata=policy_guard_metadata("leaderboard_integrity_guard"),
            )
        if host_memory_access_reason(file_path):
            return ToolResult(
                success=False,
                output="",
                error=host_memory_blocked_error(file_path),
                metadata=host_memory_block_metadata(),
            )

        if not path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")

        if path.is_dir():
            return ToolResult(success=False, output="", error=f"Path is a directory: {file_path}")

        try:
            with open(path, "r", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            start = max(0, offset - 1)
            end = min(start + limit, total_lines)
            selected = lines[start:end]

            output_lines = []
            for i, line in enumerate(selected, start=start + 1):
                output_lines.append(f"{i}\t{line.rstrip()}")

            output = "\n".join(output_lines)

            if end < total_lines:
                output += f"\n... ({total_lines - end} more lines, use offset={end + 1} to continue)"

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "total_lines": total_lines,
                    "start_line": start + 1,
                    "end_line": end,
                    "file_size": path.stat().st_size,
                },
            )
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                output="",
                error=f"Cannot read {file_path} as text (binary file)",
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
