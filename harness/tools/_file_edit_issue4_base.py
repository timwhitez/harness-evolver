"""FileEdit tool — surgical find-and-replace.

Performs exact string replacements in files.  This is much safer
than sed/awk for programmatic editing because:
  - It verifies the old_string exists (and is unique by default)
  - It can replace all occurrences
  - It preserves exact indentation
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
class FileEditTool(ToolDef):
    name: str = "edit"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = (
        "Perform exact string replacements in a file. "
        "The edit will fail if old_string is not unique in the file "
        "(use replace_all=True to replace all occurrences). "
        "Preserves exact indentation. Much safer than sed/awk."
    )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false).",
                        "default": False,
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        )

    def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        path = Path(file_path)
        prohibited_reason = prohibited_path_reason(file_path, operation="edit")
        if prohibited_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Leaderboard integrity guard blocked edit: {prohibited_reason}.",
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

        if old_string == new_string:
            return ToolResult(
                success=False, output="", error="old_string and new_string are identical"
            )

        try:
            content = path.read_text()
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Cannot read file: {e}")

        count = content.count(old_string)
        if count == 0:
            return ToolResult(
                success=False, output="", error="old_string not found in file"
            )
        if count > 1 and not replace_all:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"old_string appears {count} times in the file. "
                    "Use replace_all=True to replace all occurrences, "
                    "or provide more surrounding context to make it unique."
                ),
            )

        new_content = content.replace(old_string, new_string)
        staged_dependency_reason = staged_dependency_script_reason(file_path, new_content)
        if staged_dependency_reason:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Worker file policy blocked edit: "
                    f"{staged_dependency_reason}. Keep dependency recovery visible in "
                    "one bounded foreground shell command, or pivot to an installed, "
                    "cached, or dependency-free path."
                ),
                metadata=policy_guard_metadata("staged_dependency_script_guard"),
            )
        size_cap_reason = deliverable_size_cap_write_reason(file_path, new_content)
        if size_cap_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked edit: {size_cap_reason}",
                metadata=policy_guard_metadata(
                    "deliverable_size_cap_write_guard",
                    path=file_path,
                    content_bytes=len(new_content.encode("utf-8")),
                    limit_bytes=5000,
                ),
            )
        path.write_text(new_content)

        return ToolResult(
            success=True,
            output=f"Replaced {count} occurrence(s) in {file_path}",
            metadata={"occurrences_replaced": count},
        )
