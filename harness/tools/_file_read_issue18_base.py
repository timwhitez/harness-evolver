"""Canonical-path protected file reader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.tools._file_read_issue4_base import FileReadTool as _BaseFileReadTool
from harness.tools.base import ToolResult
from harness.tools.canonical_path_guard import guarded_path_failure, resolve_guarded_path
from harness.tools.safe_path_io import SafePathError, read_text_nofollow


@dataclass
class FileReadTool(_BaseFileReadTool):
    """Read an authorized canonical target without following later symlink swaps."""

    def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        effective_limit = self.max_lines if limit is None else limit
        if offset < 1:
            return ToolResult(success=False, output="", error="offset must be >= 1")
        if effective_limit < 1:
            return ToolResult(success=False, output="", error="limit must be >= 1")

        decision = resolve_guarded_path(
            file_path,
            operation="read",
            must_exist=True,
        )
        if not decision.allowed:
            return guarded_path_failure("read", decision)

        try:
            content, metadata = read_text_nofollow(decision.resolved, errors="replace")
        except (OSError, SafePathError) as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Cannot read authorized path safely: {exc}",
                metadata={
                    "blocked_by": "canonical_path_guard",
                    "canonical_path_checked": True,
                    "nofollow_io": True,
                },
            )

        lines = content.splitlines()
        total_lines = len(lines)
        start = min(offset - 1, total_lines)
        end = min(start + effective_limit, total_lines)
        selected = lines[start:end]
        output = "\n".join(
            f"{number}\t{line.rstrip()}"
            for number, line in enumerate(selected, start=start + 1)
        )
        if end < total_lines:
            output += (
                ("\n" if output else "")
                + f"... ({total_lines - end} more lines, use offset={end + 1} to continue)"
            )

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "total_lines": total_lines,
                "start_line": offset,
                "end_line": end,
                "file_size": metadata.st_size,
                "canonical_path_checked": True,
                "nofollow_io": True,
            },
        )
