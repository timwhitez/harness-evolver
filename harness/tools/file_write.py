"""FileWrite tool — atomic whole-file replacement.

Content is written to a same-directory temporary file and committed with
``os.replace``. A failure before replacement leaves the prior target intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import stat
import tempfile
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


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace ``path`` with UTF-8 text.

    The temporary file lives in the destination directory so ``os.replace``
    remains an atomic same-filesystem operation. Existing permission bits are
    copied before replacement. The temporary file is removed on every failure
    path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode: int | None = None
    try:
        existing_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        pass

    file_descriptor = -1
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.hl-write-",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline=None) as stream:
            file_descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        if existing_mode is not None:
            os.chmod(temporary_path, existing_mode)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


@dataclass
class FileWriteTool(ToolDef):
    name: str = "write"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = (
        "Atomically write a file to the filesystem. Overwrites existing files. "
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

        try:
            _atomic_write_text(path, content)
            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} chars to {file_path}",
                metadata={
                    "chars_written": len(content),
                    "lines": content.count("\n") + 1,
                    "atomic_replace": True,
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=str(exc),
                metadata={"atomic_replace": False},
            )
