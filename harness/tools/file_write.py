"""Canonical-path protected whole-file writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.tools._file_write_issue4_base import FileWriteTool as _BaseFileWriteTool
from harness.tools.base import ToolResult, policy_guard_metadata
from harness.tools.canonical_path_guard import guarded_path_failure, resolve_guarded_path
from harness.tools.safe_path_io import SafePathError, publish_text_nofollow
from harness.tools.shell import (
    deliverable_size_cap_write_reason,
    staged_dependency_script_reason,
)


@dataclass
class FileWriteTool(_BaseFileWriteTool):
    """Write an authorized canonical target and report publication truthfully."""

    def execute(self, file_path: str, content: str, **kwargs: Any) -> ToolResult:
        decision = resolve_guarded_path(file_path, operation="write", must_exist=False)
        if not decision.allowed:
            return guarded_path_failure("write", decision)
        staged_reason = staged_dependency_script_reason(decision.resolved, content)
        if staged_reason:
            return ToolResult(
                success=False, output="",
                error=(f"Worker file policy blocked write: {staged_reason}. Keep dependency recovery visible in one bounded "
                       "foreground shell command, or pivot to an installed, cached, or dependency-free path."),
                metadata=policy_guard_metadata("staged_dependency_script_guard"),
            )
        size_reason = deliverable_size_cap_write_reason(decision.resolved, content)
        if size_reason:
            return ToolResult(
                success=False, output="", error=f"Worker file policy blocked write: {size_reason}",
                metadata=policy_guard_metadata(
                    "deliverable_size_cap_write_guard", path=file_path,
                    content_bytes=len(content.encode("utf-8")), limit_bytes=5000,
                ),
            )
        try:
            outcome = publish_text_nofollow(decision.resolved, content)
        except (OSError, SafePathError) as exc:
            return ToolResult(
                success=False, output="", error=f"Cannot write authorized path safely: {exc}",
                metadata=policy_guard_metadata("canonical_path_guard",
                    canonical_path_checked=True, nofollow_io=True, atomic_replace=False),
            )
        metadata = {"canonical_path_checked": True, "nofollow_io": True, **outcome}
        if outcome["atomic_replace"] is not True:
            return ToolResult(success=False, output="", error=outcome["publication_error"], metadata=metadata)
        output = f"Wrote {len(content)} chars to {file_path}"
        if outcome["cleanup_warning"] or outcome["durability_warning"]:
            output += "\nWarning: content is published; do not blindly retry. Inspect publication recovery metadata."
        return ToolResult(success=True, output=output, metadata={
            **metadata, "chars_written": len(content), "lines": content.count("\n") + 1,
        })
