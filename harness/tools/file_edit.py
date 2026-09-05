"""Canonical-path protected exact-text editor."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from harness.tools._file_edit_issue4_base import FileEditTool as _BaseFileEditTool
from harness.tools.base import ToolResult, policy_guard_metadata
from harness.tools.canonical_path_guard import guarded_path_failure, resolve_guarded_path
from harness.tools.safe_path_io import (
    SafePathError,
    publish_text_nofollow,
    file_identity,
    read_text_nofollow,
)
from harness.tools.shell import (
    deliverable_size_cap_write_reason,
    staged_dependency_script_reason,
)


@dataclass
class FileEditTool(_BaseFileEditTool):
    """Edit an authorized snapshot; report committed versus uncertain outcomes."""

    def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        decision = resolve_guarded_path(file_path, operation="edit", must_exist=True)
        if not decision.allowed:
            return guarded_path_failure("edit", decision)
        if old_string == new_string:
            return ToolResult(success=False, output="", error="old_string and new_string are identical")
        try:
            content, metadata = read_text_nofollow(decision.resolved, errors="strict")
        except (OSError, UnicodeError, SafePathError) as exc:
            return ToolResult(success=False, output="", error=f"Cannot read authorized path safely: {exc}",
                metadata=policy_guard_metadata("canonical_path_guard", canonical_path_checked=True, nofollow_io=True))
        count = content.count(old_string)
        if count == 0:
            return ToolResult(success=False, output="", error="old_string not found in file")
        if count > 1 and not replace_all:
            return ToolResult(success=False, output="", error=(
                f"old_string appears {count} times in the file. Use replace_all=True "
                "to replace all occurrences, or provide more surrounding context to make it unique."))
        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
        staged_reason = staged_dependency_script_reason(decision.resolved, new_content)
        if staged_reason:
            return ToolResult(success=False, output="", error=(f"Worker file policy blocked edit: {staged_reason}. Keep dependency recovery visible in one bounded "
                       "foreground shell command, or pivot to an installed, cached, or dependency-free path."),
                metadata=policy_guard_metadata("staged_dependency_script_guard"))
        size_reason = deliverable_size_cap_write_reason(decision.resolved, new_content)
        if size_reason:
            return ToolResult(success=False, output="", error=f"Worker file policy blocked edit: {size_reason}",
                metadata=policy_guard_metadata("deliverable_size_cap_write_guard", path=file_path,
                    content_bytes=len(new_content.encode("utf-8")), limit_bytes=5000))
        try:
            outcome = publish_text_nofollow(
                decision.resolved, new_content, mode=metadata.st_mode,
                expected_identity=file_identity(metadata),
                expected_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        except (OSError, SafePathError) as exc:
            return ToolResult(success=False, output="", error=f"Cannot publish authorized edit safely: {exc}",
                metadata=policy_guard_metadata("canonical_path_guard", canonical_path_checked=True,
                    nofollow_io=True, atomic_replace=False, target_identity_verified=False))
        result_metadata = {"canonical_path_checked": True, "nofollow_io": True,
                           "target_identity_verified": outcome["atomic_replace"] is True, **outcome}
        if outcome["atomic_replace"] is not True:
            return ToolResult(success=False, output="", error=outcome["publication_error"], metadata=result_metadata)
        replaced = count if replace_all else 1
        output = f"Replaced {replaced} occurrence(s) in {file_path}"
        if outcome["cleanup_warning"] or outcome["durability_warning"]:
            output += "\nWarning: edit is published; do not blindly retry. Inspect publication recovery metadata."
        return ToolResult(success=True, output=output, metadata={**result_metadata, "occurrences_replaced": replaced})
