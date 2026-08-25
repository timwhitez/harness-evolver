"""Canonical filesystem authorization shared by local and Harbor tools."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
from typing import Literal

from harness.tools.base import ToolResult, policy_guard_metadata
from harness.tools.host_memory_guard import host_memory_access_reason, host_memory_block_metadata
from harness.tools.leaderboard_guard import prohibited_path_reason


PathOperation = Literal["read", "write", "edit"]
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[/\\]")


@dataclass(frozen=True)
class GuardedPath:
    """One canonicalization/authorization decision.

    ``resolved`` is exposed only to the caller for authorized operations. Error
    rendering deliberately omits it so a blocked alias cannot disclose a hidden
    verifier or host-memory location through tool output or metadata.
    """

    requested: str
    operation: PathOperation
    resolved: str = ""
    blocked_by: str = ""
    reason: str = ""
    error: str = ""

    @property
    def allowed(self) -> bool:
        return bool(self.resolved) and not self.blocked_by and not self.error


def guard_observed_text(text: str, *, operation: PathOperation) -> GuardedPath:
    """Apply the existing lexical policies to arbitrary observed path text."""

    blocked_by, reason = _policy_violation(text, operation=operation)
    return GuardedPath(
        requested=text,
        operation=operation,
        blocked_by=blocked_by,
        reason=reason,
    )


def resolve_guarded_path(
    path: str | os.PathLike[str],
    *,
    operation: PathOperation,
    must_exist: bool,
    allowed_root: str | os.PathLike[str] | None = None,
) -> GuardedPath:
    """Resolve local filesystem aliases before authorizing effective I/O.

    For creates, ``Path.resolve(strict=False)`` resolves every existing ancestor
    while retaining the not-yet-created suffix. Reads/edits require an existing
    target and resolve it strictly. When ``allowed_root`` is provided, the
    effective target must remain inside that already-authorized canonical root.
    """

    requested = os.fspath(path)
    unsafe_reason = unsafe_requested_path_reason(requested)
    if unsafe_reason:
        return GuardedPath(
            requested=requested,
            operation=operation,
            blocked_by="canonical_path_guard",
            reason=unsafe_reason,
        )
    lexical = guard_observed_text(requested, operation=operation)
    if lexical.blocked_by:
        return lexical

    try:
        resolved_path = Path(requested).expanduser().resolve(strict=must_exist)
    except FileNotFoundError:
        return GuardedPath(
            requested=requested,
            operation=operation,
            error=f"Path not found: {requested}",
        )
    except (OSError, RuntimeError) as exc:
        return GuardedPath(
            requested=requested,
            operation=operation,
            error=f"Cannot resolve path safely: {exc}",
        )

    canonical = guard_canonical_path_strings(
        requested=requested,
        resolved=str(resolved_path),
        operation=operation,
        allowed_root=(str(Path(allowed_root).resolve(strict=True)) if allowed_root else None),
    )
    return canonical


def guard_canonical_path_strings(
    *,
    requested: str,
    resolved: str,
    operation: PathOperation,
    allowed_root: str | None = None,
) -> GuardedPath:
    """Authorize an already-resolved path, including Harbor-side resolutions."""

    lexical = guard_observed_text(requested, operation=operation)
    if lexical.blocked_by:
        return lexical

    blocked_by, reason = _policy_violation(resolved, operation=operation)
    if blocked_by:
        return GuardedPath(
            requested=requested,
            operation=operation,
            blocked_by=blocked_by,
            reason=reason,
        )

    if allowed_root is not None and not _posix_path_within(resolved, allowed_root):
        return GuardedPath(
            requested=requested,
            operation=operation,
            blocked_by="canonical_path_guard",
            reason="resolved target escapes the authorized canonical root",
        )

    return GuardedPath(
        requested=requested,
        operation=operation,
        resolved=resolved,
    )



def unsafe_requested_path_reason(path: str) -> str:
    """Reject ambiguous parent traversal before filesystem resolution."""

    if "\x00" in path:
        return "paths containing NUL bytes are not allowed"
    normalized = path.replace("\\", "/")
    if ".." in PurePosixPath(normalized).parts:
        return "parent traversal is not allowed in filesystem paths"
    return ""

def unsafe_relative_pattern_reason(pattern: str) -> str:
    """Reject glob-like patterns that can select outside an authorized root."""

    if "\x00" in pattern:
        return "patterns containing NUL bytes are not allowed"
    normalized = pattern.replace("\\", "/")
    if normalized.startswith("/") or _WINDOWS_DRIVE.match(pattern):
        return "absolute patterns are not allowed"
    if ".." in PurePosixPath(normalized).parts:
        return "parent traversal is not allowed in search patterns"
    return ""


def guarded_path_failure(action: str, decision: GuardedPath) -> ToolResult:
    """Render one fail-closed path decision without leaking its canonical target."""

    if decision.blocked_by == "host_memory_guard":
        metadata = host_memory_block_metadata()
        metadata.update(
            {
                "canonical_path_checked": True,
                "requested_path": decision.requested,
            }
        )
        error = (
            f"Worker host-memory policy blocked {action}: {decision.reason}. "
            "Same-task memory summaries are already injected into the prompt; "
            "host trial memory is not task-workspace evidence."
        )
    elif decision.blocked_by:
        metadata = policy_guard_metadata(
            decision.blocked_by,
            canonical_path_checked=True,
            requested_path=decision.requested,
        )
        policy_name = (
            "Leaderboard integrity guard"
            if decision.blocked_by == "leaderboard_integrity_guard"
            else "Canonical path guard"
        )
        error = f"{policy_name} blocked {action}: {decision.reason}."
    else:
        metadata = policy_guard_metadata(
            "canonical_path_guard",
            canonical_path_checked=True,
            requested_path=decision.requested,
            path_resolution_failed=True,
        )
        error = decision.error or f"Cannot resolve path safely for {action}"

    return ToolResult(success=False, output="", error=error, metadata=metadata)


def _policy_violation(text: str, *, operation: PathOperation) -> tuple[str, str]:
    leaderboard_reason = prohibited_path_reason(text, operation=operation)
    if leaderboard_reason:
        return "leaderboard_integrity_guard", leaderboard_reason
    memory_reason = host_memory_access_reason(text)
    if memory_reason:
        return "host_memory_guard", memory_reason
    return "", ""


def _posix_path_within(path: str, root: str) -> bool:
    """Containment check for canonical POSIX paths returned by Harbor."""

    try:
        normalized_path = posixpath.normpath(path.replace("\\", "/"))
        normalized_root = posixpath.normpath(root.replace("\\", "/"))
        return posixpath.commonpath([normalized_path, normalized_root]) == normalized_root
    except ValueError:
        return False
