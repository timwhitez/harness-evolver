"""Canonical-path search tools with nofollow content reads."""

from __future__ import annotations

import fnmatch
import importlib
from pathlib import Path
import re
from typing import Any

from harness.tools.safe_path_io import SafePathError, read_text_nofollow

_base = importlib.import_module("harness.tools._search_issue4_fixed_base")

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


def _python_grep(
    self: Any,
    pattern: str,
    path: str,
    include: str | None,
    max_results: int,
):
    """Search canonical files without following a post-authorization symlink."""

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return ToolResult(success=False, output="", error=f"Invalid regex: {exc}")

    results: list[str] = []
    search_path = Path(path)
    targets = [search_path] if search_path.is_file() else search_path.rglob("*")

    try:
        for candidate in targets:
            try:
                if not candidate.is_file():
                    continue
            except OSError as exc:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Cannot inspect grep candidate safely: {exc}",
                    metadata=policy_guard_metadata(
                        "canonical_path_guard",
                        canonical_path_checked=True,
                        nofollow_io=True,
                    ),
                )
            if include and not fnmatch.fnmatch(candidate.name, include):
                continue

            decision = resolve_guarded_path(
                candidate,
                operation="read",
                must_exist=True,
                allowed_root=search_path,
            )
            if not decision.allowed:
                return guarded_path_failure("grep result", decision)

            try:
                content, _ = read_text_nofollow(decision.resolved, errors="replace")
            except (OSError, SafePathError) as exc:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Cannot read grep candidate safely: {exc}",
                    metadata=policy_guard_metadata(
                        "canonical_path_guard",
                        canonical_path_checked=True,
                        nofollow_io=True,
                    ),
                )

            for line_number, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    results.append(
                        f"{decision.resolved}:{line_number}: {line.rstrip()}"
                    )
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break
    except OSError as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Cannot enumerate grep targets safely: {exc}",
            metadata=policy_guard_metadata(
                "canonical_path_guard",
                canonical_path_checked=True,
                nofollow_io=True,
            ),
        )

    return ToolResult(
        success=True,
        output="\n".join(results) if results else "(no matches)",
        metadata={
            "match_count": len(results),
            "engine": "python-nofollow",
            "canonical_paths": True,
            "nofollow_io": True,
        },
    )


def _execute_secure_grep(
    self: Any,
    pattern: str,
    path: str = ".",
    include: str | None = None,
    max_results: int | None = None,
    **kwargs: Any,
):
    """Run every content read through the descriptor-relative implementation."""

    limit = self.max_results if max_results is None else max_results
    if limit < 1:
        return ToolResult(success=False, output="", error="max_results must be >= 1")

    decision = resolve_guarded_path(
        path,
        operation="read",
        must_exist=True,
    )
    if not decision.allowed:
        return guarded_path_failure("grep", decision)

    root = Path(decision.resolved)
    if root == Path(root.anchor):
        return ToolResult(
            success=False,
            output="",
            error=(
                "Canonical path guard blocked grep: filesystem-root searches "
                "are outside the authorized task workspace."
            ),
            metadata=policy_guard_metadata("canonical_path_guard"),
        )

    if include:
        unsafe_reason = unsafe_relative_pattern_reason(include)
        if unsafe_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Canonical path guard blocked grep include: {unsafe_reason}.",
                metadata=policy_guard_metadata("canonical_path_guard"),
            )

    observed = guard_observed_text(
        " ".join(value for value in [path, pattern, include or ""] if value),
        operation="read",
    )
    if observed.blocked_by:
        return guarded_path_failure("grep", observed)

    symlink_failure = _base._preflight_symlink_tree(root, action="grep symlink")
    if symlink_failure is not None:
        return symlink_failure

    result = self._python_grep(pattern, decision.resolved, include, limit)
    result.metadata = {
        **result.metadata,
        "external_search_disabled_for_path_safety": True,
    }
    return result


_base.GrepTool._python_grep = _python_grep
_base.GrepTool.execute = _execute_secure_grep
GrepTool = _base.GrepTool
GlobTool = _base.GlobTool
