"""Canonical-path search tools with stable descriptor-relative traversal."""

from __future__ import annotations

from contextlib import closing
import fnmatch
import importlib
import io
import os
from pathlib import Path
import re
from typing import Any

from harness.tools.canonical_path_guard import guard_canonical_path_strings
from harness.tools.stable_tree import StableTreeError, iter_stable_regular_files

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
    *,
    expected_root: os.stat_result | None = None,
):
    """Search only files reachable from the exact directory inode authorized."""

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return ToolResult(success=False, output="", error=f"Invalid regex: {exc}")

    results: list[str] = []
    search_path = Path(path)
    root_text = str(search_path)

    try:
        with closing(
            iter_stable_regular_files(
                search_path,
                expected_root=expected_root,
            )
        ) as targets:
            for relative, binary_stream, _metadata in targets:
                candidate = search_path if not relative.parts else search_path / relative
                if include and not fnmatch.fnmatch(candidate.name, include):
                    continue

                decision = guard_canonical_path_strings(
                    requested=str(candidate),
                    resolved=str(candidate),
                    operation="read",
                    allowed_root=root_text,
                )
                if not decision.allowed:
                    return guarded_path_failure("grep result", decision)

                with io.TextIOWrapper(
                    binary_stream,
                    encoding="utf-8",
                    errors="replace",
                    newline=None,
                ) as content:
                    for line_number, line in enumerate(content, 1):
                        if regex.search(line):
                            results.append(
                                f"{candidate}:{line_number}: {line.rstrip()}"
                            )
                            if len(results) >= max_results:
                                break
                if len(results) >= max_results:
                    break
    except (OSError, StableTreeError) as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Cannot traverse grep root safely: {exc}",
            metadata=policy_guard_metadata(
                "canonical_path_guard",
                canonical_path_checked=True,
                nofollow_io=True,
                stable_root_descriptor=True,
            ),
        )

    return ToolResult(
        success=True,
        output="\n".join(results) if results else "(no matches)",
        metadata={
            "match_count": len(results),
            "engine": "python-stable-nofollow",
            "canonical_paths": True,
            "nofollow_io": True,
            "stable_root_descriptor": True,
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
    """Authorize once, then keep traversal bound to that exact root inode."""

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

    try:
        expected_root = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Cannot capture authorized grep root identity: {exc}",
            metadata=policy_guard_metadata(
                "canonical_path_guard",
                canonical_path_checked=True,
                stable_root_descriptor=True,
            ),
        )

    symlink_failure = _base._preflight_symlink_tree(root, action="grep symlink")
    if symlink_failure is not None:
        return symlink_failure

    result = self._python_grep(
        pattern,
        decision.resolved,
        include,
        limit,
        expected_root=expected_root,
    )
    result.metadata = {
        **result.metadata,
        "external_search_disabled_for_path_safety": True,
    }
    return result


_base.GrepTool._python_grep = _python_grep
_base.GrepTool.execute = _execute_secure_grep
GrepTool = _base.GrepTool
GlobTool = _base.GlobTool
