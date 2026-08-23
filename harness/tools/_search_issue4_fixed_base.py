"""Canonical-path protected grep and glob tools."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from harness.tools._search_issue4_base import GrepTool as _BaseGrepTool
from harness.tools._search_issue4_base import GlobTool as _BaseGlobTool
from harness.tools.base import ToolResult, policy_guard_metadata
from harness.tools.canonical_path_guard import (
    guard_observed_text,
    guarded_path_failure,
    resolve_guarded_path,
    unsafe_relative_pattern_reason,
)


def _preflight_symlink_tree(root: Path, *, action: str) -> ToolResult | None:
    if not root.is_dir():
        return None
    try:
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            for name in [*dirnames, *filenames]:
                candidate = directory_path / name
                if not candidate.is_symlink():
                    continue
                decision = resolve_guarded_path(
                    candidate,
                    operation="read",
                    must_exist=True,
                    allowed_root=root,
                )
                if not decision.allowed:
                    return guarded_path_failure(action, decision)
    except (OSError, RuntimeError) as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Cannot preflight search symlinks safely: {exc}",
            metadata=policy_guard_metadata("canonical_path_guard"),
        )
    return None


@dataclass
class GrepTool(_BaseGrepTool):
    """Search a canonical root without following unvalidated file symlinks."""

    def execute(
        self,
        pattern: str,
        path: str = ".",
        include: str | None = None,
        max_results: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        max_results = max_results or self.max_results
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

        symlink_failure = _preflight_symlink_tree(root, action="grep symlink")
        if symlink_failure is not None:
            return symlink_failure

        command = [
            "rg",
            "--json",
            "--no-follow",
            "--color=never",
            "--glob",
            "!**/terminal-bench-tasks/**",
            "--glob",
            "!**/terminal-bench/**/tests/**",
            "--glob",
            "!**/terminal-bench/**/solutions/**",
            "--glob",
            "!**/terminal_bench/**/tests/**",
            "--glob",
            "!**/terminal_bench/**/solutions/**",
            "--glob",
            "!**/terminalbench/**/tests/**",
            "--glob",
            "!**/terminalbench/**/solutions/**",
            "--glob",
            "!**/trials/runs/**",
            "--glob",
            "!**/trials/summaries/**",
            "--glob",
            "!**/host/trials/**",
        ]
        if include:
            command.extend(["--glob", include])
        command.extend([pattern, decision.resolved])

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            return self._python_grep(pattern, decision.resolved, include, max_results)
        except subprocess.SubprocessError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        if completed.returncode == 1:
            return ToolResult(
                success=True,
                output="(no matches)",
                metadata={"match_count": 0, "engine": "ripgrep", "canonical_paths": True},
            )
        if completed.returncode != 0:
            return ToolResult(
                success=False,
                output="",
                error=completed.stderr.strip() or f"ripgrep exit code: {completed.returncode}",
                metadata={"exit_code": completed.returncode, "engine": "ripgrep"},
            )

        results: list[str] = []
        for raw_line in completed.stdout.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                return ToolResult(
                    success=False,
                    output="",
                    error="ripgrep returned malformed JSON search output",
                    metadata=policy_guard_metadata("canonical_path_guard"),
                )
            if event.get("type") != "match":
                continue
            data = event.get("data", {})
            path_text = data.get("path", {}).get("text")
            line_text = data.get("lines", {}).get("text", "").rstrip("\n")
            line_number = data.get("line_number")
            if not isinstance(path_text, str):
                return ToolResult(
                    success=False,
                    output="",
                    error="ripgrep match omitted a textual path",
                    metadata=policy_guard_metadata("canonical_path_guard"),
                )
            candidate = Path(path_text)
            if not candidate.is_absolute():
                candidate = root / candidate
            match = resolve_guarded_path(
                candidate,
                operation="read",
                must_exist=True,
                allowed_root=root,
            )
            if not match.allowed:
                return guarded_path_failure("grep result", match)
            results.append(f"{match.resolved}:{line_number}: {line_text}")
            if len(results) >= max_results:
                break

        return ToolResult(
            success=True,
            output="\n".join(results) if results else "(no matches)",
            metadata={
                "match_count": len(results),
                "engine": "ripgrep",
                "canonical_paths": True,
                "truncated": len(results) >= max_results,
            },
        )

    def _python_grep(
        self,
        pattern: str,
        path: str,
        include: str | None,
        max_results: int,
    ) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(success=False, output="", error=f"Invalid regex: {exc}")

        results: list[str] = []
        search_path = Path(path)
        targets = [search_path] if search_path.is_file() else search_path.rglob("*")

        for candidate in targets:
            if not candidate.is_file():
                continue
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

            safe_file = Path(decision.resolved)
            try:
                for line_number, line in enumerate(
                    safe_file.read_text(errors="replace").split("\n"),
                    1,
                ):
                    if regex.search(line):
                        results.append(f"{safe_file}:{line_number}: {line.rstrip()}")
                        if len(results) >= max_results:
                            break
            except Exception:
                continue
            if len(results) >= max_results:
                break

        return ToolResult(
            success=True,
            output="\n".join(results) if results else "(no matches)",
            metadata={"match_count": len(results), "engine": "python"},
        )


@dataclass
class GlobTool(_BaseGlobTool):
    """Return only canonical matches contained by the authorized root."""

    def execute(self, pattern: str, path: str = ".", **kwargs: Any) -> ToolResult:
        decision = resolve_guarded_path(
            path,
            operation="read",
            must_exist=True,
        )
        if not decision.allowed:
            return guarded_path_failure("glob", decision)

        unsafe_reason = unsafe_relative_pattern_reason(pattern)
        if unsafe_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Canonical path guard blocked glob: {unsafe_reason}.",
                metadata=policy_guard_metadata("canonical_path_guard"),
            )

        observed = guard_observed_text(f"{path} {pattern}", operation="read")
        if observed.blocked_by:
            return guarded_path_failure("glob", observed)

        root = Path(decision.resolved)
        symlink_failure = _preflight_symlink_tree(root, action="glob symlink")
        if symlink_failure is not None:
            return symlink_failure

        safe_matches: set[str] = set()
        try:
            candidates = root.glob(pattern)
            for candidate in candidates:
                match = resolve_guarded_path(
                    candidate,
                    operation="read",
                    must_exist=True,
                    allowed_root=root,
                )
                if not match.allowed:
                    return guarded_path_failure("glob result", match)
                safe_matches.add(match.resolved)
        except (OSError, RuntimeError) as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Cannot evaluate glob safely: {exc}",
                metadata=policy_guard_metadata("canonical_path_guard"),
            )

        matches = sorted(safe_matches)
        if not matches:
            return ToolResult(success=True, output="(no matches)", metadata={"match_count": 0})

        output = "\n".join(matches[:500])
        if len(matches) > 500:
            output += f"\n... ({len(matches) - 500} more matches)"
        return ToolResult(
            success=True,
            output=output,
            metadata={"match_count": len(matches), "canonical_paths": True},
        )
