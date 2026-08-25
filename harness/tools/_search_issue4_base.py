"""Grep and Glob tools — search without embeddings.

Speed matters.  Ripgrep over traditional grep.  No embedding-based
search — explicit patterns are more reliable and faster.
"""

from __future__ import annotations

import fnmatch
import subprocess
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
class GrepTool(ToolDef):
    """Fast regex search using ripgrep (falls back to Python re).

    Key design insight from Factory Droid: ripgrep over traditional grep
    for minimal feedback latency.
    """

    name: str = "grep"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = (
        "Search for a regex pattern in files. Fast recursive search. "
        "Use for finding function definitions, variable usages, error messages, "
        "imports, and any text pattern in the codebase."
    )
    max_results: int = 200

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: current dir).",
                    },
                    "include": {
                        "type": "string",
                        "description": "File pattern to include (e.g., '*.py', '*.ts').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": f"Maximum results to return (default: {self.max_results}).",
                    },
                },
                "required": ["pattern"],
            },
        )

    def execute(
        self,
        pattern: str,
        path: str = ".",
        include: str | None = None,
        max_results: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        max_results = max_results or self.max_results
        prohibited_reason = prohibited_path_reason(path, operation="read")
        if prohibited_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Leaderboard integrity guard blocked grep: {prohibited_reason}.",
                metadata=policy_guard_metadata("leaderboard_integrity_guard"),
            )
        observed = " ".join(value for value in [path, pattern, include or ""] if value)
        if host_memory_access_reason(observed):
            return ToolResult(
                success=False,
                output="",
                error=host_memory_blocked_error(observed),
                metadata=host_memory_block_metadata(),
            )

        # Try ripgrep first
        cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
        if include:
            cmd.extend(["--glob", include])
        cmd.extend([pattern, path])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if len(lines) > max_results:
                lines = lines[:max_results]
                lines.append(f"... ({len(lines) - max_results} more results truncated)")

            return ToolResult(
                success=True,
                output="\n".join(lines) if lines else "(no matches)",
                metadata={
                    "match_count": len(lines),
                    "truncated": len(lines) > max_results,
                    "engine": "ripgrep",
                },
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

        # Fall back to Python implementation
        return self._python_grep(pattern, path, include, max_results)

    def _python_grep(
        self, pattern: str, path: str, include: str | None, max_results: int
    ) -> ToolResult:
        import re

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid regex: {e}")

        results = []
        search_path = Path(path)
        targets = [search_path] if search_path.is_file() else search_path.rglob("*")

        for file_path in targets:
            if not file_path.is_file():
                continue
            if include and not fnmatch.fnmatch(file_path.name, include):
                continue
            if len(results) >= max_results:
                break

            try:
                for i, line in enumerate(file_path.read_text(errors="replace").split("\n"), 1):
                    if regex.search(line):
                        results.append(f"{file_path}:{i}: {line.rstrip()}")
                        if len(results) >= max_results:
                            break
            except Exception:
                continue

        return ToolResult(
            success=True,
            output="\n".join(results) if results else "(no matches)",
            metadata={"match_count": len(results), "engine": "python"},
        )


@dataclass
class GlobTool(ToolDef):
    """File pattern matching using glob patterns."""

    name: str = "glob"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = (
        "Find files matching a glob pattern. "
        "Use for discovering file structure, finding all files of a type, "
        "or locating configuration files."
    )

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.tsx').",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current dir).",
                    },
                },
                "required": ["pattern"],
            },
        )

    def execute(self, pattern: str, path: str = ".", **kwargs: Any) -> ToolResult:
        prohibited_reason = prohibited_path_reason(path, operation="read")
        if prohibited_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Leaderboard integrity guard blocked glob: {prohibited_reason}.",
                metadata=policy_guard_metadata("leaderboard_integrity_guard"),
            )
        observed = f"{path} {pattern}"
        if host_memory_access_reason(observed):
            return ToolResult(
                success=False,
                output="",
                error=host_memory_blocked_error(observed),
                metadata=host_memory_block_metadata(),
            )
        search_path = Path(path)
        matches = sorted(str(p) for p in search_path.glob(pattern))

        if not matches:
            return ToolResult(success=True, output="(no matches)", metadata={"match_count": 0})

        output = "\n".join(matches[:500])
        if len(matches) > 500:
            output += f"\n... ({len(matches) - 500} more matches)"

        return ToolResult(
            success=True,
            output=output,
            metadata={"match_count": len(matches)},
        )
