"""Canonical-path protected Harbor glob tool."""

from __future__ import annotations

import shlex
from typing import Any

from bench import _harbor_adapter_issue4_base as _base
from bench._canonical_harbor_paths import (
    _CanonicalHarborPathMixin,
    _canonical_policy_block,
)
from harness.tools.base import ToolResult
from harness.tools.canonical_path_guard import (
    guard_observed_text,
    guarded_path_failure,
    unsafe_relative_pattern_reason,
)

_HARBOR_GLOB_PYTHON = r"""
import os
from pathlib import Path

root = Path(os.environ["HL_ROOT"]).resolve(strict=True)
pattern = os.environ["HL_PATTERN"]
safe = set()


def prohibited(path: str) -> bool:
    lowered = path.lower().replace("\\", "/")
    if "/terminal-bench-tasks/" in lowered or lowered.endswith("/terminal-bench-tasks"):
        return True
    if lowered == "/tests" or lowered.startswith("/tests/"):
        return True
    if lowered in {"/solution", "/solutions"}:
        return True
    if lowered.startswith("/solution/") or lowered.startswith("/solutions/"):
        return True
    if any(marker in lowered for marker in ("/trials/runs", "/trials/summaries", "/host/trials")):
        return True
    terminal_markers = ("/terminal-bench/", "/terminal_bench/", "/terminalbench/")
    if any(marker in lowered for marker in terminal_markers):
        if any(marker in lowered for marker in ("/tests/", "/solutions/", "/solution/")):
            return True
        if lowered.endswith("/task.toml"):
            return True
    return False

for directory, dirnames, filenames in os.walk(root, followlinks=False):
    directory_path = Path(directory)
    for name in [*dirnames, *filenames]:
        candidate = directory_path / name
        if not candidate.is_symlink():
            continue
        resolved = candidate.resolve(strict=True)
        if resolved != root and root not in resolved.parents:
            raise SystemExit(73)
        if prohibited(str(resolved)):
            raise SystemExit(73)

for candidate in root.glob(pattern):
    resolved = candidate.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise SystemExit(73)
    text = str(resolved)
    if prohibited(text):
        raise SystemExit(73)
    safe.add(text)
    if len(safe) > 500:
        break
for match in sorted(safe)[:500]:
    print(match)
"""

_HARBOR_GLOB_FALLBACK = r"""
set -eu
root=${HL_ROOT:?}
pattern=${HL_PATTERN:?}
name=${pattern##*/}
[ -n "$name" ] || name='*'
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT INT TERM
out=$tmpdir/output
blocked=$tmpdir/blocked
: > "$out"
: > "$blocked"
export HL_GLOB_OUT="$out" HL_GLOB_BLOCKED="$blocked" HL_ROOT_CANON="$root"
check='for f do
  if command -v realpath >/dev/null 2>&1; then
    n=$(realpath -e -- "$f" 2>/dev/null || realpath -- "$f") || exit 74
  elif command -v readlink >/dev/null 2>&1; then
    n=$(readlink -f -- "$f") || exit 74
  else
    exit 127
  fi
  case "$n" in "$HL_ROOT_CANON"|"$HL_ROOT_CANON"/*) ;; *) echo 1 > "$HL_GLOB_BLOCKED"; continue ;; esac
  case "$n" in
    */terminal-bench-tasks|*/terminal-bench-tasks/*|/tests|/tests/*|/solutions|/solutions/*|/solution|/solution/*|*/trials/runs|*/trials/runs/*|*/trials/summaries|*/trials/summaries/*|/host/trials|/host/trials/*)
      echo 1 > "$HL_GLOB_BLOCKED"; continue ;;
  esac
  case "$n" in
    */terminal-bench/*/tests/*|*/terminal-bench/*/solutions/*|*/terminal-bench/*/solution/*|*/terminal-bench/*/task.toml|*/terminal_bench/*/tests/*|*/terminal_bench/*/solutions/*|*/terminal_bench/*/solution/*|*/terminal_bench/*/task.toml|*/terminalbench/*/tests/*|*/terminalbench/*/solutions/*|*/terminalbench/*/solution/*|*/terminalbench/*/task.toml)
      echo 1 > "$HL_GLOB_BLOCKED"; continue ;;
  esac
  printf "%s\\n" "$n" >> "$HL_GLOB_OUT"
done'
find -P "$root" -name "$name" -exec sh -c "$check" sh {} +
[ ! -s "$blocked" ] || { echo 'canonical path policy blocked a glob target' >&2; exit 73; }
sort -u "$out" | sed -n '1,500p'
"""


class HarborGlobTool(_CanonicalHarborPathMixin, _base.HarborGlobTool):
    def execute(self, pattern: str, path: str = ".", **_: Any) -> ToolResult:
        unsafe_reason = unsafe_relative_pattern_reason(pattern)
        if unsafe_reason:
            return _canonical_policy_block("glob", path, unsafe_reason)
        observed = guard_observed_text(f"{path} {pattern}", operation="read")
        if observed.blocked_by:
            return guarded_path_failure("glob", observed)

        root, failure = self._guard_environment_path(
            path,
            operation="read",
            must_exist=True,
        )
        if failure is not None:
            return failure

        env = {"HL_ROOT": root, "HL_PATTERN": pattern}
        result = self._exec(
            "python3 -c " + shlex.quote(_HARBOR_GLOB_PYTHON),
            env=env,
        )
        output = result.stdout or ""
        error = result.stderr or ""
        if result.return_code != 0 and _base._looks_like_missing_python(error, output):
            result = self._exec(
                f"sh -c {shlex.quote(_HARBOR_GLOB_FALLBACK)}",
                env=env,
            )
            output = result.stdout or ""
            error = result.stderr or ""

        unavailable = _base._terminal_environment_unavailable_text(output, error)
        if unavailable:
            return _base._terminal_environment_unavailable_result(
                output=output,
                stderr=error,
                message=unavailable,
                exit_code=result.return_code,
            )
        if result.return_code == 73:
            return _canonical_policy_block(
                "glob result",
                path,
                "a resolved glob target was prohibited or escaped the authorized root",
            )
        if result.return_code != 0:
            return ToolResult(
                success=False,
                output=output,
                error=error or f"exit code: {result.return_code}",
                metadata={"exit_code": result.return_code},
            )

        matches = [line for line in output.splitlines() if line]
        safe, match_failure = self._guard_environment_matches(
            matches,
            requested=path,
            root=root,
            action="glob result",
        )
        if match_failure is not None:
            return match_failure
        return ToolResult(
            success=True,
            output="\n".join(safe) if safe else "(no matches)",
            error="",
            metadata={
                "exit_code": result.return_code,
                "match_count": len(safe),
                "canonical_paths": True,
            },
        )


