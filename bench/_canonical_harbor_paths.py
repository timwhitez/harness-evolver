"""Canonical path resolution for Harbor file tools."""

from __future__ import annotations

import base64
import binascii
import os
import shlex
from typing import Any

from bench import _harbor_adapter_issue4_base as _base
from harness.tools.base import ToolResult, policy_guard_metadata
from harness.tools.canonical_path_guard import (
    GuardedPath,
    guard_canonical_path_strings,
    guard_observed_text,
    guarded_path_failure,
    unsafe_requested_path_reason,
)

_CANONICALIZE_PYTHON = """
import base64
import os
from pathlib import Path

path = Path(os.environ["HL_FILE_PATH"])
must_exist = os.environ.get("HL_MUST_EXIST", "1") == "1"
resolved = path.resolve(strict=must_exist)
print(base64.b64encode(os.fsencode(str(resolved))).decode("ascii"))
"""

_CANONICALIZE_FALLBACK = r"""
set -eu
p=${HL_FILE_PATH:?}
must_exist=${HL_MUST_EXIST:-1}
resolved=''
if [ "$must_exist" = '1' ]; then
  [ -e "$p" ] || { echo 'path does not exist' >&2; exit 66; }
  if command -v realpath >/dev/null 2>&1; then
    resolved=$(realpath -e -- "$p" 2>/dev/null || realpath -- "$p")
  elif command -v readlink >/dev/null 2>&1; then
    resolved=$(readlink -f -- "$p")
  else
    echo 'realpath or readlink is required for canonical path checks' >&2
    exit 127
  fi
else
  if command -v realpath >/dev/null 2>&1; then
    resolved=$(realpath -m -- "$p" 2>/dev/null) || {
      echo 'realpath does not support safe non-existing target resolution' >&2
      exit 127
    }
  else
    echo 'realpath -m is required for safe writes without Python' >&2
    exit 127
  fi
fi
command -v base64 >/dev/null 2>&1 || {
  echo 'base64 is required for canonical path transport' >&2
  exit 127
}
printf '%s' "$resolved" | base64 | tr -d '\n'
"""


def _canonical_policy_block(action: str, requested: str, reason: str) -> ToolResult:
    return guarded_path_failure(
        action,
        GuardedPath(
            requested=requested,
            operation="read",
            blocked_by="canonical_path_guard",
            reason=reason,
        ),
    )


class _CanonicalHarborPathMixin:
    """Resolve paths inside the target container before authorizing them."""

    def _guard_environment_path(
        self,
        path: str,
        *,
        operation: str,
        must_exist: bool,
        allowed_root: str | None = None,
    ) -> tuple[str, None] | tuple[None, ToolResult]:
        unsafe_reason = unsafe_requested_path_reason(path)
        if unsafe_reason:
            return None, _canonical_policy_block(operation, path, unsafe_reason)
        lexical = guard_observed_text(path, operation=operation)
        if lexical.blocked_by:
            return None, guarded_path_failure(operation, lexical)

        env = {
            "HL_FILE_PATH": path,
            "HL_MUST_EXIST": "1" if must_exist else "0",
        }
        result = self._exec(
            "python3 -c " + shlex.quote(_CANONICALIZE_PYTHON),
            env=env,
        )
        output = result.stdout or ""
        error = result.stderr or ""
        if result.return_code != 0 and _base._looks_like_missing_python(error, output):
            result = self._exec(
                f"sh -c {shlex.quote(_CANONICALIZE_FALLBACK)}",
                env=env,
            )
            output = result.stdout or ""
            error = result.stderr or ""

        unavailable = _base._terminal_environment_unavailable_text(output, error)
        if unavailable:
            return None, _base._terminal_environment_unavailable_result(
                output=output,
                stderr=error,
                message=unavailable,
                exit_code=result.return_code,
            )
        if result.return_code != 0:
            return None, ToolResult(
                success=False,
                output="",
                error=f"Cannot resolve path safely inside the task environment: {path}",
                metadata=policy_guard_metadata(
                    "canonical_path_guard",
                    canonical_path_checked=True,
                    path_resolution_failed=True,
                    exit_code=result.return_code,
                ),
            )

        try:
            raw = base64.b64decode(output.strip(), validate=True)
            resolved = os.fsdecode(raw)
        except (binascii.Error, ValueError, UnicodeError) as exc:
            return None, ToolResult(
                success=False,
                output="",
                error=f"Canonical path resolver returned invalid data: {exc}",
                metadata=policy_guard_metadata("canonical_path_guard"),
            )

        decision = guard_canonical_path_strings(
            requested=path,
            resolved=resolved,
            operation=operation,
            allowed_root=allowed_root,
        )
        if not decision.allowed:
            return None, guarded_path_failure(operation, decision)
        return resolved, None

    def _guard_environment_matches(
        self,
        matches: list[str],
        *,
        requested: str,
        root: str,
        action: str,
    ) -> tuple[list[str], None] | tuple[None, ToolResult]:
        safe: set[str] = set()
        for match in matches:
            decision = guard_canonical_path_strings(
                requested=requested,
                resolved=match,
                operation="read",
                allowed_root=root,
            )
            if not decision.allowed:
                return None, guarded_path_failure(action, decision)
            safe.add(decision.resolved)
        return sorted(safe), None


class HarborFileReadTool(_CanonicalHarborPathMixin, _base.HarborFileReadTool):
    def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int | None = 2000,
        **kwargs: Any,
    ) -> ToolResult:
        resolved, failure = self._guard_environment_path(
            file_path,
            operation="read",
            must_exist=True,
        )
        if failure is not None:
            return failure
        return super().execute(
            file_path=resolved,
            offset=offset,
            limit=limit,
            **kwargs,
        )


class HarborFileWriteTool(_CanonicalHarborPathMixin, _base.HarborFileWriteTool):
    def execute(
        self,
        file_path: str,
        content: str,
        append: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        resolved, failure = self._guard_environment_path(
            file_path,
            operation="write",
            must_exist=False,
        )
        if failure is not None:
            return failure
        return super().execute(
            file_path=resolved,
            content=content,
            append=append,
            **kwargs,
        )


class HarborFileEditTool(_CanonicalHarborPathMixin, _base.HarborFileEditTool):
    def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        resolved, failure = self._guard_environment_path(
            file_path,
            operation="edit",
            must_exist=True,
        )
        if failure is not None:
            return failure
        return super().execute(
            file_path=resolved,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            **kwargs,
        )

