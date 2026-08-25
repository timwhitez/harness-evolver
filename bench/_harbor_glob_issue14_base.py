"""Python-free glob fallback layered on the canonical Harbor glob tool."""

from __future__ import annotations

import shlex
from typing import Any

from bench import _harbor_adapter_issue14_base as _stack

_registry_module = _stack._registry_module
_original = _registry_module._base
HarborGlobTool = _registry_module.HarborGlobTool

# The primary Harbor implementation evaluates:
#
#   glob.glob(os.path.join(root, pattern), recursive=True)[:500]
#
# Bash globstar preserves the supported ordinary-tree wildcard set. Cases whose
# behavior cannot be reproduced safely without Python fail explicitly.
_GLOB_FALLBACK_SCRIPT = r'''set -eu
root=${HL_ROOT:-.}
pattern=${HL_PATTERN:?}
case "$root" in
  *"
"*|*""*) echo 'glob root containing newline or carriage return is unsupported' >&2; exit 65 ;;
esac
[ -d "$root" ] || { echo "Glob root is not a directory: $root" >&2; exit 1; }
command -v bash >/dev/null 2>&1 || { echo 'bash is required for semantics-preserving glob fallback' >&2; exit 127; }
command -v find >/dev/null 2>&1 || { echo 'find is required for glob fallback preflight' >&2; exit 127; }
exec bash -c '
set -u
root=$1
pattern=$2
if ! shopt -s globstar nullglob; then
  echo "Bash globstar support is required" >&2
  exit 127
fi
case "$root" in
  /) combined="/$pattern" ;;
  */) combined="${root}${pattern}" ;;
  *) combined="${root}/$pattern" ;;
esac

tmpdir=$(mktemp -d)
trap '\''rm -rf "$tmpdir"'\'' EXIT INT TERM
bad=$tmpdir/bad
matches=$tmpdir/matches
: > "$bad"
: > "$matches"
export HL_BAD="$bad"

# compgen emits newline-delimited paths. Preflight candidate names using argv
# transport so embedded newlines can never be split into fabricated matches.
find -P "$root" -mindepth 1 -exec sh -c '\''
for candidate do
  case "$candidate" in
    *"
"*|*""*) printf "%s\n" newline > "$HL_BAD"; exit 0 ;;
  esac
done
'\'' sh {} +
if [ -s "$bad" ]; then
  echo "glob tree contains a newline or carriage-return filename" >&2
  exit 65
fi

# Python glob follows symlink directories for recursive ** traversal; Bash
# globstar does not. Refuse that tree instead of returning incomplete results.
case "/$pattern/" in
  */\*\*/*)
    : > "$bad"
    find -P "$root" -type l -exec sh -c '\''
    for candidate do
      if [ -d "$candidate" ]; then
        printf "%s\n" symlink-directory > "$HL_BAD"
        exit 0
      fi
    done
    '\'' sh {} +
    if [ -s "$bad" ]; then
      echo "recursive glob through symlink directories is unsupported without Python" >&2
      exit 66
    fi
    ;;
esac

set +e
compgen -G "$combined" | head -n 501 > "$matches"
statuses=("${PIPESTATUS[@]}")
producer=${statuses[0]:-1}
consumer=${statuses[1]:-1}
if [ "$consumer" -ne 0 ]; then exit "$consumer"; fi
case "$producer" in
  0|1|141) ;;
  *) exit "$producer" ;;
esac

count=$(wc -l < "$matches")
if [ "$count" -gt 500 ]; then
  echo "Python-free glob produced more than 500 matches; fallback ordering cannot reproduce Python glob slicing" >&2
  exit 67
fi
cat "$matches"
' bash "$root" "$pattern"
'''


def _unsupported_glob_reason(pattern: str) -> str:
    if not pattern:
        return "empty glob patterns are unsupported"
    if pattern.startswith("/"):
        return "absolute glob patterns are unsupported; pass the root via path"
    if any(character in pattern for character in ("\x00", "\n", "\r", "\\")):
        return "NUL, newline, carriage return, and backslash are unsupported in fallback globs"
    components = pattern.removeprefix("./").split("/")
    if any(component == ".." for component in components):
        return "parent-directory traversal is unsupported in fallback globs"
    for component in components:
        cursor = 0
        while cursor < len(component):
            if component[cursor] != "[":
                cursor += 1
                continue
            close = component.find("]", cursor + 1)
            if close == -1:
                return "unterminated character class in fallback glob"
            cursor = close + 1
    return ""


def _unsupported_glob_root_reason(path: str) -> str:
    if any(character in path for character in ("\x00", "\n", "\r")):
        return "NUL, newline, and carriage return are unsupported in fallback glob roots"
    return ""


def _glob_without_python(
    self: Any,
    *,
    pattern: str,
    path: str = ".",
):
    unsupported = _unsupported_glob_reason(pattern)
    if unsupported:
        return _original.ToolResult(
            success=False,
            output="",
            error=f"Unsupported Python-free glob pattern: {unsupported}",
            metadata={
                "engine": "bash-compgen",
                "glob_pattern_unsupported": True,
            },
        )
    root_unsupported = _unsupported_glob_root_reason(path)
    if root_unsupported:
        return _original.ToolResult(
            success=False,
            output="",
            error=f"Unsupported Python-free glob root: {root_unsupported}",
            metadata={
                "engine": "bash-compgen",
                "glob_root_unsupported": True,
            },
        )

    result = self._exec(
        f"sh -c {shlex.quote(_GLOB_FALLBACK_SCRIPT)}",
        env={"HL_ROOT": path, "HL_PATTERN": pattern},
    )
    output = result.stdout or ""
    error = result.stderr or (
        "" if result.return_code == 0 else f"exit code: {result.return_code}"
    )
    unavailable = _original._terminal_environment_unavailable_text(output, error)
    if unavailable:
        return _original._terminal_environment_unavailable_result(
            output=output,
            stderr=error,
            message=unavailable,
            exit_code=result.return_code,
        )
    if result.return_code == 65:
        return _original.ToolResult(
            success=False,
            output="",
            error=(
                "Python-free glob cannot represent a root/tree containing "
                "newline or carriage-return path components unambiguously."
            ),
            metadata={
                "exit_code": 65,
                "engine": "bash-compgen",
                "glob_filename_unsupported": True,
            },
        )
    if result.return_code == 66:
        return _original.ToolResult(
            success=False,
            output="",
            error=(
                "Python-free recursive glob cannot preserve Python glob traversal "
                "through symlink directories; refusing an incomplete result."
            ),
            metadata={
                "exit_code": 66,
                "engine": "bash-compgen",
                "recursive_symlink_glob_unsupported": True,
            },
        )
    if result.return_code == 67:
        return _original.ToolResult(
            success=False,
            output="",
            error=(
                "Python-free glob found more than 500 matches; fallback ordering "
                "cannot reproduce the primary Python glob [:500] selection."
            ),
            metadata={
                "exit_code": 67,
                "engine": "bash-compgen",
                "glob_result_limit_unsupported": True,
            },
        )
    if result.return_code != 0:
        return _original.ToolResult(
            success=False,
            output=output,
            error=error,
            metadata={
                "exit_code": result.return_code,
                "engine": "bash-compgen",
                "glob_failed": True,
            },
        )

    matches = output.splitlines()
    return _original.ToolResult(
        success=True,
        output=output.rstrip("\n") if matches else "(no matches)",
        error="",
        metadata={
            "exit_code": 0,
            "engine": "bash-compgen",
            "match_count": len(matches),
            "glob_semantics": "python-glob-recursive",
            "hidden_components": "python-glob-default",
        },
    )


HarborGlobTool._glob_without_python = _glob_without_python
