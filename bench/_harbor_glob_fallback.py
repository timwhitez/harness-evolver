"""Semantics-preserving Python-free fallback for canonical Harbor glob."""

from __future__ import annotations

import shlex
from typing import Any

from bench import _canonical_harbor_glob as _canonical
from harness.tools.base import ToolResult


_GLOB_FALLBACK_SCRIPT = r"""set -eu
export LC_ALL=C
root=${HL_ROOT:?}
pattern=${HL_PATTERN:?}
recursive=${HL_RECURSIVE:-0}

for command_name in bash find head sort wc mktemp; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "$command_name is required for Python-free glob fallback" >&2
    exit 127
  }
done
if command -v realpath >/dev/null 2>&1; then
  resolver=$(command -v realpath)
  resolver_mode=realpath
elif command -v readlink >/dev/null 2>&1; then
  resolver=$(command -v readlink)
  resolver_mode=readlink
else
  echo "realpath or readlink -f is required for Python-free glob fallback" >&2
  exit 127
fi

[ -d "$root" ] || {
  echo "Glob root is not a directory: $root" >&2
  exit 1
}

tmpdir=$(mktemp -d)
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT INT TERM
bad_chars=$tmpdir/bad-chars
bad_link=$tmpdir/bad-link
recursive_link=$tmpdir/recursive-link
raw=$tmpdir/raw
canonical=$tmpdir/canonical
sorted=$tmpdir/sorted
: > "$bad_chars"
: > "$bad_link"
: > "$recursive_link"
: > "$raw"
: > "$canonical"
: > "$sorted"

export HL_GLOB_ROOT="$root"
export HL_GLOB_RESOLVER="$resolver"
export HL_GLOB_RESOLVER_MODE="$resolver_mode"
export HL_GLOB_BAD_CHARS="$bad_chars"
export HL_GLOB_BAD_LINK="$bad_link"
export HL_GLOB_RECURSIVE_LINK="$recursive_link"
export HL_GLOB_RECURSIVE="$recursive"

find -P "$root" -mindepth 1 -exec sh -c '
for candidate do
  case "$candidate" in
    *[![:print:]]*)
      printf "%s\n" unsupported > "$HL_GLOB_BAD_CHARS"
      exit 0
      ;;
  esac
done
' sh {} +
if [ -s "$bad_chars" ]; then
  echo "glob tree contains non-printable or non-ASCII path components" >&2
  exit 65
fi

find -P "$root" -type l -exec sh -c '
resolve_candidate() {
  if [ "$HL_GLOB_RESOLVER_MODE" = realpath ]; then
    "$HL_GLOB_RESOLVER" -e -- "$1"
  else
    "$HL_GLOB_RESOLVER" -f -- "$1"
  fi
}
for candidate do
  resolved=$(resolve_candidate "$candidate") || {
    printf "%s\n" broken > "$HL_GLOB_BAD_LINK"
    continue
  }
  case "$resolved" in
    "$HL_GLOB_ROOT"|"$HL_GLOB_ROOT"/*) ;;
    *)
      printf "%s\n" escape > "$HL_GLOB_BAD_LINK"
      continue
      ;;
  esac
  case "$resolved" in
    */terminal-bench-tasks|*/terminal-bench-tasks/*|/tests|/tests/*|/solutions|/solutions/*|/solution|/solution/*|*/trials/runs|*/trials/runs/*|*/trials/summaries|*/trials/summaries/*|/host/trials|/host/trials/*)
      printf "%s\n" prohibited > "$HL_GLOB_BAD_LINK"
      continue
      ;;
  esac
  case "$resolved" in
    */terminal-bench/*/tests/*|*/terminal-bench/*/solutions/*|*/terminal-bench/*/solution/*|*/terminal-bench/*/task.toml|*/terminal_bench/*/tests/*|*/terminal_bench/*/solutions/*|*/terminal_bench/*/solution/*|*/terminal_bench/*/task.toml|*/terminalbench/*/tests/*|*/terminalbench/*/solutions/*|*/terminalbench/*/solution/*|*/terminalbench/*/task.toml)
      printf "%s\n" prohibited > "$HL_GLOB_BAD_LINK"
      continue
      ;;
  esac
  if [ "$HL_GLOB_RECURSIVE" = 1 ] && [ -d "$candidate" ]; then
    printf "%s\n" recursive-symlink > "$HL_GLOB_RECURSIVE_LINK"
  fi
done
' sh {} +

if [ -s "$bad_link" ]; then
  echo "canonical path policy blocked a symlink in the glob tree" >&2
  exit 73
fi
if [ -s "$recursive_link" ]; then
  echo "recursive Python-free glob through symlink directories is unsupported" >&2
  exit 66
fi

unset GLOBIGNORE
shopt -u nocaseglob failglob extglob 2>/dev/null || true
if ! shopt -s dotglob globstar nullglob; then
  echo "Bash globstar, dotglob, and nullglob support are required" >&2
  exit 127
fi
shopt -s globskipdots 2>/dev/null || true

# Bash compgen lazily initializes dotglob. Trigger that initialization in an
# isolated one-entry directory instead of expanding the potentially large root.
probe=$tmpdir/probe
mkdir "$probe"
: > "$probe/.dotglob-probe"
old_pwd=$PWD
cd "$probe"
: *
cd "$old_pwd"

case "$root" in
  /) combined="/$pattern" ;;
  */) combined="${root}${pattern}" ;;
  *) combined="${root}/$pattern" ;;
esac
# pathlib treats a terminal ``**`` as a recursive directory selector. Bash
# globstar includes files unless the pattern has a trailing slash.
case "$pattern" in
  "**"|*/"**") combined="${combined}/" ;;
esac

set +e
compgen -G "$combined" | head -n 502 > "$raw"
statuses=("${PIPESTATUS[@]}")
set -e
producer=${statuses[0]:-1}
consumer=${statuses[1]:-1}
[ "$consumer" -eq 0 ] || exit "$consumer"
case "$producer" in
  0|1|141) ;;
  *) exit "$producer" ;;
esac

raw_count=$(wc -l < "$raw")
if [ "$raw_count" -gt 500 ]; then
  echo "Python-free glob found more than 500 raw matches" >&2
  exit 67
fi

resolve_candidate() {
  if [ "$resolver_mode" = realpath ]; then
    "$resolver" -e -- "$1"
  else
    "$resolver" -f -- "$1"
  fi
}
while IFS= read -r candidate; do
  [ -n "$candidate" ] || continue
  resolved=$(resolve_candidate "$candidate") || {
    echo "glob match disappeared during canonicalization" >&2
    exit 73
  }
  case "$resolved" in
    "$root"|"$root"/*) ;;
    *)
      echo "glob match escaped the authorized root" >&2
      exit 73
      ;;
  esac
  case "$resolved" in
    */terminal-bench-tasks|*/terminal-bench-tasks/*|/tests|/tests/*|/solutions|/solutions/*|/solution|/solution/*|*/trials/runs|*/trials/runs/*|*/trials/summaries|*/trials/summaries/*|/host/trials|/host/trials/*)
      echo "canonical path policy blocked a glob result" >&2
      exit 73
      ;;
  esac
  case "$resolved" in
    */terminal-bench/*/tests/*|*/terminal-bench/*/solutions/*|*/terminal-bench/*/solution/*|*/terminal-bench/*/task.toml|*/terminal_bench/*/tests/*|*/terminal_bench/*/solutions/*|*/terminal_bench/*/solution/*|*/terminal_bench/*/task.toml|*/terminalbench/*/tests/*|*/terminalbench/*/solutions/*|*/terminalbench/*/solution/*|*/terminalbench/*/task.toml)
      echo "canonical path policy blocked a glob result" >&2
      exit 73
      ;;
  esac
  printf "%s\n" "$resolved" >> "$canonical"
done < "$raw"

sort -u "$canonical" > "$sorted"
canonical_count=$(wc -l < "$sorted")
if [ "$canonical_count" -gt 500 ]; then
  echo "Python-free glob found more than 500 canonical matches" >&2
  exit 67
fi
cat "$sorted"
"""


def _ascii_printable(value: str) -> bool:
    return all(0x20 <= ord(character) <= 0x7E for character in value)


def _same_ascii_range_category(start: str, end: str) -> bool:
    return (
        (start.isdigit() and end.isdigit())
        or ("a" <= start <= "z" and "a" <= end <= "z")
        or ("A" <= start <= "Z" and "A" <= end <= "Z")
    )


def _supported_bracket_body(body: str) -> bool:
    if body.startswith("!"):
        body = body[1:]
    if not body or any(character in body for character in "[]"):
        return False
    if "-" in body:
        if body.count("-") != 1:
            return False
        start, end = body.split("-", 1)
        return (
            len(start) == 1
            and len(end) == 1
            and _same_ascii_range_category(start, end)
            and ord(start) <= ord(end)
        )
    return all(
        character.isascii()
        and (character.isalnum() or character == "_")
        for character in body
    )


def _unsupported_glob_reason(pattern: str) -> str:
    if not pattern:
        return "empty glob patterns are unsupported"
    if pattern in {".", "./"}:
        return "the current-directory-only pattern is not a valid pathlib glob"
    if pattern.startswith("/"):
        return "absolute glob patterns are unsupported; pass the root via path"
    if not _ascii_printable(pattern):
        return (
            "non-ASCII and control characters are unsupported because fallback "
            "matching is fixed to the C locale"
        )
    if "\\" in pattern:
        return "backslashes are unsupported in Python-free glob patterns"
    if any(marker in pattern for marker in ("@(", "+(", "?(", "*(", "!(")):
        return "extended Bash glob operators are outside the pathlib glob contract"

    components = pattern.removeprefix("./").split("/")
    if ".." in components:
        return "parent-directory traversal is unsupported in fallback globs"
    if any(component == "" for component in components[:-1]):
        return (
            "repeated path separators are unsupported because they change "
            "cross-engine result spelling"
        )
    if sum(component == "**" for component in components) > 1:
        return "multiple recursive ** components are unsupported"

    for component in components:
        cursor = 0
        while cursor < len(component):
            if component[cursor] != "[":
                cursor += 1
                continue
            close = component.find("]", cursor + 1)
            if close == -1:
                return "unterminated character class in fallback glob"
            if not _supported_bracket_body(component[cursor + 1 : close]):
                return (
                    "only literal ASCII alphanumeric/underscore classes and one "
                    "ascending same-category ASCII range are supported"
                )
            cursor = close + 1
    return ""


def _unsupported_glob_root_reason(path: str) -> str:
    if not path:
        return "empty glob roots are unsupported"
    if not _ascii_printable(path):
        return "non-ASCII and control characters are unsupported in fallback roots"
    return ""


class HarborGlobTool(_canonical.HarborGlobTool):
    """Use a conservative Bash fallback only for a verified semantic subset."""

    def _glob_without_python(
        self,
        *,
        pattern: str,
        path: str,
    ) -> ToolResult:
        unsupported = _unsupported_glob_reason(pattern)
        if unsupported:
            return ToolResult(
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
            return ToolResult(
                success=False,
                output="",
                error=f"Unsupported Python-free glob root: {root_unsupported}",
                metadata={
                    "engine": "bash-compgen",
                    "glob_root_unsupported": True,
                },
            )

        recursive = any(
            component == "**"
            for component in pattern.removeprefix("./").split("/")
        )
        result = self._exec(
            f"bash -c {shlex.quote(_GLOB_FALLBACK_SCRIPT)}",
            env={
                "HL_ROOT": path,
                "HL_PATTERN": pattern,
                "HL_RECURSIVE": "1" if recursive else "0",
            },
        )
        output = result.stdout or ""
        error = result.stderr or ""
        unavailable = _canonical._base._terminal_environment_unavailable_text(
            output,
            error,
        )
        if unavailable:
            return _canonical._base._terminal_environment_unavailable_result(
                output=output,
                stderr=error,
                message=unavailable,
                exit_code=result.return_code,
            )
        if result.return_code == 65:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Python-free glob cannot represent a tree containing "
                    "non-ASCII or control path components."
                ),
                metadata={
                    "exit_code": 65,
                    "engine": "bash-compgen",
                    "glob_filename_unsupported": True,
                },
            )
        if result.return_code == 66:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Python-free recursive glob cannot preserve pathlib traversal "
                    "semantics when the tree contains symlink directories."
                ),
                metadata={
                    "exit_code": 66,
                    "engine": "bash-compgen",
                    "recursive_symlink_glob_unsupported": True,
                },
            )
        if result.return_code == 67:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Python-free glob found more than 500 matches; refusing to "
                    "claim the same bounded subset as the primary engine."
                ),
                metadata={
                    "exit_code": 67,
                    "engine": "bash-compgen",
                    "glob_result_limit_unsupported": True,
                },
            )
        if result.return_code == 73:
            return _canonical._canonical_policy_block(
                "glob result",
                path,
                "a symlink or resolved match escaped or violated the canonical policy",
            )
        if result.return_code != 0:
            return ToolResult(
                success=False,
                output=output,
                error=error or f"exit code: {result.return_code}",
                metadata={
                    "exit_code": result.return_code,
                    "engine": "bash-compgen",
                    "glob_failed": True,
                },
            )

        matches = [line for line in output.splitlines() if line]
        return ToolResult(
            success=True,
            output="\n".join(matches) if matches else "(no matches)",
            error="",
            metadata={
                "exit_code": 0,
                "engine": "bash-compgen",
                "match_count": len(matches),
                "glob_semantics": "pathlib-glob-sorted-canonical-subset",
                "hidden_components": "pathlib-default",
                "result_order": "sorted-canonical",
                "fallback_output_bounded": True,
            },
        )

    def execute(
        self,
        pattern: str,
        path: str = ".",
        **_: Any,
    ) -> ToolResult:
        unsafe_reason = _canonical.unsafe_relative_pattern_reason(pattern)
        if unsafe_reason:
            return _canonical._canonical_policy_block("glob", path, unsafe_reason)
        observed = _canonical.guard_observed_text(
            f"{path} {pattern}",
            operation="read",
        )
        if observed.blocked_by:
            return _canonical.guarded_path_failure("glob", observed)

        root, failure = self._guard_environment_path(
            path,
            operation="read",
            must_exist=True,
        )
        if failure is not None:
            return failure

        env = {"HL_ROOT": root, "HL_PATTERN": pattern}
        result = self._exec(
            "python3 -c " + shlex.quote(_canonical._HARBOR_GLOB_PYTHON),
            env=env,
        )
        output = result.stdout or ""
        error = result.stderr or ""

        if result.return_code != 0 and _canonical._base._looks_like_missing_python(
            error,
            output,
        ):
            fallback = self._glob_without_python(pattern=pattern, path=root)
            if not fallback.success:
                return fallback
            fallback_matches = (
                []
                if fallback.output in {"", "(no matches)"}
                else [line for line in fallback.output.splitlines() if line]
            )
            safe, match_failure = self._guard_environment_matches(
                fallback_matches,
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
                    **fallback.metadata,
                    "match_count": len(safe),
                    "canonical_paths": True,
                    "fallback_revalidated": True,
                },
            )

        unavailable = _canonical._base._terminal_environment_unavailable_text(
            output,
            error,
        )
        if unavailable:
            return _canonical._base._terminal_environment_unavailable_result(
                output=output,
                stderr=error,
                message=unavailable,
                exit_code=result.return_code,
            )
        if result.return_code == 73:
            return _canonical._canonical_policy_block(
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
                "engine": "python-pathlib",
                "result_order": "sorted-canonical",
            },
        )


__all__ = [
    "HarborGlobTool",
    "_GLOB_FALLBACK_SCRIPT",
    "_unsupported_glob_reason",
    "_unsupported_glob_root_reason",
]
