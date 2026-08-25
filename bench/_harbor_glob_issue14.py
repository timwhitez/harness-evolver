"""Conservative, reachable contract for Python-free Harbor glob matching.

The locale and semantic guards are retained from the stacked Issue #14 modules.
This facade additionally overrides the *actual* public ``execute()`` branch so a
missing Python interpreter calls the reviewed Bash implementation instead of the
older basename-only canonical fallback.
"""

from __future__ import annotations

import shlex
from typing import Any

from bench import _canonical_harbor_glob as _canonical
from bench import _harbor_glob_issue14_locale_base as _base
from bench._canonical_harbor_paths import _canonical_policy_block
from harness.tools.base import ToolResult
from harness.tools.canonical_path_guard import (
    guard_observed_text,
    guarded_path_failure,
    unsafe_relative_pattern_reason,
)

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_BASE_UNSUPPORTED_PATTERN = _base._unsupported_glob_reason
_BASE_UNSUPPORTED_ROOT = _base._unsupported_glob_root_reason
_runtime_module = _base._runtime_module


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
    baseline_reason = _BASE_UNSUPPORTED_PATTERN(pattern)
    if baseline_reason:
        return baseline_reason

    # Python glob can normalize repeated separators adjacent to wildcard
    # expansion while Bash preserves them in returned strings. Reject every
    # internal empty component rather than returning differently spelled paths.
    components = pattern.removeprefix("./").split("/")
    if any(component == "" for component in components[:-1]):
        return (
            "repeated path separators are unsupported because Python glob and "
            "Bash compgen return different path spellings"
        )

    for component in components:
        cursor = 0
        while cursor < len(component):
            if component[cursor] != "[":
                cursor += 1
                continue
            close = component.find("]", cursor + 1)
            if close == -1:
                break
            if not _supported_bracket_body(component[cursor + 1 : close]):
                return (
                    "only literal ASCII alphanumeric/underscore classes and one "
                    "ascending same-category ASCII range are supported; other "
                    "hyphen, punctuation, or parser-recovery forms differ between "
                    "Python glob and Bash"
                )
            cursor = close + 1
    return ""


def _unsupported_glob_root_reason(path: str) -> str:
    baseline_reason = _BASE_UNSUPPORTED_ROOT(path)
    if baseline_reason:
        return baseline_reason
    if not path:
        return (
            "empty roots are unsupported because the shell substitutes '.', "
            "changing the primary result spelling"
        )
    return ""


# Reset caller-controlled shell glob state before enabling exactly the features
# used by the verified fallback. ``globskipdots`` is optional on older Bash.
_runtime_module._GLOB_FALLBACK_SCRIPT = _runtime_module._GLOB_FALLBACK_SCRIPT.replace(
    "set -u\nexport LC_ALL=C\nroot=$1",
    "set -u\n"
    "export LC_ALL=C\n"
    "unset GLOBIGNORE\n"
    "shopt -u dotglob nocaseglob failglob extglob 2>/dev/null || true\n"
    "shopt -s globskipdots 2>/dev/null || true\n"
    "root=$1",
    1,
)
_runtime_module._unsupported_glob_reason = _unsupported_glob_reason
_runtime_module._unsupported_glob_root_reason = _unsupported_glob_root_reason


class HarborGlobTool(_base.HarborGlobTool):
    """Use the reviewed Bash fallback from the public execution path."""

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
                "engine": "python",
            },
        )


# Patch the registry-defining module as well as exporting the public subclass.
_runtime_module.HarborGlobTool = HarborGlobTool
HLWorkerHarborAgent = _base.HLWorkerHarborAgent
