"""Final fail-closed guards for Bash/Python glob semantic divergences.

The stacked implementation is retained in :mod:`bench._harbor_glob_issue14_base`.
This facade narrows the Python-free supported subset where Bash ``compgen`` and
``glob.glob`` have different pattern or locale semantics.
"""

from __future__ import annotations

from typing import Any

from bench import _harbor_glob_issue14_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_BASE_UNSUPPORTED_PATTERN = _base._unsupported_glob_reason
_BASE_UNSUPPORTED_ROOT = _base._unsupported_glob_root_reason
_BASE_GLOB_WITHOUT_PYTHON = _base._glob_without_python


def _ascii_only(value: str) -> bool:
    return all(0x20 <= ord(character) <= 0x7E for character in value)


def _unsupported_glob_reason(pattern: str) -> str:
    baseline_reason = _BASE_UNSUPPORTED_PATTERN(pattern)
    if baseline_reason:
        return baseline_reason
    if not _ascii_only(pattern):
        return (
            "non-ASCII and control characters are unsupported because Bash glob "
            "matching is locale-dependent while Python glob uses Unicode semantics"
        )

    components = pattern.removeprefix("./").split("/")
    if sum(component == "**" for component in components) > 1:
        return (
            "multiple recursive ** components are unsupported because Python glob "
            "can emit duplicate paths that Bash globstar deduplicates"
        )

    for component in components:
        if any(marker in component for marker in ("[[:", "[[.", "[[=")):
            return (
                "POSIX character classes, collating symbols, and equivalence "
                "classes are unsupported because Python and Bash interpret them differently"
            )

        cursor = 0
        while cursor < len(component):
            if component[cursor] != "[":
                cursor += 1
                continue
            close = component.find("]", cursor + 1)
            if close == -1:
                break  # The stacked validator reports the unterminated class.
            first = cursor + 1
            if first < close and component[first] == "!":
                first += 1
            if first < close and component[first] == "^":
                return (
                    "character classes beginning with ^ are unsupported because "
                    "Python treats ^ literally while Bash treats it as negation"
                )
            cursor = close + 1
    return ""


def _unsupported_glob_root_reason(path: str) -> str:
    baseline_reason = _BASE_UNSUPPORTED_ROOT(path)
    if baseline_reason:
        return baseline_reason
    if not _ascii_only(path):
        return (
            "non-ASCII and control characters are unsupported in Python-free glob roots"
        )
    return ""


# Fix the fallback locale and reject non-ASCII/control path components before
# compgen receives newline-delimited input. Under this documented ASCII subset,
# Bash wildcard and range matching no longer depends on the container locale.
_script = _base._GLOB_FALLBACK_SCRIPT.replace(
    "set -u\nroot=$1",
    "set -u\nexport LC_ALL=C\nroot=$1",
    1,
)
_script = _script.replace(
    "    *\"\n\"*|*\"\\r\"*) printf \"%s\\n\" newline > \"$HL_BAD\"; exit 0 ;;\n",
    "    *\"\n\"*|*\"\\r\"*) printf \"%s\\n\" newline > \"$HL_BAD\"; exit 0 ;;\n"
    "    *[! -~]*) printf \"%s\\n\" non-ascii > \"$HL_BAD\"; exit 0 ;;\n",
    1,
)
_base._GLOB_FALLBACK_SCRIPT = _script
_base._unsupported_glob_reason = _unsupported_glob_reason
_base._unsupported_glob_root_reason = _unsupported_glob_root_reason


def _glob_without_python(
    self: Any,
    *,
    pattern: str,
    path: str = ".",
):
    result = _BASE_GLOB_WITHOUT_PYTHON(self, pattern=pattern, path=path)
    if result.metadata.get("glob_filename_unsupported"):
        result.error = (
            "Python-free glob cannot represent a tree containing newline, carriage-return, "
            "non-ASCII, or control path components under its text and locale contract."
        )
    return result


_base.HarborGlobTool._glob_without_python = _glob_without_python
HarborGlobTool = _base.HarborGlobTool
HLWorkerHarborAgent = _base.HLWorkerHarborAgent
