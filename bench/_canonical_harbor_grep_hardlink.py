"""Hard-link-safe Harbor grep layered on canonical no-follow traversal."""

from __future__ import annotations

from typing import Any

from bench import _canonical_harbor_grep as _base


_SCAN_REGULAR_BLOCK = """    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("grep target is not a regular file")
    with os.fdopen(
"""
_SCAN_UNIQUE_REGULAR_BLOCK = """    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("grep target is not a regular file")
    if metadata.st_nlink != 1:
        print("multiply-linked regular file", file=sys.stderr)
        raise SystemExit(74)
    with os.fdopen(
"""


def _harden_grep_script(script: str) -> str:
    if _SCAN_UNIQUE_REGULAR_BLOCK in script:
        return script
    if _SCAN_REGULAR_BLOCK not in script:
        raise RuntimeError("canonical Harbor grep descriptor contract changed unexpectedly")
    if "import sys\n" not in script:
        script = script.replace("import stat\n", "import stat\nimport sys\n", 1)
    return script.replace(
        _SCAN_REGULAR_BLOCK,
        _SCAN_UNIQUE_REGULAR_BLOCK,
        1,
    )


_base._HARBOR_GREP_PYTHON = _harden_grep_script(
    _base._HARBOR_GREP_PYTHON
)


class HarborGrepTool(_base.HarborGrepTool):
    """Reject multiply-linked regular files before exposing grep content."""

    def execute(
        self,
        pattern: str,
        path: str = ".",
        **kwargs: Any,
    ) -> _base.ToolResult:
        result = super().execute(pattern=pattern, path=path, **kwargs)
        if result.metadata.get("exit_code") != 74:
            return result
        result.success = False
        result.output = ""
        result.error = (
            "Canonical path guard blocked grep: multiply-linked regular files "
            "cannot be attributed to one authorized path."
        )
        result.metadata = _base.policy_guard_metadata(
            "canonical_path_guard",
            exit_code=74,
            canonical_path_checked=True,
            nofollow_io=True,
            hardlink_alias_blocked=True,
        )
        return result


__all__ = ["HarborGrepTool"]
