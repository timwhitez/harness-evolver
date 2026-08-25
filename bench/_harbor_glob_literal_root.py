"""Literal-root correction for the Python-free Harbor glob fallback."""

from __future__ import annotations

import shlex
from typing import Any

from bench import _harbor_glob_fallback as _base


_ROOT_PATTERN_BLOCK = '''case "$root" in
  /) combined="/$pattern" ;;
  */) combined="${root}${pattern}" ;;
  *) combined="${root}/$pattern" ;;
esac
'''
_LITERAL_ROOT_PATTERN_BLOCK = r'''escape_glob_literal() {
  local value=$1
  # Shell quoting protects the argument boundary, but metacharacters inside
  # the string passed to `compgen -G` remain active pattern syntax. Escape only
  # the authorized root; the requested pattern retains its reviewed semantics.
  value=${value//\/\\}
  value=${value//\*/\\*}
  value=${value//\?/\\?}
  value=${value//\[/\\[}
  value=${value//\]/\\]}
  printf "%s" "$value"
}
literal_root=$(escape_glob_literal "$root")
case "$literal_root" in
  /) combined="/$pattern" ;;
  */) combined="${literal_root}${pattern}" ;;
  *) combined="${literal_root}/$pattern" ;;
esac
'''
if _base._GLOB_FALLBACK_SCRIPT.count(_ROOT_PATTERN_BLOCK) != 1:
    raise RuntimeError("Harbor glob root-construction contract changed unexpectedly")
_GLOB_FALLBACK_SCRIPT = _base._GLOB_FALLBACK_SCRIPT.replace(
    _ROOT_PATTERN_BLOCK,
    _LITERAL_ROOT_PATTERN_BLOCK,
    1,
)
_OLD_FALLBACK_COMMAND = f"bash -c {shlex.quote(_base._GLOB_FALLBACK_SCRIPT)}"
_LITERAL_ROOT_FALLBACK_COMMAND = f"bash -c {shlex.quote(_GLOB_FALLBACK_SCRIPT)}"


class HarborGlobTool(_base.HarborGlobTool):
    """Run the reviewed fallback with a literal authorized root."""

    def _exec(
        self,
        command: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if command == _OLD_FALLBACK_COMMAND:
            command = _LITERAL_ROOT_FALLBACK_COMMAND
        return super()._exec(command, *args, **kwargs)


__all__ = [
    "HarborGlobTool",
    "_GLOB_FALLBACK_SCRIPT",
]
