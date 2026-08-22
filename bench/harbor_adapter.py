"""Harbor adapter with atomic write/append replacement semantics.

The audit-baseline adapter is retained in
:mod:`bench._harbor_adapter_issue16_base`; its policy checks are reused while
the final write command is replaced with a same-directory temporary-file
commit.
"""

from __future__ import annotations

from typing import Any

from bench import _harbor_adapter_issue16_base as _base

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


_ATOMIC_WRITE_SCRIPT = r'''set -eu
p=${HL_FILE_PATH:?}
case "$p" in */*) dir=${p%/*};; *) dir=.;; esac
[ -n "$dir" ] || dir=/
mkdir -p "$dir"
[ ! -d "$p" ] || { echo "Path is a directory: $p" >&2; exit 1; }
command -v base64 >/dev/null 2>&1 || { echo 'base64 command not found for structured write' >&2; exit 127; }
tmp=$(mktemp "$dir/.hl-write.XXXXXX")
cleanup() { [ -z "${tmp:-}" ] || rm -f "$tmp"; }
trap cleanup EXIT INT TERM
if [ "${HL_APPEND:-0}" = "1" ] && [ -e "$p" ]; then
  cat "$p" > "$tmp"
fi
printf '%s' "$HL_FILE_CONTENT" | base64 -d >> "$tmp"
if [ -e "$p" ]; then
  if chmod --reference="$p" "$tmp" 2>/dev/null; then :
  elif command -v stat >/dev/null 2>&1; then chmod "$(stat -c '%a' "$p")" "$tmp"; fi
else
  mask=$(umask)
  mode=$((0666 & (0777 ^ 0$mask)))
  chmod "$(printf '%03o' "$mode")" "$tmp"
fi
if command -v sync >/dev/null 2>&1; then sync -f "$tmp" 2>/dev/null || true; fi
mv -f "$tmp" "$p"
tmp=
printf '%s\n' "$p"
'''


class HarborFileWriteTool(_base.HarborFileWriteTool):
    description = (
        "Atomically write text content to a file inside the TerminalBench "
        "environment without requiring Python in the target container."
    )

    def _exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> Any:
        write_environment = env or {}
        if {
            "HL_FILE_PATH",
            "HL_FILE_CONTENT",
            "HL_APPEND",
        }.issubset(write_environment):
            command = f"sh -c {_base.shlex.quote(_ATOMIC_WRITE_SCRIPT)}"
        return super()._exec(command, timeout=timeout, env=env)

    def execute(
        self,
        file_path: str,
        content: str,
        append: bool = False,
        **kwargs: Any,
    ) -> _base.ToolResult:
        result = super().execute(
            file_path=file_path,
            content=content,
            append=append,
            **kwargs,
        )
        if result.success:
            result.metadata = {
                **result.metadata,
                "atomic_replace": True,
                "atomic_append": append,
            }
        return result


_base.HarborFileWriteTool = HarborFileWriteTool
HLWorkerHarborAgent = _base.HLWorkerHarborAgent
