"""Strict read-parameter validation for canonical Harbor file access."""

from __future__ import annotations

from typing import Any

from bench import _canonical_harbor_paths_v2 as _v2


_SECURE_READ_STRICT = _v2._SAFE_PREAMBLE + r'''
try:
    offset = int(os.environ.get("HL_OFFSET", "1"))
    limit = int(os.environ.get("HL_LIMIT", "2000"))
    if offset < 1:
        raise ValueError("offset must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    parent, name = parent_fd(os.environ["HL_FILE_PATH"], create=False)
    try:
        descriptor, _ = open_regular(parent, name)
        try:
            with os.fdopen(
                descriptor,
                "r",
                encoding="utf-8",
                errors="replace",
                closefd=False,
            ) as stream:
                lines = stream.read().splitlines()
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)

    start = offset - 1
    for number, line in enumerate(lines[start:start + limit], start=offset):
        print(f"{number}\t{line}")
    if start + limit < len(lines):
        print(f"... ({len(lines) - start - limit} more lines)")
except ValueError as exc:
    raise SystemExit(str(exc))
except Exception:
    raise SystemExit("secure nofollow read failed")
'''


def _positive_integer(value: object, *, name: str) -> tuple[int | None, _v2.ToolResult | None]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None, _v2.ToolResult(
            success=False,
            output="",
            error=f"{name} must be an integer >= 1",
            metadata={"parameter_validation_failed": True, "parameter": name},
        )
    if value < 1:
        return None, _v2.ToolResult(
            success=False,
            output="",
            error=f"{name} must be >= 1",
            metadata={"parameter_validation_failed": True, "parameter": name},
        )
    return value, None


class HarborFileReadTool(_v2.HarborFileReadTool):
    """Reject invalid windows instead of silently coercing them to one."""

    def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int | None = 2000,
        **kwargs: Any,
    ) -> _v2.ToolResult:
        validated_offset, failure = _positive_integer(offset, name="offset")
        if failure is not None:
            return failure
        effective_limit = 2000 if limit is None else limit
        validated_limit, failure = _positive_integer(effective_limit, name="limit")
        if failure is not None:
            return failure
        assert validated_offset is not None
        assert validated_limit is not None

        resolved, path_failure = self._guard_environment_path(
            file_path,
            operation="read",
            must_exist=True,
        )
        if path_failure is not None:
            return path_failure
        return self._run_secure_python(
            _SECURE_READ_STRICT,
            env={
                "HL_FILE_PATH": resolved,
                "HL_OFFSET": str(validated_offset),
                "HL_LIMIT": str(validated_limit),
            },
        )
