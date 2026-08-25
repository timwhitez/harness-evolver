"""Harbor whole-file writer that rejects existing non-regular targets."""

from __future__ import annotations

import base64
from typing import Any

from bench import _canonical_harbor_identity_guard as _base


_SPECIAL_TARGET_NEEDLE = '''        if stat.S_ISLNK(current.st_mode):
            raise RuntimeError("symlink target")
        if expected_missing:
'''
_SPECIAL_TARGET_REPLACEMENT = '''        if stat.S_ISLNK(current.st_mode):
            raise RuntimeError("symlink target")
        if not stat.S_ISREG(current.st_mode):
            raise RuntimeError("target is not a regular file")
        if expected_missing:
'''
if _SPECIAL_TARGET_NEEDLE not in _base._v3._SECURE_ATOMIC_WRITE:
    raise RuntimeError("canonical Harbor target-type contract changed unexpectedly")
_SECURE_ATOMIC_WRITE = _base._v3._SECURE_ATOMIC_WRITE.replace(
    _SPECIAL_TARGET_NEEDLE,
    _SPECIAL_TARGET_REPLACEMENT,
    1,
)


class HarborFileWriteTool(_base.HarborFileWriteTool):
    """Reject a special existing entry before unconditional publication."""

    def execute(
        self,
        file_path: str,
        content: str,
        append: bool = False,
        **kwargs: Any,
    ) -> _base._v2.ToolResult:
        # Append already uses the inherited secure snapshot and conditional
        # publication path, whose regular-file opener is stricter than this
        # unconditional whole-file target check.
        if append:
            return super().execute(
                file_path,
                content,
                append=True,
                **kwargs,
            )

        resolved, failure = self._guard_environment_path(
            file_path,
            operation="write",
            must_exist=False,
        )
        if failure is not None:
            return failure

        staged_reason = _base._v2.staged_dependency_script_reason(resolved, content)
        if staged_reason:
            return _base._v2.ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked write: {staged_reason}",
                metadata=_base._v2.policy_guard_metadata(
                    "staged_dependency_script_guard"
                ),
            )
        size_reason = _base._v2.deliverable_size_cap_write_reason(
            resolved,
            content,
        )
        if size_reason:
            return _base._v2.ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked write: {size_reason}",
                metadata=_base._v2.policy_guard_metadata(
                    "deliverable_size_cap_write_guard"
                ),
            )

        result = self._run_secure_python(
            _SECURE_ATOMIC_WRITE,
            env={
                "HL_FILE_PATH": resolved,
                "HL_FILE_CONTENT": base64.b64encode(
                    content.encode("utf-8")
                ).decode("ascii"),
            },
        )
        result = _base._publication_metadata(
            result,
            identity_verified=False,
        )
        if result.success:
            result.metadata["atomic_append"] = False
        return result


HarborFileEditTool = _base.HarborFileEditTool
HarborFileReadTool = _base.HarborFileReadTool
ToolResult = _base._v2.ToolResult

__all__ = [
    "HarborFileReadTool",
    "HarborFileEditTool",
    "HarborFileWriteTool",
    "ToolResult",
    "_SECURE_ATOMIC_WRITE",
]
