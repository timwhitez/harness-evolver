"""Harbor writes that publish both overwrite and append operations atomically."""

from __future__ import annotations

import base64
from typing import Any

from bench import _canonical_harbor_paths_v2 as _v2


_SECURE_ATOMIC_WRITE = _v2._SAFE_PREAMBLE + r'''
try:
    payload = base64.b64decode(os.environ["HL_FILE_CONTENT"], validate=True)
    parent, name = parent_fd(os.environ["HL_FILE_PATH"], create=True)
    try:
        write_atomic(parent, name, payload)
    finally:
        os.close(parent)
    print("write complete")
except Exception:
    raise SystemExit("secure nofollow write failed")
'''


class HarborFileWriteTool(_v2.HarborFileWriteTool):
    """Build the complete text payload first, then replace the path entry atomically."""

    def execute(
        self,
        file_path: str,
        content: str,
        append: bool = False,
        **kwargs: Any,
    ) -> _v2.ToolResult:
        resolved, failure = self._guard_environment_path(
            file_path,
            operation="write",
            must_exist=False,
        )
        if failure is not None:
            return failure

        effective_content = content
        if append:
            current, read_failure = self._secure_raw_read(resolved)
            if read_failure is not None:
                return read_failure
            effective_content = (current or "") + content

        staged_reason = _v2.staged_dependency_script_reason(resolved, effective_content)
        if staged_reason:
            return _v2.ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked write: {staged_reason}",
                metadata=_v2.policy_guard_metadata("staged_dependency_script_guard"),
            )
        size_reason = _v2.deliverable_size_cap_write_reason(resolved, effective_content)
        if size_reason:
            return _v2.ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked write: {size_reason}",
                metadata=_v2.policy_guard_metadata("deliverable_size_cap_write_guard"),
            )

        result = self._run_secure_python(
            _SECURE_ATOMIC_WRITE,
            env={
                "HL_FILE_PATH": resolved,
                "HL_FILE_CONTENT": base64.b64encode(
                    effective_content.encode("utf-8")
                ).decode("ascii"),
            },
        )
        if result.success:
            result.metadata = {
                **result.metadata,
                "atomic_replace": True,
                "atomic_append": append,
            }
        return result
