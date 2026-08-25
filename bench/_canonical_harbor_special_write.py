"""Race-safe Harbor whole-file publication for regular or missing targets."""

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
_CURRENT_TARGET_NEEDLE = '''    current = read_current(name)
    if mode is None and current is not None:
        mode = current.st_mode
'''
_CURRENT_TARGET_REPLACEMENT = '''    def validate_displaced_unconditional(current_name, expected):
        current = os.stat(
            current_name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(current.st_mode):
            raise RuntimeError("target is not a regular file")
        identity = (int(current.st_dev), int(current.st_ino))
        if identity != tuple(expected):
            raise RuntimeError("overwrite target changed identity before publication")

    current = read_current(name)
    unconditional_identity = (
        (int(current.st_dev), int(current.st_ino))
        if current is not None and not conditional_existing
        else None
    )
    if mode is None and current is not None:
        mode = current.st_mode
'''
_PUBLICATION_NEEDLE = '''        elif expected_missing:
            renameat2(parent, temporary, name, 1)
            temporary_exists = False
            published = True
        else:
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            temporary_exists = False
            published = True
'''
_PUBLICATION_REPLACEMENT = '''        elif expected_missing or current is None:
            renameat2(parent, temporary, name, 1)
            temporary_exists = False
            published = True
        else:
            renameat2(parent, temporary, name, 2)
            exchanged = True
            try:
                validate_displaced_unconditional(
                    temporary,
                    unconditional_identity,
                )
            except Exception as validation_error:
                try:
                    assert_name_identity(name, new_identity)
                    renameat2(parent, temporary, name, 2)
                    exchanged = False
                except Exception as rollback_error:
                    raise RuntimeError(
                        "unconditional overwrite validation failed and rollback "
                        "could not complete; displaced target retained at " + temporary
                    ) from rollback_error
                raise validation_error
            os.unlink(temporary, dir_fd=parent)
            temporary_exists = False
            exchanged = False
            published = True
'''

_SECURE_ATOMIC_WRITE = _base._v3._SECURE_ATOMIC_WRITE
for needle, replacement, contract in (
    (
        _SPECIAL_TARGET_NEEDLE,
        _SPECIAL_TARGET_REPLACEMENT,
        "target-type",
    ),
    (
        _CURRENT_TARGET_NEEDLE,
        _CURRENT_TARGET_REPLACEMENT,
        "target-identity",
    ),
    (
        _PUBLICATION_NEEDLE,
        _PUBLICATION_REPLACEMENT,
        "conditional-publication",
    ),
):
    if needle not in _SECURE_ATOMIC_WRITE:
        raise RuntimeError(
            f"canonical Harbor {contract} contract changed unexpectedly"
        )
    _SECURE_ATOMIC_WRITE = _SECURE_ATOMIC_WRITE.replace(
        needle,
        replacement,
        1,
    )


class HarborFileWriteTool(_base.HarborFileWriteTool):
    """Publish a pure overwrite only against the observed regular entry."""

    def execute(
        self,
        file_path: str,
        content: str,
        append: bool = False,
        **kwargs: Any,
    ) -> _base._v2.ToolResult:
        # Append already uses the inherited snapshot plus exact-inode
        # conditional publication path.
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
            result.metadata["conditional_overwrite_publication"] = True
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
