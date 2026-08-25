"""Hard-link-safe Harbor file operations layered on canonical no-follow I/O."""

from __future__ import annotations

import base64

from bench import _canonical_harbor_paths_v2 as _v2


_OPEN_REGULAR_BLOCK = """    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RuntimeError("not a regular file")
    return descriptor, metadata
"""
_OPEN_UNIQUE_REGULAR_BLOCK = """    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RuntimeError("not a regular file")
    if metadata.st_nlink != 1:
        os.close(descriptor)
        raise RuntimeError("multiply-linked regular file")
    return descriptor, metadata
"""

_ATOMIC_MODE_BLOCK = """        0o600,
        dir_fd=parent,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
        if mode is not None:
            os.fchmod(descriptor, stat.S_IMODE(mode))
"""
_ATOMIC_SAFE_MODE_BLOCK = """        0o666,
        dir_fd=parent,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            if mode is not None:
                # Rewriting content must never recreate setuid, setgid, or
                # sticky bits on the newly published inode.
                os.fchmod(descriptor, stat.S_IMODE(mode) & 0o777)
            os.fsync(descriptor)
"""


def _harden_content_read_script(script: str) -> str:
    """Require the embedded descriptor target to have one filesystem link."""

    if _OPEN_UNIQUE_REGULAR_BLOCK in script:
        return script
    if _OPEN_REGULAR_BLOCK not in script:
        raise RuntimeError(
            "canonical Harbor open_regular contract changed unexpectedly"
        )
    return script.replace(
        _OPEN_REGULAR_BLOCK,
        _OPEN_UNIQUE_REGULAR_BLOCK,
        1,
    )


def _harden_atomic_mode_script(script: str) -> str:
    """Apply normal umask semantics and clear special bits before fsync."""

    if _ATOMIC_SAFE_MODE_BLOCK in script:
        return script
    if _ATOMIC_MODE_BLOCK not in script:
        raise RuntimeError(
            "canonical Harbor write_atomic mode contract changed unexpectedly"
        )
    return script.replace(_ATOMIC_MODE_BLOCK, _ATOMIC_SAFE_MODE_BLOCK, 1)


def _harden_v2_script(script: str) -> str:
    return _harden_content_read_script(_harden_atomic_mode_script(script))


# Methods resolve these module globals at call time. Harden every v2 script
# before importing higher-level modules that embed the shared preamble.
_v2._SAFE_PREAMBLE = _harden_v2_script(_v2._SAFE_PREAMBLE)
_v2._SECURE_READ = _harden_v2_script(_v2._SECURE_READ)
_v2._SECURE_RAW_READ = _harden_v2_script(_v2._SECURE_RAW_READ)
_v2._SECURE_WRITE = _harden_v2_script(_v2._SECURE_WRITE)
_v2._SECURE_EDIT = _harden_v2_script(_v2._SECURE_EDIT)

# Import/patch v3 and v4 only after v2's preamble is hardened. v3's overwrite
# path does not read existing content, while append first calls v2's hardened
# raw reader. v4 restores strict offset/limit validation for public reads.
from bench import _canonical_harbor_write_v3 as _v3  # noqa: E402
from bench import _canonical_harbor_paths_v4 as _v4  # noqa: E402

_v3._SECURE_ATOMIC_WRITE = _harden_content_read_script(
    _v3._SECURE_ATOMIC_WRITE
)
_v4._SECURE_READ_STRICT = _harden_v2_script(
    _v4._SECURE_READ_STRICT
)


def _secure_raw_read_strict(
    self: _v2._SecureHarborMixin,
    path: str,
) -> tuple[str | None, _v2.ToolResult | None]:
    """Read append/edit source text without lossy replacement decoding.

    Harbor append is implemented as secure read plus complete atomic
    replacement so policy checks can inspect the final text. Replacement
    decoding would silently alter pre-existing invalid UTF-8 bytes before the
    atomic publish. A text operation must instead fail before the write phase.
    """

    result = self._run_secure_python(
        _v2._SECURE_RAW_READ,
        env={"HL_FILE_PATH": path},
    )
    if result.success:
        try:
            payload = base64.b64decode(result.output.strip(), validate=True)
        except Exception:
            return None, _v2.ToolResult(
                success=False,
                output="",
                error="Secure read returned invalid encoded content",
                metadata={
                    "canonical_path_checked": True,
                    "nofollow_io": True,
                    "secure_read_protocol_error": True,
                },
            )
        try:
            return payload.decode("utf-8", errors="strict"), None
        except UnicodeDecodeError as exc:
            return None, _v2.ToolResult(
                success=False,
                output="",
                error=f"Cannot append or edit non-UTF-8 text safely: {exc}",
                metadata={
                    "text_decode_error": True,
                    "encoding": "utf-8",
                    "canonical_path_checked": True,
                    "nofollow_io": True,
                    "atomic_replace": False,
                },
            )
    if result.metadata.get("exit_code") == 44:
        return "", None
    return None, result


_v2._SecureHarborMixin._secure_raw_read = _secure_raw_read_strict

HarborFileReadTool = _v4.HarborFileReadTool
HarborFileEditTool = _v2.HarborFileEditTool
HarborFileWriteTool = _v3.HarborFileWriteTool
ToolResult = _v2.ToolResult

__all__ = [
    "HarborFileReadTool",
    "HarborFileEditTool",
    "HarborFileWriteTool",
    "ToolResult",
]
