"""Hard-link-safe Harbor file operations layered on canonical no-follow I/O."""

from __future__ import annotations

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
