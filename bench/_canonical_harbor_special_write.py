"""Compatibility exports for the consolidated snapshot-aware Harbor writer.

Pure overwrite and append now use the same tested publication core. There is
no separate string-replacement publication implementation to drift or regress.
"""
from bench._canonical_harbor_identity_guard import (
    HarborFileReadTool,
    HarborFileEditTool,
    HarborFileWriteTool,
    ToolResult,
)
from bench import _canonical_harbor_identity_guard as _base

_SECURE_ATOMIC_WRITE = _base._v3._SECURE_ATOMIC_WRITE

__all__ = ["HarborFileReadTool", "HarborFileEditTool", "HarborFileWriteTool",
           "ToolResult", "_SECURE_ATOMIC_WRITE"]
