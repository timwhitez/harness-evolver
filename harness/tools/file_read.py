"""Public streaming reader without replacing the final path-I/O layer."""

from __future__ import annotations

import harness.tools.safe_path_io as _safe_path_io
from harness.tools.bounded_path_io import open_binary_nofollow

# The reviewed streaming implementation historically imported the reader from
# safe_path_io. Point that private lookup at the final bounded one-descriptor
# implementation introduced by the clean Issue #18 integration. No writer or
# conditional-publication behavior is modified.
_safe_path_io.open_binary_nofollow = open_binary_nofollow

import harness.tools._streaming_file_read_impl as _impl  # noqa: E402

FileReadTool = _impl.FileReadTool

__all__ = ["FileReadTool"]
