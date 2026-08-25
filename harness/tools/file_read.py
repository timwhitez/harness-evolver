"""Canonical no-follow reader with an explicit bounded UTF-8 text contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.tools._file_read_issue18_base import FileReadTool as _BaseFileReadTool
from harness.tools.base import ToolResult
from harness.tools.canonical_path_guard import guarded_path_failure, resolve_guarded_path
from harness.tools.safe_path_io import SafePathError, read_bounded_bytes_nofollow


_BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "PDF"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
)
_DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024


def _unsupported_binary_kind(payload: bytes) -> str | None:
    for signature, kind in _BINARY_SIGNATURES:
        if payload.startswith(signature):
            return kind
    if b"\x00" in payload:
        return "binary"

    disallowed_controls = sum(
        byte < 32 and byte not in {9, 10, 12, 13}
        for byte in payload
    )
    if payload and disallowed_controls / len(payload) > 0.30:
        return "binary"
    return None


@dataclass
class FileReadTool(_BaseFileReadTool):
    """Read one authorized regular file without lossy or unbounded decoding."""

    description: str = (
        "Read a bounded UTF-8 text file from the filesystem and return content "
        "with line numbers. Use this instead of 'cat' for text files. Supports "
        "offset and limit. Binary files, images, PDFs, and oversized files are "
        "rejected explicitly."
    )
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES

    def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        effective_limit = self.max_lines if limit is None else limit
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
            return ToolResult(success=False, output="", error="offset must be an integer >= 1")
        if (
            isinstance(effective_limit, bool)
            or not isinstance(effective_limit, int)
            or effective_limit < 1
        ):
            return ToolResult(success=False, output="", error="limit must be an integer >= 1")
        if (
            isinstance(self.max_file_bytes, bool)
            or not isinstance(self.max_file_bytes, int)
            or self.max_file_bytes < 1
        ):
            return ToolResult(
                success=False,
                output="",
                error="max_file_bytes must be an integer >= 1",
                metadata={"parameter_validation_failed": True},
            )

        decision = resolve_guarded_path(
            file_path,
            operation="read",
            must_exist=True,
        )
        if not decision.allowed:
            return guarded_path_failure("read", decision)

        try:
            payload, metadata, truncated = read_bounded_bytes_nofollow(
                decision.resolved,
                max_bytes=self.max_file_bytes,
            )
        except (OSError, SafePathError, ValueError) as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Cannot read authorized path safely: {exc}",
                metadata={
                    "blocked_by": "canonical_path_guard",
                    "canonical_path_checked": True,
                    "nofollow_io": True,
                },
            )

        # Classify the bytes actually retained before reporting an oversize
        # failure. A large PNG/PDF therefore remains an explicit unsupported
        # binary rather than being disguised as a generic size error, while the
        # read itself is still capped at max_file_bytes plus one probe byte.
        binary_kind = _unsupported_binary_kind(payload)
        if binary_kind is not None:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Cannot read {file_path} as UTF-8 text: "
                    f"unsupported {binary_kind} file"
                ),
                metadata={
                    "binary_file_unsupported": True,
                    "detected_file_type": binary_kind,
                    "file_size": metadata.st_size,
                    "bytes_inspected": len(payload),
                    "read_memory_bounded": True,
                    "max_file_bytes": self.max_file_bytes,
                    "canonical_path_checked": True,
                    "nofollow_io": True,
                },
            )

        if truncated:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Cannot read {file_path} as text: file exceeds the bounded "
                    f"reader limit of {self.max_file_bytes} bytes"
                ),
                metadata={
                    "file_too_large": True,
                    "file_size": max(metadata.st_size, self.max_file_bytes + 1),
                    "max_file_bytes": self.max_file_bytes,
                    "read_memory_bounded": True,
                    "canonical_path_checked": True,
                    "nofollow_io": True,
                },
            )

        try:
            content = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Cannot read {file_path} as UTF-8 text: {exc}",
                metadata={
                    "text_decode_error": True,
                    "encoding": "utf-8",
                    "file_size": metadata.st_size,
                    "read_memory_bounded": True,
                    "max_file_bytes": self.max_file_bytes,
                    "canonical_path_checked": True,
                    "nofollow_io": True,
                },
            )

        lines = content.splitlines()
        total_lines = len(lines)
        start = min(offset - 1, total_lines)
        end = min(start + effective_limit, total_lines)
        selected = lines[start:end]
        output = "\n".join(
            f"{number}\t{line.rstrip()}"
            for number, line in enumerate(selected, start=start + 1)
        )
        if end < total_lines:
            output += (
                ("\n" if output else "")
                + f"... ({total_lines - end} more lines, use offset={end + 1} to continue)"
            )

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "total_lines": total_lines,
                "start_line": offset,
                "end_line": end,
                "file_size": metadata.st_size,
                "read_memory_bounded": True,
                "max_file_bytes": self.max_file_bytes,
                "canonical_path_checked": True,
                "nofollow_io": True,
                "encoding": "utf-8",
            },
        )
