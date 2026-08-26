"""Bounded UTF-8 streaming over the canonical no-follow reader."""

from __future__ import annotations

from dataclasses import dataclass
import io
import os
from typing import Any, BinaryIO, TextIO

from harness.tools import _file_read_issue17_base as _base
from harness.tools.base import ToolResult
from harness.tools.bounded_path_io import open_binary_nofollow
from harness.tools.canonical_path_guard import guarded_path_failure, resolve_guarded_path
from harness.tools.safe_path_io import SafePathError


_BINARY_SAMPLE_BYTES = 8192
_ALLOWED_CONTROLS = {"\t", "\f", "\r"}


class _UnsupportedBinaryContent(ValueError):
    def __init__(self, kind: str = "binary") -> None:
        super().__init__(f"unsupported {kind} file")
        self.kind = kind


class _FileSizeLimitExceeded(RuntimeError):
    """Raised when one authorized descriptor grows beyond its configured cap."""


class _CappedRawReader(io.RawIOBase):
    """Expose at most ``limit`` bytes from a borrowed binary stream.

    The source is owned by :func:`open_binary_nofollow`; closing this wrapper
    deliberately does not close that source. Reaching the limit performs one
    extra-byte probe so growth while the stream is being consumed cannot pass
    silently through ``TextIOWrapper``.
    """

    def __init__(self, source: BinaryIO, limit: int) -> None:
        super().__init__()
        self._source = source
        self._limit = limit
        self._remaining = limit

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed capped reader")
        if self._remaining == 0:
            if self._source.read(1):
                raise _FileSizeLimitExceeded(
                    f"authorized file exceeded {self._limit} bytes while reading"
                )
            return 0

        view = memoryview(buffer).cast("B")
        requested = min(len(view), self._remaining)
        payload = self._source.read(requested)
        if not payload:
            return 0
        view[: len(payload)] = payload
        self._remaining -= len(payload)
        return len(payload)


@dataclass(frozen=True)
class _BoundedLine:
    text: str
    truncated: bool


def _validate_text_controls(disallowed: int, observed: int) -> None:
    if observed and disallowed / observed > 0.30:
        raise _UnsupportedBinaryContent()


def _read_bounded_utf8_line(
    stream: TextIO,
    max_line_bytes: int,
) -> _BoundedLine | None:
    """Validate one complete physical line while retaining a bounded prefix.

    Every consumed character is classified even after the retained prefix is
    full. This preserves the binary-file contract for NUL/control bytes that
    occur beyond the initial signature sample or late in a very long line.
    """

    retained: list[str] = []
    retained_bytes = 0
    truncated = False
    saw_data = False
    disallowed_controls = 0
    observed_characters = 0

    while True:
        # TextIOWrapper performs strict incremental UTF-8 validation. The size
        # is a character bound, so one chunk is at most ~4x max_line_bytes in
        # encoded form and never grows with the physical line length.
        chunk = stream.readline(max_line_bytes + 1)
        if chunk == "":
            if not saw_data:
                return None
            _validate_text_controls(disallowed_controls, observed_characters)
            return _BoundedLine("".join(retained), truncated)

        saw_data = True
        ended = chunk.endswith("\n")
        content = chunk[:-1] if ended else chunk

        if "\x00" in content:
            raise _UnsupportedBinaryContent()
        observed_characters += len(content)
        disallowed_controls += sum(
            ord(character) < 32 and character not in _ALLOWED_CONTROLS
            for character in content
        )

        if not truncated:
            for character in content:
                encoded_size = len(character.encode("utf-8"))
                if retained_bytes + encoded_size <= max_line_bytes:
                    retained.append(character)
                    retained_bytes += encoded_size
                else:
                    truncated = True
                    break

        if ended:
            _validate_text_controls(disallowed_controls, observed_characters)
            return _BoundedLine("".join(retained), truncated)
        # No newline was observed. Continue in bounded chunks so the decoder
        # validates and classifies the complete physical line even after its
        # retained prefix is full.


@dataclass
class FileReadTool(_base.FileReadTool):
    """Apply offset/limit while streaming one authorized UTF-8 regular file."""

    description: str = (
        "Stream a UTF-8 text file and return a bounded, line-numbered window. "
        "Offset and limit are 1-based; long lines and total output are capped. "
        "Binary files, images, PDFs, and invalid UTF-8 are rejected explicitly "
        "without full-file buffering; an optional explicit total-size cap is "
        "enforced while streaming."
    )
    # Kept as an optional compatibility guard for the parent #24 API. The
    # streaming implementation imposes no total-file limit by default.
    max_file_bytes: int | None = None
    max_line_bytes: int = 65_536
    max_output_chars: int = 1_000_000

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
        for name, value in (
            ("max_line_bytes", self.max_line_bytes),
            ("max_output_chars", self.max_output_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"{name} must be an integer >= 1",
                    metadata={"parameter_validation_failed": True},
                )
        if self.max_file_bytes is not None and (
            isinstance(self.max_file_bytes, bool)
            or not isinstance(self.max_file_bytes, int)
            or self.max_file_bytes < 1
        ):
            return ToolResult(
                success=False,
                output="",
                error="max_file_bytes must be an integer >= 1 or None",
                metadata={"parameter_validation_failed": True},
            )

        decision = resolve_guarded_path(
            file_path,
            operation="read",
            must_exist=True,
        )
        if not decision.allowed:
            return guarded_path_failure("read", decision)

        opened_file_size: int | None = None
        try:
            with open_binary_nofollow(decision.resolved) as (binary_stream, metadata):
                opened_file_size = metadata.st_size

                # Always retain the inherited 8 KiB classification window. A
                # tiny optional size cap must not disguise a PNG/PDF as a generic
                # oversize text file.
                sample = binary_stream.read(_BINARY_SAMPLE_BYTES)
                binary_kind = _base._unsupported_binary_kind(sample)
                if binary_kind is not None:
                    return self._binary_failure(
                        file_path,
                        binary_kind,
                        file_size=os.fstat(binary_stream.fileno()).st_size,
                        bytes_inspected=len(sample),
                    )

                current_size = os.fstat(binary_stream.fileno()).st_size
                opened_file_size = current_size
                if self.max_file_bytes is not None and current_size > self.max_file_bytes:
                    return self._file_too_large_failure(
                        file_path,
                        file_size=current_size,
                    )

                binary_stream.seek(0)
                borrowed_buffer: io.BufferedReader | None = None
                text_source: BinaryIO = binary_stream
                if self.max_file_bytes is not None:
                    capped_raw = _CappedRawReader(binary_stream, self.max_file_bytes)
                    borrowed_buffer = io.BufferedReader(capped_raw)
                    text_source = borrowed_buffer

                text_stream = io.TextIOWrapper(
                    text_source,
                    encoding="utf-8",
                    errors="strict",
                    newline=None,
                )
                try:
                    try:
                        result = self._stream_window(
                            text_stream,
                            file_size=current_size,
                            offset=offset,
                            limit=effective_limit,
                        )
                    except _FileSizeLimitExceeded:
                        observed_size = os.fstat(binary_stream.fileno()).st_size
                        return self._file_too_large_failure(
                            file_path,
                            file_size=max(
                                observed_size,
                                (self.max_file_bytes or 0) + 1,
                            ),
                        )
                finally:
                    # The outer no-follow context owns ``binary_stream``. Detach
                    # TextIOWrapper and close only the optional borrowed wrapper.
                    try:
                        text_stream.detach()
                    except (ValueError, OSError):
                        pass
                    if borrowed_buffer is not None:
                        borrowed_buffer.close()

                final_size = os.fstat(binary_stream.fileno()).st_size
                opened_file_size = final_size
                if self.max_file_bytes is not None and final_size > self.max_file_bytes:
                    return self._file_too_large_failure(
                        file_path,
                        file_size=final_size,
                    )
                result.metadata["file_size"] = final_size
                return result
        except _UnsupportedBinaryContent as exc:
            return self._binary_failure(
                file_path,
                exc.kind,
                file_size=opened_file_size,
            )
        except UnicodeDecodeError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Cannot read {file_path} as UTF-8 text: {exc}",
                metadata={
                    "text_decode_error": True,
                    "encoding": "utf-8",
                    "file_size": opened_file_size,
                    "read_memory_bounded": True,
                    "canonical_path_checked": True,
                    "nofollow_io": True,
                },
            )
        except (OSError, SafePathError) as exc:
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

    def _file_too_large_failure(
        self,
        file_path: str,
        *,
        file_size: int,
    ) -> ToolResult:
        assert self.max_file_bytes is not None
        return ToolResult(
            success=False,
            output="",
            error=(
                f"Cannot read {file_path} as text: file exceeds the "
                f"configured limit of {self.max_file_bytes} bytes"
            ),
            metadata={
                "file_too_large": True,
                "file_size": file_size,
                "max_file_bytes": self.max_file_bytes,
                "read_memory_bounded": True,
                "canonical_path_checked": True,
                "nofollow_io": True,
                "streaming_read": True,
                "live_descriptor_size_checked": True,
            },
        )

    def _binary_failure(
        self,
        file_path: str,
        kind: str,
        *,
        file_size: int | None,
        bytes_inspected: int | None = None,
    ) -> ToolResult:
        metadata: dict[str, Any] = {
            "binary_file_unsupported": True,
            "detected_file_type": kind,
            "file_size": file_size,
            "read_memory_bounded": True,
            "canonical_path_checked": True,
            "nofollow_io": True,
            "streaming_read": True,
        }
        if bytes_inspected is not None:
            metadata["bytes_inspected"] = bytes_inspected
        if self.max_file_bytes is not None:
            metadata["max_file_bytes"] = self.max_file_bytes
        return ToolResult(
            success=False,
            output="",
            error=(
                f"Cannot read {file_path} as UTF-8 text: "
                f"unsupported {kind} file"
            ),
            metadata=metadata,
        )

    def _stream_window(
        self,
        stream: TextIO,
        *,
        file_size: int,
        offset: int,
        limit: int,
    ) -> ToolResult:
        records: list[tuple[int, _BoundedLine]] = []
        line_number = 0
        eof_reached = False
        has_more = False
        next_offset: int | None = None
        output_truncated = False
        retained_chars = 0

        while line_number < offset - 1:
            skipped = _read_bounded_utf8_line(stream, self.max_line_bytes)
            if skipped is None:
                eof_reached = True
                break
            line_number += 1

        if not eof_reached:
            while len(records) < limit:
                bounded_line = _read_bounded_utf8_line(stream, self.max_line_bytes)
                if bounded_line is None:
                    eof_reached = True
                    break
                line_number += 1
                suffix = (
                    f" ... [line truncated at {self.max_line_bytes} bytes]"
                    if bounded_line.truncated
                    else ""
                )
                formatted = f"{line_number}\t{bounded_line.text}{suffix}"
                projected = retained_chars + len(formatted) + (1 if records else 0)
                if projected > self.max_output_chars:
                    has_more = True
                    next_offset = line_number
                    output_truncated = True
                    break
                records.append((line_number, bounded_line))
                retained_chars = projected

            if not has_more and not eof_reached and len(records) == limit:
                following = _read_bounded_utf8_line(stream, self.max_line_bytes)
                if following is not None:
                    line_number += 1
                    has_more = True
                    next_offset = line_number
                else:
                    eof_reached = True

        output_lines: list[str] = []
        truncated_line_count = 0
        for number, bounded_line in records:
            suffix = ""
            if bounded_line.truncated:
                truncated_line_count += 1
                suffix = f" ... [line truncated at {self.max_line_bytes} bytes]"
            output_lines.append(f"{number}\t{bounded_line.text}{suffix}")

        if has_more:
            marker = f"... (more lines, use offset={next_offset} to continue)"
            if len(marker) >= self.max_output_chars:
                output = marker[: self.max_output_chars]
                output_truncated = True
            else:
                current = "\n".join(output_lines)
                available = self.max_output_chars - len(marker) - 1
                if len(current) > available:
                    current = current[:available].rstrip()
                    output_truncated = True
                output = f"{current}\n{marker}" if current else marker
        else:
            output = "\n".join(output_lines)

        if records:
            end_line = records[-1][0]
        elif eof_reached:
            end_line = line_number
        else:
            end_line = offset - 1

        result_metadata: dict[str, Any] = {
            "start_line": offset,
            "end_line": end_line,
            "lines_returned": len(records),
            "has_more": has_more,
            "next_offset": next_offset,
            "line_truncated_count": truncated_line_count,
            "output_truncated": output_truncated,
            "max_line_bytes": self.max_line_bytes,
            "max_output_chars": self.max_output_chars,
            "file_size": file_size,
            "streaming_read": True,
            "physical_line_memory_bounded": True,
            "read_memory_bounded": True,
            "canonical_path_checked": True,
            "nofollow_io": True,
            "encoding": "utf-8",
        }
        if self.max_file_bytes is not None:
            result_metadata["max_file_bytes"] = self.max_file_bytes
            result_metadata["live_descriptor_size_checked"] = True
        if eof_reached:
            result_metadata["total_lines"] = line_number
            result_metadata["total_lines_known"] = True
        else:
            result_metadata["total_lines_known"] = False

        return ToolResult(success=True, output=output, metadata=result_metadata)


_BoundedLine = _BoundedLine
