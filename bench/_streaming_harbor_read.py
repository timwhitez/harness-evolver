"""Canonical Harbor reader with bounded strict UTF-8 streaming."""

from __future__ import annotations

from typing import Any

from harness.tools.read_window import embedded_window_source

from bench import _harbor_adapter_issue17_base as _base
from bench import _canonical_harbor_paths_v4 as _v4


_STREAMING_READ_PYTHON = _v4._v2._SAFE_PREAMBLE + embedded_window_source() + r'''
import io

BINARY_SIGNATURES = (
    (b"%PDF-", "PDF"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
)
ALLOWED_CONTROLS = {"\t", "\f", "\r"}

def binary_kind(sample):
    for signature, kind in BINARY_SIGNATURES:
        if sample.startswith(signature):
            return kind
    if b"\x00" in sample:
        return "binary"
    controls = sum(byte < 32 and byte not in (9, 10, 12, 13) for byte in sample)
    if sample and controls / len(sample) > 0.30:
        return "binary"
    return None

def validate_controls(disallowed, observed):
    if observed and disallowed / observed > 0.30:
        raise ValueError("unsupported binary file")

def read_line(stream, max_line):
    retained = []
    retained_bytes = 0
    truncated = False
    saw_data = False
    disallowed_controls = 0
    observed_characters = 0
    while True:
        chunk = stream.readline(max_line + 1)
        if chunk == "":
            if not saw_data:
                return None
            validate_controls(disallowed_controls, observed_characters)
            return "".join(retained), truncated
        saw_data = True
        ended = chunk.endswith("\n")
        content = chunk[:-1] if ended else chunk
        if "\x00" in content:
            raise ValueError("unsupported binary file")
        observed_characters += len(content)
        disallowed_controls += sum(
            ord(character) < 32 and character not in ALLOWED_CONTROLS
            for character in content
        )
        if not truncated:
            for character in content:
                size = len(character.encode("utf-8"))
                if retained_bytes + size <= max_line:
                    retained.append(character)
                    retained_bytes += size
                else:
                    truncated = True
                    break
        if ended:
            validate_controls(disallowed_controls, observed_characters)
            return "".join(retained), truncated

parent = None
descriptor = None
binary_stream = None
text_stream = None
try:
    offset = int(os.environ.get("HL_OFFSET", "1"))
    limit = int(os.environ.get("HL_LIMIT", "2000"))
    max_line = int(os.environ.get("HL_MAX_LINE_BYTES", "65536"))
    max_output = int(os.environ.get("HL_MAX_OUTPUT_CHARS", "1000000"))
    if offset < 1:
        raise ValueError("offset must be >= 1")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if max_line < 1:
        raise ValueError("max_line_bytes must be >= 1")
    if max_output < 1:
        raise ValueError("max_output_chars must be >= 1")

    parent, name = parent_fd(os.environ["HL_FILE_PATH"], create=False)
    descriptor, metadata = open_regular(parent, name)
    sample = os.read(descriptor, 8192)
    kind = binary_kind(sample)
    if kind is not None:
        raise ValueError("unsupported %s file" % kind)
    os.lseek(descriptor, 0, os.SEEK_SET)
    binary_stream = os.fdopen(descriptor, "rb", closefd=False)
    text_stream = io.TextIOWrapper(
        binary_stream,
        encoding="utf-8",
        errors="strict",
        newline=None,
    )

    page = read_window(lambda: read_line(text_stream, max_line), offset=offset,
                       limit=limit, max_line_bytes=max_line, max_output_chars=max_output)
    if not page["success"]:
        raise ValueError(page["error"])
    print(page["output"], end="")
except UnicodeDecodeError:
    raise SystemExit("invalid UTF-8 text")
except ValueError as exc:
    raise SystemExit(str(exc))
except Exception:
    raise SystemExit("secure bounded nofollow read failed")
finally:
    if text_stream is not None:
        try:
            text_stream.detach()
        except Exception:
            pass
    if binary_stream is not None:
        try:
            binary_stream.close()
        except Exception:
            pass
    if descriptor is not None:
        try:
            os.close(descriptor)
        except Exception:
            pass
    if parent is not None:
        try:
            os.close(parent)
        except Exception:
            pass
'''


class HarborFileReadTool(_base.HarborFileReadTool):
    """Stream one canonical regular file without an unsafe shell fallback."""

    max_line_bytes = 65_536
    max_output_chars = 1_000_000
    description = (
        "Read bounded UTF-8 text from a canonical regular file inside the "
        "TerminalBench environment. Binary files and invalid UTF-8 are rejected."
    )

    def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int | None = 2000,
        **kwargs: Any,
    ) -> _base.ToolResult:
        validated_offset, failure = _v4._positive_integer(offset, name="offset")
        if failure is not None:
            return failure
        effective_limit = 2000 if limit is None else limit
        validated_limit, failure = _v4._positive_integer(
            effective_limit,
            name="limit",
        )
        if failure is not None:
            return failure
        validated_max_line, failure = _v4._positive_integer(
            self.max_line_bytes,
            name="max_line_bytes",
        )
        if failure is not None:
            return failure
        validated_max_output, failure = _v4._positive_integer(
            self.max_output_chars,
            name="max_output_chars",
        )
        if failure is not None:
            return failure
        assert validated_offset is not None
        assert validated_limit is not None
        assert validated_max_line is not None
        assert validated_max_output is not None

        resolved, path_failure = self._guard_environment_path(
            file_path,
            operation="read",
            must_exist=True,
        )
        if path_failure is not None:
            return path_failure

        result = self._run_secure_python(
            _STREAMING_READ_PYTHON,
            env={
                "HL_FILE_PATH": resolved,
                "HL_OFFSET": str(validated_offset),
                "HL_LIMIT": str(validated_limit),
                "HL_MAX_LINE_BYTES": str(validated_max_line),
                "HL_MAX_OUTPUT_CHARS": str(validated_max_output),
            },
        )
        result.metadata = {
            **result.metadata,
            "streaming_read": True,
            "physical_line_memory_bounded": True,
            "max_line_bytes": validated_max_line,
            "max_output_chars": validated_max_output,
            "encoding": "utf-8",
            "unsafe_shell_fallback_allowed": False,
        }
        if result.error.startswith("Output limit too small"):
            result.metadata.update(output_limit_too_small=True, lines_returned=0,
                                   next_offset=None, retry_offset=validated_offset)
        lowered_error = result.error.lower()
        if "unsupported" in lowered_error and "file" in lowered_error:
            result.metadata["binary_file_unsupported"] = True
        if "invalid utf-8" in lowered_error:
            result.metadata["text_decode_error"] = True
        return result


_base.HarborFileReadTool = HarborFileReadTool
HLWorkerHarborAgent = _base.HLWorkerHarborAgent
