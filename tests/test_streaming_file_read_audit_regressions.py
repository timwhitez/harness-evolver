from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import MethodType

import pytest

from bench.harbor_adapter import HarborFileReadTool, _STREAMING_READ_PYTHON
from harness.tools.file_read import FileReadTool


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="canonical O_NOFOLLOW reader is POSIX-specific",
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_line_bytes", True),
        ("max_line_bytes", 64.5),
        ("max_output_chars", False),
        ("max_output_chars", "1024"),
    ],
)
def test_local_streaming_bounds_require_real_positive_integers(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    target = tmp_path / "data.txt"
    target.write_text("data\n", encoding="utf-8")
    tool = FileReadTool()
    setattr(tool, field, value)

    result = tool.execute(str(target))

    assert result.success is False
    assert result.output == ""
    assert result.error == f"{field} must be an integer >= 1"
    assert result.metadata["parameter_validation_failed"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_line_bytes", True),
        ("max_line_bytes", 64.5),
        ("max_output_chars", False),
        ("max_output_chars", "1024"),
    ],
)
def test_harbor_streaming_bounds_fail_before_environment_access(
    field: str,
    value: object,
) -> None:
    tool = object.__new__(HarborFileReadTool)
    tool.max_line_bytes = 65_536
    tool.max_output_chars = 1_000_000
    setattr(tool, field, value)

    def unexpected_guard(
        self: HarborFileReadTool,
        *args: object,
        **kwargs: object,
    ) -> tuple[str, object]:
        raise AssertionError("invalid bounds must fail before environment access")

    tool._guard_environment_path = MethodType(unexpected_guard, tool)  # type: ignore[method-assign]

    result = tool.execute("/app/data.txt")

    assert result.success is False
    assert result.output == ""
    assert field in result.error
    assert result.metadata["parameter_validation_failed"] is True


def test_known_binary_identity_precedes_a_tiny_total_size_cap(tmp_path: Path) -> None:
    target = tmp_path / "document.pdf"
    target.write_bytes(b"%PDF-1.7\n" + b"payload" * 32)

    result = FileReadTool(max_file_bytes=1).execute(str(target))

    assert result.success is False
    assert result.metadata["binary_file_unsupported"] is True
    assert result.metadata["detected_file_type"] == "PDF"
    assert result.metadata.get("file_too_large") is not True


def test_total_size_cap_detects_growth_after_the_descriptor_is_opened(
    tmp_path: Path,
) -> None:
    target = tmp_path / "growing.txt"
    target.write_text("one\n", encoding="utf-8")

    class GrowingFileReadTool(FileReadTool):
        def _stream_window(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            with target.open("ab") as stream:
                stream.write(b"x" * 128 + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            return super()._stream_window(*args, **kwargs)

    result = GrowingFileReadTool(max_file_bytes=64).execute(str(target), limit=1)

    assert result.success is False
    assert result.output == ""
    assert result.metadata["file_too_large"] is True
    assert result.metadata["live_descriptor_size_checked"] is True
    assert result.metadata["file_size"] > 64


def test_harbor_reader_rejects_nul_after_the_initial_signature_sample(
    tmp_path: Path,
) -> None:
    target = tmp_path / "late-binary.dat"
    target.write_bytes(b"a" * 9_000 + b"\x00\n")

    completed = subprocess.run(
        [sys.executable, "-c", _STREAMING_READ_PYTHON],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HL_FILE_PATH": str(target),
            "HL_OFFSET": "1",
            "HL_LIMIT": "1",
            "HL_MAX_LINE_BYTES": "64",
            "HL_MAX_OUTPUT_CHARS": "1024",
        },
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "unsupported binary file" in completed.stderr


def test_harbor_reader_rejects_late_control_heavy_content(tmp_path: Path) -> None:
    target = tmp_path / "late-controls.dat"
    target.write_bytes(b"a" * 9_000 + b"\x01" * 5_000 + b"\n")

    completed = subprocess.run(
        [sys.executable, "-c", _STREAMING_READ_PYTHON],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HL_FILE_PATH": str(target),
            "HL_OFFSET": "1",
            "HL_LIMIT": "1",
            "HL_MAX_LINE_BYTES": "64",
            "HL_MAX_OUTPUT_CHARS": "1024",
        },
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "unsupported binary file" in completed.stderr
