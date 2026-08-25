from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from bench.harbor_adapter import _STREAMING_READ_PYTHON
from harness.tools.file_read import FileReadTool


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="canonical O_NOFOLLOW reader is POSIX-specific",
)


def test_local_read_does_not_call_full_file_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "large.log"
    target.write_text(
        "".join(f"line-{index}\n" for index in range(10_000)),
        encoding="utf-8",
    )

    def fail_full_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("full-file convenience helpers must not be used")

    monkeypatch.setattr(Path, "read_text", fail_full_read)
    monkeypatch.setattr(Path, "read_bytes", fail_full_read)

    result = FileReadTool().execute(str(target), offset=5_000, limit=2)

    assert result.success is True
    assert result.output.startswith("5000\tline-4999\n5001\tline-5000")
    assert "use offset=5002" in result.output
    assert result.metadata["lines_returned"] == 2
    assert result.metadata["has_more"] is True
    assert result.metadata["streaming_read"] is True
    assert result.metadata["nofollow_io"] is True


def test_local_read_bounds_a_single_long_line(tmp_path: Path) -> None:
    target = tmp_path / "long.txt"
    target.write_bytes(b"x" * 2_000_000 + b"\nsecond\n")
    tool = FileReadTool(max_line_bytes=128, max_output_chars=256)

    result = tool.execute(str(target), limit=1)

    assert result.success is True
    assert len(result.output) <= 256
    assert "line truncated at 128 bytes" in result.output
    assert result.metadata["line_truncated_count"] == 1
    assert result.metadata["has_more"] is True
    assert result.metadata["next_offset"] == 2
    assert result.metadata["physical_line_memory_bounded"] is True


def test_invalid_utf8_after_retained_prefix_still_fails(tmp_path: Path) -> None:
    target = tmp_path / "invalid-long.txt"
    target.write_bytes(b"a" * 1_000_000 + b"\xff\n")

    result = FileReadTool(max_line_bytes=64).execute(str(target), limit=1)

    assert result.success is False
    assert result.output == ""
    assert result.metadata["text_decode_error"] is True


def test_offset_beyond_eof_reports_known_total_without_scanning_twice(tmp_path: Path) -> None:
    target = tmp_path / "small.txt"
    target.write_text("one\ntwo\n", encoding="utf-8")

    result = FileReadTool().execute(str(target), offset=10, limit=3)

    assert result.success is True
    assert result.output == ""
    assert result.metadata["total_lines_known"] is True
    assert result.metadata["total_lines"] == 2
    assert result.metadata["lines_returned"] == 0
    assert result.metadata["end_line"] == 2


def _run_harbor_script(
    target: Path,
    *,
    offset: int,
    limit: int,
    max_line: int = 65_536,
    max_output: int = 1_000_000,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _STREAMING_READ_PYTHON],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HL_FILE_PATH": str(target),
            "HL_OFFSET": str(offset),
            "HL_LIMIT": str(limit),
            "HL_MAX_LINE_BYTES": str(max_line),
            "HL_MAX_OUTPUT_CHARS": str(max_output),
        },
        check=False,
    )


def test_harbor_reader_stops_after_window_and_probe(tmp_path: Path) -> None:
    target = tmp_path / "large.log"
    target.write_text(
        "".join(f"line-{index}\n" for index in range(100)),
        encoding="utf-8",
    )

    completed = _run_harbor_script(target, offset=20, limit=2)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("20\tline-19\n21\tline-20")
    assert "use offset=22" in completed.stdout
    assert "line-99" not in completed.stdout


def test_harbor_reader_bounds_long_line_and_total_output(tmp_path: Path) -> None:
    target = tmp_path / "long.txt"
    target.write_bytes(b"x" * 2_000_000 + b"\nsecond\n")

    completed = _run_harbor_script(
        target,
        offset=1,
        limit=1,
        max_line=128,
        max_output=256,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout) <= 256
    assert completed.stdout.startswith("1\t" + "x" * 128)
    assert "line truncated at 128 bytes" in completed.stdout
    assert "use offset=2" in completed.stdout


@pytest.mark.parametrize(
    "payload",
    [
        b"%PDF-1.7\npayload",
        b"prefix\x00suffix",
        b"valid-prefix\n\xff\xfe\n",
    ],
)
def test_harbor_reader_rejects_binary_or_invalid_utf8(
    tmp_path: Path,
    payload: bytes,
) -> None:
    target = tmp_path / "invalid.dat"
    target.write_bytes(payload)

    completed = _run_harbor_script(target, offset=1, limit=10)

    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "unsupported" in completed.stderr or "invalid UTF-8" in completed.stderr
