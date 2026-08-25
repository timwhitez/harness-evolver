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


def test_exact_line_cap_with_lf_is_not_marked_truncated(tmp_path: Path) -> None:
    target = tmp_path / "lf.txt"
    target.write_bytes(b"abcd\n")

    result = FileReadTool(max_line_bytes=4).execute(str(target))

    assert result.success is True
    assert result.output == "1\tabcd"
    assert result.metadata["line_truncated_count"] == 0


def test_exact_line_cap_with_crlf_is_not_marked_truncated(tmp_path: Path) -> None:
    target = tmp_path / "crlf.txt"
    target.write_bytes(b"abcd\r\n")

    result = FileReadTool(max_line_bytes=4).execute(str(target))

    assert result.success is True
    assert result.output == "1\tabcd"
    assert result.metadata["line_truncated_count"] == 0


def test_multibyte_cap_is_measured_in_utf8_bytes(tmp_path: Path) -> None:
    exact = tmp_path / "exact-utf8.txt"
    over = tmp_path / "over-utf8.txt"
    exact.write_text("éé\n", encoding="utf-8")
    over.write_text("ééé\n", encoding="utf-8")

    exact_result = FileReadTool(max_line_bytes=4).execute(str(exact))
    over_result = FileReadTool(max_line_bytes=4).execute(str(over))

    assert exact_result.success is True
    assert exact_result.output == "1\téé"
    assert exact_result.metadata["line_truncated_count"] == 0
    assert over_result.success is True
    assert over_result.output.startswith("1\téé")
    assert "line truncated at 4 bytes" in over_result.output


def test_harbor_reader_does_not_count_crlf_in_content_cap(tmp_path: Path) -> None:
    target = tmp_path / "harbor.txt"
    target.write_bytes(b"abcd\r\n")
    env = {
        **os.environ,
        "HL_FILE_PATH": str(target),
        "HL_OFFSET": "1",
        "HL_LIMIT": "1",
        "HL_MAX_LINE_BYTES": "4",
        "HL_MAX_OUTPUT_CHARS": "1000",
    }

    completed = subprocess.run(
        [sys.executable, "-c", _STREAMING_READ_PYTHON],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "1\tabcd"
    assert "line truncated" not in completed.stdout


def test_tiny_output_limit_never_exceeds_configured_chars(tmp_path: Path) -> None:
    target = tmp_path / "tiny-output.txt"
    target.write_text("first\nsecond\n", encoding="utf-8")

    result = FileReadTool(max_output_chars=12).execute(str(target), limit=1)

    assert result.success is True
    assert len(result.output) <= 12
    assert result.metadata["output_truncated"] is True
