from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from bench.harbor_adapter import _HARBOR_GREP_COUNTING_PYTHON
from harness.tools.search import GrepTool


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="canonical descriptor-relative grep is POSIX-specific",
)


def test_local_invalid_utf8_is_a_read_failure_not_zero_match(tmp_path: Path) -> None:
    target = tmp_path / "invalid.bin"
    target.write_bytes(b"ordinary-prefix\n\xff\xfe\n")

    result = GrepTool().execute("definitely-missing", path=str(tmp_path))

    assert result.success is False
    assert result.output == ""
    assert result.metadata["search_failed"] is True
    assert result.metadata["text_decode_error"] is True
    assert result.metadata["strict_text_decoding"] is True
    assert result.metadata["read_error_count"] == 1
    assert "invalid UTF-8" in result.error


def test_local_partial_matches_do_not_hide_later_decode_failure(tmp_path: Path) -> None:
    (tmp_path / "a-valid.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "z-invalid.bin").write_bytes(b"\xff\n")

    result = GrepTool().execute("needle", path=str(tmp_path))

    assert result.success is False
    assert "needle" in result.output
    assert result.metadata["partial_results_available"] is True
    assert result.metadata["text_decode_error"] is True


def test_harbor_embedded_scanner_rejects_invalid_utf8_before_count_output(
    tmp_path: Path,
) -> None:
    (tmp_path / "invalid.bin").write_bytes(b"prefix\n\xff\xfe\n")

    completed = subprocess.run(
        [sys.executable, "-c", _HARBOR_GREP_COUNTING_PYTHON],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "HL_ROOT": str(tmp_path),
            "HL_PATTERN": "missing",
            "HL_MAX_RESULTS": "10",
            "HL_MAX_MATCH_CHARS": "100",
            "HL_MAX_INPUT_LINE_CHARS": "1000",
        },
    )

    assert completed.returncode != 0
    assert "__HL_GREP_COUNT__" not in completed.stdout
    assert "UnicodeDecodeError" in completed.stderr
