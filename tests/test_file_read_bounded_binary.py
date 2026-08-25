from __future__ import annotations

from pathlib import Path

from harness.tools.file_read import FileReadTool


def test_nul_after_the_old_sample_window_is_still_rejected(tmp_path: Path) -> None:
    target = tmp_path / "late-binary.txt"
    target.write_bytes(b"a" * 9000 + b"\x00tail")

    result = FileReadTool().execute(str(target))

    assert result.success is False
    assert result.output == ""
    assert result.metadata["binary_file_unsupported"] is True
    assert result.metadata["detected_file_type"] == "binary"


def test_oversized_file_fails_before_unbounded_buffering(tmp_path: Path) -> None:
    target = tmp_path / "large.txt"
    target.write_bytes(b"x" * 65)

    result = FileReadTool(max_file_bytes=64).execute(str(target))

    assert result.success is False
    assert result.output == ""
    assert result.metadata["file_too_large"] is True
    assert result.metadata["max_file_bytes"] == 64
    assert result.metadata["file_size"] >= 65


def test_file_growth_past_limit_is_also_rejected(tmp_path: Path) -> None:
    target = tmp_path / "bounded.txt"
    target.write_bytes(b"small text\n")

    result = FileReadTool(max_file_bytes=4).execute(str(target))

    assert result.success is False
    assert result.metadata["file_too_large"] is True
