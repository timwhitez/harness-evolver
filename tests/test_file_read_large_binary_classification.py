from __future__ import annotations

from pathlib import Path

from harness.tools.file_read import FileReadTool


def test_oversized_known_binary_keeps_binary_diagnostic(tmp_path: Path) -> None:
    target = tmp_path / "large.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 256)

    result = FileReadTool(max_file_bytes=64).execute(str(target))

    assert result.success is False
    assert result.output == ""
    assert result.metadata["binary_file_unsupported"] is True
    assert result.metadata["detected_file_type"] == "PNG"
    assert result.metadata["read_memory_bounded"] is True
    assert result.metadata["max_file_bytes"] == 64
