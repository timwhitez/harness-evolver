from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools.file_read import FileReadTool


@pytest.mark.parametrize(
    ("name", "payload", "kind"),
    [
        ("image.png", b"\x89PNG\r\n\x1a\n" + b"payload", "PNG"),
        ("photo.jpg", b"\xff\xd8\xff" + b"payload", "JPEG"),
        ("document.pdf", b"%PDF-1.7\n" + b"payload", "PDF"),
        ("opaque.bin", b"prefix\x00suffix", "binary"),
    ],
)
def test_known_binary_files_fail_explicitly(
    tmp_path: Path,
    name: str,
    payload: bytes,
    kind: str,
) -> None:
    target = tmp_path / name
    target.write_bytes(payload)

    result = FileReadTool().execute(str(target))

    assert result.success is False
    assert result.output == ""
    assert result.metadata["binary_file_unsupported"] is True
    assert result.metadata["detected_file_type"] == kind
    assert "unsupported" in result.error


def test_invalid_utf8_is_not_returned_as_replacement_text(tmp_path: Path) -> None:
    target = tmp_path / "invalid.txt"
    target.write_bytes(b"valid prefix\n\xff\xfe\n")

    result = FileReadTool().execute(str(target))

    assert result.success is False
    assert result.output == ""
    assert result.metadata["text_decode_error"] is True
    assert result.metadata["encoding"] == "utf-8"


def test_utf8_text_keeps_numbered_output(tmp_path: Path) -> None:
    target = tmp_path / "source.py"
    target.write_text("first\nsecond\n", encoding="utf-8")

    result = FileReadTool().execute(str(target), offset=2, limit=1)

    assert result.success is True
    assert result.output.startswith("2\tsecond")
    assert result.metadata["start_line"] == 2
    assert result.metadata["end_line"] == 2


def test_schema_no_longer_claims_image_or_pdf_support() -> None:
    description = FileReadTool().get_schema().description.lower()

    assert "rejected explicitly" in description
    assert "can also read images" not in description
