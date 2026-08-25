"""Compatibility base for the clean Issue #17 streaming integration."""

from __future__ import annotations

from harness.tools._file_read_issue18_base import FileReadTool


_BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "PDF"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
)


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


__all__ = ["FileReadTool", "_unsupported_binary_kind"]
