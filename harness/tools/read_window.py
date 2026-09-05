"""Whole-record pagination shared by local and embedded Harbor readers."""
from __future__ import annotations

import inspect


def read_window(next_line, *, offset: int, limit: int, max_line_bytes: int,
                max_output_chars: int) -> dict:
    """Consume bounded physical records and never skip footer-displaced text.

    next_line returns (text, explicitly_truncated) or None. The provider owns
    strict decoding and physical-line bounds. Only complete formatted records
    count as emitted. A successful continuation always advances the offset.
    """
    for name, value in (("offset", offset), ("limit", limit),
                        ("max_line_bytes", max_line_bytes),
                        ("max_output_chars", max_output_chars)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    records = []
    line_number = 0
    retained_chars = 0
    eof = False
    next_offset = None
    output_truncated = False
    blocked_record_chars = 0

    while line_number < offset - 1:
        if next_line() is None:
            eof = True
            break
        line_number += 1
    if not eof:
        while len(records) < limit:
            item = next_line()
            if item is None:
                eof = True
                break
            line_number += 1
            text, truncated = item
            suffix = f" ... [line truncated at {max_line_bytes} bytes]" if truncated else ""
            formatted = f"{line_number}\t{text}{suffix}"
            projected = retained_chars + len(formatted) + bool(records)
            if projected > max_output_chars:
                next_offset = line_number
                output_truncated = True
                blocked_record_chars = len(formatted)
                break
            records.append((line_number, formatted, truncated))
            retained_chars = projected
        if next_offset is None and not eof and len(records) == limit:
            following = next_line()
            if following is None:
                eof = True
            else:
                line_number += 1
                next_offset = line_number

    # Reserve the complete footer by removing whole records, not slicing text
    # which has already been counted. Continuation starts at the first removal.
    marker = ""
    if next_offset is not None:
        while True:
            marker = f"... (more lines, use offset={next_offset} to continue)"
            if records and retained_chars + 1 + len(marker) <= max_output_chars:
                break
            if not records:
                required = max(blocked_record_chars, 1) + 1 + len(
                    f"... (more lines, use offset={offset + 1} to continue)"
                )
                return {"success": False, "output": "",
                        "error": ("Output limit too small for one complete numbered line and "
                                  "continuation. Increase max_output_chars to at least "
                                  f"{required}, or lower the explicit physical-line cap."),
                        "metadata": {"start_line": offset, "end_line": offset - 1,
                                     "lines_returned": 0, "has_more": True,
                                     "next_offset": None, "retry_offset": offset,
                                     "output_limit_too_small": True,
                                     "required_output_chars": required,
                                     "output_truncated": True, "line_truncated_count": 0,
                                     "total_lines_known": eof}}
            number, removed, _ = records.pop()
            retained_chars -= len(removed) + bool(records)
            blocked_record_chars = len(removed)
            next_offset = number
            output_truncated = True

    output = "\n".join(record[1] for record in records)
    if marker:
        output += "\n" + marker
    metadata = {"start_line": offset,
                "end_line": records[-1][0] if records else (line_number if eof else offset - 1),
                "lines_returned": len(records), "has_more": next_offset is not None,
                "next_offset": next_offset,
                "line_truncated_count": sum(bool(record[2]) for record in records),
                "output_truncated": output_truncated, "total_lines_known": eof}
    if eof:
        metadata["total_lines"] = line_number
    return {"success": True, "output": output, "error": "", "metadata": metadata}


def embedded_window_source() -> str:
    return inspect.getsource(read_window) + "\n"
