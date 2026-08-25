"""Bounded byte readers layered on the canonical no-follow path boundary."""

from __future__ import annotations

import os
from pathlib import Path

from harness.tools import _safe_path_io_issue18_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


def _open_regular_nofollow(
    parent_fd: int,
    name: str,
    target: Path,
) -> tuple[int, os.stat_result]:
    """Open one uniquely linked regular file relative to a stable parent."""

    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        _base._validate_unique_regular_file(metadata, target)
        return descriptor, metadata
    except Exception:
        os.close(descriptor)
        raise


def read_bounded_bytes_nofollow(
    path: str | os.PathLike[str],
    *,
    max_bytes: int,
) -> tuple[bytes, os.stat_result, bool]:
    """Read at most ``max_bytes`` plus one proof byte from one descriptor."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be an integer >= 1")

    with _base._open_parent_nofollow(path, create_parents=False) as (
        parent_fd,
        name,
        target,
    ):
        descriptor, metadata = _open_regular_nofollow(parent_fd, name, target)
        try:
            remaining = max_bytes + 1
            chunks: list[bytes] = []
            while remaining > 0:
                chunk = os.read(descriptor, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            return payload[:max_bytes], metadata, len(payload) > max_bytes
        finally:
            os.close(descriptor)


def read_bytes_nofollow(
    path: str | os.PathLike[str],
) -> tuple[bytes, os.stat_result]:
    """Read one unique regular file once through a stable descriptor."""

    with _base._open_parent_nofollow(path, create_parents=False) as (
        parent_fd,
        name,
        target,
    ):
        descriptor, metadata = _open_regular_nofollow(parent_fd, name, target)
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 65_536)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks), metadata
        finally:
            os.close(descriptor)
