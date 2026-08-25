"""Bounded and streaming readers over the canonical no-follow path boundary."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import BinaryIO, Iterator

from harness.tools.safe_path_io import (
    _open_parent_nofollow,
    _validate_unique_regular_file,
)


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
        _validate_unique_regular_file(metadata, target)
        return descriptor, metadata
    except Exception:
        os.close(descriptor)
        raise


@contextmanager
def open_binary_nofollow(
    path: str | os.PathLike[str],
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Yield one stable, unique regular-file stream without reopening its path."""

    with _open_parent_nofollow(path, create_parents=False) as (
        parent_fd,
        name,
        target,
    ):
        descriptor, metadata = _open_regular_nofollow(parent_fd, name, target)
        stream = os.fdopen(descriptor, "rb", buffering=0, closefd=False)
        try:
            yield stream, metadata
        finally:
            try:
                stream.close()
            finally:
                os.close(descriptor)


def read_bounded_bytes_nofollow(
    path: str | os.PathLike[str],
    *,
    max_bytes: int,
) -> tuple[bytes, os.stat_result, bool]:
    """Read at most ``max_bytes`` plus one proof byte from one descriptor."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be an integer >= 1")

    with open_binary_nofollow(path) as (stream, metadata):
        remaining = max_bytes + 1
        chunks: list[bytes] = []
        while remaining > 0:
            chunk = stream.read(min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        return payload[:max_bytes], metadata, len(payload) > max_bytes


def read_bytes_nofollow(
    path: str | os.PathLike[str],
) -> tuple[bytes, os.stat_result]:
    """Read one unique regular file once through a stable descriptor."""

    with open_binary_nofollow(path) as (stream, metadata):
        chunks: list[bytes] = []
        while True:
            chunk = stream.read(65_536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), metadata
