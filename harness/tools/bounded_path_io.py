"""Bounded and streaming readers over the canonical no-follow path boundary."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import BinaryIO, Iterator

from harness.tools.descriptor_open import open_readonly_checked
from harness.tools.safe_path_io import SafePathError, _open_parent_nofollow


def _open_regular_nofollow(
    parent_fd: int,
    name: str,
    target: Path,
) -> tuple[int, os.stat_result]:
    """Pin/type-check before a readable open; never wait for a FIFO peer."""
    try:
        return open_readonly_checked(parent_fd, name)
    except ValueError as exc:
        raise SafePathError(f"Cannot open regular target safely: {target}: {exc}") from exc


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
        stream = None
        try:
            stream = os.fdopen(descriptor, "rb", buffering=0, closefd=False)
            yield stream, metadata
        finally:
            try:
                if stream is not None:
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
