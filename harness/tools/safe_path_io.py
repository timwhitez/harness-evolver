"""Streaming binary descriptor access layered on Issue #18 path I/O."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import BinaryIO, Iterator

from harness.tools import _safe_path_io_issue17_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


@contextmanager
def open_binary_nofollow(
    path: str | os.PathLike[str],
) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Yield one seekable stream from a unique, descriptor-bound regular file."""

    with _base._open_parent_nofollow(path, create_parents=False) as (
        parent_fd,
        name,
        target,
    ):
        descriptor, metadata = _base._open_regular_nofollow(
            parent_fd,
            name,
            target,
        )
        stream = os.fdopen(descriptor, "rb", closefd=False)
        try:
            yield stream, metadata
        finally:
            try:
                stream.close()
            finally:
                os.close(descriptor)
