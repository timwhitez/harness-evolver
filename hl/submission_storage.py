"""Exclusive, durable submission records on a shared local POSIX filesystem.

An intent is a permanent at-most-once claim, not an expiring lock. Never delete
or overwrite it automatically: after a crash, upload completion may be unknown.
All submitters for a campaign must use the same store. Network/distributed
filesystems without reliable O_EXCL and directory fsync are not supported.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any


def valid_campaign_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is not None


def _directory_fd(path: Path) -> int:
    if os.name != "posix" or not hasattr(os, "O_DIRECTORY"):
        raise OSError("durable submissions require POSIX directory fsync")
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)


def prepare_store(path: Path) -> None:
    """Create the store and persist newly created directory entries."""
    missing: list[Path] = []
    cursor = path.absolute()
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    if os.name != "posix":
        # Inspection/dry-run remains usable; claiming will fail closed.
        return
    # fsync from the leaves up, including the existing parent which acquired a
    # new directory entry. An unsupported durability primitive fails closed.
    for directory in [*missing, cursor] if missing else []:
        fd = _directory_fd(directory)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Claim once, flush bytes and the directory before returning.

    Serialization happens before claiming. Once O_EXCL succeeds, even a partial
    or empty intent is retained on error so a later caller cannot submit again.
    """
    data = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    parent = _directory_fd(path.parent)
    fd: int | None = None
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as stream:
            fd = None  # stream is now the sole owner, including on write error
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(parent)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)
