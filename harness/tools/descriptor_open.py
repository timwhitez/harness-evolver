"""Acquire readable regular-file descriptors without opening unvalidated devices.

Linux O_PATH pins and inspects the inode without invoking device/FIFO read-open
semantics. A trusted procfs then reopens that owned descriptor, not the mutable
caller path. Unsupported platforms/procfs fail closed instead of blocking.
"""
from __future__ import annotations

import inspect


def open_readonly_checked(parent_fd: int, name: str, *, unique: bool = True,
                          allow_directory: bool = False) -> tuple[int, object]:
    """Return a readable fd and its fstat; caller owns the returned fd."""
    import os
    import stat
    import sys

    if not sys.platform.startswith("linux") or not hasattr(os, "O_PATH"):
        raise OSError("safe descriptor acquisition requires Linux O_PATH and procfs")
    if not isinstance(name, str) or name in ("", ".", "..") or "/" in name:
        raise ValueError("descriptor target must be one path component")
    anchor = os.open(name, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
    readable = None
    try:
        observed = os.fstat(anchor)
        regular = stat.S_ISREG(observed.st_mode)
        directory = allow_directory and stat.S_ISDIR(observed.st_mode)
        if not (regular or directory):
            raise ValueError("not a regular file")
        if regular and unique and observed.st_nlink != 1:
            raise ValueError("Refusing multiply linked regular file")
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        # This is an owned kernel descriptor in trusted procfs, deliberately
        # not an ordinary path and not a user-controlled symlink to follow.
        readable = os.open(f"/proc/self/fd/{anchor}", flags)
        actual = os.fstat(readable)
        if ((actual.st_dev, actual.st_ino, stat.S_IFMT(actual.st_mode)) !=
                (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode))):
            raise ValueError("descriptor identity changed during acquisition")
        if regular and unique and actual.st_nlink != 1:
            raise ValueError("Refusing multiply linked regular file")
        owned_anchor, anchor = anchor, None
        os.close(owned_anchor)
        result, readable = readable, None
        return result, actual
    finally:
        if readable is not None:
            os.close(readable)
        if anchor is not None:
            os.close(anchor)


def embedded_opener_source() -> str:
    """Use the exact same implementation in trusted container-side scripts."""
    return inspect.getsource(open_readonly_checked) + "\n"
