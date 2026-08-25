"""Descriptor-relative filesystem operations that reject path-alias races."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import stat
from typing import Callable, Iterator


class SafePathError(OSError):
    """Raised when a path cannot be attributed to one safe regular-file entry."""


@contextmanager
def _open_parent_nofollow(
    path: str | os.PathLike[str],
    *,
    create_parents: bool,
) -> Iterator[tuple[int, str, Path]]:
    """Yield a stable parent descriptor and final basename for an absolute path."""

    target = Path(path)
    if not target.is_absolute():
        target = Path.cwd() / target
    target = Path(os.path.normpath(os.fspath(target)))
    if target.name in {"", ".", ".."}:
        raise SafePathError(f"Invalid final path component: {target}")

    required_flags = ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC")
    if os.name != "posix" or any(not hasattr(os, name) for name in required_flags):
        raise SafePathError(
            "Descriptor-relative O_NOFOLLOW path access is unavailable on this platform"
        )

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    parent_fd = os.open(os.sep, directory_flags)
    try:
        for component in target.parent.parts[1:]:
            try:
                next_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                if not create_parents:
                    raise
                os.mkdir(component, mode=0o755, dir_fd=parent_fd)
                next_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise SafePathError(
                    f"Cannot open path component without following symlinks: {component}"
                ) from exc
            os.close(parent_fd)
            parent_fd = next_fd
        yield parent_fd, target.name, target
    finally:
        os.close(parent_fd)


def _validate_unique_regular_file(
    metadata: os.stat_result,
    target: Path,
) -> None:
    """Reject special files and hard-link aliases before reading content.

    Canonical path resolution and ``O_NOFOLLOW`` can prove that a directory
    entry is not a symlink, but a regular file with multiple hard links can have
    an allowed-looking name while sharing the same inode as hidden verifier or
    host-memory content. The filesystem does not provide a race-free way to
    enumerate and authorize every other name for that inode. Content reads
    therefore fail closed unless the opened inode has exactly one link.
    """

    if not stat.S_ISREG(metadata.st_mode):
        raise SafePathError(f"Path is not a regular file: {target}")
    if metadata.st_nlink != 1:
        raise SafePathError(f"Refusing multiply linked regular file: {target}")


def read_text_nofollow(
    path: str | os.PathLike[str],
    *,
    errors: str = "replace",
) -> tuple[str, os.stat_result]:
    """Read one uniquely linked regular file through stable descriptors."""

    with _open_parent_nofollow(path, create_parents=False) as (parent_fd, name, target):
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            metadata = os.fstat(descriptor)
            _validate_unique_regular_file(metadata, target)
            with os.fdopen(
                descriptor,
                "r",
                encoding="utf-8",
                errors=errors,
                closefd=False,
            ) as stream:
                content = stream.read()
            return content, metadata
        finally:
            os.close(descriptor)


def atomic_write_text_nofollow(
    path: str | os.PathLike[str],
    content: str,
    *,
    mode: int | None = None,
) -> None:
    """Atomically replace a canonical path through a stable parent descriptor.

    A pure overwrite does not consume existing file content. Replacing a
    hard-linked directory entry is therefore safe and intentionally de-aliases
    it: the other inode names remain unchanged and the published target becomes
    a new uniquely linked regular file. Append/edit operations read first and
    are rejected by :func:`read_text_nofollow` when the existing inode is
    multiply linked.

    The temporary file uses normal ``0666 & ~umask`` creation semantics for a
    new target and inherits only the existing target's ordinary ``0o777``
    permission bits for replacement. Setuid, setgid, and sticky bits are never
    recreated on newly written content. Every failure before ``os.replace``
    removes the temporary file and leaves the previous target bytes unchanged.
    """

    payload = content.encode("utf-8")
    with _open_parent_nofollow(path, create_parents=True) as (parent_fd, name, target):
        existing_mode = mode
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(current.st_mode):
                raise SafePathError(f"Refusing symlink target: {target}")
            if existing_mode is None:
                existing_mode = current.st_mode
        except FileNotFoundError:
            pass

        temporary_name = f".{name}.tmp-{secrets.token_hex(8)}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = os.open(
                temporary_name,
                flags,
                0o666,
                dir_fd=parent_fd,
            )
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                if existing_mode is not None:
                    os.fchmod(
                        descriptor,
                        stat.S_IMODE(existing_mode) & 0o777,
                    )
                os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1

            try:
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if stat.S_ISLNK(current.st_mode):
                    raise SafePathError(f"Refusing symlink target: {target}")
            except FileNotFoundError:
                pass

            os.replace(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def edit_text_nofollow(
    path: str | os.PathLike[str],
    transform: Callable[[str], str],
) -> tuple[str, str]:
    """Read a stable unique file and atomically publish transformed content."""

    original, metadata = read_text_nofollow(path, errors="strict")
    updated = transform(original)
    atomic_write_text_nofollow(path, updated, mode=metadata.st_mode)
    return original, updated
