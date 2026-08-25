"""Descriptor-relative filesystem operations that reject path-alias races."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import secrets
import stat
from typing import Callable, Iterator


class SafePathError(OSError):
    """Raised when a path cannot be attributed to one safe regular-file entry."""


FileIdentity = tuple[int, int]


def file_identity(metadata: os.stat_result) -> FileIdentity:
    """Return the filesystem identity used for read/transform/publish checks."""

    return int(metadata.st_dev), int(metadata.st_ino)


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
    """Reject special files and hard-link aliases before reading content."""

    if not stat.S_ISREG(metadata.st_mode):
        raise SafePathError(f"Path is not a regular file: {target}")
    if metadata.st_nlink != 1:
        raise SafePathError(f"Refusing multiply linked regular file: {target}")


def _read_descriptor_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 65_536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _validate_expected_target(
    parent_fd: int,
    name: str,
    target: Path,
    *,
    expected_identity: FileIdentity | None,
    expected_sha256: str | None,
    expected_missing: bool,
) -> os.stat_result | None:
    """Revalidate a transform source immediately before publication.

    Whole-file overwrite deliberately passes no expectation and may de-alias a
    hard-linked destination. Edit/append-style transforms pass the identity and
    digest of the uniquely linked inode they consumed. A replaced, removed,
    relinked, or modified target then fails before publication.
    """

    try:
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if expected_missing:
            return None
        if expected_identity is not None or expected_sha256 is not None:
            raise SafePathError(f"Target disappeared before publication: {target}")
        return None

    if stat.S_ISLNK(observed.st_mode):
        raise SafePathError(f"Refusing symlink target: {target}")
    if expected_missing:
        raise SafePathError(f"Target appeared before publication: {target}")

    if expected_identity is None and expected_sha256 is None:
        return observed

    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        actual = os.fstat(descriptor)
        _validate_unique_regular_file(actual, target)
        if expected_identity is not None and file_identity(actual) != expected_identity:
            raise SafePathError(f"Target changed identity before publication: {target}")
        if expected_sha256 is not None:
            digest = hashlib.sha256(_read_descriptor_bytes(descriptor)).hexdigest()
            if digest != expected_sha256:
                raise SafePathError(f"Target content changed before publication: {target}")
        return actual
    finally:
        os.close(descriptor)


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
    expected_identity: FileIdentity | None = None,
    expected_sha256: str | None = None,
    expected_missing: bool = False,
) -> bool:
    """Atomically replace a path through a stable parent descriptor.

    Returns whether the parent-directory fsync succeeded. Once ``os.replace``
    succeeds the new inode is already published; a later directory-fsync error
    is therefore reported as a durability warning rather than a false
    pre-publication failure that callers might retry unsafely.
    """

    payload = content.encode("utf-8")
    with _open_parent_nofollow(path, create_parents=True) as (parent_fd, name, target):
        current = _validate_expected_target(
            parent_fd,
            name,
            target,
            expected_identity=expected_identity,
            expected_sha256=expected_sha256,
            expected_missing=expected_missing,
        )
        existing_mode = mode
        if existing_mode is None and current is not None:
            existing_mode = current.st_mode

        temporary_name = f".{name}.tmp-{secrets.token_hex(8)}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = -1
        published = False
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
                    os.fchmod(descriptor, stat.S_IMODE(existing_mode) & 0o777)
                os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1

            _validate_expected_target(
                parent_fd,
                name,
                target,
                expected_identity=expected_identity,
                expected_sha256=expected_sha256,
                expected_missing=expected_missing,
            )
            os.replace(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            published = True
            try:
                os.fsync(parent_fd)
            except OSError:
                return False
            return True
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if not published:
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
    atomic_write_text_nofollow(
        path,
        updated,
        mode=metadata.st_mode,
        expected_identity=file_identity(metadata),
        expected_sha256=hashlib.sha256(original.encode("utf-8")).hexdigest(),
    )
    return original, updated
