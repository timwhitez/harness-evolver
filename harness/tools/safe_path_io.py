"""Descriptor-relative filesystem operations that reject path-alias races."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import secrets
import stat
import sys
from typing import Callable, Iterator


class SafePathError(OSError):
    """Raised when a path cannot be attributed to one safe regular-file entry."""


FileIdentity = tuple[int, int]
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_RENAMEAT2_SYSCALLS = {
    "x86_64": 316,
    "amd64": 316,
    "i386": 353,
    "i686": 353,
    "aarch64": 276,
    "arm64": 276,
    "armv7l": 382,
    "ppc64": 357,
    "ppc64le": 357,
    "s390x": 347,
    "riscv64": 276,
}


def file_identity(metadata: os.stat_result) -> FileIdentity:
    """Return the filesystem identity used for read/transform/publish checks."""

    return int(metadata.st_dev), int(metadata.st_ino)


def _renameat2(
    parent_fd: int,
    source: str,
    destination: str,
    flags: int,
) -> None:
    """Invoke Linux renameat2 or fail closed when conditional rename is absent."""

    if not sys.platform.startswith("linux"):
        raise SafePathError(
            "Race-safe conditional publication requires Linux renameat2"
        )

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    function = getattr(libc, "renameat2", None)
    if function is not None:
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent_fd,
            source_bytes,
            parent_fd,
            destination_bytes,
            flags,
        )
    else:
        syscall_number = _RENAMEAT2_SYSCALLS.get(platform.machine().lower())
        if syscall_number is None:
            raise SafePathError(
                "Race-safe conditional publication is unavailable on this architecture"
            )
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        result = syscall(
            ctypes.c_long(syscall_number),
            ctypes.c_int(parent_fd),
            ctypes.c_char_p(source_bytes),
            ctypes.c_int(parent_fd),
            ctypes.c_char_p(destination_bytes),
            ctypes.c_uint(flags),
        )

    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


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


def _open_and_validate_expected(
    parent_fd: int,
    name: str,
    target: Path,
    *,
    expected_identity: FileIdentity | None,
    expected_sha256: str | None,
) -> os.stat_result:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        _validate_unique_regular_file(metadata, target)
        if expected_identity is not None and file_identity(metadata) != expected_identity:
            raise SafePathError(f"Target changed identity before publication: {target}")
        if expected_sha256 is not None:
            digest = hashlib.sha256(_read_descriptor_bytes(descriptor)).hexdigest()
            if digest != expected_sha256:
                raise SafePathError(f"Target content changed before publication: {target}")
        return metadata
    finally:
        os.close(descriptor)


def _stat_unconditional_target(
    parent_fd: int,
    name: str,
    target: Path,
) -> os.stat_result | None:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise SafePathError(f"Refusing symlink target: {target}")
    if not stat.S_ISREG(metadata.st_mode):
        raise SafePathError(f"Refusing non-regular overwrite target: {target}")
    return metadata


def _validate_displaced_unconditional_target(
    parent_fd: int,
    name: str,
    target: Path,
    *,
    expected_identity: FileIdentity,
) -> os.stat_result:
    """Validate the exact regular entry displaced by a pure overwrite.

    Multiply-linked regular files are intentionally accepted here. A pure
    whole-file replacement never consumes their old bytes and safely de-aliases
    only the selected directory entry.
    """

    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise SafePathError(f"Refusing non-regular overwrite target: {target}")
    if file_identity(metadata) != expected_identity:
        raise SafePathError(
            f"Overwrite target changed identity before publication: {target}"
        )
    return metadata


def _assert_name_identity(
    parent_fd: int,
    name: str,
    expected: FileIdentity,
    target: Path,
) -> None:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        actual = os.fstat(descriptor)
        if file_identity(actual) != expected:
            raise SafePathError(
                f"Published target changed before rollback could complete: {target}"
            )
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
                newline="",
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

    Transform writes use ``RENAME_EXCHANGE``: the new inode and current target
    are swapped atomically, then the displaced inode is validated against the
    exact identity and digest consumed by the caller. A mismatch is exchanged
    back before failure, so an ordinary-file replacement at the publication
    boundary is not overwritten. Missing-target appends use ``RENAME_NOREPLACE``.

    Pure overwrites use the same conditional-publication boundary. An initially
    missing target is published with ``RENAME_NOREPLACE``. An existing regular
    target is exchanged, then the displaced entry's type and identity are
    verified before it is removed. A special entry or identity race is rolled
    back atomically instead of being destroyed.

    Returns whether the parent-directory fsync succeeded. Once publication is
    complete, a later directory-fsync error is a durability warning rather than
    a false pre-publication failure that callers might retry unsafely.
    """

    payload = content.encode("utf-8")
    conditional_existing = expected_identity is not None or expected_sha256 is not None
    if expected_missing and conditional_existing:
        raise ValueError("expected_missing cannot be combined with existing-file expectations")

    with _open_parent_nofollow(path, create_parents=True) as (parent_fd, name, target):
        if conditional_existing:
            current = _open_and_validate_expected(
                parent_fd,
                name,
                target,
                expected_identity=expected_identity,
                expected_sha256=expected_sha256,
            )
        elif expected_missing:
            if _stat_unconditional_target(parent_fd, name, target) is not None:
                raise SafePathError(f"Target appeared before publication: {target}")
            current = None
        else:
            current = _stat_unconditional_target(parent_fd, name, target)

        unconditional_identity = (
            file_identity(current)
            if current is not None and not conditional_existing
            else None
        )

        existing_mode = mode
        if existing_mode is None and current is not None:
            existing_mode = current.st_mode

        temporary_name = f".{name}.tmp-{secrets.token_hex(8)}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = -1
        temporary_exists = False
        exchanged = False
        published = False
        try:
            descriptor = os.open(
                temporary_name,
                flags,
                0o666,
                dir_fd=parent_fd,
            )
            temporary_exists = True
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
                if existing_mode is not None:
                    os.fchmod(descriptor, stat.S_IMODE(existing_mode) & 0o777)
                os.fsync(descriptor)
            new_identity = file_identity(os.fstat(descriptor))
            os.close(descriptor)
            descriptor = -1

            if conditional_existing:
                _renameat2(parent_fd, temporary_name, name, _RENAME_EXCHANGE)
                exchanged = True
                try:
                    _open_and_validate_expected(
                        parent_fd,
                        temporary_name,
                        target,
                        expected_identity=expected_identity,
                        expected_sha256=expected_sha256,
                    )
                except Exception as validation_error:
                    try:
                        _assert_name_identity(parent_fd, name, new_identity, target)
                        _renameat2(parent_fd, temporary_name, name, _RENAME_EXCHANGE)
                        exchanged = False
                    except Exception as rollback_error:
                        raise SafePathError(
                            "Conditional publication validation failed and atomic "
                            f"rollback could not complete; displaced target retained at "
                            f"{temporary_name}"
                        ) from rollback_error
                    raise validation_error

                os.unlink(temporary_name, dir_fd=parent_fd)
                temporary_exists = False
                exchanged = False
                published = True
            elif expected_missing or current is None:
                _renameat2(parent_fd, temporary_name, name, _RENAME_NOREPLACE)
                temporary_exists = False
                published = True
            else:
                assert unconditional_identity is not None
                _renameat2(parent_fd, temporary_name, name, _RENAME_EXCHANGE)
                exchanged = True
                try:
                    _validate_displaced_unconditional_target(
                        parent_fd,
                        temporary_name,
                        target,
                        expected_identity=unconditional_identity,
                    )
                except Exception as validation_error:
                    try:
                        _assert_name_identity(parent_fd, name, new_identity, target)
                        _renameat2(parent_fd, temporary_name, name, _RENAME_EXCHANGE)
                        exchanged = False
                    except Exception as rollback_error:
                        raise SafePathError(
                            "Unconditional overwrite validation failed and atomic "
                            f"rollback could not complete; displaced target retained at "
                            f"{temporary_name}"
                        ) from rollback_error
                    raise validation_error

                os.unlink(temporary_name, dir_fd=parent_fd)
                temporary_exists = False
                exchanged = False
                published = True

            try:
                os.fsync(parent_fd)
            except OSError:
                return False
            return True
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_exists and not exchanged:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            if not published and exchanged:
                # Preserve the displaced inode as a recovery artifact rather
                # than deleting data after an unsuccessful rollback.
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
