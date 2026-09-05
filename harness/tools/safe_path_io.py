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

from harness.tools.descriptor_open import open_readonly_checked
from harness.tools.publication import publish_bytes


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
    try:
        descriptor, _ = open_readonly_checked(parent_fd, name)
    except ValueError as exc:
        raise SafePathError(f"Cannot open regular target safely: {target}: {exc}") from exc
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
    actual = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISREG(actual.st_mode) or file_identity(actual) != expected:
        raise SafePathError(
            f"Published target changed before rollback could complete: {target}"
        )


def read_text_nofollow(
    path: str | os.PathLike[str],
    *,
    errors: str = "replace",
) -> tuple[str, os.stat_result]:
    """Read one uniquely linked regular file through stable descriptors."""

    with _open_parent_nofollow(path, create_parents=False) as (parent_fd, name, target):
        try:
            descriptor, _ = open_readonly_checked(parent_fd, name)
        except ValueError as exc:
            raise SafePathError(f"Cannot open regular target safely: {target}: {exc}") from exc
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


def publish_text_nofollow(
    path: str | os.PathLike[str],
    content: str,
    *,
    mode: int | None = None,
    expected_identity: FileIdentity | None = None,
    expected_sha256: str | None = None,
    expected_missing: bool = False,
) -> dict:
    """Return an explicit publication outcome; see docs/file_publication.md.

    Cooperating writers serialize. External post-exchange races are
    indeterminate and retain recovery data, never a destructive rollback.
    """
    payload = content.encode("utf-8")
    outcome = None
    try:
        with _open_parent_nofollow(path, create_parents=True) as (parent_fd, name, target):
            outcome = publish_bytes(
                parent_fd, name, payload, _renameat2, mode=mode,
                expected_identity=expected_identity, expected_sha256=expected_sha256,
                expected_missing=expected_missing,
            )
            outcome["recovery_directory"] = str(target.parent)
    except OSError as exc:
        if outcome is None:
            raise
        # Even closing the parent descriptor is post-publication housekeeping.
        # Preserve the known publication state if that final close reports an error.
        outcome["cleanup_warning"] = True
        outcome["publication_error"] = str(exc)
        outcome["no_auto_retry"] = True
    return outcome


def atomic_write_text_nofollow(
    path: str | os.PathLike[str],
    content: str,
    *,
    mode: int | None = None,
    expected_identity: FileIdentity | None = None,
    expected_sha256: str | None = None,
    expected_missing: bool = False,
) -> bool:
    """Compatibility API: bool durability on success, structured error otherwise.

    Public tools use publish_text_nofollow to expose cleanup/recovery metadata.
    A legacy caller never receives a false uncommitted error after a confirmed
    publication; retained cleanup artifacts are also reported as a warning.
    """
    outcome = publish_text_nofollow(
        path, content, mode=mode, expected_identity=expected_identity,
        expected_sha256=expected_sha256, expected_missing=expected_missing,
    )
    if outcome["atomic_replace"] is not True:
        error = SafePathError(outcome["publication_error"] or "publication indeterminate")
        error.publication_outcome = outcome
        raise error
    if outcome["cleanup_warning"]:
        import warnings
        warnings.warn("Content published; cleanup/reconciliation is required: " +
                      repr(outcome["recovery_entries"]), RuntimeWarning, stacklevel=2)
    return bool(outcome["directory_fsync"])


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
