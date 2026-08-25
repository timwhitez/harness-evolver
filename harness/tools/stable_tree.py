"""Stable descriptor-relative traversal for authorized local search roots."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import stat
from typing import BinaryIO, Iterator

from harness.tools.safe_path_io import (
    SafePathError,
    _open_parent_nofollow,
    _validate_unique_regular_file,
)


class StableTreeError(SafePathError):
    """Raised when a search tree changes identity or contains an unsafe entry."""


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _directory_flags() -> int:
    required = ("O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW")
    if os.name != "posix" or any(not hasattr(os, name) for name in required):
        raise StableTreeError(
            "Stable descriptor-relative directory traversal is unavailable"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


@contextmanager
def _open_stable_directory(path: Path) -> Iterator[int]:
    expected = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(expected.st_mode):
        raise StableTreeError(f"Search root is not a directory: {path}")

    with _open_parent_nofollow(path, create_parents=False) as (
        parent_fd,
        name,
        target,
    ):
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
        try:
            actual = os.fstat(descriptor)
            if not stat.S_ISDIR(actual.st_mode) or not _same_inode(expected, actual):
                raise StableTreeError(
                    f"Search root changed identity before traversal: {target}"
                )
            yield descriptor
        finally:
            os.close(descriptor)


def _walk_directory(
    directory_fd: int,
    relative: Path,
) -> Iterator[tuple[Path, BinaryIO, os.stat_result]]:
    with os.scandir(directory_fd) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)

    for entry in entries:
        child_relative = relative / entry.name
        try:
            observed = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise StableTreeError(
                f"Cannot inspect search entry safely: {child_relative}"
            ) from exc

        if stat.S_ISLNK(observed.st_mode):
            raise StableTreeError(
                f"Search tree contains a symlink: {child_relative}"
            )

        if stat.S_ISDIR(observed.st_mode):
            try:
                child_fd = os.open(
                    entry.name,
                    _directory_flags(),
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise StableTreeError(
                    f"Cannot open search directory without following aliases: "
                    f"{child_relative}"
                ) from exc
            try:
                actual = os.fstat(child_fd)
                if not stat.S_ISDIR(actual.st_mode) or not _same_inode(
                    observed,
                    actual,
                ):
                    raise StableTreeError(
                        f"Search directory changed identity: {child_relative}"
                    )
                yield from _walk_directory(child_fd, child_relative)
            finally:
                os.close(child_fd)
            continue

        if not stat.S_ISREG(observed.st_mode):
            raise StableTreeError(
                f"Search tree contains a non-regular entry: {child_relative}"
            )

        try:
            file_fd = os.open(
                entry.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise StableTreeError(
                f"Cannot open search file without following aliases: {child_relative}"
            ) from exc
        try:
            actual = os.fstat(file_fd)
            _validate_unique_regular_file(actual, child_relative)
            if not _same_inode(observed, actual):
                raise StableTreeError(
                    f"Search file changed identity: {child_relative}"
                )
            with os.fdopen(file_fd, "rb", closefd=False) as stream:
                yield child_relative, stream, actual
        finally:
            os.close(file_fd)


def iter_stable_regular_files(
    root: str | os.PathLike[str],
) -> Iterator[tuple[Path, BinaryIO, os.stat_result]]:
    """Yield regular files anchored to the directory inode first authorized."""

    root_path = Path(root)
    observed = os.stat(root_path, follow_symlinks=False)
    if stat.S_ISREG(observed.st_mode):
        with _open_parent_nofollow(root_path, create_parents=False) as (
            parent_fd,
            name,
            target,
        ):
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            try:
                actual = os.fstat(descriptor)
                _validate_unique_regular_file(actual, target)
                if not _same_inode(observed, actual):
                    raise StableTreeError(
                        f"Search file changed identity before reading: {target}"
                    )
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    yield Path(), stream, actual
            finally:
                os.close(descriptor)
        return

    if not stat.S_ISDIR(observed.st_mode):
        raise StableTreeError(f"Search root is not a regular file or directory: {root_path}")

    with _open_stable_directory(root_path) as root_fd:
        yield from _walk_directory(root_fd, Path())
