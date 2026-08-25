"""Materialize bundled Worker and configuration resources into a writable cache."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import lru_cache
from hashlib import sha256
from importlib import resources
from importlib.resources.abc import Traversable
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time


_RESOURCE_PACKAGE = "bench.resources"
_RESOURCE_MARKER = ".harness-evolver-resources"
_WORKER_DIRECTORY = "hl-worker-core"
_CONFIG_DIRECTORY = "config"
_CONFIG_NAMES = frozenset({"benchmark.yaml", "default.yaml", "models.yaml", "trials.yaml"})
_LOCK_WAIT_SECONDS = 30.0
_STALE_LOCK_SECONDS = 300.0
_REQUIRED_WORKER_FILES = (
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("src/main.rs"),
)


def _resource_root() -> Traversable:
    return resources.files(_RESOURCE_PACKAGE)


def _resource_files() -> Iterable[tuple[Path, Traversable]]:
    """Yield every bundled file with a stable relative path."""

    def walk(node: Traversable, relative: Path) -> Iterator[tuple[Path, Traversable]]:
        for child in sorted(node.iterdir(), key=lambda item: item.name):
            child_relative = relative / child.name
            if child.is_dir():
                yield from walk(child, child_relative)
            elif child.is_file():
                yield child_relative, child

    yield from walk(_resource_root(), Path())


def _update_content_digest(
    digest: object,
    relative: Path,
    payload: bytes,
) -> None:
    digest.update(relative.as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


@lru_cache(maxsize=1)
def runtime_resource_digest() -> str:
    """Return a content address for the complete bundled runtime resource set."""

    digest = sha256()
    for relative, resource in _resource_files():
        if "__pycache__" in relative.parts:
            continue
        _update_content_digest(digest, relative, resource.read_bytes())
    return digest.hexdigest()


def default_runtime_cache_root() -> Path:
    """Return a user-writable cache location without depending on the process CWD."""

    explicit = os.environ.get("HL_RUNTIME_CACHE_DIR") or os.environ.get(
        "HL_WORKER_CACHE_DIR"
    )
    if explicit:
        return Path(explicit).expanduser()

    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"]).expanduser()
    else:
        try:
            base = Path.home() / ".cache"
        except (RuntimeError, OSError):
            base = Path(tempfile.gettempdir())
    return base / "harness-evolver" / "runtime"


def _reject_symlinked_cache_path(path: Path) -> None:
    """Reject every existing symlink component in the writable cache path."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor) if absolute.anchor else Path()
    components = absolute.parts[1:] if absolute.anchor else absolute.parts
    for component in components:
        current = current / component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(
                f"Cannot inspect runtime cache path safely: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(
                f"Runtime cache path contains a symlink: {current}"
            )


def materialize_runtime_resources(
    cache_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Copy package resources to a verified content-addressed runtime directory."""

    digest = runtime_resource_digest()
    root = (
        Path(cache_root).expanduser()
        if cache_root is not None
        else default_runtime_cache_root()
    )
    _reject_symlinked_cache_path(root)
    destination = root / digest
    if _materialization_is_complete(destination, digest):
        return destination

    root.mkdir(parents=True, exist_ok=True)
    # Re-check after creation: an existing parent or the requested root itself
    # must never redirect writes into another filesystem location.
    _reject_symlinked_cache_path(root)
    lock = root / f".{digest}.lock"
    if not _acquire_materialization_lock(lock, destination, digest):
        return destination

    temporary: Path | None = None
    try:
        if _materialization_is_complete(destination, digest):
            return destination
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or destination.is_file():
                destination.unlink()
            else:
                shutil.rmtree(destination)

        temporary = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=root))
        for relative, resource in _resource_files():
            if "__pycache__" in relative.parts:
                continue
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(resource.read_bytes())
        (temporary / _RESOURCE_MARKER).write_text(digest + "\n", encoding="ascii")
        _validate_materialized_tree(temporary)
        if _materialized_resource_digest(temporary) != digest:
            raise RuntimeError("Materialized runtime resource digest does not match package data")
        temporary.rename(destination)
        temporary = None
        return destination
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(lock, ignore_errors=True)


def worker_crate_root(cache_root: str | os.PathLike[str] | None = None) -> Path:
    """Return the writable packaged Rust Worker crate root."""

    root = materialize_runtime_resources(cache_root)
    crate = root / _WORKER_DIRECTORY
    _validate_worker_crate(crate)
    return crate


def bundled_config_path(
    name: str,
    cache_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a persistent path to one known bundled YAML configuration file."""

    if name not in _CONFIG_NAMES:
        raise ValueError(f"Unknown bundled configuration file: {name!r}")
    path = materialize_runtime_resources(cache_root) / _CONFIG_DIRECTORY / name
    if not path.is_file():
        raise RuntimeError(f"Bundled configuration resource is missing: {name}")
    return path


def _acquire_materialization_lock(
    lock: Path,
    destination: Path,
    digest: str,
) -> bool:
    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    while True:
        if _materialization_is_complete(destination, digest):
            return False
        try:
            lock.mkdir()
            return True
        except FileExistsError:
            try:
                stale = time.time() - lock.stat().st_mtime > _STALE_LOCK_SECONDS
            except FileNotFoundError:
                continue
            if stale:
                shutil.rmtree(lock, ignore_errors=True)
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for runtime resource materialization lock: {lock}"
                )
            time.sleep(0.05)


def _materialization_is_complete(destination: Path, digest: str) -> bool:
    if destination.is_symlink() or not destination.is_dir():
        return False
    marker = destination / _RESOURCE_MARKER
    try:
        if marker.read_text(encoding="ascii").strip() != digest:
            return False
        _validate_materialized_tree(destination)
        return _materialized_resource_digest(destination) == digest
    except (OSError, RuntimeError):
        return False


def _materialized_resource_digest(root: Path) -> str:
    """Hash only declared package resources, ignoring Cargo target/build outputs."""

    digest = sha256()
    for relative, _resource in _resource_files():
        if "__pycache__" in relative.parts:
            continue
        payload = (root / relative).read_bytes()
        _update_content_digest(digest, relative, payload)
    return digest.hexdigest()


def _validate_materialized_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("Materialized runtime root is not a regular directory")

    for relative, _resource in _resource_files():
        if "__pycache__" in relative.parts:
            continue
        candidate = root
        for component in relative.parts:
            candidate = candidate / component
            if candidate.is_symlink():
                raise RuntimeError(
                    f"Materialized runtime resource contains a symlink: {relative}"
                )
        if not candidate.is_file():
            raise RuntimeError(
                f"Materialized runtime resource is missing or not a file: {relative}"
            )

    _validate_worker_crate(root / _WORKER_DIRECTORY)
    for name in _CONFIG_NAMES:
        if not (root / _CONFIG_DIRECTORY / name).is_file():
            raise RuntimeError(f"Bundled configuration resource is missing: {name}")


def _validate_worker_crate(crate: Path) -> None:
    missing = [
        str(relative)
        for relative in _REQUIRED_WORKER_FILES
        if not (crate / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Bundled Rust Worker crate is incomplete; missing: " + ", ".join(missing)
        )
