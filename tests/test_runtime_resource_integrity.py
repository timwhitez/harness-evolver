from __future__ import annotations

from importlib import resources
import os
from pathlib import Path

import pytest

from bench.runtime_resources import materialize_runtime_resources


def _packaged_models_bytes() -> bytes:
    return resources.files("bench.resources").joinpath(
        "config",
        "models.yaml",
    ).read_bytes()


def test_modified_content_addressed_cache_is_repaired(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    root = materialize_runtime_resources(cache)
    models = root / "config" / "models.yaml"
    models.write_text("roles: {tampered: true}\n", encoding="utf-8")

    repaired = materialize_runtime_resources(cache)

    assert repaired == root
    assert models.read_bytes() == _packaged_models_bytes()


@pytest.mark.skipif(os.name == "nt", reason="symlink cache fixture is POSIX-specific")
def test_symlinked_cached_resource_is_replaced_with_regular_package_bytes(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    root = materialize_runtime_resources(cache)
    models = root / "config" / "models.yaml"
    outside = tmp_path / "outside-models.yaml"
    outside.write_text("roles: {redirected: true}\n", encoding="utf-8")
    models.unlink()
    models.symlink_to(outside)

    repaired = materialize_runtime_resources(cache)

    assert repaired == root
    assert models.is_file()
    assert not models.is_symlink()
    assert models.read_bytes() == _packaged_models_bytes()
    assert outside.read_text(encoding="utf-8") == "roles: {redirected: true}\n"


def test_untracked_cargo_build_outputs_do_not_invalidate_resource_cache(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    root = materialize_runtime_resources(cache)
    artifact = root / "hl-worker-core" / "target" / "debug" / "artifact"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"derived")

    reused = materialize_runtime_resources(cache)

    assert reused == root
    assert artifact.read_bytes() == b"derived"
