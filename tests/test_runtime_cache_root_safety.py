from __future__ import annotations

import os
from pathlib import Path

import pytest

from bench.runtime_resources import materialize_runtime_resources


pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="symlinked cache-root validation is POSIX-specific",
)


def test_symlinked_cache_root_is_rejected_before_any_resource_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    cache = tmp_path / "cache"
    cache.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="cache path contains a symlink"):
        materialize_runtime_resources(cache)

    assert list(outside.iterdir()) == []


def test_symlinked_cache_parent_is_rejected_before_directory_creation(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "redirect"
    parent.symlink_to(outside, target_is_directory=True)
    cache = parent / "nested-cache"

    with pytest.raises(RuntimeError, match="cache path contains a symlink"):
        materialize_runtime_resources(cache)

    assert not (outside / "nested-cache").exists()
