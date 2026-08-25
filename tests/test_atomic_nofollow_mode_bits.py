from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from harness.tools.safe_path_io import atomic_write_text_nofollow, edit_text_nofollow


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor-relative no-follow I/O is POSIX-only",
)


def test_pure_replacement_preserves_only_ordinary_permission_bits(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o4755)

    atomic_write_text_nofollow(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_edit_does_not_recreate_setgid_or_sticky_bits(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o3750)

    edit_text_nofollow(target, lambda text: text.replace("old", "new"))

    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o750
