from __future__ import annotations

import base64
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from bench import _canonical_harbor_hardlink as hardlink


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="secure Harbor descriptor I/O is POSIX-only",
)


def _run(script: str, target: Path, payload: bytes) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HL_FILE_PATH": str(target),
            "HL_FILE_CONTENT": base64.b64encode(payload).decode("ascii"),
        },
        check=False,
    )


def test_secure_harbor_overwrite_clears_special_mode_bits(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o4755)

    completed = _run(hardlink._v3._SECURE_ATOMIC_WRITE, target, b"new")

    assert completed.returncode == 0, completed.stderr
    assert target.read_bytes() == b"new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_secure_harbor_new_file_uses_process_umask(tmp_path: Path) -> None:
    target = tmp_path / "created.txt"
    program = "import os\nos.umask(0o027)\n" + hardlink._v3._SECURE_ATOMIC_WRITE

    completed = _run(program, target, b"created")

    assert completed.returncode == 0, completed.stderr
    assert target.read_bytes() == b"created"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_secure_harbor_fsync_failure_keeps_old_target(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    failing = hardlink._v3._SECURE_ATOMIC_WRITE.replace(
        "            os.fsync(descriptor)",
        "            raise OSError('injected fsync failure')",
        1,
    )
    assert failing != hardlink._v3._SECURE_ATOMIC_WRITE

    completed = _run(failing, target, b"new")

    assert completed.returncode != 0
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".target.txt.tmp-*")) == []
