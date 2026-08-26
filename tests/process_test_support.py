"""Host-observable assertions for commands running in private PID namespaces."""

from __future__ import annotations

import fcntl
from pathlib import Path
import time


def assert_descendant_lock_released(path: Path, timeout: float = 3.0) -> None:
    """Assert the kernel released a descendant-owned advisory file lock."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ready = path.read_text(encoding="utf-8") == "locked"
        except OSError:
            ready = False
        if ready:
            break
        time.sleep(0.02)
    else:
        raise AssertionError(f"descendant never confirmed lock ownership: {path}")

    with path.open("a+", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(0.02)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
                return
    raise AssertionError(f"descendant still owns its process-lifetime lock: {path}")
