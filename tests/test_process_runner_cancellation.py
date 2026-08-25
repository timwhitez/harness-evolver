from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys
import threading
import time

import pytest

from harness.tools.process_runner import (
    _supervisor_exit_confirms_cleanup,
    run_bounded_shell,
)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_exit(pid: int, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while _pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_exists(pid)


def test_only_preliminary_supervisor_timeout_status_is_accepted() -> None:
    assert _supervisor_exit_confirms_cleanup(124) is True
    for returncode in (None, 0, 1, 125, 126, 137, 143, 255, -9, -15):
        assert _supervisor_exit_confirms_cleanup(returncode) is False


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="cleanup proof is attached to the Linux supervisor",
)
def test_managed_command_normal_exit_124_is_not_cleanup_attestation() -> None:
    result = run_bounded_shell("exit 124", timeout_seconds=5.0)

    assert result.timed_out is False
    assert result.cancelled is False
    assert result.returncode == 124
    assert result.managed_process_group_terminated is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX descendant cancellation assertion")
def test_cancellation_terminates_and_reaps_descendant_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "spawn_child.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')",
                "print('child-ready', flush=True)",
                "time.sleep(60)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    command = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote(str(script)),
            shlex.quote(str(pid_file)),
        ]
    )
    cancel = threading.Event()

    def request_cancel() -> None:
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        cancel.set()

    trigger = threading.Thread(target=request_cancel)
    trigger.start()
    result = run_bounded_shell(
        command,
        timeout_seconds=10.0,
        cancel_event=cancel,
    )
    trigger.join(timeout=1)

    assert result.cancelled is True
    assert result.timed_out is False
    assert result.managed_process_group_terminated is True
    assert "child-ready" in result.stdout
    _wait_for_pid_exit(int(pid_file.read_text(encoding="utf-8")))
