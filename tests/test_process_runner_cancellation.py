from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

import pytest

from harness.tools.process_runner import (
    _supervisor_exit_confirms_cleanup,
    run_bounded_shell,
)
from tests.process_test_support import assert_descendant_lock_released


def test_descendant_lock_assertion_rejects_a_live_owner(tmp_path: Path) -> None:
    process_lock = tmp_path / "live-owner.lock"
    child_code = (
        "import fcntl,pathlib,sys,time; "
        "h=pathlib.Path(sys.argv[1]).open('w'); "
        "fcntl.flock(h, fcntl.LOCK_EX); h.write('locked'); h.flush(); "
        "time.sleep(60)"
    )
    child = subprocess.Popen([sys.executable, "-c", child_code, str(process_lock)])
    try:
        deadline = time.monotonic() + 3
        while (
            (not process_lock.exists() or not process_lock.read_text())
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        with pytest.raises(AssertionError, match="still owns"):
            assert_descendant_lock_released(process_lock, timeout=0.2)
    finally:
        child.kill()
        child.wait(timeout=3)


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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descendant cancellation assertion")
def test_cancellation_terminates_and_reaps_descendant_tree(tmp_path: Path) -> None:
    process_lock = tmp_path / "child.lock"
    script = tmp_path / "spawn_child.py"
    script.write_text(
        "\n".join(
            [
                "import subprocess, sys, time",
                "child_code = \"import fcntl,pathlib,sys,time; h=pathlib.Path(sys.argv[1]).open('w'); fcntl.flock(h, fcntl.LOCK_EX); h.write('locked'); h.flush()\\nwhile True: time.sleep(60)\"",
                "subprocess.Popen([sys.executable, '-c', child_code, sys.argv[1]])",
                "deadline = time.monotonic() + 5",
                "p = __import__('pathlib').Path(sys.argv[1])",
                "while (not p.exists() or p.read_text() != 'locked') and time.monotonic() < deadline: time.sleep(0.01)",
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
            shlex.quote(str(process_lock)),
        ]
    )
    cancel = threading.Event()

    def request_cancel() -> None:
        deadline = time.monotonic() + 3
        while (
            (not process_lock.exists() or process_lock.read_text() != "locked")
            and time.monotonic() < deadline
        ):
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
    assert_descendant_lock_released(process_lock)
