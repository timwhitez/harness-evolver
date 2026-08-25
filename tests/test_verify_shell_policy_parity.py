from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys
import time

import pytest

from harness.tools.shell import ShellTool
from harness.tools.verify import VerifyTool


@pytest.mark.parametrize(
    "command",
    [
        "cat /tests/test.sh",
        "codex exec 'do work'",
        "pip install example-package &",
        "find / -type f",
    ],
)
def test_verify_returns_the_same_policy_block_as_shell(command: str) -> None:
    shell_result = ShellTool().execute(command=command)
    verify_result = VerifyTool().execute(command=command)

    assert shell_result.success is False
    assert verify_result.success is False
    assert shell_result.metadata.get("blocked_by")
    assert verify_result.metadata == shell_result.metadata
    assert verify_result.output == shell_result.output
    assert verify_result.error == shell_result.error


def test_verify_keeps_allowed_shell_execution_behavior() -> None:
    shell_result = ShellTool().execute(command="printf 'ok'")
    verify_result = VerifyTool().execute(command="printf 'ok'")

    assert shell_result.success is True
    assert verify_result.success is True
    assert verify_result.output == shell_result.output == "ok"
    assert verify_result.metadata["exit_code"] == shell_result.metadata["exit_code"] == 0


def test_verify_adds_verifier_specific_semantic_failure_after_shell_authorization() -> None:
    command = "printf 'Only achieved 10%% (need 90%%+)\\n'"

    shell_result = ShellTool().execute(command=command)
    verify_result = VerifyTool().execute(command=command)

    assert shell_result.success is True
    assert verify_result.success is False
    assert verify_result.metadata["semantic_failure_detected"] is True
    assert verify_result.metadata["semantic_failure_kind"] == "verification_threshold_failure"
    assert "unmet threshold" in verify_result.error


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


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not Path("/proc/self/stat").is_file(),
    reason="Linux subreaper descendant assertion",
)
def test_shell_timeout_preserves_issue19_cleanup_after_child_clears_environment(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "setsid-child.pid"
    child_code = "\n".join(
        [
            "import os, pathlib, signal, sys, time",
            # Clearing inherited variables proves cleanup does not depend on the
            # earlier environment-token implementation.
            "os.environ.clear()",
            "os.setsid()",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')",
            "time.sleep(60)",
        ]
    )
    script = tmp_path / "spawn_setsid_child.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}, sys.argv[1]])",
                "deadline = time.monotonic() + 5",
                "while not pathlib.Path(sys.argv[1]).exists() and time.monotonic() < deadline: time.sleep(0.01)",
                "print('setsid-child-ready', flush=True)",
                "time.sleep(60)",
            ]
        ),
        encoding="utf-8",
    )
    command = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote(str(script)),
            shlex.quote(str(pid_file)),
        ]
    )

    result = ShellTool(timeout_seconds=0.75).execute(command=command)

    assert result.success is False
    assert result.metadata["timed_out"] is True
    assert "setsid-child-ready" in result.output
    _wait_for_pid_exit(int(pid_file.read_text(encoding="utf-8")))
