from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import sys
import time

import pytest

from harness.tools.process_runner import run_bounded_shell
from harness.tools.verify import VerifyTool


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group assertion")
def test_verify_timeout_terminates_descendant_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "spawn_child.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')",
                "print('child-ready', flush=True)",
                "print('diagnostic-before-timeout', file=sys.stderr, flush=True)",
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

    result = VerifyTool(timeout_seconds=0.5).execute(command)

    assert result.success is False
    assert result.metadata["timed_out"] is True
    assert result.metadata["process_tree_terminated"] is True
    assert result.metadata["partial_output_available"] is True
    assert "child-ready" in result.output
    assert "diagnostic-before-timeout" in result.output

    child_pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3.0
    while _pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pid_exists(child_pid)


def test_process_runner_bounds_stdout_and_stderr() -> None:
    payload_size = 20_000
    program = (
        "import sys; "
        f"sys.stdout.write('o' * {payload_size}); "
        f"sys.stderr.write('e' * {payload_size})"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"

    result = run_bounded_shell(
        command,
        timeout_seconds=5.0,
        output_limit_bytes=4096,
    )

    assert result.timed_out is False
    assert result.returncode == 0
    assert len(result.stdout.encode("utf-8")) <= 4096
    assert len(result.stderr.encode("utf-8")) <= 4096
    assert "output bytes omitted" in result.stdout
    assert "output bytes omitted" in result.stderr


def test_non_positive_timeout_is_rejected() -> None:
    result = VerifyTool().execute("echo ok", timeout=0)

    assert result.success is False
    assert "timeout_seconds must be > 0" in result.error
