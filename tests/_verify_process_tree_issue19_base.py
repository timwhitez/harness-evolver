from __future__ import annotations

import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

import harness.tools.process_runner as process_runner
from harness.tools.process_runner import run_bounded_shell
from harness.tools.verify import VerifyTool
from tests.process_test_support import assert_descendant_lock_released


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group assertion")
def test_verify_timeout_terminates_descendant_process(tmp_path: Path) -> None:
    process_lock = tmp_path / "child.lock"
    script = tmp_path / "spawn_child.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                "child_code = \"import fcntl,pathlib,sys,time; h=pathlib.Path(sys.argv[1]).open('w'); fcntl.flock(h, fcntl.LOCK_EX); h.write('locked'); h.flush()\\nwhile True: time.sleep(60)\"",
                "subprocess.Popen([sys.executable, '-c', child_code, sys.argv[1]])",
                "deadline = time.monotonic() + 5",
                "p = pathlib.Path(sys.argv[1])",
                "while (not p.exists() or p.read_text() != 'locked') and time.monotonic() < deadline: time.sleep(0.01)",
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
            shlex.quote(str(process_lock)),
        ]
    )

    result = VerifyTool(timeout_seconds=0.5).execute(command)

    assert result.success is False
    assert result.metadata["timed_out"] is True
    assert result.metadata["process_tree_terminated"] is True
    assert result.metadata["partial_output_available"] is True
    assert "child-ready" in result.output
    assert "diagnostic-before-timeout" in result.output

    assert_descendant_lock_released(process_lock)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group assertion")
def test_timeout_kills_descendant_after_group_leader_exits_on_sigterm(tmp_path: Path) -> None:
    process_lock = tmp_path / "resistant-child.lock"
    child_code = (
        "import fcntl,pathlib,signal,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "h=pathlib.Path(sys.argv[1]).open('w'); fcntl.flock(h, fcntl.LOCK_EX); "
        "h.write('locked'); h.flush()\n"
        "while True: time.sleep(60)"
    )
    script = tmp_path / "spawn_resistant_child.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}, sys.argv[1]])",
                "deadline = time.monotonic() + 5",
                "p = pathlib.Path(sys.argv[1])",
                "while (not p.exists() or p.read_text() != 'locked') and time.monotonic() < deadline: time.sleep(0.01)",
                "print('resistant-child-ready', flush=True)",
                "time.sleep(60)",
            ]
        ),
        encoding="utf-8",
    )
    command = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote(str(script)),
            shlex.quote(str(process_lock)),
        ]
    )

    started = time.monotonic()
    result = run_bounded_shell(command, timeout_seconds=0.5)

    assert result.timed_out is True
    assert result.managed_process_group_terminated is True
    assert "resistant-child-ready" in result.stdout
    assert time.monotonic() - started < 3.5
    assert_descendant_lock_released(process_lock)


@pytest.mark.skipif(
    sys.platform == "win32" or not Path("/proc/self/stat").is_file(),
    reason="Linux /proc descendant discovery assertion",
)
def test_timeout_kills_descendant_that_escapes_with_setsid(tmp_path: Path) -> None:
    process_lock = tmp_path / "setsid-child.lock"
    child_code = "\n".join(
        [
            "import fcntl, os, pathlib, signal, sys, time",
            "os.setsid()",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "h = pathlib.Path(sys.argv[1]).open('w')",
            "fcntl.flock(h, fcntl.LOCK_EX); h.write('locked'); h.flush()",
            "print('setsid-child-ready', flush=True)",
            "while True: time.sleep(60)",
        ]
    )
    script = tmp_path / "spawn_setsid_child.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}, sys.argv[1]])",
                "deadline = time.monotonic() + 5",
                "p = pathlib.Path(sys.argv[1])",
                "while (not p.exists() or p.read_text() != 'locked') and time.monotonic() < deadline: time.sleep(0.01)",
                "print('parent-observed-setsid-child', flush=True)",
                "time.sleep(60)",
            ]
        ),
        encoding="utf-8",
    )
    command = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote(str(script)),
            shlex.quote(str(process_lock)),
        ]
    )

    result = run_bounded_shell(command, timeout_seconds=0.75)

    assert result.timed_out is True
    assert result.managed_process_group_terminated is True
    assert "parent-observed-setsid-child" in result.stdout
    assert_descendant_lock_released(process_lock)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group assertion")
def test_normal_parent_exit_does_not_leave_pipe_holding_background_child(
    tmp_path: Path,
) -> None:
    process_lock = tmp_path / "background.lock"
    script = tmp_path / "spawn_and_exit.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                "child_code = \"import fcntl,pathlib,sys,time; h=pathlib.Path(sys.argv[1]).open('w'); fcntl.flock(h, fcntl.LOCK_EX); h.write('locked'); h.flush()\\nwhile True: time.sleep(60)\"",
                "subprocess.Popen([sys.executable, '-c', child_code, sys.argv[1]])",
                "deadline = time.monotonic() + 5",
                "p = pathlib.Path(sys.argv[1])",
                "while (not p.exists() or p.read_text() != 'locked') and time.monotonic() < deadline: time.sleep(0.01)",
                "print('parent-exiting', flush=True)",
            ]
        ),
        encoding="utf-8",
    )
    command = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote(str(script)),
            shlex.quote(str(process_lock)),
        ]
    )

    result = run_bounded_shell(command, timeout_seconds=5.0)

    assert result.timed_out is False
    assert result.managed_process_group_terminated is True
    assert "parent-exiting" in result.stdout
    assert_descendant_lock_released(process_lock)


def test_windows_timeout_always_invokes_taskkill_for_the_tree(monkeypatch) -> None:
    class FakeProcess:
        pid = 4242
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 1
            return 1

        def kill(self):
            raise AssertionError("root-only kill should not be needed")

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(process_runner.os, "name", "nt")
    monkeypatch.setattr(process_runner.subprocess, "run", fake_run)

    terminated = process_runner._terminate_process_tree(FakeProcess())  # type: ignore[arg-type]

    assert terminated is True
    assert calls == [["taskkill", "/PID", "4242", "/T", "/F"]]


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


def test_tiny_output_limit_never_returns_an_oversized_marker() -> None:
    program = "print('abcdef')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"

    result = run_bounded_shell(command, timeout_seconds=5.0, output_limit_bytes=1)

    assert len(result.stdout.encode("utf-8")) <= 1


@pytest.mark.parametrize("timeout", [0.0, -1.0, math.nan, math.inf])
def test_non_finite_or_non_positive_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        run_bounded_shell("echo ok", timeout_seconds=timeout)


def test_verify_reports_invalid_timeout_as_a_tool_failure() -> None:
    result = VerifyTool().execute("echo ok", timeout=0)

    assert result.success is False
    assert "timeout_seconds must be finite and > 0" in result.error
