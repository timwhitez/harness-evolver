"""Bounded subprocess execution with process-tree cleanup."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import IO


_OUTPUT_LIMIT_BYTES = 1_000_000
_TERMINATION_GRACE_SECONDS = 1.0


@dataclass(frozen=True)
class ProcessOutcome:
    """Result of one bounded shell process."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_ms: float


def run_bounded_shell(
    command: str,
    *,
    timeout_seconds: float,
    cwd: str | Path | None = None,
    output_limit_bytes: int = _OUTPUT_LIMIT_BYTES,
) -> ProcessOutcome:
    """Run a shell command and terminate its complete process tree on timeout.

    Output is redirected to temporary files instead of pipes so a descendant
    cannot deadlock timeout cleanup by retaining or filling inherited pipe
    descriptors. Returned stdout and stderr are independently bounded.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if output_limit_bytes < 1:
        raise ValueError("output_limit_bytes must be >= 1")

    popen_kwargs: dict[str, object] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=stdout_file,
            stderr=stderr_file,
            cwd=cwd,
            **popen_kwargs,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process)

        elapsed_ms = (time.monotonic() - started) * 1000
        stdout = _read_bounded_output(stdout_file, output_limit_bytes)
        stderr = _read_bounded_output(stderr_file, output_limit_bytes)
        return ProcessOutcome(
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            elapsed_ms=elapsed_ms,
        )


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate and reap a shell process plus every descendant."""

    if process.poll() is not None:
        return

    if os.name == "posix":
        _signal_posix_group(process.pid, signal.SIGTERM)
        if _wait_for_exit(process, _TERMINATION_GRACE_SECONDS):
            return
        _signal_posix_group(process.pid, signal.SIGKILL)
        _wait_for_exit(process, _TERMINATION_GRACE_SECONDS)
        return

    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError):
            pass
        if _wait_for_exit(process, _TERMINATION_GRACE_SECONDS):
            return
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        _wait_for_exit(process, _TERMINATION_GRACE_SECONDS)
        return

    process.terminate()
    if not _wait_for_exit(process, _TERMINATION_GRACE_SECONDS):
        process.kill()
        _wait_for_exit(process, _TERMINATION_GRACE_SECONDS)


def _signal_posix_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return


def _wait_for_exit(process: subprocess.Popen[bytes], timeout: float) -> bool:
    try:
        process.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def _read_bounded_output(stream: IO[bytes], limit_bytes: int) -> str:
    stream.flush()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(0)
    if size <= limit_bytes:
        payload = stream.read()
    else:
        marker = f"\n... [{size - limit_bytes} output bytes omitted] ...\n".encode()
        retained = max(0, limit_bytes - len(marker))
        head_size = retained // 2
        tail_size = retained - head_size
        head = stream.read(head_size)
        stream.seek(-tail_size, os.SEEK_END)
        tail = stream.read(tail_size)
        payload = head + marker + tail
    return payload.decode("utf-8", errors="replace")
