"""Bounded subprocess execution with managed process-group cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import BinaryIO


_OUTPUT_LIMIT_BYTES = 1_000_000
_TERMINATION_GRACE_SECONDS = 1.0
_TERMINATION_POLL_SECONDS = 0.02
_STREAM_JOIN_SECONDS = 1.0
_READ_CHUNK_BYTES = 65_536


@dataclass(frozen=True)
class ProcessOutcome:
    """Result of one bounded shell process."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_ms: float
    managed_process_group_terminated: bool = False


@dataclass
class _BoundedCapture:
    """Retain a bounded head/tail while continuously draining one pipe."""

    limit_bytes: int
    total_bytes: int = 0
    head: bytearray = field(default_factory=bytearray)
    tail: bytearray = field(default_factory=bytearray)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def feed(self, payload: bytes) -> None:
        if not payload:
            return
        with self._lock:
            self.total_bytes += len(payload)
            remaining = payload
            if len(self.head) < self.limit_bytes:
                take = min(self.limit_bytes - len(self.head), len(remaining))
                self.head.extend(remaining[:take])
                remaining = remaining[take:]
            if remaining:
                self.tail.extend(remaining)
                if len(self.tail) > self.limit_bytes:
                    del self.tail[: len(self.tail) - self.limit_bytes]

    def text(self) -> str:
        with self._lock:
            total_bytes = self.total_bytes
            head = bytes(self.head)
            tail = bytes(self.tail)
        if total_bytes <= self.limit_bytes:
            payload = head[:total_bytes]
        else:
            marker = (
                f"\n... [{total_bytes - self.limit_bytes} output bytes omitted] ...\n"
            ).encode("ascii")
            if len(marker) >= self.limit_bytes:
                payload = marker[: self.limit_bytes]
            else:
                retained = self.limit_bytes - len(marker)
                head_size = retained // 2
                tail_size = retained - head_size
                payload = (
                    head[:head_size]
                    + marker
                    + (tail[-tail_size:] if tail_size else b"")
                )
        return payload.decode("utf-8", errors="replace")


def run_bounded_shell(
    command: str,
    *,
    timeout_seconds: float,
    cwd: str | Path | None = None,
    output_limit_bytes: int = _OUTPUT_LIMIT_BYTES,
) -> ProcessOutcome:
    """Run a shell command and clean up its managed process group on timeout.

    Stdout and stderr are drained concurrently into bounded in-memory head/tail
    buffers. This avoids pipe backpressure deadlocks without an unbounded disk
    spool. POSIX descendants that remain in the session's process group are
    terminated with TERM then KILL. Windows uses ``taskkill /T /F`` for the
    process tree rooted at the shell PID.
    """

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and > 0")
    if output_limit_bytes < 1:
        raise ValueError("output_limit_bytes must be >= 1")

    popen_kwargs: dict[str, object] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        **popen_kwargs,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_capture = _BoundedCapture(output_limit_bytes)
    stderr_capture = _BoundedCapture(output_limit_bytes)
    stdout_thread = _start_drain_thread(process.stdout, stdout_capture, "stdout")
    stderr_thread = _start_drain_thread(process.stderr, stderr_capture, "stderr")

    timed_out = False
    group_terminated = False
    try:
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            group_terminated = _terminate_process_tree(process)

        # A shell may exit while a background descendant still owns the pipes.
        # Verification commands must not leave that unaccounted work running.
        _join_stream_threads(stdout_thread, stderr_thread, timeout=_STREAM_JOIN_SECONDS)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            group_terminated = _terminate_process_tree(process) or group_terminated
            _join_stream_threads(stdout_thread, stderr_thread, timeout=_STREAM_JOIN_SECONDS)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            _close_stream(process.stdout)
            _close_stream(process.stderr)
            _join_stream_threads(stdout_thread, stderr_thread, timeout=_STREAM_JOIN_SECONDS)
    finally:
        _reap_process(process)
        _close_stream(process.stdout)
        _close_stream(process.stderr)

    elapsed_ms = (time.monotonic() - started) * 1000
    return ProcessOutcome(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=stdout_capture.text(),
        stderr=stderr_capture.text(),
        timed_out=timed_out,
        elapsed_ms=elapsed_ms,
        managed_process_group_terminated=group_terminated,
    )


def _start_drain_thread(
    stream: BinaryIO,
    capture: _BoundedCapture,
    label: str,
) -> threading.Thread:
    def drain() -> None:
        try:
            while True:
                chunk = stream.read(_READ_CHUNK_BYTES)
                if not chunk:
                    return
                capture.feed(chunk)
        except (OSError, ValueError):
            return

    thread = threading.Thread(
        target=drain,
        name=f"harness-evolver-{label}-drain",
        daemon=True,
    )
    thread.start()
    return thread


def _join_stream_threads(*threads: threading.Thread, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for thread in threads:
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(remaining)


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> bool:
    """Terminate and reap the process group/tree managed for one command."""

    if os.name == "posix":
        process_group = process.pid
        if not _posix_group_exists(process_group):
            _reap_process(process)
            return False
        _signal_posix_group(process_group, signal.SIGTERM)
        terminated = True
        if not _wait_for_posix_group_exit(
            process,
            process_group,
            _TERMINATION_GRACE_SECONDS,
        ):
            _signal_posix_group(process_group, signal.SIGKILL)
            _wait_for_posix_group_exit(
                process,
                process_group,
                _TERMINATION_GRACE_SECONDS,
            )
        _reap_process(process)
        return terminated

    if os.name == "nt":
        terminated = False
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_TERMINATION_GRACE_SECONDS,
            )
            terminated = completed.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            try:
                process.kill()
                terminated = True
            except (OSError, ProcessLookupError):
                pass
        _reap_process(process)
        return terminated

    if process.poll() is not None:
        return False
    terminated = True
    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        terminated = False
    if not _wait_for_exit(process, _TERMINATION_GRACE_SECONDS):
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
    _reap_process(process)
    return terminated


def _signal_posix_group(process_group: int, sig: signal.Signals) -> None:
    try:
        os.killpg(process_group, sig)
    except (ProcessLookupError, PermissionError):
        return


def _posix_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_posix_group_exit(
    process: subprocess.Popen[bytes],
    process_group: int,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        # poll() reaps the direct child when it has exited. Without this, the
        # zombie group leader keeps killpg(..., 0) true for the full grace time.
        process.poll()
        if not _posix_group_exists(process_group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_TERMINATION_POLL_SECONDS)


def _reap_process(process: subprocess.Popen[bytes]) -> None:
    if process.returncode is not None:
        return
    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=_TERMINATION_GRACE_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass
    except OSError:
        pass


def _wait_for_exit(process: subprocess.Popen[bytes], timeout: float) -> bool:
    try:
        process.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def _close_stream(stream: BinaryIO | None) -> None:
    if stream is None:
        return
    try:
        stream.close()
    except (OSError, ValueError):
        pass
