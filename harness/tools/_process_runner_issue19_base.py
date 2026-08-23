"""Bounded subprocess execution with Linux subreaper containment."""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import BinaryIO, Mapping, Sequence


_OUTPUT_LIMIT_BYTES = 1_000_000
_READ_CHUNK_BYTES = 65_536
_STREAM_JOIN_SECONDS = 1.0
_TERMINATION_GRACE_SECONDS = 1.0
_SUPERVISOR_SHUTDOWN_SECONDS = 3.0

_PROCESS_TOKEN_ENV = "HL_PROCESS_TREE_TOKEN"  # compatibility for stacked PRs
_SUPERVISOR_TIMEOUT_EXIT = 124
_SUPERVISOR_CLEANUP_FAILED_EXIT = 125
_SUPERVISOR_SETUP_FAILED_EXIT = 126


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
    """Continuously drain a stream while retaining a bounded head/tail."""

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


def _is_linux_subreaper_available() -> bool:
    return sys.platform.startswith("linux") and Path("/proc/self/stat").is_file()


def _encode_supervisor_argv(argv: Sequence[str]) -> str:
    raw = json.dumps(
        list(argv),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def supervised_command_for_argv(argv: Sequence[str]) -> list[str]:
    """Return an isolated Python command that supervises exactly ``argv``."""

    if not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("supervised argv must be a non-empty list of strings")
    script = Path(__file__).with_name("process_supervisor.py")
    if not script.is_file():
        raise RuntimeError(f"process supervisor is missing: {script}")
    return [
        sys.executable,
        "-I",
        "-S",
        str(script),
        "--payload",
        _encode_supervisor_argv(argv),
    ]


def run_bounded_shell(
    command: str,
    *,
    timeout_seconds: float,
    cwd: str | Path | None = None,
    output_limit_bytes: int = _OUTPUT_LIMIT_BYTES,
    env: Mapping[str, str] | None = None,
) -> ProcessOutcome:
    """Run one shell command with bounded output and descendant cleanup.

    Linux commands execute below a dedicated ``PR_SET_CHILD_SUBREAPER`` process.
    That kernel reparenting boundary catches double-fork/``setsid`` descendants
    even when they clear their environment. Other POSIX systems retain
    process-group cleanup; Windows uses ``taskkill /T /F``.
    """

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and > 0")
    if output_limit_bytes < 1:
        raise ValueError("output_limit_bytes must be >= 1")

    child_env = dict(os.environ)
    if env is not None:
        child_env.update({str(key): str(value) for key, value in env.items()})

    supervised = _is_linux_subreaper_available()
    if supervised:
        argv: str | list[str] = supervised_command_for_argv(
            ["/bin/sh", "-c", command]
        )
        use_shell = False
    else:
        argv = command
        use_shell = True

    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        shell=use_shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=child_env,
        start_new_session=(os.name == "posix"),
    )
    if supervised:
        setattr(process, "_harness_evolver_supervised", True)

    assert process.stdout is not None
    assert process.stderr is not None
    stdout_capture = _BoundedCapture(output_limit_bytes)
    stderr_capture = _BoundedCapture(output_limit_bytes)
    stdout_thread = _start_drain_thread(
        process.stdout,
        stdout_capture,
        "stdout",
    )
    stderr_thread = _start_drain_thread(
        process.stderr,
        stderr_capture,
        "stderr",
    )

    timed_out = False
    managed_tree_terminated = False
    try:
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            managed_tree_terminated = _terminate_process_tree(process)

        _join_stream_threads(
            stdout_thread,
            stderr_thread,
            timeout=_STREAM_JOIN_SECONDS,
        )
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            managed_tree_terminated = (
                _terminate_process_tree(process) or managed_tree_terminated
            )
            _join_stream_threads(
                stdout_thread,
                stderr_thread,
                timeout=_STREAM_JOIN_SECONDS,
            )
    finally:
        _reap_process(process)
        _close_stream(process.stdout)
        _close_stream(process.stderr)
        _join_stream_threads(
            stdout_thread,
            stderr_thread,
            timeout=_STREAM_JOIN_SECONDS,
        )

    elapsed_ms = (time.monotonic() - started) * 1000
    return ProcessOutcome(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=stdout_capture.text(),
        stderr=stderr_capture.text(),
        timed_out=timed_out,
        elapsed_ms=elapsed_ms,
        managed_process_group_terminated=managed_tree_terminated,
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


def _join_stream_threads(
    *threads: threading.Thread,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    *,
    process_token: str | None = None,
) -> bool:
    """Terminate the managed command tree.

    ``process_token`` remains accepted for compatibility with stacked branches;
    Linux completeness is provided by the dedicated subreaper process instead.
    """

    del process_token
    if getattr(process, "_harness_evolver_supervised", False):
        return _terminate_supervised_process(process)

    if os.name == "posix":
        return _terminate_posix_process_group(process)

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
        if process.poll() is None and not _wait_for_exit(
            process,
            _TERMINATION_GRACE_SECONDS,
        ):
            try:
                process.kill()
                terminated = True
            except (OSError, ProcessLookupError):
                pass
        _reap_process(process)
        return terminated

    if process.poll() is not None:
        return False
    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        return False
    if not _wait_for_exit(process, _TERMINATION_GRACE_SECONDS):
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
    _reap_process(process)
    return process.poll() is not None


def _terminate_supervised_process(process: subprocess.Popen[bytes]) -> bool:
    """Ask the subreaper to clean descendants before killing the supervisor."""

    if process.poll() is not None:
        return process.returncode not in {
            _SUPERVISOR_CLEANUP_FAILED_EXIT,
            _SUPERVISOR_SETUP_FAILED_EXIT,
        }

    try:
        process.send_signal(signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return False

    try:
        process.wait(timeout=_SUPERVISOR_SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        # The supervisor failed to complete its kernel-backed cleanup. Kill the
        # original session as a final best effort, but report cleanup failure.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        else:
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
        _reap_process(process)
        return False

    return process.returncode not in {
        _SUPERVISOR_CLEANUP_FAILED_EXIT,
        _SUPERVISOR_SETUP_FAILED_EXIT,
    }


def _terminate_posix_process_group(process: subprocess.Popen[bytes]) -> bool:
    process_group = process.pid
    if not _posix_group_exists(process_group):
        _reap_process(process)
        return process.poll() is not None

    _signal_posix_group(process_group, signal.SIGTERM)
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
    return not _posix_group_exists(process_group)


def _signal_posix_group(
    process_group: int,
    sig: signal.Signals,
) -> None:
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
        process.poll()
        if not _posix_group_exists(process_group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


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


def _wait_for_exit(
    process: subprocess.Popen[bytes],
    timeout: float,
) -> bool:
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
