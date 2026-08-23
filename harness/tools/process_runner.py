"""Bounded subprocess execution with managed descendant cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
import time
from typing import BinaryIO, Mapping


_OUTPUT_LIMIT_BYTES = 1_000_000
_TERMINATION_GRACE_SECONDS = 1.0
_TERMINATION_POLL_SECONDS = 0.02
_STREAM_JOIN_SECONDS = 1.0
_READ_CHUNK_BYTES = 65_536
_PROCESS_TOKEN_ENV = "HL_PROCESS_TREE_TOKEN"


@dataclass(frozen=True)
class ProcessOutcome:
    """Result of one bounded shell process."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_ms: float
    managed_process_group_terminated: bool = False


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    start_time: int


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
    env: Mapping[str, str] | None = None,
) -> ProcessOutcome:
    """Run a shell command and clean up all discoverable managed descendants.

    Stdout/stderr are drained concurrently into bounded in-memory head/tail
    buffers. On Linux, every spawned process inherits a unique token; cleanup
    combines process-group signals, `/proc` parent traversal, and token-based
    discovery. This catches descendants that call ``setsid()`` or double-fork
    out of the original process group. Other POSIX systems retain process-group
    cleanup, while Windows uses ``taskkill /T /F``.
    """

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and > 0")
    if output_limit_bytes < 1:
        raise ValueError("output_limit_bytes must be >= 1")

    process_token = secrets.token_hex(24)
    child_env = dict(os.environ)
    if env is not None:
        child_env.update({str(key): str(value) for key, value in env.items()})
    child_env[_PROCESS_TOKEN_ENV] = process_token

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
        env=child_env,
        **popen_kwargs,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_capture = _BoundedCapture(output_limit_bytes)
    stderr_capture = _BoundedCapture(output_limit_bytes)
    stdout_thread = _start_drain_thread(process.stdout, stdout_capture, "stdout")
    stderr_thread = _start_drain_thread(process.stderr, stderr_capture, "stderr")

    timed_out = False
    managed_tree_terminated = False
    try:
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            managed_tree_terminated = _terminate_process_tree(
                process,
                process_token=process_token,
            )

        # A root shell may exit while background or daemonized descendants keep
        # running and retain the output pipes. Give normal commands a chance to
        # finish draining, then clean up any remaining managed process tree.
        _join_stream_threads(stdout_thread, stderr_thread, timeout=_STREAM_JOIN_SECONDS)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            managed_tree_terminated = (
                _terminate_process_tree(process, process_token=process_token)
                or managed_tree_terminated
            )
            _join_stream_threads(stdout_thread, stderr_thread, timeout=_STREAM_JOIN_SECONDS)
        elif _managed_descendants_exist(process.pid, process_token):
            managed_tree_terminated = (
                _terminate_process_tree(process, process_token=process_token)
                or managed_tree_terminated
            )

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


def _join_stream_threads(*threads: threading.Thread, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for thread in threads:
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(remaining)


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    *,
    process_token: str | None = None,
) -> bool:
    """Terminate and reap the process group/tree managed for one command."""

    if os.name == "posix":
        if process_token and _linux_proc_available():
            return _terminate_linux_process_tree(process, process_token)
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


def _managed_descendants_exist(root_pid: int, process_token: str) -> bool:
    if not _linux_proc_available():
        return _posix_group_exists(root_pid) if os.name == "posix" else False
    return bool(_linux_managed_processes(root_pid, process_token, {}))


def _linux_proc_available() -> bool:
    return os.name == "posix" and Path("/proc/self/stat").is_file()


def _terminate_linux_process_tree(
    process: subprocess.Popen[bytes],
    process_token: str,
) -> bool:
    root_pid = process.pid
    known: dict[int, _ProcessIdentity] = {}

    def refresh() -> dict[int, _ProcessIdentity]:
        known.update(_linux_managed_processes(root_pid, process_token, known))
        return {
            pid: identity
            for pid, identity in known.items()
            if _identity_is_alive(identity)
        }

    alive = refresh()
    _signal_posix_group(root_pid, signal.SIGTERM)
    _signal_identities(alive.values(), signal.SIGTERM)

    deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        alive = refresh()
        if not alive and not _posix_group_exists(root_pid):
            break
        time.sleep(_TERMINATION_POLL_SECONDS)

    alive = refresh()
    if alive or _posix_group_exists(root_pid):
        _signal_posix_group(root_pid, signal.SIGKILL)
        _signal_identities(alive.values(), signal.SIGKILL)
        deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
        while time.monotonic() < deadline:
            process.poll()
            alive = refresh()
            if not alive and not _posix_group_exists(root_pid):
                break
            _signal_identities(alive.values(), signal.SIGKILL)
            time.sleep(_TERMINATION_POLL_SECONDS)

    _reap_process(process)
    _reap_known_children(known.values())
    return not refresh() and not _posix_group_exists(root_pid)


def _linux_managed_processes(
    root_pid: int,
    process_token: str,
    known: Mapping[int, _ProcessIdentity],
) -> dict[int, _ProcessIdentity]:
    table = _linux_process_table()
    roots = {root_pid, *known.keys(), *_linux_token_pids(process_token)}
    selected = set(roots)

    changed = True
    while changed:
        changed = False
        for pid, (_, parent_pid) in table.items():
            if parent_pid in selected and pid not in selected:
                selected.add(pid)
                changed = True

    return {
        pid: _ProcessIdentity(pid=pid, start_time=table[pid][0])
        for pid in selected
        if pid in table and pid != os.getpid()
    }


def _linux_process_table() -> dict[int, tuple[int, int]]:
    table: dict[int, tuple[int, int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return table
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8", errors="strict")
            _, remainder = stat_text.rsplit(") ", 1)
            fields = remainder.split()
            table[int(entry.name)] = (int(fields[19]), int(fields[1]))
        except (OSError, ValueError, IndexError):
            continue
    return table


def _linux_token_pids(process_token: str) -> set[int]:
    marker = f"{_PROCESS_TOKEN_ENV}={process_token}".encode() + b"\0"
    matches: set[int] = set()
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return matches
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if marker in environ:
            matches.add(int(entry.name))
    return matches


def _identity_is_alive(identity: _ProcessIdentity) -> bool:
    current = _linux_process_table().get(identity.pid)
    return current is not None and current[0] == identity.start_time


def _signal_identities(
    identities: object,
    sig: signal.Signals,
) -> None:
    for identity in list(identities):
        if not isinstance(identity, _ProcessIdentity) or not _identity_is_alive(identity):
            continue
        try:
            os.kill(identity.pid, sig)
        except (ProcessLookupError, PermissionError):
            continue


def _reap_known_children(identities: object) -> None:
    for identity in list(identities):
        if not isinstance(identity, _ProcessIdentity):
            continue
        try:
            os.waitpid(identity.pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError, OSError):
            continue


def _terminate_posix_process_group(process: subprocess.Popen[bytes]) -> bool:
    process_group = process.pid
    if not _posix_group_exists(process_group):
        _reap_process(process)
        return process.poll() is not None
    _signal_posix_group(process_group, signal.SIGTERM)
    if not _wait_for_posix_group_exit(process, process_group, _TERMINATION_GRACE_SECONDS):
        _signal_posix_group(process_group, signal.SIGKILL)
        _wait_for_posix_group_exit(process, process_group, _TERMINATION_GRACE_SECONDS)
    _reap_process(process)
    return not _posix_group_exists(process_group)


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
