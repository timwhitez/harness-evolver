"""Public bounded process runner with challenge-response cleanup attribution.

The Linux namespace/subreaper implementation lives in
:mod:`harness.tools._process_runner_issue19_namespace_base`. This facade keeps
that containment boundary, adds exact-argv execution and cooperative
cancellation, and reports successful cleanup only after the supervisor returns a
random challenge that was never inherited by the managed command.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import secrets
import select
import subprocess
import threading
import time
from typing import BinaryIO, Mapping, Sequence

from harness.tools import _process_runner_issue19_namespace_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_runtime = _base._base
_CLEANUP_TOKEN_BYTES = 32
_CLEANUP_PROOF_WAIT_SECONDS = 0.5


def _supervisor_exit_confirms_cleanup(returncode: int | None) -> bool:
    """Return the preliminary handled-termination status.

    Exit status 124 is necessary but not sufficient: a managed command can also
    return 124 normally. Public cleanup attribution additionally requires the
    random proof pipe validated by :func:`_cleanup_proof_matches`.
    """

    return returncode == _runtime._SUPERVISOR_TIMEOUT_EXIT


_base._supervisor_exit_confirms_cleanup = _supervisor_exit_confirms_cleanup
_runtime.supervised_command_for_argv = _base.supervised_command_for_argv
_runtime._terminate_supervised_process = _base._terminate_supervised_process


@dataclass(frozen=True)
class ProcessOutcome:
    """Result of one bounded command execution."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_ms: float
    managed_process_group_terminated: bool = False
    cancelled: bool = False


def _validated_timeout(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("timeout_seconds must be finite and > 0")
    try:
        timeout = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timeout_seconds must be finite and > 0") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and > 0")
    return timeout


def _validated_output_limit(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("output_limit_bytes must be >= 1")
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("output_limit_bytes must be >= 1") from exc
    if limit < 1:
        raise ValueError("output_limit_bytes must be >= 1")
    return limit


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _write_all_fd(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while sending cleanup challenge")
        view = view[written:]


def _close_raw_descriptor(stream: BinaryIO | None) -> None:
    """Close a pipe fd without contending on a buffered reader lock."""

    if stream is None:
        return
    try:
        descriptor = stream.fileno()
    except (OSError, ValueError):
        return
    _close_fd(descriptor)


def _prepare_supervised_launch(
    argv: Sequence[str],
) -> tuple[list[str], tuple[int, int], int, int, bytes]:
    """Build the supervisor argv and two parent-owned attestation pipes."""

    challenge_read, challenge_write = os.pipe()
    proof_read, proof_write = os.pipe()
    token = secrets.token_bytes(_CLEANUP_TOKEN_BYTES)
    command = [
        *_base.supervised_command_for_argv(argv),
        "--cleanup-challenge-fd",
        str(challenge_read),
        "--cleanup-proof-fd",
        str(proof_write),
    ]
    return (
        command,
        (challenge_read, proof_write),
        challenge_write,
        proof_read,
        token,
    )


def _attach_cleanup_proof(
    process: subprocess.Popen[bytes],
    *,
    proof_fd: int,
    token: bytes,
) -> None:
    setattr(process, "_harness_cleanup_proof_fd", proof_fd)
    setattr(process, "_harness_cleanup_token", token)
    setattr(process, "_harness_cleanup_proof_result", None)


def _cleanup_proof_matches(process: subprocess.Popen[bytes]) -> bool:
    """Consume and validate one supervisor proof exactly once."""

    cached = getattr(process, "_harness_cleanup_proof_result", None)
    if isinstance(cached, bool):
        return cached
    fd = getattr(process, "_harness_cleanup_proof_fd", None)
    token = getattr(process, "_harness_cleanup_token", None)
    if not isinstance(fd, int) or not isinstance(token, bytes):
        return False

    payload = bytearray()
    deadline = time.monotonic() + _CLEANUP_PROOF_WAIT_SECONDS
    try:
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                break
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > len(token):
                break
    except (OSError, ValueError):
        pass
    finally:
        _close_fd(fd)
        setattr(process, "_harness_cleanup_proof_fd", None)

    matched = bytes(payload) == token
    setattr(process, "_harness_cleanup_proof_result", matched)
    return matched


def _discard_cleanup_proof(process: subprocess.Popen[bytes]) -> None:
    fd = getattr(process, "_harness_cleanup_proof_fd", None)
    if isinstance(fd, int):
        _close_fd(fd)
        setattr(process, "_harness_cleanup_proof_fd", None)


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    *,
    process_token: str | None = None,
) -> bool:
    """Terminate the managed tree and require proof for a supervisor process."""

    terminated = _runtime._terminate_process_tree(
        process,
        process_token=process_token,
    )
    if getattr(process, "_harness_evolver_supervised", False):
        return bool(terminated and _cleanup_proof_matches(process))
    return terminated


def _run_bounded_argv(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    cwd: str | Path | None = None,
    output_limit_bytes: int = _runtime._OUTPUT_LIMIT_BYTES,
    env: Mapping[str, str] | None = None,
    cancel_event: threading.Event | None = None,
) -> ProcessOutcome:
    if not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("argv must be a non-empty sequence of strings")
    timeout = _validated_timeout(timeout_seconds)
    output_limit = _validated_output_limit(output_limit_bytes)

    # Match subprocess.Popen's established environment contract exactly:
    # ``None`` inherits the parent environment, while an explicit mapping is a
    # complete replacement. Overlaying it onto ``os.environ`` can reintroduce
    # credentials, proxy settings, or loader variables the caller deliberately
    # removed from a verification command.
    child_env = (
        None
        if env is None
        else {str(key): str(value) for key, value in env.items()}
    )

    supervised = _runtime._is_linux_subreaper_available()
    child_fds: tuple[int, int] = ()
    challenge_write: int | None = None
    proof_read: int | None = None
    cleanup_token: bytes | None = None
    if supervised:
        (
            launched_argv,
            child_fds,
            challenge_write,
            proof_read,
            cleanup_token,
        ) = _prepare_supervised_launch(argv)
    else:
        launched_argv = list(argv)

    started = time.monotonic()
    try:
        process = subprocess.Popen(
            launched_argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=child_env,
            start_new_session=(os.name == "posix"),
            pass_fds=child_fds,
        )
    except Exception:
        for fd in (*child_fds, challenge_write, proof_read):
            _close_fd(fd)
        raise

    if supervised:
        setattr(process, "_harness_evolver_supervised", True)
        for fd in child_fds:
            _close_fd(fd)
        assert challenge_write is not None
        assert proof_read is not None
        assert cleanup_token is not None
        _attach_cleanup_proof(
            process,
            proof_fd=proof_read,
            token=cleanup_token,
        )
        try:
            _write_all_fd(challenge_write, cleanup_token)
        except OSError:
            # The supervisor will fail setup and no proof can be accepted.
            _discard_cleanup_proof(process)
        finally:
            _close_fd(challenge_write)

    assert process.stdout is not None
    assert process.stderr is not None
    stdout_capture = _runtime._BoundedCapture(output_limit)
    stderr_capture = _runtime._BoundedCapture(output_limit)
    stdout_thread = _runtime._start_drain_thread(
        process.stdout,
        stdout_capture,
        "stdout",
    )
    stderr_thread = _runtime._start_drain_thread(
        process.stderr,
        stderr_capture,
        "stderr",
    )

    timed_out = False
    cancelled = False
    cleanup_confirmed = False
    deadline = started + timeout
    try:
        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                cleanup_confirmed = _terminate_process_tree(process)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                cleanup_confirmed = _terminate_process_tree(process)
                break
            time.sleep(min(0.02, remaining))

        _runtime._join_stream_threads(
            stdout_thread,
            stderr_thread,
            timeout=_runtime._STREAM_JOIN_SECONDS,
        )
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            background_cleanup = _terminate_process_tree(process)
            if timed_out or cancelled:
                cleanup_confirmed = cleanup_confirmed or background_cleanup
            _runtime._join_stream_threads(
                stdout_thread,
                stderr_thread,
                timeout=_runtime._STREAM_JOIN_SECONDS,
            )
    finally:
        _runtime._reap_process(process)
        if stdout_thread.is_alive():
            _close_raw_descriptor(process.stdout)
        if stderr_thread.is_alive():
            _close_raw_descriptor(process.stderr)
        _runtime._join_stream_threads(
            stdout_thread,
            stderr_thread,
            timeout=_runtime._STREAM_JOIN_SECONDS,
        )
        if not stdout_thread.is_alive():
            _runtime._close_stream(process.stdout)
        if not stderr_thread.is_alive():
            _runtime._close_stream(process.stderr)
        _discard_cleanup_proof(process)

    return ProcessOutcome(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=stdout_capture.text(),
        stderr=stderr_capture.text(),
        timed_out=timed_out,
        cancelled=cancelled,
        elapsed_ms=(time.monotonic() - started) * 1000,
        managed_process_group_terminated=cleanup_confirmed,
    )


def run_bounded_argv(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    cwd: str | Path | None = None,
    output_limit_bytes: int = _runtime._OUTPUT_LIMIT_BYTES,
    env: Mapping[str, str] | None = None,
    cancel_event: threading.Event | None = None,
) -> ProcessOutcome:
    """Run exact argv with bounded output and complete managed-tree cleanup."""

    return _run_bounded_argv(
        argv,
        timeout_seconds=timeout_seconds,
        cwd=cwd,
        output_limit_bytes=output_limit_bytes,
        env=env,
        cancel_event=cancel_event,
    )


def run_bounded_shell(
    command: str,
    *,
    timeout_seconds: float,
    cwd: str | Path | None = None,
    output_limit_bytes: int = _runtime._OUTPUT_LIMIT_BYTES,
    env: Mapping[str, str] | None = None,
    cancel_event: threading.Event | None = None,
) -> ProcessOutcome:
    """Run one platform shell command through the shared bounded runner."""

    if not isinstance(command, str):
        raise TypeError("command must be a string")
    if os.name == "nt":
        shell = os.environ.get("COMSPEC") or "cmd.exe"
        argv = [shell, "/d", "/s", "/c", command]
    else:
        argv = ["/bin/sh", "-c", command]
    return _run_bounded_argv(
        argv,
        timeout_seconds=timeout_seconds,
        cwd=cwd,
        output_limit_bytes=output_limit_bytes,
        env=env,
        cancel_event=cancel_event,
    )


supervised_command_for_argv = _base.supervised_command_for_argv
_pid_namespace_prefix = _base._pid_namespace_prefix
_is_linux_subreaper_available = _runtime._is_linux_subreaper_available
