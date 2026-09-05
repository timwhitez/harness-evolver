"""Bounded exact-argv execution with single-owner pipes and cleanup attestation.

The existing namespace/subreaper owns process-tree containment. Output is read
nonblockingly by the caller, never by a thread sharing an owning stream. One
resource scope covers every allocation, launch, setup failure, and teardown.
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
from typing import Mapping, Sequence

from harness.tools import _process_runner_issue19_namespace_base as _base
from harness.tools.pipe_io import PipeReader

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, _value)

_runtime = _base._base
_CLEANUP_TOKEN_BYTES = 32
_CLEANUP_PROOF_WAIT_SECONDS = 0.5


def _supervisor_exit_confirms_cleanup(returncode: int | None) -> bool:
    # Status 124 alone is not proof: managed commands can return it normally.
    return returncode == _runtime._SUPERVISOR_TIMEOUT_EXIT


_base._supervisor_exit_confirms_cleanup = _supervisor_exit_confirms_cleanup
_runtime.supervised_command_for_argv = _base.supervised_command_for_argv
_runtime._terminate_supervised_process = _base._terminate_supervised_process


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_ms: float
    managed_process_group_terminated: bool = False
    cancelled: bool = False
    output_eof: bool = True


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


def _validated_argv(value: object) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("argv must be a non-empty sequence of strings")
    argv = list(value)
    if not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("argv must be a non-empty sequence of strings")
    return argv


def _close_fd(fd: int | None) -> None:
    if fd is not None:
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


def _prepare_supervised_launch(
    argv: Sequence[str],
) -> tuple[list[str], tuple[int, int], int, int, bytes]:
    owned: list[int] = []
    try:
        challenge_read, challenge_write = os.pipe()
        owned.extend((challenge_read, challenge_write))
        proof_read, proof_write = os.pipe()
        owned.extend((proof_read, proof_write))
        token = secrets.token_bytes(_CLEANUP_TOKEN_BYTES)
        command = [*_base.supervised_command_for_argv(argv),
                   "--cleanup-challenge-fd", str(challenge_read),
                   "--cleanup-proof-fd", str(proof_write)]
        return command, (challenge_read, proof_write), challenge_write, proof_read, token
    except BaseException:
        for fd in owned:
            _close_fd(fd)
        raise


def _attach_cleanup_proof(process, *, proof_fd: int, token: bytes) -> None:
    process._harness_cleanup_proof_fd = proof_fd
    process._harness_cleanup_token = token
    process._harness_cleanup_proof_result = None


def _cleanup_proof_matches(process) -> bool:
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
            readable, _, _ = select.select([fd], [], [], max(0.0, deadline - time.monotonic()))
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
        process._harness_cleanup_proof_fd = None
        _close_fd(fd)
    matched = bytes(payload) == token
    process._harness_cleanup_proof_result = matched
    return matched


def _discard_cleanup_proof(process) -> None:
    fd = getattr(process, "_harness_cleanup_proof_fd", None)
    process._harness_cleanup_proof_fd = None
    if isinstance(fd, int):
        _close_fd(fd)


def _terminate_process_tree(process, *, process_token: str | None = None) -> bool:
    terminated = _runtime._terminate_process_tree(process, process_token=process_token)
    if getattr(process, "_harness_evolver_supervised", False):
        return bool(terminated and _cleanup_proof_matches(process))
    return terminated


def _run_bounded_argv(
    argv: Sequence[str], *, timeout_seconds: float,
    cwd: str | Path | None = None,
    output_limit_bytes: int = _runtime._OUTPUT_LIMIT_BYTES,
    env: Mapping[str, str] | None = None,
    cancel_event: threading.Event | None = None,
) -> ProcessOutcome:
    normalized_argv = _validated_argv(argv)
    timeout = _validated_timeout(timeout_seconds)
    output_limit = _validated_output_limit(output_limit_bytes)
    child_env = None if env is None else {str(k): str(v) for k, v in env.items()}
    # Capture allocation cannot strand a child: do it before acquiring resources.
    stdout_capture = _runtime._BoundedCapture(output_limit)
    stderr_capture = _runtime._BoundedCapture(output_limit)
    supervised = _runtime._is_linux_subreaper_available()
    child_fds: tuple[int, ...] = ()
    challenge_write = proof_read = None
    process = None
    readers: list[tuple[PipeReader, object]] = []
    timed_out = cancelled = cleanup_confirmed = False
    def pump() -> bool:
        progress = False
        for reader, capture in list(readers):
            data = reader.read()
            if data == b"":
                readers.remove((reader, capture))
            elif data is not None:
                capture.feed(data)
                progress = True
        return progress

    def drain_for(seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while readers and time.monotonic() < deadline:
            if not pump():
                time.sleep(0.01)

    try:
        launched_argv = normalized_argv
        if supervised:
            launched_argv, child_fds, challenge_write, proof_read, token = (
                _prepare_supervised_launch(normalized_argv)
            )
        # Keep the established timeout boundary: capability discovery/setup
        # precedes the command's execution budget.
        started = time.monotonic()
        process = subprocess.Popen(
            launched_argv, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=cwd, env=child_env, bufsize=0,
            start_new_session=(os.name == "posix"), pass_fds=child_fds,
        )
        if supervised:
            process._harness_evolver_supervised = True
            while child_fds:
                fd, *remaining = child_fds
                child_fds = tuple(remaining)
                _close_fd(fd)
            _attach_cleanup_proof(process, proof_fd=proof_read, token=token)
            proof_read = None  # ownership transferred to process attribute
            try:
                _write_all_fd(challenge_write, token)
            except OSError:
                _discard_cleanup_proof(process)
            finally:
                fd, challenge_write = challenge_write, None
                _close_fd(fd)
        assert process.stdout is not None and process.stderr is not None
        readers.append((PipeReader(process.stdout), stdout_capture))
        readers.append((PipeReader(process.stderr), stderr_capture))
        deadline = started + timeout
        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                cleanup_confirmed = _terminate_process_tree(process)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                cleanup_confirmed = _terminate_process_tree(process)
                break
            if not pump():
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        drain_for(_runtime._STREAM_JOIN_SECONDS)
        if readers:
            background_cleanup = _terminate_process_tree(process)
            if timed_out or cancelled:
                cleanup_confirmed = cleanup_confirmed or background_cleanup
            drain_for(_runtime._STREAM_JOIN_SECONDS)
    finally:
        # Also covers Popen, reader initialization, capture.feed, and cancellation
        # exceptions. Never raw-close a descriptor owned by a Python stream.
        try:
            if process is not None:
                try:
                    if process.poll() is None:
                        _terminate_process_tree(process)
                except Exception:
                    # Preserve the original setup/read error; reaping is still tried.
                    pass
                finally:
                    try:
                        _runtime._reap_process(process)
                    finally:
                        _runtime._close_stream(process.stdout)
                        _runtime._close_stream(process.stderr)
                        _discard_cleanup_proof(process)
        finally:
            for fd in (*child_fds, challenge_write, proof_read):
                _close_fd(fd)
    return ProcessOutcome(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=stdout_capture.text(), stderr=stderr_capture.text(),
        timed_out=timed_out, cancelled=cancelled,
        elapsed_ms=(time.monotonic() - started) * 1000,
        managed_process_group_terminated=cleanup_confirmed,
        output_eof=not readers,
    )


def run_bounded_argv(
    argv: Sequence[str], *, timeout_seconds: float,
    cwd: str | Path | None = None,
    output_limit_bytes: int = _runtime._OUTPUT_LIMIT_BYTES,
    env: Mapping[str, str] | None = None,
    cancel_event: threading.Event | None = None,
) -> ProcessOutcome:
    return _run_bounded_argv(argv, timeout_seconds=timeout_seconds, cwd=cwd,
                             output_limit_bytes=output_limit_bytes, env=env,
                             cancel_event=cancel_event)


def run_bounded_shell(
    command: str, *, timeout_seconds: float,
    cwd: str | Path | None = None,
    output_limit_bytes: int = _runtime._OUTPUT_LIMIT_BYTES,
    env: Mapping[str, str] | None = None,
    cancel_event: threading.Event | None = None,
) -> ProcessOutcome:
    if not isinstance(command, str):
        raise TypeError("command must be a string")
    argv = ([os.environ.get("COMSPEC") or "cmd.exe", "/d", "/s", "/c", command]
            if os.name == "nt" else ["/bin/sh", "-c", command])
    return _run_bounded_argv(argv, timeout_seconds=timeout_seconds, cwd=cwd,
                             output_limit_bytes=output_limit_bytes, env=env,
                             cancel_event=cancel_event)


supervised_command_for_argv = _base.supervised_command_for_argv
_pid_namespace_prefix = _base._pid_namespace_prefix
_is_linux_subreaper_available = _runtime._is_linux_subreaper_available
