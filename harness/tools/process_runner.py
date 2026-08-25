"""Public bounded process runner with verified cleanup attribution.

The Linux namespace/subreaper implementation lives in
:mod:`harness.tools._process_runner_issue19_namespace_base`.  This facade keeps
that containment boundary, adds exact-argv execution and cooperative
cancellation, and accepts cleanup only when the supervisor explicitly returns
its handled timeout status.  Shell-style 137/143 wrapper exits are never treated
as proof that descendants were reaped.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import BinaryIO, Mapping, Sequence

from harness.tools import _process_runner_issue19_namespace_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_runtime = _base._base


def _supervisor_exit_confirms_cleanup(returncode: int | None) -> bool:
    """Require the supervisor's explicit handled-termination status.

    ``process_supervisor.py`` returns 124 only after its TERM handler has
    terminated and reaped the complete descendant closure. Namespace and shell
    wrappers can map signal death to positive statuses such as 137 or 143; those
    values are failure evidence, not successful cleanup.
    """

    return returncode == _runtime._SUPERVISOR_TIMEOUT_EXIT


# Functions defined in the namespace module resolve globals dynamically, so
# replacing this module attribute tightens its timeout path without forking the
# namespace-discovery implementation.
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


def _close_raw_descriptor(stream: BinaryIO | None) -> None:
    """Close a pipe fd without contending on a buffered reader lock."""

    if stream is None:
        return
    try:
        descriptor = stream.fileno()
    except (OSError, ValueError):
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


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

    child_env = dict(os.environ)
    if env is not None:
        child_env.update({str(key): str(value) for key, value in env.items()})

    supervised = _runtime._is_linux_subreaper_available()
    launched_argv = (
        _base.supervised_command_for_argv(argv)
        if supervised
        else list(argv)
    )

    started = time.monotonic()
    process = subprocess.Popen(
        launched_argv,
        shell=False,
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
                cleanup_confirmed = _runtime._terminate_process_tree(process)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                cleanup_confirmed = _runtime._terminate_process_tree(process)
                break
            time.sleep(min(0.02, remaining))

        _runtime._join_stream_threads(
            stdout_thread,
            stderr_thread,
            timeout=_runtime._STREAM_JOIN_SECONDS,
        )
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            # A root process may exit while a background child retains a pipe.
            # Perform the same tree cleanup, but reserve the public attribution
            # flag for an explicit timeout/cancellation operation.
            background_cleanup = _runtime._terminate_process_tree(process)
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


_terminate_process_tree = _runtime._terminate_process_tree
supervised_command_for_argv = _base.supervised_command_for_argv
_pid_namespace_prefix = _base._pid_namespace_prefix
_is_linux_subreaper_available = _runtime._is_linux_subreaper_available
