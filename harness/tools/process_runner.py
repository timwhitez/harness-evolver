"""Graceful termination for the namespace-hardened process runner.

The namespace-capability and signal-attribution implementation is retained in
:mod:`harness.tools._process_runner_issue19_namespace_base`. When an ``unshare``
wrapper is active, this facade signals the inner ``process_supervisor.py``
process rather than terminating the outer namespace owner first. The supervisor
can then run its TERM/KILL/reap loop and return a trustworthy status through
``unshare``.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess

from harness.tools import _process_runner_issue19_namespace_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_runtime = _base._base


def _linux_process_table() -> dict[int, int]:
    """Return a best-effort PID -> PPID table from Linux procfs."""

    table: dict[int, int] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return table
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(
                encoding="utf-8",
                errors="strict",
            )
            _, remainder = stat_text.rsplit(") ", 1)
            fields = remainder.split()
            table[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    return table


def _descendant_pids(root_pid: int) -> list[int]:
    table = _linux_process_table()
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid in table.items():
            if parent_pid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return sorted(selected - {root_pid})


def _namespace_supervisor_pid(root_pid: int) -> int | None:
    """Find the inner standard-library supervisor below an ``unshare`` process."""

    if not os.path.isdir("/proc"):
        return None
    for pid in _descendant_pids(root_pid):
        try:
            command_line = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        if b"process_supervisor.py" in command_line:
            return pid
    return None


def _request_supervisor_cleanup(process: subprocess.Popen[bytes]) -> bool:
    """Signal the inner namespace init when present, otherwise the root process."""

    supervisor_pid = _namespace_supervisor_pid(process.pid)
    try:
        if supervisor_pid is not None:
            os.kill(supervisor_pid, signal.SIGTERM)
        else:
            process.send_signal(signal.SIGTERM)
    except (OSError, ProcessLookupError, PermissionError):
        return False
    return True


def _terminate_supervised_process(process: subprocess.Popen[bytes]) -> bool:
    """Run graceful cleanup before allowing the namespace owner to exit.

    Sending SIGTERM directly to ``unshare --kill-child=KILL`` can kill the
    namespace-init supervisor before its cleanup handler runs. Signal the inner
    supervisor instead; its handled exit is propagated by ``unshare``. If the
    inner process cannot be found, retain the conservative old fallback and do
    not claim cleanup success after a signal-killed wrapper.
    """

    if process.poll() is not None:
        return _base._supervisor_exit_confirms_cleanup(process.returncode)

    if not _request_supervisor_cleanup(process):
        return False

    try:
        process.wait(timeout=_runtime._SUPERVISOR_SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
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
        _runtime._reap_process(process)
        return False

    return _base._supervisor_exit_confirms_cleanup(process.returncode)


_runtime._terminate_supervised_process = _terminate_supervised_process
run_bounded_shell = _runtime.run_bounded_shell
_terminate_process_tree = _runtime._terminate_process_tree
supervised_command_for_argv = _base.supervised_command_for_argv
_pid_namespace_prefix = _base._pid_namespace_prefix
_supervisor_exit_confirms_cleanup = _base._supervisor_exit_confirms_cleanup
