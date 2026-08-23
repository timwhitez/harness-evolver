"""Hardened facade for the bounded process runner.

The subreaper implementation is retained in
:mod:`harness.tools._process_runner_issue19_base`. On Linux, this facade places
the standalone supervisor below a verified user/PID namespace boundary when the
host permits it. The capability probe requires a private procfs mounted for the
child PID namespace; a PID namespace with the host's procfs view is rejected
because namespace-relative PIDs cannot safely be compared with host-relative
``/proc/*/stat`` records.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import Sequence

from harness.tools import _process_runner_issue19_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_BASE_SUPERVISED_COMMAND = _base.supervised_command_for_argv
_PID_NAMESPACE_PROBE_SECONDS = 2.0
_PID_NAMESPACE_PROBE = (
    "import os; from pathlib import Path; "
    "self_pid=int(Path('/proc/self/stat').read_text(encoding='utf-8').split(maxsplit=1)[0]); "
    "init_pid=int(Path('/proc/1/stat').read_text(encoding='utf-8').split(maxsplit=1)[0]); "
    "raise SystemExit(0 if os.getpid() == self_pid == init_pid == 1 else 1)"
)


@lru_cache(maxsize=1)
def _pid_namespace_prefix() -> tuple[str, ...] | None:
    """Return a verified user/PID namespace wrapper with a private procfs.

    ``unshare --pid`` alone is insufficient: unless procfs is remounted inside
    the namespace, ``os.getpid()`` is namespace-relative while ``/proc`` may
    still expose host-relative identifiers. The supervisor relies on procfs for
    PPID closure and PID start-time validation, so strong containment is enabled
    only when an executable probe proves that the command is PID 1 and
    ``/proc/self`` plus ``/proc/1`` describe that same namespace PID.
    """

    if not sys.platform.startswith("linux"):
        return None
    unshare = shutil.which("unshare")
    if unshare is None:
        return None

    prefix = (
        unshare,
        "--map-current-user",
        "--pid",
        "--fork",
        "--mount-proc",
        "--kill-child=KILL",
        "--",
    )
    try:
        probe = subprocess.run(
            [
                *prefix,
                sys.executable,
                "-I",
                "-S",
                "-c",
                _PID_NAMESPACE_PROBE,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_PID_NAMESPACE_PROBE_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return prefix if probe.returncode == 0 else None


def supervised_command_for_argv(argv: Sequence[str]) -> list[str]:
    """Build the supervisor argv, optionally isolated in a verified namespace."""

    supervisor = _BASE_SUPERVISED_COMMAND(argv)
    prefix = _pid_namespace_prefix()
    if prefix is None:
        return supervisor
    return [*prefix, *supervisor]


def _supervisor_exit_confirms_cleanup(returncode: int | None) -> bool:
    """Accept only an orderly supervisor/wrapper exit as cleanup evidence.

    A negative return code means the outer supervisor or namespace wrapper died
    from a signal. Its cleanup path cannot be assumed to have run, so the result
    must never advertise successful tree termination merely because the code is
    different from the two positive sentinel values.
    """

    return (
        returncode is not None
        and returncode >= 0
        and returncode
        not in {
            _base._SUPERVISOR_CLEANUP_FAILED_EXIT,
            _base._SUPERVISOR_SETUP_FAILED_EXIT,
        }
    )


def _terminate_supervised_process(process: subprocess.Popen[bytes]) -> bool:
    """Ask the managed supervisor/namespace boundary to clean descendants."""

    if process.poll() is not None:
        return _supervisor_exit_confirms_cleanup(process.returncode)

    try:
        process.send_signal(signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return False

    try:
        process.wait(timeout=_base._SUPERVISOR_SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        # A namespace-enabled unshare process or the standalone supervisor did
        # not finish cleanup. Kill the outer session only as a final best effort
        # and report failure rather than claiming complete containment.
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
        _base._reap_process(process)
        return False

    return _supervisor_exit_confirms_cleanup(process.returncode)


_base.supervised_command_for_argv = supervised_command_for_argv
_base._terminate_supervised_process = _terminate_supervised_process

# Re-export the patched callables explicitly after the compatibility surface was
# copied above.
run_bounded_shell = _base.run_bounded_shell
_terminate_process_tree = _base._terminate_process_tree
