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

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Sequence

from harness.tools import _process_runner_issue19_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, _value)

_BASE_SUPERVISED_COMMAND = _base.supervised_command_for_argv
_PID_NAMESPACE_PROBE_SECONDS = 2.0
_NAMESPACE_SUPERVISOR_DISCOVERY_SECONDS = 1.0
_NAMESPACE_SUPERVISOR_POLL_SECONDS = 0.02
_PID_NAMESPACE_PROBE = (
    "import os; from pathlib import Path; "
    "self_pid=int(Path('/proc/self/stat').read_text(encoding='utf-8').split(maxsplit=1)[0]); "
    "init_pid=int(Path('/proc/1/stat').read_text(encoding='utf-8').split(maxsplit=1)[0]); "
    "raise SystemExit(0 if os.getpid() == self_pid == init_pid == 1 else 1)"
)


@dataclass(frozen=True)
class _HostProcessIdentity:
    pid: int
    parent_pid: int
    start_time: int
    argv: tuple[bytes, ...]


@lru_cache(maxsize=1)
def _pid_namespace_prefix() -> tuple[str, ...] | None:
    """Return a verified user/PID namespace wrapper with a private procfs."""

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
    """Accept only an orderly supervisor/wrapper exit as cleanup evidence."""

    return (
        returncode is not None
        and returncode >= 0
        and returncode
        not in {
            _base._SUPERVISOR_CLEANUP_FAILED_EXIT,
            _base._SUPERVISOR_SETUP_FAILED_EXIT,
        }
    )


def _process_uses_pid_namespace(process: subprocess.Popen[bytes]) -> bool:
    prefix = _pid_namespace_prefix()
    args = process.args
    if prefix is None or not isinstance(args, (list, tuple)):
        return False
    return list(args[: len(prefix)]) == list(prefix)


def _read_host_process_identity(pid: int) -> _HostProcessIdentity | None:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(
            encoding="utf-8",
            errors="strict",
        )
        _, remainder = stat_text.rsplit(") ", 1)
        fields = remainder.split()
        argv = tuple(
            part
            for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if part
        )
        return _HostProcessIdentity(
            pid=pid,
            parent_pid=int(fields[1]),
            start_time=int(fields[19]),
            argv=argv,
        )
    except (OSError, ValueError, IndexError):
        return None


def _host_process_table() -> dict[int, _HostProcessIdentity]:
    table: dict[int, _HostProcessIdentity] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return table
    for entry in entries:
        if not entry.name.isdigit():
            continue
        identity = _read_host_process_identity(int(entry.name))
        if identity is not None:
            table[identity.pid] = identity
    return table


def _namespace_supervisor_identity(
    outer_pid: int,
) -> _HostProcessIdentity | None:
    """Find the unique closest process_supervisor.py descendant on host procfs."""

    table = _host_process_table()
    depths = {outer_pid: 0}
    changed = True
    while changed:
        changed = False
        for pid, identity in table.items():
            if identity.parent_pid in depths and pid not in depths:
                depths[pid] = depths[identity.parent_pid] + 1
                changed = True

    script = os.fsencode(str(Path(__file__).with_name("process_supervisor.py").resolve()))
    candidates = [
        (depths[pid], identity)
        for pid, identity in table.items()
        if pid in depths
        and pid != outer_pid
        and script in identity.argv
        and b"--payload" in identity.argv
    ]
    if not candidates:
        return None
    minimum_depth = min(depth for depth, _ in candidates)
    nearest = [identity for depth, identity in candidates if depth == minimum_depth]
    return nearest[0] if len(nearest) == 1 else None


def _signal_namespace_supervisor(
    process: subprocess.Popen[bytes],
) -> bool:
    """Signal the inner namespace init, never the outer unshare wrapper first."""

    deadline = time.monotonic() + _NAMESPACE_SUPERVISOR_DISCOVERY_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        identity = _namespace_supervisor_identity(process.pid)
        if identity is not None:
            current = _read_host_process_identity(identity.pid)
            if current is None or current.start_time != identity.start_time:
                time.sleep(_NAMESPACE_SUPERVISOR_POLL_SECONDS)
                continue
            try:
                os.kill(identity.pid, signal.SIGTERM)
                return True
            except (OSError, ProcessLookupError, PermissionError):
                return False
        time.sleep(_NAMESPACE_SUPERVISOR_POLL_SECONDS)
    return False


def _kill_outer_session(process: subprocess.Popen[bytes]) -> None:
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


def _terminate_supervised_process(process: subprocess.Popen[bytes]) -> bool:
    """Ask the inner supervisor to clean descendants before stopping wrappers."""

    if process.poll() is not None:
        return _supervisor_exit_confirms_cleanup(process.returncode)

    if _process_uses_pid_namespace(process):
        # Signalling unshare first activates --kill-child=KILL and can destroy
        # namespace PID 1 before its TERM/KILL/reap handler runs. Discover and
        # signal the exact inner supervisor through the host procfs view.
        if not _signal_namespace_supervisor(process):
            _kill_outer_session(process)
            return False
    else:
        try:
            process.send_signal(signal.SIGTERM)
        except (OSError, ProcessLookupError):
            return False

    try:
        process.wait(timeout=_base._SUPERVISOR_SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_outer_session(process)
        return False

    return _supervisor_exit_confirms_cleanup(process.returncode)


_base.supervised_command_for_argv = supervised_command_for_argv
_base._terminate_supervised_process = _terminate_supervised_process

run_bounded_shell = _base.run_bounded_shell
_terminate_process_tree = _base._terminate_process_tree
