"""Linux subreaper used to contain one shell command and all descendants.

This module is launched as an isolated script (``python -I -S``) by
``harness.tools.process_runner``. It intentionally depends only on the Python
standard library so site customizations cannot create untracked children before
the subreaper boundary is established.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Sequence


PR_SET_CHILD_SUBREAPER = 36
POLL_SECONDS = 0.02
TERM_GRACE_SECONDS = 1.0
KILL_GRACE_SECONDS = 1.0

TIMEOUT_EXIT_CODE = 124
CLEANUP_FAILED_EXIT_CODE = 125
SETUP_FAILED_EXIT_CODE = 126
_CLEANUP_TOKEN_BYTES = 32

_stop_signal: int | None = None


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    parent_pid: int
    state: str
    start_time: int


def encode_argv(argv: Sequence[str]) -> str:
    """Encode a child argv list for transport through one supervisor argument."""

    raw = json.dumps(
        list(argv),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_argv(encoded: str) -> list[str]:
    """Decode and validate an argv payload produced by :func:`encode_argv`."""

    padding = "=" * (-len(encoded) % 4)
    raw = base64.urlsafe_b64decode(encoded + padding)
    value = json.loads(raw)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError("supervisor payload must be a non-empty list of strings")
    return value


def _set_child_subreaper() -> None:
    """Ask Linux to reparent orphaned double-fork descendants to this process."""

    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _record_stop_signal(signum: int, _frame: object) -> None:
    global _stop_signal
    _stop_signal = signum


def _read_exact(fd: int, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        chunk = os.read(fd, size - len(payload))
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) != size:
        raise ValueError("cleanup challenge has an invalid length")
    return bytes(payload)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while publishing cleanup proof")
        view = view[written:]


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _read_cleanup_challenge(fd: int) -> bytes:
    try:
        return _read_exact(fd, _CLEANUP_TOKEN_BYTES)
    finally:
        _close_fd(fd)


def _linux_process_table() -> dict[int, _ProcessIdentity]:
    table: dict[int, _ProcessIdentity] = {}
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
            identity = _ProcessIdentity(
                pid=int(entry.name),
                state=fields[0],
                parent_pid=int(fields[1]),
                start_time=int(fields[19]),
            )
        except (OSError, ValueError, IndexError):
            continue
        table[identity.pid] = identity
    return table


def _descendants(supervisor_pid: int) -> dict[int, _ProcessIdentity]:
    """Return the current complete PPID closure below the supervisor."""

    table = _linux_process_table()
    selected = {supervisor_pid}
    changed = True
    while changed:
        changed = False
        for pid, identity in table.items():
            if identity.parent_pid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return {
        pid: table[pid]
        for pid in selected
        if pid in table and pid != supervisor_pid
    }


def _identity_is_alive(identity: _ProcessIdentity) -> bool:
    current = _linux_process_table().get(identity.pid)
    return (
        current is not None
        and current.start_time == identity.start_time
        and current.state != "Z"
    )


def _signal_identities(
    identities: object,
    sig: signal.Signals,
) -> None:
    for identity in list(identities):
        if not isinstance(identity, _ProcessIdentity):
            continue
        if not _identity_is_alive(identity):
            continue
        try:
            os.kill(identity.pid, sig)
        except (ProcessLookupError, PermissionError):
            continue


def _reap_orphaned_children(root_pid: int) -> None:
    """Reap direct orphaned descendants without stealing Popen's root status."""

    supervisor_pid = os.getpid()
    for pid, identity in _linux_process_table().items():
        if identity.parent_pid != supervisor_pid or pid == root_pid:
            continue
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError, OSError):
            continue


def _terminate_descendants(root: subprocess.Popen[bytes]) -> bool:
    """TERM/KILL every current or newly reparented descendant and reap it."""

    known: dict[int, _ProcessIdentity] = {}

    def refresh() -> dict[int, _ProcessIdentity]:
        known.update(_descendants(os.getpid()))
        return {
            pid: identity
            for pid, identity in known.items()
            if _identity_is_alive(identity)
        }

    live = refresh()
    _signal_identities(live.values(), signal.SIGTERM)

    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        root.poll()
        _reap_orphaned_children(root.pid)
        live = refresh()
        if not live:
            break
        _signal_identities(live.values(), signal.SIGTERM)
        time.sleep(POLL_SECONDS)

    live = refresh()
    if live:
        _signal_identities(live.values(), signal.SIGKILL)
        deadline = time.monotonic() + KILL_GRACE_SECONDS
        while time.monotonic() < deadline:
            root.poll()
            _reap_orphaned_children(root.pid)
            live = refresh()
            if not live:
                break
            _signal_identities(live.values(), signal.SIGKILL)
            time.sleep(POLL_SECONDS)

    try:
        root.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        try:
            root.kill()
        except ProcessLookupError:
            pass
        try:
            root.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass

    for _ in range(5):
        _reap_orphaned_children(root.pid)
        live = refresh()
        if not live:
            break
        _signal_identities(live.values(), signal.SIGKILL)
        time.sleep(POLL_SECONDS)

    return not refresh()


def _normalized_exit_code(returncode: int) -> int:
    if returncode >= 0:
        return min(returncode, 255)
    return min(128 + (-returncode), 255)


def main(argv: Sequence[str] | None = None) -> int:
    global _stop_signal
    _stop_signal = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--cleanup-challenge-fd", type=int)
    parser.add_argument("--cleanup-proof-fd", type=int)
    arguments = parser.parse_args(argv)

    proof_fd = arguments.cleanup_proof_fd
    challenge_fd = arguments.cleanup_challenge_fd
    cleanup_token: bytes | None = None

    if not sys.platform.startswith("linux") or not Path("/proc/self/stat").is_file():
        print("process supervisor requires Linux /proc", file=sys.stderr)
        _close_fd(challenge_fd)
        _close_fd(proof_fd)
        return SETUP_FAILED_EXIT_CODE

    try:
        if (challenge_fd is None) != (proof_fd is None):
            raise ValueError("cleanup challenge and proof descriptors must be paired")
        _set_child_subreaper()
        child_argv = decode_argv(arguments.payload)
        if challenge_fd is not None:
            cleanup_token = _read_cleanup_challenge(challenge_fd)
            challenge_fd = None
    except Exception as exc:
        _close_fd(challenge_fd)
        _close_fd(proof_fd)
        print(f"process supervisor setup failed: {exc}", file=sys.stderr)
        return SETUP_FAILED_EXIT_CODE

    signal.signal(signal.SIGTERM, _record_stop_signal)
    signal.signal(signal.SIGINT, _record_stop_signal)

    try:
        # close_fds prevents the managed command from inheriting the proof
        # channel. It can observe descriptor numbers in argv, but never receives
        # the random challenge token read and closed before this launch.
        root = subprocess.Popen(child_argv, close_fds=True)
    except Exception as exc:
        _close_fd(proof_fd)
        print(f"process supervisor child launch failed: {exc}", file=sys.stderr)
        return SETUP_FAILED_EXIT_CODE

    try:
        while True:
            if _stop_signal is not None:
                cleaned = _terminate_descendants(root)
                if not cleaned:
                    return CLEANUP_FAILED_EXIT_CODE
                if cleanup_token is not None and proof_fd is not None:
                    try:
                        _write_all(proof_fd, cleanup_token)
                    except OSError as exc:
                        print(
                            f"process supervisor cleanup proof failed: {exc}",
                            file=sys.stderr,
                        )
                        return CLEANUP_FAILED_EXIT_CODE
                return TIMEOUT_EXIT_CODE

            root_returncode = root.poll()
            if root_returncode is not None:
                if _descendants(os.getpid()) and not _terminate_descendants(root):
                    return CLEANUP_FAILED_EXIT_CODE
                return _normalized_exit_code(root_returncode)

            _reap_orphaned_children(root.pid)
            time.sleep(POLL_SECONDS)
    finally:
        _close_fd(proof_fd)


if __name__ == "__main__":
    raise SystemExit(main())
