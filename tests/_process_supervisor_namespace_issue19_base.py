from __future__ import annotations

import inspect
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

import pytest

from harness.tools import process_runner
from harness.tools.process_runner import run_bounded_shell


def test_signal_killed_supervisor_never_claims_successful_cleanup() -> None:
    assert process_runner._supervisor_exit_confirms_cleanup(-signal.SIGKILL) is False
    assert process_runner._supervisor_exit_confirms_cleanup(-signal.SIGTERM) is False
    assert process_runner._supervisor_exit_confirms_cleanup(
        process_runner._SUPERVISOR_CLEANUP_FAILED_EXIT
    ) is False
    assert process_runner._supervisor_exit_confirms_cleanup(
        process_runner._SUPERVISOR_SETUP_FAILED_EXIT
    ) is False
    assert process_runner._supervisor_exit_confirms_cleanup(0) is False
    assert process_runner._supervisor_exit_confirms_cleanup(7) is False
    assert process_runner._supervisor_exit_confirms_cleanup(137) is False
    assert process_runner._supervisor_exit_confirms_cleanup(143) is False
    assert process_runner._supervisor_exit_confirms_cleanup(
        process_runner._SUPERVISOR_TIMEOUT_EXIT
    ) is True


def test_namespace_capability_probe_requires_a_private_procfs() -> None:
    source = inspect.getsource(process_runner._pid_namespace_prefix)

    assert '"--mount-proc"' in source
    assert "_PID_NAMESPACE_PROBE" in source
    assert "/proc/self/stat" in process_runner._PID_NAMESPACE_PROBE
    assert "/proc/1/stat" in process_runner._PID_NAMESPACE_PROBE


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or process_runner._pid_namespace_prefix() is None,
    reason="verified unprivileged Linux user/PID namespaces with private procfs are unavailable",
)
def test_verified_namespace_procfs_matches_namespace_pids() -> None:
    prefix = process_runner._pid_namespace_prefix()
    assert prefix is not None

    probe = (
        "import os; from pathlib import Path; "
        "a=os.getpid(); "
        "b=int(Path('/proc/self/stat').read_text().split(maxsplit=1)[0]); "
        "c=int(Path('/proc/1/stat').read_text().split(maxsplit=1)[0]); "
        "print(a,b,c)"
    )
    completed = subprocess.run(
        [*prefix, sys.executable, "-I", "-S", "-c", probe],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=5.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "1 1 1"


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or process_runner._pid_namespace_prefix() is None,
    reason="verified unprivileged Linux user/PID namespaces with private procfs are unavailable",
)
def test_managed_command_cannot_sigkill_the_namespace_init_supervisor(
    tmp_path: Path,
) -> None:
    script = tmp_path / "attack_supervisor.py"
    script.write_text(
        """
import os
from pathlib import Path
import re
import signal


def status(path):
    text = Path(path).read_text(encoding="utf-8")
    current = int(re.search(r"^Pid:\\s+(\\d+)", text, re.M).group(1))
    parent = int(re.search(r"^PPid:\\s+(\\d+)", text, re.M).group(1))
    return current, parent


_, parent = status("/proc/self/status")
supervisor = None
while parent > 0:
    try:
        cmdline = Path(f"/proc/{parent}/cmdline").read_bytes()
        if b"process_supervisor.py" in cmdline:
            supervisor = parent
            break
        _, parent = status(f"/proc/{parent}/status")
    except OSError:
        break

assert supervisor is not None
try:
    os.kill(supervisor, signal.SIGKILL)
except (ProcessLookupError, PermissionError):
    print("supervisor-kill-blocked", flush=True)
else:
    print("supervisor-kill-succeeded", flush=True)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    command = " ".join([shlex.quote(sys.executable), shlex.quote(str(script))])

    result = run_bounded_shell(command, timeout_seconds=5.0)

    assert result.timed_out is False
    assert result.returncode == 0
    assert "supervisor-kill-blocked" in result.stdout
    assert "supervisor-kill-succeeded" not in result.stdout


def test_namespace_wrapper_is_outside_the_standalone_supervisor() -> None:
    prefix = process_runner._pid_namespace_prefix()
    if prefix is None:
        pytest.skip("verified unprivileged Linux user/PID namespaces with private procfs are unavailable")

    command = process_runner.supervised_command_for_argv(["/bin/true"])

    assert tuple(command[: len(prefix)]) == prefix
    assert "--mount-proc" in command
    assert "--kill-child=KILL" in command
    assert any(Path(item).name == "process_supervisor.py" for item in command)
