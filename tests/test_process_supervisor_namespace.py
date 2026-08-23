"""Namespace containment tests plus graceful-timeout status propagation."""

from __future__ import annotations

import shlex
import sys

import pytest

from tests import _process_supervisor_namespace_issue19_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


@pytest.mark.skipif(
    not sys.platform.startswith("linux")
    or _base.process_runner._pid_namespace_prefix() is None,
    reason="verified unprivileged Linux user/PID namespaces are unavailable",
)
def test_namespace_timeout_signals_inner_supervisor_before_unshare() -> None:
    program = "import time; print('ready', flush=True); time.sleep(60)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"

    result = _base.run_bounded_shell(command, timeout_seconds=0.5)

    assert result.timed_out is True
    assert result.managed_process_group_terminated is True
    assert "ready" in result.stdout
