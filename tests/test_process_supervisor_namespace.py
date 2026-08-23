"""Namespace containment tests plus graceful-timeout status propagation."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shlex
import sys

import pytest

_BASE_PATH = Path(__file__).with_name("_process_supervisor_namespace_issue19_base.py")
_SPEC = spec_from_file_location(
    "_harness_process_supervisor_namespace_issue19_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load namespace test base: {_BASE_PATH}")
_base = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _base
_SPEC.loader.exec_module(_base)

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
