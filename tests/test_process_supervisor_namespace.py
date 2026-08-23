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


def test_namespace_timeout_signals_inner_supervisor_not_outer_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _base.process_runner._base
    calls: list[str] = []

    class FakeProcess:
        pid = 4242
        args = ["unshare", "--pid", "--", "python", "process_supervisor.py"]
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, sig):
            raise AssertionError(f"outer wrapper received signal {sig}")

        def wait(self, timeout=None):
            calls.append(f"wait:{timeout}")
            self.returncode = namespace._base._SUPERVISOR_TIMEOUT_EXIT
            return self.returncode

    monkeypatch.setattr(namespace, "_process_uses_pid_namespace", lambda process: True)
    monkeypatch.setattr(
        namespace,
        "_signal_namespace_supervisor",
        lambda process: calls.append("inner-term") or True,
    )

    assert namespace._terminate_supervised_process(FakeProcess()) is True
    assert calls[0] == "inner-term"
    assert calls[1].startswith("wait:")


def test_namespace_supervisor_selection_prefers_unique_closest_ancestor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _base.process_runner._base
    script = str(Path(namespace.__file__).with_name("process_supervisor.py").resolve()).encode()
    table = {
        100: namespace._HostProcessIdentity(100, 1, 10, (b"unshare",)),
        101: namespace._HostProcessIdentity(
            101,
            100,
            11,
            (b"python", script, b"--payload", b"real"),
        ),
        102: namespace._HostProcessIdentity(102, 101, 12, (b"worker",)),
        103: namespace._HostProcessIdentity(
            103,
            102,
            13,
            (b"python", script, b"--payload", b"decoy"),
        ),
    }
    monkeypatch.setattr(namespace, "_host_process_table", lambda: table)

    selected = namespace._namespace_supervisor_identity(100)

    assert selected is not None
    assert selected.pid == 101
    assert selected.start_time == 11


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
