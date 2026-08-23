"""Public facade for the namespace-hardened bounded process runner.

All lifecycle decisions live in
:mod:`harness.tools._process_runner_issue19_namespace_base`. Keeping one
implementation avoids a second, weaker supervisor-discovery path from
silently overriding the verified private-procfs and PID-start-time checks.
"""

from __future__ import annotations

from harness.tools import _process_runner_issue19_namespace_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_runtime = _base._base
_runtime.supervised_command_for_argv = _base.supervised_command_for_argv
_runtime._terminate_supervised_process = _base._terminate_supervised_process

run_bounded_shell = _runtime.run_bounded_shell
_terminate_process_tree = _runtime._terminate_process_tree
supervised_command_for_argv = _base.supervised_command_for_argv
_pid_namespace_prefix = _base._pid_namespace_prefix
_supervisor_exit_confirms_cleanup = _base._supervisor_exit_confirms_cleanup
