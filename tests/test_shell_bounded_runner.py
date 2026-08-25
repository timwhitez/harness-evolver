from __future__ import annotations

import shlex
import sys

import pytest

from harness.tools.shell import ShellTool
from harness.tools.verify import VerifyTool


def test_shell_and_verify_reject_explicit_zero_timeout() -> None:
    shell = ShellTool().execute("printf ok", timeout=0)
    verify = VerifyTool().execute("printf ok", timeout=0)

    assert shell.success is False
    assert verify.success is False
    assert "timeout_seconds must be finite and > 0" in shell.error
    assert verify.error == shell.error


@pytest.mark.parametrize("tool", [ShellTool, VerifyTool])
def test_shell_execution_bounds_stdout_and_stderr_before_return(tool: type) -> None:
    program = (
        "import sys; "
        "sys.stdout.write('o' * 200000); "
        "sys.stderr.write('e' * 200000)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"

    result = tool(max_output_chars=4096).execute(command, timeout=5)

    assert result.success is True
    assert len(result.output) <= 4096
    assert result.metadata["output_bounded"] is True


def test_verify_keeps_shell_timeout_cleanup_metadata() -> None:
    program = "import time; print('ready', flush=True); time.sleep(60)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"

    shell = ShellTool(timeout_seconds=0.25).execute(command)
    verify = VerifyTool(timeout_seconds=0.25).execute(command)

    for result in (shell, verify):
        assert result.success is False
        assert result.metadata["timed_out"] is True
        assert result.metadata["output_bounded"] is True
        assert "ready" in result.output
        assert "process_tree_terminated" in result.metadata
