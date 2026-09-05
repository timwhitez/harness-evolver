from __future__ import annotations

import sys

import pytest

from harness.tools import process_runner as runner


pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux supervisor fixture")


def test_supervisor_capability_setup_does_not_consume_execution_timeout(monkeypatch):
    import time
    monkeypatch.setattr(runner._runtime, '_is_linux_subreaper_available', lambda: True)
    monkeypatch.setattr(runner._base, '_pid_namespace_prefix', lambda: None)
    original = runner._prepare_supervised_launch
    def slow_setup(argv):
        time.sleep(1.5)
        return original(argv)
    monkeypatch.setattr(runner, '_prepare_supervised_launch', slow_setup)
    result = runner.run_bounded_argv([sys.executable, '-c', 'print("ready")'], timeout_seconds=1.0)
    assert result.returncode == 0 and not result.timed_out
    assert result.stdout == 'ready\n'
