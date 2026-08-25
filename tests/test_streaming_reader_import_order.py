from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_fresh_import(program: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_tools_package_does_not_mutate_safe_path_io() -> None:
    completed = _run_fresh_import(
        """
import harness.tools
import harness.tools.safe_path_io as safe
from harness.tools.file_read import FileReadTool
from harness.tools._streaming_file_read_impl import FileReadTool as Implementation

assert not hasattr(safe, "open_binary_nofollow")
assert FileReadTool is Implementation
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_streaming_implementation_reloads_without_injected_reader() -> None:
    completed = _run_fresh_import(
        """
import importlib
import harness.tools._streaming_file_read_impl as implementation
import harness.tools.safe_path_io as safe
from harness.tools.bounded_path_io import open_binary_nofollow

if hasattr(safe, "open_binary_nofollow"):
    del safe.open_binary_nofollow
reloaded = importlib.reload(implementation)

assert reloaded.open_binary_nofollow is open_binary_nofollow
assert reloaded.FileReadTool.__module__ == "harness.tools._streaming_file_read_impl"
"""
    )

    assert completed.returncode == 0, completed.stderr
