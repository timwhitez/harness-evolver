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


def test_streaming_implementation_imports_before_public_facade() -> None:
    completed = _run_fresh_import(
        "from harness.tools._streaming_file_read_impl import FileReadTool; "
        "assert FileReadTool.__module__ == 'harness.tools._streaming_file_read_impl'"
    )

    assert completed.returncode == 0, completed.stderr


def test_public_facade_does_not_mutate_safe_path_io() -> None:
    completed = _run_fresh_import(
        "import harness.tools.safe_path_io as safe; "
        "assert not hasattr(safe, 'open_binary_nofollow'); "
        "from harness.tools.file_read import FileReadTool; "
        "assert not hasattr(safe, 'open_binary_nofollow'); "
        "from harness.tools._streaming_file_read_impl import FileReadTool as Impl; "
        "assert FileReadTool is Impl"
    )

    assert completed.returncode == 0, completed.stderr
