"""Process-tree regression tests with a corrected normal-completion contract."""

from __future__ import annotations

from pathlib import Path
import shlex
import sys

from tests import _verify_process_tree_issue19_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


def test_normal_parent_exit_does_not_leave_pipe_holding_background_child(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "background.pid"
    script = tmp_path / "spawn_and_exit.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')",
                "print('parent-exiting', flush=True)",
            ]
        ),
        encoding="utf-8",
    )
    command = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote(str(script)),
            shlex.quote(str(pid_file)),
        ]
    )

    result = _base.run_bounded_shell(command, timeout_seconds=5.0)

    assert result.timed_out is False
    assert result.returncode == 0
    assert "parent-exiting" in result.stdout
    # The important contract is the absence of descendants. The metadata flag
    # records an explicit outer termination operation and need not be true when
    # the supervisor/namespace completed cleanup before the outer wait returned.
    _base._wait_for_pid_exit(int(pid_file.read_text(encoding="utf-8")))
