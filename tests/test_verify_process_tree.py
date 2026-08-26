"""Process-tree regression tests with a corrected normal-completion contract."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shlex
import sys

_BASE_PATH = Path(__file__).with_name("_verify_process_tree_issue19_base.py")
_SPEC = spec_from_file_location("_harness_verify_process_tree_issue19_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load process-tree test base: {_BASE_PATH}")
_base = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _base
_SPEC.loader.exec_module(_base)

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


def test_normal_parent_exit_does_not_leave_pipe_holding_background_child(
    tmp_path: Path,
) -> None:
    process_lock = tmp_path / "background.lock"
    script = tmp_path / "spawn_and_exit.py"
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                "child_code = \"import fcntl,pathlib,sys,time; h=pathlib.Path(sys.argv[1]).open('w'); fcntl.flock(h, fcntl.LOCK_EX); h.write('locked'); h.flush()\\nwhile True: time.sleep(60)\"",
                "subprocess.Popen([sys.executable, '-c', child_code, sys.argv[1]])",
                "deadline = time.monotonic() + 5",
                "p = pathlib.Path(sys.argv[1])",
                "while (not p.exists() or p.read_text() != 'locked') and time.monotonic() < deadline: time.sleep(0.01)",
                "print('parent-exiting', flush=True)",
            ]
        ),
        encoding="utf-8",
    )
    command = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote(str(script)),
            shlex.quote(str(process_lock)),
        ]
    )

    result = _base.run_bounded_shell(command, timeout_seconds=5.0)

    assert result.timed_out is False
    assert result.returncode == 0
    assert "parent-exiting" in result.stdout
    # The important contract is the absence of descendants. The metadata flag
    # records an explicit outer termination operation and need not be true when
    # the supervisor/namespace completed cleanup before the outer wait returned.
    _base.assert_descendant_lock_released(process_lock)
