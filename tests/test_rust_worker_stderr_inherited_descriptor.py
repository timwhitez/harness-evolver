from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import threading
import time

import pytest

from bench.agent import HLAgent
from hl.types import TrialStatus


def _worker_script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _final_payload(task_id: str) -> str:
    return json.dumps(
        {
            "type": "final",
            "result": {
                "trial_id": task_id,
                "task_id": task_id,
                "status": "failed",
                "score": 0.0,
                "verified": False,
                "tool_calls": [],
                "trajectory": [],
                "token_usage": {},
                "error_log": [],
                "metadata": {},
            },
        }
    )


@pytest.mark.skipif(os.name != "posix", reason="uses fork and POSIX signals")
def test_worker_exit_does_not_wait_for_descendant_to_close_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descendant_pid_path = tmp_path / "descendant.pid"
    final = _final_payload("inherited-stderr")
    worker = _worker_script(
        tmp_path / "worker.py",
        "\n".join(
            [
                "import os, pathlib, sys, time",
                "sys.stdin.readline()",
                "child = os.fork()",
                "if child == 0:",
                "    time.sleep(30)",
                "    os._exit(0)",
                f"pathlib.Path({str(descendant_pid_path)!r}).write_text(str(child))",
                "sys.stderr.write('worker finished while descendant keeps fd 2\\n')",
                "sys.stderr.flush()",
                f"print({final!r}, flush=True)",
            ]
        )
        + "\n",
    )
    monkeypatch.setenv("HL_WORKER_RUST_BIN", str(worker))

    holder: dict[str, object] = {}
    thread = threading.Thread(
        target=lambda: holder.setdefault(
            "result",
            HLAgent(rust_stderr_tail_bytes=4096).run(
                "test",
                {"task_id": "inherited-stderr"},
            ),
        )
    )
    thread.start()

    deadline = time.monotonic() + 5
    while not descendant_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert descendant_pid_path.exists(), "worker did not create its descendant"

    thread.join(timeout=3)
    completed_without_descendant_eof = not thread.is_alive()

    descendant_pid = int(descendant_pid_path.read_text())
    try:
        os.kill(descendant_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    thread.join(timeout=5)

    assert completed_without_descendant_eof, (
        "bridge remained blocked until the inherited stderr descriptor reached EOF"
    )
    assert not thread.is_alive()
    assert holder["result"].status == TrialStatus.FAILED
