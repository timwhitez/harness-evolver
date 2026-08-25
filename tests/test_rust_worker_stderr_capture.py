from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import threading
import time

import pytest

from bench.agent import HLAgent, _BoundedStderrTail
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


def test_stderr_capture_memory_is_independent_of_total_diagnostics() -> None:
    capture = _BoundedStderrTail(4096)

    for _ in range(64):
        capture.feed(b"x" * 65_536)
    capture.feed(b"TAIL-MARKER")

    assert capture.total_bytes > 4_000_000
    assert len(capture.tail) <= 4096
    assert "stderr bytes omitted" in capture.text()
    assert capture.text().endswith("TAIL-MARKER")


@pytest.mark.skipif(os.name != "posix", reason="uses executable test script")
def test_large_stderr_cannot_block_jsonl_final_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _final_payload("stderr-flood")
    worker = _worker_script(
        tmp_path / "worker.py",
        "\n".join(
            [
                "import sys",
                "sys.stdin.readline()",
                "sys.stderr.write('diagnostic-' + 'x' * 2_000_000)",
                "sys.stderr.flush()",
                f"print({final!r}, flush=True)",
            ]
        )
        + "\n",
    )
    monkeypatch.setenv("HL_WORKER_RUST_BIN", str(worker))

    started = time.monotonic()
    result = HLAgent(rust_stderr_tail_bytes=4096).run(
        "test",
        {"task_id": "stderr-flood"},
    )

    assert time.monotonic() - started < 10
    assert result.status == TrialStatus.FAILED
    assert result.metadata.get("rust_worker_core_error") is not True


@pytest.mark.skipif(os.name != "posix", reason="uses executable test script")
def test_failure_reports_only_bounded_stderr_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "TAIL-DIAGNOSTIC-MARKER"
    worker = _worker_script(
        tmp_path / "worker.py",
        "\n".join(
            [
                "import sys",
                "sys.stdin.readline()",
                f"sys.stderr.write('prefix-' + 'x' * 2_000_000 + {marker!r})",
                "sys.stderr.flush()",
                "raise SystemExit(7)",
            ]
        )
        + "\n",
    )
    monkeypatch.setenv("HL_WORKER_RUST_BIN", str(worker))

    result = HLAgent(rust_stderr_tail_bytes=4096).run(
        "test",
        {"task_id": "stderr-error"},
    )

    assert result.status == TrialStatus.ERROR
    error = result.error_log[0]
    assert marker in error
    assert "stderr bytes omitted" in error
    assert len(error) < 5000


@pytest.mark.skipif(os.name != "posix", reason="uses process-group cancellation")
def test_cancellation_completes_after_stderr_flood(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready = tmp_path / "ready"
    worker = _worker_script(
        tmp_path / "worker.py",
        "\n".join(
            [
                "import os, pathlib, sys, time",
                "sys.stdin.readline()",
                "sys.stderr.write('x' * 2_000_000)",
                "sys.stderr.flush()",
                "pathlib.Path(os.environ['READY_PATH']).write_text('ready')",
                "time.sleep(60)",
            ]
        )
        + "\n",
    )
    monkeypatch.setenv("HL_WORKER_RUST_BIN", str(worker))
    monkeypatch.setenv("READY_PATH", str(ready))
    agent = HLAgent(rust_stderr_tail_bytes=4096)
    holder: dict[str, object] = {}

    thread = threading.Thread(
        target=lambda: holder.setdefault(
            "result",
            agent.run("test", {"task_id": "stderr-cancel"}),
        )
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists(), "worker did not pass the stderr flood"

    agent.cancel_current_run("test cancellation")
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert holder["result"].status == TrialStatus.ERROR
