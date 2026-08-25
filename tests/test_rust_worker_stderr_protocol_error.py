from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from bench.agent import HLAgent
from hl.types import TrialStatus


def _worker_script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.mark.skipif(os.name != "posix", reason="uses executable test script")
def test_malformed_protocol_reports_bounded_stderr_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "MALFORMED-PROTOCOL-TAIL"
    worker = _worker_script(
        tmp_path / "worker.py",
        "\n".join(
            [
                "import sys",
                "sys.stdin.readline()",
                f"sys.stderr.write('prefix-' + 'x' * 2_000_000 + {marker!r})",
                "sys.stderr.flush()",
                "print('{malformed-json', flush=True)",
            ]
        )
        + "\n",
    )
    monkeypatch.setenv("HL_WORKER_RUST_BIN", str(worker))

    result = HLAgent(rust_stderr_tail_bytes=4096).run(
        "test",
        {"task_id": "stderr-malformed-protocol"},
    )

    assert result.status == TrialStatus.ERROR
    error = result.error_log[0]
    assert marker in error
    assert "stderr bytes omitted" in error
    assert len(error) < 5000
