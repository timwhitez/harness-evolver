"""Public compatibility controls for the scoped validation-lint cleanup."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bench import _canonical_harbor_identity_guard as identity_guard
from bench import harbor, network_environment
from bench.harbor_adapter import HarborFileEditTool, HarborFileWriteTool
from harness.tools.base import ToolResult
from hl.types import TrialStatus
from tests.test_harbor_identity_publication import _outcome


@pytest.mark.parametrize("operation", ["write", "edit"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("directory_fsync", 1),
        ("directory_fsync", None),
        ("directory_fsync", "true"),
        ("publication_error", 0),
        ("publication_error", None),
        ("publication_error", []),
        (None, None),
    ],
)
def test_public_publication_receipts_keep_result_contract(monkeypatch, operation, field, value):
    outcome = _outcome()
    if field:
        outcome[field] = value
    response = "write complete\n__HL_PUBLICATION__" + json.dumps(outcome) + "\n"
    calls = []

    def receipt(*args, **kwargs):
        calls.append((args, kwargs))
        return ToolResult(True, response)

    if operation == "write":
        tool = object.__new__(HarborFileWriteTool)
        tool._guard_environment_path = lambda path, **_: (path, None)
        tool._run_secure_python = receipt
        result = tool.execute("/workspace/target.txt", "content\n")
    else:
        monkeypatch.setattr(identity_guard._base.HarborFileEditTool, "execute", receipt)
        result = object.__new__(HarborFileEditTool).execute("/workspace/target.txt", "old", "new")

    assert len(calls) == 1
    if field:
        assert not result.success
        assert "Secure publication outcome unknown; reconcile before retry" in result.error
        assert result.metadata["secure_write_protocol_error"] is True
        assert result.metadata["publication_state"] == "indeterminate"
        assert result.metadata["atomic_replace"] is None
        assert result.metadata["no_auto_retry"] is True
    else:
        assert result.success and result.error == ""
        assert result.output == "write complete"
        assert result.metadata["publication_state"] == "published"
        assert result.metadata["atomic_replace"] is True
        assert result.metadata["no_auto_retry"] is True


def test_public_network_default_and_identity_tool_exports_remain_available():
    assert harbor.DEFAULT_DOCKER_LABELS is network_environment.DEFAULT_DOCKER_LABELS
    assert set(identity_guard.__all__) == {
        "_SECURE_SNAPSHOT",
        "HarborFileEditTool",
        "HarborFileReadTool",
        "HarborFileWriteTool",
        "ToolResult",
    }


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_runner_keeps_unknown_error_result_and_process_exit_boundary(monkeypatch, tmp_path, error_type):
    runner = harbor.HarborRunner()
    command = SimpleNamespace(argv=["mock-harbor"], job_name="fixture", job_dir=tmp_path)
    monkeypatch.setattr(runner, "build_command", lambda *args, **kwargs: command)

    def fail(*args, **kwargs):
        raise error_type("fixture failure")

    monkeypatch.setattr(runner, "_run_command", fail)
    kwargs = {
        "task_id": "fixture",
        "agent_config": {},
        "job_name": None,
        "jobs_dir": tmp_path,
        "timeout_audit": 1,
    }
    if error_type is RuntimeError:
        result = runner._run_task_once(**kwargs)
        assert result.status == TrialStatus.ERROR
        assert result.error_log == ["fixture failure"]
        assert not result.verified and result.score == 0
    else:
        with pytest.raises(error_type, match="fixture failure"):
            runner._run_task_once(**kwargs)
