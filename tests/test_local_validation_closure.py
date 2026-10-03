"""Offline controls for the local-validation contract repair."""

from __future__ import annotations

import json

import pytest

from bench._canonical_harbor_identity_guard import _Snapshot
from bench.harbor import HarborRunner
from bench.harbor_adapter import HarborFileWriteTool
from harness.tools.base import ToolResult
from tests.test_harbor_identity_publication import _response
from tests.worker_contract_support import assert_worker_waits_are_cleanup_only


@pytest.mark.parametrize("top", [[], None, 7, "unexpected"])
@pytest.mark.parametrize("attempts", [0, 1, 2])
def test_non_object_top_level_is_conservative_or_uses_existing_multi_attempt_contract(
    tmp_path, top, attempts
):
    (tmp_path / "result.json").write_text(json.dumps(top))
    for index in range(attempts):
        trial = tmp_path / f"fixture__{index}"
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "fixture",
                    "trial_name": trial.name,
                    "agent_result": {
                        "metadata": {
                            "turn_count": 3,
                            "worker_metrics_observation": {
                                "schema": "worker_metrics_v1",
                                "status": "partial",
                            },
                        }
                    },
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            )
        )
    result = HarborRunner().parse_job_dir(tmp_path, task_id="fixture")
    if attempts < 2:
        assert result.status.value == "error" and result.score == 0 and not result.verified
        assert any("object" in message for message in result.error_log)
        assert result.turn_count == (3 if attempts else None)
    else:
        # Existing recovery accepts independently written Harbor verifier evidence.
        assert result.verified and result.score == 1 and result.turn_count == 6
        assert result.metadata["attempt_count"] == 2


@pytest.mark.parametrize(
    "identities",
    [
        [{"task_name": "foreign"}],
        [{"task_name": "fixture", "task_id": {"path": "/tasks/foreign"}}],
        [
            {"task_name": "fixture", "task_id": {"path": "/a/fixture"}},
            {"task_name": "fixture", "task_id": {"path": "/b/fixture"}},
        ],
    ],
)
def test_non_object_top_level_never_borrows_ambiguous_or_foreign_evidence(tmp_path, identities):
    (tmp_path / "result.json").write_text("[]")
    for index, identity in enumerate(identities):
        trial = tmp_path / f"attempt-{index}"
        trial.mkdir()
        (trial / "result.json").write_text(
            json.dumps(
                {
                    **identity,
                    "agent_result": {"metadata": {"turn_count": 9}},
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            )
        )
    result = HarborRunner().parse_job_dir(tmp_path, task_id="fixture")
    assert result.status.value == "error" and not result.verified and result.score == 0
    assert result.turn_count is None


@pytest.mark.parametrize(
    "path,content,guard",
    [
        (
            "/workspace/download_httpstan.py",
            "import urllib.request\nurllib.request.urlopen('https://pypi.org/pypi/httpstan/4.13.0/json')\n",
            "staged_dependency_script_guard",
        ),
        (
            "/workspace/delegate.js",
            "import {spawnSync} from 'child_process'; spawnSync('codex', ['exec', 'fix'])\n",
            "staged_dependency_script_guard",
        ),
        ("/workspace/app/gpt2.c", "x" * 5000, "deliverable_size_cap_write_guard"),
    ],
    ids=["staged-dependency", "nested-agent", "size-cap"],
)
def test_static_write_rejections_do_not_touch_environment(path, content, guard):
    tool = object.__new__(HarborFileWriteTool)

    def unexpected(*args, **kwargs):
        pytest.fail("content-only rejection must happen before environment execution")

    tool._guard_environment_path = unexpected
    tool._run_secure_python = unexpected
    result = tool.execute(path, content)
    assert not result.success and result.metadata["blocked_by"] == guard
    assert result.metadata["loop_stop_condition"] is False


@pytest.mark.parametrize("authorized", [False, True])
def test_allowed_write_still_requires_canonical_authorization_and_publication_receipt(authorized):
    tool = object.__new__(HarborFileWriteTool)
    calls = []
    denied = ToolResult(
        False, "", error="fixture canonical denial", metadata={"blocked_by": "canonical_path_guard"}
    )

    def guard(path, **kwargs):
        calls.append(("guard", path, kwargs))
        return path, None if authorized else denied

    def publish(script, *, env):
        calls.append(("publish", env))
        return ToolResult(True, _response(), metadata={"exit_code": 0})

    tool._guard_environment_path = guard
    tool._run_secure_python = publish
    result = tool.execute("/workspace/ordinary.txt", "ordinary text")
    assert calls[0][0] == "guard"
    assert len(calls) == (2 if authorized else 1)
    assert result.success is authorized
    if authorized:
        assert result.metadata["publication_state"] == "published"
    else:
        assert result is denied


def test_nonappend_alias_keeps_resolved_size_guard():
    tool = object.__new__(HarborFileWriteTool)
    calls = []

    def guard(path, **kwargs):
        calls.append(path)
        return "/workspace/app/gpt2.c", None

    tool._guard_environment_path = guard
    tool._run_secure_python = lambda *args, **kwargs: pytest.fail(
        "resolved oversized write must not publish"
    )
    result = tool.execute("/workspace/ordinary-alias", "x" * 5000)
    assert calls == ["/workspace/ordinary-alias"]
    assert not result.success
    assert result.metadata["blocked_by"] == "deliverable_size_cap_write_guard"
    assert result.metadata["path"] == "/workspace/app/gpt2.c"
    assert result.metadata["content_bytes"] == 5000


def test_append_size_guard_uses_canonical_snapshot_and_composed_content():
    tool = object.__new__(HarborFileWriteTool)
    calls = []

    def guard(path, **kwargs):
        calls.append("guard")
        return "/workspace/app/gpt2.c", None

    def snapshot(path):
        calls.append("snapshot")
        return _Snapshot(True, "x" * 4990, 1, 2, "fixture"), None

    tool._guard_environment_path = guard
    tool._secure_snapshot = snapshot
    tool._run_secure_python = lambda *args, **kwargs: pytest.fail(
        "oversized append must not publish"
    )
    result = tool.execute("/workspace/alias", "x" * 10, append=True)
    assert calls == ["guard", "snapshot"]
    assert result.metadata["blocked_by"] == "deliverable_size_cap_write_guard"
    assert result.metadata["content_bytes"] == 5000


@pytest.mark.parametrize(
    "source",
    [
        "def _run_rust_core():\n    process.wait(timeout=1)\n",
        "def _run_rust_core():\n    process.wait(1)\n",
        "def _run_rust_core():\n    process.communicate(None, 1)\n",
        "def _run_rust_core():\n    while True:\n        self._terminate_process(process)\n",
    ],
)
def test_worker_cleanup_scope_check_rejects_task_loop_limits(source):
    with pytest.raises(AssertionError):
        assert_worker_waits_are_cleanup_only(source)


def test_worker_cleanup_scope_check_allows_only_error_and_cancel_cleanup():
    assert_worker_waits_are_cleanup_only("""
def _terminate_process(process):
    process.wait(timeout=.5)
def cancel_current_run():
    self._terminate_process(process)
def _run_rust_core():
    try:
        while True:
            request()
    except Exception:
        self._terminate_process(process)
    finally:
        self._terminate_process(process)
""")
