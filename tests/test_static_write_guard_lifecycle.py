"""Static policy assertions must not depend on a running event-loop thread."""

import pytest

from tests import test_roadmap_harbor as roadmap

STATIC_CASES = [
    roadmap.test_harbor_write_tool_blocks_staged_dependency_script_before_exec,
    roadmap.test_harbor_write_tool_blocks_oversized_gpt2_codegolf_before_exec,
    roadmap.test_harbor_write_tool_blocks_staged_nested_agent_script_before_exec,
]


@pytest.mark.parametrize("case", STATIC_CASES, ids=lambda case: case.__name__)
def test_static_guard_fixture_never_allocates_loop_or_thread(case, monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("content-only rejection fixture must not allocate an event loop or thread")

    monkeypatch.setattr(roadmap.asyncio, "new_event_loop", unexpected)
    monkeypatch.setattr(roadmap.threading, "Thread", unexpected)
    case(tmp_path)


def _without_static_prechecks(original):
    """Mutation keeps canonical/resolved checks and removes only early denials."""
    import ast
    import inspect
    import textwrap
    from types import CodeType, FunctionType

    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    early = [
        node
        for node in function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "append"
    ]
    assert len(early) == 1, "mutation must remove exactly the nonappend precheck block"
    function.body.remove(early[0])
    module_code = compile(tree, "<static-prechecks-removed>", "exec")
    function_code = next(
        code
        for code in module_code.co_consts
        if isinstance(code, CodeType) and code.co_name == function.name
    )
    return FunctionType(function_code, original.__globals__, argdefs=original.__defaults__)


@pytest.mark.parametrize("case", STATIC_CASES, ids=lambda case: case.__name__)
def test_static_guard_fixture_detects_removed_prechecks(case, monkeypatch, tmp_path):
    from bench.harbor_adapter import HarborFileWriteTool

    monkeypatch.setattr(
        HarborFileWriteTool, "execute", _without_static_prechecks(HarborFileWriteTool.execute)
    )
    with pytest.raises(AssertionError, match="fixture attempted environment.exec"):
        case(tmp_path)


def test_ordinary_write_reaches_environment_boundary_in_rejection_spy():
    from tests.static_guard_support import no_dispatch_write_tool

    tool, environment, loop = no_dispatch_write_tool()
    with pytest.raises(AssertionError, match="fixture attempted environment.exec"):
        tool.execute("/workspace/ordinary.txt", "ordinary text")
    environment.exec.assert_called_once()
    assert loop.mock_calls == []
    assert tool.timeout_seconds == 5


@pytest.mark.parametrize("valid_receipt", [True, False])
def test_ordinary_write_preserves_canonical_publication_order_and_receipt(valid_receipt):
    import base64
    from types import SimpleNamespace

    from tests.static_guard_support import no_dispatch_write_tool
    from tests.test_harbor_identity_publication import _response

    tool, environment, loop = no_dispatch_write_tool()
    calls = []
    path = "/workspace/ordinary.txt"
    content = "ordinary \u00e9\n"

    def synchronous_exec(command, *, timeout=None, env=None):
        calls.append(dict(env))
        if "HL_MUST_EXIST" in env:
            output = base64.b64encode(path.encode()).decode()
        else:
            assert "HL_FILE_CONTENT" in env, "unexpected environment operation"
            output = _response() if valid_receipt else "write complete\n"
        return SimpleNamespace(return_code=0, stdout=output, stderr="")

    # Only the environment response boundary is simulated; real public
    # canonical authorization, resolved policy and publication receipt run.
    tool._exec = synchronous_exec
    result = tool.execute(path, content)
    assert calls[0] == {"HL_FILE_PATH": path, "HL_MUST_EXIST": "0"}
    assert len(calls) == 2 and calls[1]["HL_FILE_PATH"] == path
    assert base64.b64decode(calls[1]["HL_FILE_CONTENT"]) == content.encode()
    assert result.success is valid_receipt
    assert result.metadata["publication_state"] == (
        "published" if valid_receipt else "indeterminate"
    )
    assert result.metadata["atomic_replace"] is (True if valid_receipt else None)
    assert result.metadata["no_auto_retry"] is True
    environment.exec.assert_not_called()
    assert loop.mock_calls == [] and tool.timeout_seconds == 5
