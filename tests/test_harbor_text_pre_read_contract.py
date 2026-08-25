from __future__ import annotations

import base64
from types import MethodType

from bench.harbor_adapter import HarborFileEditTool, HarborFileWriteTool
from harness.tools.base import ToolResult


def _guard(self: object, *args: object, **kwargs: object) -> tuple[str, None]:
    return "/workspace/target.txt", None


def _success_with_payload(payload: bytes) -> ToolResult:
    return ToolResult(
        success=True,
        output=base64.b64encode(payload).decode("ascii"),
        error="",
        metadata={"exit_code": 0},
    )


def test_append_rejects_invalid_utf8_before_atomic_publish() -> None:
    tool = object.__new__(HarborFileWriteTool)
    calls: list[tuple[str, dict[str, str]]] = []

    def run_secure(
        self: HarborFileWriteTool,
        script: str,
        *,
        env: dict[str, str],
    ) -> ToolResult:
        calls.append((script, env))
        if len(calls) == 1:
            return _success_with_payload(b"old-\xff-bytes")
        raise AssertionError("write phase must not run after text decode failure")

    tool._guard_environment_path = MethodType(_guard, tool)  # type: ignore[method-assign]
    tool._run_secure_python = MethodType(run_secure, tool)  # type: ignore[method-assign]

    result = tool.execute("/workspace/target.txt", "+tail", append=True)

    assert result.success is False
    assert result.output == ""
    assert result.metadata["text_decode_error"] is True
    assert result.metadata["atomic_replace"] is False
    assert len(calls) == 1


def test_edit_rejects_invalid_utf8_before_transform_or_publish() -> None:
    tool = object.__new__(HarborFileEditTool)
    calls: list[tuple[str, dict[str, str]]] = []

    def run_secure(
        self: HarborFileEditTool,
        script: str,
        *,
        env: dict[str, str],
    ) -> ToolResult:
        calls.append((script, env))
        return _success_with_payload(b"old-\xfe-bytes")

    tool._guard_environment_path = MethodType(_guard, tool)  # type: ignore[method-assign]
    tool._run_secure_python = MethodType(run_secure, tool)  # type: ignore[method-assign]

    result = tool.execute(
        "/workspace/target.txt",
        old_string="old",
        new_string="new",
    )

    assert result.success is False
    assert result.metadata["text_decode_error"] is True
    assert len(calls) == 1


def test_valid_utf8_append_builds_one_complete_atomic_payload() -> None:
    tool = object.__new__(HarborFileWriteTool)
    calls: list[tuple[str, dict[str, str]]] = []

    def run_secure(
        self: HarborFileWriteTool,
        script: str,
        *,
        env: dict[str, str],
    ) -> ToolResult:
        calls.append((script, env))
        if len(calls) == 1:
            return _success_with_payload("旧内容".encode("utf-8"))
        return ToolResult(
            success=True,
            output="write complete\n",
            error="",
            metadata={"exit_code": 0},
        )

    tool._guard_environment_path = MethodType(_guard, tool)  # type: ignore[method-assign]
    tool._run_secure_python = MethodType(run_secure, tool)  # type: ignore[method-assign]

    result = tool.execute("/workspace/target.txt", "+tail", append=True)

    assert result.success is True
    assert result.metadata["atomic_replace"] is True
    assert result.metadata["atomic_append"] is True
    assert len(calls) == 2
    published = base64.b64decode(calls[1][1]["HL_FILE_CONTENT"], validate=True)
    assert published.decode("utf-8") == "旧内容+tail"
