from __future__ import annotations

from bench.harbor_adapter import (
    HLWorkerHarborAgent,
    HarborGlobTool,
    HarborGrepTool,
)


def test_harbor_agent_registry_keeps_stacked_grep_and_glob_classes() -> None:
    agent = object.__new__(HLWorkerHarborAgent)
    agent.tool_timeout_seconds = 1.0
    agent._goal_path = lambda: None  # type: ignore[method-assign]

    registry = agent._build_environment_registry(object(), object())

    assert type(registry.get("grep")) is HarborGrepTool
    assert type(registry.get("glob")) is HarborGlobTool
    assert (
        HarborGlobTool._glob_without_python.__module__
        == "bench._harbor_glob_issue14_order_guard"
    )
