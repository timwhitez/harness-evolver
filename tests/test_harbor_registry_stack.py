from __future__ import annotations

from bench.harbor_adapter import (
    HLWorkerHarborAgent,
    HarborFileReadTool,
    HarborGlobTool,
    HarborGrepTool,
)


def test_harbor_agent_registry_uses_the_integrated_file_search_classes() -> None:
    agent = object.__new__(HLWorkerHarborAgent)
    agent.tool_timeout_seconds = 1.0
    agent._goal_path = lambda: None  # type: ignore[method-assign]

    registry = agent._build_environment_registry(object(), object())

    assert type(registry.get("read")) is HarborFileReadTool
    assert type(registry.get("grep")) is HarborGrepTool
    assert type(registry.get("glob")) is HarborGlobTool
    assert registry.get("grep").max_results == 200
    assert registry.get("grep").max_match_chars == 4000
