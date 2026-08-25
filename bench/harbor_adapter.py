"""Harbor adapter with canonical path authorization at every file boundary."""

from __future__ import annotations

from bench._harbor_adapter_issue4_base import *  # noqa: F401,F403
from bench import _harbor_adapter_issue4_base as _base
from bench._canonical_harbor_identity_guard import (
    HarborFileEditTool,
    HarborFileWriteTool,
)
from bench._harbor_adapter_issue13_audit import (
    HarborGrepTool,
    _HARBOR_GREP_COUNTING_PYTHON,
)
from bench._harbor_glob_fallback import (
    HarborGlobTool,
    _GLOB_FALLBACK_SCRIPT,
)
from bench._streaming_harbor_read import (
    HarborFileReadTool,
    _STREAMING_READ_PYTHON,
)


class HLWorkerHarborAgent(_base.HLWorkerHarborAgent):
    """Build a registry containing the canonical-path protected tool classes."""

    def _build_environment_registry(self, environment, loop):
        registry = _base.ToolRegistry()
        kwargs = {
            "environment": environment,
            "loop": loop,
            "timeout_seconds": float(self.tool_timeout_seconds),
        }
        todo_store = _base.TodoStore()
        for tool in [
            _base.HarborShellTool(**kwargs),
            HarborFileReadTool(**kwargs),
            HarborFileEditTool(**kwargs),
            HarborFileWriteTool(**kwargs),
            HarborGrepTool(**kwargs),
            HarborGlobTool(**kwargs),
            _base.TodoReadTool(store=todo_store),
            _base.TodoWriteTool(store=todo_store),
            _base.GoalReadTool(goal_path=self._goal_path()),
            _base.HarborVerifyTool(**kwargs),
            _base.DoneTool(),
        ]:
            registry.register(tool)
        return registry


__all__ = [
    "HLWorkerHarborAgent",
    "HarborFileReadTool",
    "HarborFileEditTool",
    "HarborFileWriteTool",
    "HarborGrepTool",
    "HarborGlobTool",
    "_STREAMING_READ_PYTHON",
    "_HARBOR_GREP_COUNTING_PYTHON",
    "_GLOB_FALLBACK_SCRIPT",
]
