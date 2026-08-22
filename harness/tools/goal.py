"""Read-only campaign goal tool for the Worker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hl.goals import normalize_goal_status_payload
from harness.tools.base import ToolDef, ToolResult, ToolSchema


@dataclass
class GoalReadTool(ToolDef):
    name: str = "goal_read"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = (
        "Read task-scoped HL campaign context when the harness provides one. "
        "This is read-only outer-loop context; it is not TerminalBench task "
        "state and the Worker must not mark campaign goals complete."
    )
    goal_path: Path | None = None

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            description=self.description,
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, **_: Any) -> ToolResult:
        if self.goal_path is None:
            return ToolResult(
                success=True,
                output=(
                    "No task-scoped HL campaign goal was provided to this Worker. "
                    "Continue solving the current TerminalBench task; Harbor/verifier "
                    "decides pass/fail."
                ),
            )
        if not self.goal_path.exists():
            return ToolResult(
                success=True,
                output=(
                    "No active task-scoped HL campaign goal exists at the configured "
                    f"path: {self.goal_path}. Continue solving the current "
                    "TerminalBench task; Harbor/verifier decides pass/fail."
                ),
            )
        raw = self.goal_path.read_text(errors="replace")
        scoped_payload: dict[str, Any] = {
            "scope": "outer_hl_campaign_context_not_terminalbench_task_state",
            "worker_guidance": (
                "Use this only for prioritization context. Do not treat campaign "
                "status, score, token budget, or completion_reason as the current "
                "TerminalBench task status. Do not stop, mark todos complete, or "
                "declare failure because the outer campaign is stopped or budget "
                "exhausted; continue solving until local task evidence says the "
                "work is ready for Harbor verification."
            ),
        }
        try:
            campaign_goal = json.loads(raw)
            if isinstance(campaign_goal, dict):
                normalize_goal_status_payload(campaign_goal)
            scoped_payload["campaign_goal"] = campaign_goal
        except json.JSONDecodeError:
            scoped_payload["campaign_goal_text"] = raw
        return ToolResult(success=True, output=json.dumps(scoped_payload, indent=2))
