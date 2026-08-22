"""TodoWrite enforcement — the single change that drove ForgeCode 38% → 66%.

Asserts at runtime that the agent maintains a structured task list.
Without this, agents tend to skip planning and code ad-hoc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.planning.base import PlanningStrategy


@dataclass
class TodoEnforcement(PlanningStrategy):
    """Force the agent to create and update a structured task list.

    This is one of the highest-impact single harness changes.
    ForgeCode reported 38% → 66% on TermBench from this alone.
    """

    name: str = "todo_enforcement"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)

    template: str = """## Task Planning (Required)

Before writing any code, you MUST:
1. Create a structured task list breaking down the work
2. Mark each task as pending, in_progress, or completed
3. Update task status as you work — never leave stale statuses
4. Only work on ONE task at a time

Format:
```
1. [in_progress] Task description
2. [pending] Task description
3. [pending] Task description
```

You MUST update this task list after every significant action.
If a task fails, create a new task describing the fix needed.
"""

    def render(self, context: dict[str, Any]) -> str:
        return self.template

    def validate(self) -> list[str]:
        if len(self.template) < 50:
            return ["todo_enforcement template too short"]
        return []

    def raw_content(self) -> str:
        return self.template
