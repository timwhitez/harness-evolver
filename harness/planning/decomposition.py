"""Task decomposition — split complex tasks into parallel subtasks.

Factory Missions pattern: orchestrator decomposes into milestones,
each milestone becomes an independent worker task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.planning.base import PlanningStrategy


@dataclass
class TaskDecomposition(PlanningStrategy):
    """Decompose complex tasks into independent subtasks.

    For TerminalBench, this means identifying when a task has
    independent components that can be solved separately and
    then integrated.
    """

    name: str = "task_decomposition"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)

    # Compatibility name: this caps one planning decomposition prompt, not the
    # task-solving loop or the number of recovery passes the Worker may take.
    max_subtasks: int = 5
    parallel_subtasks: bool = False  # Serial-first policy

    def render(self, context: dict[str, Any]) -> str:
        return f"""## Task Decomposition

For complex tasks, start with up to {self.max_subtasks} visible subtasks in one plan.
Each subtask should be independently verifiable.
Complete subtasks {"sequentially" if not self.parallel_subtasks else "in parallel where possible"}.
Integrate and verify the full solution before declaring completion.
This planning cap is not a loop stop condition; if evidence changes, rewrite the
todo list and continue with the next concrete task-solving step.
"""

    def validate(self) -> list[str]:
        return []

    def raw_content(self) -> str:
        return self.render({})
