"""Progressive thinking policy — tiered reasoning effort guidance.

ForgeCode pattern: high budget early (planning), low budget later
(execution), re-elevated at verification/decision points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.planning.base import PlanningStrategy


@dataclass
class ProgressiveThinkingPolicy(PlanningStrategy):
    """Describe reasoning effort preferences based on task phase.

    - Planning phase: high thinking budget (deep analysis)
    - Execution phase: low thinking budget (fast coding)
    - Verification phase: elevated budget (thorough checking)
    - Decision points: re-elevated budget (architecture choices)
    """

    name: str = "progressive_thinking"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)

    # Historical field names are kept for provider config compatibility. These
    # numbers describe requested reasoning effort, not loop or token stop caps.
    planning_budget: int = 16000
    execution_budget: int = 4000
    verification_budget: int = 12000
    decision_budget: int = 16000

    def get_budget_for_phase(self, phase: str) -> int:
        budgets = {
            "planning": self.planning_budget,
            "execution": self.execution_budget,
            "verification": self.verification_budget,
            "decision": self.decision_budget,
        }
        return budgets.get(phase, self.execution_budget)

    def render(self, context: dict[str, Any]) -> str:
        return f"""## Reasoning Effort Policy

Requested reasoning effort varies by phase:
- **Planning**: {self.planning_budget} reference tokens — think deeply about approach
- **Execution**: {self.execution_budget} reference tokens — act quickly and decisively
- **Verification**: {self.verification_budget} reference tokens — check thoroughly
- **Decision Points**: {self.decision_budget} reference tokens — re-evaluate carefully

These values are provider/request hints and audit metadata, not Worker-loop stop
conditions and not reasons to abandon a task.
Keep routine execution concise.
Re-elevate at architectural decisions and verification checkpoints.
"""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.execution_budget > self.planning_budget:
            errors.append("execution budget should not exceed planning budget")
        return errors

    def raw_content(self) -> str:
        return self.render({})
