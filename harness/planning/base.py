"""Base planning strategy class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanningStrategy:
    """Base class for planning strategies.

    Planning strategies control how the agent:
    - Decomposes complex tasks into subtasks
    - Describes reasoning effort hints across phases
    - Enforces planning discipline (e.g., todo_write)
    """

    name: str
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)

    def render(self, context: dict[str, Any]) -> str:
        return ""

    def validate(self) -> list[str]:
        return []

    def raw_content(self) -> str:
        return ""
