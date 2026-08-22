"""Base context management class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextManager:
    name: str
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    max_context_tokens: int = 200000

    def render(self, context: dict[str, Any]) -> str:
        return ""

    def validate(self) -> list[str]:
        return []

    def raw_content(self) -> str:
        return ""
