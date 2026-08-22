"""Base verification strategy class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationStrategy:
    name: str
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)

    def render(self, context: dict[str, Any]) -> str:
        return ""

    def validate(self) -> list[str]:
        return []

    def raw_content(self) -> str:
        return ""
