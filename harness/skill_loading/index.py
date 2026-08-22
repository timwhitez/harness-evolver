"""Skill index — registry of available skills with triggers.

Maps task domains and patterns to available skill modules.
The meta-agent can add new skills as it discovers useful patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.skill_loading.base import SkillLoader


@dataclass
class SkillIndex(SkillLoader):
    name: str = "skill_index"
    version: str = "0.1.0"

    skill_registry: dict[str, str] = field(default_factory=dict)
