"""Dynamic skill loader — load specialized instruction sets by task profile.

ForgeCode pattern: skills are only loaded when the task domain/profile
requires them. This keeps the context window lean and focused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.skill_loading.base import SkillLoader


@dataclass
class DynamicSkillLoader(SkillLoader):
    name: str = "dynamic_skill_loader"
    version: str = "0.1.0"
