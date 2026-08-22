"""Dynamic skill loading — load specialized instructions by task profile.

ForgeCode pattern: skills loaded only when task profile requires them,
keeping context lean.
"""

from harness.skill_loading.base import SkillLoader
from harness.skill_loading.loader import DynamicSkillLoader
from harness.skill_loading.index import SkillIndex

__all__ = ["SkillLoader", "DynamicSkillLoader", "SkillIndex"]
