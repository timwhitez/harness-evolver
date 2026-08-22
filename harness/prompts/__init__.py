"""Prompt templates — the core Policy components that shape agent behavior.

Inspired by Factory Droid's three-tier prompting:
  System prompt  → Role, constraints, environment
  Tool prompt    → Tool descriptions (kept separate to avoid attention dilution)
  Notification   → Dynamic guidance injected during execution
"""

from harness.prompts.base import PromptTemplate
from harness.prompts.system import SystemPrompt, ThreeTierPromptSystem
from harness.prompts.task import TaskPrompt
from harness.prompts.recovery import RecoveryPrompt

__all__ = [
    "PromptTemplate",
    "SystemPrompt",
    "ThreeTierPromptSystem",
    "TaskPrompt",
    "RecoveryPrompt",
]
