"""Context management — control what the agent sees in its context window.

Key patterns:
- Compaction pipeline: 5-stage pre-model compaction (Claude Code)
- Context isolation: sub-agent windows don't pollute parent (Claude Code)
- Progressive disclosure: most relevant info first
"""

from harness.context.base import ContextManager
from harness.context.compaction import CompactionStrategy
from harness.context.isolation import ContextIsolation
from harness.context.trajectory_pack import TrajectoryPack

__all__ = ["ContextManager", "CompactionStrategy", "ContextIsolation", "TrajectoryPack"]
