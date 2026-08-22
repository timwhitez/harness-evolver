"""Compaction strategy — 5-layer pipeline (Claude Code pattern).

Budget Reduction → Snip → Microcompact → Context Collapse → Auto-Compact
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.context.base import ContextManager


@dataclass
class CompactionStrategy(ContextManager):
    name: str = "compaction"
    version: str = "0.1.0"

    # Compaction thresholds (percentage of max context)
    budget_reduction_pct: float = 0.80
    snip_pct: float = 0.85
    microcompact_pct: float = 0.90
    collapse_pct: float = 0.95
    auto_compact_pct: float = 0.98
