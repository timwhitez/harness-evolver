"""Coupling complexity tracker.

From the paper:
  "耦合复杂度是一次更新必须同时照顾多少相互牵连的状态、
   规则、测试、反馈和历史。这个量不能按代码行数算。"

Coupling complexity measures how many harness components each edit
touches.  High coupling means the system is becoming harder to maintain
and needs compression (folding patches into simpler representations).

Key insights from the paper:
  - Good modularity cuts global coupling into local coupling
  - Good tests let the agent avoid simulating the whole system mentally
  - Memory and tools increase the agent's effective context
  - An HS that only grows without compressing will exceed maintenance capacity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from hl.types import HarnessPatch, TrialResult


@dataclass
class CouplingTracker:
    """Tracks coupling complexity across the Heuristic System lifecycle.

    Each patch is scored by how many components it touches.
    Each component has a dependency fan-in/fan-out count.
    When coupling exceeds threshold, compression is triggered.
    """

    # Per-component coupling stats
    _component_deps: dict[str, set[str]] = field(default_factory=dict)
    _patch_history: list[dict] = field(default_factory=list)
    _compression_events: list[dict] = field(default_factory=list)

    # Thresholds
    max_coupling_per_patch: int = 3
    patch_to_compression_ratio: float = 5.0

    def register_component(self, name: str, dependencies: list[str]) -> None:
        self._component_deps[name] = set(dependencies)

    def remove_component(self, name: str) -> None:
        self._component_deps.pop(name, None)
        for deps in self._component_deps.values():
            deps.discard(name)

    def touch_count(self, patch: HarnessPatch) -> int:
        """How many components does this patch directly touch?"""
        touched = {patch.component_name}
        for name, deps in self._component_deps.items():
            if patch.component_name in deps:
                touched.add(name)
        return len(touched)

    def record_patch(self, patch: HarnessPatch) -> int:
        """Record a patch and return its coupling score."""
        score = self.touch_count(patch)
        self._patch_history.append({
            "patch_id": f"{patch.component_name}@{patch.after_version}",
            "component": patch.component_name,
            "coupling_score": score,
            "timestamp": patch.timestamp.isoformat(),
        })
        return score

    def record_trial_failure(
        self, trial: TrialResult, affected_components: list[str]
    ) -> None:
        """Track which components were implicated in a failure."""
        for comp in affected_components:
            if comp not in self._component_deps:
                self._component_deps[comp] = set()

    def record_compression(self, components_merged: list[str], new_component: str) -> None:
        """Record a compression event (folding patches into simpler form)."""
        self._compression_events.append({
            "timestamp": datetime.now().isoformat(),
            "merged": components_merged,
            "into": new_component,
        })
        # Update dependency graph
        merged_deps: set[str] = set()
        for comp in components_merged:
            if comp in self._component_deps:
                merged_deps.update(self._component_deps[comp])
                del self._component_deps[comp]
        self._component_deps[new_component] = merged_deps

    @property
    def average_coupling(self) -> float:
        if not self._patch_history:
            return 0.0
        return sum(p["coupling_score"] for p in self._patch_history) / len(
            self._patch_history
        )

    @property
    def component_count(self) -> int:
        return len(self._component_deps)

    @property
    def compression_count(self) -> int:
        return len(self._compression_events)

    def needs_compression(self) -> bool:
        """Returns True if coupling is too high and compression is needed.

        Two signals:
          1. Average coupling per patch exceeds threshold
          2. Too many patches relative to component count
        """
        if self.component_count == 0:
            return False
        if self.average_coupling > self.max_coupling_per_patch:
            return True
        if len(self._patch_history) / self.component_count > self.patch_to_compression_ratio:
            return True
        return False

    def summary(self) -> dict:
        return {
            "component_count": self.component_count,
            "patch_count": len(self._patch_history),
            "average_coupling": round(self.average_coupling, 2),
            "compression_count": self.compression_count,
            "needs_compression": self.needs_compression(),
            "per_component_deps": {
                k: len(v) for k, v in self._component_deps.items()
            },
        }
