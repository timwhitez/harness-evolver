"""HeuristicSystem — the central registry that wires everything together.

A Heuristic System (HS) is more than an isolated policy.py.
It contains at minimum:
  - Programmatic policy (harness components)
  - State representation (environment awareness)
  - Feedback channels (multiple signal extractors)
  - Experiment records (trials, summaries)
  - Memory (regression snapshots, patches, diffs)
  - Update mechanism (meta-coding-agent)

Rules, feedback, history, and the next update path all need to
connect before it becomes an HS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hl.protocol import FeedbackChannel, MemoryStore, Policy, StateProvider, UpdateEngine


@dataclass
class HeuristicSystem:
    """Central registry for all HL components.

    This is the "living system" — not a static config but a
    versioned, evolving collection of interconnected components.
    Each component is independently editable and testable.

    Two required operations for a healthy HS:
      1. Absorb feedback: write new failures, logs, rewards into the system
      2. Compress history: fold local patches into simpler representations
    """

    name: str = "harness-evolver"
    version: str = "0.1.0"

    # ── Registered components ──
    policies: dict[str, Policy] = field(default_factory=dict)
    state_providers: dict[str, StateProvider] = field(default_factory=dict)
    feedback_channels: dict[str, FeedbackChannel] = field(default_factory=dict)
    memory: MemoryStore | None = None
    update_engine: UpdateEngine | None = None

    # ── Metadata ──
    trial_count: int = 0
    patch_count: int = 0
    solved_tasks: set[str] = field(default_factory=set)
    failed_directions: dict[str, list[str]] = field(default_factory=dict)

    def register_policy(self, policy: Policy) -> None:
        self.policies[policy.name] = policy

    def register_state_provider(self, provider: StateProvider) -> None:
        self.state_providers[provider.name] = provider

    def register_feedback_channel(self, channel: FeedbackChannel) -> None:
        self.feedback_channels[channel.name] = channel

    def set_memory(self, memory: MemoryStore) -> None:
        self.memory = memory

    def set_update_engine(self, engine: UpdateEngine) -> None:
        self.update_engine = engine

    def get_policy(self, name: str) -> Policy | None:
        return self.policies.get(name)

    def get_active_policies(self) -> list[Policy]:
        """Return all registered policies in dependency order."""
        return sorted(
            self.policies.values(),
            key=lambda p: len(p.dependencies),
        )

    def render_context(self, task_context: dict[str, Any]) -> str:
        """Render all active policies into the agent context window."""
        parts: list[str] = []
        for policy in self.get_active_policies():
            rendered = policy.render(task_context)
            if rendered:
                parts.append(rendered)
        return "\n\n".join(parts)

    def record_solved(self, task_id: str) -> None:
        self.solved_tasks.add(task_id)

    def record_failed_direction(self, task_id: str, direction: str) -> None:
        if task_id not in self.failed_directions:
            self.failed_directions[task_id] = []
        self.failed_directions[task_id].append(direction)

    def is_solved(self, task_id: str) -> bool:
        return task_id in self.solved_tasks

    def needs_compression(self, coupling_threshold: int = 5) -> bool:
        """Heuristic: if patch count >> component count, compression needed.

        A healthy HS periodically folds patches into simpler representations.
        This prevents the "big ball of mud" problem described in the paper.
        """
        if len(self.policies) == 0:
            return False
        return self.patch_count / len(self.policies) > coupling_threshold

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "policies": list(self.policies.keys()),
            "state_providers": list(self.state_providers.keys()),
            "feedback_channels": list(self.feedback_channels.keys()),
            "trial_count": self.trial_count,
            "patch_count": self.patch_count,
            "solved_tasks": len(self.solved_tasks),
            "failed_direction_count": sum(
                len(v) for v in self.failed_directions.values()
            ),
            "needs_compression": self.needs_compression(),
        }
