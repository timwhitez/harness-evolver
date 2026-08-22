"""Harness — the editable components that determine agent behavior.

Each component here is a "Policy" in HL terms: an independently
versioned, editable, testable piece of the agent's behavior.
The meta-coding-agent edits these files to improve TerminalBench scores.
"""

from harness.config import HarnessConfig
from harness.registry import HarnessRegistry

__all__ = ["HarnessConfig", "HarnessRegistry"]
