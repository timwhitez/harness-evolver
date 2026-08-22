"""Entry-point discovery — find starting points before exploration begins.

Key patterns:
- Semantic scanning: lightweight pre-scan identifies likely files (ForgeCode)
- Environment awareness: auto-bootstrap system info at startup (Factory Droid)
"""

from harness.entrypoint.base import EntryPointDiscovery
from harness.entrypoint.semantic import SemanticScanner
from harness.entrypoint.environment import EnvironmentAwareness

__all__ = ["EntryPointDiscovery", "SemanticScanner", "EnvironmentAwareness"]
