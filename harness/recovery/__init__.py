"""Error recovery — patterns for recovering from common failures.

Key patterns:
- Known failure → recovery mapping (learned over time by meta-agent)
- Exponential backoff retry with fresh context
- Escalation: when to abandon approach and try something different
"""

from harness.recovery.base import ErrorRecovery
from harness.recovery.patterns import ErrorPatterns
from harness.recovery.retry import RetryStrategy

__all__ = ["ErrorRecovery", "ErrorPatterns", "RetryStrategy"]
