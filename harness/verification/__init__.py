"""Verification strategies — check solution correctness before declaring done.

Key patterns:
- Post-task checks: verify solution completeness
- Self-test: agent runs its own tests before declaring done
"""

from harness.verification.base import VerificationStrategy
from harness.verification.checks import PostTaskChecks
from harness.verification.self_test import SelfTest

__all__ = ["VerificationStrategy", "PostTaskChecks", "SelfTest"]
