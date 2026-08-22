"""Self-test — agent runs its own tests before declaring done."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.verification.base import VerificationStrategy


@dataclass
class SelfTest(VerificationStrategy):
    name: str = "self_test"
    version: str = "0.1.0"

    template: str = """## Self-Testing

Before submitting your solution:
1. Run the project's test suite
2. If any tests fail, debug and fix them — do NOT skip failing tests
3. Add a small test for your specific change if appropriate
4. Run your solution with example inputs to verify behavior
5. Check edge cases: empty input, large input, invalid input
6. For subprocess, CLI, signal, interrupt, timeout, cancellation, stdout/stderr,
   or exit-status behavior, add a bounded check that exercises the real process
   boundary and asserts the caller-visible result.
7. For async or concurrent cleanup, assert cleanup side effects visible to the
   caller after cancellation/interruption, not only internal coroutine state.
"""

    def render(self, context: dict[str, Any]) -> str:
        return self.template

    def validate(self) -> list[str]:
        return []

    def raw_content(self) -> str:
        return self.template
