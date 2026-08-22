"""Post-task checks — verify solution completeness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.verification.base import VerificationStrategy


@dataclass
class PostTaskChecks(VerificationStrategy):
    name: str = "post_task_checks"
    version: str = "0.1.0"

    template: str = """## Before Declaring Completion

You MUST verify:
1. All tests pass (run the test suite)
2. The task requirements are fully met (re-read the instruction)
3. No temporary files or debug output remain
4. The solution is in the expected location
5. Any modified configuration files are correct
6. Required output files, directories, commands, and report formats exist and
   are visible from the path the verifier will inspect
7. If requirements depend on signals, interrupts, subprocesses, timeouts,
   stdout/stderr, or exit status, verify through the same observable process
   boundary the verifier will use rather than only with in-process mocks.

Do NOT declare completion until ALL checks pass.
"""

    def render(self, context: dict[str, Any]) -> str:
        return self.template

    def validate(self) -> list[str]:
        return []

    def raw_content(self) -> str:
        return self.template
