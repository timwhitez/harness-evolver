"""ImprovementSuggester — generate candidate harness edits.

Takes analyzed failure findings and generates specific,
minimal edits to harness components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hl.types import HarnessPatch


@dataclass
class ImprovementSuggester:
    """Generates candidate harness edits from failure analysis.

    Each suggestion is a HarnessPatch with before/after content,
    rationale, and the failure IDs it addresses.
    """

    name: str = "improvement_suggester"

    def suggest(
        self,
        findings: list[dict[str, Any]],
        component_contents: dict[str, str],
    ) -> list[HarnessPatch]:
        """Generate candidate edits from failure analysis findings.

        In the full implementation, this calls an LLM to generate
        the actual edits. The current version generates structured
        suggestions that a human or LLM can implement.
        """
        patches: list[HarnessPatch] = []

        for finding in findings:
            for component in finding.get("affected_components", []):
                if component in component_contents:
                    patch = self._build_patch(
                        component=component,
                        finding=finding,
                        current_content=component_contents[component],
                    )
                    if patch:
                        patches.append(patch)

        return patches

    def _build_patch(
        self,
        component: str,
        finding: dict[str, Any],
        current_content: str,
    ) -> HarnessPatch | None:
        """Build a single HarnessPatch suggestion."""
        # This is a template — in production, an LLM fills in the actual edit
        suggestion_note = (
            f"# Suggested edit for {component}\n"
            f"# Root cause: {finding.get('root_cause', 'unknown')}\n"
            f"# Action: {finding.get('suggested_action', 'review and improve')}\n"
            f"# Task: {finding.get('task_id', 'unknown')}\n"
        )

        return HarnessPatch(
            component_name=component,
            before_version="0.1.0",
            after_version="0.1.1",
            file_path=f"harness/{component.replace('/', '/')}.py",
            diff=suggestion_note,
            rationale=finding.get("root_cause", "Unknown failure"),
            failure_ids=[finding.get("task_id", "unknown")],
        )
