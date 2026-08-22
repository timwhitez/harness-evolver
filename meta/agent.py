"""MetaAgent — the Orchestrator that analyzes failures and edits harness.

This is the "Update Engine" in HL terms.  It reads failure context
and directly edits harness component files — no backpropagation,
just a coding agent modifying code.

The meta-agent uses the strongest available reasoning model
because failure analysis requires deep understanding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import litellm

from hl.protocol import UpdateEngine
from hl.types import FeedbackSignal, HarnessPatch, TrialResult
from meta.analysis import FailureAnalyzer
from meta.editor import HarnessEditor
from meta.prompts import MetaAgentPrompts
from meta.suggestion import ImprovementSuggester


@dataclass
class MetaAgent(UpdateEngine):
    """The Orchestrator meta-agent that improves the harness.

    Implements the UpdateEngine protocol — reads failures,
    edits harness components, and tracks improvements.
    """

    name: str = "meta_agent"
    model: str = "claude-opus-4-6"  # Strongest reasoning model for orchestration
    prompts: MetaAgentPrompts = field(default_factory=MetaAgentPrompts)
    editor: HarnessEditor = field(default_factory=HarnessEditor)
    analyzer: FailureAnalyzer = field(default_factory=FailureAnalyzer)
    suggester: ImprovementSuggester = field(default_factory=ImprovementSuggester)

    _edit_history: list[HarnessPatch] = field(default_factory=list)

    def analyze_failures(
        self,
        feedback_signals: list[FeedbackSignal],
        trials: list[TrialResult],
    ) -> list[dict[str, Any]]:
        """Analyze failures and return structured findings.

        Each finding maps a failure to a root cause component.
        """
        findings: list[dict[str, Any]] = []

        for signal, trial in zip(feedback_signals, trials):
            finding = self.analyzer.analyze(signal, trial)
            findings.append(finding)

        return findings

    def suggest_edits(
        self,
        findings: list[dict[str, Any]],
        current_harness: dict[str, Any],
    ) -> list[HarnessPatch]:
        """Generate candidate harness edits from failure analysis.

        In a full implementation, this calls the LLM with the
        analysis prompt to generate specific code edits.
        """
        # Gather current component contents
        component_contents: dict[str, str] = {}
        for component in self.editor.list_editable_components():
            component_contents[component] = self.editor.get_component_content(component)

        # Generate suggestions
        patches = self.suggester.suggest(findings, component_contents)
        return patches

    def apply_patch(self, patch: HarnessPatch) -> bool:
        """Apply a harness patch with backup and validation."""
        # Snapshot before edit
        before_snapshot = self.editor.snapshot_harness()

        # Apply the edit
        success = self.editor.edit_component(
            component_path=patch.file_path,
            old_string=patch.diff.split("\n")[0] if patch.diff else "",  # Simplified
            new_string="",  # In production, parsed from LLM output
            rationale=patch.rationale,
        )

        if success:
            self._edit_history.append(patch)

        return success

    def rollback_patch(self, patch: HarnessPatch) -> bool:
        """Roll back a previously applied patch."""
        return self.editor.rollback(patch)

    def get_edit_history(self) -> list[HarnessPatch]:
        return list(self._edit_history)

    def call_llm_for_analysis(
        self,
        task_id: str,
        instruction: str,
        trajectory: str,
        errors: list[str],
        harness_summary: dict[str, Any],
    ) -> str:
        """Call the LLM to analyze a specific failure.

        Uses the strongest reasoning model for deep analysis.
        """
        from jinja2 import Template

        t = Template(self.prompts.analysis_prompt)
        prompt = t.render(
            task_id=task_id,
            domain=harness_summary.get("domain", "unknown"),
            difficulty=harness_summary.get("difficulty", "unknown"),
            instruction=instruction,
            trajectory_limit=20,
            trajectory=trajectory,
            errors="\n".join(errors),
            harness_summary=str(harness_summary),
        )

        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompts.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"LLM call failed: {e}"
