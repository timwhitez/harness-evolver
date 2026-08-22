"""FailureAnalyzer — categorize failures and extract root causes.

Reads trial trajectories and feedback signals to identify
which harness component caused each failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hl.types import FeedbackSignal, TrialResult


@dataclass
class FailureAnalyzer:
    """Analyzes task failures to identify root causes.

    Maps failure patterns to harness components so the
    meta-agent knows what to edit.
    """

    name: str = "failure_analyzer"

    # Pattern → component mapping (grows over time via meta-agent)
    pattern_to_component: dict[str, str] = field(default_factory=lambda: {
        "command_not_found": "tools/shell",
        "permission_denied": "tools/shell",
        "file_not_found": "tools/file_read",
        "syntax_error": "tools/file_edit",
        "timeout": "context/compaction",
        "tool_call_error": "tools/correction",
        "planning_error": "planning/todo_enforcement",
        "verification_error": "verification/checks",
        "context_overflow": "context/compaction",
        "wrong_approach": "prompts/system",
        "incomplete_solution": "prompts/task",
        "recovery_failure": "recovery/patterns",
        "entrypoint_miss": "entrypoint/semantic",
        "thinking_exhausted": "planning/progressive_thinking",
    })

    def analyze(
        self,
        feedback: FeedbackSignal,
        trial: TrialResult | None = None,
    ) -> dict[str, Any]:
        """Analyze a failure and return structured findings.

        Returns a dict with:
        - root_cause: human-readable description
        - affected_components: list of component names to edit
        - severity: 0.0-1.0
        - suggested_action: what kind of edit is needed
        """
        affected = self._categorize(feedback, trial)
        return {
            "task_id": feedback.trial_id,
            "status": feedback.status.value,
            "score": feedback.score,
            "root_cause": feedback.error_summary or "Unknown failure",
            "affected_components": affected,
            "severity": 1.0 if feedback.score == 0.0 else 0.5,
            "suggested_action": self._suggest_action(affected, feedback),
            "error_patterns": feedback.raw_errors,
        }

    def _categorize(
        self, feedback: FeedbackSignal, trial: TrialResult | None
    ) -> list[str]:
        """Map feedback signals to affected harness components."""
        affected: list[str] = []

        # Check raw error messages for known patterns
        for error in feedback.raw_errors:
            for pattern, component in self.pattern_to_component.items():
                if pattern.replace("_", " ") in error.lower():
                    if component not in affected:
                        affected.append(component)

        # If tool success rate is low, check tool definitions
        if feedback.tool_call_success_rate < 0.5:
            affected.append("tools/correction")

        # If trajectory is very long, check context management
        if feedback.trajectory_length > 50:
            affected.append("context/compaction")

        # Fallback: system prompt may need improvement
        if not affected:
            affected.append("prompts/system")

        return affected

    def _suggest_action(
        self, affected: list[str], feedback: FeedbackSignal
    ) -> str:
        """Suggest the type of edit needed."""
        if "tools/shell" in affected:
            return "improve error handling in shell tool or add package installation guidance"
        if "tools/file_read" in affected:
            return "improve file path resolution or add existence checks"
        if "planning/todo_enforcement" in affected:
            return "strengthen todo_write enforcement or add planning checklist"
        if "context/compaction" in affected:
            return "adjust compaction thresholds or improve context prioritization"
        if "recovery/patterns" in affected:
            return "add new error recovery pattern or improve retry strategy"
        if "verification/checks" in affected:
            return "add verification step or improve post-task checks"
        if "prompts/system" in affected:
            return "improve system prompt guidance or add specific constraint"
        return "analyze trajectory and improve relevant component"
