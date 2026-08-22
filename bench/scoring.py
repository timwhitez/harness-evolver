"""Scoring — parse Harbor verifier output and compute per-component impact.

The scoring system:
1. Reads verifier reward files from Harbor output
2. Categorizes failures by affected harness component
3. Tracks score deltas between trials
4. Computes per-component impact scores
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hl.types import TrialResult, TrialSummary, trial_is_infrastructure_failure


class Scoring:
    """Score computation and component impact analysis."""

    @staticmethod
    def parse_verifier_score(trial_dir: Path) -> float:
        """Parse the Harbor verifier reward.txt file."""
        for reward_path in [
            trial_dir / "verifier" / "reward.txt",
            trial_dir / "logs" / "verifier" / "reward.txt",
        ]:
            if not reward_path.exists():
                continue
            try:
                return float(reward_path.read_text().strip())
            except (ValueError, FileNotFoundError):
                pass
        return 0.0

    @staticmethod
    def parse_verifier_detail(trial_dir: Path) -> dict[str, Any]:
        """Parse detailed verifier output."""
        for detail_path in [
            trial_dir / "verifier" / "reward.json",
            trial_dir / "logs" / "verifier" / "results.json",
        ]:
            if not detail_path.exists():
                continue
            try:
                return json.loads(detail_path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {}

    @staticmethod
    def compute_component_impact(
        before_scores: dict[str, float],
        after_scores: dict[str, float],
        component_name: str,
    ) -> float:
        """Compute the per-task score delta attributed to a component change.

        Positive = improvement, negative = regression.
        """
        deltas = []
        for task_id in before_scores:
            if task_id in after_scores:
                deltas.append(after_scores[task_id] - before_scores[task_id])

        if not deltas:
            return 0.0

        return sum(deltas) / len(deltas)

    @staticmethod
    def build_summary(
        summary_id: str,
        trials: list[TrialResult],
        patches_applied: list[str] | None = None,
    ) -> TrialSummary:
        """Build a TrialSummary from a list of trial results."""
        if not trials:
            return TrialSummary(summary_id=summary_id)

        total = len(trials)
        passed = sum(1 for t in trials if t.status.value == "passed")
        failed = sum(1 for t in trials if t.status.value == "failed")
        timeout = sum(1 for t in trials if t.status.value == "timeout")
        error = sum(1 for t in trials if t.status.value == "error")

        # Infrastructure/environment failures (e.g. prebuilt image registry
        # denial, environment build/start timeout) are not Worker capability
        # failures and must not lower the score. Exclude them from every score
        # denominator while keeping full status counts for reporting.
        scored_trials = [t for t in trials if not trial_is_infrastructure_failure(t)]
        infrastructure_excluded = total - len(scored_trials)
        scored_tasks = len(scored_trials)

        overall = (
            sum(t.score for t in scored_trials) / scored_tasks
            if scored_tasks > 0
            else 0.0
        )

        # Per-domain scores
        domain_scores: dict[str, list[float]] = {}
        for t in scored_trials:
            domain_scores.setdefault(_value_key(t.task_domain), []).append(t.score)

        per_domain = {
            d: sum(s) / len(s) for d, s in domain_scores.items()
        }

        # Per-difficulty scores
        diff_scores: dict[str, list[float]] = {}
        for t in scored_trials:
            diff_scores.setdefault(_value_key(t.task_difficulty), []).append(t.score)

        per_difficulty = {
            d: sum(s) / len(s) for d, s in diff_scores.items()
        }

        return TrialSummary(
            summary_id=summary_id,
            trial_ids=[t.trial_id for t in trials],
            total_tasks=total,
            passed=passed,
            failed=failed,
            timeout=timeout,
            error=error,
            infrastructure_excluded=infrastructure_excluded,
            scored_tasks=scored_tasks,
            overall_score=round(overall, 4),
            per_domain_scores={k: round(v, 4) for k, v in per_domain.items()},
            per_difficulty_scores={k: round(v, 4) for k, v in per_difficulty.items()},
            patches_applied=patches_applied or [],
        )


def _value_key(value: Any) -> str:
    return str(getattr(value, "value", value))
