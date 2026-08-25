"""Aggregate-aware facade for provenance-based infrastructure attribution."""

from __future__ import annotations

import subprocess
from typing import Any

from bench import _harbor_issue5_logic as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


class HarborRunner(_base.HarborRunner):
    """Require independently structured provenance for infrastructure exclusion."""

    def _trusted_early_process_failure(
        self,
        command: _base.HarborCommand,
        completed: subprocess.CompletedProcess[str],
    ) -> tuple[str, dict[str, Any]] | None:
        """Never promote an untyped completed-process stream to provenance.

        A missing/non-executable Harbor binary is already captured at its typed
        exception source by ``_run_task_once``. Ordinary nonzero stdout/stderr
        can contain Worker-controlled Docker, DNS, or timeout-looking text and is
        therefore retained only as diagnostics.
        """

        return None

    def is_infra_error(self, trial: _base.TrialResult) -> bool:
        if bool(getattr(trial, "verified", False)):
            return False

        metadata = trial.metadata or {}
        attempt_results = metadata.get("attempt_results")
        if isinstance(attempt_results, list) and attempt_results:
            decisions = [
                self._attempt_snapshot_is_infrastructure(attempt)
                for attempt in attempt_results
                if isinstance(attempt, dict)
            ]
            # A malformed/partial aggregate cannot safely be excluded. Every
            # configured attempt must have its own trusted provenance.
            if len(decisions) != len(attempt_results):
                return False
            return all(decisions)

        return self._metadata_is_infrastructure(metadata)

    def _attempt_snapshot_is_infrastructure(self, attempt: dict[str, Any]) -> bool:
        if bool(attempt.get("verified")):
            return False
        metadata = attempt.get("metadata")
        if not isinstance(metadata, dict):
            return False
        return self._metadata_is_infrastructure(metadata)

    def _metadata_is_infrastructure(self, metadata: dict[str, Any]) -> bool:
        """Accept typed launch evidence or phase-owned textual evidence only."""

        phase = str(metadata.get("infrastructure_phase") or "")
        launch_evidence = metadata.get("harbor_launch_evidence")
        if (
            phase == "harbor_launch"
            and isinstance(launch_evidence, dict)
            and str(launch_evidence.get("kind") or "")
            in _base._DETERMINISTIC_LAUNCH_KINDS
        ):
            return True

        # ``timeout_phase`` and booleans such as
        # ``verifier_runtime_prepare_timeout`` are summaries, not provenance.
        # Only evidence fields written by the owning runner/parser phase may
        # carry the text needed for classification. Untyped Harbor/Worker/
        # verifier streams never enter this helper.
        return self._structured_text_is_infrastructure(
            self._structured_environment_evidence(metadata)
        )

    def _mark_retry_policy_finite(self, trial: _base.TrialResult) -> None:
        """Persist one final, versioned attribution decision for score consumers."""

        # The inherited implementation calls ``self.is_infra_error()``, so the
        # value below is already computed through the strict evidence rules in
        # this public facade rather than the compatibility base's old heuristics.
        super()._mark_retry_policy_finite(trial)
        metadata = trial.metadata
        current_infra = metadata.get("infra_error_detected") is True
        metadata["infra_attribution_finalized"] = True
        metadata["infra_attribution_policy"] = "phase_owned_evidence_v2"

        if current_infra:
            metadata["score_exclusion_reason"] = "infrastructure_error"
        elif metadata.get("score_exclusion_reason") == "infrastructure_error":
            metadata.pop("score_exclusion_reason", None)
