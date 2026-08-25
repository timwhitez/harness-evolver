"""Shared types with finalized infrastructure score attribution.

PR #39 owns evidence validation and persists one versioned final decision. Issue
#21 narrows scoring to consume only that decision: generic Worker/process
timeouts remain in the denominator, and stale phase/flag fields cannot exclude a
trial independently.
"""

from __future__ import annotations

from typing import Any

from hl import _types_issue21_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


# Retained as a public compatibility constant for callers that display known
# infrastructure phases. It is deliberately *not* used to infer exclusion.
INFRASTRUCTURE_TIMEOUT_PHASES = frozenset(
    {
        "environment_start",
        "environment_build",
        "verifier_runtime_prepare",
    }
)

_INFRA_ATTRIBUTION_POLICY = "phase_owned_evidence_v2"


def trial_is_infrastructure_failure(trial: Any) -> bool:
    """Return only PR #39's finalized, provenance-checked exclusion decision."""

    if bool(getattr(trial, "verified", False)):
        return False

    metadata = getattr(trial, "metadata", {}) or {}
    return (
        metadata.get("infra_attribution_finalized") is True
        and metadata.get("infra_attribution_policy") == _INFRA_ATTRIBUTION_POLICY
        and metadata.get("infra_error_detected") is True
        and metadata.get("score_exclusion_reason") == "infrastructure_error"
    )


_base.INFRASTRUCTURE_TIMEOUT_PHASES = INFRASTRUCTURE_TIMEOUT_PHASES
_base.trial_is_infrastructure_failure = trial_is_infrastructure_failure
