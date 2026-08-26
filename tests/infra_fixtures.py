"""Shared fixtures for finalized infrastructure-attribution decisions."""

from __future__ import annotations


def finalized_infra_metadata(**extra: object) -> dict[str, object]:
    """Return the complete score-authoritative v2 attribution contract."""

    return {
        "infra_attribution_finalized": True,
        "infra_attribution_policy": "phase_owned_evidence_v2",
        "infra_error_detected": True,
        "score_exclusion_reason": "infrastructure_error",
        **extra,
    }
