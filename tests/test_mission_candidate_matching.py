"""T6: mission-candidate matching is precise-id-first and marker-deduped."""

from meta.codex_update import _matching_mission_candidates


def test_verbatim_id_selects_that_candidate():
    candidates = [
        {"id": "mission-attributed-verifier-mismatch", "failure_category": "verifier_mismatch"},
        {"id": "mission-attributed-agent-timeout", "failure_category": "agent_execution_timeout"},
    ]
    report = "we selected mission-attributed-agent-timeout as the slice"
    selected = _matching_mission_candidates(candidates, report)
    assert [c["id"] for c in selected] == ["mission-attributed-agent-timeout"]


def test_two_verbatim_ids_still_ambiguous():
    candidates = [
        {"id": "mission-attributed-verifier-mismatch", "failure_category": "verifier_mismatch"},
        {"id": "mission-attributed-agent-timeout", "failure_category": "agent_execution_timeout"},
    ]
    report = (
        "mission-attributed-verifier-mismatch and mission-attributed-agent-timeout"
    )
    selected = _matching_mission_candidates(candidates, report)
    assert len(selected) == 2


def test_shared_mechanism_marker_dedupes_to_single_candidate():
    # Historical deadlock: two candidates share a mechanism failure_category, and
    # the report names only the mechanism. Old behavior matched both ->
    # "matched multiple". New behavior dedupes the marker fallback to one.
    candidates = [
        {
            "id": "mission-attributed-cross-arch-pivot",
            "failure_category": "missing_output_artifact_contract",
        },
        {
            "id": "mission-attributed-stan-dependency-pivot",
            "failure_category": "missing_output_artifact_contract",
        },
    ]
    report = "this slice targets missing_output_artifact_contract broadly"
    selected = _matching_mission_candidates(candidates, report)
    assert len(selected) == 1
