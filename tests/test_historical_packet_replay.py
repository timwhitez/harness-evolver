"""Convergence replay: the three historical Codex packets under the fixed gates.

Loads the real artifacts recorded for the three rejected updates and re-runs the
report gates with the fixed logic, asserting:

* packet 1 (023213, rust/main.rs) and packet 3 (180021, tool/shell.py): the
  qualified diff's report-narrative reasons are now advisory, so the update is
  accepted (the historical rollback is undone);
* the anti-benchmark-leakage gate still rejects a hardcoded task id, and a fatal
  gate mixed with advisory reasons still blocks (regression proof: fatals were
  not weakened).

If the recorded artifacts are absent the artifact-backed cases skip, but the
severity regression proof (equivalent fixtures) always runs.
"""

import json
from pathlib import Path

import pytest

from meta import report_contract
from meta.codex_update import CodexUpdateEngine, _blocking_reasons
from meta.reviewer import PatchReviewResult

PACKET_DIR = Path(__file__).resolve().parents[1] / "trials" / "diffs"


def _load(packet: str) -> dict | None:
    path = PACKET_DIR / packet / "final_message.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _digest(packet: str) -> dict:
    path = PACKET_DIR / packet / "codex_update_packet.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {
        "failure_pattern_digest": data.get("failure_pattern_digest") or {},
        "mission_debug": data.get("mission_debug") or {},
        "rejected_update_buffer": data.get("rejected_update_buffer") or [],
    }


@pytest.mark.parametrize(
    "packet_id,changed_files,ignore_files",
    [
        ("codex_packet_20260709_023213", ["crates/hl-worker-core/src/main.rs"], []),
        (
            "codex_packet_20260709_180021",
            ["harness/tools/shell.py", "tests/test_tool_registry.py"],
            ["AGENTS.md"],
        ),
    ],
)
def test_historical_packet_selected_problem_class_no_longer_blocks(
    tmp_path, packet_id, changed_files, ignore_files
):
    final_report = _load(packet_id)
    if final_report is None:
        pytest.skip(f"historical artifact missing: {packet_id}")
    ctx = _digest(packet_id)
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=list(changed_files)),
        exit_code=0,
        final_report=final_report,
        required_validation_commands=[],
        failure_pattern_digest=ctx["failure_pattern_digest"],
        mission_debug=ctx["mission_debug"],
        rejected_update_buffer=ctx["rejected_update_buffer"],
        ignore_files=ignore_files,
    )

    # The historical blocking reasons are now advisory (report severity), so none
    # of them appear in the blocking set. The over-reported dirty baseline file
    # (AGENTS.md for packet 3) is exempted, not blocking.
    blocking = _blocking_reasons(review.violations)
    assert not any(
        "selected_problem_class must reference a concrete" in item.reason
        for item in blocking
    ), blocking
    assert not any("includes files not changed" in item.reason for item in blocking), blocking
    assert not any(
        "primary_layer or component_type must match" in item.reason
        for item in blocking
    ), blocking


def test_hardcoded_task_id_still_fatal():
    finding = report_contract.violation(
        "patch.task_id_hardcoding",
        "production diff hardcodes TerminalBench task ids: "
        "crates/hl-worker-core/src/main.rs: path-tracing",
    )
    assert finding.severity == report_contract.FATAL
    assert _blocking_reasons([finding]) == [finding]


def test_fatal_plus_advisory_still_rejects():
    findings = [
        report_contract.violation(
            "report.cross_round_problem_class",
            "cross_round_evidence.selected_problem_class must reference a concrete "
            "failure_pattern_digest label",
        ),
        report_contract.violation(
            "patch.task_id_hardcoding",
            "production diff hardcodes TerminalBench task ids: main.rs: path-tracing",
        ),
        report_contract.violation(
            "patch.gate_weakening",
            "patch appears to weaken verifier/regression/submit gates",
        ),
    ]
    blocking = _blocking_reasons(findings)
    assert len(blocking) == 2
    assert all("must reference a concrete" not in item.reason for item in blocking)
