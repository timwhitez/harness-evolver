"""Typed report-contract severity and fail-closed behavior."""

from meta import report_contract
from meta.codex_update import _blocking_reasons
from meta.reviewer import PatchReviewResult


def test_registry_contains_explicit_fatal_and_report_rules():
    severities = {rule.severity for rule in report_contract.final_report_rules()}
    assert severities == {report_contract.FATAL, report_contract.REPORT}


def test_advisory_violations_do_not_block():
    finding = report_contract.violation(
        "report.cross_round_problem_class",
        "selected problem class did not cite the packet mechanism",
    )
    assert finding.rule_id == "report.cross_round_problem_class"
    assert finding.severity == report_contract.REPORT
    assert _blocking_reasons([finding]) == []


def test_fatal_violations_block_even_with_advisory_present():
    advisory = report_contract.violation(
        "report.implementation_layer",
        "reported layer did not match the isolated diff",
    )
    fatal = report_contract.violation(
        "patch.task_id_hardcoding",
        "production diff hardcodes TerminalBench task ids",
    )
    assert _blocking_reasons([advisory, fatal]) == [fatal]


def test_unknown_rule_id_fails_closed_as_internal_contract_error():
    finding = report_contract.violation("new.unregistered.rule", "new failure")
    assert finding.rule_id == report_contract.INTERNAL_RULE_ID
    assert finding.severity == report_contract.FATAL
    assert "unregistered rule id" in finding.reason


def test_untyped_patch_review_reason_gets_stable_internal_rule_id():
    review = PatchReviewResult(accepted=True, reasons=["legacy finding"])
    assert review.accepted is False
    assert review.reason_details == [
        {
            "reason": "legacy finding",
            "rule_id": report_contract.INTERNAL_RULE_ID,
            "severity": report_contract.FATAL,
        }
    ]
