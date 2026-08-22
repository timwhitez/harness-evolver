"""C3/B2: report_lint dry-run self-check reuses the contract registry."""

import json
import hashlib
import subprocess
import sys

from meta.codex_update import CodexUpdateEngine
from meta.reviewer import PatchReviewer
from meta.update_policy import merge_validation_commands, validation_ladder_for_changed_files
from scripts.report_lint import _lint_from_packet, lint_report, load_packet_lint_context


def _valid_report() -> dict:
    return {
        "status": "edited",
        "summary": "bounded worker edit",
        "changed_files": ["bench/agent.py"],
        "validation_commands": ["pytest tests/ -v"],
        "component_type": "adapter",
        "strategy_confidence": "medium",
        "loophole_review": ["reviewed regression risk"],
        "loophole_fixes": ["kept regression gate"],
        "generalization": {
            "problem_class": "reusable worker policy",
            "applies_to": ["software_engineering tasks"],
            "anti_overfit_checks": ["does not branch on task id"],
            "why_not_task_specific": "no task-id logic",
        },
        "cross_round_evidence": {
            "used": True,
            "recent_summary_ids": ["summary_001"],
            "dominant_patterns": ["entrypoint_miss"],
            "selected_problem_class": "reusable worker policy",
            "why_this_slice_generalizes": "reusable failure class",
        },
        "memory_record": {
            "concise": "edit",
            "detailed": "bounded harness edit",
            "failed_directions_to_avoid": [],
            "supported_directions_to_preserve": [],
        },
        "framework_comparison": {
            "before": "a",
            "after": "b",
            "expected_effect": "c",
            "rollback_trigger": "d",
        },
        "prediction": {
            "expected_fixed_task_classes": ["entrypoint_miss"],
            "risk_task_classes": ["regression_gate"],
            "expected_metric_delta": 0.1,
            "confidence": "medium",
            "falsification_window": "next comparable summary",
        },
        "implementation_scope": {
            "primary_layer": "adapter",
            "architectural_change_considered": True,
            "structural_files_changed": ["bench/agent.py"],
            "why_prompt_only_is_sufficient": "not prompt-only",
        },
        "leaderboard_compliance": {
            "harbor_official_harness_preserved": True,
            "self_owned_worker_preserved": True,
            "benchmark_integrity_preserved": True,
            "timeouts_resources_unchanged": True,
            "submit_gate_preserved": True,
            "official_dataset_preserved": True,
            "five_attempts_per_task_preserved": True,
            "no_prohibited_terminal_bench_access": True,
            "upload_artifacts_trace_preserved": True,
        },
        "external_research": {
            "used": False,
            "sources": [],
            "fetches": [],
            "reason": "enough local evidence",
            "impact": "",
        },
    }


def test_lint_clean_report_has_no_fatal(tmp_path):
    result = lint_report(
        _valid_report(),
        changed_files=["bench/agent.py"],
        required_validation_commands=["pytest tests/ -v"],
        repo_root=tmp_path,
    )
    assert result["fatal"] == []


def test_lint_flags_missing_summary_as_fatal(tmp_path):
    report = _valid_report()
    report["summary"] = ""
    result = lint_report(
        report,
        changed_files=["bench/agent.py"],
        required_validation_commands=["pytest tests/ -v"],
        repo_root=tmp_path,
    )
    assert any("summary is required" in d["reason"] for d in result["fatal"])
    assert result["accepted"] is False


def test_lint_reports_advisory_with_rule_id(tmp_path):
    report = _valid_report()
    report["implementation_scope"]["primary_layer"] = "config"
    report["component_type"] = "config"
    result = lint_report(
        report,
        changed_files=["bench/agent.py"],
        required_validation_commands=["pytest tests/ -v"],
        repo_root=tmp_path,
    )
    advisory_ids = {d["rule_id"] for d in result["advisory"]}
    assert "report.implementation_layer" in advisory_ids
    # advisory-only means still accepted
    assert result["fatal"] == []


def _packet_payload() -> dict:
    return {
        "allowed_edit_paths": ["bench", "harness", "crates", "tests", "config"],
        "required_validation_commands": ["pytest tests/ -v"],
        "failure_pattern_digest": {},
        "mission_debug": {},
        "rejected_update_buffer": [],
        "runner_pivot_policy": {},
        "change_evaluation_digest": {},
        "prior_update_lesson_entries": [],
        "external_research_policy": {},
    }


def _write_packet_dir(tmp_path, *, changed_path: str = "bench/agent.py"):
    packet_dir = tmp_path / "codex_packet_fixture"
    packet_dir.mkdir()
    packet = _packet_payload()
    (packet_dir / "codex_update_packet.json").write_text(json.dumps(packet))
    (packet_dir / "git.diff").write_text(
        f"diff --git a/{changed_path} b/{changed_path}\n"
        f"--- a/{changed_path}\n"
        f"+++ b/{changed_path}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    return packet_dir, packet


def _host_review(report, packet_context, packet, tmp_path):
    changed_files = packet_context["changed_files"]
    base = PatchReviewer(
        tmp_path,
        allowed_roots=packet["allowed_edit_paths"],
    ).review_delta(changed_files, packet_context["diff_text"])
    required = packet["required_validation_commands"]
    host_commands = merge_validation_commands(
        required,
        validation_ladder_for_changed_files(changed_files, repo_root=tmp_path),
    )
    return CodexUpdateEngine(
        repo_root=tmp_path,
        events_dir=tmp_path / "diffs",
    )._apply_report_gates(
        base,
        exit_code=0,
        final_report=report,
        required_validation_commands=required,
        host_validation_commands=host_commands,
        failure_pattern_digest=packet["failure_pattern_digest"],
        mission_debug=packet["mission_debug"],
        rejected_update_buffer=packet["rejected_update_buffer"],
        runner_pivot_policy=packet["runner_pivot_policy"],
        change_evaluation_digest=packet["change_evaluation_digest"],
        prior_update_lesson_entries=packet["prior_update_lesson_entries"],
        external_research_policy=packet["external_research_policy"],
    )


def test_packet_lint_matches_host_for_accepted_and_rejected_reports(tmp_path):
    packet_dir, packet = _write_packet_dir(tmp_path)
    context = load_packet_lint_context(packet_dir, repo_root=tmp_path)

    for report in (_valid_report(), {**_valid_report(), "summary": ""}):
        linted = _lint_from_packet(
            report,
            context,
            changed_files=None,
            ignore_files=[],
            required_validation_commands=[],
            repo_root=tmp_path,
        )
        host = _host_review(report, context, packet, tmp_path)
        assert linted["accepted"] == host.accepted
        assert linted["reason_details"] == host.reason_details


def test_packet_lint_loads_all_evidence_binding_context(tmp_path):
    packet_dir, packet = _write_packet_dir(tmp_path)
    packet.update(
        {
            "failure_pattern_digest": {
                "dominant_pattern": {"failure_category": "entrypoint_miss"},
                "patterns": [{"failure_category": "entrypoint_miss"}],
            },
            "mission_debug": {
                "evidence_summary": {
                    "selected_candidate_id": "mission-attributed-verifier-gap"
                },
                "feature_candidates": [
                    {
                        "id": "mission-attributed-verifier-gap",
                        "failure_category": "verifier_gap",
                        "allowed_edit_paths": ["bench", "tests"],
                    }
                ],
            },
            "rejected_update_buffer": [
                {"packet_id": "codex_packet_old", "failure_class": "bad_direction"}
            ],
            "runner_pivot_policy": {
                "supported": [
                    {
                        "packet_id": "codex_packet_good",
                        "failure_class": "good_direction",
                        "component_layer": "adapter",
                    }
                ]
            },
            "change_evaluation_digest": {
                "miss_classes": [{"class": "missed_class"}],
                "risk_classes": [{"class": "risk_class"}],
            },
            "prior_update_lesson_entries": [
                {"packet_id": "codex_packet_lesson", "outcome": "validation_failed"}
            ],
        }
    )
    (packet_dir / "codex_update_packet.json").write_text(json.dumps(packet))
    context = load_packet_lint_context(packet_dir, repo_root=tmp_path)

    result = _lint_from_packet(
        _valid_report(),
        context,
        changed_files=None,
        ignore_files=[],
        required_validation_commands=[],
        repo_root=tmp_path,
    )

    ids = {item["rule_id"] for item in result["reason_details"]}
    assert "report.generalization_evidence" in ids
    assert "report.mission_selection" in ids
    assert "report.memory_failed_directions" in ids
    assert "report.memory_supported_directions" in ids
    assert "report.change_evaluation_misses" in ids
    assert "report.change_evaluation_risks" in ids


def test_packet_lint_supports_draft_report_without_final_message(tmp_path):
    packet_dir, _packet = _write_packet_dir(tmp_path)
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(_valid_report()))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report_lint.py",
            "--packet-dir",
            str(packet_dir),
            "--report",
            str(draft),
            "--repo-root",
            str(tmp_path),
            "--json",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout)
    assert result["changed_files"] == ["bench/agent.py"]
    assert not (packet_dir / "final_message.json").exists()


def test_post_edit_layer_budget_is_precise_for_worker_and_config_diffs(tmp_path):
    worker_dir, _packet = _write_packet_dir(
        tmp_path,
        changed_path="crates/hl-worker-core/src/main.rs",
    )
    worker = load_packet_lint_context(worker_dir, repo_root=tmp_path)
    assert worker["post_edit_value_budget"]["valid_primary_layers"] == [
        "planning",
        "tool",
        "recovery",
        "verification",
        "context",
        "adapter",
        "architecture",
    ]

    other_root = tmp_path / "config_case"
    other_root.mkdir()
    config_dir, _packet = _write_packet_dir(
        other_root,
        changed_path="config/harness.yaml",
    )
    config = load_packet_lint_context(config_dir, repo_root=other_root)
    assert config["post_edit_value_budget"]["valid_primary_layers"] == ["config"]
    assert "planning" not in config["post_edit_value_budget"]["valid_primary_layers"]


def test_report_only_lint_does_not_trust_report_changed_files(tmp_path):
    result = lint_report(_valid_report(), repo_root=tmp_path)
    assert result["changed_files"] == []
    assert any(
        item["rule_id"] == "report.changed_files" for item in result["fatal"]
    )


def test_pre_delivery_context_derives_isolated_delta_from_baseline_hashes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bench").mkdir()
    (repo / "bench" / "agent.py").write_text("before\n")
    (repo / "AGENTS.md").write_text("tracked\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    baseline_content = b"user dirty baseline\n"
    (repo / "AGENTS.md").write_bytes(baseline_content)
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    (packet_dir / "codex_update_packet.json").write_text(
        json.dumps(_packet_payload())
    )
    (packet_dir / "review_context.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_files": [
                    {
                        "path": "AGENTS.md",
                        "exists": True,
                        "sha256": hashlib.sha256(baseline_content).hexdigest(),
                        "size_bytes": len(baseline_content),
                    }
                ],
            }
        )
    )
    (repo / "bench" / "agent.py").write_text("after\n")

    context = load_packet_lint_context(packet_dir, repo_root=repo)

    assert context["changed_files"] == ["bench/agent.py"]
    assert context["ignore_files"] == ["AGENTS.md"]
    assert context["changed_files_source"] == (
        "current_worktree_minus_baseline_hashes"
    )
