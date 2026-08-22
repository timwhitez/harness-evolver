#!/usr/bin/env python3
"""Lint a draft/final Codex report with the exact host report-gate context."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from meta import report_contract  # noqa: E402
from meta.codex_update import CodexUpdateEngine  # noqa: E402
from meta.reviewer import PatchReviewer, PatchReviewResult  # noqa: E402
from meta.update_policy import (  # noqa: E402
    merge_validation_commands,
    validation_ladder_for_changed_files,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _changed_files_from_diff(diff_text: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+)$", diff_text, flags=re.M):
        before, after = match.groups()
        path = after if after != "/dev/null" else before
        if path and path not in paths:
            paths.append(path)
    return paths


def _current_isolated_delta(
    *,
    repo_root: Path,
    review_context: dict[str, Any],
) -> tuple[list[str], list[str]]:
    baseline_entries = {
        str(entry.get("path") or ""): entry
        for entry in review_context.get("baseline_files") or []
        if isinstance(entry, dict) and str(entry.get("path") or "")
    }
    current_paths = PatchReviewer(repo_root).changed_files()
    changed: list[str] = []
    for path in current_paths:
        baseline = baseline_entries.get(path)
        if baseline is None:
            changed.append(path)
            continue
        full_path = repo_root / path
        current_exists = full_path.is_file()
        baseline_exists = bool(baseline.get("exists"))
        if current_exists != baseline_exists:
            changed.append(path)
            continue
        if current_exists and _file_sha256(full_path) != str(baseline.get("sha256") or ""):
            changed.append(path)
    ignored = [path for path in baseline_entries if path not in set(changed)]
    return changed, ignored


def load_packet_lint_context(
    packet_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Load packet evidence plus the actual isolated delta used by host review."""

    directory = Path(packet_dir)
    packet_path = directory / "codex_update_packet.json"
    if not packet_path.is_file():
        raise FileNotFoundError(f"packet not found: {packet_path}")
    packet = _load_json(packet_path)
    root = Path(repo_root)
    review_context_path = directory / "review_context.json"
    diff_path = directory / "git.diff"
    review_path = directory / "review.json"

    changed_files: list[str] = []
    ignore_files: list[str] = []
    diff_text = ""
    changed_files_source = ""
    if diff_path.is_file() and diff_path.read_text().strip():
        diff_text = diff_path.read_text()
        changed_files = _changed_files_from_diff(diff_text)
        changed_files_source = "git.diff"
    elif review_context_path.is_file():
        review_context = _load_json(review_context_path)
        changed_files, ignore_files = _current_isolated_delta(
            repo_root=root,
            review_context=review_context,
        )
        changed_files_source = "current_worktree_minus_baseline_hashes"
    elif review_path.is_file():
        host_review = _load_json(review_path)
        changed_files = [str(path) for path in host_review.get("changed_files") or []]
        changed_files_source = "review.json.changed_files"
    else:
        raise ValueError(
            "packet dir has no git.diff, review_context.json, or review.json actual-delta source"
        )

    required = [str(command) for command in packet.get("required_validation_commands") or []]
    dynamic_ladder = validation_ladder_for_changed_files(
        changed_files,
        repo_root=root,
    )
    host_commands = merge_validation_commands(required, dynamic_ladder)
    return {
        "packet": packet,
        "changed_files": changed_files,
        "ignore_files": ignore_files,
        "diff_text": diff_text,
        "changed_files_source": changed_files_source,
        "required_validation_commands": required,
        "host_validation_commands": host_commands,
        "post_edit_value_budget": {
            "valid_primary_layers": (
                report_contract.valid_primary_layers_for_changed_files(changed_files)
            ),
            "changed_files": changed_files,
            "source": changed_files_source,
        },
    }


def lint_report(
    final_report: dict[str, Any],
    *,
    changed_files: list[str] | None = None,
    ignore_files: list[str] | None = None,
    required_validation_commands: list[str] | None = None,
    host_validation_commands: list[str] | None = None,
    failure_pattern_digest: dict[str, Any] | None = None,
    mission_debug: dict[str, Any] | None = None,
    rejected_update_buffer: list[dict[str, Any]] | None = None,
    runner_pivot_policy: dict[str, Any] | None = None,
    change_evaluation_digest: dict[str, Any] | None = None,
    prior_update_lesson_entries: list[dict[str, Any]] | None = None,
    external_research_policy: dict[str, Any] | None = None,
    external_research_recommended: bool = False,
    base_review: PatchReviewResult | None = None,
    repo_root: str | Path = ".",
    changed_files_source: str = "explicit",
) -> dict[str, Any]:
    """Run the same typed report validators as host review, without Codex exec."""

    actual_changed_files = list(changed_files or [])
    engine = CodexUpdateEngine(
        repo_root=repo_root,
        events_dir=Path(repo_root) / "trials" / "report_lint_scratch",
        dry_run=True,
    )
    review = engine._apply_report_gates(
        base_review
        or PatchReviewResult(accepted=True, changed_files=actual_changed_files),
        exit_code=0,
        final_report=final_report,
        required_validation_commands=required_validation_commands or [],
        host_validation_commands=host_validation_commands or [],
        ignore_files=ignore_files or [],
        failure_pattern_digest=failure_pattern_digest or {},
        mission_debug=mission_debug or {},
        rejected_update_buffer=rejected_update_buffer or [],
        runner_pivot_policy=runner_pivot_policy or {},
        change_evaluation_digest=change_evaluation_digest or {},
        prior_update_lesson_entries=prior_update_lesson_entries or [],
        external_research_policy=external_research_policy or {},
        external_research_recommended=external_research_recommended,
    )
    details = review.reason_details
    fatal = [item for item in details if item["severity"] == report_contract.FATAL]
    advisory = [item for item in details if item["severity"] == report_contract.REPORT]
    return {
        "accepted": review.accepted,
        "fatal": fatal,
        "advisory": advisory,
        "reason_details": details,
        "changed_files": actual_changed_files,
        "ignored_baseline_files": list(ignore_files or []),
        "changed_files_source": changed_files_source,
        "post_edit_value_budget": {
            "valid_primary_layers": (
                report_contract.valid_primary_layers_for_changed_files(actual_changed_files)
            ),
            "changed_files": actual_changed_files,
            "source": changed_files_source,
        },
    }


def _lint_from_packet(
    final_report: dict[str, Any],
    packet_context: dict[str, Any],
    *,
    changed_files: list[str] | None,
    ignore_files: list[str],
    required_validation_commands: list[str],
    repo_root: str | Path,
) -> dict[str, Any]:
    packet = packet_context["packet"]
    actual_changed = (
        list(changed_files)
        if changed_files is not None
        else list(packet_context["changed_files"])
    )
    ignored = list(dict.fromkeys([*packet_context["ignore_files"], *ignore_files]))
    required = (
        required_validation_commands
        or list(packet_context["required_validation_commands"])
    )
    host_commands = merge_validation_commands(
        required,
        validation_ladder_for_changed_files(actual_changed, repo_root=repo_root),
    )
    base_review: PatchReviewResult | None = None
    diff_text = str(packet_context.get("diff_text") or "")
    if diff_text:
        base_review = PatchReviewer(
            repo_root,
            allowed_roots=[str(path) for path in packet.get("allowed_edit_paths") or []]
            or None,
        ).review_delta(actual_changed, diff_text)
    return lint_report(
        final_report,
        changed_files=actual_changed,
        ignore_files=ignored,
        required_validation_commands=required,
        host_validation_commands=host_commands,
        failure_pattern_digest=packet.get("failure_pattern_digest") or {},
        mission_debug=packet.get("mission_debug") or {},
        rejected_update_buffer=packet.get("rejected_update_buffer") or [],
        runner_pivot_policy=packet.get("runner_pivot_policy") or {},
        change_evaluation_digest=packet.get("change_evaluation_digest") or {},
        prior_update_lesson_entries=packet.get("prior_update_lesson_entries") or [],
        external_research_policy=packet.get("external_research_policy") or {},
        external_research_recommended=(
            (packet.get("external_research_policy") or {}).get("status")
            == "recommended"
        ),
        base_review=base_review,
        repo_root=repo_root,
        changed_files_source=(
            "cli --changed-file"
            if changed_files is not None
            else str(packet_context["changed_files_source"])
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint a Codex final/draft report")
    parser.add_argument("--report", help="Path to final or draft report JSON")
    parser.add_argument(
        "--packet-dir",
        help="Codex packet run dir; loads packet evidence and the isolated delta",
    )
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="Explicit actual changed file override (repeatable)",
    )
    parser.add_argument(
        "--ignore-file",
        action="append",
        default=[],
        help="Baseline dirty file exemption override (repeatable)",
    )
    parser.add_argument(
        "--required-validation-command",
        action="append",
        default=[],
        help="Required validation override (repeatable)",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    if not args.report and not args.packet_dir:
        parser.error("one of --report or --packet-dir is required")
    report_path = (
        Path(args.report)
        if args.report
        else Path(args.packet_dir) / "final_message.json"
    )
    if not report_path.is_file():
        print(f"report not found: {report_path}", file=sys.stderr)
        return 2

    final_report = _load_json(report_path)
    if args.packet_dir:
        try:
            packet_context = load_packet_lint_context(
                args.packet_dir,
                repo_root=args.repo_root,
            )
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"report_lint context error: {exc}", file=sys.stderr)
            return 2
        result = _lint_from_packet(
            final_report,
            packet_context,
            changed_files=args.changed_file or None,
            ignore_files=args.ignore_file,
            required_validation_commands=args.required_validation_command,
            repo_root=args.repo_root,
        )
    else:
        result = lint_report(
            final_report,
            changed_files=args.changed_file,
            ignore_files=args.ignore_file,
            required_validation_commands=args.required_validation_command,
            repo_root=args.repo_root,
            changed_files_source="cli --changed-file or empty isolated delta",
        )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        budget = result["post_edit_value_budget"]
        print(
            "report_lint: isolated changed files: "
            + (", ".join(result["changed_files"]) or "none")
        )
        print(
            "report_lint: valid primary layers: "
            + (", ".join(budget["valid_primary_layers"]) or "none")
        )
        if not result["fatal"] and not result["advisory"]:
            print("report_lint: clean (no contract violations)")
        if result["fatal"]:
            print(f"report_lint: {len(result['fatal'])} FATAL violation(s):")
            for detail in result["fatal"]:
                print(
                    f"  [fatal] {detail['rule_id']}: {detail['reason']}"
                )
        if result["advisory"]:
            print(f"report_lint: {len(result['advisory'])} advisory violation(s):")
            for detail in result["advisory"]:
                print(
                    f"  [report] {detail['rule_id']}: {detail['reason']}"
                )
    return 1 if result["fatal"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
