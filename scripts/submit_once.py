#!/usr/bin/env python3
"""Evaluate and optionally perform one Harbor leaderboard upload."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a Harbor job at most once")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--best-job-dir", required=True)
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--tasks-evaluated", type=int, required=True)
    parser.add_argument("--full-regression-passed", action="store_true")
    parser.add_argument("--config", default="config/trials.yaml")
    parser.add_argument("--submissions-dir", default="trials/submissions")
    parser.add_argument("--harbor-bin", default="harbor")
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--trigger-score", type=float, default=None)
    parser.add_argument("--min-tasks-evaluated", type=int, default=None)
    parser.add_argument("--min-attempts-per-task", type=int, default=None)
    parser.add_argument(
        "--attempts-per-task-json",
        default=None,
        help="JSON object mapping task id to observed Harbor attempt count.",
    )
    parser.add_argument("--visibility", choices=["private", "public"], default=None)
    parser.add_argument("--share-org", action="append", default=None)
    parser.add_argument("--share-user", action="append", default=None)
    parser.add_argument(
        "--share-yes",
        action="store_true",
        help="Pass Harbor upload --yes for non-interactive share confirmation",
    )
    parser.add_argument("--no-require-full-regression", action="store_true")
    parser.add_argument("--no-require-clean-git", action="store_true")
    parser.add_argument("--no-require-no-uncommitted-harness-diff", action="store_true")
    parser.add_argument("--no-harbor-upload", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable gate result")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--record-dry-run",
        action="store_true",
        help=(
            "When used with --dry-run, write a non-terminal dry-run evidence file "
            "without creating the once-per-campaign submit marker."
        ),
    )
    args = parser.parse_args()
    if args.record_dry_run and not args.dry_run:
        parser.error("--record-dry-run requires --dry-run")

    from hl.submit import SubmitConfig, SubmitGate
    from hl.submission_storage import valid_campaign_id

    if not valid_campaign_id(args.campaign_id):
        parser.error("--campaign-id must be one safe ASCII identifier")
    config = _load_submit_config(Path(args.config))
    if args.enabled:
        config.enabled = True
    if args.trigger_score is not None:
        config.trigger_score = args.trigger_score
    if args.min_tasks_evaluated is not None:
        config.min_tasks_evaluated = args.min_tasks_evaluated
    if args.min_attempts_per_task is not None:
        config.min_attempts_per_task = args.min_attempts_per_task
    if args.visibility:
        config.visibility = args.visibility
    if args.share_org:
        config.share_orgs = args.share_org
    if args.share_user:
        config.share_users = args.share_user
    if args.share_yes:
        config.share_yes = True
    if args.no_require_full_regression:
        config.require_full_regression = False
    if args.no_require_clean_git:
        config.require_clean_git = False
    if args.no_require_no_uncommitted_harness_diff:
        config.require_no_uncommitted_harness_diff = False
    if args.no_harbor_upload:
        config.harbor_upload = False

    gate = SubmitGate(
        config,
        submissions_dir=args.submissions_dir,
        harbor_bin=args.harbor_bin,
    )
    result = gate.submit_once(
        campaign_id=args.campaign_id,
        best_job_dir=args.best_job_dir,
        score=args.score,
        tasks_evaluated=args.tasks_evaluated,
        full_regression_passed=args.full_regression_passed,
        attempts_per_task=_attempts_per_task(args, parser),
        dry_run=args.dry_run,
    )
    payload = _result_payload(result, args, config)
    if args.record_dry_run:
        record_path = Path(args.submissions_dir) / f"{args.campaign_id}.dry_run.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_payload = {
            **payload,
            "campaign_id": args.campaign_id,
            "best_job_dir": args.best_job_dir,
            "score": args.score,
            "tasks_evaluated": args.tasks_evaluated,
            "full_regression_passed": args.full_regression_passed,
            "recorded_at": datetime.now().isoformat(),
            "non_terminal_marker": True,
        }
        record_path.write_text(json.dumps(record_payload, indent=2))
        payload["dry_run_record_path"] = str(record_path)
    if args.json:
        print(json.dumps(payload, indent=2))
        return _exit_code(result, args.dry_run)

    print(f"Eligible: {result.eligible}")
    if result.attempted:
        print(f"Attempted: {result.attempted}")
        print(f"Submitted: {result.submitted}")
        if result.returncode is not None:
            print(f"Return code: {result.returncode}")
    if result.command:
        print("Command: " + " ".join(result.command))
    elif not config.harbor_upload:
        print("Command: <harbor upload disabled>")
    if result.reasons:
        print("Reasons:")
        for reason in result.reasons:
            print(f"- {reason}")
    print(f"Intent: {result.intent_path}")
    print(f"Result: {result.result_path}")
    if args.record_dry_run:
        print(f"Dry-run record: {payload['dry_run_record_path']}")
    return _exit_code(result, args.dry_run)


def _load_submit_config(path: Path) -> "SubmitConfig":
    from hl.submit import SubmitConfig

    if not path.exists():
        return SubmitConfig()
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    submit = data.get("submit", {}) if isinstance(data, dict) else {}
    allowed = set(SubmitConfig.__dataclass_fields__)
    values: dict[str, Any] = {
        key: value for key, value in submit.items() if key in allowed
    }
    return SubmitConfig(**values)


def _attempts_per_task(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, int] | None:
    raw = args.attempts_per_task_json
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        parser.error(f"--attempts-per-task-json must be a JSON object: {exc}")
    if not isinstance(data, dict):
        parser.error("--attempts-per-task-json must be a JSON object")
    from hl.submission_evidence import component, positive_int

    if not data or any(not component(key) or not positive_int(value)
                       for key, value in data.items()):
        parser.error("--attempts-per-task-json must map task names to positive integer counts")
    return data



def _exit_code(result, dry_run: bool) -> int:
    if dry_run:
        return 0
    if not result.eligible:
        return 1
    if result.attempted and result.returncode not in (None, 0):
        return int(result.returncode or 1)
    return 0


def _result_payload(result, args: argparse.Namespace, config) -> dict[str, Any]:
    return {
        "eligible": result.eligible,
        "reasons": result.reasons,
        "command": result.command,
        "intent_path": result.intent_path,
        "result_path": result.result_path,
        "attempted": result.attempted,
        "submitted": result.submitted,
        "returncode": result.returncode,
        "upload_skipped": result.upload_skipped,
        "terminal": result.terminal,
        "evidence": result.evidence,
        "dry_run": args.dry_run,
        "harbor_upload": config.harbor_upload,
        "min_attempts_per_task": config.min_attempts_per_task,
        "require_integrity_scan": config.require_integrity_scan,
        "require_atif_trajectory": config.require_atif_trajectory,
        "visibility": config.visibility,
        "share_orgs": config.share_orgs,
        "share_users": config.share_users,
        "share_yes": config.share_yes,
    }


if __name__ == "__main__":
    raise SystemExit(main())
