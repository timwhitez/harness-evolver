#!/usr/bin/env python3
"""Rebuild durable analysis reports from existing trial memory.

This is an offline evidence-replay utility. It reads recorded campaign summary
state and trial result JSON files, then re-runs the current analysis/reporting
logic without starting Harbor, Worker, Codex, or any benchmark task.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.harbor import HarborRunner  # noqa: E402
from hl.memory import FileSystemMemory  # noqa: E402
from hl.types import TrialResult, TrialSummary  # noqa: E402
from scripts.run_campaign import (  # noqa: E402
    _attach_campaign_analysis_digest,
    _normalized_codex_update_events,
    _normalized_state_analysis_reports,
    _write_iteration_analysis_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild trials/analysis reports from recorded trial memory. This "
            "is offline analysis replay only; it does not run Harbor, Worker, "
            "Codex, or TerminalBench tasks."
        )
    )
    parser.add_argument("--memory-path", default="trials")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--campaign-state",
        default=None,
        help=(
            "Path to a campaign_state.json file. Defaults to "
            "<memory-path>/summaries/<campaign-id>_campaign_state.json."
        ),
    )
    parser.add_argument(
        "--summary-id",
        action="append",
        default=None,
        help="Summary id to rebuild; repeatable. Defaults to every summary in campaign state.",
    )
    parser.add_argument(
        "--summary-json",
        action="append",
        default=None,
        help=(
            "Standalone TrialSummary JSON file to rebuild when campaign state is "
            "unavailable; repeatable."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the summaries/trials that would be analyzed without writing reports.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help=(
            "Refresh campaign_state/campaign_report derived fields from existing "
            "analysis_reports without rebuilding summaries or reading trial memory."
        ),
    )
    refresh = parser.add_mutually_exclusive_group()
    refresh.add_argument(
        "--refresh-harbor-jobs",
        dest="refresh_harbor_jobs",
        action="store_true",
        default=True,
        help=(
            "When trial memory records a readable Harbor job directory, reparse "
            "that raw job with the current parser before writing analysis. This "
            "keeps rebuilt reports aligned with parser fixes without rerunning "
            "Harbor, Worker, Codex, or benchmark tasks."
        ),
    )
    refresh.add_argument(
        "--no-refresh-harbor-jobs",
        dest="refresh_harbor_jobs",
        action="store_false",
        help="Use recorded trial result JSON exactly as-is when rebuilding analysis.",
    )
    args = parser.parse_args()

    memory_path = Path(args.memory_path)
    if args.sync_only:
        record = _sync_campaign_outputs(
            memory_path=memory_path,
            campaign_id=args.campaign_id,
            campaign_state_path=(
                Path(args.campaign_state) if args.campaign_state else None
            ),
        )
        print(json.dumps({"synced": [record]}, indent=2))
        return 0

    summary_entries = _summary_entries(
        memory_path=memory_path,
        campaign_id=args.campaign_id,
        campaign_state_path=Path(args.campaign_state) if args.campaign_state else None,
        summary_ids=set(args.summary_id or []),
        summary_json_paths=[Path(path) for path in args.summary_json or []],
    )
    if not summary_entries:
        parser.error("no matching summaries found to rebuild")

    records: list[dict[str, Any]] = []
    for summary in summary_entries:
        trials, missing, refreshed = _load_trials(
            memory_path,
            summary.trial_ids,
            refresh_harbor_jobs=args.refresh_harbor_jobs,
        )
        record: dict[str, Any] = {
            "campaign_id": args.campaign_id,
            "summary_id": summary.summary_id,
            "trial_ids": list(summary.trial_ids),
            "loaded_trials": len(trials),
            "missing_trials": missing,
            "refreshed_trials": refreshed,
        }
        if args.dry_run:
            records.append(record | {"dry_run": True})
            continue
        if refreshed:
            for trial in trials:
                if trial.trial_id in refreshed:
                    _persist_replayed_trial(memory_path, trial)
        paths = _write_iteration_analysis_report(
            memory_path=memory_path,
            campaign_id=args.campaign_id,
            summary=summary,
            trials=trials,
            campaign_state={},
        )
        state_update = _sync_campaign_state_analysis_report(
            memory_path=memory_path,
            campaign_id=args.campaign_id,
            campaign_state_path=(
                Path(args.campaign_state) if args.campaign_state else None
            ),
            analysis_paths=paths,
        )
        report_update = _sync_campaign_report_analysis_report(
            memory_path=memory_path,
            campaign_id=args.campaign_id,
            analysis_paths=paths,
        )
        records.append(
            record
            | {
                "paths": paths,
                "campaign_state": state_update,
                "campaign_report": report_update,
            }
        )

    print(json.dumps({"rebuilt": records}, indent=2))
    return 0


def _sync_campaign_outputs(
    *,
    memory_path: Path,
    campaign_id: str,
    campaign_state_path: Path | None,
) -> dict[str, Any]:
    state_update = _sync_campaign_state_derived_fields(
        memory_path=memory_path,
        campaign_id=campaign_id,
        campaign_state_path=campaign_state_path,
    )
    report_update = _sync_campaign_report_derived_fields(
        memory_path=memory_path,
        campaign_id=campaign_id,
    )
    return {
        "campaign_id": campaign_id,
        "campaign_state": state_update,
        "campaign_report": report_update,
    }


def _sync_campaign_state_derived_fields(
    *,
    memory_path: Path,
    campaign_id: str,
    campaign_state_path: Path | None,
) -> dict[str, Any]:
    state_path = campaign_state_path or _campaign_state_path(memory_path, campaign_id)
    state, error = _read_json_object(state_path, label="campaign_state")
    if error:
        return error
    original_state = json.loads(json.dumps(state, sort_keys=True))
    legacy_events = _normalize_campaign_container_codex_events(state)
    analysis_reports = _normalized_state_analysis_reports(state)
    state["analysis_reports"] = analysis_reports
    _attach_campaign_analysis_digest(state, analysis_reports)
    changed = state != original_state
    if not changed:
        return {
            "updated": False,
            "reason": "already_current",
            "path": str(state_path),
            "legacy_codex_update_events_normalized": legacy_events,
            "analysis_digest_summary_id": state.get("analysis_digest_summary_id", ""),
        }
    state["updated_at"] = datetime.now().isoformat()
    state_path.write_text(json.dumps(state, indent=2))
    return {
        "updated": True,
        "path": str(state_path),
        "legacy_codex_update_events_normalized": legacy_events,
        "analysis_digest_summary_id": state.get("analysis_digest_summary_id", ""),
    }


def _sync_campaign_report_derived_fields(
    *,
    memory_path: Path,
    campaign_id: str,
) -> dict[str, Any]:
    report_path = _campaign_report_path(memory_path, campaign_id)
    report, error = _read_json_object(report_path, label="campaign_report")
    if error:
        return error
    original_report = json.loads(json.dumps(report, sort_keys=True))
    legacy_events = _normalize_campaign_container_codex_events(report)
    analysis_reports = _normalized_state_analysis_reports(report)
    report["analysis_reports"] = analysis_reports
    _attach_campaign_analysis_digest(report, analysis_reports)
    nested_legacy_events = 0
    nested_state = report.get("campaign_state")
    if isinstance(nested_state, dict):
        nested_legacy_events = _normalize_campaign_container_codex_events(nested_state)
        nested_analysis_reports = _normalized_state_analysis_reports(nested_state)
        nested_state["analysis_reports"] = nested_analysis_reports
        _attach_campaign_analysis_digest(nested_state, nested_analysis_reports)
    changed = report != original_report
    if not changed:
        return {
            "updated": False,
            "reason": "already_current",
            "path": str(report_path),
            "legacy_codex_update_events_normalized": legacy_events,
            "campaign_state_legacy_codex_update_events_normalized": nested_legacy_events,
            "analysis_digest_summary_id": report.get("analysis_digest_summary_id", ""),
        }
    if isinstance(nested_state, dict):
        nested_state["updated_at"] = datetime.now().isoformat()
    report["updated_at"] = datetime.now().isoformat()
    report_path.write_text(json.dumps(report, indent=2))
    return {
        "updated": True,
        "path": str(report_path),
        "legacy_codex_update_events_normalized": legacy_events,
        "campaign_state_legacy_codex_update_events_normalized": nested_legacy_events,
        "analysis_digest_summary_id": report.get("analysis_digest_summary_id", ""),
    }


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        return {}, {"updated": False, "reason": f"{label}_missing"}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {}, {"updated": False, "reason": f"{label}_invalid_json", "error": str(exc)}
    if not isinstance(data, dict):
        return {}, {"updated": False, "reason": f"{label}_not_object"}
    return data, {}


def _summary_entries(
    *,
    memory_path: Path,
    campaign_id: str,
    campaign_state_path: Path | None,
    summary_ids: set[str],
    summary_json_paths: list[Path],
) -> list[SimpleNamespace]:
    entries: list[SimpleNamespace] = []
    if summary_json_paths:
        entries.extend(_summary_from_json(path) for path in summary_json_paths)
    else:
        state_path = campaign_state_path or _campaign_state_path(memory_path, campaign_id)
        state = json.loads(state_path.read_text())
        for item in state.get("summaries") or []:
            summary_id = str(item.get("summary_id") or "")
            if summary_ids and summary_id not in summary_ids:
                continue
            entries.append(
                SimpleNamespace(
                    summary_id=summary_id,
                    trial_ids=[str(trial_id) for trial_id in item.get("trial_ids") or []],
                    overall_score=float(item.get("overall_score") or 0.0),
                    patches_applied=[str(patch) for patch in item.get("patches_applied") or []],
                )
            )
    if summary_ids:
        entries = [entry for entry in entries if entry.summary_id in summary_ids]
    return entries


def _sync_campaign_state_analysis_report(
    *,
    memory_path: Path,
    campaign_id: str,
    campaign_state_path: Path | None,
    analysis_paths: dict[str, Any],
) -> dict[str, Any]:
    if not analysis_paths:
        return {"updated": False, "reason": "no_analysis_paths"}
    state_path = campaign_state_path or _campaign_state_path(memory_path, campaign_id)
    if not state_path.exists():
        return {"updated": False, "reason": "campaign_state_missing"}
    try:
        state = json.loads(state_path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "updated": False,
            "reason": "campaign_state_invalid_json",
            "error": str(exc),
        }
    if not isinstance(state, dict):
        return {"updated": False, "reason": "campaign_state_not_object"}

    replacement = dict(analysis_paths)
    reports, action = _upsert_analysis_report(
        state.get("analysis_reports"),
        replacement,
    )
    state["analysis_reports"] = reports
    normalized_codex_update_events = _normalize_campaign_container_codex_events(state)
    state["updated_at"] = datetime.now().isoformat()
    state_path.write_text(json.dumps(state, indent=2))
    return {
        "updated": True,
        "path": str(state_path),
        "summary_id": str(analysis_paths.get("summary_id") or ""),
        "action": action,
        "legacy_codex_update_events_normalized": normalized_codex_update_events,
    }


def _sync_campaign_report_analysis_report(
    *,
    memory_path: Path,
    campaign_id: str,
    analysis_paths: dict[str, Any],
) -> dict[str, Any]:
    if not analysis_paths:
        return {"updated": False, "reason": "no_analysis_paths"}
    report_path = _campaign_report_path(memory_path, campaign_id)
    if not report_path.exists():
        return {"updated": False, "reason": "campaign_report_missing"}
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "updated": False,
            "reason": "campaign_report_invalid_json",
            "error": str(exc),
        }
    if not isinstance(report, dict):
        return {"updated": False, "reason": "campaign_report_not_object"}

    replacement = dict(analysis_paths)
    top_legacy_codex_update_events = _normalize_campaign_container_codex_events(report)
    top_reports, top_action = _upsert_analysis_report(
        report.get("analysis_reports"),
        replacement,
    )
    report["analysis_reports"] = top_reports
    top_state_for_digest = {"analysis_reports": top_reports}
    top_analysis_reports = _normalized_state_analysis_reports(top_state_for_digest)
    report["analysis_reports"] = top_analysis_reports
    _attach_campaign_analysis_digest(report, top_analysis_reports)

    nested_action = "missing"
    nested_legacy_codex_update_events = 0
    nested_state = report.get("campaign_state")
    if isinstance(nested_state, dict):
        nested_legacy_codex_update_events = _normalize_campaign_container_codex_events(
            nested_state
        )
        nested_reports, nested_action = _upsert_analysis_report(
            nested_state.get("analysis_reports"),
            replacement,
        )
        nested_state_for_digest = {"analysis_reports": nested_reports}
        nested_analysis_reports = _normalized_state_analysis_reports(
            nested_state_for_digest
        )
        nested_state["analysis_reports"] = nested_analysis_reports
        _attach_campaign_analysis_digest(nested_state, nested_analysis_reports)
        nested_state["updated_at"] = datetime.now().isoformat()

    report["updated_at"] = datetime.now().isoformat()
    report_path.write_text(json.dumps(report, indent=2))
    return {
        "updated": True,
        "path": str(report_path),
        "summary_id": str(analysis_paths.get("summary_id") or ""),
        "analysis_reports_action": top_action,
        "campaign_state_analysis_reports_action": nested_action,
        "legacy_codex_update_events_normalized": top_legacy_codex_update_events,
        "campaign_state_legacy_codex_update_events_normalized": (
            nested_legacy_codex_update_events
        ),
    }


def _normalize_campaign_container_codex_events(container: dict[str, Any]) -> int:
    raw_events = container.get("codex_update_events")
    if not isinstance(raw_events, list):
        return 0
    normalized_events = _normalized_codex_update_events(container)
    container["codex_update_events"] = normalized_events
    return sum(
        1
        for event in normalized_events
        if event.get("legacy_limit_driven_skip_normalized") is True
    )


def _upsert_analysis_report(
    reports_value: Any,
    replacement: dict[str, Any],
) -> tuple[list[Any], str]:
    reports = reports_value if isinstance(reports_value, list) else []
    summary_id = str(replacement.get("summary_id") or "")
    for index, existing in enumerate(reports):
        if not isinstance(existing, dict):
            continue
        if str(existing.get("summary_id") or "") != summary_id:
            continue
        reports[index] = _merged_rebuilt_analysis_report(existing, replacement)
        return reports, "replaced"
    reports.append(replacement)
    return reports, "appended"


def _merged_rebuilt_analysis_report(
    existing: dict[str, Any],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing)
    merged.update(replacement)
    old_candidates = [
        str(item) for item in existing.get("candidate_update_classes") or []
    ]
    new_candidates = [
        str(item) for item in replacement.get("candidate_update_classes") or []
    ]
    if old_candidates and old_candidates != new_candidates:
        merged["raw_candidate_update_classes"] = existing.get(
            "raw_candidate_update_classes",
            old_candidates,
        )
        merged["candidate_update_classes_normalized_from"] = "rebuild_analysis"
    return merged


def _campaign_state_path(memory_path: Path, campaign_id: str) -> Path:
    return memory_path / "summaries" / f"{_safe_campaign_id(campaign_id)}_campaign_state.json"


def _campaign_report_path(memory_path: Path, campaign_id: str) -> Path:
    return memory_path / "summaries" / f"{_safe_campaign_id(campaign_id)}_campaign.json"


def _safe_campaign_id(campaign_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in campaign_id)


def _summary_from_json(path: Path) -> SimpleNamespace:
    summary = TrialSummary.model_validate_json(path.read_text())
    return SimpleNamespace(
        summary_id=summary.summary_id,
        trial_ids=list(summary.trial_ids),
        overall_score=summary.overall_score,
        patches_applied=list(summary.patches_applied),
    )


def _load_trials(
    memory_path: Path,
    trial_ids: list[str],
    *,
    refresh_harbor_jobs: bool = True,
) -> tuple[list[TrialResult], list[str], list[str]]:
    trials: list[TrialResult] = []
    missing: list[str] = []
    refreshed: list[str] = []
    for trial_id in trial_ids:
        result_path = memory_path / "runs" / trial_id / "result.json"
        if not result_path.exists():
            missing.append(trial_id)
            continue
        trial = TrialResult.model_validate_json(result_path.read_text())
        if refresh_harbor_jobs:
            reparsed = _refresh_trial_from_harbor_job(trial)
            if reparsed is not None:
                trial = reparsed
                refreshed.append(trial_id)
        trials.append(trial)
    return trials, missing, refreshed


def _refresh_trial_from_harbor_job(trial: TrialResult) -> TrialResult | None:
    job_dir = Path(str(trial.harbor_job_dir or ""))
    if not job_dir.exists() or not (job_dir / "result.json").exists():
        return None
    try:
        return HarborRunner(output_dir=Path("__analysis_replay__")).parse_job_dir(
            job_dir,
            task_id=trial.task_id,
            returncode=int(trial.metadata.get("harbor_returncode") or 0),
            stdout=trial.harbor_stdout,
            stderr=trial.harbor_stderr,
            wall_time=trial.wall_time_seconds,
            agent_config=(trial.metadata.get("model_config") or {}),
        )
    except Exception:
        return None


def _persist_replayed_trial(memory_path: Path, trial: TrialResult) -> None:
    """Rewrite durable trial memory after offline Harbor reparse.

    This intentionally avoids appending to scoreboard.csv: the trial already
    happened, and rebuild_analysis is correcting stale parser-derived memory,
    not recording a new benchmark run.
    """
    FileSystemMemory(str(memory_path)).record_trial(trial, append_scoreboard=False)


if __name__ == "__main__":
    raise SystemExit(main())
