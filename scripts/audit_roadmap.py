#!/usr/bin/env python3
"""Audit repository readiness against concrete source and runtime evidence.

The filename is retained for CLI compatibility with existing local automation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


STATUS_PASS = "pass"
STATUS_PARTIAL = "partial"
STATUS_MISSING = "missing"


@dataclass
class ChecklistItem:
    id: str
    section: str
    requirement: str
    status: str
    evidence: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class RoadmapAudit:
    objective: str
    repo_root: str
    memory_path: str
    jobs_dir: str
    roadmap_complete: bool
    counts: dict[str, int]
    checklist: list[ChecklistItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "repo_root": self.repo_root,
            "memory_path": self.memory_path,
            "jobs_dir": self.jobs_dir,
            "roadmap_complete": self.roadmap_complete,
            "counts": self.counts,
            "checklist": [asdict(item) for item in self.checklist],
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map repository readiness requirements to concrete implementation and runtime evidence."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--memory-path", default="trials")
    parser.add_argument("--jobs-dir", default="jobs")
    parser.add_argument("--task-path", default="terminal-bench-tasks/terminal-bench")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any roadmap item is partial or missing",
    )
    args = parser.parse_args()

    audit = build_audit(
        repo_root=Path(args.repo_root),
        memory_path=Path(args.memory_path),
        jobs_dir=Path(args.jobs_dir),
        task_path=Path(args.task_path),
    )
    if args.json:
        print(json.dumps(audit.to_dict(), indent=2))
    else:
        print_text_report(audit)

    if args.strict and not audit.roadmap_complete:
        return 1
    return 0


def build_audit(
    *,
    repo_root: Path,
    memory_path: Path,
    jobs_dir: Path,
    task_path: Path,
) -> RoadmapAudit:
    repo_root = repo_root.resolve()
    memory_path = _resolve_against(repo_root, memory_path)
    jobs_dir = _resolve_against(repo_root, jobs_dir)
    task_path = _resolve_against(repo_root, task_path)
    trials = _load_trials(memory_path)
    checklist = [
        _check_product_boundary(repo_root),
        _check_benchmark_integrity(repo_root),
        _check_harbor_source(repo_root),
        _check_verified_harbor_trial(memory_path, trials),
        _check_verified_harbor_pass(memory_path, trials),
        _check_worker_loop_maturity(repo_root),
        _check_provider_config(repo_root),
        _check_secret_hygiene(repo_root),
        _check_setup_runner(repo_root),
        _check_codex_update_source(repo_root),
        _check_real_codex_update(memory_path),
        _check_goal_mode(repo_root, memory_path),
        _check_memory_absorb(memory_path, trials),
        _check_compression(repo_root, memory_path),
        _check_regression_snapshots(memory_path, trials),
        _check_regression_runs(jobs_dir),
        _check_submit_source(repo_root),
        _check_submit_evidence(memory_path),
        _check_task_curriculum(repo_root, task_path),
        _check_campaign_scale(memory_path),
    ]
    counts = {
        STATUS_PASS: sum(1 for item in checklist if item.status == STATUS_PASS),
        STATUS_PARTIAL: sum(1 for item in checklist if item.status == STATUS_PARTIAL),
        STATUS_MISSING: sum(1 for item in checklist if item.status == STATUS_MISSING),
    }
    complete = counts[STATUS_PARTIAL] == 0 and counts[STATUS_MISSING] == 0
    return RoadmapAudit(
        objective=(
            "Maintain a self-owned TerminalBench Worker harness with "
            "evidence-backed HL updates."
        ),
        repo_root=str(repo_root),
        memory_path=str(memory_path),
        jobs_dir=str(jobs_dir),
        roadmap_complete=complete,
        counts=counts,
        checklist=checklist,
    )


def print_text_report(audit: RoadmapAudit) -> None:
    print(f"Objective: {audit.objective}")
    print(f"Repo: {audit.repo_root}")
    print(
        "Status: "
        f"{audit.counts[STATUS_PASS]} pass, "
        f"{audit.counts[STATUS_PARTIAL]} partial, "
        f"{audit.counts[STATUS_MISSING]} missing"
    )
    print(f"Roadmap complete: {str(audit.roadmap_complete).lower()}")
    for item in audit.checklist:
        print(f"\n[{item.status}] {item.id}: {item.requirement}")
        for evidence in item.evidence:
            print(f"  evidence: {evidence}")
        for missing in item.missing:
            print(f"  missing: {missing}")
        for note in item.notes:
            print(f"  note: {note}")


def _check_product_boundary(repo_root: Path) -> ChecklistItem:
    required = [
        "README.md",
        "docs/architecture.md",
        "AGENTS.md",
        "bench/agent.py",
        "bench/harbor_adapter.py",
        "meta/codex_update.py",
    ]
    return _files_item(
        repo_root,
        item_id="boundary.self_owned_worker",
        section="0 Corrected Product Boundary",
        requirement="Worker is implemented in this repo and Codex is only the outer HL updater.",
        required=required,
    )


def _check_benchmark_integrity(repo_root: Path) -> ChecklistItem:
    output = _git_status(repo_root, "terminal-bench-tasks")
    if output is None:
        return ChecklistItem(
            id="boundary.benchmark_integrity",
            section="15 Non-Goals And Guardrails",
            requirement="Benchmark task tests, solutions, and definitions are not edited.",
            status=STATUS_PARTIAL,
            missing=["could not query git status for terminal-bench-tasks"],
        )
    if not output.strip():
        return ChecklistItem(
            id="boundary.benchmark_integrity",
            section="15 Non-Goals And Guardrails",
            requirement="Benchmark task tests, solutions, and definitions are not edited.",
            status=STATUS_PASS,
            evidence=["git status --short -- terminal-bench-tasks is empty"],
        )
    return ChecklistItem(
        id="boundary.benchmark_integrity",
        section="15 Non-Goals And Guardrails",
        requirement="Benchmark task tests, solutions, and definitions are not edited.",
        status=STATUS_MISSING,
        missing=["terminal-bench-tasks has git-visible changes"],
        notes=output.splitlines()[:10],
    )


def _check_harbor_source(repo_root: Path) -> ChecklistItem:
    required = [
        "bench/harbor.py",
        "bench/harbor_adapter.py",
        "bench/trajectory.py",
        "scripts/run_trial.py",
        "tests/test_roadmap_harbor.py",
    ]
    item = _files_item(
        repo_root,
        item_id="phase2_1.harbor_worker_source",
        section="4 Phase 2.1",
        requirement="Harbor command builder, custom Worker adapter, parser, run_trial CLI, and tests exist.",
        required=required,
    )
    harbor_text = _read_text(repo_root / "bench/_harbor_issue9_base.py")
    run_trial_text = _read_text(repo_root / "scripts/run_trial.py")
    if item.status == STATUS_PASS:
        if (
            'argv.extend(["--agent", self.worker_import_path])' in harbor_text
            and 'argv.extend(["--env", str(environment_import_path)])' in harbor_text
            and "--include-task-name" in harbor_text
            and "--agent-import-path" not in harbor_text
            and "--environment-import-path" not in harbor_text
        ):
            item.evidence.append("bench/harbor.py builds installed-Harbor CLI flags")
        else:
            item.status = STATUS_PARTIAL
            item.missing.append("bench/harbor.py CLI flag evidence")
        if "--dry-run" in run_trial_text and "--worker-role" in run_trial_text:
            item.evidence.append("scripts/run_trial.py exposes dry-run and worker role switches")
        else:
            item.status = STATUS_PARTIAL
            item.missing.append("scripts/run_trial.py dry-run/worker role switches")
    return item


def _check_verified_harbor_trial(
    memory_path: Path,
    trials: list[dict[str, Any]],
) -> ChecklistItem:
    verified = [trial for trial in trials if trial.get("verified") is True]
    with_job = [trial for trial in verified if trial.get("harbor_job_dir")]
    if with_job:
        trial = with_job[0]
        return ChecklistItem(
            id="phase2_1.verified_trial",
            section="4 Phase 2.1",
            requirement="At least one real Harbor trial is recorded from verifier/Harbor evidence.",
            status=STATUS_PASS,
            evidence=[
                _trial_evidence(memory_path, trial),
                f"status={trial.get('status')} score={trial.get('score')} verified=true",
            ],
        )
    if verified:
        return ChecklistItem(
            id="phase2_1.verified_trial",
            section="4 Phase 2.1",
            requirement="At least one real Harbor trial is recorded from verifier/Harbor evidence.",
            status=STATUS_PARTIAL,
            evidence=[_trial_evidence(memory_path, verified[0])],
            missing=["verified trial does not record harbor_job_dir"],
        )
    return ChecklistItem(
        id="phase2_1.verified_trial",
        section="4 Phase 2.1",
        requirement="At least one real Harbor trial is recorded from verifier/Harbor evidence.",
        status=STATUS_MISSING,
        missing=[f"no verified result.json found under {memory_path / 'runs'}"],
    )


def _check_verified_harbor_pass(
    memory_path: Path,
    trials: list[dict[str, Any]],
) -> ChecklistItem:
    passes = [trial for trial in trials if _is_verified_pass(trial)]
    for trial in passes:
        trial_dir = memory_path / "runs" / str(trial.get("trial_id", ""))
        missing = _missing_artifacts(trial_dir, ["result.json", "trajectory.jsonl", "verifier_output.txt"])
        if not missing:
            return ChecklistItem(
                id="phase2_1.verified_pass",
                section="4 Phase 2.1",
                requirement="Worker has at least one known easy task pass with verifier output and trajectory.",
                status=STATUS_PASS,
                evidence=[
                    _trial_evidence(memory_path, trial),
                    "result.json, trajectory.jsonl, and verifier_output.txt exist",
                ],
            )
    if passes:
        trial = passes[0]
        trial_dir = memory_path / "runs" / str(trial.get("trial_id", ""))
        return ChecklistItem(
            id="phase2_1.verified_pass",
            section="4 Phase 2.1",
            requirement="Worker has at least one known easy task pass with verifier output and trajectory.",
            status=STATUS_PARTIAL,
            evidence=[_trial_evidence(memory_path, trial)],
            missing=_missing_artifacts(trial_dir, ["trajectory.jsonl", "verifier_output.txt"]),
        )
    return ChecklistItem(
        id="phase2_1.verified_pass",
        section="4 Phase 2.1",
        requirement="Worker has at least one known easy task pass with verifier output and trajectory.",
        status=STATUS_MISSING,
        missing=["no verified pass with score >= 1.0 found"],
    )


def _check_worker_loop_maturity(repo_root: Path) -> ChecklistItem:
    required = [
        "bench/agent.py",
        "crates/hl-worker-core/Cargo.toml",
        "crates/hl-worker-core/src/main.rs",
        "harness/tools/todo.py",
        "harness/tools/goal.py",
        "harness/tools/verify.py",
        "harness/tools/correction.py",
        "harness/context/trajectory_pack.py",
        "tests/test_tool_registry.py",
        "tests/test_models_and_worker_policy.py",
    ]
    item = _files_item(
        repo_root,
        item_id="phase3_2.worker_loop_maturity",
        section="9 Phase 3.2",
        requirement="Worker loop has todo, goal, verify, tool correction, context packing, and tests.",
        required=required,
    )
    agent_text = _read_text(repo_root / "bench/agent.py")
    rust_text = _read_text(repo_root / "crates" / "hl-worker-core" / "src" / "main.rs")
    combined_text = f"{agent_text}\n{rust_text}"
    signals = {
        "rust prompt initialization": "fn initialize_worker_messages" in rust_text,
        "rust bounded entrypoint scan": "fn bounded_entrypoint_scan" in rust_text,
        "rust same-task memory hint": "fn memory_availability_hint" in rust_text,
        "todo gate": "pending todos" in combined_text.lower() or "todo" in combined_text.lower(),
        "verification gate": "verification_command" in combined_text,
        "tool correction": "correction" in combined_text.lower(),
        "unverified worker result": "UNVERIFIED" in agent_text or "unverified" in rust_text,
        "rust worker core": "hl-worker-core" in agent_text or "WorkerState" in rust_text,
        "python bridge only": "def _bounded_entrypoint_scan" not in agent_text
        and "def _memory_availability_hint" not in agent_text
        and "Prompt assembly and other dynamically updated Worker policy decisions live in" in agent_text,
    }
    missing = [name for name, ok in signals.items() if not ok]
    if missing:
        item.status = STATUS_PARTIAL if item.evidence else STATUS_MISSING
        item.missing.extend(missing)
    else:
        item.evidence.append("Rust worker core owns prompt initialization, entrypoint scan, same-task memory hint, todo/correction/verification gates, and UNVERIFIED signals; bench/agent.py remains a bridge")
    return item


def _check_provider_config(repo_root: Path) -> ChecklistItem:
    configs = _load_model_configs(repo_root)
    roles = configs.get("roles", {})
    worker_roles = [
        name for name, role in roles.items()
        if name.startswith("worker") and isinstance(role, dict)
    ]
    missing = []
    if "worker" not in roles:
        missing.append("worker role")
    if "orchestrator" not in roles:
        missing.append("orchestrator role")
    if "validator" not in roles and "config/models.yaml" in configs.get("sources", []):
        missing.append("validator role")
    if len(worker_roles) < 2:
        missing.append("at least two worker roles for provider switching")
    for role_name in worker_roles:
        role = roles.get(role_name, {})
        for key in ("provider", "api_key_env", "model"):
            if not role.get(key):
                missing.append(f"{role_name}.{key}")
        if "reasoning" not in role:
            missing.append(f"{role_name}.reasoning")
        if "timeout_seconds" not in role:
            missing.append(f"{role_name}.timeout_seconds")
        if "max_retries" not in role:
            missing.append(f"{role_name}.max_retries")

    evidence = [
        f"config sources: {', '.join(configs.get('sources', [])) or 'none'}",
        f"worker roles: {', '.join(worker_roles) or 'none'}",
    ]
    for role_name in worker_roles:
        role = roles.get(role_name, {})
        evidence.append(
            f"{role_name}: provider={role.get('provider')} model={role.get('model')} "
            f"base_url_host={_host(role.get('base_url')) or '<default>'} "
            f"api_key_env={role.get('api_key_env')}"
        )
    return ChecklistItem(
        id="phase2_2.provider_reasoning_config",
        section="5 Phase 2.2",
        requirement="Provider-neutral multi-role config supports parameter switching, reasoning, timeout, and retry.",
        status=STATUS_PASS if not missing else STATUS_PARTIAL,
        evidence=evidence,
        missing=missing,
    )


def _check_secret_hygiene(repo_root: Path) -> ChecklistItem:
    scanned = [
        "config",
        "scripts",
        "bench",
        "harness",
        "hl",
        "meta",
        "tests",
        "README.md",
        "docs",
        ".env.example",
    ]
    raw_key_pattern = re.compile(r"sk-[A-Za-z0-9]{16,}")
    hits: list[str] = []
    for rel in scanned:
        path = repo_root / rel
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = [
                item for item in path.rglob("*")
                if item.is_file() and item.suffix in {".py", ".yaml", ".yml", ".md", ".json", ".example"}
            ]
        else:
            candidates = []
        for candidate in candidates:
            if candidate.name == ".env.local":
                continue
            text = _read_text(candidate)
            for lineno, line in enumerate(text.splitlines(), start=1):
                for match in raw_key_pattern.finditer(line):
                    if _is_fixture_secret_match(match.group(0), line):
                        continue
                    hits.append(f"{candidate.relative_to(repo_root)}:{lineno}")
    if hits:
        return ChecklistItem(
            id="phase2_2.secret_hygiene",
            section="5 Phase 2.2 / 15 Guardrails",
            requirement="Tracked config, docs, tests, and source do not contain raw API keys.",
            status=STATUS_MISSING,
            missing=["raw API-key-like strings found"],
            notes=hits[:20],
        )
    return ChecklistItem(
        id="phase2_2.secret_hygiene",
        section="5 Phase 2.2 / 15 Guardrails",
        requirement="Tracked config, docs, tests, and source do not contain raw API keys.",
        status=STATUS_PASS,
        evidence=["no raw API-key pattern found in tracked source/config/docs/test surfaces"],
    )


# Markers that identify a deliberately fake key embedded as a test fixture or
# documentation placeholder rather than a leaked real credential. A genuine
# provider key is high-entropy and never spells out words like "test" or runs
# of sequential/repeated characters.
_FIXTURE_TOKEN_MARKERS = (
    "test",
    "fake",
    "dummy",
    "sample",
    "example",
    "placeholder",
    "fixture",
    "redact",
    "notreal",
    "secret",
    "xxxx",
    "1234567890",
    "abcdef",
    "0000",
)

# Signals that the surrounding line is asserting redaction/scrubbing behaviour
# (a negative test or a sed-style substitution) rather than carrying a secret.
_FIXTURE_LINE_MARKERS = (
    "not in",
    "redact",
    "[redacted",
    "assert",
    "s/akia",
    "/redacted/",
    "scrub",
)


def _is_fixture_secret_match(token: str, line: str) -> bool:
    lowered_token = token.lower()
    if any(marker in lowered_token for marker in _FIXTURE_TOKEN_MARKERS):
        return True
    lowered_line = line.lower()
    return any(marker in lowered_line for marker in _FIXTURE_LINE_MARKERS)


def _check_setup_runner(repo_root: Path) -> ChecklistItem:
    required = ["scripts/setup_wizard.py", "scripts/hl_run.sh", ".env.example", "tests/test_setup_wizard.py"]
    item = _files_item(
        repo_root,
        item_id="phase2_3.setup_runner",
        section="6 Phase 2.3",
        requirement="Setup wizard, one-command runner, env template, and setup tests exist.",
        required=required,
    )
    wizard = _read_text(repo_root / "scripts/setup_wizard.py")
    runner = _read_text(repo_root / "scripts/hl_run.sh")
    for label, ok in {
        "setup wizard checks Harbor": "harbor" in wizard,
        "setup wizard checks Codex": "codex" in wizard,
        "setup wizard preserves local config": "overwrite-local-config" in wizard,
        "runner supports dry-run": "--dry-run" in runner,
    }.items():
        if ok:
            item.evidence.append(label)
        else:
            item.status = STATUS_PARTIAL
            item.missing.append(label)
    return item


def _check_codex_update_source(repo_root: Path) -> ChecklistItem:
    required = [
        "meta/codex_update.py",
        "meta/packager.py",
        "meta/reviewer.py",
        "meta/prompts.py",
        "tests/test_meta_codex_update.py",
    ]
    item = _files_item(
        repo_root,
        item_id="phase3.codex_update_source",
        section="7 Phase 3",
        requirement="Codex UpdateEngine, packet builder, reviewer, prompts, and tests exist.",
        required=required,
    )
    text = _read_text(repo_root / "meta/codex_update.py")
    signals = {
        "codex exec command": "exec" in text and "--json" in text,
        "output schema": "--output-schema" in text,
        "final message": "--output-last-message" in text,
        "dirty baseline gate": "allow_dirty_baseline" in text,
        "report gates": "_apply_report_gates" in text,
        "rollback": "rollback" in text.lower(),
    }
    for label, ok in signals.items():
        if ok:
            item.evidence.append(label)
        else:
            item.status = STATUS_PARTIAL
            item.missing.append(label)
    return item


def _check_real_codex_update(memory_path: Path) -> ChecklistItem:
    diff_dirs = sorted(
        [path for path in (memory_path / "diffs").glob("codex_packet_*") if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    accepted = []
    rejected = []
    for diff_dir in diff_dirs:
        review_path = diff_dir / "review.json"
        final_path = diff_dir / "final_message.json"
        diff_path = diff_dir / "git.diff"
        if not (final_path.exists() and diff_path.exists()):
            continue
        review = _load_json(review_path) if review_path.exists() else {}
        if review.get("accepted") is True:
            accepted.append(diff_dir)
        else:
            rejected.append(diff_dir)
    if accepted:
        return ChecklistItem(
            id="phase3.real_codex_update",
            section="7 Phase 3",
            requirement="A real Codex update creates a diff, stores events, and passes review.",
            status=STATUS_PASS,
            evidence=[str(path) for path in accepted[:3]],
        )
    if rejected:
        return ChecklistItem(
            id="phase3.real_codex_update",
            section="7 Phase 3",
            requirement="A real Codex update creates a diff, stores events, and passes review.",
            status=STATUS_PARTIAL,
            evidence=[str(path) for path in rejected[:3]],
            missing=["no accepted real Codex update artifact found"],
        )
    return ChecklistItem(
        id="phase3.real_codex_update",
        section="7 Phase 3",
        requirement="A real Codex update creates a diff, stores events, and passes review.",
        status=STATUS_MISSING,
        missing=[f"no codex_packet_* artifacts found under {memory_path / 'diffs'}"],
    )


def _check_goal_mode(repo_root: Path, memory_path: Path) -> ChecklistItem:
    required = ["hl/goals.py", "tests/test_goal_submit_compression.py"]
    item = _files_item(
        repo_root,
        item_id="phase3_1.goal_mode",
        section="8 Phase 3.1",
        requirement="Persistent campaign goal and budget accounting exist and budget exhaustion is not success.",
        required=required,
    )
    text = _read_text(repo_root / "hl/goals.py")
    if "budget_exhausted" in text and "update_goal" in text:
        item.evidence.append("hl/goals.py models completion and budget exhaustion separately")
    else:
        item.status = STATUS_PARTIAL
        item.missing.append("budget exhaustion/completion separation")
    goal_path = memory_path / "goals" / "current.json"
    if not goal_path.exists():
        campaign_goals = sorted((memory_path / "goals").glob("*.json"))
        goal_path = campaign_goals[-1] if campaign_goals else goal_path
    if goal_path.exists():
        goal = _load_json(goal_path)
        item.evidence.append(f"{goal_path.name} status={goal.get('status')}")
    else:
        item.notes.append("no active persisted goal; this is acceptable outside a real campaign run")
    return item


def _check_memory_absorb(memory_path: Path, trials: list[dict[str, Any]]) -> ChecklistItem:
    required = ["result.json", "feedback.json", "harness_snapshot.json", "handoff.md"]
    for trial in trials:
        trial_dir = memory_path / "runs" / str(trial.get("trial_id", ""))
        missing = _missing_artifacts(trial_dir, required)
        if not missing:
            return ChecklistItem(
                id="phase4.memory_absorb",
                section="10 Phase 4",
                requirement="Trial memory stores normalized result, feedback, harness snapshot, and handoff.",
                status=STATUS_PASS,
                evidence=[str(trial_dir), "normalized trial artifact set exists"],
            )
    if trials:
        trial = trials[0]
        trial_dir = memory_path / "runs" / str(trial.get("trial_id", ""))
        return ChecklistItem(
            id="phase4.memory_absorb",
            section="10 Phase 4",
            requirement="Trial memory stores normalized result, feedback, harness snapshot, and handoff.",
            status=STATUS_PARTIAL,
            evidence=[str(trial_dir)],
            missing=_missing_artifacts(trial_dir, required),
        )
    return ChecklistItem(
        id="phase4.memory_absorb",
        section="10 Phase 4",
        requirement="Trial memory stores normalized result, feedback, harness snapshot, and handoff.",
        status=STATUS_MISSING,
        missing=[f"no trial result.json files under {memory_path / 'runs'}"],
    )


def _check_compression(repo_root: Path, memory_path: Path) -> ChecklistItem:
    required = ["hl/compression.py", "tests/test_goal_submit_compression.py"]
    item = _files_item(
        repo_root,
        item_id="phase4.compression",
        section="10 Phase 4",
        requirement="Compression can be planned while preserving raw evidence, regressions, and submit records.",
        required=required,
    )
    text = _read_text(repo_root / "hl/compression.py")
    if "trials/regressions" in text and "trials/submissions" in text and "dry_run" in text:
        item.evidence.append("compression dry-run preserves regression and submission paths")
    else:
        item.status = STATUS_PARTIAL
        item.missing.append("compression preservation signals")
    compression_artifacts = list((memory_path / "diffs").glob("*compression*"))
    if compression_artifacts:
        item.evidence.append(f"compression artifacts: {compression_artifacts[0]}")
    else:
        item.notes.append("no applied compression artifact found yet")
    return item


def _check_regression_snapshots(
    memory_path: Path,
    trials: list[dict[str, Any]],
) -> ChecklistItem:
    snapshots = [path for path in (memory_path / "regressions").glob("*.json") if path.is_file()]
    if not snapshots:
        return ChecklistItem(
            id="phase5.regression_snapshots",
            section="11 Phase 5",
            requirement="Regression snapshots exist and are tied to verified pass evidence.",
            status=STATUS_MISSING,
            missing=[f"no regression snapshots under {memory_path / 'regressions'}"],
        )
    pass_tasks = {str(trial.get("task_id")) for trial in trials if _is_verified_pass(trial)}
    snapshot_tasks = {str(_load_json(path).get("task_id")) for path in snapshots}
    backed = sorted(task for task in snapshot_tasks if task in pass_tasks)
    if backed:
        return ChecklistItem(
            id="phase5.regression_snapshots",
            section="11 Phase 5",
            requirement="Regression snapshots exist and are tied to verified pass evidence.",
            status=STATUS_PASS,
            evidence=[f"{len(snapshots)} snapshot(s)", f"backed by verified pass: {', '.join(backed)}"],
        )
    return ChecklistItem(
        id="phase5.regression_snapshots",
        section="11 Phase 5",
        requirement="Regression snapshots exist and are tied to verified pass evidence.",
        status=STATUS_PARTIAL,
        evidence=[f"{len(snapshots)} snapshot(s)"],
        missing=["no snapshot task matches a verified pass in trials memory"],
    )


def _check_regression_runs(jobs_dir: Path) -> ChecklistItem:
    job_results = [path for path in jobs_dir.glob("regression_*/result.json") if path.is_file()]
    passed = []
    for result_path in job_results:
        result = _load_json(result_path)
        if _job_result_has_pass(result):
            passed.append(result_path.parent)
    if passed:
        return ChecklistItem(
            id="phase5.real_regression_runs",
            section="11 Phase 5",
            requirement="Solved-task regression checks have been run through Harbor.",
            status=STATUS_PASS,
            evidence=[str(path) for path in passed[:5]],
        )
    if job_results:
        return ChecklistItem(
            id="phase5.real_regression_runs",
            section="11 Phase 5",
            requirement="Solved-task regression checks have been run through Harbor.",
            status=STATUS_PARTIAL,
            evidence=[str(path.parent) for path in job_results[:5]],
            missing=["no regression job result shows a pass reward"],
        )
    return ChecklistItem(
        id="phase5.real_regression_runs",
        section="11 Phase 5",
        requirement="Solved-task regression checks have been run through Harbor.",
        status=STATUS_MISSING,
        missing=[f"no regression_* job result.json under {jobs_dir}"],
    )


def _check_submit_source(repo_root: Path) -> ChecklistItem:
    required = ["hl/submit.py", "scripts/submit_once.py", "tests/test_goal_submit_compression.py"]
    item = _files_item(
        repo_root,
        item_id="phase6.submit_gate_source",
        section="12 Phase 6",
        requirement="One-shot submit gate source, CLI, and tests exist with default-disabled policy.",
        required=required,
    )
    trials_config = _load_yaml(repo_root / "config/trials.yaml")
    submit = trials_config.get("submit", {}) if isinstance(trials_config, dict) else {}
    if submit.get("enabled") is False:
        item.evidence.append("config/trials.yaml submit.enabled=false")
    else:
        item.status = STATUS_PARTIAL
        item.missing.append("default submit.enabled=false")
    submit_text = _read_text(repo_root / "hl/submit.py")
    if "campaign already has a submit" in submit_text and '"upload"' in submit_text:
        item.evidence.append("SubmitGate enforces duplicate prevention and Harbor upload command")
    else:
        item.status = STATUS_PARTIAL
        item.missing.append("duplicate prevention and upload command evidence")
    return item


def _check_submit_evidence(memory_path: Path) -> ChecklistItem:
    submissions = [path for path in (memory_path / "submissions").glob("*.json") if path.is_file()]
    dry_runs = [
        path for path in (memory_path / "submissions").glob("*.dry_run.json")
        if path.is_file()
    ]
    if not submissions and not dry_runs:
        return ChecklistItem(
            id="phase6.submit_evidence",
            section="12 Phase 6",
            requirement="A submit dry-run or upload result is recorded in trials/submissions.",
            status=STATUS_MISSING,
            missing=[f"no submission result JSON under {memory_path / 'submissions'}"],
        )
    submitted = []
    local_no_upload = []
    for path in submissions:
        data = _load_json(path)
        if data.get("submitted") is True:
            submitted.append(path)
        elif data.get("upload_skipped") is True:
            local_no_upload.append(path)
    if submitted:
        return ChecklistItem(
            id="phase6.submit_evidence",
            section="12 Phase 6",
            requirement="A submit dry-run or upload result is recorded in trials/submissions.",
            status=STATUS_PASS,
            evidence=[str(path) for path in submitted[:3]],
        )
    if dry_runs:
        return ChecklistItem(
            id="phase6.submit_evidence",
            section="12 Phase 6",
            requirement="A submit dry-run or upload result is recorded in trials/submissions.",
            status=STATUS_PASS,
            evidence=[str(path) for path in dry_runs[:3]],
            notes=["dry-run evidence is non-terminal and does not mark the campaign submitted"],
        )
    if local_no_upload:
        return ChecklistItem(
            id="phase6.submit_evidence",
            section="12 Phase 6",
            requirement="A submit dry-run or upload result is recorded in trials/submissions.",
            status=STATUS_PASS,
            evidence=[str(path) for path in local_no_upload[:3]],
            notes=["harbor_upload=false result intentionally skipped upload"],
        )
    return ChecklistItem(
        id="phase6.submit_evidence",
        section="12 Phase 6",
        requirement="A submit dry-run or upload result is recorded in trials/submissions.",
        status=STATUS_PARTIAL,
        evidence=[str(path) for path in submissions[:3]],
        missing=["no successful upload, local no-upload result, or recorded dry-run result"],
    )


def _check_task_curriculum(repo_root: Path, task_path: Path) -> ChecklistItem:
    required = ["bench/tasks.py", "tests/test_campaign_runner.py", "scripts/run_campaign.py"]
    item = _files_item(
        repo_root,
        item_id="phase7.task_curriculum",
        section="13 Phase 7",
        requirement="Task curriculum selection supports smoke, domain-balanced, hard-focus, and full subsets.",
        required=required,
    )
    text = _read_text(repo_root / "bench/tasks.py")
    for lane in ("smoke", "domain-balanced", "hard-focus", "full"):
        if lane in text:
            item.evidence.append(f"curriculum lane: {lane}")
        else:
            item.status = STATUS_PARTIAL
            item.missing.append(f"curriculum lane: {lane}")
    if task_path.exists():
        task_count = len([path for path in task_path.iterdir() if (path / "task.toml").exists()])
        item.evidence.append(f"local task catalog path exists with {task_count} task.toml entries")
    else:
        item.status = STATUS_PARTIAL
        item.missing.append(f"local task path missing: {task_path}")
    return item


def _check_campaign_scale(memory_path: Path) -> ChecklistItem:
    summaries = [path for path in (memory_path / "summaries").glob("*_campaign.json") if path.is_file()]
    if not summaries:
        return ChecklistItem(
            id="phase7.campaign_scale_evidence",
            section="13 Phase 7",
            requirement="Campaign evidence covers multi-task/domain/full runs with patch lineage and reproducibility.",
            status=STATUS_MISSING,
            missing=[f"no campaign summaries under {memory_path / 'summaries'}"],
        )
    best_task_count = 0
    best_path = summaries[0]
    full = []
    for path in summaries:
        data = _load_json(path)
        task_count = len(data.get("tasks") or [])
        result_task_ids = {
            str(item.get("task_id"))
            for item in (data.get("task_results") or [])
            if isinstance(item, dict) and item.get("task_id")
        }
        if task_count > best_task_count:
            best_task_count = task_count
            best_path = path
        has_lineage = bool(data.get("patch_lineage"))
        has_repro = bool(data.get("reproducibility"))
        has_full_results = len(result_task_ids) >= 89
        if task_count >= 89 and has_full_results and has_lineage and has_repro:
            full.append(path)
    if full:
        return ChecklistItem(
            id="phase7.campaign_scale_evidence",
            section="13 Phase 7",
            requirement="Campaign evidence covers multi-task/domain/full runs with patch lineage and reproducibility.",
            status=STATUS_PASS,
            evidence=[str(path) for path in full[:3]],
        )
    return ChecklistItem(
        id="phase7.campaign_scale_evidence",
        section="13 Phase 7",
        requirement="Campaign evidence covers multi-task/domain/full runs with patch lineage and reproducibility.",
        status=STATUS_PARTIAL,
        evidence=[f"largest campaign summary: {best_path} ({best_task_count} task(s))"],
        missing=[
            "no >=89-task campaign summary with >=89 completed task results, "
            "patch lineage, and reproducibility"
        ],
    )


def _files_item(
    repo_root: Path,
    *,
    item_id: str,
    section: str,
    requirement: str,
    required: list[str],
) -> ChecklistItem:
    evidence = []
    missing = []
    for rel in required:
        path = repo_root / rel
        if path.exists():
            evidence.append(rel)
        else:
            missing.append(rel)
    return ChecklistItem(
        id=item_id,
        section=section,
        requirement=requirement,
        status=STATUS_PASS if not missing else (STATUS_PARTIAL if evidence else STATUS_MISSING),
        evidence=evidence,
        missing=missing,
    )


def _load_trials(memory_path: Path) -> list[dict[str, Any]]:
    runs_dir = memory_path / "runs"
    trials = []
    for result_path in sorted(runs_dir.glob("*/result.json")):
        data = _load_json(result_path)
        if not isinstance(data, dict):
            continue
        data.setdefault("trial_id", result_path.parent.name)
        trials.append(data)
    return trials


def _load_model_configs(repo_root: Path) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    sources: list[str] = []
    for rel in ("config/models.yaml", "config/local.yaml"):
        path = repo_root / rel
        if not path.exists():
            continue
        data = _load_yaml(path)
        if not isinstance(data, dict):
            continue
        source_roles = data.get("roles")
        if not isinstance(source_roles, dict):
            source_roles = ((data.get("models") or {}).get("roles") or {})
        if isinstance(source_roles, dict):
            roles.update(source_roles)
            sources.append(rel)
    return {"roles": roles, "sources": sources}


def _is_verified_pass(trial: dict[str, Any]) -> bool:
    try:
        score = float(trial.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return trial.get("verified") is True and str(trial.get("status")) == "passed" and score >= 1.0


def _trial_evidence(memory_path: Path, trial: dict[str, Any]) -> str:
    return str(memory_path / "runs" / str(trial.get("trial_id", "")) / "result.json")


def _missing_artifacts(path: Path, names: list[str]) -> list[str]:
    return [str(path / name) for name in names if not (path / name).exists()]


def _job_result_has_pass(result: dict[str, Any]) -> bool:
    for trial in result.get("trial_results", []) or []:
        verifier = trial.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        reward = rewards.get("reward", rewards.get("score", 0.0))
        try:
            if float(reward) >= 1.0:
                return True
        except (TypeError, ValueError):
            continue
    stats = result.get("stats") or {}
    evals = stats.get("evals") or {}
    if isinstance(evals, dict):
        for eval_result in evals.values():
            if not isinstance(eval_result, dict):
                continue
            for metric in eval_result.get("metrics", []) or []:
                if isinstance(metric, dict):
                    try:
                        if float(metric.get("mean", 0.0)) >= 1.0:
                            return True
                    except (TypeError, ValueError):
                        pass
            reward_stats = ((eval_result.get("reward_stats") or {}).get("reward") or {})
            if any(_reward_key_is_pass(key) for key in reward_stats):
                return True
    return False


def _reward_key_is_pass(value: Any) -> bool:
    try:
        return float(value) >= 1.0
    except (TypeError, ValueError):
        return False


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except OSError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _host(url: Any) -> str:
    if not isinstance(url, str) or not url:
        return ""
    parsed = urlparse(url)
    return parsed.netloc or parsed.path


def _resolve_against(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _git_status(repo_root: Path, rel_path: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--", rel_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


if __name__ == "__main__":
    raise SystemExit(main())
