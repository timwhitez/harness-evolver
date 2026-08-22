"""Mission-style meta debugging packets for the outer HL loop.

This module adapts Factory-style Missions into a deterministic planning surface
for HarnessEvolver. It does not spawn external agents and does not solve
TerminalBench tasks. It turns real trial/campaign evidence into bounded feature
candidates, validation contracts, and external-loop controls that another
orchestrator can choose from.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from hl.loop_limits import base_loop_limit_contract
from hl.types import TrialResult

REPO_EDIT_ROOTS = {
    "bench",
    "hl",
    "meta",
    "harness",
    "scripts",
    "config",
    "crates",
    "tests",
}

LOGICAL_COMPONENT_EDIT_ROOTS: dict[str, tuple[str, ...]] = {
    "context": ("harness", "crates"),
    "entrypoint": ("bench", "harness", "crates"),
    "planning": ("harness", "crates"),
    "prompts": ("harness",),
    "recovery": ("harness", "crates"),
    "tools": ("harness",),
    "verification": ("harness",),
    "worker_loop": ("bench", "crates", "harness"),
}


MISSIONS_SOURCE_URL = "https://factory.ai/news/missions-architecture"


class MissionValidationContract(BaseModel):
    """Validation that must exist before an external loop accepts a feature."""

    id: str
    description: str
    commands: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    pass_condition: str


class MissionFeatureCandidate(BaseModel):
    """One bounded harness-improvement slice an external loop may select."""

    id: str
    title: str
    rationale: str
    target_tasks: list[str] = Field(default_factory=list)
    affected_components: list[str] = Field(default_factory=list)
    allowed_edit_paths: list[str] = Field(default_factory=list)
    validation_contracts: list[str] = Field(default_factory=list)
    success_signal: str
    priority: str = "P2"


class MissionDebugPacket(BaseModel):
    """Serializable mission packet consumed by Codex or another outer loop."""

    mission_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    source: str = ""
    source_url: str = MISSIONS_SOURCE_URL
    objective: str
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    validation_contracts: list[MissionValidationContract] = Field(default_factory=list)
    feature_candidates: list[MissionFeatureCandidate] = Field(default_factory=list)
    candidate_audit: list[MissionFeatureCandidate] = Field(default_factory=list)
    external_loop_controls: list[dict[str, Any]] = Field(default_factory=list)
    loop_limit_contract: dict[str, Any] = Field(default_factory=dict)
    blocked_actions: list[str] = Field(default_factory=list)
    architecture_notes: list[str] = Field(default_factory=list)


class MissionPlanner:
    """Build mission debug packets from already-recorded evidence."""

    def from_trials(
        self,
        trials: list[TrialResult],
        *,
        source: str = "trial_failures",
        max_features: int = 6,
    ) -> MissionDebugPacket:
        task_results = [self._trial_result_dict(trial) for trial in trials]
        summary = {
            "campaign_id": source,
            "task_results": task_results,
            "tasks_completed": len(task_results),
            "tasks_pending": 0,
            "score_history": [self._score_history_from_task_results(task_results)],
        }
        return self.from_campaign_summary(
            summary,
            source_path=source,
            max_features=max_features,
        )

    def from_campaign_file(
        self,
        path: str | Path,
        *,
        max_features: int = 6,
        covered_mechanism_signatures: list[str] | None = None,
    ) -> MissionDebugPacket:
        import json

        summary_path = Path(path)
        summary = json.loads(summary_path.read_text())
        return self.from_campaign_summary(
            summary,
            source_path=str(summary_path),
            max_features=max_features,
            covered_mechanism_signatures=covered_mechanism_signatures,
        )

    def from_campaign_summary(
        self,
        summary: dict[str, Any],
        *,
        source_path: str = "",
        max_features: int = 6,
        covered_mechanism_signatures: list[str] | None = None,
    ) -> MissionDebugPacket:
        covered = self._normalize_covered_signatures(covered_mechanism_signatures)
        task_results = list(summary.get("task_results") or [])
        score_history = list(summary.get("score_history") or [])
        last_score = score_history[-1] if score_history else {}
        status_counts = self._status_counts(task_results, last_score)
        passed_tasks = self._tasks_with_status(task_results, {"passed"})
        packet = MissionDebugPacket(
            mission_id=f"mission_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            source=source_path or str(summary.get("campaign_id") or "campaign_summary"),
            objective=(
                "Expose validation-first, bounded meta-improvement choices for "
                "the external HL loop while preserving the self-owned Worker boundary."
            ),
            evidence_summary={
                "campaign_id": summary.get("campaign_id"),
                "tasks_completed": summary.get("tasks_completed", len(task_results)),
                "tasks_pending": summary.get("tasks_pending"),
                "task_results": len(task_results),
                "score": last_score.get("score", summary.get("overall_score", 0.0)),
                "status_counts": dict(status_counts),
                "top_domains": self._top_values(task_results, "domain"),
                "top_difficulties": self._top_values(task_results, "difficulty"),
                "has_patch_lineage": bool(summary.get("patch_lineage")),
                "has_reproducibility": bool(summary.get("reproducibility")),
                "max_features_audit_only": max_features,
                "max_features_stop_condition": False,
                "covered_mechanism_signatures": sorted(covered),
                "filtered_covered_candidate_ids": [],
            },
            validation_contracts=self._validation_contracts(passed_tasks),
            blocked_actions=[
                "Do not edit terminal-bench-tasks, task tests, task solutions, or verifier code.",
                "Do not delegate benchmark task execution to Factory, Codex, ForgeCode, or another external agent.",
                "Do not mark campaign goals complete from mission packet text; require audit/test evidence.",
                "Do not submit/upload from this debug layer; use the one-shot submit gate only.",
            ],
            architecture_notes=[
                "Mission state is an external-loop artifact, not Worker session state.",
                "Validation contracts are produced before feature candidates are accepted.",
                "Feature candidates are scoped harness/policy slices, not benchmark-task solutions.",
                "External loops may select, reorder, or discard candidates based on fresh tests.",
                "Mission feature counts, target-task counts, elapsed time, and round counts are audit or packet-size metadata only; they must not stop master, Codex update, diagnostic/context, mission-debug, or Worker loops.",
            ],
            external_loop_controls=self._external_loop_controls(),
            loop_limit_contract=self._loop_limit_contract(max_features=max_features),
        )
        candidates = self._feature_candidates(
            task_results=task_results,
            status_counts=status_counts,
            validation_contracts=[contract.id for contract in packet.validation_contracts],
            max_features=max_features,
        )
        packet.candidate_audit = list(candidates)
        if covered:
            kept, filtered_ids = self.filter_covered_candidates(
                candidates,
                list(covered),
            )
            if kept:
                candidates = kept
                packet.evidence_summary["filtered_covered_candidate_ids"] = filtered_ids
            else:
                candidates = []
                packet.evidence_summary["filtered_covered_candidate_ids"] = filtered_ids
                packet.evidence_summary["all_candidates_covered"] = True
                packet.evidence_summary["skip_codex_update"] = {
                    "reason": "all mission candidates are already covered by current policy/tests or accepted update memory",
                    "covered_candidate_ids": filtered_ids,
                    "covered_mechanism_signatures": sorted(covered),
                }
        packet.feature_candidates = candidates
        return packet

    def filter_covered_candidates(
        self,
        candidates: list[MissionFeatureCandidate],
        covered_mechanism_signatures: list[str] | None,
    ) -> tuple[list[MissionFeatureCandidate], list[str]]:
        """Return uncovered candidates and the ids removed by policy coverage."""

        covered = self._normalize_covered_signatures(covered_mechanism_signatures)
        if not covered:
            return list(candidates), []
        kept: list[MissionFeatureCandidate] = []
        filtered_ids: list[str] = []
        for candidate in candidates:
            if self._candidate_is_covered(candidate, covered):
                filtered_ids.append(candidate.id)
            else:
                kept.append(candidate)
        return kept, filtered_ids

    def _validation_contracts(
        self,
        passed_tasks: list[str],
    ) -> list[MissionValidationContract]:
        contracts = [
            MissionValidationContract(
                id="contract-readiness-audit",
                description="Repository readiness must remain true after any mission-selected edit.",
                commands=["python scripts/audit_roadmap.py --json"],
                required_artifacts=["trials/summaries/*_campaign.json"],
                pass_condition="The readiness audit has zero partial or missing checklist items.",
            ),
            MissionValidationContract(
                id="contract-unit-suite",
                description="Repository behavior must remain covered by the deterministic unit suite.",
                commands=["pytest tests/ -q"],
                required_artifacts=[],
                pass_condition="All tests pass.",
            ),
            MissionValidationContract(
                id="contract-dry-run-regression",
                description="Regression wiring and Harbor command construction remain valid.",
                commands=["python scripts/regression_check.py --dry-run"],
                required_artifacts=["trials/regressions"],
                pass_condition="Dry-run exits successfully and preserves solved-task regression gates.",
            ),
        ]
        if passed_tasks:
            contracts.append(
                MissionValidationContract(
                    id="contract-solved-task-protection",
                    description=(
                        "Known verified passes stay protected before accepting a new harness slice; "
                        "the command lists every observed passed task so mission-debug task counts "
                        "do not become a validation/regression, master, sub-agent, or Worker loop limit."
                    ),
                    commands=[
                        "python scripts/regression_check.py "
                        + " ".join(f"--task {task}" for task in passed_tasks)
                    ],
                    required_artifacts=["trials/regressions"],
                    pass_condition=(
                        "All observed solved-task regressions in the packet do not lose verified pass evidence; "
                        "full same-model regression coverage remains owned by the regression gate, not by "
                        "mission-debug feature, target-task, validation-contract, time, round, or task-count limits."
                    ),
                )
            )
        return contracts

    def _feature_candidates(
        self,
        *,
        task_results: list[dict[str, Any]],
        status_counts: Counter[str],
        validation_contracts: list[str],
        max_features: int,
    ) -> list[MissionFeatureCandidate]:
        attributed_candidates = self._attributed_feature_candidates(
            task_results=task_results,
            validation_contracts=validation_contracts,
        )
        _ = max_features
        if attributed_candidates:
            return attributed_candidates

        candidates: list[MissionFeatureCandidate] = []
        if status_counts["error"]:
            candidates.append(
                MissionFeatureCandidate(
                    id="mission-harbor-error-attribution",
                    title="Make Harbor/environment errors first-class repair signals",
                    rationale=(
                        f"{status_counts['error']} task(s) ended as error. The external loop "
                        "needs structured root-cause buckets before selecting a Codex update."
                    ),
                    target_tasks=self._tasks_with_status(task_results, {"error"}),
                    affected_components=[
                        "bench/harbor.py",
                        "bench/trajectory.py",
                        "hl/attribution.py",
                        "meta/packager.py",
                    ],
                    allowed_edit_paths=["bench", "hl", "meta", "scripts", "tests"],
                    validation_contracts=validation_contracts,
                    success_signal="Error trials include actionable category, component, and repair-scope evidence.",
                    priority="P1",
                )
            )
        if status_counts["timeout"]:
            candidates.append(
                MissionFeatureCandidate(
                    id="mission-timeout-recovery-policy",
                    title="Separate build/runtime/model timeouts for targeted recovery",
                    rationale=(
                        f"{status_counts['timeout']} task(s) timed out. A single timeout bucket "
                        "is too coarse for external selection and provider/timeout adjustment."
                    ),
                    target_tasks=self._tasks_with_status(task_results, {"timeout"}),
                    affected_components=[
                        "bench/harbor.py",
                        "bench/trajectory.py",
                        "harness/recovery/retry.py",
                        "config/trials.yaml",
                    ],
                    allowed_edit_paths=["bench", "harness", "config", "tests"],
                    validation_contracts=validation_contracts,
                    success_signal="Timeout evidence distinguishes Docker build, task runtime, verifier, and LLM phases.",
                    priority="P1",
                )
            )
        if status_counts["failed"]:
            candidates.append(
                MissionFeatureCandidate(
                    id="mission-verified-failure-learning",
                    title="Turn verified failures into bounded Worker policy updates",
                    rationale=(
                        f"{status_counts['failed']} verified or verifier-grounded failure(s) "
                        "can train Rust Worker policy and tool behavior without weakening verifier gates."
                    ),
                    target_tasks=self._tasks_with_status(task_results, {"failed"}),
                    affected_components=[
                        "crates/hl-worker-core/src/main.rs",
                        "harness/tools/verify.py",
                        "harness/context/trajectory_pack.py",
                    ],
                    allowed_edit_paths=["crates", "harness", "tests"],
                    validation_contracts=validation_contracts,
                    success_signal="A focused rerun improves score or produces richer verifier-grounded diagnostics.",
                    priority="P2",
                )
            )
        if status_counts["passed"]:
            candidates.append(
                MissionFeatureCandidate(
                    id="mission-regression-contract-hardening",
                    title="Promote solved tasks into stronger validation contracts",
                    rationale=(
                        f"{status_counts['passed']} task(s) passed and should constrain future "
                        "mission-selected updates."
                    ),
                    target_tasks=self._tasks_with_status(task_results, {"passed"}),
                    affected_components=["hl/memory.py", "hl/loop.py", "scripts/regression_check.py"],
                    allowed_edit_paths=["hl", "scripts", "tests"],
                    validation_contracts=["contract-unit-suite", "contract-dry-run-regression"],
                    success_signal="Solved-task snapshots are discoverable and run before accepting edits.",
                    priority="P2",
                )
            )
        if not candidates:
            candidates.append(
                MissionFeatureCandidate(
                    id="mission-next-score-slice",
                    title="Select the next score-improvement slice from current evidence",
                    rationale="No failure-heavy bucket dominates; external loop should select a narrow next experiment.",
                    target_tasks=self._all_task_ids(task_results),
                    affected_components=["crates/hl-worker-core/src/main.rs", "harness", "meta"],
                    allowed_edit_paths=["crates", "harness", "meta", "tests"],
                    validation_contracts=validation_contracts,
                    success_signal="Next campaign summary shows non-negative score movement with no regression loss.",
                    priority="P3",
                )
            )
        return candidates

    def _normalize_covered_signatures(
        self,
        covered_mechanism_signatures: list[str] | None,
    ) -> set[str]:
        if not covered_mechanism_signatures:
            return set()
        normalized: set[str] = set()
        for signature in covered_mechanism_signatures:
            text = str(signature or "").strip().lower()
            if len(text) >= 3:
                normalized.add(text)
        return normalized

    def _candidate_is_covered(
        self,
        candidate: MissionFeatureCandidate,
        covered: set[str],
    ) -> bool:
        """True when a candidate's mechanism signature is already Worker-covered.

        The candidate id encodes its failure_category + mechanism slug (e.g.
        ``mission-attributed-<category>-<mechanism>``). If any covered signature
        appears in the id, title, or rationale, re-proposing a patch is
        speculative -- exactly the packet-2 noop trap.
        """

        haystack = " ".join(
            [
                str(candidate.id or ""),
                str(candidate.title or ""),
                str(candidate.rationale or ""),
            ]
        ).lower()
        normalized_haystack = haystack.replace("-", "_").replace(" ", "_")
        for signature in covered:
            if signature in haystack:
                return True
            normalized_signature = signature.replace("-", "_").replace(" ", "_")
            if normalized_signature in normalized_haystack:
                return True
        return False

    def _attributed_feature_candidates(
        self,
        *,
        task_results: list[dict[str, Any]],
        validation_contracts: list[str],
    ) -> list[MissionFeatureCandidate]:
        mechanism_buckets = self._mechanism_attribution_buckets(task_results)
        if mechanism_buckets:
            return [
                self._feature_candidate_from_bucket(
                    bucket,
                    validation_contracts=validation_contracts,
                    mechanism_scoped=True,
                )
                for bucket in mechanism_buckets
            ]
        buckets = self._attribution_buckets(task_results)
        if not buckets:
            return []
        return [
            self._feature_candidate_from_bucket(
                bucket,
                validation_contracts=validation_contracts,
                mechanism_scoped=False,
            )
            for bucket in buckets
        ]

    def _feature_candidate_from_bucket(
        self,
        bucket: dict[str, Any],
        *,
        validation_contracts: list[str],
        mechanism_scoped: bool,
    ) -> MissionFeatureCandidate:
        category = str(bucket["failure_category"])
        mechanism = str(bucket.get("mechanism") or "")
        components = bucket["affected_components"]
        infrastructure = bool(bucket.get("infrastructure"))
        allowed_paths = self._allowed_paths_for_components(components)
        if infrastructure and "bench" not in allowed_paths:
            allowed_paths.insert(0, "bench")
        if mechanism_scoped:
            candidate_id = "mission-attributed-" + self._slug(
                category + "-" + mechanism
            )
            title = "Repair " + mechanism.replace("_", " ") + " evidence path"
            rationale = (
                f"{bucket['count']} task(s) share failure_category={category} "
                f"and mechanism={mechanism}; mission selection should target this "
                "specific root mechanism instead of the broader category bucket."
            )
            success_signal = (
                "Next campaign report changes this mechanism count, adds richer "
                "mechanism/trajectory evidence, or improves score without solved-task regression."
            )
        else:
            candidate_id = "mission-attributed-" + self._slug(category)
            title = "Repair " + category.replace("_", " ") + " evidence path"
            rationale = (
                f"{bucket['count']} task(s) share failure_category={category}; "
                "mission selection should target this root-cause bucket "
                "instead of a broad status-only slice."
            )
            success_signal = (
                "Next campaign report changes this failure_category count, "
                "adds richer verifier/trajectory evidence, or improves score "
                "without solved-task regression."
            )
        return MissionFeatureCandidate(
            id=candidate_id,
            title=title,
            rationale=rationale,
            target_tasks=bucket["task_ids"],
            affected_components=components,
            allowed_edit_paths=allowed_paths,
            validation_contracts=validation_contracts,
            success_signal=success_signal,
            priority="P1" if infrastructure or bucket["count"] > 1 else "P2",
        )

    def _mechanism_attribution_buckets(
        self,
        task_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, str], dict[str, Any]] = {}
        for item in task_results:
            if str(item.get("status") or "").lower() == "passed":
                continue
            category = str(item.get("failure_category") or "").strip()
            if not category:
                continue
            mechanism_names = self._failure_mechanism_names(item)
            if not mechanism_names:
                continue
            for mechanism in mechanism_names:
                bucket = buckets.setdefault(
                    (category, mechanism),
                    {
                        "failure_category": category,
                        "mechanism": mechanism,
                        "count": 0,
                        "task_ids": [],
                        "affected_components": [],
                        "timeout_phases": set(),
                        "statuses": Counter(),
                        "infrastructure": False,
                    },
                )
                self._add_task_result_to_bucket(bucket, item)
        return self._finalize_attribution_buckets(buckets.values(), mechanism_scoped=True)

    def _attribution_buckets(
        self,
        task_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        buckets: dict[str, dict[str, Any]] = {}
        for item in task_results:
            if str(item.get("status") or "").lower() == "passed":
                continue
            category = str(item.get("failure_category") or "").strip()
            if not category:
                continue
            bucket = buckets.setdefault(
                category,
                {
                    "failure_category": category,
                    "count": 0,
                    "task_ids": [],
                    "affected_components": [],
                    "timeout_phases": set(),
                    "statuses": Counter(),
                    "infrastructure": False,
                },
            )
            self._add_task_result_to_bucket(bucket, item)
        return self._finalize_attribution_buckets(buckets.values(), mechanism_scoped=False)

    def _add_task_result_to_bucket(
        self,
        bucket: dict[str, Any],
        item: dict[str, Any],
    ) -> None:
        bucket["count"] += 1
        task_id = str(item.get("task_id") or "").strip()
        if task_id and task_id not in bucket["task_ids"]:
            bucket["task_ids"].append(task_id)
        for component in item.get("affected_components") or []:
            component_text = str(component).strip()
            if component_text and component_text not in bucket["affected_components"]:
                bucket["affected_components"].append(component_text)
        timeout_phase = str(item.get("timeout_phase") or "").strip()
        if timeout_phase:
            bucket["timeout_phases"].add(timeout_phase)
        status = str(item.get("status") or "").lower()
        if status:
            bucket["statuses"][status] += 1
        if bool(item.get("infra_error_detected")) or str(
            item.get("score_exclusion_reason") or ""
        ) == "infrastructure_error":
            bucket["infrastructure"] = True

    def _finalize_attribution_buckets(
        self,
        buckets: Any,
        *,
        mechanism_scoped: bool,
    ) -> list[dict[str, Any]]:
        result = []
        for bucket in buckets:
            if not bucket["affected_components"]:
                continue
            entry = {
                "failure_category": bucket["failure_category"],
                "count": int(bucket["count"]),
                "task_ids": bucket["task_ids"],
                "affected_components": list(bucket["affected_components"]),
                "affected_components_count_stop_condition": False,
                "component_count_stop_condition": False,
                "timeout_phases": sorted(bucket["timeout_phases"]),
                "statuses": dict(bucket["statuses"]),
                "infrastructure": bool(bucket["infrastructure"]),
            }
            if mechanism_scoped:
                entry["mechanism"] = str(bucket.get("mechanism") or "")
                entry["mechanism_count_stop_condition"] = False
            result.append(entry)
        result.sort(
            key=lambda item: (
                0 if item["infrastructure"] else 1,
                -int(item["count"]),
                str(item["failure_category"]),
                str(item.get("mechanism") or ""),
            )
        )
        return result

    def _failure_mechanism_names(self, item: dict[str, Any]) -> list[str]:
        names: list[str] = []
        raw_mechanisms = item.get("failure_mechanisms")
        if isinstance(raw_mechanisms, list):
            for raw_mechanism in raw_mechanisms:
                if isinstance(raw_mechanism, dict):
                    name = str(raw_mechanism.get("name") or "").strip()
                else:
                    name = str(raw_mechanism or "").strip()
                if name and name not in names:
                    names.append(name)
        raw_entries = item.get("mechanism_update_entries")
        if isinstance(raw_entries, list):
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, dict):
                    continue
                name = str(raw_entry.get("mechanism") or "").strip()
                if name and name not in names:
                    names.append(name)
        return names

    def _allowed_paths_for_components(self, components: list[str]) -> list[str]:
        paths: list[str] = []
        for component in components:
            root = str(component).strip().split("/", 1)[0]
            if root in REPO_EDIT_ROOTS and root not in paths:
                paths.append(root)
            for mapped_root in LOGICAL_COMPONENT_EDIT_ROOTS.get(root, ()):
                if mapped_root not in paths:
                    paths.append(mapped_root)
        if "tests" not in paths:
            paths.append("tests")
        return paths or ["harness", "tests"]

    def _slug(self, value: str) -> str:
        chars = []
        for char in value.lower():
            if char.isalnum():
                chars.append(char)
            elif chars and chars[-1] != "-":
                chars.append("-")
        return "".join(chars).strip("-") or "unknown"

    def _external_loop_controls(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "select_candidate",
                "description": (
                    "Choose exactly one feature candidate for the next scoped Codex update; "
                    "candidate selection must not impose time, round, attempt, or feature-count stops."
                ),
            },
            {
                "name": "adjust_worker_role",
                "description": "Switch --worker-role, for example worker_deepseek for tests or worker_gpt for formal runs.",
            },
            {
                "name": "adjust_run_cap",
                "description": (
                    "Record the legacy --run-task-cap audit value without shrinking "
                    "the campaign task list, truncating the per-round task list, or "
                    "stopping the outer loop."
                ),
            },
            {
                "name": "request_more_evidence",
                "description": "Rerun selected tasks when the current packet lacks verifier or trajectory evidence.",
            },
            {
                "name": "defer_update",
                "description": "Record no-op when validation contracts are not strong enough for a safe edit.",
            },
        ]

    def _loop_limit_contract(self, *, max_features: int) -> dict[str, Any]:
        contract = base_loop_limit_contract(
            notes={
                "all_loops": (
                    "Mission debug emits feature-selection evidence for the outer HL loop. "
                    "Its max_features, candidate count, target-task count, validation-contract "
                    "count, elapsed time, round, attempt, budget, cooldown, context-depth, "
                    "token, timeout, and max_turns values are audit or packet-size metadata "
                    "only; they must not stop master, Codex update, diagnostic/context, "
                    "validation/regression, mission-debug, or Worker loops."
                ),
                "master_loop": (
                    "Mission-debug packets may guide the master loop, but packet fields such "
                    "as max_features, elapsed time, rounds, attempts, task counts, budgets, "
                    "cooldowns, and token windows are audit metadata only and must not stop it."
                ),
                "codex_update_sub_agent": (
                    "Mission-debug evidence can select a Codex update slice, but it must not "
                    "turn time, round, interval, cooldown, attempt, K, feature, provider, or "
                    "budget fields into Codex update sub-agent stop conditions."
                ),
                "validation_regression_sub_agents": (
                    "Mission-selected validation contracts, regression lanes, snapshot "
                    "counts, selection caps, retries, transient cooldowns, validation "
                    "timeout references, project-test duration, and task-concurrency "
                    "values are audit, validation, retry, or throughput metadata only. "
                    "They may reject an unsafe candidate on real evidence, but must not "
                    "stop validation/regression, master, Codex update, diagnostic/context, "
                    "mission-debug, or Worker loops because a time, round, attempt, "
                    "snapshot-count, or timeout value was reached."
                ),
                "mission_debug_sub_agent": (
                    "max_features is retained for compatibility and reporting only. "
                    "Do not truncate feature candidates, target tasks, validation contracts, "
                    "or mission-debug execution because this value, a round count, or a "
                    "wall-clock value was reached."
                ),
            }
        )
        contract["mission_debug_sub_agent"].update(
            {
                "max_features_audit_only": max_features,
                "max_features_stop_condition": False,
            }
        )
        return contract

    def _score_history_from_task_results(self, task_results: list[dict[str, Any]]) -> dict[str, Any]:
        counts = self._status_counts(task_results, {})
        completed = max(len(task_results), 1)
        return {
            "score": round(counts["passed"] / completed, 4),
            "passed": counts["passed"],
            "failed": counts["failed"],
            "timeout": counts["timeout"],
            "error": counts["error"],
        }

    def _trial_result_dict(self, trial: TrialResult) -> dict[str, Any]:
        return {
            "trial_id": trial.trial_id,
            "task_id": trial.task_id,
            "domain": trial.task_domain.value,
            "difficulty": trial.task_difficulty.value,
            "status": trial.status.value,
            "score": trial.score,
            "verified": trial.verified,
            "model": trial.model_used,
            "wall_time_seconds": trial.wall_time_seconds,
            "harbor_job_dir": trial.harbor_job_dir,
        }

    def _status_counts(
        self,
        task_results: list[dict[str, Any]],
        score: dict[str, Any],
    ) -> Counter[str]:
        counts: Counter[str] = Counter()
        for item in task_results:
            status = str(item.get("status") or "").lower()
            if status:
                counts[status] += 1
        for status in ("passed", "failed", "timeout", "error"):
            if counts[status] == 0 and score.get(status) is not None:
                counts[status] = int(score.get(status) or 0)
        return counts

    def _tasks_with_status(
        self,
        task_results: list[dict[str, Any]],
        statuses: set[str],
        *,
        limit: int | None = None,
    ) -> list[str]:
        """Return all matching tasks; ``limit`` is legacy audit metadata only."""
        _ = limit
        tasks = []
        for item in task_results:
            status = str(item.get("status") or "").lower()
            task_id = item.get("task_id")
            task_id_text = str(task_id).strip() if task_id else ""
            if status in statuses and task_id_text and task_id_text not in tasks:
                tasks.append(task_id_text)
        return tasks

    def _all_task_ids(self, task_results: list[dict[str, Any]]) -> list[str]:
        tasks: list[str] = []
        for item in task_results:
            task_id = str(item.get("task_id") or "").strip()
            if task_id and task_id not in tasks:
                tasks.append(task_id)
        return tasks

    def _top_values(
        self,
        task_results: list[dict[str, Any]],
        key: str,
        *,
        limit: int = 5,
    ) -> dict[str, int]:
        """Return all ranked values; ``limit`` is legacy audit metadata only."""
        _ = limit
        values = Counter(str(item.get(key) or "unknown") for item in task_results)
        return dict(values.most_common())
