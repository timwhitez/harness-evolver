"""Deterministic failure categorization and component attribution."""

from __future__ import annotations

from dataclasses import dataclass, field

from hl.failure_mechanisms import (
    DEPENDENCY_PIVOT_MECHANISM_NAMES,
    PRIMARY_VERIFIER_CONTRACT_MECHANISM_NAMES,
    affected_components_for_failure_mechanism,
    dependency_loop_failure_category_for_trial,
    dependency_loop_mechanism_for_failure_category,
    failure_mechanisms_for_trial,
)
from hl.types import TrialResult, TrialStatus


@dataclass(frozen=True)
class AttributionResult:
    failure_category: str
    affected_components: list[str] = field(default_factory=list)
    component_confidence: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


class FailureAttributor:
    """Map trial evidence to failure categories and likely harness components."""

    def analyze(
        self,
        trial: TrialResult,
        *,
        tool_success_rate: float | None = None,
    ) -> AttributionResult:
        if trial.status == TrialStatus.PASSED and trial.verified:
            return AttributionResult(failure_category="passed")

        evidence = self._evidence(trial)
        text = "\n".join(evidence).lower()
        verifier_failure_text = self._verifier_failure_text(trial).lower()
        trajectory_text = self._trajectory_text(trial).lower()
        tool_rate = 1.0 if tool_success_rate is None else tool_success_rate
        timeout_phase = str(trial.metadata.get("timeout_phase") or "")

        if timeout_phase == "verifier_runtime_prepare":
            return self._result(
                "verifier_runtime_prepare_timeout",
                ["bench/network_environment", "bench/harbor"],
                evidence,
            )

        if trial.metadata.get("verifier_infra_error"):
            return self._result(
                "verifier_environment_error",
                ["bench/harbor", "verification/checks"],
                evidence,
            )

        if self._has_semantic_verifier_failure(verifier_failure_text, trial=trial):
            mechanism_result = self._primary_verifier_contract_result(trial, evidence)
            if mechanism_result is not None:
                return mechanism_result
            mechanism_result = self._recovery_mechanism_result(trial, evidence)
            if mechanism_result is not None:
                return mechanism_result
            if timeout_phase == "agent_execution" and trial.status == TrialStatus.TIMEOUT:
                return self._result(
                    "agent_timeout_with_verifier_mismatch",
                    [
                        "bench/agent",
                        "recovery/patterns",
                        "context/compaction",
                        "verification/checks",
                    ],
                    evidence,
                )
            return self._result(
                "verifier_mismatch",
                ["verification/checks", "harness/tools/verify"],
                evidence,
            )

        mechanism_result = self._primary_verifier_contract_result(trial, evidence)
        if mechanism_result is not None:
            return mechanism_result

        mechanism_result = self._recovery_mechanism_result(trial, evidence)
        if mechanism_result is not None:
            return mechanism_result

        if timeout_phase == "agent_execution":
            return self._result(
                "agent_execution_timeout",
                ["bench/agent", "recovery/patterns", "context/compaction"],
                evidence,
            )

        if timeout_phase:
            if timeout_phase == "verifier":
                return self._result(
                    "verifier_timeout",
                    ["bench/harbor", "verification/checks"],
                    evidence,
                )
            if timeout_phase == "environment_build":
                return self._result(
                    "environment_build_timeout",
                    ["bench/harbor", "bench/network_environment"],
                    evidence,
                )
            if timeout_phase == "environment_start":
                return self._result(
                    "environment_start_timeout",
                    ["bench/harbor", "bench/network_environment"],
                    evidence,
                )
            if timeout_phase in {"harbor_process", "harbor_cancelled"}:
                return self._result(
                    "harbor_process_timeout",
                    ["bench/harbor", "orchestration/run_campaign"],
                    evidence,
                )

        if trial.metadata.get("post_completion_agent_exception") or self._has_done_before_agent_exception(
            trial, text
        ):
            return self._result(
                "post_completion_agent_exception",
                ["bench/harbor", "bench/harbor_adapter", "bench/agent"],
                evidence,
            )

        if self._has_verifier_cache_permission_signal(verifier_failure_text):
            return self._result(
                "verifier_environment_error",
                ["bench/harbor", "bench/network_environment", "verification/checks"],
                evidence,
            )

        if self._has_any(
            verifier_failure_text,
            [
                "ssl certificate problem",
                "unable to get local issuer certificate",
                "curl: (60)",
                "/root/.local/bin/env",
                "uvx: command not found",
            ],
        ):
            return self._result(
                "verifier_environment_error",
                ["bench/harbor", "verification/checks"],
                evidence,
            )

        if trial.metadata.get("infra_error_detected") or self._has_any(
            text,
            [
                "docker compose command failed for environment",
                "failed to solve: process",
                "did not complete successfully: exit code: 5",
                "dl-cdn.alpinelinux.org",
            ],
        ):
            if trial.verified and trial.score < 1.0 and not trial.metadata.get(
                "verifier_infra_error"
            ):
                return self._result(
                    "verifier_mismatch",
                    ["verification/checks", "harness/tools/verify"],
                    evidence,
                )
            return self._result(
                "harbor_environment_error",
                ["bench/harbor", "bench/network_environment"],
                evidence,
            )

        if trial.status != TrialStatus.TIMEOUT and self._has_dependency_signal(trajectory_text):
            return self._result(
                "dependency_issue",
                ["tools/shell", "recovery/patterns"],
                evidence,
            )

        if trial.status == TrialStatus.TIMEOUT:
            return self._result(
                "timeout",
                ["recovery/patterns", "context/compaction"],
                evidence,
            )

        if self._has_any(text, ["timeout", "timed out"]):
            if self._is_verified_non_timeout_failure(trial):
                return self._result(
                    "verifier_mismatch",
                    ["verification/checks", "harness/tools/verify"],
                    evidence,
                )
            return self._result(
                "timeout",
                ["recovery/patterns", "context/compaction"],
                evidence,
            )

        if self._has_any(
            text,
            [
                "context length",
                "context window",
                "maximum context",
                "token limit",
                "too many tokens",
            ],
        ):
            return self._result("context_overflow", ["context/compaction"], evidence)

        if self._has_any(text, ["refusal", "refused", "policy violation", "safety policy"]):
            return self._result("model_refusal", ["prompts/system"], evidence)

        if self._has_any(
            text,
            [
                "tool call",
                "invalid json",
                "malformed json",
                "missing required",
                "invalid arguments",
                "schema validation",
            ],
        ):
            return self._result("tool_misuse", ["tools/correction"], evidence)

        if self._has_any(
            text,
            [
                "command not found",
                "no module named",
                "module not found",
                "package not found",
                "uvx command not found",
                "dependency",
                "could not resolve",
            ],
        ):
            return self._result(
                "dependency_issue",
                ["tools/shell", "recovery/patterns"],
                evidence,
            )

        if self._has_any(
            text,
            [
                "no such file",
                "file not found",
                "can't cd",
                "cannot cd",
                "not a directory",
                "workspace not found",
            ],
        ):
            return self._result(
                "entrypoint_miss",
                ["entrypoint/semantic", "tools/file_read"],
                evidence,
            )

        if self._has_any(
            text,
            [
                "artifact not found",
                "result.json not found",
                "reward.txt",
                "missing artifact",
                "trajectory not found",
            ],
        ):
            return self._result(
                "missing_artifact",
                ["bench/harbor", "verification/checks"],
                evidence,
            )

        if self._has_any(text, ["verifier", "assert", "pytest", "test failed", "reward"]):
            return self._result(
                "verifier_mismatch",
                ["verification/checks", "harness/tools/verify"],
                evidence,
            )

        if self._has_any(text, ["traceback", "exception", "harness", "adapter"]):
            return self._result(
                "harness_bug",
                ["bench/harbor_adapter", "bench/harbor"],
                evidence,
            )

        if tool_rate < 0.5:
            return self._result("tool_misuse", ["tools/correction"], evidence)

        return self._result(
            "task_misunderstanding",
            ["prompts/task", "planning/todo_enforcement"],
            evidence,
            base_confidence=0.45,
        )

    def _primary_verifier_contract_result(
        self,
        trial: TrialResult,
        evidence: list[str],
    ) -> AttributionResult | None:
        all_mechanisms = failure_mechanisms_for_trial(trial)
        primary_names = sorted(
            dict.fromkeys(
                mechanism.name
                for mechanism in all_mechanisms
                if mechanism.name in PRIMARY_VERIFIER_CONTRACT_MECHANISM_NAMES
            )
        )
        if not primary_names:
            return None
        components = self._components_for_mechanisms(
            sorted(dict.fromkeys(mechanism.name for mechanism in all_mechanisms))
        )
        for component in ["bench/agent", "harness/tools/verify", "verification/checks"]:
            if component not in components:
                components.append(component)
        return self._result(
            ",".join(primary_names),
            components,
            evidence,
        )

    def _components_for_mechanisms(
        self,
        mechanism_names: list[str],
        *,
        fallback: list[str] | None = None,
    ) -> list[str]:
        components: list[str] = []
        for mechanism_name in mechanism_names:
            for component in affected_components_for_failure_mechanism(mechanism_name):
                if component not in components:
                    components.append(component)
        return components or list(fallback or [])

    def _recovery_mechanism_result(
        self,
        trial: TrialResult,
        evidence: list[str],
    ) -> AttributionResult | None:
        mechanisms = failure_mechanisms_for_trial(trial)
        mechanism_names = sorted(dict.fromkeys(mechanism.name for mechanism in mechanisms))
        if not mechanism_names:
            return None

        dependency_category = dependency_loop_failure_category_for_trial(
            trial,
            mechanism_names,
        )
        if dependency_category:
            category_mechanism = dependency_loop_mechanism_for_failure_category(
                dependency_category
            )
            components = self._components_for_mechanisms(
                [category_mechanism],
                fallback=["bench/agent", "recovery/patterns"],
            )
            return self._result(dependency_category, components, evidence)

        recovery_names = [
            name for name in mechanism_names if name in DEPENDENCY_PIVOT_MECHANISM_NAMES
        ]
        if not recovery_names:
            return None
        return self._result(
            ",".join(recovery_names),
            self._components_for_mechanisms(
                recovery_names,
                fallback=["bench/agent", "recovery/patterns"],
            ),
            evidence,
        )

    def _result(
        self,
        failure_category: str,
        components: list[str],
        evidence: list[str],
        *,
        base_confidence: float = 0.7,
    ) -> AttributionResult:
        confidence = {
            component: round(max(0.1, base_confidence - index * 0.1), 2)
            for index, component in enumerate(components)
        }
        return AttributionResult(
            failure_category=failure_category,
            affected_components=components,
            component_confidence=confidence,
            evidence=evidence[:5],
        )

    def _evidence(self, trial: TrialResult) -> list[str]:
        evidence: list[str] = []
        evidence.extend(str(error) for error in trial.error_log[:8])
        for event in trial.trajectory[:40]:
            if not isinstance(event, dict):
                continue
            value = event.get("error") or event.get("stderr") or event.get("output")
            if value:
                evidence.append(str(value)[:1000])
        if trial.verifier_output:
            evidence.append(trial.verifier_output[:2000])
        verifier_logs = trial.metadata.get("verifier_logs")
        if verifier_logs:
            evidence.append(str(verifier_logs)[:2000])
        if trial.harbor_stderr:
            evidence.append(trial.harbor_stderr[:2000])
        if trial.harbor_stdout and not evidence:
            evidence.append(trial.harbor_stdout[:1000])
        for call in trial.tool_calls[:10]:
            if call.get("success") is False:
                evidence.append(str(call.get("error") or call.get("output") or call)[:1000])
        return [item for item in evidence if item]

    def _verifier_failure_text(self, trial: TrialResult) -> str:
        parts: list[str] = []
        parts.extend(str(error) for error in trial.error_log[:8])
        if trial.verifier_output:
            parts.append(trial.verifier_output[:2000])
        verifier_logs = trial.metadata.get("verifier_logs")
        if verifier_logs:
            parts.append(str(verifier_logs)[:2000])
        return "\n".join(parts)

    def _trajectory_text(self, trial: TrialResult) -> str:
        parts: list[str] = []
        for event in trial.trajectory[:40]:
            if not isinstance(event, dict):
                continue
            value = event.get("error") or event.get("stderr") or event.get("output")
            if value:
                parts.append(str(value)[:1000])
        for call in trial.tool_calls[:10]:
            if call.get("success") is False:
                parts.append(str(call.get("error") or call.get("output") or call)[:1000])
        return "\n".join(parts)

    def _has_done_before_agent_exception(self, trial: TrialResult, text: str) -> bool:
        if trial.status not in {TrialStatus.ERROR, TrialStatus.TIMEOUT}:
            return False
        saw_successful_done = False
        for event in trial.trajectory:
            if not isinstance(event, dict):
                continue
            if event.get("type") == "tool_call" and event.get("tool") == "done":
                saw_successful_done = event.get("success") is not False
        if not saw_successful_done:
            return False
        return self._has_any(
            text,
            [
                "command timed out after",
                "agent execution timed out",
                "runtimeerror",
                "agenttimeouterror",
            ],
        )

    def _is_verified_non_timeout_failure(self, trial: TrialResult) -> bool:
        if trial.status != TrialStatus.FAILED or not trial.verified:
            return False
        metadata = trial.metadata or {}
        authoritative_timeout_keys = [
            "timeout_phase",
            "timeout_source",
            "outer_harbor_timeout",
            "outer_harbor_interrupted",
            "timed_out_process",
        ]
        return not any(metadata.get(key) for key in authoritative_timeout_keys)

    def _has_semantic_verifier_failure(self, text: str, *, trial: TrialResult) -> bool:
        if not (
            trial.verified
            or "ctrf.json" in text
            or "test_outputs.py" in text
            or "pytest" in text
        ):
            return False
        return self._has_any(
            text,
            [
                "assertionerror",
                "valueerror:",
                '"raw_status": "call_failed"',
                "test failed in the call phase",
                "failed ../tests/",
                "failed test_outputs.py",
                "e       assert",
                "e       valueerror",
            ],
        )

    def _has_dependency_signal(self, text: str) -> bool:
        return self._has_any(
            text,
            [
                "command not found",
                "no module named",
                "module not found",
                "package not found",
                "dependency",
                "could not resolve",
                "no matching distribution found",
                "ssl certificate problem",
                "unable to get local issuer certificate",
                "certificate verify failed",
                "curl: (60)",
            ],
        )

    def _has_verifier_cache_permission_signal(self, text: str) -> bool:
        lowered = text.lower()
        if "/tmp/hl-verifier-cache" not in lowered and "distribution cache" not in lowered:
            return False
        return (
            "permission denied" in lowered
            or "os error 13" in lowered
            or "failed to write to the distribution cache" in lowered
            or "failed to rename file from /tmp/hl-verifier-cache" in lowered
        )

    def _has_any(self, text: str, needles: list[str]) -> bool:
        return any(needle in text for needle in needles)
