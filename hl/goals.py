"""Persistent HL campaign goals and budget accounting."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from hl.loop_limits import is_limit_terminal_reason, normalize_limit_reason_text


GoalStatus = Literal["active", "complete", "stopped"]
TokenBudgetScope = Literal["iteration", "campaign"]
_UNSET = object()


_EXPLICIT_STOP_REASON_MARKERS = (
    "explicit user",
    "user requested",
    "user stop",
    "manual stop",
    "operator stop",
    "external cancellation",
    "process cancellation",
)

_EXPLICIT_COMPLETE_REASON_MARKERS = (
    "explicit user",
    "user requested",
    "manual complete",
    "manual completion",
    "operator complete",
    "operator completion",
    "explicit campaign goal complete",
    "explicit local target completion",
    "submit terminal action",
    "one-shot submit terminal action",
    "submit gate passed",
    "leaderboard submit",
)


def _is_legacy_audit_only_stop_reason(reason: str) -> bool:
    if is_explicit_goal_stop_reason(reason) or is_explicit_goal_completion_reason(reason):
        return False
    return is_limit_terminal_reason(reason)


def is_explicit_goal_stop_reason(reason: str) -> bool:
    """Return true only for operator-owned stop/cancel reasons."""
    normalized = normalize_limit_reason_text(reason)
    return any(marker in normalized for marker in _EXPLICIT_STOP_REASON_MARKERS)


def is_explicit_goal_completion_reason(reason: str) -> bool:
    """Return true only for explicit completion or submit-terminal reasons."""
    normalized = normalize_limit_reason_text(reason)
    return any(marker in normalized for marker in _EXPLICIT_COMPLETE_REASON_MARKERS)


def _normalize_stopped_payload(raw_goal: dict[str, object], *, reason: str) -> None:
    raw_goal["legacy_status"] = "stopped"
    raw_goal["status"] = "active"
    raw_goal["limit_stop_audit_only"] = True
    raw_goal["completion_reason"] = reason


def _normalize_complete_payload(raw_goal: dict[str, object], *, reason: str) -> None:
    raw_goal["legacy_status"] = "complete"
    raw_goal["status"] = "active"
    raw_goal["limit_completion_audit_only"] = True
    raw_goal["completed_at"] = None
    raw_goal["completion_reason"] = reason


def normalize_goal_status_payload(raw_goal: dict[str, object]) -> None:
    """Normalize historical limit-driven goal statuses to active audit state."""
    status = str(raw_goal.get("status") or "")
    if status == "budget_exhausted":
        raw_goal["legacy_status"] = "budget_exhausted"
        raw_goal["status"] = "active"
        raw_goal["budget_exhaustion_audit_only"] = True
        raw_goal["completion_reason"] = (
            "Legacy budget_exhausted status was normalized to active because "
            "budget, token, and wall-time fields are audit metadata only."
        )
        return
    if status == "complete" and _is_legacy_audit_only_stop_reason(
        str(raw_goal.get("completion_reason") or "")
    ):
        _normalize_complete_payload(
            raw_goal,
            reason=(
                "Legacy complete status from a time, round, token, budget, "
                "patience, cooldown, max_turns, deadline, or timeout limit was "
                "normalized to active because those fields are audit metadata "
                "only."
            ),
        )
        return
    if status == "complete" and not is_explicit_goal_completion_reason(
        str(raw_goal.get("completion_reason") or "")
    ):
        _normalize_complete_payload(
            raw_goal,
            reason=(
                "Non-explicit complete status was normalized to active because "
                "master, sub-agent, Codex update, diagnostic/context, "
                "validation/regression, mission-debug, and Worker loops may only "
                "stop on explicit user/operator completion, one-shot submit "
                "terminal action, external cancellation, or hard non-limit "
                "process errors."
            ),
        )
        raw_goal["non_explicit_completion_audit_only"] = True
        return
    if status == "stopped" and _is_legacy_audit_only_stop_reason(
        str(raw_goal.get("completion_reason") or "")
    ):
        _normalize_stopped_payload(
            raw_goal,
            reason=(
                "Legacy stopped status from a time, round, token, budget, patience, "
                "cooldown, max_turns, deadline, or timeout limit was normalized to "
                "active because those fields are audit metadata only."
            ),
        )
        return
    if status == "stopped" and not is_explicit_goal_stop_reason(
        str(raw_goal.get("completion_reason") or "")
    ):
        _normalize_stopped_payload(
            raw_goal,
            reason=(
                "Non-explicit stopped status was normalized to active because "
                "master, sub-agent, Codex update, diagnostic/context, "
                "validation/regression, mission-debug, and Worker loops may only "
                "stop on explicit user/operator stop, "
                "external cancellation, completion, submit terminal action, or hard "
                "non-limit process errors."
            ),
        )


class BudgetUsage(BaseModel):
    worker_input_tokens: int = 0
    worker_output_tokens: int = 0
    worker_cache_tokens: int = 0
    codex_input_tokens: int = 0
    codex_output_tokens: int = 0
    harbor_wall_time_seconds: float = 0.0
    patch_count: int = 0
    last_iteration_tokens: int = 0
    max_iteration_tokens: int = 0
    token_budget_overruns: int = 0
    token_budget_observations: int = 0
    wall_time_budget_observations: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.worker_input_tokens
            + self.worker_output_tokens
            + self.worker_cache_tokens
            + self.codex_input_tokens
            + self.codex_output_tokens
        )


class GoalState(BaseModel):
    objective: str
    status: GoalStatus = "active"
    score_target: float | None = None
    guard_convergence_score_floor: float | None = None
    guard_budget_baseline_total: int | None = None
    guard_budget_target_total: int | None = None
    guard_budget_reduction_target_fraction: float | None = None
    convergence_plateau_rounds: int | None = None
    convergence_min_score_delta: float | None = None
    token_budget: int | None = None
    token_budget_scope: TokenBudgetScope = "iteration"
    wall_time_budget_seconds: int | None = None
    submit_when_reached: bool = False
    best_score: float = 0.0
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    completion_reason: str = ""

    def remaining_tokens(self) -> int | None:
        if self.token_budget is None:
            return None
        if self.token_budget_scope == "iteration":
            return self.token_budget
        return max(0, self.token_budget - self.usage.total_tokens)

    def latest_iteration_remaining_tokens(self) -> int | None:
        if self.token_budget is None:
            return None
        return max(0, self.token_budget - self.usage.last_iteration_tokens)

    def remaining_wall_time(self) -> float | None:
        if self.wall_time_budget_seconds is None:
            return None
        return max(0.0, self.wall_time_budget_seconds - self.usage.harbor_wall_time_seconds)


class GoalStore:
    """Campaign goal store.

    When ``path`` is omitted the store is in-memory only. Campaign CLIs pass an
    explicit path under the configured memory root; library construction and
    tests should not implicitly read or create repository runtime state.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._goal: GoalState | None = None

    def get_goal(self) -> GoalState | None:
        if self.path is None:
            if self._goal is None:
                return None
            self._goal, _ = self._normalize_loaded_goal(self._goal)
            return self._goal
        if not self.path.exists():
            return None
        raw_goal = json.loads(self.path.read_text())
        if isinstance(raw_goal, dict):
            normalize_goal_status_payload(raw_goal)
        goal = GoalState.model_validate(raw_goal)
        goal, _ = self._normalize_loaded_goal(goal)
        return goal

    def create_goal(
        self,
        objective: str,
        *,
        score_target: float | None = None,
        guard_convergence_score_floor: float | None = None,
        guard_budget_baseline_total: int | None = None,
        guard_budget_target_total: int | None = None,
        guard_budget_reduction_target_fraction: float | None = None,
        convergence_plateau_rounds: int | None = None,
        convergence_min_score_delta: float | None = None,
        token_budget: int | None = None,
        token_budget_scope: TokenBudgetScope = "iteration",
        wall_time_budget_seconds: int | None = None,
        submit_when_reached: bool = False,
    ) -> GoalState:
        existing = self.get_goal()
        if existing and existing.status == "active":
            raise RuntimeError("An active goal already exists")
        token_budget = self._normalize_optional_positive_budget(token_budget)
        wall_time_budget_seconds = self._normalize_optional_positive_budget(
            wall_time_budget_seconds
        )
        goal = GoalState(
            objective=objective,
            score_target=score_target,
            guard_convergence_score_floor=guard_convergence_score_floor,
            guard_budget_baseline_total=self._normalize_optional_positive_budget(
                guard_budget_baseline_total
            ),
            guard_budget_target_total=self._normalize_optional_positive_budget(
                guard_budget_target_total
            ),
            guard_budget_reduction_target_fraction=guard_budget_reduction_target_fraction,
            convergence_plateau_rounds=self._normalize_optional_positive_budget(
                convergence_plateau_rounds
            ),
            convergence_min_score_delta=convergence_min_score_delta,
            token_budget=token_budget,
            token_budget_scope=token_budget_scope,
            wall_time_budget_seconds=wall_time_budget_seconds,
            submit_when_reached=submit_when_reached,
        )
        self._save(goal)
        return goal

    def update_budget(
        self,
        *,
        token_budget: int | None | object = _UNSET,
        token_budget_scope: TokenBudgetScope | None = None,
        wall_time_budget_seconds: int | None | object = _UNSET,
    ) -> GoalState:
        goal = self.get_goal()
        if goal is None:
            raise FileNotFoundError("No goal exists")
        if token_budget is not _UNSET:
            goal.token_budget = self._normalize_optional_positive_budget(token_budget)
        if token_budget_scope is not None:
            goal.token_budget_scope = token_budget_scope
        if wall_time_budget_seconds is not _UNSET:
            goal.wall_time_budget_seconds = self._normalize_optional_positive_budget(
                wall_time_budget_seconds
            )
        goal, _ = self._normalize_loaded_goal(goal)
        goal.updated_at = datetime.now()
        self._save(goal)
        return goal

    def update_usage(
        self,
        *,
        worker_tokens: dict[str, int] | None = None,
        codex_tokens: dict[str, int] | None = None,
        harbor_wall_time_seconds: float = 0.0,
        patch_count: int = 0,
        best_score: float | None = None,
    ) -> GoalState | None:
        goal = self.get_goal()
        if goal is None:
            return None
        latest_iteration_tokens = 0
        if worker_tokens:
            latest_iteration_tokens += int(worker_tokens.get("input", 0))
            latest_iteration_tokens += int(worker_tokens.get("output", 0))
            latest_iteration_tokens += int(worker_tokens.get("cache", 0))
            goal.usage.worker_input_tokens += int(worker_tokens.get("input", 0))
            goal.usage.worker_output_tokens += int(worker_tokens.get("output", 0))
            goal.usage.worker_cache_tokens += int(worker_tokens.get("cache", 0))
        if codex_tokens:
            latest_iteration_tokens += int(codex_tokens.get("input", 0))
            latest_iteration_tokens += int(codex_tokens.get("output", 0))
            goal.usage.codex_input_tokens += int(codex_tokens.get("input", 0))
            goal.usage.codex_output_tokens += int(codex_tokens.get("output", 0))
        goal.usage.last_iteration_tokens = latest_iteration_tokens
        goal.usage.max_iteration_tokens = max(
            goal.usage.max_iteration_tokens,
            latest_iteration_tokens,
        )
        if (
            goal.token_budget_scope == "iteration"
            and goal.token_budget is not None
            and latest_iteration_tokens >= goal.token_budget
        ):
            goal.usage.token_budget_overruns += 1
            goal.usage.token_budget_observations += 1
            goal.completion_reason = (
                "Latest iteration crossed the per-iteration token budget "
                "reference. This is an audit observation only; the master, "
                "sub-agent, validation/regression, mission-debug, context, "
                "Codex update, and Worker loops remain active."
            )
        elif goal.completion_reason.startswith(
            "Latest iteration crossed the per-iteration token budget reference"
        ):
            goal.completion_reason = ""
        goal.usage.harbor_wall_time_seconds += harbor_wall_time_seconds
        goal.usage.patch_count += patch_count
        if best_score is not None:
            goal.best_score = max(goal.best_score, best_score)
        if goal.status == "active" and self._budget_exhausted(goal):
            if self._campaign_token_budget_observed(goal):
                goal.usage.token_budget_observations += 1
            if self._wall_time_budget_observed(goal):
                goal.usage.wall_time_budget_observations += 1
            goal.completion_reason = (
                "Configured goal budget reference was crossed. This is an audit "
                "observation only; the master, sub-agent, validation/regression, "
                "mission-debug, context, Codex update, and Worker loops remain "
                "active until explicit user goal completion/stop, one-shot submit "
                "terminal action, external cancellation, or a hard non-limit "
                "validation/regression process error."
            )
        goal.updated_at = datetime.now()
        self._save(goal)
        return goal

    def update_goal(self, status: GoalStatus, *, reason: str = "") -> GoalState:
        goal = self.get_goal()
        if goal is None:
            raise FileNotFoundError("No goal exists")
        if status == "budget_exhausted":
            status = "active"
            reason = (
                reason
                or "Budget exhaustion is audit metadata only; the HL loop remains active."
            )
        elif status == "complete" and _is_legacy_audit_only_stop_reason(reason):
            status = "active"
            reason = (
                "Limit-driven complete status was normalized to active because "
                "master, sub-agent, Codex update, diagnostic/context, "
                "mission-debug, validation/regression, and Worker loops must "
                "not stop on time, round, token, budget, patience, cooldown, "
                "max_turns, deadline, timeout, or other limit metadata. "
                f"Original reason: {reason}"
            )
        elif status == "complete" and not is_explicit_goal_completion_reason(reason):
            status = "active"
            reason = (
                reason
                or "Non-explicit complete status was treated as audit metadata only."
            )
            reason = (
                "Non-explicit complete status was normalized to active because "
                "master, sub-agent, Codex update, diagnostic/context, "
                "validation/regression, mission-debug, and Worker loops must not "
                "stop on time, round, token, budget, patience, cooldown, "
                "max_turns, deadline, timeout, attempt, cap, count, or other "
                "limit metadata. Original reason: "
                f"{reason}"
            )
        elif status == "stopped" and _is_legacy_audit_only_stop_reason(reason):
            status = "active"
            reason = (
                "Limit-driven stopped status was normalized to active because "
                "master, sub-agent, Codex update, diagnostic/context, "
                "mission-debug, validation/regression, and Worker loops must "
                "not stop on time, round, token, budget, patience, cooldown, "
                "max_turns, deadline, timeout, attempt, cap, count, or other "
                "limit metadata. Original reason: "
                f"{reason}"
            )
        elif status == "stopped" and not is_explicit_goal_stop_reason(reason):
            status = "active"
            reason = (
                reason
                or "Non-explicit stopped status was treated as audit metadata only."
            )
            reason = (
                "Non-explicit stopped status was normalized to active because "
                "master, sub-agent, Codex update, diagnostic/context, "
                "validation/regression, mission-debug, and Worker loops must not "
                "stop on time, round, token, budget, patience, cooldown, "
                "max_turns, deadline, timeout, or other limit metadata. Original reason: "
                f"{reason}"
            )
        goal.status = status
        goal.completion_reason = reason
        goal.updated_at = datetime.now()
        if status == "complete":
            goal.completed_at = goal.updated_at
        elif status == "active":
            goal.completed_at = None
        self._save(goal)
        return goal

    def continuation_prompt(self) -> str:
        goal = self.get_goal()
        if goal is None:
            return "No active HL campaign goal."
        return (
            f"HL campaign goal: {goal.objective}\n"
            f"Status: {goal.status}\n"
            f"Best score: {goal.best_score}\n"
            f"Score target: {goal.score_target}\n"
            f"Guard fixed-eval score floor: {goal.guard_convergence_score_floor}\n"
            f"Guard budget baseline total: {goal.guard_budget_baseline_total}\n"
            f"Guard budget target total: {goal.guard_budget_target_total}\n"
            f"Guard budget reduction target fraction: {goal.guard_budget_reduction_target_fraction}\n"
            f"Convergence plateau rounds: {goal.convergence_plateau_rounds}\n"
            f"Convergence min score delta: {goal.convergence_min_score_delta}\n"
            f"Token budget scope: {goal.token_budget_scope}\n"
            f"Token budget: {goal.token_budget}\n"
            f"Cumulative tokens used: {goal.usage.total_tokens}\n"
            f"Latest iteration tokens: {goal.usage.last_iteration_tokens}\n"
            f"Remaining tokens for next iteration/campaign: {goal.remaining_tokens()}\n"
            f"Remaining wall time seconds: {goal.remaining_wall_time()}\n"
            "Budget exhaustion is not success; complete only after configured gates pass."
        )

    def _budget_exhausted(self, goal: GoalState) -> bool:
        if self._campaign_token_budget_observed(goal):
            return True
        if self._wall_time_budget_observed(goal):
            return True
        return False

    def _campaign_token_budget_observed(self, goal: GoalState) -> bool:
        return (
            goal.token_budget_scope == "campaign"
            and goal.token_budget is not None
            and goal.usage.total_tokens >= goal.token_budget
        )

    def _wall_time_budget_observed(self, goal: GoalState) -> bool:
        return (
            goal.wall_time_budget_seconds is not None
            and goal.usage.harbor_wall_time_seconds >= goal.wall_time_budget_seconds
        )

    def _normalize_loaded_goal(self, goal: GoalState) -> tuple[GoalState, bool]:
        changed = False
        if str(goal.status) == "budget_exhausted":
            self._normalize_goal_status(
                goal,
                reason=(
                    "Legacy budget_exhausted status was normalized to active because "
                    "budget, token, and wall-time fields are audit metadata only."
                ),
            )
            changed = True
        elif goal.status == "complete" and _is_legacy_audit_only_stop_reason(
            goal.completion_reason
        ):
            self._normalize_goal_status(
                goal,
                reason=(
                    "Legacy complete status from a time, round, token, budget, "
                    "patience, cooldown, max_turns, deadline, or timeout limit was "
                    "normalized to active because those fields are audit metadata "
                    "only."
                ),
            )
            changed = True
        elif goal.status == "complete" and not is_explicit_goal_completion_reason(
            goal.completion_reason
        ):
            self._normalize_goal_status(
                goal,
                reason=(
                    "Non-explicit complete status was normalized to active because "
                    "master, sub-agent, Codex update, diagnostic/context, "
                    "validation/regression, mission-debug, and Worker loops may "
                    "only stop on explicit user/operator completion, one-shot "
                    "submit terminal action, external cancellation, or hard "
                    "non-limit process errors."
                ),
            )
            changed = True
        elif goal.status == "stopped" and _is_legacy_audit_only_stop_reason(
            goal.completion_reason
        ):
            self._normalize_goal_status(
                goal,
                reason=(
                    "Legacy stopped status from a time, round, token, budget, "
                    "patience, cooldown, max_turns, deadline, or timeout limit was "
                    "normalized to active because those fields are audit metadata "
                    "only."
                ),
            )
            changed = True
        elif goal.status == "stopped" and not is_explicit_goal_stop_reason(
            goal.completion_reason
        ):
            self._normalize_goal_status(
                goal,
                reason=(
                    "Non-explicit stopped status was normalized to active because "
                    "master, sub-agent, Codex update, diagnostic/context, "
                    "validation/regression, mission-debug, and Worker loops may "
                    "only stop on explicit user/operator stop, external cancellation, "
                    "completion, submit terminal action, or hard non-limit process errors."
                ),
            )
            changed = True
        normalized_token_budget = self._normalize_optional_positive_budget(goal.token_budget)
        if normalized_token_budget != goal.token_budget:
            goal.token_budget = normalized_token_budget
            changed = True
        normalized_wall_time = self._normalize_optional_positive_budget(
            goal.wall_time_budget_seconds
        )
        if normalized_wall_time != goal.wall_time_budget_seconds:
            goal.wall_time_budget_seconds = normalized_wall_time
            changed = True
        normalized_guard_baseline = self._normalize_optional_positive_budget(
            goal.guard_budget_baseline_total
        )
        if normalized_guard_baseline != goal.guard_budget_baseline_total:
            goal.guard_budget_baseline_total = normalized_guard_baseline
            changed = True
        normalized_guard_target = self._normalize_optional_positive_budget(
            goal.guard_budget_target_total
        )
        if normalized_guard_target != goal.guard_budget_target_total:
            goal.guard_budget_target_total = normalized_guard_target
            changed = True
        normalized_plateau = self._normalize_optional_positive_budget(
            goal.convergence_plateau_rounds
        )
        if normalized_plateau != goal.convergence_plateau_rounds:
            goal.convergence_plateau_rounds = normalized_plateau
            changed = True
        return goal, changed

    def _normalize_goal_status(
        self,
        goal: GoalState,
        *,
        reason: str,
    ) -> None:
        goal.status = "active"
        goal.completion_reason = reason
        goal.completed_at = None
        goal.updated_at = datetime.now()

    def _normalize_optional_positive_budget(self, value: int | None | object) -> int | None:
        if value is None or value is _UNSET:
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None

    def _save(self, goal: GoalState) -> None:
        if self.path is None:
            self._goal = goal
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(goal.model_dump_json(indent=2))
