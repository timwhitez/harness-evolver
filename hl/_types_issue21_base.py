"""Shared data types for the Heuristic Learning system.

Every type here is a Pydantic model for serialization/deserialization.
All trial data must be JSON-serializable for file-system memory storage.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskDomain(str, Enum):
    SOFTWARE_ENGINEERING = "software_engineering"
    SCIENTIFIC_COMPUTING = "scientific_computing"
    SECURITY = "security"
    SYSTEM_ADMINISTRATION = "system_administration"
    ML_ENGINEERING = "ml_engineering"
    DATA_ENGINEERING = "data_engineering"
    WEB_DEVELOPMENT = "web_development"
    DEVOPS = "devops"
    NETWORKING = "networking"
    DATABASE = "database"


class TrialStatus(str, Enum):
    RUNNING = "running"
    UNVERIFIED = "unverified"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


# Timeout phases that indicate a Harbor/environment infrastructure failure rather
# than a Worker capability failure. Kept here as the single source of truth for
# every module that needs to attribute or exclude infrastructure failures.
INFRASTRUCTURE_TIMEOUT_PHASES = frozenset(
    {
        "environment_start",
        "environment_build",
        "verifier_runtime_prepare",
        "harbor_process",
        "harbor_cancelled",
    }
)


def trial_is_infrastructure_failure(trial: Any) -> bool:
    """Return True when a trial failed for infrastructure/environment reasons
    (e.g. prebuilt image registry denial, environment build/start timeout) rather
    than Worker capability. Such trials must not count against Worker score.
    """
    metadata = getattr(trial, "metadata", {}) or {}
    if metadata.get("score_exclusion_reason") == "infrastructure_error":
        return True
    if metadata.get("infra_error_detected") and not bool(getattr(trial, "verified", False)):
        return True
    timeout_phase = str(metadata.get("timeout_phase") or "")
    return timeout_phase in INFRASTRUCTURE_TIMEOUT_PHASES


class ComponentVersion(BaseModel):
    """Versioned snapshot of a single harness component."""

    name: str
    version: str
    git_commit: str
    content_hash: str
    dependencies: list[str] = Field(default_factory=list)


class HarnessPatch(BaseModel):
    """A single edit to a harness component, produced by the meta-agent."""

    component_name: str
    before_version: str
    after_version: str
    file_path: str
    diff: str
    rationale: str
    failure_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class FeedbackSignal(BaseModel):
    """Structured feedback extracted from a trial run."""

    trial_id: str
    task_id: str
    status: TrialStatus
    score: float = 0.0
    affected_components: list[str] = Field(default_factory=list)
    failure_category: str = ""
    component_confidence: dict[str, float] = Field(default_factory=dict)
    error_summary: str = ""
    tool_call_success_rate: float = 1.0
    trajectory_length: int = 0
    wall_time_seconds: float = 0.0
    raw_errors: list[str] = Field(default_factory=list)


class TrialResult(BaseModel):
    """Complete result of a single TerminalBench task execution."""

    trial_id: str
    task_id: str
    task_domain: TaskDomain
    task_difficulty: TaskDifficulty
    status: TrialStatus
    score: float = 0.0
    harness_version: str = "0.1.0"
    component_versions: dict[str, ComponentVersion] = Field(default_factory=dict)
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    error_log: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    wall_time_seconds: float = 0.0
    model_used: str = ""
    token_usage: dict[str, int] = Field(default_factory=dict)
    verified: bool = False
    verifier_output: str = ""
    harbor_job_dir: str = ""
    harbor_trial_dir: str = ""
    harbor_stdout: str = ""
    harbor_stderr: str = ""
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class TrialSummary(BaseModel):
    """Aggregated summary across multiple trials."""

    summary_id: str
    trial_ids: list[str] = Field(default_factory=list)
    total_tasks: int = 0
    passed: int = 0
    failed: int = 0
    timeout: int = 0
    error: int = 0
    infrastructure_excluded: int = 0
    scored_tasks: int = 0
    overall_score: float = 0.0
    per_domain_scores: dict[str, float] = Field(default_factory=dict)
    per_difficulty_scores: dict[str, float] = Field(default_factory=dict)
    component_impacts: dict[str, float] = Field(default_factory=dict)
    patches_applied: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class RegressionSnapshot(BaseModel):
    """Golden snapshot of a solved task — the validation contract."""

    task_id: str
    harness_version: str
    model_scope: str = ""
    scope_config: dict[str, str] = Field(default_factory=dict)
    component_hashes: dict[str, str] = Field(default_factory=dict)
    solved_at: datetime = Field(default_factory=datetime.now)
    verification_output: str = ""
    required_assertions: list[str] = Field(default_factory=list)
    source_trial_id: str = ""
    source_summary_id: str = ""
    validation_status: str = "stable"
    invalidation_reason: str = ""
    regression_runs: int = 0
    regression_failures: int = 0
    regression_transient_failures: int = 0
    last_regression_status: str = ""
    last_regression_at: datetime | None = None
    last_regression_wall_time_seconds: float = 0.0
    regression_cooldown_until: datetime | None = None
    regression_cooldown_reason: str = ""
