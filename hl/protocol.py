"""Abstract protocols for the Heuristic Learning system.

Every entity in the HL framework implements one of these protocols.
This maps directly to the HL definition from the Learning Beyond Gradients paper:

    HL 由程序代码构成，共享状态、动作、反馈、更新的闭环。
    更新对象从神经网络参数换成了软件结构。

    Policy     → 程序策略（代码规则、状态机、controller、MPC）
    State      → 显式变量、检测器、缓存
    Action     → 执行代码逻辑生成
    Feedback   → coding agent context（testcase、环境反馈、日志、回放）
    Update     → coding agent 直接修改代码
    Memory     → trials、summary、失败原因、回放、版本 diff
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from hl.types import (
    FeedbackSignal,
    HarnessPatch,
    RegressionSnapshot,
    TrialResult,
    TrialSummary,
)


# ── Policy ──────────────────────────────────────────────────────────
# "Policy 由代码构成，可以是代码规则、状态机、controller、MPC、宏动作"


@runtime_checkable
class Policy(Protocol):
    """A harness component that determines agent behavior when rendered.

    Every harness component (prompt template, tool definition, planning
    strategy, recovery pattern) is a Policy.  The agent's behavior is the
    sum of all active policies rendered into its context window.
    """

    name: str
    version: str
    dependencies: list[str]

    def render(self, context: dict[str, Any]) -> str:
        """Render this policy into tokens for the agent context window."""
        ...

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        ...


# ── StateProvider ────────────────────────────────────────────────────
# "State 通常显式写成变量、检测器、缓存等可读表示的东西"


class StateProvider(ABC):
    """Captures environment and codebase state for the agent.

    State is explicit, readable, serializable — not hidden in weights.
    """

    name: str

    @abstractmethod
    def snapshot(self, env: Any) -> dict[str, Any]:
        """Capture current state from the environment."""
        ...

    @abstractmethod
    def diff(self, before: dict[str, Any], after: dict[str, Any]) -> str:
        """Human-readable diff between two state snapshots."""
        ...


# ── FeedbackChannel ──────────────────────────────────────────────────
# "Feedback 由 coding agent 根据 context 提供，
#  testcase、环境反馈、日志、回放都算 context"


class FeedbackChannel(ABC):
    """Collects and structures feedback from a trial run.

    Multiple channels can coexist: verifier score, error traces,
    trajectory analysis, tool call logs.  Each channel extracts
    a different signal from the same trial.
    """

    name: str

    @abstractmethod
    def collect(self, trial: TrialResult) -> FeedbackSignal:
        """Extract structured feedback from a completed trial."""
        ...

    @abstractmethod
    def categorize(self, signal: FeedbackSignal) -> list[str]:
        """Map feedback to affected harness component names."""
        ...

    @abstractmethod
    def severity(self, signal: FeedbackSignal) -> float:
        """Return 0.0 (trivial) to 1.0 (critical failure)."""
        ...


# ── MemoryStore ──────────────────────────────────────────────────────
# "Memory 可以显式记录 trials、summary、失败原因、回放、版本 diff"


class MemoryStore(ABC):
    """File-system memory for trials, summaries, regressions, and diffs.

    Unlike neural network memory (replay buffers, weight updates),
    HL memory is explicit, readable, deletable, and refactorable.
    """

    base_path: str

    @abstractmethod
    def record_trial(self, trial: TrialResult, *, append_scoreboard: bool = True) -> str:
        """Store a trial result, return trial_id."""
        ...

    @abstractmethod
    def get_trial(self, trial_id: str) -> TrialResult:
        """Retrieve a stored trial."""
        ...

    @abstractmethod
    def list_trials(self, task_id: str | None = None) -> list[str]:
        """List trial IDs, optionally filtered by task."""
        ...

    @abstractmethod
    def record_summary(self, summary: TrialSummary) -> str:
        """Store an aggregated summary."""
        ...

    @abstractmethod
    def get_latest_summary(self) -> TrialSummary | None:
        """Get the most recent summary."""
        ...

    @abstractmethod
    def save_regression(self, task_id: str, snapshot: RegressionSnapshot) -> None:
        """Save a known-good state for regression testing.

        This is how HL 'remembers' solved tasks — not in weights,
        but as explicit, versioned snapshots scoped to comparable model config.
        """
        ...

    @abstractmethod
    def check_regression(self, task_id: str, result: TrialResult) -> bool:
        """Return True if a previously-solved task now fails."""
        ...

    @abstractmethod
    def save_patch(self, patch: HarnessPatch) -> str:
        """Save a harness edit patch with its diff and rationale."""
        ...

    @abstractmethod
    def list_patches(self, component_name: str | None = None) -> list[str]:
        """List patches, optionally filtered by component."""
        ...


# ── UpdateEngine ─────────────────────────────────────────────────────
# "Update 由 coding agent 直接修改代码"


class UpdateEngine(ABC):
    """Executes harness edits via the meta-coding-agent.

    The update does not use backpropagation.  The meta-agent reads
    failure context and directly edits harness component files.
    """

    name: str

    @abstractmethod
    def analyze_failures(
        self, feedback_signals: list[FeedbackSignal], trials: list[TrialResult]
    ) -> list[dict[str, Any]]:
        """Analyze failures and return structured findings.

        Each finding maps a failure to a root cause component.
        """
        ...

    @abstractmethod
    def suggest_edits(
        self, findings: list[dict[str, Any]], current_harness: dict[str, Any]
    ) -> list[HarnessPatch]:
        """Generate candidate harness edits from failure analysis."""
        ...

    @abstractmethod
    def apply_patch(self, patch: HarnessPatch) -> bool:
        """Apply a harness patch. Returns True on success."""
        ...

    @abstractmethod
    def rollback_patch(self, patch: HarnessPatch) -> bool:
        """Roll back a previously applied patch."""
        ...
