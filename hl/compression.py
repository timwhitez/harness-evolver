"""Absorb/compress operations for HL memory and harness clutter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CompressionPlan:
    apply: bool
    reasons: list[str] = field(default_factory=list)
    before: dict[str, int] = field(default_factory=dict)
    after_estimate: dict[str, int] = field(default_factory=dict)
    preserved_paths: list[str] = field(default_factory=list)


class CompressionEngine:
    """Decides when a Codex compression task should be requested."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        prompt_length_threshold: int = 60000,
        recovery_rule_threshold: int = 80,
        touched_component_threshold: int = 8,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.prompt_length_threshold = prompt_length_threshold
        self.recovery_rule_threshold = recovery_rule_threshold
        self.touched_component_threshold = touched_component_threshold

    def dry_run(self, patch_metadata: list[dict[str, Any]] | None = None) -> CompressionPlan:
        patch_metadata = patch_metadata or []
        before = {
            "prompt_chars": self._total_chars(self.repo_root / "harness" / "prompts"),
            "recovery_rule_lines": self._line_count(self.repo_root / "harness" / "recovery"),
            "touched_components": len(
                {
                    component
                    for patch in patch_metadata
                    for component in patch.get("touched_components", [])
                }
            ),
        }
        reasons: list[str] = []
        if before["prompt_chars"] > self.prompt_length_threshold:
            reasons.append("prompt length threshold exceeded")
        if before["recovery_rule_lines"] > self.recovery_rule_threshold:
            reasons.append("recovery rule threshold exceeded")
        if before["touched_components"] > self.touched_component_threshold:
            reasons.append("too many touched components in recent patches")

        return CompressionPlan(
            apply=bool(reasons),
            reasons=reasons,
            before=before,
            after_estimate={
                "prompt_chars": min(before["prompt_chars"], self.prompt_length_threshold),
                "recovery_rule_lines": min(
                    before["recovery_rule_lines"], self.recovery_rule_threshold
                ),
            },
            preserved_paths=[
                "trials/regressions",
                "trials/submissions",
                "trials/runs",
                "trials/diffs",
            ],
        )

    def build_codex_instruction(self, plan: CompressionPlan) -> str:
        return (
            "Compress HarnessEvolver harness policy while preserving behavior.\n"
            f"Reasons: {', '.join(plan.reasons) or 'manual request'}\n"
            f"Before: {plan.before}\n"
            "Do not delete regression snapshots, submit records, raw run artifacts, "
            "or benchmark task definitions. Preserve existing tests and add focused "
            "tests when behavior changes."
        )

    def _total_chars(self, path: Path) -> int:
        if not path.exists():
            return 0
        return sum(
            len(file.read_text(errors="replace"))
            for file in path.rglob("*.py")
            if file.is_file()
        )

    def _line_count(self, path: Path) -> int:
        if not path.exists():
            return 0
        return sum(
            len(file.read_text(errors="replace").splitlines())
            for file in path.rglob("*.py")
            if file.is_file()
        )
