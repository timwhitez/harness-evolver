"""Guards that keep host-side HL memory out of Worker task tools."""

from __future__ import annotations

from typing import Any

from harness.tools.base import policy_guard_metadata


HOST_MEMORY_BLOCKED_BY = "host_memory_guard"
HOST_MEMORY_SEMANTIC_KIND = "blocked_host_memory_search"


def host_memory_access_reason(text: str) -> str:
    """Return a reason when text points at host-side HL memory artifacts.

    The guard is intentionally narrower than a generic ``trials`` ban: task
    workspaces may legitimately contain local files named ``trials`` or
    ``trajectory.jsonl``. We block known HL memory roots and broad host searches
    for HL artifacts, matching the Rust Worker pre-dispatch policy.
    """

    lowered = text.lower()
    mentions_memory_root = any(
        marker in lowered
        for marker in (
            "/trials/runs",
            "trials/runs",
            "/trials/summaries",
            "trials/summaries",
            "/host/trials",
            "hl_memory_path",
            "memory_path",
        )
    )
    if mentions_memory_root:
        return "host HL trial memory is not TerminalBench task workspace evidence"

    mentions_trial_artifact = any(
        marker in lowered
        for marker in (
            "trajectory.jsonl",
            "harness_snapshot.json",
            "handoff.md",
            "feedback.json",
        )
    )
    broad_or_host_search = any(
        marker in lowered
        for marker in (
            "find /",
            "find ..",
            "/host",
            "/mnt/",
            "/root/",
            "../trials",
            "/trials",
        )
    )
    if mentions_trial_artifact and broad_or_host_search:
        return "host-side HL artifact searches are not task workspace evidence"

    return ""


def host_memory_blocked_error(observed: str) -> str:
    """Build a consistent model-facing block message."""

    return (
        "Worker host-memory policy blocked access: "
        f"{host_memory_access_reason(observed)}. Same-task memory summaries "
        "are already injected into the prompt; trials/runs logs and host "
        "memory paths are not part of the TerminalBench task workspace. Use "
        "current task files plus the bounded prior-failure summary instead."
    )


def host_memory_block_metadata() -> dict[str, Any]:
    return policy_guard_metadata(
        HOST_MEMORY_BLOCKED_BY,
        semantic_failure_kind=HOST_MEMORY_SEMANTIC_KIND,
        blocked_reason="host_memory_search",
    )
