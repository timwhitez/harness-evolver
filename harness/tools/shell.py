"""Canonical ShellTool policy plus the shared bounded process runner."""

from __future__ import annotations

import math
from typing import Any

from harness.tools import _shell_issue3_base as _base
from harness.tools import process_runner

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, _value)


def _invalid_timeout_result(value: object) -> _base.ToolResult | None:
    if isinstance(value, bool):
        return _base.ToolResult(
            success=False,
            output="",
            error="timeout_seconds must be finite and > 0",
        )
    try:
        timeout = float(value)
    except (TypeError, ValueError, OverflowError):
        return _base.ToolResult(
            success=False,
            output="",
            error="timeout_seconds must be finite and > 0",
        )
    if not math.isfinite(timeout) or timeout <= 0:
        return _base.ToolResult(
            success=False,
            output="",
            error="timeout_seconds must be finite and > 0",
        )
    return None


def _policy_block(command: str) -> _base.ToolResult | None:
    reason = _base.prohibited_command_reason(command)
    if reason:
        return _base.ToolResult(
            success=False,
            output="",
            error=f"Leaderboard integrity guard blocked command: {reason}.",
            duration_ms=0.0,
            metadata=_base.policy_guard_metadata("leaderboard_integrity_guard"),
        )
    if _base.host_memory_access_reason(command):
        return _base.ToolResult(
            success=False,
            output="",
            error=_base.host_memory_blocked_error(command),
            duration_ms=0.0,
            metadata=_base.host_memory_block_metadata(),
        )

    reason = _base.external_agent_command_reason(command)
    if reason:
        return _base.ToolResult(
            success=False,
            output="",
            error=(
                "Worker shell policy blocked command: "
                f"{reason}. Keep agent creation in the master campaign "
                "orchestrator and solve the task with this Worker loop."
            ),
            duration_ms=0.0,
            metadata=_base.policy_guard_metadata(
                "nested_sub_agent_creation_guard",
                sub_agent_creation_guard=True,
                nested_sub_agent_creation_allowed=False,
                only_master_loop_may_create_sub_agents=True,
                sub_agent_creation_loop_stop_condition=False,
                nested_sub_agent_creation_stop_condition=False,
            ),
        )

    checks = (
        (
            _base.background_package_command_reason,
            "background_package_command_guard",
            "Run one foreground, bounded install/download step with visible output, "
            "or pivot to an existing dependency-free implementation path.",
        ),
        (
            _base.manual_dependency_download_reason,
            "manual_dependency_download_guard",
            "Use an existing installed or cached artifact, one foreground package-"
            "manager command capped by policy, or a dependency-free implementation "
            "plus a short visible check.",
        ),
        (
            _base.scripted_package_manager_command_reason,
            "scripted_package_manager_guard",
            "Use a visible foreground package-manager step capped by policy only "
            "when necessary, or pivot to an installed/cached artifact or a "
            "dependency-free implementation.",
        ),
        (
            _base.heavy_scientific_dependency_install_reason,
            "heavy_scientific_dependency_guard",
            "Use an installed/cached artifact, inspect for a smaller explicit "
            "requirement, or implement a dependency-free/sampled path plus a short check.",
        ),
        (
            _base.heavy_graphics_runtime_install_reason,
            "heavy_graphics_runtime_dependency_guard",
            "Re-read the visible image/output contract, use installed lightweight "
            "libraries or stdlib, and produce the smallest dependency-light artifact.",
        ),
        (
            _base.large_toolchain_install_command_reason,
            "large_toolchain_install_guard",
            "Use an existing compiler/toolchain, a smaller build target, or a "
            "dependency-free implementation with a short visible check.",
        ),
        (
            _base.manual_deb_dependency_chase_reason,
            "manual_deb_dependency_chase_guard",
            "Use an installed/cached artifact, a smaller explicit requirement, or a "
            "dependency-free implementation plus a short visible check.",
        ),
        (
            _base.broad_root_find_command_reason,
            "broad_root_find_guard",
            "Narrow the search to /app, /tmp, or another task-relevant prefix, or "
            "add -maxdepth before expanding it.",
        ),
        (
            _base.broad_proc_scan_command_reason,
            "broad_proc_scan_guard",
            "Use ps, pgrep, pidof, /proc/net, or a specific known PID instead of "
            "scanning every process.",
        ),
    )
    for detector, guard, guidance in checks:
        reason = detector(command)
        if reason:
            return _base.ToolResult(
                success=False,
                output="",
                error=f"Worker shell policy blocked command: {reason}. {guidance}",
                duration_ms=0.0,
                metadata=_base.policy_guard_metadata(guard),
            )
    return None


def _bounded_output(
    stdout: str,
    stderr: str,
    *,
    prefix_note: str,
    limit: int,
) -> str:
    output = stdout
    if stderr:
        output += f"\n[stderr]\n{stderr}"
    if prefix_note:
        output = f"{prefix_note}\n{output}" if output else prefix_note
    if len(output) <= limit:
        return output
    marker = f"\n... (truncated, at least {len(output)} captured chars)"
    if len(marker) >= limit:
        return marker[:limit]
    return output[: limit - len(marker)] + marker


class ShellTool(_base.ShellTool):
    """Authorize once and execute exact Bash argv through bounded supervision."""

    def execute(
        self,
        command: str,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> _base.ToolResult:
        blocked = _policy_block(command)
        if blocked is not None:
            return blocked

        requested_timeout: object = self.timeout_seconds if timeout is None else timeout
        invalid = _invalid_timeout_result(requested_timeout)
        if invalid is not None:
            return invalid
        effective_timeout, timeout_note = _base.package_manager_timeout_cap(
            command,
            float(requested_timeout),
        )

        try:
            outcome = process_runner.run_bounded_argv(
                ["bash", "-o", "pipefail", "-c", command],
                timeout_seconds=effective_timeout,
                cwd=kwargs.get("cwd"),
                env=kwargs.get("env"),
                cancel_event=kwargs.get("cancel_event"),
                output_limit_bytes=max(1, int(self.max_output_chars)),
            )
        except (OSError, TypeError, ValueError) as exc:
            return _base.ToolResult(success=False, output="", error=str(exc))

        output = _bounded_output(
            outcome.stdout,
            outcome.stderr,
            prefix_note=timeout_note,
            limit=max(1, int(self.max_output_chars)),
        )
        if outcome.timed_out:
            cleanup_state = (
                "The managed command tree was terminated and reaped."
                if outcome.managed_process_group_terminated
                else "Managed command-tree cleanup could not be confirmed."
            )
            metadata = _base.operation_timeout_metadata(
                timeout_seconds=effective_timeout,
                requested_timeout_seconds=timeout,
                timeout_capped=bool(timeout_note),
                elapsed_ms=outcome.elapsed_ms,
                stdout=outcome.stdout,
                stderr=outcome.stderr,
                telemetry_source="shell",
                managed_process_group_terminated=(
                    outcome.managed_process_group_terminated
                ),
                process_tree_terminated=outcome.managed_process_group_terminated,
                output_bounded=True,
            )
            return _base.ToolResult(
                success=False,
                output=output,
                error=(
                    f"Command timed out after {effective_timeout:g}s. {cleanup_state} "
                    "This timeout is an operation result, not a master, sub-agent, "
                    "or Worker loop stop condition."
                ),
                duration_ms=outcome.elapsed_ms,
                metadata=metadata,
            )
        if outcome.cancelled:
            return _base.ToolResult(
                success=False,
                output=output,
                error="Command was cancelled; managed command-tree cleanup was requested.",
                duration_ms=outcome.elapsed_ms,
                metadata={
                    "cancelled": True,
                    "process_tree_terminated": outcome.managed_process_group_terminated,
                    "managed_process_group_terminated": (
                        outcome.managed_process_group_terminated
                    ),
                    "output_bounded": True,
                    "loop_stop_condition": False,
                },
            )

        result = self._result_from_completed(
            command=command,
            returncode=outcome.returncode,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            duration_ms=outcome.elapsed_ms,
            prefix_note=timeout_note,
            timeout_seconds=effective_timeout,
            requested_timeout_seconds=(
                None if timeout is None else float(requested_timeout)
            ),
            timeout_capped=bool(timeout_note),
        )
        result.metadata = {**result.metadata, "output_bounded": True}
        result.output = _bounded_output(
            outcome.stdout,
            outcome.stderr,
            prefix_note=timeout_note,
            limit=max(1, int(self.max_output_chars)),
        )
        return result

    def _terminate_process_tree(self, process: Any) -> None:
        process_runner._terminate_process_tree(getattr(process, "_process", process))
