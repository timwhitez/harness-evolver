"""HLAgent Python boundary for the Rust TerminalBench Worker core.

The task-solving loop is implemented in ``crates/hl-worker-core``. This module
keeps the Harbor/LiteLLM/Python-tool compatibility surface: it launches the
Rust JSONL worker, executes requested Python tools, adapts model responses, and
converts the final Rust payload into the existing ``TrialResult`` contract.
Prompt assembly and other dynamically updated Worker policy decisions live in
the Rust core.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import litellm

from harness.config import HarnessConfig, RoleModelConfig
from harness.tools.registry import ToolRegistry
from harness.tools.shell import ShellTool
from harness.tools.file_read import FileReadTool
from harness.tools.file_edit import FileEditTool
from harness.tools.file_write import FileWriteTool
from harness.tools.search import GrepTool, GlobTool
from harness.tools.goal import GoalReadTool
from harness.tools.todo import TodoReadTool, TodoStore, TodoWriteTool
from harness.tools.verify import VerifyTool
from harness.tools.done import DoneTool
from harness.tools.base import ToolResult

from hl.types import TrialResult, TrialStatus


@dataclass
class HLAgent:
    """Harbor-facing Worker adapter backed by the Rust harness agent core."""

    config: HarnessConfig = field(default_factory=HarnessConfig.create_default)
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    role_config: RoleModelConfig | None = None

    # Runtime state mirrored from Rust for existing Harbor/trial contracts.
    # ``0`` means no turn-count reference; the Rust core never stops solely
    # because a turn counter reached this value.
    max_turns_audit: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    tool_call_history: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    todo_store: TodoStore = field(default_factory=TodoStore)
    trajectory_event_sink: Callable[[dict[str, Any]], None] | None = None
    completion_blockers: list[str] = field(default_factory=list)
    _active_process: subprocess.Popen[str] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _process_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    # Policy knobs forwarded to the Rust core. They remain configurable from
    # Python/Harbor while Rust owns the Worker loop decisions.
    compaction_char_threshold: int = 120000
    verification_checkpoint_turns: tuple[int, ...] = (8, 16, 24, 32)
    verification_checkpoint_cooldown_turns: int = 4
    verification_checkpoint_max: int = 4
    verification_checkpoint_min_tool_calls: int = 4
    no_progress_checkpoint_tool_calls: int = 12
    no_progress_checkpoint_cooldown_turns: int = 8
    no_progress_checkpoint_max: int = 2
    # Compatibility/audit-only field kept for older configs. The Rust Worker
    # no-progress pivot is evidence-driven and does not use max_turns percent
    # thresholds as a stop or near-stop signal.
    late_no_progress_turn_percent: int = 70
    stalled_tool_loop_calls: int = 20
    stalled_tool_loop_cooldown_turns: int = 8
    stalled_tool_loop_max: int = 2
    deliverable_checkpoint_min_tool_calls: int = 8
    deliverable_checkpoint_cooldown_turns: int = 8
    deliverable_checkpoint_max: int = 2
    empty_response_recovery_limit: int = 3
    ephemeral_tool_output_max_chars: int = 4000
    ephemeral_tool_output_keep_recent: int = 4

    def __post_init__(self):
        if len(self.tool_registry) == 0:
            self._register_default_tools()

    @property
    def max_turns(self) -> int:
        """Backward-compatible alias for the audit-only turn reference."""
        return self.max_turns_audit

    @max_turns.setter
    def max_turns(self, value: int) -> None:
        self.max_turns_audit = value

    def _register_default_tools(self) -> None:
        """Register the core Python tool adapters exposed to the Rust worker."""
        for tool in [
            ShellTool(),
            FileReadTool(),
            FileEditTool(),
            FileWriteTool(),
            GrepTool(),
            GlobTool(),
            TodoReadTool(store=self.todo_store),
            TodoWriteTool(store=self.todo_store),
            GoalReadTool(),
            VerifyTool(),
            DoneTool(),
        ]:
            self.tool_registry.register(tool)

    def _configure_goal_tool(self, task_context: dict[str, Any]) -> None:
        tool = self.tool_registry.get("goal_read")
        if tool is None or not hasattr(tool, "goal_path"):
            return
        tool.goal_path = self._goal_path_from_context(task_context)

    def _goal_path_from_context(self, task_context: dict[str, Any]) -> Path | None:
        raw = task_context.get("goal_path") or task_context.get("hl_goal_path")
        if isinstance(raw, str) and raw.strip():
            return Path(raw)
        if isinstance(raw, Path):
            return raw
        if raw is not None:
            try:
                return Path(raw)
            except TypeError:
                return None
        return None

    def run(self, task_instruction: str, task_context: dict[str, Any]) -> TrialResult:
        """Execute a TerminalBench task through the Rust Worker core."""

        self._initialize_run_state(task_instruction, task_context)
        try:
            return self._run_rust_core(task_instruction, task_context)
        except Exception as exc:
            self._append_trajectory(
                {
                    "type": "rust_worker_core_error",
                    "error": str(exc),
                }
            )
            return TrialResult(
                trial_id=task_context.get("task_id", "unknown"),
                task_id=task_context.get("task_id", "unknown"),
                task_domain=self._task_domain(task_context.get("domain")),
                task_difficulty=self._task_difficulty(task_context.get("difficulty")),
                status=TrialStatus.ERROR,
                score=0.0,
                verified=False,
                verifier_output=(
                    "Rust Worker core failed before reliable Harbor verification."
                ),
                tool_calls=self.tool_call_history,
                trajectory=self.trajectory,
                model_used=self._model_name(),
                token_usage=self.token_usage,
                error_log=[f"Rust Worker core failed: {exc}"],
                metadata={"rust_worker_core_error": True},
            )

    def cancel_current_run(self, reason: str = "cancelled") -> None:
        """Terminate the active Rust Worker process after Harbor cancels a trial."""

        with self._process_lock:
            process = self._active_process
        if process is None:
            return
        self._append_trajectory(
            {
                "type": "rust_worker_core_cancelled",
                "reason": reason,
                "loop_stop_condition": False,
                "timeout_seconds_stop_condition": False,
            }
        )
        self._terminate_process(process)

    def _initialize_run_state(
        self,
        task_instruction: str,
        task_context: dict[str, Any],
    ) -> None:
        self.turn_count = 0
        self.messages = []
        self.tool_call_history = []
        self.trajectory = []
        self.token_usage = {}
        self.completion_blockers = []
        self._configure_goal_tool(task_context)

    def _run_rust_core(
        self,
        task_instruction: str,
        task_context: dict[str, Any],
    ) -> TrialResult:
        process = subprocess.Popen(
            self._rust_worker_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        with self._process_lock:
            self._active_process = process

        try:
            self._write_bridge_event(
                process,
                {
                    "type": "start",
                    "request": self._rust_worker_request(task_instruction, task_context),
                },
            )

            final_payload: dict[str, Any] | None = None
            for line in process.stdout:
                if not line.strip():
                    continue
                event = json.loads(line)
                event_type = event.get("type")
                if event_type == "llm_request":
                    self.messages = list(event.get("messages") or [])
                    try:
                        response = litellm.completion(
                            **self._completion_kwargs_for_bridge(
                                self.messages,
                                list(event.get("tool_schemas") or []),
                            )
                        )
                    except Exception as exc:
                        self._write_bridge_event(
                            process,
                            {
                                "type": "llm_response",
                                "payload": self._llm_error_response_payload(exc),
                            },
                        )
                        continue
                    try:
                        payload = self._llm_response_payload(response)
                    except Exception as exc:
                        self._write_bridge_event(
                            process,
                            {
                                "type": "llm_response",
                                "payload": self._llm_error_response_payload(exc),
                            },
                        )
                        continue
                    self._write_bridge_event(
                        process,
                        {
                            "type": "llm_response",
                            "payload": payload,
                        },
                    )
                elif event_type == "tool_request":
                    self._write_bridge_event(
                        process,
                        {
                            "type": "tool_response",
                            "payload": self._execute_bridge_tool(event),
                        },
                    )
                elif event_type == "trajectory_event":
                    trajectory_event = event.get("event")
                    if isinstance(trajectory_event, dict):
                        self._append_trajectory(trajectory_event)
                elif event_type == "final":
                    final_payload = dict(event.get("result") or {})
                    break
                elif event_type == "fatal":
                    raise RuntimeError(str(event.get("error") or "Rust Worker core fatal error"))
                else:
                    raise RuntimeError(f"Unknown Rust Worker core event: {event_type!r}")

            stderr = process.stderr.read() if process.stderr is not None else ""
            return_code = process.wait()
            if final_payload is None:
                raise RuntimeError(
                    "Rust Worker core exited without a final result"
                    + (f": {stderr.strip()}" if stderr.strip() else "")
                )
            if return_code != 0:
                raise RuntimeError(
                    f"Rust Worker core exited with code {return_code}"
                    + (f": {stderr.strip()}" if stderr.strip() else "")
                )
            return self._trial_result_from_rust(final_payload, task_context)
        finally:
            if process.poll() is None:
                self._terminate_process(process)
            with self._process_lock:
                if self._active_process is process:
                    self._active_process = None

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        self._send_process_signal(process, signal.SIGTERM)
        if process.poll() is None:
            killed = self._send_process_signal(process, signal.SIGKILL)
            if killed:
                process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        if process.poll() is not None:
            process.wait()

    def _send_process_signal(
        self,
        process: subprocess.Popen[str],
        sig: signal.Signals,
    ) -> bool:
        try:
            os.killpg(process.pid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except Exception:
            try:
                process.send_signal(sig)
                return True
            except Exception:
                return False

    def _rust_worker_command(self) -> list[str]:
        explicit = os.environ.get("HL_WORKER_RUST_BIN")
        if explicit:
            return [explicit]
        repo_root = Path(__file__).resolve().parents[1]
        crate_root = repo_root / "crates" / "hl-worker-core"
        # Prefer an already-built binary. Launching via ``cargo run`` forces a
        # crates.io index refresh on every trial, which fails on offline or
        # read-only registry mounts; the prebuilt binary avoids that and is
        # faster. Skip stale binaries so local source edits and tests do not
        # silently run an older Worker core.
        for profile in ("release", "debug"):
            candidate = crate_root / "target" / profile / "hl-worker-core"
            if (
                candidate.is_file()
                and os.access(candidate, os.X_OK)
                and self._rust_worker_binary_is_current(candidate, crate_root)
            ):
                return [str(candidate)]
        manifest = crate_root / "Cargo.toml"
        return [
            "cargo",
            "+stable",
            "run",
            "--quiet",
            "--manifest-path",
            str(manifest),
            "--",
        ]

    def _rust_worker_binary_is_current(self, candidate: Path, crate_root: Path) -> bool:
        try:
            binary_mtime = candidate.stat().st_mtime
            source_mtime = max(
                path.stat().st_mtime
                for path in (crate_root / "src").rglob("*.rs")
                if path.is_file()
            )
        except (OSError, ValueError):
            return True
        return binary_mtime >= source_mtime

    def _rust_worker_request(
        self,
        task_instruction: str,
        task_context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "task_instruction": task_instruction,
            "task_context": self._jsonable(task_context),
            "initial_messages": self.messages,
            "initial_trajectory": self.trajectory,
            "tool_schemas": self.tool_registry.get_schemas(),
            "max_turns_audit": self.max_turns_audit,
            "model_used": self._model_name(),
            "todo_items": self._todo_items_payload(),
            "verification_command": "",
            "prompt_policy": self._prompt_policy_payload(task_context),
            "thresholds": {
                "compaction_char_threshold": self.compaction_char_threshold,
                "verification_checkpoint_turns": list(self.verification_checkpoint_turns),
                "verification_checkpoint_cooldown_turns": self.verification_checkpoint_cooldown_turns,
                "verification_checkpoint_max": self.verification_checkpoint_max,
                "verification_checkpoint_min_tool_calls": self.verification_checkpoint_min_tool_calls,
                "no_progress_checkpoint_tool_calls": self.no_progress_checkpoint_tool_calls,
                "no_progress_checkpoint_cooldown_turns": self.no_progress_checkpoint_cooldown_turns,
                "no_progress_checkpoint_max": self.no_progress_checkpoint_max,
                "late_no_progress_turn_percent": self.late_no_progress_turn_percent,
                "stalled_tool_loop_calls": self.stalled_tool_loop_calls,
                "stalled_tool_loop_cooldown_turns": self.stalled_tool_loop_cooldown_turns,
                "stalled_tool_loop_max": self.stalled_tool_loop_max,
                "deliverable_checkpoint_min_tool_calls": self.deliverable_checkpoint_min_tool_calls,
                "deliverable_checkpoint_cooldown_turns": self.deliverable_checkpoint_cooldown_turns,
                "deliverable_checkpoint_max": self.deliverable_checkpoint_max,
                "empty_response_recovery_limit": self.empty_response_recovery_limit,
                "ephemeral_tool_output_max_chars": self.ephemeral_tool_output_max_chars,
                "ephemeral_tool_output_keep_recent": self.ephemeral_tool_output_keep_recent,
            },
        }

    def _completion_kwargs_for_bridge(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.messages = messages
        if self.role_config is not None:
            kwargs = self.role_config.litellm_kwargs()
        else:
            kwargs = {"model": self.config.model}
        kwargs.update(
            {
                "messages": self._provider_messages(),
                "tools": tool_schemas,
                "tool_choice": "auto",
            }
        )
        return kwargs

    def _llm_response_payload(self, response: Any) -> dict[str, Any]:
        choice = response.choices[0]
        message = choice.message
        tool_calls = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            )
        return {
            "message": {
                "content": getattr(message, "content", None) or "",
                "reasoning_content": self._message_reasoning_content(message),
                "tool_calls": tool_calls,
            },
            "usage": self._usage_payload(response),
        }

    def _llm_error_response_payload(self, exc: Exception) -> dict[str, Any]:
        return {
            "message": {
                "content": "",
                "reasoning_content": None,
                "tool_calls": [],
            },
            "usage": {},
            "error": {
                "type": exc.__class__.__name__,
                "message": _truncate(str(exc) or repr(exc), 1000),
            },
        }

    def _usage_payload(self, response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return {}
        payload: dict[str, int] = {}
        for key in (
            "prompt_tokens",
            "input_tokens",
            "completion_tokens",
            "output_tokens",
            "cache_read_input_tokens",
        ):
            value = self._usage_value(usage, key)
            if value is not None:
                payload[key] = value
        return payload

    def _execute_bridge_tool(self, event: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(event.get("tool") or "")
        args = event.get("args")
        if not isinstance(args, dict):
            args = {}
        blocked = self._blocked_verifier_artifact_search(tool_name, args)
        if blocked is not None:
            return {
                "success": blocked.success,
                "output": blocked.output,
                "error": blocked.error,
                "duration_ms": blocked.duration_ms,
                "metadata": blocked.metadata,
            }
        result = self.tool_registry.execute(tool_name, **args)
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "duration_ms": result.duration_ms,
            "metadata": result.metadata,
        }

    def _blocked_verifier_artifact_search(
        self, tool_name: str, args: dict[str, Any]
    ) -> ToolResult | None:
        observed = _tool_argument_search_text(tool_name, args)
        if observed is None or not _looks_like_verifier_artifact_search(observed):
            return None
        return ToolResult(
            success=False,
            output="",
            error=(
                "Blocked hidden verifier artifact search: "
                f"{_truncate(observed.strip(), 500)}. "
                "Harbor/verifier runtime logs and dependency caches are not "
                "TerminalBench task workspace evidence. Do not inspect "
                "/logs/verifier, /tmp/hl-verifier-cache, hidden verifier "
                "artifacts, or Harbor internals; use the current workspace, "
                "visible task files/tests, task-provided checks, and bounded "
                "same-task memory summaries instead."
            ),
            metadata={
                "semantic_failure_kind": "blocked_verifier_artifact_search",
                "blocked_reason": "verifier_artifact_search",
            },
        )

    def _write_bridge_event(
        self,
        process: subprocess.Popen[str],
        payload: dict[str, Any],
    ) -> None:
        if process.stdin is None:
            raise RuntimeError("Rust Worker core stdin is closed")
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def _trial_result_from_rust(
        self,
        payload: dict[str, Any],
        task_context: dict[str, Any],
    ) -> TrialResult:
        self.messages = list(payload.get("messages") or self.messages)
        self.turn_count = int(payload.get("turn_count") or self.turn_count)
        self.tool_call_history = list(payload.get("tool_calls") or [])
        self.trajectory = list(payload.get("trajectory") or self.trajectory)
        self.token_usage = {
            str(key): int(value)
            for key, value in dict(payload.get("token_usage") or {}).items()
            if isinstance(value, int)
        }
        self.completion_blockers = [
            str(item)
            for item in list(payload.get("error_log") or [])
            if isinstance(item, str)
        ]
        status = self._trial_status(payload.get("status"))
        return TrialResult(
            trial_id=payload.get("trial_id") or task_context.get("task_id", "unknown"),
            task_id=payload.get("task_id") or task_context.get("task_id", "unknown"),
            task_domain=self._task_domain(task_context.get("domain")),
            task_difficulty=self._task_difficulty(task_context.get("difficulty")),
            status=status,
            score=float(payload.get("score") or 0.0),
            verified=bool(payload.get("verified", False)),
            verifier_output=str(payload.get("verifier_output") or ""),
            tool_calls=self.tool_call_history,
            trajectory=self.trajectory,
            model_used=str(payload.get("model_used") or self._model_name()),
            token_usage=self.token_usage,
            error_log=[
                str(item)
                for item in list(payload.get("error_log") or [])
                if isinstance(item, str)
            ],
            metadata=dict(payload.get("metadata") or {}),
        )

    def _trial_status(self, raw: Any) -> TrialStatus:
        try:
            return TrialStatus(str(raw))
        except ValueError:
            return TrialStatus.ERROR

    def _todo_items_payload(self) -> list[dict[str, str]]:
        for tool_name in ("todo_read", "todo_write"):
            tool = self.tool_registry.get(tool_name)
            store = getattr(tool, "store", None)
            if store is None:
                continue
            return [
                {
                    "id": item.id,
                    "content": item.content,
                    "status": item.status,
                }
                for item in store.items
            ]
        return []

    def _prompt_policy_payload(self, task_context: dict[str, Any]) -> dict[str, str]:
        return {
            "tool_descriptions": self.tool_registry.render_descriptions(),
        }

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        return value

    def _provider_messages(self) -> list[dict[str, Any]]:
        """Return a provider-valid copy of the conversation history."""
        repaired: list[dict[str, Any]] = []
        index = 0
        while index < len(self.messages):
            message = dict(self.messages[index])
            if message.get("role") == "assistant" and message.get("tool_calls"):
                expected = [
                    str(tool_call.get("id"))
                    for tool_call in message.get("tool_calls") or []
                    if tool_call.get("id")
                ]
                tool_results: list[dict[str, Any]] = []
                seen: set[str] = set()
                next_index = index + 1
                while next_index < len(self.messages):
                    candidate = dict(self.messages[next_index])
                    if candidate.get("role") != "tool":
                        break
                    tool_call_id = str(candidate.get("tool_call_id") or "")
                    if tool_call_id in expected and tool_call_id not in seen:
                        tool_results.append(candidate)
                        seen.add(tool_call_id)
                    next_index += 1
                    if len(seen) == len(expected):
                        break
                if expected and len(seen) == len(expected):
                    repaired.append(message)
                    repaired.extend(tool_results)
                    index = next_index
                    continue
                repaired.append(self._assistant_tool_calls_as_text(message, expected))
                index += 1
                continue
            if message.get("role") == "tool":
                repaired.append(self._orphan_tool_result_as_user_message(message))
                index += 1
                continue
            repaired.append(message)
            index += 1
        return repaired

    def _assistant_tool_calls_as_text(
        self,
        message: dict[str, Any],
        expected_tool_call_ids: list[str],
    ) -> dict[str, Any]:
        content = str(message.get("content") or "").strip()
        ids = ", ".join(expected_tool_call_ids) or "unknown"
        suffix = (
            "Provider protocol repair: omitted non-contiguous assistant "
            f"tool_calls from replay history. tool_call_ids={ids}."
        )
        return {
            "role": "assistant",
            "content": f"{content}\n\n{suffix}".strip(),
        }

    def _orphan_tool_result_as_user_message(
        self,
        message: dict[str, Any],
    ) -> dict[str, Any]:
        tool_call_id = str(message.get("tool_call_id") or "unknown")
        content = str(message.get("content") or "")
        return {
            "role": "user",
            "content": (
                "Provider protocol repair: preserved orphan tool result "
                f"as observation for tool_call_id={tool_call_id}.\n"
                f"{content[:4000]}"
            ),
        }

    def _model_name(self) -> str:
        return self.role_config.model if self.role_config is not None else self.config.model

    def _assistant_message_payload(
        self,
        message: Any,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
        }
        if tool_calls is not None:
            payload["tool_calls"] = tool_calls
        reasoning_content = self._message_reasoning_content(message)
        if reasoning_content:
            payload["reasoning_content"] = reasoning_content
        return payload

    def _message_reasoning_content(self, message: Any) -> str | None:
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            return str(reasoning_content)
        provider_fields = getattr(message, "provider_specific_fields", None)
        if isinstance(provider_fields, dict) and provider_fields.get("reasoning_content"):
            return str(provider_fields["reasoning_content"])
        if hasattr(message, "model_dump"):
            data = message.model_dump()
            if data.get("reasoning_content"):
                return str(data["reasoning_content"])
            provider_fields = data.get("provider_specific_fields")
            if isinstance(provider_fields, dict) and provider_fields.get("reasoning_content"):
                return str(provider_fields["reasoning_content"])
        return None

    def _append_trajectory(self, event: dict[str, Any]) -> None:
        """Record one trajectory event and opportunistically persist it live."""
        self.trajectory.append(event)
        if self.trajectory_event_sink is None:
            return
        try:
            self.trajectory_event_sink(event)
        except Exception:
            return

    def _usage_value(self, usage: Any, key: str) -> int | None:
        if isinstance(usage, dict):
            value = usage.get(key)
        else:
            value = getattr(usage, key, None)
        return int(value) if isinstance(value, int) else None

    def _task_domain(self, raw: Any):
        from hl.types import TaskDomain

        try:
            return TaskDomain(str(raw))
        except ValueError:
            if str(raw).lower() == "security":
                return TaskDomain.SECURITY
            return TaskDomain.SOFTWARE_ENGINEERING

    def _task_difficulty(self, raw: Any):
        from hl.types import TaskDifficulty

        try:
            return TaskDifficulty(str(raw))
        except ValueError:
            return TaskDifficulty.MEDIUM


def _tool_argument_search_text(tool_name: str, args: dict[str, Any]) -> str | None:
    if tool_name == "bash":
        command = args.get("command")
        return str(command) if isinstance(command, str) else None
    if tool_name in {"read", "grep", "glob"}:
        parts = [
            str(value)
            for key in ("file_path", "path", "pattern", "query", "glob")
            if isinstance((value := args.get(key)), str)
        ]
        return " ".join(parts) if parts else None
    return None


def _looks_like_verifier_artifact_search(text: str) -> bool:
    lowered = text.lower()
    return any(
        re.search(rf"(?<![\w./-]){re.escape(marker)}(?:/|(?![\w./-]))", lowered)
        for marker in ("/logs/verifier", "/tmp/hl-verifier-cache")
    )


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."
