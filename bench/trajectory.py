"""TrajectoryReader — parse ATIF trajectory files from Harbor.

Extracts failure patterns, tool call sequences, and timing data
from agent execution trajectories for meta-agent analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrajectoryReader:
    """Parse and analyze agent execution trajectories."""

    @staticmethod
    def load(trajectory_path: Path) -> list[dict[str, Any]]:
        """Load a trajectory file (JSONL or JSON)."""
        if not trajectory_path.exists():
            return []

        if trajectory_path.suffix == ".jsonl":
            events = []
            with open(trajectory_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return events

        if trajectory_path.suffix == ".json":
            try:
                data = json.loads(trajectory_path.read_text())
                return data if isinstance(data, list) else [data]
            except (json.JSONDecodeError, FileNotFoundError):
                return []

        return []

    @staticmethod
    def extract_errors(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract error events from a trajectory."""
        errors = []
        for event in trajectory:
            if event.get("type") == "error":
                errors.append(event)
            elif event.get("success") is False:
                errors.append(event)
            elif "error" in event.get("output", "").lower():
                errors.append(event)
        return errors

    @staticmethod
    def extract_tool_sequence(
        trajectory: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract the sequence of tool calls from a trajectory."""
        tools = []
        for event in trajectory:
            if "tool" in event or "function" in event:
                tools.append(event)
        return tools

    @staticmethod
    def extract_failure_patterns(
        trajectory: list[dict[str, Any]],
    ) -> list[str]:
        """Extract recurring failure patterns from a trajectory.

        These patterns are fed to the meta-agent for root cause analysis.
        """
        patterns: list[str] = []
        errors = TrajectoryReader.extract_errors(trajectory)

        for error in errors:
            msg = str(error.get("error", error.get("output", "")))
            if "command not found" in msg.lower():
                patterns.append("command_not_found")
            elif "permission denied" in msg.lower():
                patterns.append("permission_denied")
            elif "no such file" in msg.lower():
                patterns.append("file_not_found")
            elif "syntax error" in msg.lower():
                patterns.append("syntax_error")
            elif "timeout" in msg.lower():
                patterns.append("timeout")

        return list(set(patterns))

    @staticmethod
    def summarize_timing(
        trajectory: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Summarize timing data from a trajectory."""
        tool_times: dict[str, list[float]] = {}
        for event in trajectory:
            tool_name = event.get("tool", event.get("function", ""))
            duration = event.get("duration_ms", event.get("duration", 0))
            if tool_name and duration:
                tool_times.setdefault(tool_name, []).append(float(duration))

        return {
            tool: sum(times) / len(times)
            for tool, times in tool_times.items()
        }
