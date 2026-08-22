"""Trajectory selection for Worker continuation and Codex update packets.

Legacy ``max_*`` fields are retained as audit references only. They must not
truncate context for master, diagnostic/context sub-agent, Codex update
sub-agent, validation/regression, mission-debug, or Worker loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrajectoryPack:
    max_events: int = 80
    max_output_chars: int = 4000

    def select(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _ = self.max_events, self.max_output_chars
        return [dict(event) for event in events]

    def audit_metadata(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "trajectory_event_count": len(events),
            "max_events_audit_only": self.max_events,
            "max_output_chars_audit_only": self.max_output_chars,
            "max_events_stop_condition": False,
            "max_output_chars_stop_condition": False,
            "trajectory_event_count_stop_condition": False,
            "trajectory_output_chars_stop_condition": False,
            "trajectory_event_count_truncation_stop_condition": False,
            "trajectory_output_char_truncation_stop_condition": False,
            "context_sub_agent_stop_condition": False,
            "codex_update_sub_agent_stop_condition": False,
            "worker_loop_stop_condition": False,
            "loop_stop_condition": False,
            "time_round_token_limit_driven": False,
        }
