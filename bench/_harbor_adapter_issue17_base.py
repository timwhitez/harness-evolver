"""Minimal current-main base for the Harbor streaming reader."""

from __future__ import annotations

from bench import _harbor_adapter_issue4_base as _base
from bench import _canonical_harbor_identity_guard as _identity

HarborFileReadTool = _identity.HarborFileReadTool
HLWorkerHarborAgent = _base.HLWorkerHarborAgent
ToolResult = _base.ToolResult

__all__ = ["HarborFileReadTool", "HLWorkerHarborAgent", "ToolResult"]
