"""Minimal current-main base for the reliable Harbor grep integration."""

from __future__ import annotations

from bench import _harbor_adapter_issue4_base as _base
from bench._canonical_harbor_grep_hardlink import HarborGrepTool

HLWorkerHarborAgent = _base.HLWorkerHarborAgent
ToolResult = _base.ToolResult

__all__ = ["HarborGrepTool", "HLWorkerHarborAgent", "ToolResult"]
