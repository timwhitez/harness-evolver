"""Meta-coding exports for the external HL update layer.

The package is imported whenever a concrete ``meta.*`` module is loaded. Keep
the public convenience exports lazy so deterministic campaign bookkeeping and
dry-run commands do not import the legacy LiteLLM-backed ``meta.agent`` module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "MetaAgent",
    "MetaAgentPrompts",
    "HarnessEditor",
    "ImprovementSuggester",
    "FailureAnalyzer",
    "CodexRunResult",
    "CodexUpdateEngine",
    "CodexWorkPacket",
    "MissionDebugPacket",
    "MissionFeatureCandidate",
    "MissionPlanner",
    "MissionValidationContract",
    "WorkPacketBuilder",
    "PatchReviewer",
    "PatchReviewResult",
]


_LAZY_EXPORTS = {
    "MetaAgent": ("meta.agent", "MetaAgent"),
    "MetaAgentPrompts": ("meta.prompts", "MetaAgentPrompts"),
    "HarnessEditor": ("meta.editor", "HarnessEditor"),
    "ImprovementSuggester": ("meta.suggestion", "ImprovementSuggester"),
    "FailureAnalyzer": ("meta.analysis", "FailureAnalyzer"),
    "CodexRunResult": ("meta.codex_update", "CodexRunResult"),
    "CodexUpdateEngine": ("meta.codex_update", "CodexUpdateEngine"),
    "CodexWorkPacket": ("meta.packager", "CodexWorkPacket"),
    "MissionDebugPacket": ("meta.missions", "MissionDebugPacket"),
    "MissionFeatureCandidate": ("meta.missions", "MissionFeatureCandidate"),
    "MissionPlanner": ("meta.missions", "MissionPlanner"),
    "MissionValidationContract": ("meta.missions", "MissionValidationContract"),
    "WorkPacketBuilder": ("meta.packager", "WorkPacketBuilder"),
    "PatchReviewer": ("meta.reviewer", "PatchReviewer"),
    "PatchReviewResult": ("meta.reviewer", "PatchReviewResult"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
