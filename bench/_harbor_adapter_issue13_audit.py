"""Strict host-side and UTF-8 contracts for the Issue #13 Harbor grep."""

from __future__ import annotations

from typing import Any

from bench import _harbor_adapter_issue13_impl as _original

# The embedded scanner must not replacement-decode invalid bytes into apparent
# searchable text. Mutate the implementation module before exporting its public
# compatibility surface so direct script tests and the class method see the same
# strict UTF-8 contract.
_original._HARBOR_GREP_COUNTING_PYTHON = (
    _original._HARBOR_GREP_COUNTING_PYTHON.replace(
        'errors="replace"',
        'errors="strict"',
    )
)

# Preserve the implementation module's complete public and compatibility
# surface, including the embedded secure script used by direct regression tests.
for _name, _value in vars(_original).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_OriginalHarborGrepTool = _original.HarborGrepTool


class HarborGrepTool(_OriginalHarborGrepTool):
    """Validate every host-side bound before touching the Harbor environment."""

    def execute(
        self,
        pattern: str,
        path: str = ".",
        **kwargs: Any,
    ) -> _original._base.ToolResult:
        for name in (
            "max_results",
            "max_match_chars",
            "max_input_line_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                return _original._base.ToolResult(
                    success=False,
                    output="",
                    error=f"{name} must be an integer >= 1",
                    metadata={
                        "search_failed": True,
                        "parameter_validation_failed": True,
                        "validated_parameter": name,
                    },
                )
        result = super().execute(pattern, path, **kwargs)
        if result.metadata.get("search_failed"):
            result.metadata.setdefault("strict_text_decoding", True)
            result.metadata.setdefault("text_encoding", "utf-8")
        return result


_original._base.HarborGrepTool = HarborGrepTool
_base = _original._base
HLWorkerHarborAgent = _original.HLWorkerHarborAgent
