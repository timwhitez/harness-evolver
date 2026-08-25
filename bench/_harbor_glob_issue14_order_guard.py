"""Fail closed when Bash glob output has an unprovable result order.

Python's :func:`glob.glob` preserves filesystem enumeration order, while Bash
``compgen -G`` applies shell ordering.  The two engines can therefore return the
same paths in a different sequence.  Because the primary implementation slices
that sequence before returning it, the Python-free fallback can only make an
exact claim when it observes zero or one match.
"""

from __future__ import annotations

from typing import Any

from bench import _harbor_glob_issue14 as _base
from harness.tools.base import ToolResult


class HarborGlobTool(_base.HarborGlobTool):
    """Retain only fallback results whose sequence is engine-independent."""

    def _glob_without_python(
        self,
        *,
        pattern: str,
        path: str = ".",
        **_: Any,
    ) -> ToolResult:
        result = super()._glob_without_python(pattern=pattern, path=path)
        if not result.success:
            if result.metadata.get("glob_result_limit_unsupported"):
                result.metadata["glob_result_order_unsupported"] = True
                result.error = (
                    "Python-free glob produced too many matches to prove the "
                    "same ordered [:500] result as Python glob."
                )
            return result

        matches = (
            []
            if result.output in {"", "(no matches)"}
            else [line for line in result.output.splitlines() if line]
        )
        if len(matches) > 1:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Python-free glob produced multiple matches, but Bash and "
                    "Python glob do not guarantee the same result order; "
                    "refusing an order-dependent result."
                ),
                metadata={
                    **result.metadata,
                    "glob_result_order_unsupported": True,
                    "observed_match_count": len(matches),
                },
            )
        return result


HLWorkerHarborAgent = _base.HLWorkerHarborAgent
