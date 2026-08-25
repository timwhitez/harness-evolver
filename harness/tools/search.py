"""Stable no-follow grep with exact result/error accounting."""

from __future__ import annotations

from contextlib import closing
import fnmatch
import importlib
import io
import os
from pathlib import Path
import re
from typing import Any, Iterator, TextIO

from harness.tools.canonical_path_guard import guard_canonical_path_strings
from harness.tools.stable_tree import StableTreeError, iter_stable_regular_files

_base = importlib.import_module("harness.tools._search_issue13_base")

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

_DEFAULT_MAX_INPUT_LINE_CHARS = 1_000_000
_MAX_FAILURE_SAMPLES = 5
_MAX_FAILURE_SAMPLE_CHARS = 2_000
_ORIGINAL_GET_SCHEMA_ATTRIBUTE = "_harness_evolver_uncapped_get_schema"
_BASE_GET_SCHEMA = getattr(
    _base.GrepTool,
    _ORIGINAL_GET_SCHEMA_ATTRIBUTE,
    _base.GrepTool.get_schema,
)
setattr(
    _base.GrepTool,
    _ORIGINAL_GET_SCHEMA_ATTRIBUTE,
    _BASE_GET_SCHEMA,
)


class _InputLineLimitExceeded(Exception):
    def __init__(self, line_number: int, limit: int) -> None:
        self.line_number = line_number
        self.limit = limit
        super().__init__(
            f"physical line {line_number} exceeds grep input limit of {limit} characters"
        )


def _bounded_physical_lines(
    stream: TextIO,
    *,
    max_chars: int,
) -> Iterator[tuple[int, str]]:
    """Yield complete lines without allocating an unbounded physical record."""

    line_number = 0
    while True:
        line = stream.readline(max_chars + 1)
        if line == "":
            return
        line_number += 1
        # Exactly ``max_chars`` content characters followed by a newline are
        # valid. A read that fills max_chars + 1 without seeing the newline
        # proves that the physical record exceeds the configured semantic cap.
        if len(line) > max_chars and not line.endswith("\n"):
            raise _InputLineLimitExceeded(line_number, max_chars)
        yield line_number, line


def _parameter_failure(name: str) -> ToolResult:
    return ToolResult(
        success=False,
        output="",
        error=f"{name} must be an integer >= 1",
        metadata={
            "engine": "python-stable-nofollow",
            "search_failed": True,
            "parameter_validation_failed": True,
            "canonical_paths": True,
            "nofollow_io": True,
            "stable_root_descriptor": True,
        },
    )


def _configured_result_cap(self: Any) -> int | None:
    value = getattr(self, "max_results", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _result_cap_failure(configured_cap: int) -> ToolResult:
    return ToolResult(
        success=False,
        output="",
        error=(
            "max_results cannot exceed the configured GrepTool cap "
            f"of {configured_cap}"
        ),
        metadata={
            "engine": "python-stable-nofollow",
            "search_failed": True,
            "parameter_validation_failed": True,
            "configured_max_results": configured_cap,
            "canonical_paths": True,
            "nofollow_io": True,
            "stable_root_descriptor": True,
        },
    )


def _bounded_grep_schema(self: Any):
    schema = _BASE_GET_SCHEMA(self)
    configured_cap = _configured_result_cap(self)
    properties = schema.parameters.get("properties", {})
    result_property = properties.get("max_results")
    if isinstance(result_property, dict):
        result_property["minimum"] = 1
        if configured_cap is not None:
            result_property["maximum"] = configured_cap
    return schema


def _validate_result_window(
    self: Any,
    requested: object,
) -> tuple[int | None, ToolResult | None]:
    configured_cap = _configured_result_cap(self)
    if configured_cap is None:
        return None, _parameter_failure("configured max_results")
    effective = configured_cap if requested is None else requested
    if isinstance(effective, bool) or not isinstance(effective, int) or effective < 1:
        return None, _parameter_failure("max_results")
    if effective > configured_cap:
        return None, _result_cap_failure(configured_cap)
    return effective, None


def _python_grep(
    self: Any,
    pattern: str,
    path: str,
    include: str | None,
    max_results: int,
    *,
    expected_root: os.stat_result | None = None,
):
    """Scan every authorized target; retain only bounded results and diagnostics.

    Invalid UTF-8 is a read failure, never replacement-decoded negative evidence.
    The full stable traversal is consumed so match/omission and failure counts
    are exact, while retained records, physical input lines, and diagnostics are
    independently bounded.
    """

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Invalid regex: {exc}",
            metadata={
                "engine": "python-stable-nofollow",
                "search_failed": True,
                "canonical_paths": True,
                "nofollow_io": True,
                "stable_root_descriptor": True,
            },
        )

    validated_results, result_failure = _validate_result_window(self, max_results)
    if result_failure is not None:
        return result_failure
    assert validated_results is not None
    max_results = validated_results
    max_input_line_chars = getattr(
        self,
        "max_input_line_chars",
        _DEFAULT_MAX_INPUT_LINE_CHARS,
    )
    if (
        isinstance(max_input_line_chars, bool)
        or not isinstance(max_input_line_chars, int)
        or max_input_line_chars < 1
    ):
        return _parameter_failure("max_input_line_chars")

    search_path = Path(path)
    root_text = str(search_path)
    returned: list[str] = []
    failure_samples: list[str] = []
    failure_count = 0
    total_matches = 0
    input_line_limit_exceeded = False
    text_decode_error = False

    def record_failure(message: object) -> None:
        nonlocal failure_count
        failure_count += 1
        if len(failure_samples) >= _MAX_FAILURE_SAMPLES:
            return
        # Exceptions and paths are external data. Flatten and truncate each
        # sample so even a maliciously verbose error cannot defeat the bound.
        rendered = " ".join(str(message).splitlines())
        failure_samples.append(rendered[:_MAX_FAILURE_SAMPLE_CHARS])

    try:
        with closing(
            iter_stable_regular_files(
                search_path,
                expected_root=expected_root,
            )
        ) as targets:
            for relative, binary_stream, _metadata in targets:
                candidate = search_path if not relative.parts else search_path / relative
                if include and not fnmatch.fnmatch(candidate.name, include):
                    binary_stream.close()
                    continue

                decision = guard_canonical_path_strings(
                    requested=str(candidate),
                    resolved=str(candidate),
                    operation="read",
                    allowed_root=root_text,
                )
                if not decision.allowed:
                    binary_stream.close()
                    return guarded_path_failure("grep result", decision)

                try:
                    with io.TextIOWrapper(
                        binary_stream,
                        encoding="utf-8",
                        errors="strict",
                        newline=None,
                    ) as text_stream:
                        for line_number, line in _bounded_physical_lines(
                            text_stream,
                            max_chars=max_input_line_chars,
                        ):
                            if not regex.search(line):
                                continue
                            total_matches += 1
                            if len(returned) < max_results:
                                rendered = line.rstrip()[:4000]
                                returned.append(
                                    f"{candidate}:{line_number}: {rendered}"
                                )
                except _InputLineLimitExceeded as exc:
                    input_line_limit_exceeded = True
                    record_failure(f"{candidate}: {exc}")
                except UnicodeError as exc:
                    text_decode_error = True
                    record_failure(f"{candidate}: invalid UTF-8 text: {exc}")
                except OSError as exc:
                    record_failure(f"{candidate}: {exc}")
    except (OSError, StableTreeError) as exc:
        record_failure(f"{search_path}: {exc}")

    omitted_count = max(0, total_matches - len(returned))
    output_lines = list(returned)
    if omitted_count:
        output_lines.append(f"... ({omitted_count} more results truncated)")
    partial_output = "\n".join(output_lines)

    metadata = {
        "engine": "python-stable-nofollow",
        "match_count": total_matches,
        "returned_count": len(returned),
        "omitted_count": omitted_count,
        "truncated": omitted_count > 0,
        "read_error_count": failure_count,
        "diagnostic_sample_count": len(failure_samples),
        "diagnostic_sample_limit": _MAX_FAILURE_SAMPLES,
        "canonical_paths": True,
        "nofollow_io": True,
        "stable_root_descriptor": True,
        "host_output_bounded": True,
        "failure_diagnostics_bounded": True,
        "physical_input_line_bounded": True,
        "max_input_line_chars": max_input_line_chars,
        "text_encoding": "utf-8",
        "strict_text_decoding": True,
    }
    if input_line_limit_exceeded:
        metadata["input_line_limit_exceeded"] = True
    if text_decode_error:
        metadata["text_decode_error"] = True
    if failure_count:
        metadata["search_failed"] = True
        metadata["partial_results_available"] = bool(returned)
        metadata["diagnostics_omitted_count"] = max(
            0,
            failure_count - len(failure_samples),
        )
        diagnostic = "; ".join(failure_samples)
        omitted_failures = failure_count - len(failure_samples)
        if omitted_failures:
            diagnostic += f"; ... ({omitted_failures} additional failures omitted)"
        return ToolResult(
            success=False,
            output=partial_output,
            error=(
                "Python grep could not read every authorized target"
                + (f": {diagnostic}" if diagnostic else "")
            ),
            metadata=metadata,
        )

    return ToolResult(
        success=True,
        output=partial_output or "(no matches)",
        metadata=metadata,
    )


def _execute_secure_grep(
    self: Any,
    pattern: str,
    path: str = ".",
    include: str | None = None,
    max_results: int | None = None,
    **kwargs: Any,
):
    """Authorize once, then keep traversal bound to that exact root inode."""

    limit, result_failure = _validate_result_window(self, max_results)
    if result_failure is not None:
        return result_failure
    assert limit is not None
    max_input_line_chars = getattr(
        self,
        "max_input_line_chars",
        _DEFAULT_MAX_INPUT_LINE_CHARS,
    )
    if (
        isinstance(max_input_line_chars, bool)
        or not isinstance(max_input_line_chars, int)
        or max_input_line_chars < 1
    ):
        return _parameter_failure("max_input_line_chars")

    decision = resolve_guarded_path(
        path,
        operation="read",
        must_exist=True,
    )
    if not decision.allowed:
        return guarded_path_failure("grep", decision)

    root = Path(decision.resolved)
    if root == Path(root.anchor):
        return ToolResult(
            success=False,
            output="",
            error=(
                "Canonical path guard blocked grep: filesystem-root searches "
                "are outside the authorized task workspace."
            ),
            metadata=policy_guard_metadata("canonical_path_guard"),
        )

    if include:
        unsafe_reason = unsafe_relative_pattern_reason(include)
        if unsafe_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Canonical path guard blocked grep include: {unsafe_reason}.",
                metadata=policy_guard_metadata("canonical_path_guard"),
            )

    observed = guard_observed_text(
        " ".join(value for value in [path, pattern, include or ""] if value),
        operation="read",
    )
    if observed.blocked_by:
        return guarded_path_failure("grep", observed)

    try:
        expected_root = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Cannot capture authorized grep root identity: {exc}",
            metadata=policy_guard_metadata(
                "canonical_path_guard",
                canonical_path_checked=True,
                stable_root_descriptor=True,
            ),
        )

    symlink_failure = _base._preflight_symlink_tree(root, action="grep symlink")
    if symlink_failure is not None:
        return symlink_failure

    result = self._python_grep(
        pattern,
        decision.resolved,
        include,
        limit,
        expected_root=expected_root,
    )
    result.metadata = {
        **result.metadata,
        "external_search_disabled_for_path_safety": True,
    }
    return result


_base.GrepTool._python_grep = _python_grep
_base.GrepTool.execute = _execute_secure_grep
_base.GrepTool.get_schema = _bounded_grep_schema
_base.GrepTool.max_input_line_chars = _DEFAULT_MAX_INPUT_LINE_CHARS
GrepTool = _base.GrepTool
GlobTool = _base.GlobTool
