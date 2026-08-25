"""Canonical Harbor grep with reliable three-state and count handling."""

from __future__ import annotations

from typing import Any

from bench import _harbor_adapter_issue13_base as _base
from bench import _canonical_harbor_grep as _canonical


_COUNT_PREFIX = "__HL_GREP_COUNT__"

_HARBOR_GREP_COUNTING_PYTHON = r'''
import os
from pathlib import Path
import re
import stat
import sys

root_text = os.path.normpath(os.environ["HL_ROOT"])
max_results = int(os.environ.get("HL_MAX_RESULTS", "200"))
max_match_chars = int(os.environ.get("HL_MAX_MATCH_CHARS", "4000"))
max_input_line_chars = int(os.environ.get("HL_MAX_INPUT_LINE_CHARS", "1000000"))
try:
    regex = re.compile(os.environ["HL_PATTERN"])
except re.error as exc:
    print("invalid regular expression: %s" % exc, file=sys.stderr)
    raise SystemExit(2)
if max_results < 1 or max_match_chars < 1 or max_input_line_chars < 1:
    print("grep output and input-line bounds must be positive", file=sys.stderr)
    raise SystemExit(2)
results = []
total_matches = 0


def prohibited(path: str) -> bool:
    lowered = path.lower().replace("\\", "/")
    if "/terminal-bench-tasks/" in lowered or lowered.endswith("/terminal-bench-tasks"):
        return True
    if lowered == "/tests" or lowered.startswith("/tests/"):
        return True
    if lowered in {"/solution", "/solutions"}:
        return True
    if lowered.startswith("/solution/") or lowered.startswith("/solutions/"):
        return True
    if any(marker in lowered for marker in ("/trials/runs", "/trials/summaries", "/host/trials")):
        return True
    terminal_markers = ("/terminal-bench/", "/terminal_bench/", "/terminalbench/")
    if any(marker in lowered for marker in terminal_markers):
        if any(marker in lowered for marker in ("/tests/", "/solutions/", "/solution/")):
            return True
        if lowered.endswith("/task.toml"):
            return True
    return False


def within_root(path: str) -> bool:
    normalized = os.path.normpath(path)
    try:
        return os.path.commonpath([normalized, root_text]) == root_text
    except ValueError:
        return False


def authorize_path(path: str) -> None:
    if not within_root(path) or prohibited(path):
        raise SystemExit(73)


def authorize_symlink(path: str) -> None:
    try:
        resolved = str(Path(path).resolve(strict=True))
    except OSError:
        raise SystemExit(73)
    authorize_path(resolved)


def open_root_nofollow(path: str):
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("O_NOFOLLOW directory traversal is unavailable")
    target = Path(path)
    if not target.is_absolute():
        target = Path.cwd() / target
    target = Path(os.path.normpath(os.fspath(target)))
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    parent = os.open("/", directory_flags)
    try:
        for component in target.parts[1:-1]:
            next_parent = os.open(component, directory_flags, dir_fd=parent)
            os.close(parent)
            parent = next_parent
        descriptor = os.open(
            target.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent,
        )
        return parent, descriptor, target
    except Exception:
        os.close(parent)
        raise


def scan_descriptor(descriptor: int, display_path: str) -> None:
    global total_matches
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("grep target is not a regular file")
    if metadata.st_nlink != 1:
        print("multiply-linked regular file", file=sys.stderr)
        raise SystemExit(74)
    with os.fdopen(
        descriptor,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=None,
        closefd=False,
    ) as handle:
        line_number = 0
        while True:
            line = handle.readline(max_input_line_chars + 1)
            if line == "":
                break
            line_number += 1
            if len(line) > max_input_line_chars and not line.endswith("\n"):
                print(
                    "%s: physical line %d exceeds grep input limit of %d characters"
                    % (display_path, line_number, max_input_line_chars),
                    file=sys.stderr,
                )
                raise SystemExit(75)
            if not regex.search(line):
                continue
            total_matches += 1
            if len(results) < max_results:
                rendered = line.rstrip()[:max_match_chars]
                results.append(f"{display_path}:{line_number}: {rendered}")


def scan_root_file(descriptor: int, display_path: str) -> None:
    try:
        scan_descriptor(descriptor, display_path)
    finally:
        os.close(descriptor)


def scan_root_directory(root_descriptor: int) -> None:
    try:
        for relative_dir, dirnames, filenames, directory_fd in os.fwalk(
            ".",
            topdown=True,
            follow_symlinks=False,
            dir_fd=root_descriptor,
        ):
            relative_clean = "" if relative_dir == "." else relative_dir.removeprefix("./")
            display_dir = (
                root_text
                if not relative_clean
                else os.path.normpath(os.path.join(root_text, relative_clean))
            )
            authorize_path(display_dir)

            for name in list(dirnames):
                candidate = os.path.join(display_dir, name)
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    authorize_symlink(candidate)
                    dirnames.remove(name)
                    continue
                if not stat.S_ISDIR(metadata.st_mode):
                    dirnames.remove(name)
                    continue
                authorize_path(candidate)

            for name in filenames:
                candidate = os.path.join(display_dir, name)
                authorize_path(candidate)
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    authorize_symlink(candidate)
                    continue
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                try:
                    scan_descriptor(descriptor, candidate)
                finally:
                    os.close(descriptor)
    finally:
        os.close(root_descriptor)


authorize_path(root_text)
parent_descriptor, root_descriptor, root_path = open_root_nofollow(root_text)
os.close(parent_descriptor)
root_metadata = os.fstat(root_descriptor)
if stat.S_ISREG(root_metadata.st_mode):
    scan_root_file(root_descriptor, str(root_path))
elif stat.S_ISDIR(root_metadata.st_mode):
    scan_root_directory(root_descriptor)
else:
    os.close(root_descriptor)
    raise RuntimeError("grep root is not a regular file or directory")

print("__HL_GREP_COUNT__%d" % total_matches)
if results:
    print("\n".join(results))
'''


class HarborGrepTool(_canonical.HarborGrepTool):
    """Retain canonical scanning and add exact count/error semantics."""

    max_results = 200
    max_match_chars = 4000
    max_input_line_chars = 1_000_000

    def execute(self, pattern: str, path: str = ".", **_: Any) -> _base.ToolResult:
        observed = _canonical.guard_observed_text(
            f"{path} {pattern}",
            operation="read",
        )
        if observed.blocked_by:
            return _canonical.guarded_path_failure("grep", observed)

        root, failure = self._guard_environment_path(
            path,
            operation="read",
            must_exist=True,
        )
        if failure is not None:
            return failure
        if root == "/":
            return _canonical._canonical_policy_block(
                "grep",
                path,
                "filesystem-root searches are outside the authorized task workspace",
            )
        if self.max_results < 1 or self.max_match_chars < 1 or self.max_input_line_chars < 1:
            return _base.ToolResult(
                success=False,
                output="",
                error="grep output and input-line bounds must be positive",
                metadata={"search_failed": True, "parameter_validation_failed": True},
            )

        result = self._exec(
            "python3 -c " + _canonical.shlex.quote(_HARBOR_GREP_COUNTING_PYTHON),
            env={
                "HL_ROOT": root,
                "HL_PATTERN": pattern,
                "HL_MAX_RESULTS": str(self.max_results),
                "HL_MAX_MATCH_CHARS": str(self.max_match_chars),
                "HL_MAX_INPUT_LINE_CHARS": str(self.max_input_line_chars),
            },
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        unavailable = _canonical._base._terminal_environment_unavailable_text(
            stdout,
            stderr,
        )
        if unavailable:
            return _canonical._base._terminal_environment_unavailable_result(
                output=stdout,
                stderr=stderr,
                message=unavailable,
                exit_code=result.return_code,
            )
        if result.return_code != 0 and _canonical._base._looks_like_missing_python(
            stderr,
            stdout,
        ):
            return _base.ToolResult(
                success=False,
                output="",
                error=(
                    "Canonical path guard requires Python O_NOFOLLOW support for "
                    "Harbor grep; refusing an unsafe shell fallback."
                ),
                metadata=_canonical.policy_guard_metadata(
                    "canonical_path_guard",
                    nofollow_io_unavailable=True,
                    exit_code=result.return_code,
                    search_failed=True,
                ),
            )
        if result.return_code == 73:
            return _canonical._canonical_policy_block(
                "grep result",
                path,
                "a resolved search target was prohibited or escaped the authorized root",
            )
        if result.return_code == 74:
            return _base.ToolResult(
                success=False,
                output="",
                error=(
                    "Canonical path guard blocked grep: multiply linked regular files "
                    "cannot be attributed to one authorized path."
                ),
                metadata=_canonical.policy_guard_metadata(
                    "canonical_path_guard",
                    exit_code=74,
                    search_failed=True,
                    canonical_path_checked=True,
                    nofollow_io=True,
                    hardlink_alias_blocked=True,
                ),
            )
        if result.return_code == 75:
            return _base.ToolResult(
                success=False,
                output="",
                error=stderr.strip() or "grep input line exceeded the configured safe limit",
                metadata={
                    "exit_code": 75,
                    "engine": "python-nofollow",
                    "search_failed": True,
                    "input_line_limit_exceeded": True,
                    "physical_input_line_bounded": True,
                    "max_input_line_chars": self.max_input_line_chars,
                    "host_output_bounded": True,
                    "canonical_paths": True,
                    "nofollow_io": True,
                },
            )
        if result.return_code != 0:
            return _base.ToolResult(
                success=False,
                output=stdout.strip(),
                error=stderr.strip() or f"grep exited with code {result.return_code}",
                metadata={
                    "exit_code": result.return_code,
                    "engine": "python-nofollow",
                    "search_failed": True,
                    "physical_input_line_bounded": True,
                    "max_input_line_chars": self.max_input_line_chars,
                    "host_output_bounded": True,
                    "canonical_paths": True,
                    "nofollow_io": True,
                },
            )

        lines = stdout.splitlines()
        if not lines or not lines[0].startswith(_COUNT_PREFIX):
            return _base.ToolResult(
                success=False,
                output="",
                error="bounded canonical grep returned malformed count metadata",
                metadata={
                    "exit_code": 0,
                    "engine": "python-nofollow",
                    "search_failed": True,
                    "physical_input_line_bounded": True,
                    "max_input_line_chars": self.max_input_line_chars,
                    "host_output_bounded": True,
                    "canonical_paths": True,
                    "nofollow_io": True,
                },
            )
        try:
            total_matches = int(lines[0][len(_COUNT_PREFIX) :])
        except ValueError:
            return _base.ToolResult(
                success=False,
                output="",
                error="bounded canonical grep returned a non-numeric match count",
                metadata={
                    "exit_code": 0,
                    "engine": "python-nofollow",
                    "search_failed": True,
                    "physical_input_line_bounded": True,
                    "max_input_line_chars": self.max_input_line_chars,
                    "host_output_bounded": True,
                    "canonical_paths": True,
                    "nofollow_io": True,
                },
            )

        returned_lines = lines[1 : 1 + self.max_results]
        omitted_count = max(0, total_matches - len(returned_lines))
        output_lines = list(returned_lines)
        if omitted_count:
            output_lines.append(f"... ({omitted_count} more results truncated)")
        return _base.ToolResult(
            success=True,
            output="\n".join(output_lines) if output_lines else "(no matches)",
            error="",
            metadata={
                "exit_code": 0,
                "engine": "python-nofollow",
                "match_count": total_matches,
                "returned_count": len(returned_lines),
                "omitted_count": omitted_count,
                "truncated": omitted_count > 0,
                "search_failed": False,
                "physical_input_line_bounded": True,
                "max_input_line_chars": self.max_input_line_chars,
                "host_output_bounded": True,
                "canonical_paths": True,
                "nofollow_io": True,
            },
        )


_base.HarborGrepTool = HarborGrepTool
HLWorkerHarborAgent = _base.HLWorkerHarborAgent
