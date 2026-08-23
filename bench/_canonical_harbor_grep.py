"""Canonical-path protected Harbor grep tool."""

from __future__ import annotations

import shlex
from typing import Any

from bench import _harbor_adapter_issue4_base as _base
from bench._canonical_harbor_paths import (
    _CanonicalHarborPathMixin,
    _canonical_policy_block,
)
from harness.tools.base import ToolResult, policy_guard_metadata
from harness.tools.canonical_path_guard import (
    guard_observed_text,
    guarded_path_failure,
)

_HARBOR_GREP_PYTHON = r"""
import os
from pathlib import Path
import re
import stat

root_text = os.path.normpath(os.environ["HL_ROOT"])
regex = re.compile(os.environ["HL_PATTERN"])
results = []


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
            next_parent = os.open(
                component,
                directory_flags,
                dir_fd=parent,
            )
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
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("grep target is not a regular file")
    with os.fdopen(
        descriptor,
        "r",
        encoding="utf-8",
        errors="replace",
        closefd=False,
    ) as handle:
        for line_number, line in enumerate(handle, 1):
            if regex.search(line):
                results.append(f"{display_path}:{line_number}: {line.rstrip()}")
                if len(results) >= 200:
                    return


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
                if len(results) >= 200:
                    return
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

print("\n".join(results))
"""


class HarborGrepTool(_CanonicalHarborPathMixin, _base.HarborGrepTool):
    def execute(self, pattern: str, path: str = ".", **_: Any) -> ToolResult:
        observed = guard_observed_text(f"{path} {pattern}", operation="read")
        if observed.blocked_by:
            return guarded_path_failure("grep", observed)

        root, failure = self._guard_environment_path(
            path,
            operation="read",
            must_exist=True,
        )
        if failure is not None:
            return failure
        if root == "/":
            return _canonical_policy_block(
                "grep",
                path,
                "filesystem-root searches are outside the authorized task workspace",
            )

        env = {"HL_ROOT": root, "HL_PATTERN": pattern}
        result = self._exec(
            "python3 -c " + shlex.quote(_HARBOR_GREP_PYTHON),
            env=env,
        )
        output = result.stdout or ""
        error = result.stderr or ""

        unavailable = _base._terminal_environment_unavailable_text(output, error)
        if unavailable:
            return _base._terminal_environment_unavailable_result(
                output=output,
                stderr=error,
                message=unavailable,
                exit_code=result.return_code,
            )
        if result.return_code != 0 and _base._looks_like_missing_python(error, output):
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Canonical path guard requires Python O_NOFOLLOW support for "
                    "Harbor grep; refusing an unsafe shell fallback."
                ),
                metadata=policy_guard_metadata(
                    "canonical_path_guard",
                    nofollow_io_unavailable=True,
                    exit_code=result.return_code,
                ),
            )
        if result.return_code == 73:
            return _canonical_policy_block(
                "grep result",
                path,
                "a resolved search target was prohibited or escaped the authorized root",
            )
        return ToolResult(
            success=result.return_code == 0,
            output=output or "(no matches)",
            error=error or (
                "" if result.return_code == 0 else f"exit code: {result.return_code}"
            ),
            metadata={
                "exit_code": result.return_code,
                "canonical_paths": True,
                "nofollow_io": True,
            },
        )
