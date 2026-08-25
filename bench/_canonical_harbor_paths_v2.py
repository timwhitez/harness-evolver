"""Harbor file tools that operate on authorized canonical paths without symlink following."""

from __future__ import annotations

import base64
import hashlib
import shlex
from typing import Any

from bench import _harbor_adapter_issue4_base as _base
from bench._canonical_harbor_paths import _CanonicalHarborPathMixin
from harness.tools.base import ToolResult, policy_guard_metadata
from harness.tools.shell import (
    deliverable_size_cap_write_reason,
    staged_dependency_script_reason,
)


_SAFE_PREAMBLE = r'''
import base64, hashlib, os, pathlib, secrets, stat, sys

def parent_fd(raw_path, create=False):
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW unavailable")
    path = pathlib.Path(raw_path)
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    path = pathlib.Path(os.path.normpath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parent.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=descriptor)
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, path.name
    except Exception:
        os.close(descriptor)
        raise

def open_regular(parent, name):
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent,
    )
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RuntimeError("not a regular file")
    return descriptor, metadata

def write_atomic(parent, name, payload, mode=None):
    try:
        current = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISLNK(current.st_mode):
            raise RuntimeError("symlink target")
        if mode is None:
            mode = current.st_mode
    except FileNotFoundError:
        pass
    temporary = ".%s.tmp-%s" % (name, secrets.token_hex(8))
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
        if mode is not None:
            os.fchmod(descriptor, stat.S_IMODE(mode))
    finally:
        os.close(descriptor)
    try:
        try:
            current = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if stat.S_ISLNK(current.st_mode):
                raise RuntimeError("symlink target")
        except FileNotFoundError:
            pass
        os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass
'''

_SECURE_READ = _SAFE_PREAMBLE + r'''
try:
    parent, name = parent_fd(os.environ["HL_FILE_PATH"], create=False)
    try:
        descriptor, _ = open_regular(parent, name)
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8", errors="replace", closefd=False) as stream:
                lines = stream.read().splitlines()
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)
    offset = max(int(os.environ.get("HL_OFFSET", "1")), 1)
    limit = max(int(os.environ.get("HL_LIMIT", "2000")), 1)
    start = offset - 1
    for number, line in enumerate(lines[start:start + limit], start=offset):
        print(f"{number}\t{line}")
    if start + limit < len(lines):
        print(f"... ({len(lines) - start - limit} more lines)")
except Exception:
    raise SystemExit("secure nofollow read failed")
'''

_SECURE_RAW_READ = _SAFE_PREAMBLE + r'''
try:
    parent, name = parent_fd(os.environ["HL_FILE_PATH"], create=False)
    try:
        descriptor, metadata = open_regular(parent, name)
        try:
            chunks = []
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)
    payload = b"".join(chunks)
    print(base64.b64encode(payload).decode("ascii"))
except FileNotFoundError:
    raise SystemExit(44)
except Exception:
    raise SystemExit("secure nofollow read failed")
'''

_SECURE_WRITE = _SAFE_PREAMBLE + r'''
try:
    payload = base64.b64decode(os.environ["HL_FILE_CONTENT"], validate=True)
    append = os.environ.get("HL_APPEND") == "1"
    parent, name = parent_fd(os.environ["HL_FILE_PATH"], create=True)
    try:
        if append:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o666,
                dir_fd=parent,
            )
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            write_atomic(parent, name, payload)
    finally:
        os.close(parent)
    print("write complete")
except Exception:
    raise SystemExit("secure nofollow write failed")
'''

_SECURE_EDIT = _SAFE_PREAMBLE + r'''
try:
    old = base64.b64decode(os.environ["HL_OLD"], validate=True).decode("utf-8")
    new = base64.b64decode(os.environ["HL_NEW"], validate=True).decode("utf-8")
    replace_all = os.environ.get("HL_ALL") == "1"
    expected = os.environ["HL_EXPECTED_SHA256"]
    parent, name = parent_fd(os.environ["HL_FILE_PATH"], create=False)
    try:
        descriptor, metadata = open_regular(parent, name)
        try:
            chunks = []
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
        payload = b"".join(chunks)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError("target changed after authorization")
        text = payload.decode("utf-8")
        count = text.count(old)
        if count == 0:
            raise RuntimeError("old_string not found")
        if count > 1 and not replace_all:
            raise RuntimeError("old_string is not unique")
        updated = text.replace(old, new, -1 if replace_all else 1).encode("utf-8")
        write_atomic(parent, name, updated, metadata.st_mode)
    finally:
        os.close(parent)
    print(f"replaced {count if replace_all else 1} occurrence(s)")
except Exception:
    raise SystemExit("secure nofollow edit failed")
'''


class _SecureHarborMixin(_CanonicalHarborPathMixin):
    def _run_secure_python(
        self,
        script: str,
        *,
        env: dict[str, str],
    ) -> ToolResult:
        result = self._exec(f"python3 -c {shlex.quote(script)}", env=env)
        output = result.stdout or ""
        error = result.stderr or (
            "" if result.return_code == 0 else f"exit code: {result.return_code}"
        )
        unavailable = _base._terminal_environment_unavailable_text(output, error)
        if unavailable:
            return _base._terminal_environment_unavailable_result(
                output=output,
                stderr=error,
                message=unavailable,
                exit_code=result.return_code,
            )
        if result.return_code == 0:
            return ToolResult(
                success=True,
                output=output,
                error="",
                metadata={
                    "exit_code": 0,
                    "canonical_path_checked": True,
                    "nofollow_io": True,
                },
            )
        if _base._looks_like_missing_python(error, output):
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Canonical path guard requires Python O_NOFOLLOW support in the "
                    "task environment; refusing an unsafe shell fallback."
                ),
                metadata=policy_guard_metadata(
                    "canonical_path_guard",
                    nofollow_io_unavailable=True,
                    exit_code=result.return_code,
                ),
            )
        return ToolResult(
            success=False,
            output=output,
            error=error or "secure nofollow operation failed",
            metadata=policy_guard_metadata(
                "canonical_path_guard",
                canonical_path_checked=True,
                nofollow_io=True,
                exit_code=result.return_code,
            ),
        )

    def _secure_raw_read(self, path: str) -> tuple[str | None, ToolResult | None]:
        result = self._run_secure_python(
            _SECURE_RAW_READ,
            env={"HL_FILE_PATH": path},
        )
        if result.success:
            try:
                payload = base64.b64decode(result.output.strip(), validate=True)
                return payload.decode("utf-8", errors="replace"), None
            except Exception:
                return None, ToolResult(
                    success=False,
                    output="",
                    error="Secure read returned invalid encoded content",
                    metadata=policy_guard_metadata("canonical_path_guard"),
                )
        if result.metadata.get("exit_code") == 44:
            return "", None
        return None, result


class HarborFileReadTool(_SecureHarborMixin, _base.HarborFileReadTool):
    def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int | None = 2000,
        **kwargs: Any,
    ) -> ToolResult:
        resolved, failure = self._guard_environment_path(
            file_path,
            operation="read",
            must_exist=True,
        )
        if failure is not None:
            return failure
        return self._run_secure_python(
            _SECURE_READ,
            env={
                "HL_FILE_PATH": resolved,
                "HL_OFFSET": str(offset),
                "HL_LIMIT": str(limit or 2000),
            },
        )


class HarborFileWriteTool(_SecureHarborMixin, _base.HarborFileWriteTool):
    def execute(
        self,
        file_path: str,
        content: str,
        append: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        resolved, failure = self._guard_environment_path(
            file_path,
            operation="write",
            must_exist=False,
        )
        if failure is not None:
            return failure

        effective_content = content
        if append:
            current, read_failure = self._secure_raw_read(resolved)
            if read_failure is not None:
                return read_failure
            effective_content = (current or "") + content
        staged_reason = staged_dependency_script_reason(resolved, effective_content)
        if staged_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked write: {staged_reason}",
                metadata=policy_guard_metadata("staged_dependency_script_guard"),
            )
        size_reason = deliverable_size_cap_write_reason(resolved, effective_content)
        if size_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked write: {size_reason}",
                metadata=policy_guard_metadata("deliverable_size_cap_write_guard"),
            )
        return self._run_secure_python(
            _SECURE_WRITE,
            env={
                "HL_FILE_PATH": resolved,
                "HL_FILE_CONTENT": base64.b64encode(content.encode("utf-8")).decode("ascii"),
                "HL_APPEND": "1" if append else "0",
            },
        )


class HarborFileEditTool(_SecureHarborMixin, _base.HarborFileEditTool):
    def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        resolved, failure = self._guard_environment_path(
            file_path,
            operation="edit",
            must_exist=True,
        )
        if failure is not None:
            return failure
        current, read_failure = self._secure_raw_read(resolved)
        if read_failure is not None:
            return read_failure
        assert current is not None
        count = current.count(old_string)
        if count == 0:
            return ToolResult(success=False, output="", error="old_string not found")
        if count > 1 and not replace_all:
            return ToolResult(
                success=False,
                output="",
                error=f"old_string occurs {count} times",
            )
        updated = current.replace(old_string, new_string, -1 if replace_all else 1)
        staged_reason = staged_dependency_script_reason(resolved, updated)
        if staged_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked edit: {staged_reason}",
                metadata=policy_guard_metadata("staged_dependency_script_guard"),
            )
        size_reason = deliverable_size_cap_write_reason(resolved, updated)
        if size_reason:
            return ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked edit: {size_reason}",
                metadata=policy_guard_metadata("deliverable_size_cap_write_guard"),
            )
        expected = hashlib.sha256(current.encode("utf-8")).hexdigest()
        return self._run_secure_python(
            _SECURE_EDIT,
            env={
                "HL_FILE_PATH": resolved,
                "HL_OLD": base64.b64encode(old_string.encode("utf-8")).decode("ascii"),
                "HL_NEW": base64.b64encode(new_string.encode("utf-8")).decode("ascii"),
                "HL_ALL": "1" if replace_all else "0",
                "HL_EXPECTED_SHA256": expected,
            },
        )
