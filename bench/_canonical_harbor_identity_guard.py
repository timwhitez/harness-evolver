'''Snapshot-aware Harbor publication with explicit committed/uncertain outcomes.

Local and Harbor writers embed the same tested publication core. Cooperating
writers serialize; external races retain durable recovery information instead
of performing a destructive exchange-back. This is not an inode-CAS primitive.
'''
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import PurePosixPath
import json
import re
from typing import Any

from bench import _canonical_harbor_hardlink as _base
from harness.tools.publication import embedded_publication_source

_v2 = _base._v2
_v3 = _base._v3
_v4 = _base._v4

_RENAME_FUNCTION = r'''def renameat2(parent, source, destination, flags):
    import ctypes, platform
    if not sys.platform.startswith("linux"):
        raise RuntimeError("race-safe conditional publication requires Linux renameat2")
    numbers = {
        "x86_64": 316, "amd64": 316, "i386": 353, "i686": 353,
        "aarch64": 276, "arm64": 276, "armv7l": 382,
        "ppc64": 357, "ppc64le": 357, "s390x": 347, "riscv64": 276,
    }
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    function = getattr(libc, "renameat2", None)
    if function is not None:
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                             ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(parent, source_bytes, parent, destination_bytes, flags)
    else:
        number = numbers.get(platform.machine().lower())
        if number is None:
            raise RuntimeError("renameat2 syscall number unavailable")
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        result = syscall(ctypes.c_long(number), ctypes.c_int(parent),
                         ctypes.c_char_p(source_bytes), ctypes.c_int(parent),
                         ctypes.c_char_p(destination_bytes), ctypes.c_uint(flags))
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)

'''
_ATOMIC_FUNCTION = _RENAME_FUNCTION + embedded_publication_source(include_opener=False) + r'''
def write_atomic(parent, name, payload, mode=None, expected_identity=None,
                 expected_sha256=None, expected_missing=False):
    import json
    outcome = publish_bytes(parent, name, payload, renameat2, mode=mode,
                            expected_identity=expected_identity,
                            expected_sha256=expected_sha256,
                            expected_missing=expected_missing)
    print("__HL_PUBLICATION__" + json.dumps(outcome, allow_nan=False), flush=True)
    if outcome["atomic_replace"] is not True:
        raise RuntimeError(outcome["publication_error"] or "publication indeterminate")
    return outcome["directory_fsync"]
'''
_ATOMIC_FUNCTION_PATTERN = re.compile(
    r"def write_atomic\(parent, name, payload, mode=None\):\n"
    r".*?"
    r"    finally:\n"
    r"        try:\n"
    r"            os\.unlink\(temporary, dir_fd=parent\)\n"
    r"        except FileNotFoundError:\n"
    r"            pass\n", flags=re.DOTALL,
)


def _replace_atomic_function(script: str) -> str:
    # A replacement callable preserves literal backslashes in embedded Python.
    updated, count = _ATOMIC_FUNCTION_PATTERN.subn(lambda _: _ATOMIC_FUNCTION, script, count=1)
    if count != 1:
        raise RuntimeError("canonical Harbor write_atomic contract changed unexpectedly")
    return updated


_v2._SAFE_PREAMBLE = _replace_atomic_function(_v2._SAFE_PREAMBLE)
_v2._SECURE_EDIT = _replace_atomic_function(_v2._SECURE_EDIT)
_v3._SECURE_ATOMIC_WRITE = _replace_atomic_function(_v3._SECURE_ATOMIC_WRITE)

_EDIT_WRITE = "        write_atomic(parent, name, updated, metadata.st_mode)\n"
_EDIT_WRITE_GUARDED = '''        directory_synced = write_atomic(
            parent, name, updated, metadata.st_mode,
            expected_identity=(int(metadata.st_dev), int(metadata.st_ino)),
            expected_sha256=expected,
        )
'''
if _EDIT_WRITE not in _v2._SECURE_EDIT:
    raise RuntimeError("canonical Harbor edit publication contract changed unexpectedly")
_v2._SECURE_EDIT = _v2._SECURE_EDIT.replace(_EDIT_WRITE, _EDIT_WRITE_GUARDED, 1)

_WRITE_CALL = "        write_atomic(parent, name, payload)\n"
_WRITE_CALL_GUARDED = '''        expected_identity = None
        expected_present = os.environ.get("HL_EXPECTED_PRESENT")
        if expected_present == "1":
            expected_identity = (int(os.environ["HL_EXPECTED_DEV"]), int(os.environ["HL_EXPECTED_INO"]))
        write_atomic(
            parent, name, payload,
            expected_identity=expected_identity,
            expected_sha256=(os.environ.get("HL_EXPECTED_SHA256") if expected_present == "1" else None),
            expected_missing=expected_present == "0",
        )
'''
if _WRITE_CALL not in _v3._SECURE_ATOMIC_WRITE:
    raise RuntimeError("canonical Harbor write publication contract changed unexpectedly")
_v3._SECURE_ATOMIC_WRITE = _v3._SECURE_ATOMIC_WRITE.replace(_WRITE_CALL, _WRITE_CALL_GUARDED, 1)

_SECURE_SNAPSHOT = _v2._SAFE_PREAMBLE + r'''
import json
try:
    parent, name = parent_fd(os.environ["HL_FILE_PATH"], create=True)
    try:
        try:
            descriptor, metadata = open_regular(parent, name)
        except FileNotFoundError:
            snapshot = {"present": False, "payload": "", "sha256": hashlib.sha256(b"").hexdigest()}
        else:
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
            snapshot = {"present": True, "dev": int(metadata.st_dev), "ino": int(metadata.st_ino),
                        "payload": base64.b64encode(payload).decode("ascii"),
                        "sha256": hashlib.sha256(payload).hexdigest()}
    finally:
        os.close(parent)
    print(json.dumps(snapshot, sort_keys=True))
except Exception:
    raise SystemExit("secure nofollow snapshot failed")
'''


@dataclass(frozen=True)
class _Snapshot:
    present: bool
    text: str
    dev: int | None
    ino: int | None
    sha256: str


def _publication_metadata(result: _v2.ToolResult, *, identity_verified: bool) -> _v2.ToolResult:
    lines = result.output.splitlines()
    prefix = "__HL_PUBLICATION__"
    records = [line[len(prefix):] for line in lines if line.startswith(prefix)]
    if not records and not result.metadata.get("publication_attempted") and not result.success:
        # Authorization, input matching, and snapshot failures precede publication.
        return result
    try:
        if len(records) != 1:
            raise ValueError("publication must report exactly one outcome")
        outcome = json.loads(records[0])
        states = {"not_published": False, "published": True, "indeterminate": None}
        state = outcome["publication_state"]
        if (outcome.get("publication_protocol") != "hl-publication-v1" or state not in states
                or outcome.get("atomic_replace") is not states[state]):
            raise ValueError("invalid publication state")
        for flag in ("directory_fsync", "durability_warning", "cleanup_warning", "no_auto_retry"):
            if not isinstance(outcome.get(flag), bool):
                raise ValueError("invalid publication flag")
        entries = outcome.get("recovery_entries")
        if not isinstance(entries, list) or any(not isinstance(e, str) or '/' in e or e in ('', '.', '..') for e in entries):
            raise ValueError("invalid recovery entries")
        if not isinstance(outcome.get("publication_error"), str):
            raise ValueError("invalid publication diagnostic")
    except (TypeError, ValueError, KeyError) as exc:
        result.success = False
        result.error = f"Secure publication outcome unknown; reconcile before retry: {exc}"
        result.metadata.update(secure_write_protocol_error=True, publication_state="indeterminate",
                               atomic_replace=None, no_auto_retry=True)
        return result
    previous_error = result.error
    result.success = outcome["atomic_replace"] is True
    result.error = "" if result.success else outcome["publication_error"]
    result.output = "\n".join(line for line in lines if not line.startswith(prefix))
    result.metadata.update(outcome)
    result.metadata["target_identity_verified"] = result.success and identity_verified
    if result.success and (outcome["cleanup_warning"] or outcome["durability_warning"] or previous_error):
        result.output += "\nWarning: content is published; do not blindly retry. Inspect publication recovery metadata."
    return result


class HarborFileWriteTool(_base.HarborFileWriteTool):
    '''Publish over the exact observed entry and report recovery state.'''

    def _secure_snapshot(self, path: str) -> tuple[_Snapshot | None, _v2.ToolResult | None]:
        result = self._run_secure_python(_SECURE_SNAPSHOT, env={"HL_FILE_PATH": path})
        if not result.success:
            return None, result
        try:
            payload = json.loads(result.output)
            present = payload["present"]
            if not isinstance(present, bool):
                raise TypeError("present is not boolean")
            raw = base64.b64decode(payload.get("payload", ""), validate=True)
            digest = str(payload["sha256"])
            dev = int(payload["dev"]) if present else None
            ino = int(payload["ino"]) if present else None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return None, _v2.ToolResult(success=False, output="", error=f"Secure snapshot protocol failed: {exc}",
                metadata={"canonical_path_checked": True, "nofollow_io": True,
                          "secure_read_protocol_error": True, "atomic_replace": False})
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            return None, _v2.ToolResult(success=False, output="", error=f"Cannot append non-UTF-8 text safely: {exc}",
                metadata={"text_decode_error": True, "encoding": "utf-8",
                          "canonical_path_checked": True, "nofollow_io": True, "atomic_replace": False})
        return _Snapshot(present, text, dev, ino, digest), None

    def execute(self, file_path: str, content: str, append: bool = False, **kwargs: Any) -> _v2.ToolResult:
        resolved, failure = self._guard_environment_path(file_path, operation="write", must_exist=False)
        if failure is not None:
            return failure
        snapshot: _Snapshot | None = None
        effective_content = content
        if append:
            snapshot, read_failure = self._secure_snapshot(resolved)
            if read_failure is not None:
                return read_failure
            assert snapshot is not None
            effective_content = snapshot.text + content
        staged_reason = _v2.staged_dependency_script_reason(resolved, effective_content)
        if staged_reason:
            return _v2.ToolResult(success=False, output="", error=f"Worker file policy blocked write: {staged_reason}",
                metadata=_v2.policy_guard_metadata("staged_dependency_script_guard"))
        size_reason = _v2.deliverable_size_cap_write_reason(resolved, effective_content)
        if size_reason:
            return _v2.ToolResult(success=False, output="", error=f"Worker file policy blocked write: {size_reason}",
                metadata=_v2.policy_guard_metadata("deliverable_size_cap_write_guard", path=resolved,
                    content_bytes=len(effective_content.encode("utf-8")), limit_bytes=5000))
        environment = {"HL_FILE_PATH": resolved,
                       "HL_FILE_CONTENT": base64.b64encode(effective_content.encode("utf-8")).decode("ascii")}
        if snapshot is not None:
            environment["HL_EXPECTED_PRESENT"] = "1" if snapshot.present else "0"
            if snapshot.present:
                environment["HL_EXPECTED_SHA256"] = snapshot.sha256
                assert snapshot.dev is not None and snapshot.ino is not None
                environment["HL_EXPECTED_DEV"] = str(snapshot.dev)
                environment["HL_EXPECTED_INO"] = str(snapshot.ino)
        result = self._run_secure_python(_v3._SECURE_ATOMIC_WRITE, env=environment)
        result.metadata["publication_attempted"] = True
        result = _publication_metadata(result, identity_verified=append)
        result.metadata["recovery_directory"] = str(PurePosixPath(resolved).parent)
        if result.success:
            result.metadata["atomic_append"] = append
        return result


class HarborFileEditTool(_base.HarborFileEditTool):
    '''Retain guarded snapshot editing and expose publication outcomes.'''

    def execute(self, file_path: str, old_string: str, new_string: str,
                replace_all: bool = False, **kwargs: Any) -> _v2.ToolResult:
        staged_reason = _v2.staged_dependency_script_reason(file_path, new_string)
        if staged_reason:
            return _v2.ToolResult(success=False, output="", error=f"Worker file policy blocked edit: {staged_reason}",
                metadata=_v2.policy_guard_metadata("staged_dependency_script_guard"))
        return _publication_metadata(super().execute(file_path, old_string, new_string,
            replace_all=replace_all, **kwargs), identity_verified=True)


HarborFileReadTool = _base.HarborFileReadTool
ToolResult = _v2.ToolResult
__all__ = ["HarborFileReadTool", "HarborFileEditTool", "HarborFileWriteTool", "ToolResult", "_SECURE_SNAPSHOT"]
