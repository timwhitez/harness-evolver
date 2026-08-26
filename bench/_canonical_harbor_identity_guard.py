'''Conditional Harbor publication for append/edit transforms.

The canonical/hard-link layer authorizes and reads content through stable
descriptor-relative opens. This final layer binds publication to the exact
unique inode and byte snapshot consumed by the transform, and treats a
post-replace directory-fsync failure as a durability warning rather than a
false pre-publication failure.
'''

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import re
from typing import Any

from bench import _canonical_harbor_hardlink as _base

_v2 = _base._v2
_v3 = _base._v3
_v4 = _base._v4


_ATOMIC_FUNCTION = r'''def renameat2(parent, source, destination, flags):
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
        function.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            parent, source_bytes, parent, destination_bytes, flags
        )
    else:
        number = numbers.get(platform.machine().lower())
        if number is None:
            raise RuntimeError("renameat2 syscall number unavailable")
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        result = syscall(
            ctypes.c_long(number), ctypes.c_int(parent),
            ctypes.c_char_p(source_bytes), ctypes.c_int(parent),
            ctypes.c_char_p(destination_bytes), ctypes.c_uint(flags),
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)

def write_atomic(
    parent,
    name,
    payload,
    mode=None,
    expected_identity=None,
    expected_sha256=None,
    expected_missing=False,
):
    conditional_existing = expected_identity is not None or expected_sha256 is not None
    if expected_missing and conditional_existing:
        raise RuntimeError("missing and existing target expectations conflict")

    def read_current(current_name):
        try:
            current = os.stat(current_name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            if expected_missing:
                return None
            if conditional_existing:
                raise RuntimeError("target disappeared before publication")
            return None
        if stat.S_ISLNK(current.st_mode):
            raise RuntimeError("symlink target")
        if expected_missing:
            raise RuntimeError("target appeared before publication")
        if not conditional_existing:
            return current
        descriptor, metadata = open_regular(parent, current_name)
        try:
            chunks = []
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(descriptor)
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if expected_identity is not None and identity != tuple(expected_identity):
            raise RuntimeError("target changed identity before publication")
        if (
            expected_sha256 is not None
            and hashlib.sha256(b"".join(chunks)).hexdigest() != expected_sha256
        ):
            raise RuntimeError("target content changed before publication")
        return metadata

    def assert_name_identity(current_name, expected):
        descriptor = os.open(
            current_name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent,
        )
        try:
            actual = os.fstat(descriptor)
            identity = (int(actual.st_dev), int(actual.st_ino))
            if identity != tuple(expected):
                raise RuntimeError("published target changed before rollback")
        finally:
            os.close(descriptor)

    current = read_current(name)
    if mode is None and current is not None:
        mode = current.st_mode

    temporary = ".%s.tmp-%s" % (name, secrets.token_hex(8))
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o666,
        dir_fd=parent,
    )
    temporary_exists = True
    exchanged = False
    published = False
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            if mode is not None:
                os.fchmod(descriptor, stat.S_IMODE(mode) & 0o777)
            os.fsync(descriptor)
        new_metadata = os.fstat(descriptor)
        new_identity = (int(new_metadata.st_dev), int(new_metadata.st_ino))
        os.close(descriptor)
        descriptor = -1

        if conditional_existing:
            renameat2(parent, temporary, name, 2)
            exchanged = True
            try:
                read_current(temporary)
            except Exception as validation_error:
                try:
                    assert_name_identity(name, new_identity)
                    renameat2(parent, temporary, name, 2)
                    exchanged = False
                except Exception as rollback_error:
                    raise RuntimeError(
                        "conditional publication validation failed and rollback "
                        "could not complete; displaced target retained at " + temporary
                    ) from rollback_error
                raise validation_error
            os.unlink(temporary, dir_fd=parent)
            temporary_exists = False
            exchanged = False
            published = True
        elif expected_missing:
            renameat2(parent, temporary, name, 1)
            temporary_exists = False
            published = True
        else:
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            temporary_exists = False
            published = True

        try:
            os.fsync(parent)
        except OSError:
            return False
        return True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_exists and not exchanged:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
        if not published and exchanged:
            pass
'''


_ATOMIC_FUNCTION_PATTERN = re.compile(
    r"def write_atomic\(parent, name, payload, mode=None\):\n"
    r".*?"
    r"    finally:\n"
    r"        try:\n"
    r"            os\.unlink\(temporary, dir_fd=parent\)\n"
    r"        except FileNotFoundError:\n"
    r"            pass\n",
    flags=re.DOTALL,
)


def _replace_atomic_function(script: str) -> str:
    updated, count = _ATOMIC_FUNCTION_PATTERN.subn(_ATOMIC_FUNCTION, script, count=1)
    if count != 1:
        raise RuntimeError("canonical Harbor write_atomic contract changed unexpectedly")
    return updated


_v2._SAFE_PREAMBLE = _replace_atomic_function(_v2._SAFE_PREAMBLE)
_v2._SECURE_EDIT = _replace_atomic_function(_v2._SECURE_EDIT)
_v3._SECURE_ATOMIC_WRITE = _replace_atomic_function(_v3._SECURE_ATOMIC_WRITE)


_EDIT_WRITE = "        write_atomic(parent, name, updated, metadata.st_mode)\n"
_EDIT_WRITE_GUARDED = '''        directory_synced = write_atomic(
            parent,
            name,
            updated,
            metadata.st_mode,
            expected_identity=(int(metadata.st_dev), int(metadata.st_ino)),
            expected_sha256=expected,
        )
'''
if _EDIT_WRITE not in _v2._SECURE_EDIT:
    raise RuntimeError("canonical Harbor edit publication contract changed unexpectedly")
_v2._SECURE_EDIT = _v2._SECURE_EDIT.replace(
    _EDIT_WRITE,
    _EDIT_WRITE_GUARDED,
    1,
)
_EDIT_RESULT = '    print(f"replaced {count if replace_all else 1} occurrence(s)")\n'
if _EDIT_RESULT not in _v2._SECURE_EDIT:
    raise RuntimeError("canonical Harbor edit result contract changed unexpectedly")
_v2._SECURE_EDIT = _v2._SECURE_EDIT.replace(
    _EDIT_RESULT,
    _EDIT_RESULT
    + '    print("directory_synced=1" if directory_synced else "directory_synced=0")\n',
    1,
)


_WRITE_CALL = "        write_atomic(parent, name, payload)\n"
_WRITE_CALL_GUARDED = '''        expected_identity = None
        expected_present = os.environ.get("HL_EXPECTED_PRESENT")
        if expected_present == "1":
            expected_identity = (
                int(os.environ["HL_EXPECTED_DEV"]),
                int(os.environ["HL_EXPECTED_INO"]),
            )
        directory_synced = write_atomic(
            parent,
            name,
            payload,
            expected_identity=expected_identity,
            expected_sha256=(
                os.environ.get("HL_EXPECTED_SHA256")
                if expected_present == "1"
                else None
            ),
            expected_missing=expected_present == "0",
        )
'''
if _WRITE_CALL not in _v3._SECURE_ATOMIC_WRITE:
    raise RuntimeError("canonical Harbor write publication contract changed unexpectedly")
_v3._SECURE_ATOMIC_WRITE = _v3._SECURE_ATOMIC_WRITE.replace(
    _WRITE_CALL,
    _WRITE_CALL_GUARDED,
    1,
)
_WRITE_RESULT = '    print("write complete")\n'
if _WRITE_RESULT not in _v3._SECURE_ATOMIC_WRITE:
    raise RuntimeError("canonical Harbor write result contract changed unexpectedly")
_v3._SECURE_ATOMIC_WRITE = _v3._SECURE_ATOMIC_WRITE.replace(
    _WRITE_RESULT,
    _WRITE_RESULT
    + '    print("directory_synced=1" if directory_synced else "directory_synced=0")\n',
    1,
)


_SECURE_SNAPSHOT = _v2._SAFE_PREAMBLE + r'''
import json
try:
    parent, name = parent_fd(os.environ["HL_FILE_PATH"], create=True)
    try:
        try:
            descriptor, metadata = open_regular(parent, name)
        except FileNotFoundError:
            snapshot = {
                "present": False,
                "payload": "",
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
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
            snapshot = {
                "present": True,
                "dev": int(metadata.st_dev),
                "ino": int(metadata.st_ino),
                "payload": base64.b64encode(payload).decode("ascii"),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
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


def _publication_metadata(
    result: _v2.ToolResult,
    *,
    identity_verified: bool,
) -> _v2.ToolResult:
    if not result.success:
        return result
    lines = result.output.splitlines()
    marker = next(
        (line for line in reversed(lines) if line.startswith("directory_synced=")),
        "",
    )
    if not marker:
        result.success = False
        result.error = "Secure publication omitted its durability result"
        result.metadata = {
            **result.metadata,
            "secure_write_protocol_error": True,
            "atomic_replace": False,
        }
        return result

    directory_synced = marker == "directory_synced=1"
    result.output = "\n".join(line for line in lines if line != marker)
    result.metadata = {
        **result.metadata,
        "atomic_replace": True,
        "target_identity_verified": identity_verified,
        "directory_fsync": directory_synced,
        "durability_warning": not directory_synced,
    }
    if not directory_synced:
        result.output += (
            "\nWarning: content was atomically published, but the parent "
            "directory could not be fsynced."
        )
    return result


class HarborFileWriteTool(_base.HarborFileWriteTool):
    '''Publish append only when its pre-read target snapshot is unchanged.'''

    def _secure_snapshot(
        self,
        path: str,
    ) -> tuple[_Snapshot | None, _v2.ToolResult | None]:
        result = self._run_secure_python(
            _SECURE_SNAPSHOT,
            env={"HL_FILE_PATH": path},
        )
        if not result.success:
            return None, result
        try:
            payload = json.loads(result.output)
            present = payload["present"]
            if not isinstance(present, bool):
                raise TypeError("present is not boolean")
            raw = base64.b64decode(payload.get("payload", ""), validate=True)
            text = raw.decode("utf-8", errors="strict")
            digest = str(payload["sha256"])
            dev = int(payload["dev"]) if present else None
            ino = int(payload["ino"]) if present else None
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            return None, _v2.ToolResult(
                success=False,
                output="",
                error=f"Secure snapshot protocol failed: {exc}",
                metadata={
                    "canonical_path_checked": True,
                    "nofollow_io": True,
                    "secure_read_protocol_error": True,
                    "atomic_replace": False,
                },
            )
        return _Snapshot(present, text, dev, ino, digest), None

    def execute(
        self,
        file_path: str,
        content: str,
        append: bool = False,
        **kwargs: Any,
    ) -> _v2.ToolResult:
        resolved, failure = self._guard_environment_path(
            file_path,
            operation="write",
            must_exist=False,
        )
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
            return _v2.ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked write: {staged_reason}",
                metadata=_v2.policy_guard_metadata("staged_dependency_script_guard"),
            )
        size_reason = _v2.deliverable_size_cap_write_reason(
            resolved,
            effective_content,
        )
        if size_reason:
            return _v2.ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked write: {size_reason}",
                metadata=_v2.policy_guard_metadata(
                    "deliverable_size_cap_write_guard",
                    path=resolved,
                    content_bytes=len(effective_content.encode("utf-8")),
                    limit_bytes=5000,
                ),
            )

        environment = {
            "HL_FILE_PATH": resolved,
            "HL_FILE_CONTENT": base64.b64encode(
                effective_content.encode("utf-8")
            ).decode("ascii"),
        }
        if snapshot is not None:
            environment["HL_EXPECTED_PRESENT"] = "1" if snapshot.present else "0"
            if snapshot.present:
                environment["HL_EXPECTED_SHA256"] = snapshot.sha256
                assert snapshot.dev is not None and snapshot.ino is not None
                environment["HL_EXPECTED_DEV"] = str(snapshot.dev)
                environment["HL_EXPECTED_INO"] = str(snapshot.ino)

        result = self._run_secure_python(
            _v3._SECURE_ATOMIC_WRITE,
            env=environment,
        )
        result = _publication_metadata(
            result,
            identity_verified=append,
        )
        if result.success:
            result.metadata["atomic_append"] = append
        return result


class HarborFileEditTool(_base.HarborFileEditTool):
    '''Retain the exact-inode edit script and expose publication durability.'''

    def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> _v2.ToolResult:
        staged_reason = _v2.staged_dependency_script_reason(
            file_path,
            new_string,
        )
        if staged_reason:
            return _v2.ToolResult(
                success=False,
                output="",
                error=f"Worker file policy blocked edit: {staged_reason}",
                metadata=_v2.policy_guard_metadata(
                    "staged_dependency_script_guard"
                ),
            )
        return _publication_metadata(
            super().execute(
                file_path,
                old_string,
                new_string,
                replace_all=replace_all,
                **kwargs,
            ),
            identity_verified=True,
        )


HarborFileReadTool = _base.HarborFileReadTool
ToolResult = _v2.ToolResult

__all__ = [
    "HarborFileReadTool",
    "HarborFileEditTool",
    "HarborFileWriteTool",
    "ToolResult",
    "_SECURE_SNAPSHOT",
]
