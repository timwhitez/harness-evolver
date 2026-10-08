"""Opt-in, attempt-local exact captures; never reexecute a task tool."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import weakref
from pathlib import Path
from uuid import uuid4

from harness.tools.base import ToolDef, ToolResult, ToolSchema
from harness.tools.file_edit import FileEditTool
from harness.tools.file_read import FileReadTool
from harness.tools.file_write import FileWriteTool
from harness.tools.host_memory_guard import host_memory_access_reason
from harness.tools.safe_path_io import _open_parent_nofollow, file_identity
from harness.tools.todo import TodoReadTool, TodoWriteTool

# Storage bounds only: exhausting them never stops the Worker loop.
MAX_BYTES = 16 * 1024 * 1024
MAX_ENTRIES = 2048
MAX_INDEX_BYTES = 1024 * 1024
READ_LIMIT = 1024
INDEX_LIMIT = 16
_SECRET_KEY = re.compile(r'api[_-]?key|token|password|secret|credential|authorization', re.IGNORECASE)
_SECRET_TEXT = re.compile(
    r'Bearer\s+\S+|\bsk-[A-Za-z0-9_-]{16,}|\bAKIA[A-Z0-9]{16}'
    r'|(?:api[_-]?key|password|credential|secret|access[_-]?token)\s*[=:]\s*\S+',
    re.IGNORECASE)
_QUOTED_ASSIGNMENT = re.compile(r'''["']([^"'\r\n]+)["']\s*[=:]\s*["']([^"'\r\n]+)''')
_OMISSION_TEXT = re.compile(
    r'output bytes omitted|\[line truncated at '
    r'|\.\.\. \(\d+ (?:more (?:matches|results)(?: truncated)?|additional failures omitted)\)')
# Exact producer types, not registry names or subclasses: custom tools cannot
# inherit a coverage claim. See docs/architecture.md for each audited contract.
_COMPLETE_PRODUCERS = (FileReadTool, FileWriteTool, FileEditTool, TodoReadTool, TodoWriteTool)
_PARTIAL_FIELDS = (
    'truncated', 'output_truncated', 'line_truncated_count', 'has_more',
    'timed_out', 'cancelled', 'partial_output_available', 'partial_results_available',
    'omitted_count', 'diagnostics_omitted_count', 'read_error_count', 'search_failed',
    'input_line_limit_exceeded', 'text_decode_error', 'parameter_validation_failed',
    'output_limit_too_small', 'file_too_large', 'binary_file_unsupported',
    'blocked_by', 'blocked_reason', 'semantic_failure_detected',
    'publication_error', 'cleanup_warning', 'durability_warning',
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _sensitive(value, secrets=()):
    """Withhold captures rather than redact and claim they are exact."""
    if isinstance(value, dict):
        return any((isinstance(item, str) and item and _SECRET_KEY.search(str(key))) or _sensitive(key, secrets)
                   or _sensitive(item, secrets)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_sensitive(item, secrets) for item in value)
    if not isinstance(value, str):
        return False
    return bool(host_memory_access_reason(value) or _SECRET_TEXT.search(value)
                or any(_SECRET_KEY.search(key) for key, _ in _QUOTED_ASSIGNMENT.findall(value))
                or any(path in value.lower() for path in ('/logs/verifier', '/tmp/hl-verifier-cache'))
                or any(secret in value for secret in secrets if secret)
                or any(secret in value for key, secret in os.environ.items()
                       if _SECRET_KEY.search(key) and len(secret) >= 4))


def _partial(metadata):
    return (any(metadata.get(key) for key in _PARTIAL_FIELDS)
            or metadata.get('total_lines_known') is False
            or (type(metadata.get('start_line')) is int and metadata['start_line'] > 1)
            or metadata.get('status') in ('partial', 'cancelled', 'failed')
            or metadata.get('publication_state') in ('not_published', 'indeterminate'))


def _capture_status(result, partial, producer):
    metadata = result.get('metadata')
    if (partial or result.get('success') is False or result.get('error')
            or any(_OMISSION_TEXT.search(result.get(key, '')) for key in ('output', 'error'))):
        return 'partial'
    if not isinstance(metadata, dict):
        return 'unknown'
    if _partial(metadata):
        return 'partial'
    if (result.get('success') is not True or type(producer) not in _COMPLETE_PRODUCERS
            or metadata.get('status') == 'unknown'
            or metadata.get('output_bounded') or metadata.get('host_output_bounded')):
        return 'unknown'
    if type(producer) is FileReadTool:
        total = metadata.get('total_lines')
        if not (metadata.get('total_lines_known') is True and type(total) is int and total >= 0
                and metadata.get('start_line') == 1 and metadata.get('end_line') == total
                and metadata.get('lines_returned') == total and metadata.get('has_more') is False
                and metadata.get('next_offset') is None and metadata.get('output_truncated') is False
                and metadata.get('line_truncated_count') == 0):
            return 'unknown'
    if (type(producer) in (FileWriteTool, FileEditTool)
            and (metadata.get('atomic_replace') is not True
                 or metadata.get('publication_state') != 'published')):
        return 'unknown'
    return 'complete'


def _window(offset, limit, maximum):
    if type(offset) is not int or offset < 0:
        raise ValueError('offset must be a nonnegative integer')
    if type(limit) is not int or not 1 <= limit <= maximum:
        raise ValueError(f'limit must be an integer in 1..{maximum}')


def _result(envelope, error=''):
    return ToolResult(not error, _json(envelope), error=error)


def _unknown(reason):
    return _result({'capture_status': 'unknown', 'reason': reason,
                    'next_offset': None, 'truncated': False}, error=reason)


class ToolResultArchive:
    """The root is supplied by the host adapter, never by model tool arguments.

    Ref lookup uses only journalled sequence numbers in a private scope directory.
    The model cannot choose a scope, filename or host path. Recovery can reopen
    the same scope via a trusted host caller; a new run always creates a new scope.
    """

    MAX_BYTES = MAX_BYTES
    MAX_ENTRIES = MAX_ENTRIES

    def __init__(self, root: Path | str | None, *, scope: str | None = None, secret_env_names=()):
        self.secrets = tuple(os.environ.get(name, "") for name in secret_env_names if name)
        try:
            self.root = Path(root) if root is not None else None
        except TypeError:
            self.root = None
        self.scope = scope or uuid4().hex
        if re.fullmatch(r'[a-f0-9]{32}', self.scope) is None:
            raise ValueError('invalid archive scope')
        self.directory = self.root / 'tool-results' / self.scope if self.root else None
        self.entries = []
        self.bytes_used = 0
        self.index_bytes = 0
        self.available = False
        self._directory_fd = None
        self._close_directory = None
        if self.directory is None:
            return
        try:
            with _open_parent_nofollow(self.directory, create_parents=scope is None) as (parent, name, _):
                if scope is None:
                    os.mkdir(name, mode=0o700, dir_fd=parent)
                self._directory_fd = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent)
                self._close_directory = weakref.finalize(self, os.close, self._directory_fd)
            if scope is not None:
                journal = self._load('index.jsonl', MAX_INDEX_BYTES).decode('utf-8')
                for line in journal.splitlines():
                    entry = json.loads(line)
                    number = len(self.entries) + 1
                    if (number > self.MAX_ENTRIES or type(entry['sequence']) is not int
                            or entry['sequence'] != number
                            or entry['ref'] not in (None, f'{self.scope}:{number}')
                            or type(entry['bytes']) is not int or entry['bytes'] < 0
                            or entry['bytes'] > self.MAX_BYTES
                            or entry['status'] not in ('complete', 'partial', 'unknown')
                            or any(not isinstance(entry[key], str) or len(entry[key]) > 64
                                   for key in ('call_id', 'tool', 'reason'))
                            or _sensitive(entry, self.secrets)
                            or (entry['ref'] is not None and
                                re.fullmatch(r'[a-f0-9]{64}', entry['sha256']) is None)):
                        raise ValueError('invalid archive journal')
                    self.entries.append(entry)
                    self.bytes_used += entry['bytes']
                if self.bytes_used > self.MAX_BYTES:
                    raise ValueError('invalid archive size')
                self.index_bytes = len(journal.encode('utf-8'))
            self.available = True
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            self.entries = []
            if self._close_directory is not None:
                self._close_directory()
            self._directory_fd = None

    def _open(self, name, flags):
        if self._directory_fd is None:
            raise OSError('archive unavailable')
        # Reject replacement of any ancestor; file operations use the original
        # descriptor even if the path changes after this check.
        with _open_parent_nofollow(self.directory, create_parents=False) as (parent, scope, _):
            current = os.stat(scope, dir_fd=parent, follow_symlinks=False)
            if (not stat.S_ISDIR(current.st_mode)
                    or file_identity(current) != file_identity(os.fstat(self._directory_fd))):
                raise OSError('archive directory changed')
        return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                       dir_fd=self._directory_fd)

    def _write(self, name, payload):
        with os.fdopen(self._open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'wb') as handle:
            handle.write(payload)

    def _load(self, name, maximum):
        with os.fdopen(self._open(name, os.O_RDONLY | os.O_NONBLOCK), 'rb') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise ValueError('invalid archive file')
            payload = handle.read(maximum + 1)
            if len(payload) > maximum:
                raise ValueError('invalid archive file')
            return payload

    def capture(self, call_id, tool, result, *, partial=False, producer=None):
        unknown = {'ref': None, 'status': 'unknown', 'reason': 'archive_unavailable'}
        if not self.available:
            return unknown
        if len(self.entries) >= self.MAX_ENTRIES:
            return {**unknown, 'reason': 'index_quota'}
        if self.index_bytes + 1024 > MAX_INDEX_BYTES:
            return {**unknown, 'reason': 'index_quota'}
        number = len(self.entries) + 1
        entry = {'sequence': number, 'call_id': str(call_id)[:32], 'tool': str(tool)[:32],
                     'ref': None, 'status': 'unknown', 'reason': '', 'bytes': 0, 'sha256': '',
                     'call_id_truncated': len(str(call_id)) > 32, 'tool_truncated': len(str(tool)) > 32}
        try:
            if _sensitive([call_id, tool, result], self.secrets):
                # Do not persist sensitive identifiers in the discovery journal.
                entry.update(call_id='withheld', tool='withheld', reason='sensitive_capture')
            else:
                payload = _json(result).encode('utf-8')
                if self.bytes_used + len(payload) > self.MAX_BYTES:
                    entry['reason'] = 'storage_quota'
                else:
                    self._write(f'{number}.json', payload)
                    self.bytes_used += len(payload)
                    entry.update(ref=f'{self.scope}:{number}', bytes=len(payload),
                                 sha256=hashlib.sha256(payload).hexdigest(),
                                 status=_capture_status(result, partial, producer))
            line = (_json(entry) + '\n').encode('utf-8')
            if self.index_bytes + len(line) > MAX_INDEX_BYTES:
                return {**unknown, 'reason': 'index_quota'}
            with os.fdopen(self._open('index.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND), 'wb') as handle:
                handle.write(line)
            self.index_bytes += len(line)
        except (OSError, ValueError, TypeError, UnicodeError):
            entry.update(ref=None, status='unknown', reason='archive_unavailable')
            # The journal may contain an incomplete record; no further writes.
            self.available = False
        self.entries.append(entry)
        return {key: entry[key] for key in ('ref', 'status', 'reason')}

    def _text(self, entry):
        payload = self._load(f"{entry['sequence']}.json", self.MAX_BYTES)
        if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError('archive integrity failure')
        text = payload.decode('utf-8')
        if _sensitive(json.loads(text), self.secrets):
            raise ValueError('sensitive capture')
        return text

    def index(self, *, offset=0, limit=INDEX_LIMIT):
        try:
            _window(offset, limit, INDEX_LIMIT)
        except ValueError as exc:
            return _unknown(str(exc))
        entries = []
        for original in self.entries[offset:offset + limit]:
            entry = {key: original[key] for key in ('call_id', 'tool', 'ref', 'status', 'reason')}
            entry.update({key: original.get(key, False)
                          for key in ('call_id_truncated', 'tool_truncated')})
            if entry['ref'] is not None:
                try:
                    self._text(original)
                except (OSError, ValueError, UnicodeError):
                    entry.update(ref=None, status='unknown', reason='archive_unavailable')
            entries.append(entry)
        following = offset + len(entries)
        more = following < len(self.entries)
        return _result({'entries': entries, 'next_offset': following if more else None,
                            'truncated': more, 'capture_status': 'complete' if self.available else 'unknown',
                            'history_status': 'unknown', 'scope': 'current_attempt'})

    def read(self, *, ref, offset=0, limit=READ_LIMIT):
        try:
            _window(offset, limit, READ_LIMIT)
            if not isinstance(ref, str) or re.fullmatch(rf'{self.scope}:[1-9][0-9]*', ref) is None:
                return _unknown('invalid_or_cross_attempt_ref')
            entry = next((entry for entry in self.entries if entry['ref'] == ref), None)
            if entry is None:
                return _unknown('missing_capture')
            text = self._text(entry)
            content = text[offset:offset + limit]
            following = offset + len(content)
            more = following < len(text)
            return _result({'ref': ref, 'content': content, 'offset': offset,
                                'next_offset': following if more else None, 'total_chars': len(text),
                                'truncated': more, 'capture_status': entry['status'],
                                'encoding': 'json_utf8', 'offset_unit': 'unicode_characters'})
        except (OSError, ValueError, UnicodeError):
            return _unknown('archive_unavailable')


class ArchiveTool(ToolDef):
    def __init__(self, archive, *, read=False):
        self.archive = archive
        self.read = read
        self.name = 'tool_result_read' if read else 'tool_result_index'
        self.description = ('Read captured ToolResult JSON by current-attempt ref without replay; '
                            'concatenate content pages using next_offset. Capture may be partial.'
                            if read else 'Discover current-attempt captured tool results; use next_offset '
                            'to page. Missing historical captures remain unknown. No task action.')

    def get_schema(self):
        properties = {'offset': {'type': 'integer', 'minimum': 0},
                      'limit': {'type': 'integer', 'minimum': 1,
                                'maximum': READ_LIMIT if self.read else INDEX_LIMIT}}
        if self.read:
            properties['ref'] = {'type': 'string'}
        return ToolSchema(parameters={'type': 'object', 'properties': properties,
                                     'required': ['ref'] if self.read else [],
                                     'additionalProperties': False}, description=self.description)

    def execute(self, **kwargs):
        try:
            return self.archive.read(**kwargs) if self.read else self.archive.index(**kwargs)
        except TypeError:
            return _unknown('invalid_archive_arguments')
