"""Offline attempt-local archive contracts, including the real Rust bridge."""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import litellm
import pytest

from bench.agent import HLAgent
from harness.config import HarnessConfig
from harness.tools.base import ToolDef, ToolResult
from hl.types import TrialStatus
from tests.test_worker_context_overflow import assert_protocol, history, tool_response
from tests.test_worker_tool_delivery import ResultSpy, round_response
from tests.test_worker_usage import worker_binary as worker_binary  # noqa: PLC0414


@pytest.fixture
def archive(tmp_path):
    from bench.tool_result_archive import ToolResultArchive
    return ToolResultArchive(tmp_path)


def captured(output='HEAD\n' + '界🙂' * 1500 + '\nMIDDLE\n' + '界🙂' * 1500 + '\nTAIL', **metadata):
    return {'success': True, 'output': output, 'error': '', 'duration_ms': 1, 'metadata': metadata}


def decode(result):
    assert result.success, result.error
    return json.loads(result.output)


def recover(archive, ref, limit=127):
    pages = []
    offset = 0
    while True:
        page = decode(archive.read(ref=ref, offset=offset, limit=limit))
        assert len(page['content']) <= limit
        pages.append(page['content'])
        if page['next_offset'] is None:
            break
        assert page['next_offset'] > offset
        offset = page['next_offset']
    return json.loads(''.join(pages))


def test_exact_unicode_capture_no_replay_and_reopen(archive):
    original = captured()
    saved = archive.capture('call-a', 'read', original)
    assert saved['status'] == 'unknown'  # A tool name alone proves no producer coverage.
    original['output'] = 'changed source'
    assert recover(archive, saved['ref']) == captured()
    reopened = type(archive)(archive.root, scope=archive.scope)
    assert recover(reopened, saved['ref']) == captured()
    assert decode(reopened.index())['entries'][0]['ref'] == saved['ref']


@pytest.mark.parametrize('metadata', [{'truncated': True}, {'output_truncated': True},
    {'line_truncated_count': 1}, {'has_more': True}, {'timed_out': True},
    {'cancelled': True}, {'partial_output_available': True}, {'total_lines_known': False}])
def test_upstream_partial_is_never_complete(archive, metadata):
    result = captured(**metadata)
    saved = archive.capture('partial', 'read', result)
    assert saved['status'] == 'partial'
    assert recover(archive, saved['ref']) == result
    assert decode(archive.read(ref=saved['ref']))['capture_status'] == 'partial'


def test_error_output_and_full_envelope(archive):
    result = {'success': False, 'output': 'partial stderr\n界', 'error': 'command failed',
                  'duration_ms': 12, 'metadata': {'exit_code': 2}}
    saved = archive.capture('failure', 'bash', result)
    assert recover(archive, saved['ref']) == result


@pytest.mark.parametrize('ref', ['../index.jsonl', '/etc/passwd', 'other:1',
                                  'file:///root/trials/runs/hidden', ''])
def test_reject_arbitrary_paths_cross_attempt_and_legacy_refs(archive, tmp_path, ref):
    other = type(archive)(tmp_path / 'other')
    cross = other.capture('same-call', 'read', captured())['ref']
    for invalid in (ref, cross):
        result = archive.read(ref=invalid)
        assert not result.success
        assert decode_error(result)['capture_status'] == 'unknown'


def decode_error(result):
    return json.loads(result.output)


@pytest.mark.parametrize('mutation', ['delete', 'tamper', 'symlink'])
def test_missing_tampered_symlink_data_stays_unknown(archive, tmp_path, mutation):
    saved = archive.capture('x', 'read', captured())
    path = archive.directory / '1.json'
    if mutation == 'delete':
        path.unlink()
    elif mutation == 'tamper':
        path.write_text('secret forged output')
    else:
        path.unlink()
        hidden = tmp_path / 'hidden'
        hidden.write_text('host secret')
        path.symlink_to(hidden)
    result = archive.read(ref=saved['ref'])
    assert not result.success
    assert decode_error(result)['capture_status'] == 'unknown'
    assert 'secret' not in result.output
    assert decode(archive.index())['entries'][0]['status'] == 'unknown'


def test_quota_and_storage_failure_keep_unknown_index(archive, monkeypatch):
    monkeypatch.setattr(archive, 'MAX_BYTES', 1)
    saved = archive.capture('quota', 'bash', captured())
    assert saved['status'] == 'unknown'
    assert saved['reason'] == 'storage_quota'
    assert saved['ref'] is None
    assert decode(archive.index())['entries'][0]['status'] == 'unknown'
    monkeypatch.setattr(archive, 'MAX_BYTES', 1024 * 1024)
    def fail(*args, **kwargs):
        raise OSError('host secret path')
    monkeypatch.setattr(archive, '_write', fail)
    saved = archive.capture('io', 'bash', captured())
    assert saved['status'] == 'unknown'
    assert saved['reason'] == 'archive_unavailable'
    assert 'secret' not in str(saved)


@pytest.mark.parametrize('kwargs', [{'offset': -1}, {'offset': True}, {'limit': 0},
                                   {'limit': 1000000}, {'limit': '2'}])
def test_bounded_read_and_index_validate_windows(archive, kwargs):
    saved = archive.capture('x', 'read', captured())
    assert not archive.read(ref=saved['ref'], **kwargs).success
    assert not archive.index(**kwargs).success


def test_index_pagination_rediscovery_and_storage_cap(archive, monkeypatch):
    for number in range(9):
        archive.capture(str(number), 'read', captured('tiny'))
    offset, refs = 0, []
    while True:
        page = decode(archive.index(offset=offset, limit=2))
        assert len(page['entries']) <= 2
        refs.extend(entry['ref'] for entry in page['entries'])
        if page['next_offset'] is None:
            break
        offset = page['next_offset']
    assert len(set(refs)) == 9
    assert all(recover(archive, ref)['output'] == 'tiny' for ref in refs)
    monkeypatch.setattr(archive, 'MAX_ENTRIES', 9)
    assert archive.capture('overflow', 'read', captured())['status'] == 'unknown'
    assert len(decode(archive.index(limit=16))['entries']) == 9


@pytest.mark.parametrize('output,metadata', [('/host/trials/runs/hidden', {}),
    ('/logs/verifier/reward.txt', {}), ('Bearer abcdefghijklmnop', {}),
    ('ordinary', {'api_key': 'private'}), ('known-private-value', {}),
    ('password=task-private', {})])
def test_sensitive_capture_is_withheld(archive, output, metadata, monkeypatch):
    monkeypatch.setenv('WORKER_API_KEY', 'known-private-value')
    saved = archive.capture('secret', 'bash', captured(output, **metadata))
    assert saved['status'] == 'unknown'
    assert saved['reason'] == 'sensitive_capture'
    assert not list(archive.directory.glob('*.json'))
    assert output not in json.dumps(decode(archive.index()))


class StableSpy(ResultSpy):
    def _timed_execute(self, **kwargs):
        return self.execute(**kwargs)


def worker_setup(worker_binary, monkeypatch, tmp_path, enabled):
    monkeypatch.setenv('HL_WORKER_RUST_BIN', str(worker_binary))
    monkeypatch.setattr(ToolDef, '_timed_execute', lambda self, **kwargs: self.execute(**kwargs))
    worker = HLAgent(config=HarnessConfig.model_validate({'version': '0.1.0', 'tool_result_archive_enabled': enabled}))
    dispatches = []
    results = {f'c{i}': ToolResult(True, f'HEAD{i}\n' + '界🙂' * 3000 + f'\nMIDDLE{i}\nTAIL{i}',
                                  duration_ms=1) for i in range(5)}
    worker.tool_registry.register(StableSpy('irreversible', results, dispatches))
    original = worker._rust_worker_request
    initial = history(7)
    def request(instruction, context):
        payload = original(instruction, context)
        payload['initial_messages'] = deepcopy(initial)
        payload['thresholds']['compaction_char_threshold'] = 30000
        return payload
    monkeypatch.setattr(worker, '_rust_worker_request', request)
    # Avoid bootstrap environment actions in this deterministic fixture.
    worker.tool_registry.unregister('bash')
    return worker, dispatches, results, {'task_id': 'offline', 'evidence_path': str(tmp_path)}


def test_default_off_byte_identical_to_baseline(worker_binary, monkeypatch, tmp_path):
    worker, dispatches, results, context = worker_setup(worker_binary, monkeypatch, tmp_path, False)
    observed = []
    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            return round_response(list(results), ['irreversible'] * 5)
        return tool_response()
    def checked(**kwargs):
        try:
            return completion(**kwargs)
        except AssertionError as exc:
            raise RuntimeError(f'Insufficient Balance: offline assertion: {exc}') from exc
    monkeypatch.setattr('bench.agent.litellm.completion', checked)
    result = worker.run('Inspect visible inputs.', context)
    assert result.status == TrialStatus.UNVERIFIED, result.error_log
    assert len(observed) == 2
    payload = json.dumps({'requests': observed, 'trajectory': result.trajectory,
                          'tools': result.tool_calls}, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    assert digest == '34f651a4660632aefb47d530da62fb92fcf35d6b3a699a668dcdceac1e9cdf7c', digest
    assert not list(tmp_path.glob('tool-results*'))
    assert dispatches == [('irreversible', key) for key in results]


@pytest.mark.parametrize('projection_limit', [100, 4000])
@pytest.mark.parametrize('recovery', ['normal', 'overflow', 'transient'])
def test_archive_read_envelope_reaches_provider_after_compaction_without_replay(
        worker_binary, monkeypatch, tmp_path, recovery, projection_limit):
    worker, dispatches, results, context = worker_setup(worker_binary, monkeypatch, tmp_path, True)
    for key, value in results.items():
        value.output = f'HEAD:{key}\n' + '界🙂' * 1500 + f'\nMIDDLE:{key}\n' + '界🙂' * 1500 + f'\nTAIL:{key}'
    worker.ephemeral_tool_output_max_chars = projection_limit
    worker._initialize_run_state('Inspect visible inputs.', context)
    assert {'tool_result_index', 'tool_result_read'} <= set(worker.tool_registry.list_tools())
    observed, ref = [], None
    def completion(**kwargs):
        nonlocal ref
        observed.append(deepcopy(kwargs))
        assert_protocol(kwargs['messages'])
        step = len(observed)
        names = {schema['function']['name'] for schema in kwargs['tools']}
        assert {'tool_result_index', 'tool_result_read'} <= names
        if step == 1:
            return round_response(list(results), ['irreversible'] * 5)
        if step == 2:
            fresh = {message['tool_call_id']: message['content']
                     for message in kwargs['messages'] if message['role'] == 'tool'}
            for key in results:
                assert fresh[key].endswith(f'TAIL:{key}')
                assert f'MIDDLE:{key}' not in fresh[key]
                assert 'tool_result_read' in fresh[key]
        if step == 2 and recovery == 'overflow':
            raise litellm.ContextWindowExceededError('offline overflow', 'mock', 'openai')
        if step == 2 and recovery == 'transient':
            raise TimeoutError('offline transient')
        tools = {message['tool_call_id']: message['content'] for message in kwargs['messages']
                 if message['role'] == 'tool'}
        if 'page' in tools:
            envelope = json.loads(tools['page'])
            assert envelope['ref'] == ref
            assert envelope['capture_status'] == 'unknown'  # Custom irreversible producer.
            assert 'MIDDLE:c0' in envelope['content']
            assert envelope['next_offset'] == 3090
            assert envelope['truncated'] is True
            assert envelope['total_chars'] > 6000
            return tool_response()
        if 'index' in tools:
            entries = json.loads(tools['index'])['entries']
            ref = next(entry['ref'] for entry in entries if entry['call_id'] == 'c0')
            call = SimpleNamespace(id='page', function=SimpleNamespace(name='tool_result_read',
                       arguments=json.dumps({'ref': ref, 'offset': 2990, 'limit': 100})))
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
                content='', reasoning_content=None, tool_calls=[call]))])
        call = SimpleNamespace(id='index', function=SimpleNamespace(name='tool_result_index',
                                    arguments='{"offset":0,"limit":16}'))
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
            content='', reasoning_content=None, tool_calls=[call]))])
    def checked(**kwargs):
        try:
            return completion(**kwargs)
        except AssertionError as exc:
            raise RuntimeError(f'Insufficient Balance: offline assertion: {exc}') from exc
    monkeypatch.setattr('bench.agent.litellm.completion', checked)
    result = worker.run('Inspect visible inputs.', context)
    assert result.error_log == []
    assert dispatches == [('irreversible', key) for key in results]
    assert any(event['type'] == 'context_compaction' for event in result.trajectory)
    assert len(observed) == (4 if recovery == 'normal' else 5)
    if recovery != 'overflow':
        assert any('tool_result_read' in message['content'] for message in observed[1]['messages']
                   if message['role'] == 'tool')


def test_changed_source_file_and_environment_actions_are_not_replayed(archive, tmp_path):
    source = tmp_path / 'visible.txt'
    source.write_text('HEAD\n' + '中' * 3000 + '\nMIDDLE\nTAIL')
    original = source.read_text()
    saved = archive.capture('read-once', 'read', captured(original))
    source.write_text('new output that rerunning would return')
    assert recover(archive, saved['ref'])['output'] == original
    assert source.read_text() == 'new output that rerunning would return'


@pytest.mark.parametrize('journal', ['broken-json', 'cross-scope', 'symlink'])
def test_corrupt_recovery_journal_never_fabricates_history(archive, tmp_path, journal):
    archive.capture('x', 'read', captured())
    path = archive.directory / 'index.jsonl'
    if journal == 'symlink':
        path.unlink()
        path.symlink_to(tmp_path / 'hidden-host-path')
    elif journal == 'cross-scope':
        entry = json.loads(path.read_text())
        entry['ref'] = '0' * 32 + ':1'
        path.write_text(json.dumps(entry))
    else:
        path.write_text('{broken')
    reopened = type(archive)(archive.root, scope=archive.scope)
    page = decode(reopened.index())
    assert page['capture_status'] == 'unknown'
    assert page['entries'] == []
    assert page['history_status'] == 'unknown'


def test_yaml_flag_strict_default_and_harbor_evidence_path(tmp_path):
    from bench.harbor_adapter import HLWorkerHarborAgent
    from harness.tools.registry import ToolRegistry
    assert HarnessConfig.create_default().tool_result_archive_enabled is False
    config = tmp_path / 'config.yaml'
    config.write_text('version: 0.1.0\ntool_result_archive_enabled: true\n')
    adapter = HLWorkerHarborAgent(logs_dir=tmp_path, harness_config=str(config))
    worker = adapter._build_agent(ToolRegistry())
    assert worker.config.tool_result_archive_enabled is True
    assert worker.tool_result_evidence_path == tmp_path
    with pytest.raises(ValueError):
        HarnessConfig.model_validate({'version': '0.1.0', 'tool_result_archive_enabled': 'yes'})


def test_cancellation_missing_evidence_and_new_run_scopes(tmp_path):
    from threading import Event
    worker = HLAgent(config=HarnessConfig.model_validate(
        {'version': '0.1.0', 'tool_result_archive_enabled': True}))
    context = {'task_id': 'same-task', 'evidence_path': str(tmp_path)}
    worker._initialize_run_state('visible instruction', context)
    schemas = worker.tool_registry.get_schemas()
    original_schemas = HLAgent().tool_registry.get_schemas()
    assert [s for s in schemas if not s['function']['name'].startswith('tool_result_')] == original_schemas
    # No output has completed: the archive cannot fabricate an in-flight result.
    assert decode(worker.tool_registry.execute('tool_result_index'))['history_status'] == 'unknown'
    worker._run_cancellation = Event()
    worker._run_cancellation.set()
    saved = worker._archive_bridge_result({'id': 'late', 'tool': 'read'}, captured('late output'))
    assert saved['status'] == 'partial'
    assert recover(worker._tool_result_archive, saved['ref']) == captured('late output')
    scope = worker._tool_result_archive.scope
    worker._initialize_run_state('visible instruction', context)
    assert worker._tool_result_archive.scope != scope
    assert not worker.tool_registry.execute('tool_result_read', ref=saved['ref']).success
    worker._initialize_run_state('visible instruction', {'task_id': 'missing-evidence'})
    saved = worker._archive_bridge_result({'id': 'missing', 'tool': 'read'}, captured('output'))
    assert saved['ref'] is None and saved['status'] == 'unknown'
    worker.config.tool_result_archive_enabled = False
    worker._initialize_run_state('visible instruction', context)
    assert worker.tool_registry.get_schemas() == original_schemas


def test_configured_credential_env_with_arbitrary_name_is_not_archived(tmp_path, monkeypatch):
    from bench.tool_result_archive import ToolResultArchive
    monkeypatch.setenv('CUSTOM_AUTH_VALUE', 'sensitive-provider-value')
    archive = ToolResultArchive(tmp_path, secret_env_names=('CUSTOM_AUTH_VALUE',))
    saved = archive.capture('read', 'read', captured('sensitive-provider-value'))
    assert saved['reason'] == 'sensitive_capture'
    assert not list(archive.directory.glob('*.json'))


def test_paginated_rediscovery_across_repeated_compactions(worker_binary, monkeypatch, tmp_path):
    worker, dispatches, results, context = worker_setup(worker_binary, monkeypatch, tmp_path, True)
    for number in range(5, 25):
        results[f'c{number}'] = ToolResult(True, 'large observation ' * 400, duration_ms=1)
    observed, target = [], None
    offset = 0
    def completion(**kwargs):
        nonlocal target, offset
        observed.append(deepcopy(kwargs))
        assert_protocol(kwargs['messages'])
        step = len(observed)
        if step <= 5:
            keys = [f'c{number}' for number in range((step - 1) * 5, step * 5)]
            return round_response(keys, ['irreversible'] * 5)
        tools = {message['tool_call_id']: message['content'] for message in kwargs['messages']
                 if message['role'] == 'tool'}
        if 'read-rediscovered' in tools:
            assert json.loads(tools['read-rediscovered'])['content'].startswith('{')
            return tool_response()
        previous = tools.get(f'index-{offset}')
        if previous:
            envelope = json.loads(previous)
            target = next((entry['ref'] for entry in envelope['entries']
                           if entry['call_id'] == 'c4'), None)
            if target is None:
                offset = envelope['next_offset']
                assert offset is not None
        if target:
            call = SimpleNamespace(id='read-rediscovered', function=SimpleNamespace(
                name='tool_result_read', arguments=json.dumps({'ref': target, 'limit': 1024})))
        else:
            call = SimpleNamespace(id=f'index-{offset}', function=SimpleNamespace(
                name='tool_result_index', arguments=json.dumps({'offset': offset, 'limit': 2})))
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
            content='', reasoning_content=None, tool_calls=[call]))])
    def checked(**kwargs):
        try:
            return completion(**kwargs)
        except AssertionError as exc:
            raise RuntimeError(f'Insufficient Balance: offline assertion: {exc}') from exc
    monkeypatch.setattr('bench.agent.litellm.completion', checked)
    result = worker.run('Inspect visible inputs.', context)
    assert result.error_log == []
    assert len(dispatches) == 25
    assert offset == 4
    assert sum(event['type'] == 'context_compaction' for event in result.trajectory) >= 2
    assert any('tool_result_index' in message['content'] for request in observed
               for message in request['messages'] if message.get('name') == 'worker_context_state')
    assert not any(message.get('tool_call_id') == 'c4' for message in observed[-1]['messages'])


def test_rust_local_rejection_capture_has_no_environment_dispatch(worker_binary, monkeypatch, tmp_path):
    worker, dispatches, _, context = worker_setup(worker_binary, monkeypatch, tmp_path, True)
    observed = []
    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            call = SimpleNamespace(id='malformed', function=SimpleNamespace(
                name='irreversible', arguments='{broken json'))
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
                content='', reasoning_content=None, tool_calls=[call]))])
        return tool_response()
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', context)
    assert not dispatches
    entry = decode(worker._tool_result_archive.index())['entries'][0]
    assert entry['call_id'] == 'malformed'
    captured_result = recover(worker._tool_result_archive, entry['ref'])
    assert captured_result['success'] is False
    assert 'Malformed JSON' in captured_result['error']
    assert result.tool_calls[0]['metadata']['tool_result_archive']['ref'] == entry['ref']


def test_storage_exact_byte_boundary_and_empty_pages(archive, monkeypatch):
    result = captured('界🙂')
    size = len(json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode())
    monkeypatch.setattr(archive, 'MAX_BYTES', size)
    saved = archive.capture('boundary', 'read', result)
    assert saved['status'] == 'unknown'
    assert archive.bytes_used == size
    assert archive.capture('over', 'read', result)['reason'] == 'storage_quota'
    page = decode(archive.read(ref=saved['ref'], offset=100000))
    assert page['content'] == '' and page['next_offset'] is None
    page = decode(archive.index(offset=100000))
    assert page['entries'] == [] and page['next_offset'] is None


def test_index_quota_failure_and_invalid_roots_remain_unknown(archive, tmp_path, monkeypatch):
    import bench.tool_result_archive as module
    monkeypatch.setattr(module, 'MAX_INDEX_BYTES', 1023)
    assert archive.capture('full-index', 'read', captured())['reason'] == 'index_quota'
    assert not list(archive.directory.glob('*.json'))
    monkeypatch.setattr(module, 'MAX_INDEX_BYTES', 1024 * 1024)
    original = archive._open
    def fail_journal(name, flags):
        if name == 'index.jsonl':
            raise OSError('private host path')
        return original(name, flags)
    monkeypatch.setattr(archive, '_open', fail_journal)
    saved = archive.capture('broken-journal', 'read', captured())
    assert saved['status'] == 'unknown' and saved['ref'] is None
    assert decode(archive.index())['entries'][0]['status'] == 'unknown'
    root_file = tmp_path / 'file-root'
    root_file.write_text('ordinary host file')
    for root in (None, root_file, {}):
        unavailable = module.ToolResultArchive(root)
        assert unavailable.capture('x', 'read', captured())['status'] == 'unknown'
    for scope in ('../../host', '/root/host', 'other:attempt'):
        with pytest.raises(ValueError):
            module.ToolResultArchive(tmp_path, scope=scope)


def test_unknown_upstream_metadata_keeps_exact_capture_without_complete_claim(archive):
    result = captured()
    result['metadata'] = None
    saved = archive.capture('legacy-metadata', 'read', result)
    assert saved['status'] == 'unknown'
    assert recover(archive, saved['ref']) == result


def test_failed_archive_notice_has_no_fake_ref_at_provider(worker_binary, monkeypatch, tmp_path):
    from bench.tool_result_archive import ToolResultArchive
    worker, dispatches, results, context = worker_setup(worker_binary, monkeypatch, tmp_path, True)
    def fail(*args, **kwargs):
        raise OSError('private host path')
    monkeypatch.setattr(ToolResultArchive, '_write', fail)
    observed = []
    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            return round_response(list(results), ['irreversible'] * 5)
        return tool_response()
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', context)
    assert result.status == TrialStatus.UNVERIFIED
    tools = [message for message in observed[1]['messages']
             if message['role'] == 'tool' and message['tool_call_id'] in results]
    for message in tools:
        assert 'Capture unavailable/unknown' in message['content']
        assert 'ref=' not in message['content']
        assert 'private host path' not in message['content']
    assert len(dispatches) == 5
    assert result.tool_calls[0]['metadata']['tool_result_archive']['status'] == 'unknown'


class BootstrapSpy(ToolDef):
    name = 'bash'
    description = 'Offline bootstrap fixture'

    def __init__(self):
        self.commands = []

    def get_schema(self):
        from harness.tools.base import ToolSchema
        return ToolSchema(parameters={'type': 'object'}, description=self.description)

    def execute(self, **kwargs):
        self.commands.append(kwargs['command'])
        return ToolResult(True, 'PWD: /app\nTop-level files: visible.txt\nLikely entrypoints: ./app.py',
                          duration_ms=1)


@pytest.mark.parametrize('enabled', [False, True])
def test_bootstrap_prompts_default_off_baseline_and_enabled_capture(
        worker_binary, monkeypatch, tmp_path, enabled):
    monkeypatch.setenv('HL_WORKER_RUST_BIN', str(worker_binary))
    monkeypatch.setattr(ToolDef, '_timed_execute', lambda self, **kwargs: self.execute(**kwargs))
    worker = HLAgent(config=HarnessConfig.model_validate(
        {'version': '0.1.0', 'tool_result_archive_enabled': enabled}))
    scan = BootstrapSpy()
    worker.tool_registry.register(scan)
    observed = []
    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        return tool_response()
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', {'task_id': 'offline', 'workspace': '/app',
                                                  'evidence_path': str(tmp_path)})
    assert result.status == TrialStatus.UNVERIFIED, result.error_log
    assert len(observed) == 1
    assert len(scan.commands) == 1
    assert 'find . -maxdepth 2' in scan.commands[0]
    assert 'PWD: /app' in observed[0]['messages'][1]['content']
    if enabled:
        entry = decode(worker._tool_result_archive.index())['entries'][0]
        assert entry['call_id'] == 'entrypoint-scan'
        assert recover(worker._tool_result_archive, entry['ref'])['output'].startswith('PWD: /app')
    else:
        payload = json.dumps({'requests': observed, 'trajectory': result.trajectory,
                              'tools': result.tool_calls}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        assert digest == '78bf8d58b80ba29160c17bdaed4aef405b2d9e70bb86e9e62f7809ddba4aad93', digest
        assert not (tmp_path / 'tool-results').exists()


def test_discovery_identifier_truncation_is_explicit_and_capture_remains_exact(archive):
    original = captured('known original')
    saved = archive.capture('界' * 100, 'tool' * 30, original)
    entry = decode(archive.index())['entries'][0]
    assert entry['call_id_truncated'] and entry['tool_truncated']
    assert len(entry['call_id']) == len(entry['tool']) == 32
    assert recover(archive, saved['ref']) == original


def test_configured_credential_as_nested_key_is_withheld(tmp_path, monkeypatch):
    from bench.tool_result_archive import ToolResultArchive
    credential = 'fixture-opaque-abc123'
    monkeypatch.setenv('REVIEW152_AUTH_VALUE', credential)
    archive = ToolResultArchive(tmp_path, secret_env_names=('REVIEW152_AUTH_VALUE',))
    saved = archive.capture('key', 'read', captured('safe', nested=[{credential: 'label'}]))
    assert saved == {'ref': None, 'status': 'unknown', 'reason': 'sensitive_capture'}
    assert not list(archive.directory.glob('*.json'))
    assert credential not in archive.index().output
    assert credential not in archive.read(ref=f'{archive.scope}:1').output


def test_configured_credential_as_key_is_checked_at_read_time(tmp_path, monkeypatch):
    from bench.tool_result_archive import ToolResultArchive
    credential = 'fixture-opaque-abc123'
    archive = ToolResultArchive(tmp_path)
    saved = archive.capture('key', 'read', captured('safe', nested={credential: 'label'}))
    monkeypatch.setenv('REVIEW152_AUTH_VALUE', credential)
    reopened = ToolResultArchive(tmp_path, scope=archive.scope,
                                 secret_env_names=('REVIEW152_AUTH_VALUE',))
    result = reopened.read(ref=saved['ref'])
    assert not result.success
    assert credential not in result.output
    index = decode(reopened.index())
    assert index['entries'][0]['status'] == 'unknown'
    assert index['entries'][0]['ref'] is None
    assert credential not in json.dumps(index)


@pytest.mark.parametrize('component', ['root', 'tool-results', 'ancestor'])
def test_archive_rejects_symlinked_directory_chain(tmp_path, component):
    from bench.tool_result_archive import ToolResultArchive
    outside = tmp_path / 'outside'
    outside.mkdir()
    evidence = tmp_path / 'evidence'
    if component == 'root':
        evidence.symlink_to(outside, target_is_directory=True)
    elif component == 'tool-results':
        evidence.mkdir()
        (evidence / 'tool-results').symlink_to(outside, target_is_directory=True)
    else:
        (tmp_path / 'ancestor').symlink_to(outside, target_is_directory=True)
        evidence = tmp_path / 'ancestor' / 'evidence'
    archive = ToolResultArchive(evidence)
    assert archive.capture('safe', 'read', captured('safe'))['status'] == 'unknown'
    assert not archive.available
    assert list(outside.iterdir()) == []
    assert decode(archive.index())['capture_status'] == 'unknown'


@pytest.mark.parametrize('during_init', [False, True])
def test_archive_ancestor_replacement_fails_closed(tmp_path, monkeypatch, during_init):
    from bench.tool_result_archive import ToolResultArchive
    evidence = tmp_path / 'evidence'
    outside = tmp_path / 'outside'
    outside.mkdir()
    displaced = evidence / 'original-tool-results'
    def replace():
        (evidence / 'tool-results').rename(displaced)
        (evidence / 'tool-results').symlink_to(outside, target_is_directory=True)
    original_open = os.open
    if during_init:
        def swap_scope(path, flags, *args, **kwargs):
            fd = original_open(path, flags, *args, **kwargs)
            if flags & os.O_DIRECTORY and len(str(path)) == 32:
                replace()
            return fd
        monkeypatch.setattr(os, 'open', swap_scope)
    archive = ToolResultArchive(evidence)
    if not during_init:
        replace()
    saved = archive.capture('safe', 'read', captured('safe'))
    assert saved['status'] == 'unknown'
    assert list(outside.iterdir()) == []
    assert not list(displaced.rglob('*.json*'))


def test_real_shell_upstream_omission_is_not_complete(archive):
    from harness.tools.shell import ShellTool
    result = ShellTool(max_output_chars=128).execute("printf '%1000s' ''")
    assert result.success, result.error
    assert len(result.output) == 128
    assert 'output bytes omitted' in result.output
    assert result.metadata['truncated'] is False
    original = vars(result)
    saved = archive.capture('bounded', 'bash', original)
    assert saved['status'] == 'partial'
    assert recover(archive, saved['ref']) == original
    assert decode(archive.read(ref=saved['ref']))['capture_status'] == 'partial'


def test_bounded_output_without_coverage_is_unknown(archive):
    original = captured('small boundary output', output_bounded=True, truncated=False)
    saved = archive.capture('bounded', 'bash', original)
    assert saved['status'] == 'unknown'
    assert recover(archive, saved['ref']) == original


@pytest.mark.parametrize('metadata', [None, ['legacy'], 'legacy', 7])
def test_non_mapping_metadata_reaches_next_provider_request(
        worker_binary, monkeypatch, tmp_path, metadata):
    worker, dispatches, results, context = worker_setup(worker_binary, monkeypatch, tmp_path, True)
    results['c0'] = ToolResult(True, 'ORIGINAL RESULT', metadata=metadata, duration_ms=1)
    observed = []
    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            return round_response(['c0'], ['irreversible'])
        return tool_response()
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', context)
    assert result.status == TrialStatus.UNVERIFIED, result.error_log
    assert len(observed) == 2
    tool = result.tool_calls[0]
    assert tool['success'] and tool['output'] == 'ORIGINAL RESULT'
    saved = tool['metadata']['tool_result_archive']
    assert saved['status'] == 'unknown'
    assert recover(worker._tool_result_archive, saved['ref'])['metadata'] == metadata
    delivered = next(message['content'] for message in observed[1]['messages']
                     if message.get('tool_call_id') == 'c0')
    assert delivered.startswith('ORIGINAL RESULT')
    assert f"ref={saved['ref']}, capture_status=unknown" in delivered
    assert dispatches == [('irreversible', 'c0')]


def test_delivered_archive_ref_survives_later_pruning(worker_binary, monkeypatch, tmp_path):
    worker, dispatches, results, context = worker_setup(worker_binary, monkeypatch, tmp_path, True)
    worker.ephemeral_tool_output_max_chars = 1000
    worker.ephemeral_tool_output_keep_recent = 1
    request = worker._rust_worker_request
    def unseeded(instruction, context):
        payload = request(instruction, context)
        payload['initial_messages'] = history(0)
        payload['thresholds']['compaction_char_threshold'] = 1000000
        return payload
    monkeypatch.setattr(worker, '_rust_worker_request', unseeded)
    observed = []
    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) <= 2:
            return round_response([f'c{len(observed) - 1}'], ['irreversible'])
        return tool_response()
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', context)
    assert result.status == TrialStatus.UNVERIFIED, result.error_log
    assert len(observed) == 3
    saved = result.tool_calls[0]['metadata']['tool_result_archive']
    first = next(message['content'] for message in observed[1]['messages']
                 if message.get('tool_call_id') == 'c0')
    pruned = next(message['content'] for message in observed[2]['messages']
                  if message.get('tool_call_id') == 'c0')
    assert not first.startswith('[pruned ')
    assert pruned.startswith('[pruned ')
    assert f"ref={saved['ref']}, capture_status={saved['status']}" in pruned
    assert 'Capture unavailable/unknown' not in pruned
    assert recover(worker._tool_result_archive, saved['ref'])['output'] == results['c0'].output
    assert dispatches == [('irreversible', 'c0'), ('irreversible', 'c1')]


@pytest.mark.parametrize('case,expected', [('long-line', 'unknown'), ('decode-error', 'partial')])
def test_real_grep_coverage_capture_index_and_read(archive, tmp_path, case, expected):
    from harness.tools.search import GrepTool
    source = tmp_path / 'visible'
    source.mkdir()
    (source / 'match.txt').write_text('needle:' + 'x' * 7000 + 'UNIQUE_TAIL\n')
    if case == 'decode-error':
        (source / 'invalid.txt').write_bytes(b'\xff')
    tool = GrepTool()
    result = tool.execute(pattern='needle', path=str(source))
    assert 'UNIQUE_TAIL' not in result.output
    if case == 'long-line':
        assert result.success and result.metadata['truncated'] is False
        assert result.metadata['host_output_bounded'] is True
    else:
        assert not result.success
        assert result.metadata['partial_results_available'] is True
        assert result.metadata['read_error_count'] == 1
        assert result.metadata['search_failed'] is True
    saved = archive.capture('grep', tool.name, vars(result), producer=tool)
    assert saved['status'] == expected
    assert decode(archive.index())['entries'][0]['status'] == expected
    assert decode(archive.read(ref=saved['ref']))['capture_status'] == expected
    assert recover(archive, saved['ref']) == vars(result)


@pytest.mark.parametrize('field', ['output', 'error'])
@pytest.mark.parametrize('assignment', ['"api_key": "fixture-json-credential-152"',
    "'password': 'fixture-json-password-152'", '"access_token": "fixture-json-token-152"',
    '"authorization": "fixture-json-auth-152"', '"client_secret": "fixture-json-secret-152"'])
def test_quoted_credentials_withheld_at_capture_and_reopened_read(
        archive, monkeypatch, field, assignment):
    import bench.tool_result_archive as module
    original = captured('safe')
    original[field] = '{' + assignment + '}'
    saved = archive.capture('quoted', 'read', original)
    assert saved == {'ref': None, 'status': 'unknown', 'reason': 'sensitive_capture'}
    assert not list(archive.directory.glob('*.json'))
    assert assignment not in archive.index().output
    # Simulate a historical capture made before credential recognition improved.
    with monkeypatch.context() as patch:
        patch.setattr(module, '_sensitive', lambda *args: False)
        historical = archive.capture('historical', 'read', original)
    reopened = type(archive)(archive.root, scope=archive.scope)
    read = reopened.read(ref=historical['ref'])
    assert not read.success
    assert decode_error(read)['capture_status'] == 'unknown'
    assert assignment not in read.output
    entry = decode(reopened.index())['entries'][1]
    assert entry['status'] == 'unknown' and entry['ref'] is None


@pytest.mark.parametrize('value', [42, 0, True, False, None])
@pytest.mark.parametrize('key', ['token_count', 'api_key', 'password'])
def test_non_string_credential_key_values_are_safe(archive, key, value):
    original = captured('safe', **{key: value})
    saved = archive.capture('metric', 'custom', original)
    assert saved['ref'] is not None and saved['reason'] == ''
    assert recover(archive, saved['ref']) == original


@pytest.mark.parametrize('name,expected', [('read', 'complete'), ('write', 'complete'),
    ('edit', 'complete'), ('todo_read', 'complete'), ('todo_write', 'complete'),
    ('bash', 'unknown'), ('grep', 'unknown'), ('glob', 'unknown'),
    ('goal_read', 'unknown'), ('verify', 'unknown'), ('done', 'unknown')])
def test_all_builtin_coverage_end_to_end(tmp_path, name, expected):
    from harness.tools import (
        FileEditTool,
        FileReadTool,
        FileWriteTool,
        GlobTool,
        GoalReadTool,
        GrepTool,
        ShellTool,
        TodoReadTool,
        TodoWriteTool,
        VerifyTool,
    )
    from harness.tools.done import DoneTool
    path = tmp_path / 'visible.txt'
    path.write_text('needle\n')
    tools = [FileReadTool(), FileWriteTool(), FileEditTool(), TodoReadTool(),
             TodoWriteTool(), ShellTool(), GrepTool(), GlobTool(), GoalReadTool(),
             VerifyTool(), DoneTool()]
    args = {'read': {'file_path': str(path)},
        'write': {'file_path': str(path), 'content': 'needle\n'},
        'edit': {'file_path': str(path), 'old_string': 'needle', 'new_string': 'changed'},
        'todo_write': {'items': [{'content': 'inspect', 'status': 'pending'}]},
        'bash': {'command': "printf 'needle'"},
        'grep': {'pattern': 'needle', 'path': str(path)},
        'glob': {'pattern': '*.txt', 'path': str(tmp_path)},
        'verify': {'command': "printf 'needle'"}}
    worker = HLAgent(config=HarnessConfig.model_validate(
        {'version': '0.1.0', 'tool_result_archive_enabled': True}))
    worker._initialize_run_state('inspect', {'evidence_path': str(tmp_path)})
    for tool in tools:
        worker.tool_registry.register(tool)
    result = worker._execute_bridge_tool({'id': name, 'tool': name, 'args': args.get(name, {})})
    assert result['success'], result['error']
    saved = result['metadata'].pop('tool_result_archive')
    assert saved['status'] == expected
    archive = worker._tool_result_archive
    assert decode(archive.index())['entries'][0]['status'] == expected
    assert decode(archive.read(ref=saved['ref']))['capture_status'] == expected
    assert recover(archive, saved['ref']) == result


@pytest.mark.parametrize('name', ['read', 'write', 'edit', 'todo_read', 'todo_write', 'unknown'])
def test_registered_custom_tools_cannot_claim_complete(tmp_path, name):
    worker = HLAgent(config=HarnessConfig.model_validate(
        {'version': '0.1.0', 'tool_result_archive_enabled': True}))
    worker._initialize_run_state('inspect', {'evidence_path': str(tmp_path)})
    worker.tool_registry.register(StableSpy(name, {'x': ToolResult(True, 'safe')}, []))
    result = worker._execute_bridge_tool({'id': 'x', 'tool': name, 'args': {'key': 'x'}})
    assert result['success'], result['error']
    assert result['metadata']['tool_result_archive']['status'] == 'unknown'


@pytest.mark.parametrize('metadata,expected', [
    ({'partial_results_available': True}, 'partial'), ({'read_error_count': 1}, 'partial'),
    ({'search_failed': True}, 'partial'), ({'omitted_count': 1}, 'partial'),
    ({'diagnostics_omitted_count': 1}, 'partial'), ({'input_line_limit_exceeded': True}, 'partial'),
    ({'text_decode_error': True}, 'partial'), ({'parameter_validation_failed': True}, 'partial'),
    ({'output_limit_too_small': True}, 'partial'), ({'file_too_large': True}, 'partial'),
    ({'binary_file_unsupported': True}, 'partial'), ({'cleanup_warning': True}, 'partial'),
    ({'durability_warning': True}, 'partial'), ({'publication_error': 'fixture failure'}, 'partial'),
    ({'blocked_by': 'policy'}, 'partial'), ({'semantic_failure_detected': True}, 'partial'),
    ({'publication_state': 'indeterminate'}, 'partial'), ({'host_output_bounded': True}, 'unknown'),
    ({'output_bounded': True}, 'unknown'), ({'status': 'unknown'}, 'unknown')])
@pytest.mark.parametrize('producer_name', ['FileReadTool', 'FileWriteTool', 'FileEditTool',
                                          'TodoReadTool', 'TodoWriteTool'])
def test_allowlisted_producers_deny_incomplete_signals(archive, producer_name, metadata, expected):
    from harness import tools
    tool = getattr(tools, producer_name)()
    original = captured('safe', total_lines_known=True, total_lines=1, start_line=1,
        end_line=1, lines_returned=1, has_more=False, next_offset=None,
        output_truncated=False, line_truncated_count=0, atomic_replace=True,
        publication_state='published')
    original['metadata'].update(metadata)
    saved = archive.capture('signals', tool.name, original, producer=tool)
    assert saved['status'] == expected
    assert decode(archive.index())['entries'][0]['status'] == expected
    assert decode(archive.read(ref=saved['ref']))['capture_status'] == expected
    assert recover(archive, saved['ref']) == original


@pytest.mark.parametrize('kwargs,expected', [({}, 'complete'), ({'offset': 2}, 'partial'),
    ({'limit': 1}, 'partial'), ({'max_line_bytes': 3}, 'partial'),
    ({'max_output_chars': 40}, 'partial')])
def test_real_read_requires_full_range(archive, tmp_path, kwargs, expected):
    from harness.tools.file_read import FileReadTool
    path = tmp_path / 'visible.txt'
    path.write_text('first long line\nsecond long line\nthird long line\n')
    settings = {key: value for key, value in kwargs.items() if key.startswith('max_')}
    args = {key: value for key, value in kwargs.items() if not key.startswith('max_')}
    tool = FileReadTool(**settings)
    original = vars(tool.execute(str(path), **args))
    saved = archive.capture('range', tool.name, original, producer=tool)
    assert saved['status'] == expected
    assert recover(archive, saved['ref']) == original


def test_read_without_full_range_metadata_and_subclasses_remain_unknown(archive):
    from harness.tools.file_read import FileReadTool
    from harness.tools.todo import TodoReadTool
    class CustomTodo(TodoReadTool):
        pass
    for tool in (FileReadTool(), CustomTodo()):
        saved = archive.capture('unknown', tool.name, captured('safe'), producer=tool)
        assert saved['status'] == 'unknown'


def test_failed_boundary_result_is_partial_even_with_no_metadata(archive):
    original = captured('surviving output')
    original.update(success=False, error='fixture failure', metadata=None)
    saved = archive.capture('failed', 'custom', original)
    assert saved['status'] == 'partial'
    assert recover(archive, saved['ref']) == original


def test_real_glob_omission_marker_is_partial(archive, tmp_path):
    from harness.tools.search import GlobTool
    source = tmp_path / 'visible'
    source.mkdir()
    for number in range(501):
        (source / f'{number}.txt').touch()
    tool = GlobTool()
    original = vars(tool.execute('*.txt', path=str(source)))
    assert original['success']
    assert original['metadata']['match_count'] == 501
    assert '... (1 more matches)' in original['output']
    saved = archive.capture('glob', tool.name, original, producer=tool)
    assert saved['status'] == 'partial'
    assert decode(archive.index())['entries'][0]['status'] == 'partial'
    assert decode(archive.read(ref=saved['ref']))['capture_status'] == 'partial'
    assert recover(archive, saved['ref']) == original
