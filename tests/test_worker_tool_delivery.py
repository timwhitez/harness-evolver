"""First delivery of a complete parallel tool round across the real Rust bridge."""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import litellm
import pytest

from bench.agent import HLAgent
from harness.tools.base import ToolDef, ToolResult, ToolSchema
from harness.tools.done import DoneTool
from harness.tools.registry import ToolRegistry
from hl.types import TrialStatus
from tests.test_worker_context_overflow import (
    assert_protocol,
    history,
    payload_bytes,
    tool_response,
)
from tests.test_worker_usage import worker_binary as worker_binary  # noqa: PLC0414 (pytest fixture)


class ResultSpy(ToolDef):
    def __init__(self, name, results, dispatches):
        self.name = name
        self.description = 'Offline scripted tool'
        self.results = results
        self.dispatches = dispatches

    def get_schema(self):
        return ToolSchema(parameters={'type': 'object'}, description=self.description)

    def execute(self, **kwargs):
        key = kwargs['key']
        self.dispatches.append((self.name, key))
        # The synthetic irreversible tool must never be replayed.
        assert sum(item[1] == key for item in self.dispatches) == 1
        return deepcopy(self.results[key])


def round_response(keys, names=None):
    names = names or ['read'] * len(keys)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
        content='', reasoning_content=None, tool_calls=[SimpleNamespace(
            id=key, function=SimpleNamespace(name=name, arguments=json.dumps({'key': key})))
            for key, name in zip(keys, names, strict=True)]))])


def result_output(key, size=5000, fill='X'):
    marker = f'<TAIL:{key}>'
    body_size = size - len(marker)
    return (fill * size)[:body_size] + marker if size else ''


def projection(result, limit=4000):
    raw = result.output if result.success else (
        f'{result.error.strip()}\n[tool output]\n{result.output.strip()}'
        if result.output.strip() else result.error.strip())
    prefix = '' if result.success else 'Error: '
    if len(raw) <= limit:
        return prefix + raw
    head = limit * 2 // 3
    tail = max(limit - head, 1)
    omitted = len(raw) - limit
    return (f'{prefix}[tool output truncated for model context: original_chars={len(raw)}, '
            f'shown_chars={limit}, omitted_chars={omitted}. Full output is preserved in '
            'trajectory artifacts when available; rerun a narrower command if exact omitted '
            f'lines are needed.]\n{raw[:head]}\n...[omitted {omitted} chars]...\n{raw[-tail:]}')


def setup_worker(worker_binary, monkeypatch, results, threshold=120000):
    monkeypatch.setenv('HL_WORKER_RUST_BIN', str(worker_binary))
    worker = HLAgent()
    worker.tool_registry = ToolRegistry()
    dispatches = []
    for name in ('read', 'verify', 'irreversible'):
        worker.tool_registry.register(ResultSpy(name, results, dispatches))
    # Keep the real done tool; only model/provider and task-tool execution are scripted.
    worker.tool_registry.register(DoneTool())
    original_request = worker._rust_worker_request
    initial = [{'role': 'system', 'content': 'Use visible task inputs only.'},
               {'role': 'user', 'content': 'Inspect visible inputs.'}]

    def request(instruction, context):
        payload = original_request(instruction, context)
        payload['initial_messages'] = deepcopy(initial)
        payload['thresholds']['compaction_char_threshold'] = threshold
        return payload

    monkeypatch.setattr(worker, '_rust_worker_request', request)
    return worker, dispatches, initial


def assert_delivered(request, keys, results):
    assert_protocol(request['messages'])
    tools = {message['tool_call_id']: message['content'] for message in request['messages']
             if message['role'] == 'tool'}
    for key in keys:
        assert tools[key] == projection(results[key])
        if results[key].output:
            assert tools[key].endswith(f'<TAIL:{key}>')


@pytest.mark.parametrize('count', [1, 4, 5, 9])
def test_fresh_read_round_delivers_every_bounded_projection(worker_binary, monkeypatch, count):
    keys = [f'read-{index}' for index in range(count)]
    results = {key: ToolResult(True, result_output(key)) for key in keys}
    original = deepcopy(results)
    worker, dispatches, initial = setup_worker(worker_binary, monkeypatch, results)
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            return round_response(keys)
        if len(observed) == 2:
            return tool_response()
        raise RuntimeError('Insufficient Balance: unexpected request')

    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', {'task_id': 'task-a'})
    assert result.status == TrialStatus.UNVERIFIED
    assert len(observed) == 2
    assert_delivered(observed[1], keys, results)
    assert payload_bytes(observed[1]) < 120000
    assert observed[1]['messages'][:2] == initial
    assert results == original
    assert dispatches == [('read', key) for key in keys]


def test_mixed_unicode_error_verify_and_irreversible_results(worker_binary, monkeypatch):
    sizes = [5000, 0, 20, 3999, 4000, 4001, 6000, 100]
    keys = [f'mixed-{index}' for index in range(len(sizes))]
    names = ['read', 'read', 'read', 'verify', 'irreversible', 'read', 'verify', 'read']
    results = {key: ToolResult(index != 0, result_output(key, size, '界🙂'),
                               error='read failed' if index == 0 else '')
               for index, (key, size) in enumerate(zip(keys, sizes, strict=True))}
    original = deepcopy(results)
    worker, dispatches, _ = setup_worker(worker_binary, monkeypatch, results)
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            return round_response(keys, names)
        if len(observed) == 2:
            return tool_response()
        raise RuntimeError('Insufficient Balance: unexpected request')

    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', {'task_id': 'task-a'})
    assert len(observed) == 2
    assert result.status == TrialStatus.UNVERIFIED
    assert_delivered(observed[1], keys, results)
    assert results == original
    assert dispatches == list(zip(names, keys, strict=True))


def test_old_rounds_prune_and_protection_expires_after_delivery(worker_binary, monkeypatch):
    rounds = [[f'{round_index}-{index}' for index in range(count)]
              for round_index, count in enumerate((5, 4, 5))]
    results = {key: ToolResult(True, result_output(key)) for keys in rounds for key in keys}
    worker, dispatches, _ = setup_worker(worker_binary, monkeypatch, results)
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) <= 3:
            return round_response(rounds[len(observed)-1])
        if len(observed) == 4:
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
                content='Continue inspecting inputs.', reasoning_content=None, tool_calls=[]))])
        if len(observed) == 5:
            return tool_response()
        raise RuntimeError('Insufficient Balance: unexpected request')

    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', {'task_id': 'task-a'})
    assert result.status == TrialStatus.UNVERIFIED
    assert len(observed) == 5
    for request, keys in zip(observed[1:4], rounds, strict=True):
        assert_delivered(request, keys, results)
    for request in observed[2:]:
        assert_protocol(request['messages'])
        old = [message for message in request['messages'] if message.get('tool_call_id') == '0-0']
        assert old[0]['content'].startswith('[pruned ')
        assert '<TAIL:0-0>' not in old[0]['content']
    newest_old = next(message for message in observed[4]['messages']
                      if message.get('tool_call_id') == '2-0')
    assert newest_old['content'].startswith('[pruned ')
    assert len(dispatches) == len(results)


@pytest.mark.parametrize('recovery', ['transient', 'overflow'])
def test_fresh_round_recovery_has_no_replay(worker_binary, monkeypatch, recovery):
    keys = [f'recovery-{index}' for index in range(5)]
    results = {key: ToolResult(True, result_output(key)) for key in keys}
    worker, dispatches, _ = setup_worker(worker_binary, monkeypatch, results)
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            return round_response(keys, ['read', 'verify', 'irreversible', 'read', 'read'])
        if len(observed) == 2:
            if recovery == 'transient':
                raise TimeoutError('temporary provider failure')
            raise litellm.ContextWindowExceededError('scripted overflow', 'mock', 'openai')
        if len(observed) == 3:
            return tool_response()
        raise RuntimeError('Insufficient Balance: unexpected request')

    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', {'task_id': 'task-a'})
    assert result.status == TrialStatus.UNVERIFIED
    assert len(observed) == 3
    assert_delivered(observed[1], keys, results)
    assert_protocol(observed[2]['messages'])
    if recovery == 'transient':
        assert_delivered(observed[2], keys, results)
    else:
        assert payload_bytes(observed[2]) < payload_bytes(observed[1])
        assert not any(message['role'] == 'tool' for message in observed[2]['messages'])
        assert any(event['type'] == 'tool_result_delivery_incomplete'
                   for event in result.trajectory)
    assert len(dispatches) == len(keys)


def test_above_advisory_target_delivers_every_fresh_projection(worker_binary, monkeypatch):
    keys = [f'capacity-{index}' for index in range(11)]
    results = {key: ToolResult(True, result_output(key, fill='界')) for key in keys}
    original = deepcopy(results)
    worker, dispatches, initial = setup_worker(worker_binary, monkeypatch, results)
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        # This provider accepts the complete group above the advisory preference.
        assert payload_bytes(kwargs) < 200000
        if len(observed) == 1:
            return round_response(keys)
        if len(observed) == 2:
            return tool_response()
        raise RuntimeError('Insufficient Balance: unexpected request')

    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', {'task_id': 'task-a'})
    assert result.status == TrialStatus.UNVERIFIED
    assert len(observed) == 2
    assert_delivered(observed[1], keys, results)
    assert 120000 < payload_bytes(observed[1]) < 200000
    assert observed[1]['messages'][:2] == initial
    assert results == original
    assert dispatches == [('read', key) for key in keys]
    assert not any(message.get('name') == 'worker_tool_delivery'
                   for message in observed[1]['messages'])
    assert not any(event['type'] in ('tool_result_delivery_incomplete', 'context_overflow_recovery')
                   for event in result.trajectory)


def test_rejected_fresh_group_is_omitted_only_after_provider_overflow(worker_binary, monkeypatch):
    keys = [f'overflow-{index}' for index in range(11)]
    names = ['read', 'verify', 'irreversible'] + ['read'] * 8
    results = {key: ToolResult(True, result_output(key, fill='界')) for key in keys}
    worker, dispatches, initial = setup_worker(worker_binary, monkeypatch, results)
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if payload_bytes(kwargs) > 120000:
            raise litellm.ContextWindowExceededError('scripted capacity rejection', 'mock', 'openai')
        if len(observed) == 1:
            return round_response(keys, names)
        if len(observed) == 3:
            return tool_response()
        raise RuntimeError('Insufficient Balance: unexpected request')

    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', {'task_id': 'task-a'})
    assert result.status == TrialStatus.UNVERIFIED
    assert len(observed) == 3
    assert_delivered(observed[1], keys, results)
    assert 120000 < payload_bytes(observed[1]) < 200000
    assert payload_bytes(observed[2]) < 120000 < payload_bytes(observed[1])
    assert observed[2]['messages'][:2] == initial
    assert_protocol(observed[2]['messages'])
    assert not any(message['role'] == 'tool' or message.get('tool_calls')
                   for message in observed[2]['messages'])
    notice = next(message for message in observed[2]['messages']
                  if message.get('name') == 'worker_tool_delivery')
    assert 'could not be delivered in full' in notice['content']
    assert 'Do not replay writes' in notice['content']
    events = [event for event in result.trajectory
              if event['type'] == 'tool_result_delivery_incomplete']
    assert len(events) == 1
    event = events[0]
    assert event['tool_call_ids'] == keys
    assert event['reason'] == 'context_overflow'
    assert event['target_bytes'] == event['before_bytes'] == payload_bytes(observed[1])
    assert event['after_bytes'] == payload_bytes(observed[2])
    recovery = next(event for event in result.trajectory
                    if event['type'] == 'context_overflow_recovery')
    assert recovery['request_reduced'] is True
    assert dispatches == list(zip(names, keys, strict=True))


def test_normal_compaction_keeps_recent_units_and_fresh_round_above_target(worker_binary, monkeypatch):
    keys = [f'compact-{index}' for index in range(5)]
    results = {key: ToolResult(True, result_output(key)) for key in keys}
    worker, dispatches, initial = setup_worker(worker_binary, monkeypatch, results, threshold=30000)
    initial.extend(history(7)[2:])
    original = deepcopy(initial)
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            return round_response(keys)
        if len(observed) == 2:
            return tool_response()
        raise RuntimeError('Insufficient Balance: unexpected request')

    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', {'task_id': 'task-a'})
    assert result.status == TrialStatus.UNVERIFIED
    assert len(observed) == 2
    assert_delivered(observed[1], keys, results)
    assert payload_bytes(observed[1]) > 30000
    assert len(observed[1]['messages']) == 14
    retained_old_ids = {message['tool_call_id'] for message in observed[1]['messages']
                        if message['role'] == 'tool' and message['tool_call_id'] not in keys}
    assert retained_old_ids == {'5-0', '5-1', '6-0', '6-1'}
    assert observed[1]['messages'][:2] == initial[:2]
    assert initial == original
    assert dispatches == [('read', key) for key in keys]
    assert any(event['type'] == 'context_compaction' for event in result.trajectory)
    assert not any(event['type'] == 'tool_result_delivery_incomplete' for event in result.trajectory)


def test_irreducible_fresh_input_reports_existing_context_input_unfit(worker_binary, monkeypatch):
    keys = ['irreducible']
    results = {key: ToolResult(True, result_output(key)) for key in keys}
    worker, dispatches, initial = setup_worker(worker_binary, monkeypatch, results, threshold=1000)
    initial[1]['content'] += 'visible instruction' * 1000
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            return round_response(keys)
        if len(observed) <= 3:
            raise litellm.ContextWindowExceededError('irreducible instruction', 'mock', 'openai')
        raise RuntimeError('Insufficient Balance: unexpected repeated overflow')

    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', {'task_id': 'task-a'})
    assert result.status == TrialStatus.ERROR
    assert len(observed) == 3
    assert result.metadata['context_input_unfit'] is True
    assert result.metadata['time_round_token_limit_driven'] is False
    assert dispatches == [('read', keys[0])]
    assert_delivered(observed[1], keys, results)
    assert payload_bytes(observed[2]) < payload_bytes(observed[1])
    assert observed[2]['messages'][:2] == initial
    assert any(event['type'] == 'tool_result_delivery_incomplete' for event in result.trajectory)


@pytest.mark.parametrize('terminal', ['done', 'environment'])
def test_terminal_round_does_not_dispatch_a_delivery_request(worker_binary, monkeypatch, terminal):
    results = {'terminal': ToolResult(True, 'service "main" is not running')}
    worker, dispatches, _ = setup_worker(worker_binary, monkeypatch, results)
    observed = []

    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) > 1:
            raise RuntimeError('Insufficient Balance: unexpected request after termination')
        return tool_response() if terminal == 'done' else round_response(['terminal'])

    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('Inspect visible inputs.', {'task_id': 'task-a'})
    assert len(observed) == 1
    assert result.status == (TrialStatus.UNVERIFIED if terminal == 'done' else TrialStatus.ERROR)
    assert dispatches == ([] if terminal == 'done' else [('read', 'terminal')])
    assert not any(event['type'] == 'tool_result_delivery_incomplete' for event in result.trajectory)
