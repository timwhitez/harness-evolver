from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import litellm
import pytest

from bench.agent import HLAgent
from hl.types import TrialStatus
from tests.test_worker_usage import worker_binary  # Reuse the source/packaged JSONL contract fixture.


def payload_bytes(kwargs):
    return len(json.dumps({key: kwargs[key] for key in ('messages', 'tools', 'tool_choice')},
                          ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def tool_response(name='done', args=None):
    args = args if args is not None else {'summary': 'ready'}
    return SimpleNamespace(usage={'prompt_tokens': 10, 'completion_tokens': 1,
                                  'cache_read_input_tokens': 0}, choices=[SimpleNamespace(
        message=SimpleNamespace(content='', reasoning_content=None, tool_calls=[
            SimpleNamespace(id='next-'+name, function=SimpleNamespace(
                name=name, arguments=json.dumps(args)))]) )])


def history(units, dominant='arguments'):
    messages = [{'role': 'system', 'content': 'Preserve these rules'},
                {'role': 'user', 'content': 'Original task: finish the task'}]
    for index in range(units):
        calls = [{'id': f'{index}-{tool}', 'type': 'function', 'function': {
            'name': 'write', 'arguments': json.dumps({'file_path': '/tmp/past',
                'content': 'code'*2000 if dominant == 'arguments' else ''})}}
                 for tool in range(2)]
        messages.append({'role': 'assistant', 'content': '', 'tool_calls': calls,
                         'reasoning_content': '思考'*2000 if dominant == 'reasoning' else ''})
        messages.extend({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Historical write completed'}
                        for call in reversed(calls))
    return messages


def assert_protocol(messages):
    index = 0
    while index < len(messages):
        message = messages[index]
        assert message['role'] != 'tool', 'orphan result'
        calls = message.get('tool_calls') or []
        if calls:
            ids = {call['id'] for call in calls}
            assert len(ids) == len(calls)
            for call in calls:
                json.loads(call['function']['arguments'])
            results = messages[index+1:index+1+len(calls)]
            assert len(results) == len(calls)
            assert all(result['role'] == 'tool' for result in results)
            assert {result['tool_call_id'] for result in results} == ids
        index += 1 + len(calls)


def setup_worker(worker_binary, monkeypatch, messages, threshold=10**9, pending=False):
    monkeypatch.setenv('HL_WORKER_RUST_BIN', str(worker_binary))
    worker = HLAgent()
    original_request = worker._rust_worker_request
    def request(instruction, context):
        payload = original_request(instruction, context)
        payload['initial_messages'] = deepcopy(messages)
        payload['thresholds']['compaction_char_threshold'] = threshold
        # Tools contribute substantial fixed request size even with tiny content.
        payload['tool_schemas'][0]['function']['description'] += 'schema'*1000
        if pending:
            payload['todo_items'] = [{'id': 'pending', 'content': 'finish work', 'status': 'pending'}]
        return payload
    monkeypatch.setattr(worker, '_rust_worker_request', request)
    return worker


@pytest.mark.parametrize('units', [2, 7])
@pytest.mark.parametrize('dominant', ['arguments', 'reasoning'])
def test_overflow_second_request_shrinks_without_replaying_parallel_writes(
    worker_binary, monkeypatch, units, dominant,
):
    messages = history(units, dominant)
    worker = setup_worker(worker_binary, monkeypatch, messages, pending=True)
    observed = []
    boundary = None
    def completion(**kwargs):
        nonlocal boundary
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            boundary = payload_bytes(kwargs)-1000
            raise litellm.ContextWindowExceededError('Mock context boundary', 'mock', 'openai')
        assert payload_bytes(kwargs) < boundary
        assert_protocol(kwargs['messages'])
        assert kwargs['messages'][:2] == messages[:2]
        if len(observed) == 2:
            note = next(message for message in kwargs['messages']
                        if message.get('name') == 'worker_context_state')
            assert 'pending' in note['content'] and 'finish work' in note['content']
            assert 'not semantically summarized' in note['content']
            return tool_response('todo_write', {'items': []})
        if len(observed) == 3:
            return tool_response()
        raise RuntimeError('Insufficient Balance: unexpected replay')
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('finish the task', {'task_id': 'task-a'})
    assert result.status == TrialStatus.UNVERIFIED
    assert len(observed) == 3
    assert [call['tool'] for call in result.tool_calls] == ['todo_write', 'done']
    recovery = next(event for event in result.trajectory if event['type'] == 'context_overflow_recovery')
    assert recovery['request_reduced'] is True
    assert recovery['rejected_bytes'] == payload_bytes(observed[0])
    assert recovery['next_bytes'] == payload_bytes(observed[1])
    compaction = next(event for event in result.trajectory if event['type'] == 'context_compaction')
    assert compaction['reason'] == 'context_overflow'
    assert compaction['semantic_summary_generated'] is False
    assert_protocol(compaction['omitted_history'])
    assert compaction['omitted_history'][0]['tool_calls'] == messages[2]['tool_calls']
    assert compaction['after_bytes'] < compaction['before_bytes']


def test_normal_threshold_counts_arguments_with_short_content(worker_binary, monkeypatch):
    messages = history(7)
    worker = setup_worker(worker_binary, monkeypatch, messages, threshold=5000)
    observed = []
    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        assert_protocol(kwargs['messages'])
        return tool_response()
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('finish the task', {'task_id': 'task-a'})
    assert result.status == TrialStatus.UNVERIFIED
    assert len(observed[0]['messages']) < len(messages)
    assert observed[0]['messages'][:2] == messages[:2]
    assert any(event['type'] == 'context_compaction' and event['reason'] == 'payload_threshold'
               for event in result.trajectory)


@pytest.mark.parametrize('unfit', ['task', 'schemas'])
def test_irreducible_overflow_has_one_diagnostic_and_one_request(worker_binary, monkeypatch, unfit):
    messages = history(0)
    if unfit == 'task':
        messages[1]['content'] += 'huge instruction'*1000
    worker = setup_worker(worker_binary, monkeypatch, messages)
    observed = []
    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) > 1:
            raise RuntimeError('Insufficient Balance: repeated invalid request')
        raise litellm.ContextWindowExceededError('Unfit '+unfit, 'mock', 'openai')
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('finish the task', {'task_id': 'task-a'})
    assert result.status == TrialStatus.ERROR
    assert result.metadata['context_input_unfit'] is True
    assert result.metadata['time_round_token_limit_driven'] is False
    assert len(observed) == 1
    assert result.tool_calls == []
    assert any('context_input_unfit' in error for error in result.error_log)
    assert sum(event['type'] == 'context_overflow_recovery' for event in result.trajectory) == 1


def test_transient_retry_keeps_existing_recovery_semantics(worker_binary, monkeypatch):
    messages = history(2)
    worker = setup_worker(worker_binary, monkeypatch, messages)
    observed = []
    def completion(**kwargs):
        observed.append(deepcopy(kwargs))
        if len(observed) == 1:
            raise TimeoutError('429 temporarily unavailable')
        if len(observed) == 2:
            return tool_response()
        raise RuntimeError('Insufficient Balance: unexpected retry')
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('finish the task', {'task_id': 'task-a'})
    assert result.status == TrialStatus.UNVERIFIED
    assert len(observed) == 2
    assert observed[1]['messages'][:len(messages)] == messages
    assert any('Model-provider recovery checkpoint' in message.get('content', '')
               for message in observed[1]['messages'])
    assert any(event['type'] == 'llm_error_recovery_prompt' for event in result.trajectory)
    assert not any(event['type'] == 'context_overflow_recovery' for event in result.trajectory)


def test_bridge_classifies_context_overflow_without_changing_transient_errors():
    agent = HLAgent()
    overflow = litellm.ContextWindowExceededError('overflow', 'mock', 'openai')
    assert agent._llm_error_response_payload(overflow)['error']['kind'] == 'context_overflow'
    coded = RuntimeError('bad request')
    coded.code = 'context_length_exceeded'
    assert agent._llm_error_response_payload(coded)['error']['kind'] == 'context_overflow'
    assert agent._llm_error_response_payload(TimeoutError('429'))['error']['kind'] == 'transient'


def test_small_completed_unit_can_shrink_schema_dominated_request(worker_binary, monkeypatch):
    messages = history(1, dominant='small')
    worker = setup_worker(worker_binary, monkeypatch, messages)
    sizes = []
    def completion(**kwargs):
        sizes.append(payload_bytes(kwargs))
        if len(sizes) == 1:
            raise litellm.ContextWindowExceededError('schema-dominated', 'mock', 'openai')
        assert len(sizes) == 2 and sizes[1] < sizes[0]
        assert_protocol(kwargs['messages'])
        return tool_response()
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    result = worker.run('finish the task', {'task_id': 'task-a'})
    assert result.status == TrialStatus.UNVERIFIED
    assert len(sizes) == 2
