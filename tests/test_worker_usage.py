from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from bench.agent import HLAgent
from bench.harbor import HarborRunner
from bench.harbor_adapter import HLWorkerHarborAgent
from bench.runtime_resources import worker_crate_root
from bench.usage import normalize_worker_usage, USAGE_SCHEMA
from harness.tools.registry import ToolRegistry
from hl.types import TrialResult
from scripts.run_campaign import _aggregate_efficiency


@pytest.mark.parametrize('objects', [False, True])
@pytest.mark.parametrize('cache_fields', [
    {'prompt_tokens_details': {'cached_tokens': 1920}},
    {'cache_read_input_tokens': 1920},
    {'prompt_tokens_details': {'cached_tokens': 1920}, 'cache_read_input_tokens': 1920},
])
def test_normalize_standard_usage_and_equal_aliases(objects, cache_fields):
    usage = dict(prompt_tokens=2006, input_tokens=2006,
                 completion_tokens=300, output_tokens=300, **cache_fields)
    if objects:
        if 'prompt_tokens_details' in usage:
            usage['prompt_tokens_details'] = SimpleNamespace(**usage['prompt_tokens_details'])
        usage = SimpleNamespace(**usage)
    counts, observation = normalize_worker_usage(usage)
    assert counts == {'input': 86, 'cache': 1920, 'output': 300}
    assert observation == {'schema': USAGE_SCHEMA, 'status': 'complete',
                           'unknown_fields': [], 'diagnostics': [],
                           'source_input_semantics': 'litellm_inclusive'}


@pytest.mark.parametrize('usage,expected,status', [
    (None, {}, 'incomplete'),
    ({'prompt_tokens': 10, 'completion_tokens': 2}, {'output': 2}, 'incomplete'),
    ({'prompt_tokens': 10, 'completion_tokens': 2, 'cache_read_input_tokens': 0},
     {'input': 10, 'cache': 0, 'output': 2}, 'complete'),
    ({'prompt_tokens': True, 'cache_read_input_tokens': 0}, {}, 'invalid'),
    ({'prompt_tokens': -1, 'cache_read_input_tokens': 0}, {}, 'invalid'),
    ({'prompt_tokens': 10, 'input_tokens': 11, 'cache_read_input_tokens': 0}, {}, 'invalid'),
    ({'prompt_tokens': 10, 'cache_read_input_tokens': 11}, {}, 'invalid'),
    ({'prompt_tokens': 10, 'cache_read_input_tokens': False}, {}, 'invalid'),
    ({'prompt_tokens': 10, 'cache_read_input_tokens': 3,
      'prompt_tokens_details': {'cached_tokens': 4}}, {}, 'invalid'),
    ({'completion_tokens': 2, 'output_tokens': 3}, {}, 'invalid'),
    ({'prompt_tokens': 2**63, 'cache_read_input_tokens': 0}, {}, 'invalid'),
])
def test_missing_invalid_conflicting_usage_is_not_known_zero(usage, expected, status):
    counts, observation = normalize_worker_usage(usage)
    assert counts == expected
    assert observation['status'] == status
    assert observation['unknown_fields'] == sorted({'input', 'cache', 'output'} - counts.keys())


def test_raw_anthropic_exclusive_input_is_not_subtracted_again():
    counts, observation = normalize_worker_usage({
        'input_tokens': 86, 'cache_read_input_tokens': 1920, 'output_tokens': 300})
    assert counts == {'input': 86, 'cache': 1920, 'output': 300}
    assert observation['source_input_semantics'] == 'raw_anthropic_exclusive'
    counts, observation = normalize_worker_usage({
        'input_tokens': 86, 'cache_read_input_tokens': 1920,
        'cache_creation_input_tokens': 100, 'output_tokens': 300})
    assert counts == {'input': 186, 'cache': 1920, 'output': 300}
    assert observation['status'] == 'complete'
    counts, observation = normalize_worker_usage({
        'input_tokens': 86, 'cache_read_input_tokens': 1920,
        'cache_creation_input_tokens': True, 'output_tokens': 300})
    assert counts == {'output': 300}
    assert observation['status'] == 'invalid'


@pytest.fixture(scope='module', params=['source', 'packaged'])
def worker_binary(request, tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    source = root / 'crates/hl-worker-core'
    crate = source if request.param == 'source' else worker_crate_root(
        tmp_path_factory.mktemp('usage-runtime'))
    assert (crate / 'src/main.rs').read_bytes() == (source / 'src/main.rs').read_bytes()
    target = tmp_path_factory.mktemp('usage-target-' + request.param)
    subprocess.run(['cargo', 'build', '--quiet', '--locked', '--manifest-path',
                    str(crate / 'Cargo.toml'), '--target-dir', str(target)], check=True)
    return target / 'debug/hl-worker-core'


def response(usage, done=False):
    calls = [SimpleNamespace(id='finish', function=SimpleNamespace(
        name='done', arguments='{"summary":"ready"}'))] if done else [SimpleNamespace(
            id='todo', function=SimpleNamespace(name='todo_read', arguments='{}'))]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='', tool_calls=calls, reasoning_content=None))], usage=usage)


def run_script(worker_binary, monkeypatch, usages):
    monkeypatch.setenv('HL_WORKER_RUST_BIN', str(worker_binary))
    script = [response(usage, index == len(usages)-1) for index, usage in enumerate(usages)]
    def completion(**kwargs):
        if not script:
            raise RuntimeError('Insufficient Balance: test script exhausted')
        return script.pop(0)
    monkeypatch.setattr('bench.agent.litellm.completion', completion)
    worker = HLAgent()
    return worker, worker.run('Finish the task', {'task_id': 'task-a'})


def write_job(tmp_path, agent_results):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'result.json').write_text(json.dumps({'trial_results': [
        {'task_name': 'task-a', 'trial_name': f'task-a__{index}',
         'agent_result': result, 'verifier_result': {'rewards': {'reward': 0.0}}}
        for index, result in enumerate(agent_results)]}))
    return HarborRunner().parse_job_dir(tmp_path, task_id='task-a')


def test_real_bridge_adapter_parser_and_aggregate_conserve_usage(worker_binary, monkeypatch, tmp_path):
    usage = {'prompt_tokens': 2006, 'input_tokens': 2006,
             'completion_tokens': 300, 'output_tokens': 300,
             'prompt_tokens_details': {'cached_tokens': 1920}}
    worker, result = run_script(worker_binary, monkeypatch, [usage, usage])
    assert result.token_usage == {'input': 172, 'cache': 3840, 'output': 600}
    assert result.metadata['token_usage_observation']['status'] == 'complete'

    adapter = HLWorkerHarborAgent(logs_dir=tmp_path / 'agent', model_name='test')
    # Exercise the actual adapter run; repeat the result without making model calls.
    monkeypatch.setattr(worker, 'run', lambda *args: result)
    monkeypatch.setattr(adapter, '_build_agent', lambda *args: worker)
    monkeypatch.setattr(adapter, '_build_environment_registry', lambda *args: ToolRegistry())
    context = SimpleNamespace()
    asyncio.run(adapter.run('Finish the task', SimpleNamespace(environment_name='task-a'), context))
    assert context.n_cache_tokens == 3840
    parsed = write_job(tmp_path / 'job', [vars(context)])
    assert parsed.token_usage == result.token_usage
    assert parsed.metadata['token_usage_observation']['schema'] == USAGE_SCHEMA
    assert parsed.metadata['trial_metrics']['cache_hit_ratio'] == 0.9571
    aggregate = write_job(tmp_path / 'attempts', [vars(context), vars(context)])
    assert aggregate.token_usage == {'input': 344, 'cache': 7680, 'output': 1200}
    assert aggregate.metadata['trial_metrics']['cache_hit_ratio'] == 0.9571
    campaign = _aggregate_efficiency([parsed, parsed])
    assert campaign['token_usage'] == aggregate.token_usage
    assert campaign['trial_metrics']['cache_hit_ratio_mean'] == 0.9571
    assert campaign['trial_metrics']['cache_hit_ratio_samples'] == 2
    assert campaign['token_usage_coverage']['known_trials']['cache'] == 2


def test_any_unknown_call_remains_unknown_in_final_usage(worker_binary, monkeypatch):
    usage = {'prompt_tokens': 10, 'completion_tokens': 2, 'cache_read_input_tokens': 0}
    _, result = run_script(worker_binary, monkeypatch, [
        usage, {'prompt_tokens': 10, 'completion_tokens': 2}, usage])
    assert result.token_usage == {'output': 6}
    assert result.metadata['token_usage_observation']['status'] == 'incomplete'
    assert 'missing_cache' in result.metadata['token_usage_observation']['diagnostics']


def test_zero_cache_survives_adapter(worker_binary, monkeypatch, tmp_path):
    worker, result = run_script(worker_binary, monkeypatch, [
        {'prompt_tokens': 10, 'completion_tokens': 2, 'cache_read_input_tokens': 0}])
    assert result.token_usage['cache'] == 0
    adapter = HLWorkerHarborAgent(logs_dir=tmp_path / 'agent', model_name='test')
    monkeypatch.setattr(worker, 'run', lambda *args: result)
    monkeypatch.setattr(adapter, '_build_agent', lambda *args: worker)
    monkeypatch.setattr(adapter, '_build_environment_registry', lambda *args: ToolRegistry())
    context = SimpleNamespace()
    asyncio.run(adapter.run('Finish the task', SimpleNamespace(environment_name='task-a'), context))
    assert context.n_cache_tokens == 0
    # The bridge/parser must preserve known zero, unlike absent cache.
    parsed = write_job(tmp_path / 'zero', [{'n_input_tokens': 10, 'n_cache_tokens': 0,
                                          'n_output_tokens': 2}])
    assert parsed.token_usage['cache'] == 0
    assert parsed.metadata['trial_metrics']['cache_hit_ratio'] == 0


def test_unknown_attempts_and_legacy_reports_are_labelled(tmp_path):
    complete = {'n_input_tokens': 86, 'n_cache_tokens': 1920, 'n_output_tokens': 300}
    unknown = {'n_input_tokens': 10, 'n_output_tokens': 2}
    parsed = write_job(tmp_path / 'legacy', [unknown])
    assert parsed.token_usage == {'input': 10, 'output': 2}
    assert parsed.metadata['token_usage_observation']['schema'] == 'legacy'
    assert 'cache_hit_ratio' not in parsed.metadata['trial_metrics']
    aggregate = write_job(tmp_path / 'mixed', [complete, unknown])
    assert 'cache' not in aggregate.token_usage
    assert 'cache_hit_ratio' not in aggregate.metadata['trial_metrics']
    known = write_job(tmp_path / 'known', [{**complete, 'metadata': {
        'token_usage_observation': {'schema': USAGE_SCHEMA, 'status': 'complete'}}}])
    campaign = _aggregate_efficiency([known, parsed])
    assert campaign['token_usage'] == {'output': 302}
    assert campaign['trial_metrics']['cache_hit_ratio_mean'] == 0.9571
    assert campaign['trial_metrics']['cache_hit_ratio_samples'] == 1
    assert campaign['token_usage_coverage']['schema'] == 'mixed_or_empty'
    assert campaign['token_usage_coverage']['known_trials']['cache'] == 1


def test_mixed_legacy_and_v1_denominators_are_not_combined(tmp_path):
    canonical = {'n_input_tokens': 86, 'n_cache_tokens': 1920, 'n_output_tokens': 300,
                 'metadata': {'token_usage_observation': {'schema': USAGE_SCHEMA, 'status': 'complete'}}}
    legacy = {'n_input_tokens': 2006, 'n_cache_tokens': 1920, 'n_output_tokens': 300}
    combined = write_job(tmp_path / 'mixed', [canonical, legacy])
    assert combined.token_usage == {'output': 600}
    assert 'cache_hit_ratio' not in combined.metadata['trial_metrics']
    assert combined.metadata['token_usage_observation']['status'] == 'legacy_or_mixed'
