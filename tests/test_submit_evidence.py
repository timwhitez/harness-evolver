from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import threading

import pytest

from hl import submission_evidence as evidence
from hl import submit as module
from hl.submit import SubmitConfig, SubmitGate


pytestmark = pytest.mark.skipif(not hasattr(os, 'O_PATH'), reason='secure evidence needs Linux O_PATH')


def atif() -> dict:
    return {
        'schema_version': 'ATIF-v1.4', 'session_id': 'fixture-session',
        'agent': {'name': 'fixture', 'version': '1'},
        'steps': [
            {'step_id': 1, 'source': 'user', 'message': 'Run the visible check.'},
            {'step_id': 2, 'source': 'agent', 'message': 'Checking.',
             'tool_calls': [{'tool_call_id': 'c1', 'function_name': 'bash',
                             'arguments': {'command': 'true'}}],
             'observation': {'results': [{'source_call_id': 'c1', 'content': 'ok'}]}},
        ],
    }


def trial(task='task-a', attempt=1, reward=1.0, error=None) -> dict:
    return {'trial_name': f'{task}__{attempt}', 'task_name': task,
            'verifier_result': {'rewards': {'reward': reward}} if reward is not None else None,
            'exception_info': error}


def write_job(root: Path, trials=None, layout='inline') -> Path:
    root.mkdir(parents=True, exist_ok=True)
    trials = [trial()] if trials is None else trials
    job = {'trial_results': trials if layout in ('inline', 'both') else []}
    if layout != 'inline':
        job.update(n_total_trials=len(trials), finished_at='2026-09-05T00:00:00Z', stats={
            'n_completed_trials': len(trials), 'n_errored_trials': sum(bool(t.get('exception_info')) for t in trials),
            'n_running_trials': 0, 'n_pending_trials': 0,
        })
    (root / 'result.json').write_text(json.dumps(job), encoding='utf-8')
    for record in trials:
        directory = root / record['trial_name']
        (directory / 'agent').mkdir(parents=True, exist_ok=True)
        (directory / 'agent/trajectory.json').write_text(json.dumps(atif()), encoding='utf-8')
        if layout in ('native', 'both'):
            (directory / 'result.json').write_text(json.dumps(record), encoding='utf-8')
    return root


@pytest.fixture
def setup(tmp_path, monkeypatch):
    job = write_job(tmp_path / 'job')
    config = SubmitConfig(enabled=True, trigger_score=0, min_tasks_evaluated=1,
                          min_attempts_per_task=1, require_clean_git=False,
                          require_no_uncommitted_harness_diff=False)
    gate = SubmitGate(config, submissions_dir=tmp_path / 'submissions')
    monkeypatch.setattr(gate, '_harbor_auth_ok', lambda: True)
    uploads = []
    def upload(*args, **kwargs):
        uploads.append(args[0])
        return subprocess.CompletedProcess(args[0], 0, 'fixture upload', '')
    monkeypatch.setattr(module.subprocess, 'run', upload)
    args = dict(campaign_id='fixture', best_job_dir=job, score=1.0, tasks_evaluated=1,
                full_regression_passed=True)
    return gate, args, job, uploads


def replace_root(job, document):
    (job / 'result.json').write_text(json.dumps(document), encoding='utf-8')


@pytest.mark.parametrize('layout', ['inline', 'native', 'both'])
def test_valid_job_is_eligible_and_uploads_once(setup, layout):
    gate, args, job, uploads = setup
    write_job(job, layout=layout)
    first = gate.submit_once(**args)
    second = gate.submit_once(**args)
    assert first.eligible and first.submitted and first.intent_persisted, first.reasons
    assert not second.eligible and not second.attempted
    assert len(uploads) == 1
    saved = json.loads(Path(first.intent_path).read_text())
    assert saved['evidence'] == first.evidence
    assert saved['evidence']['score'] == 1.0
    assert saved['evidence']['attempts_per_task'] == {'task-a': 1}
    assert len(saved['evidence']['fingerprint']) == 64


@pytest.mark.parametrize('value', [None, '1', True, False, float('nan'), float('inf'),
                                  -float('inf'), -0.01, 1.01, 10**1000, {}, []])
def test_invalid_score_fails_before_any_external_command(setup, value):
    gate, args, _, uploads = setup
    args['score'] = value
    result = gate.submit_once(**args)
    assert not result.eligible and not result.attempted
    assert any('score must be' in r for r in result.reasons)
    assert uploads == [] and not list(gate.submissions_dir.iterdir())


@pytest.mark.parametrize('field', ['tasks_evaluated', 'min_tasks_evaluated', 'min_attempts_per_task'])
@pytest.mark.parametrize('value', [True, 0, -1, 1.5, '1', None])
def test_counts_must_be_positive_integers(setup, field, value):
    gate, args, _, uploads = setup
    if field == 'tasks_evaluated': args[field] = value
    else: setattr(gate.config, field, value)
    result = gate.check(**args)
    assert not result.eligible and any(field in r for r in result.reasons)
    assert uploads == []


@pytest.mark.parametrize('value', [True, None, float('nan'), float('inf'), -1, 1.01, '0.5'])
def test_invalid_trigger_cannot_weaken_gate(setup, value):
    gate, args, _, _ = setup
    gate.config.trigger_score = value
    assert not gate.check(**args).eligible


@pytest.mark.parametrize('mapping', [{}, [], 'x', {'task-a': True}, {'task-a': 0},
                                    {'task-a': -1}, {'task-a': 1.1}, {'task-a': '1'},
                                    {1: 1}, {'': 1}])
def test_invalid_attempt_mapping_is_not_coerced(setup, mapping):
    gate, args, _, _ = setup
    args['attempts_per_task'] = mapping
    assert not gate.check(**args).eligible


@pytest.mark.parametrize('records', [[None], [{}], [1], ['fake'], {}, None, True, 'fake'])
def test_malformed_trial_collection_never_uploads(setup, records):
    gate, args, job, uploads = setup
    replace_root(job, {'trial_results': records})
    result = gate.submit_once(**args)
    assert not result.eligible and not result.attempted
    assert not uploads and not list(gate.submissions_dir.iterdir())


@pytest.mark.parametrize('changes', [
    {'trial_name': ''}, {'task_name': ''}, {'trial_name': '../escape'},
    {'trial_name': '/absolute'}, {'trial_name': 'a/b'}, {'trial_name': 'a\\b'},
    {'trial_name': 'a\n'}, {'task_name': None}, {'verifier_result': []},
    {'verifier_result': {'rewards': []}}, {'verifier_result': {'rewards': {}}},
    {'verifier_result': {'rewards': {'reward': True}}},
    {'verifier_result': {'rewards': {'reward': '1'}}},
    {'verifier_result': {'rewards': {'reward': 2}}},
    {'verifier_result': {'rewards': {'reward': -1}}},
    {'verifier_result': None}, {'exception_info': {}},
    {'exception_info': {'exception_type': 'Error'}}, {'exception_info': 'Error'},
    {'finished_at': None}, {'config': []}, {'config': {'task': []}},
    {'step_results': [{'step_name': 'unsupported-multistep'}]},
])
def test_malformed_attempt_is_not_skipped_or_scored(setup, changes):
    gate, args, job, uploads = setup
    record = trial(); record.update(changes)
    replace_root(job, {'trial_results': [record]})
    result = gate.submit_once(**args)
    assert not result.eligible and not result.attempted
    assert uploads == []


@pytest.mark.parametrize('document', [b'{"trial_results":[],"trial_results":[]}',
                                     b'{"trial_results":[NaN]}', b'{"extra":1e999}',
                                     b'\xff', b'x', b'[]', b'null'])
def test_ambiguous_or_invalid_json_is_rejected(setup, document):
    gate, args, job, _ = setup
    (job / 'result.json').write_bytes(document)
    assert not gate.check(**args).eligible


def test_valid_failures_remain_in_the_denominator(setup):
    gate, args, job, _ = setup
    records = [trial(attempt=1), trial(attempt=2, reward=0), trial(attempt=3, reward=None,
               error={'exception_type': 'TimeoutError', 'exception_message': 'fixture timeout'})]
    write_job(job, records, layout='native')
    args.update(score=round(1/3, 4), attempts_per_task={'task-a': 3})
    result = gate.check(**args)
    assert result.eligible, result.reasons
    assert result.evidence['score'] == pytest.approx(1/3)
    assert result.evidence['attempts_per_task'] == {'task-a': 3}
    args['score'] = 1.0
    assert not gate.check(**args).eligible


def test_macro_task_mean_not_attempt_weighted_mean(setup):
    gate, args, job, _ = setup
    write_job(job, [trial(reward=0), trial(attempt=2, reward=0), trial(task='task-b')])
    args.update(score=0.5, tasks_evaluated=2, attempts_per_task={'task-a': 2, 'task-b': 1})
    assert gate.check(**args).eligible
    args['score'] = 1/3
    assert not gate.check(**args).eligible


def test_exception_does_not_become_pass_from_stale_reward(setup):
    gate, args, job, _ = setup
    write_job(job, [trial(error={'exception_type': 'WorkerError', 'exception_message': ''})])
    args['score'] = 0
    result = gate.check(**args)
    assert result.eligible and result.evidence['score'] == 0


@pytest.mark.parametrize('changes', [{'score': 0.9}, {'tasks_evaluated': 89},
                                    {'attempts_per_task': {'task-a': 5}},
                                    {'attempts_per_task': {'other': 1}}])
def test_supplied_summary_cannot_describe_another_job(setup, changes):
    gate, args, _, uploads = setup
    args.update(changes)
    result = gate.submit_once(**args)
    assert not result.eligible and any('disagrees' in r for r in result.reasons)
    assert uploads == []


def test_rounded_score_cannot_cross_raw_trigger(setup):
    gate, args, job, _ = setup
    write_job(job, [trial(reward=.69996)])
    args['score'] = .7; gate.config.trigger_score = .7
    result = gate.check(**args)
    assert not result.eligible
    assert 'verified Harbor job score is below the submit trigger' in result.reasons


def test_minimum_attempts_comes_from_records_not_caller(setup):
    gate, args, job, _ = setup
    gate.config.min_attempts_per_task = 2
    assert not gate.check(**args).eligible
    write_job(job, [trial(), trial(attempt=2)])
    assert gate.check(**args).eligible


def test_duplicate_trial_name_does_not_inflate_attempt_count(setup):
    gate, args, job, _ = setup
    replace_root(job, {'trial_results': [trial(), trial()]})
    assert not gate.check(**args).eligible


def test_same_task_name_different_sources_is_ambiguous(setup):
    gate, args, job, _ = setup
    a, b = trial(), trial(attempt=2)
    a['source'], b['source'] = 'dataset-A', 'dataset-B'
    write_job(job, [a, b])
    result = gate.check(**args)
    assert not result.eligible and any('ambiguous' in r for r in result.reasons)


@pytest.mark.parametrize('mutation', ['reward', 'extra', 'name'])
def test_inline_and_disk_result_disagreement_is_fatal(setup, mutation):
    gate, args, job, _ = setup
    write_job(job, layout='both')
    path = job / 'task-a__1/result.json'
    record = json.loads(path.read_text())
    if mutation == 'reward': record['verifier_result']['rewards']['reward'] = 0
    elif mutation == 'name': record['trial_name'] = 'other'
    else:
        (job / 'extra').mkdir()
        (job / 'extra/result.json').write_text(json.dumps(trial(task='extra')))
    path.write_text(json.dumps(record))
    assert not gate.check(**args).eligible


@pytest.mark.parametrize('field,value', [('n_total_trials', 2), ('n_total_trials', True),
     ('n_completed_trials', 0), ('n_completed_trials', True), ('n_pending_trials', 1),
     ('n_running_trials', 1), ('n_errored_trials', 1), ('n_cancelled_trials', 1)])
def test_native_stats_must_match_completed_records(setup, field, value):
    gate, args, job, _ = setup
    write_job(job, layout='native')
    data = json.loads((job / 'result.json').read_text())
    (data if field == 'n_total_trials' else data['stats'])[field] = value
    replace_root(job, data)
    assert not gate.check(**args).eligible


def test_missing_trial_file_is_not_recovered_from_stats_only(setup):
    gate, args, job, _ = setup
    write_job(job, layout='native')
    (job / 'task-a__1/result.json').unlink()
    assert not gate.check(**args).eligible


@pytest.mark.parametrize('payload', [b'x', b'{}', b'[]', b'{"type":"tool_call"}', b'\xff'])
def test_non_atif_nonempty_file_is_not_evidence(setup, payload):
    gate, args, job, uploads = setup
    (job / 'task-a__1/agent/trajectory.json').write_bytes(payload)
    result = gate.submit_once(**args)
    assert not result.eligible and not result.attempted and not uploads


@pytest.mark.parametrize('mutation', ['version', 'steps', 'id', 'source', 'call', 'unknown', 'nan'])
def test_official_atif_model_rejects_bad_structures(setup, mutation):
    gate, args, job, _ = setup
    doc = atif()
    if mutation == 'version': doc['schema_version'] = 'invented'
    elif mutation == 'steps': doc['steps'] = []
    elif mutation == 'id': doc['steps'][0]['step_id'] = 2
    elif mutation == 'source': doc['steps'][0]['source'] = 'tool'
    elif mutation == 'call': doc['steps'][1]['observation']['results'][0]['source_call_id'] = 'missing'
    elif mutation == 'nan': doc['final_metrics'] = {'total_cost_usd': float('nan')}
    else: doc['unrecognized_field'] = True
    (job / 'task-a__1/agent/trajectory.json').write_text(json.dumps(doc))
    assert not gate.check(**args).eligible


def test_canonical_atif_takes_priority_over_native_event_log(setup):
    gate, args, job, _ = setup
    (job / 'task-a__1/agent/trajectory.jsonl').write_text('{"type":"tool_call"}\n')
    assert gate.check(**args).eligible


def test_invalid_preferred_atif_cannot_fall_back_to_a_valid_other_file(setup):
    gate, args, job, _ = setup
    directory = job / 'task-a__1/agent'
    (directory / 'trajectory.jsonl').write_text(json.dumps(atif())+'\n')
    (directory / 'trajectory.json').write_text('x')
    assert not gate.check(**args).eligible


def test_legacy_jsonl_full_atif_document_is_supported(setup):
    gate, args, job, _ = setup
    directory = job / 'task-a__1/agent'
    (directory / 'trajectory.json').rename(directory / 'trajectory.jsonl')
    assert gate.check(**args).eligible


def test_trajectory_not_required_for_verified_failure(setup):
    gate, args, job, _ = setup
    write_job(job, [trial(reward=0)])
    (job / 'task-a__1/agent/trajectory.json').unlink()
    args['score'] = 0
    assert gate.check(**args).eligible


def test_explicit_no_atif_policy_does_not_skip_trial_validation(setup):
    gate, args, job, _ = setup
    gate.config.require_atif_trajectory = False
    (job / 'task-a__1/agent/trajectory.json').unlink()
    assert gate.check(**args).eligible
    replace_root(job, {'trial_results': [None]})
    assert not gate.check(**args).eligible


def test_unresolved_external_trajectory_does_not_prove_completeness(setup):
    gate, args, job, _ = setup
    doc = atif(); doc['continued_trajectory_ref'] = 'https://example.invalid/part.json'
    (job / 'task-a__1/agent/trajectory.json').write_text(json.dumps(doc))
    assert not gate.check(**args).eligible


@pytest.mark.parametrize('kind', ['fifo', 'symlink', 'hardlink', 'directory'])
def test_required_evidence_must_be_unique_regular_file(setup, tmp_path, kind):
    gate, args, job, _ = setup
    path = job / 'task-a__1/agent/trajectory.json'
    original = path.read_bytes(); path.unlink()
    outside = tmp_path / 'outside.json'; outside.write_bytes(original)
    if kind == 'fifo': os.mkfifo(path)
    elif kind == 'symlink': path.symlink_to(outside)
    elif kind == 'hardlink': os.link(outside, path)
    else: path.mkdir()
    old = signal.signal(signal.SIGALRM, lambda *_: pytest.fail('evidence read blocked'))
    signal.alarm(3)
    try: assert not gate.check(**args).eligible
    finally: signal.alarm(0); signal.signal(signal.SIGALRM, old)
    assert outside.read_bytes() == original


def test_symlinked_job_root_is_not_followed(setup, tmp_path):
    gate, args, job, _ = setup
    link = tmp_path / 'alias'; link.symlink_to(job, target_is_directory=True)
    args['best_job_dir'] = link
    assert not gate.check(**args).eligible


@pytest.mark.parametrize('constant,cap', [('_MAX_FILE_BYTES', 20), ('_MAX_READ_BYTES', 50),
                                       ('_MAX_ENTRIES', 3), ('_MAX_DEPTH', 1)])
def test_inspection_limit_fails_explicitly_not_silently_skips(setup, monkeypatch, constant, cap):
    gate, args, _, _ = setup
    monkeypatch.setattr(evidence, constant, cap)
    assert not gate.check(**args).eligible


def test_policy_scan_still_rejects_solution_fetch_in_native_log(setup):
    gate, args, job, _ = setup
    (job / 'task-a__1/agent/native.log').write_text('curl https://example.com/solutions/task.py')
    result = gate.check(**args)
    assert not result.eligible and any('External solution URL access' in r for r in result.reasons)


def test_evidence_modification_during_validation_is_detected(setup, monkeypatch):
    gate, args, job, _ = setup
    original = evidence._JobFiles.read
    changed = False
    def read(files, relative):
        nonlocal changed
        data = original(files, relative)
        if not changed:
            changed = True
            (job / 'late-file.txt').write_text('changed')
        return data
    monkeypatch.setattr(evidence._JobFiles, 'read', read)
    assert not gate.check(**args).eligible


def test_job_change_after_claim_cancels_without_upload_and_retains_intent(setup, monkeypatch):
    gate, args, job, uploads = setup
    original = module.write_exclusive_json
    def write(path, payload):
        original(path, payload)
        if path.name.endswith('.intent.json'):
            write_job(job, [trial(reward=0)])
    monkeypatch.setattr(module, 'write_exclusive_json', write)
    result = gate.submit_once(**args)
    assert not result.eligible and not result.attempted and not result.submitted
    assert result.intent_persisted and Path(result.intent_path).exists() and not uploads
    assert any('before launch' in r for r in result.reasons)


def test_valid_integrity_check_does_not_weaken_exclusive_claim(setup, monkeypatch):
    gate, args, _, uploads = setup
    barrier = threading.Barrier(2, timeout=5)
    original = gate.check
    def check(**kwargs):
        result = original(**kwargs); barrier.wait(); return result
    monkeypatch.setattr(gate, 'check', check)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: gate.submit_once(**args), range(2)))
    assert len(uploads) == 1 and sum(r.attempted for r in results) == 1


def test_dry_run_validates_without_claim_or_upload(setup):
    gate, args, _, uploads = setup
    assert gate.submit_once(**args, dry_run=True).eligible
    assert uploads == [] and not list(gate.submissions_dir.iterdir())


def test_explicit_upload_skip_retains_existing_behavior(setup):
    gate, args, job, uploads = setup
    gate.config.harbor_upload = False
    (job / 'result.json').unlink()
    result = gate.submit_once(**args)
    assert result.eligible and result.upload_skipped and not result.submitted
    assert result.attempted and uploads == []


@pytest.mark.parametrize('changes', [
    {'finished_at': 'not-a-date'}, {'finished_at': True}, {'finished_at': 7},
    {'step_results': {}}, {'step_results': False}, {'step_results': ''},
    {'source': []}, {'source': ''}, {'task_checksum': 7},
    {'agent_info': []}, {'agent_info': {'name': 7, 'version': '1'}},
    {'agent_info': {'name': 'fixture', 'version': '1', 'model_info': []}},
    {'task_id': []}, {'task_id': {}}, {'config': {'task': {'path': []}}},
])
def test_optional_identity_and_finalization_fields_are_not_opaque(setup, changes):
    gate, args, job, uploads = setup
    record = trial(); record.update(changes)
    replace_root(job, {'trial_results': [record]})
    assert not gate.submit_once(**args).eligible
    assert not uploads


@pytest.mark.parametrize('layout', ['inline', 'native'])
def test_malformed_job_finalization_is_rejected(setup, layout):
    gate, args, job, _ = setup
    write_job(job, layout=layout)
    root = json.loads((job / 'result.json').read_text())
    root['finished_at'] = 'done'
    replace_root(job, root)
    assert not gate.check(**args).eligible


def test_native_job_must_have_finalization_and_completed_count(setup):
    gate, args, job, _ = setup
    write_job(job, layout='native')
    root = json.loads((job / 'result.json').read_text())
    for broken in [dict(root, finished_at=None), {k: v for k, v in root.items() if k != 'finished_at'},
                   dict(root, stats={})]:
        replace_root(job, broken)
        assert not gate.check(**args).eligible


def test_distinct_tasks_cannot_mix_model_or_dataset_scopes(setup):
    gate, args, job, _ = setup
    records = [trial(), trial(task='task-b')]
    records[0]['source'] = 'dataset-a'
    records[1]['source'] = 'dataset-b'
    write_job(job, records)
    args['tasks_evaluated'] = 2
    assert not gate.check(**args).eligible


def test_valid_identity_and_iso_finalization_are_supported(setup):
    gate, args, job, _ = setup
    record = trial()
    record.update(source='dataset', task_checksum='abc', task_id={'path': 'task-a'},
                  config={'task': {'path': 'task-a'}},
                  agent_info={'name': 'hl', 'version': '1', 'model_info': {'name': 'model'}},
                  finished_at='2026-09-05T00:00:00Z', step_results=[])
    write_job(job, [record], layout='native')
    assert gate.check(**args).eligible


@pytest.mark.parametrize('field', ['require_integrity_scan', 'require_atif_trajectory', 'harbor_upload'])
@pytest.mark.parametrize('value', [None, 0, 'false', []])
def test_malformed_boolean_config_cannot_disable_evidence_validation(setup, field, value):
    gate, args, _, uploads = setup
    setattr(gate.config, field, value)
    result = gate.submit_once(**args)
    assert not result.eligible and not result.attempted
    assert any(field in reason for reason in result.reasons)
    assert not uploads


def test_missing_official_atif_validator_fails_closed(setup, monkeypatch):
    import builtins
    gate, args, _, uploads = setup
    original = builtins.__import__
    def missing(name, *values, **kwargs):
        if name == 'harbor.models.trajectories.trajectory':
            raise ModuleNotFoundError(name)
        return original(name, *values, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    result = gate.submit_once(**args)
    assert not result.eligible and 'unavailable' in result.reasons[0]
    assert not uploads
