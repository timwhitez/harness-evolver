from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import threading

import pytest

import hl.submit as module
from hl.submit import SubmitConfig, SubmitGate
from hl.submission_storage import write_exclusive_json


def gate_for(root: Path, **options) -> SubmitGate:
    config = SubmitConfig(enabled=True, min_tasks_evaluated=1, min_attempts_per_task=1,
                          require_clean_git=False, require_no_uncommitted_harness_diff=False,
                          require_integrity_scan=False, **options)
    gate = SubmitGate(config, submissions_dir=root / 'submissions')
    gate._harbor_auth_ok = lambda: True
    return gate


def arguments(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return dict(campaign_id='test-campaign', best_job_dir=root, score=1.0,
                tasks_evaluated=1, full_regression_passed=True)


def test_two_threads_only_one_upload(tmp_path, monkeypatch):
    gate = gate_for(tmp_path)
    args = arguments(tmp_path / 'job')
    barrier = threading.Barrier(2, timeout=5)
    original = gate.check
    def check(**kwargs):
        result = original(**kwargs)
        barrier.wait()
        return result
    gate.check = check
    calls = []
    def upload(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '', '')
    monkeypatch.setattr(module.subprocess, 'run', upload)
    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(lambda _: gate.submit_once(**args), range(2)))
    assert len(calls) == 1
    assert sum(result.submitted for result in outcomes) == 1
    assert sum(result.attempted for result in outcomes) == 1
    assert json.loads((tmp_path / 'submissions/test-campaign.intent.json').read_text())['campaign_id'] == 'test-campaign'


@pytest.mark.skipif(os.name != 'posix', reason='fork fixture')
def test_two_processes_only_one_upload(tmp_path, monkeypatch):
    context = multiprocessing.get_context('fork')
    gate = gate_for(tmp_path)
    args = arguments(tmp_path / 'job')
    barrier = context.Barrier(2, timeout=5)
    calls = context.Value('i', 0)
    original = gate.check
    def check(**kwargs):
        result = original(**kwargs)
        barrier.wait()
        return result
    gate.check = check
    def upload(*args, **kwargs):
        with calls.get_lock():
            calls.value += 1
        return subprocess.CompletedProcess(args, 0, '', '')
    monkeypatch.setattr(module.subprocess, 'run', upload)
    children = [context.Process(target=gate.submit_once, kwargs=args) for _ in range(2)]
    try:
        for child in children:
            child.start()
        for child in children:
            child.join(8)
            assert not child.is_alive()
            assert child.exitcode == 0
        assert calls.value == 1
    finally:
        for child in children:
            if child.is_alive(): child.kill()
            child.join(2)


@pytest.mark.parametrize('existing', [b'', b'partial', b'{"owner":"other"}'])
def test_existing_intent_never_overwritten(tmp_path, monkeypatch, existing):
    gate = gate_for(tmp_path)
    marker = gate.submissions_dir / 'test-campaign.intent.json'
    marker.write_bytes(existing)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: pytest.fail('must not upload'))
    result = gate.submit_once(**arguments(tmp_path / 'job'))
    assert not result.eligible and not result.attempted
    assert marker.read_bytes() == existing


@pytest.mark.parametrize('failure_call', [1, 2])
def test_intent_fsync_failure_preserves_claim_and_prevents_upload(tmp_path, monkeypatch, failure_call):
    gate = gate_for(tmp_path)
    args = arguments(tmp_path / 'job')
    original = os.fsync
    calls = 0
    def fail(fd):
        nonlocal calls
        calls += 1
        if calls == failure_call: raise OSError('injected fsync failure')
        return original(fd)
    monkeypatch.setattr(os, 'fsync', fail)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: pytest.fail('must not upload'))
    result = gate.submit_once(**args)
    assert not result.eligible and not result.attempted
    assert Path(result.intent_path).exists()
    assert 'durably persisted' in result.reasons[0]
    assert not gate.submit_once(**args).eligible


def test_upload_error_is_uncertain_and_not_retried(tmp_path, monkeypatch):
    gate = gate_for(tmp_path)
    args = arguments(tmp_path / 'job')
    def fail(*a, **k): raise OSError('transport interrupted')
    monkeypatch.setattr(module.subprocess, 'run', fail)
    result = gate.submit_once(**args)
    assert result.intent_persisted and result.attempted and result.outcome_unknown
    assert not result.submitted
    assert Path(result.intent_path).exists()
    assert not gate.submit_once(**args).eligible


def test_result_failure_does_not_undo_successful_upload(tmp_path, monkeypatch):
    gate = gate_for(tmp_path)
    args = arguments(tmp_path / 'job')
    original = module.write_exclusive_json
    def write(path, payload):
        if not path.name.endswith('.intent.json'): raise OSError('result disk failure')
        return original(path, payload)
    monkeypatch.setattr(module, 'write_exclusive_json', write)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, '', ''))
    result = gate.submit_once(**args)
    assert result.submitted and result.result_persistence_failed
    assert Path(result.intent_path).exists()
    assert not gate.submit_once(**args).eligible


def test_dry_run_and_disabled_do_not_claim(tmp_path):
    gate = gate_for(tmp_path)
    args = arguments(tmp_path / 'job')
    assert gate.submit_once(**args, dry_run=True).eligible
    assert not list(gate.submissions_dir.iterdir())
    gate.config.enabled = False
    assert not gate.submit_once(**args).eligible
    assert not list(gate.submissions_dir.iterdir())


def test_no_upload_still_records_explicit_skip(tmp_path):
    gate = gate_for(tmp_path, harbor_upload=False)
    result = gate.submit_once(**arguments(tmp_path / 'job'))
    assert result.upload_skipped and result.attempted and not result.submitted
    assert result.intent_persisted


@pytest.mark.parametrize('campaign', ['../escape', '/absolute', '.', '', 'a/b', 'x' * 129, 42])
def test_unsafe_campaign_cannot_choose_claim_path(tmp_path, campaign):
    gate = gate_for(tmp_path)
    args = arguments(tmp_path / 'job')
    args['campaign_id'] = campaign
    result = gate.submit_once(**args)
    assert not result.eligible and not result.attempted
    assert not list(gate.submissions_dir.iterdir())


def test_exclusive_json_is_owner_only_and_cannot_replace_symlink(tmp_path):
    path = tmp_path / 'claim.json'
    write_exclusive_json(path, {'ok': True})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError): write_exclusive_json(path, {'other': True})
    dangling = tmp_path / 'dangling.json'
    dangling.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError): write_exclusive_json(dangling, {'other': True})
    assert dangling.is_symlink()


@pytest.mark.skipif(os.name != 'posix', reason='process-exit fixture')
def test_exit_after_intent_does_not_allow_resubmission(tmp_path):
    gate = gate_for(tmp_path)
    args = arguments(tmp_path / 'job')
    code = '''from pathlib import Path
from hl.submission_storage import write_exclusive_json
import os, sys
write_exclusive_json(Path(sys.argv[1]), {"phase": "intent"})
os._exit(0)
'''
    subprocess.run([os.sys.executable, '-c', code, str(gate.submissions_dir / 'test-campaign.intent.json')], check=True, timeout=5)
    result = gate.submit_once(**args)
    assert not result.eligible and not result.attempted
