from __future__ import annotations

import argparse
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.submit_once import _attempts_per_task


@pytest.mark.parametrize('value', [True, False, 1.5, '1', None, 0, -1])
def test_cli_does_not_coerce_invalid_attempt_counts(value):
    args = SimpleNamespace(attempts_per_task_json=json.dumps({'task': value}))
    with pytest.raises(SystemExit) as caught:
        _attempts_per_task(args, argparse.ArgumentParser())
    assert caught.value.code == 2


def test_cli_accepts_exact_integer_evidence():
    args = SimpleNamespace(attempts_per_task_json='{"task": 5}')
    assert _attempts_per_task(args, argparse.ArgumentParser()) == {'task': 5}


def test_cli_nan_is_ineligible_without_upload(tmp_path):
    completed = subprocess.run([
        sys.executable, 'scripts/submit_once.py', '--campaign-id', 'nan-fixture',
        '--best-job-dir', str(tmp_path), '--score', 'nan', '--tasks-evaluated', '1',
        '--submissions-dir', str(tmp_path / 'records'), '--enabled', '--no-harbor-upload',
        '--min-tasks-evaluated', '1', '--min-attempts-per-task', '1',
        '--no-require-full-regression', '--no-require-clean-git',
        '--no-require-no-uncommitted-harness-diff', '--dry-run', '--json',
    ], capture_output=True, text=True, timeout=5)
    assert completed.returncode == 0  # dry-run retains its inspection exit contract
    payload = json.loads(completed.stdout)
    assert not payload['eligible'] and not payload['attempted']
    assert payload['evidence'] == {}
    assert not list((tmp_path / 'records').iterdir())


def test_cli_invalid_campaign_cannot_write_dry_run_outside_store(tmp_path):
    completed = subprocess.run([
        sys.executable, 'scripts/submit_once.py', '--campaign-id', '../escape',
        '--best-job-dir', str(tmp_path), '--score', '1', '--tasks-evaluated', '1',
        '--submissions-dir', str(tmp_path / 'records'), '--dry-run', '--record-dry-run',
    ], capture_output=True, text=True, timeout=5)
    assert completed.returncode == 2
    assert not (tmp_path / 'escape.dry_run.json').exists()
