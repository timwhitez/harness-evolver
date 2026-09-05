from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

fcntl = pytest.importorskip("fcntl")

from harness.tools.publication import publish_bytes, embedded_publication_source
from harness.tools.safe_path_io import _renameat2


pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux local-filesystem publication')


@pytest.fixture(params=['host', 'embedded'])
def publish(request, tmp_path):
    function = publish_bytes
    if request.param == 'embedded':
        namespace = {}
        exec(embedded_publication_source(), namespace)
        function = namespace['publish_bytes']
    def call(name, data, rename=_renameat2, **kwargs):
        fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            return function(fd, name, data, rename, **kwargs)
        finally:
            os.close(fd)
    return call


def files(root):
    return sorted(p.name for p in root.iterdir())


def identity(path):
    m = path.lstat()
    return m.st_dev, m.st_ino


def test_create_overwrite_and_permissions(publish, tmp_path):
    previous = os.umask(0o027)
    try:
        created = publish('file', b'old')
    finally:
        os.umask(previous)
    assert created['atomic_replace'] is True and created['directory_fsync'] is True
    path = tmp_path / 'file'
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    path.chmod(0o4755)
    replaced = publish('file', b'new')
    assert replaced['publication_state'] == 'published'
    assert path.read_bytes() == b'new'
    assert stat.S_IMODE(path.stat().st_mode) == 0o755
    assert files(tmp_path) == ['file']


@pytest.mark.parametrize('kind', ['fifo', 'socket', 'directory', 'symlink'])
def test_special_targets_are_untouched_before_claim(publish, tmp_path, kind):
    path = tmp_path / 'endpoint'
    server = None
    if kind == 'fifo': os.mkfifo(path)
    elif kind == 'directory': path.mkdir()
    elif kind == 'symlink': path.symlink_to(tmp_path / 'absent')
    else:
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(path))
    try:
        before = path.lstat()
        result = publish('endpoint', b'candidate')
        assert result['atomic_replace'] is False
        assert result['publication_state'] == 'not_published'
        assert path.lstat().st_ino == before.st_ino
        assert path.lstat().st_mode == before.st_mode
        assert files(tmp_path) == ['endpoint']
    finally:
        if server: server.close()


def test_hardlink_pure_overwrite_dealiases_without_reading_old_content(publish, tmp_path, monkeypatch):
    path = tmp_path / 'file'
    path.write_bytes(b'old')
    os.link(path, tmp_path / 'sibling')
    with monkeypatch.context() as patch:
        patch.setattr(os, 'read', lambda *a: pytest.fail('pure overwrite must not read aliased bytes'))
        result = publish('file', b'new')
    assert result['atomic_replace'] is True
    assert (tmp_path / 'sibling').read_bytes() == b'old'
    assert path.read_bytes() == b'new'
    assert identity(path) != identity(tmp_path / 'sibling')


def test_transform_requires_exact_source_and_unique_link(publish, tmp_path):
    path = tmp_path / 'file'
    path.write_bytes(b'old')
    expected = identity(path)
    digest = hashlib.sha256(b'old').hexdigest()
    valid = publish('file', b'old+append', expected_identity=expected, expected_sha256=digest)
    assert valid['atomic_replace'] is True
    rejected = publish('file', b'wrong', expected_identity=expected, expected_sha256=digest)
    assert rejected['atomic_replace'] is False
    assert path.read_bytes() == b'old+append'
    os.link(path, tmp_path / 'sibling')
    linked = publish('file', b'wrong', expected_identity=identity(path))
    assert linked['atomic_replace'] is False and 'linked' in linked['publication_error']


@pytest.mark.parametrize('at', [1, 2, 3])
def test_prepublication_fsync_failure_preserves_old_bytes(publish, tmp_path, monkeypatch, at):
    path = tmp_path / 'file'; path.write_bytes(b'old')
    original = os.fsync
    count = 0
    def sync(fd):
        nonlocal count
        count += 1
        if count == at: raise OSError('prepublication fsync fault')
        return original(fd)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'fsync', sync)
        result = publish('file', b'new')
    assert result['atomic_replace'] is False
    assert path.read_bytes() == b'old'
    assert files(tmp_path) == ['file']


@pytest.mark.parametrize('at', [4, 5, 6])
def test_postpublication_fsync_failure_is_not_uncommitted(publish, tmp_path, monkeypatch, at):
    path = tmp_path / 'file'; path.write_bytes(b'old')
    original = os.fsync
    count = 0
    def sync(fd):
        nonlocal count
        count += 1
        if count == at: raise OSError('postpublication fsync fault')
        return original(fd)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'fsync', sync)
        result = publish('file', b'new')
    assert result['atomic_replace'] is True
    assert result['publication_state'] == 'published'
    assert result['durability_warning'] and result['no_auto_retry']
    assert path.read_bytes() == b'new'


@pytest.mark.parametrize('target', ['data', 'intent'])
def test_postpublication_unlink_failure_reports_commit_and_blocks_retry(publish, tmp_path, monkeypatch, target):
    path = tmp_path / 'file'; path.write_bytes(b'old')
    original = os.unlink
    def unlink(name, *args, **kwargs):
        if str(name).endswith('.' + target): raise OSError('injected cleanup fault')
        return original(name, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'unlink', unlink)
        result = publish('file', b'old+once')
    assert result['atomic_replace'] is True and result['cleanup_warning']
    assert result['recovery_entries'] and result['no_auto_retry']
    assert path.read_bytes() == b'old+once'
    again = publish('file', b'old+once+twice')
    assert again['atomic_replace'] is False and again['no_auto_retry']
    assert 'unreconciled' in again['publication_error']
    assert path.read_bytes() == b'old+once'


def test_two_external_updates_cannot_be_deleted_by_rollback(publish, tmp_path):
    path = tmp_path / 'file'; path.write_bytes(b'old')
    calls = 0
    def interleave(parent, source, destination, flags):
        nonlocal calls
        calls += 1
        assert calls == 1, 'must not perform a second/rollback exchange'
        external = tmp_path / 'external'
        external.write_bytes(b'concurrent-A')
        os.replace(external, path)
        _renameat2(parent, source, destination, flags)
        external.write_bytes(b'concurrent-B')
        os.replace(external, path)
    result = publish('file', b'candidate', rename=interleave)
    assert result['publication_state'] == 'indeterminate'
    assert result['atomic_replace'] is None and result['no_auto_retry']
    assert path.read_bytes() == b'concurrent-B'
    backups = list(tmp_path.glob('*.data'))
    assert len(backups) == 1 and backups[0].read_bytes() == b'concurrent-A'
    assert any(name.endswith('.intent') for name in result['recovery_entries'])


def test_visible_candidate_is_never_reported_as_untouched_rejection(publish, tmp_path):
    path = tmp_path / 'file'; path.write_bytes(b'old')
    observed = []
    def interleave(parent, source, destination, flags):
        external = tmp_path / 'external'; external.write_bytes(b'other')
        os.replace(external, path)
        _renameat2(parent, source, destination, flags)
        observed.append(path.read_bytes())
    result = publish('file', b'candidate', rename=interleave)
    assert observed == [b'candidate']
    assert result['publication_state'] == 'indeterminate'
    assert result['atomic_replace'] is None
    assert path.read_bytes() == b'candidate'


def test_missing_target_appearance_uses_noreplace(publish, tmp_path):
    path = tmp_path / 'file'
    def interleave(parent, source, destination, flags):
        assert flags == 1
        os.mkfifo(path)
        return _renameat2(parent, source, destination, flags)
    result = publish('file', b'new', rename=interleave)
    assert result['atomic_replace'] is False
    assert stat.S_ISFIFO(path.lstat().st_mode)
    assert files(tmp_path) == ['file']


def test_cooperating_transforms_do_not_silently_lose_updates(publish, tmp_path):
    path = tmp_path / 'file'; path.write_bytes(b'old')
    expected = identity(path); digest = hashlib.sha256(b'old').hexdigest()
    barrier = threading.Barrier(2, timeout=3)
    def worker(index):
        barrier.wait()
        return publish('file', f'old+{index}'.encode(), expected_identity=expected, expected_sha256=digest)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(worker, [1, 2]))
    assert sum(r['atomic_replace'] is True for r in results) == 1
    assert sum(r['atomic_replace'] is False for r in results) == 1
    assert path.read_bytes() in (b'old+1', b'old+2')
    assert files(tmp_path) == ['file']


def test_busy_directory_returns_bounded_prepublication_failure(publish, tmp_path):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        result = publish('file', b'new', lock_timeout=.05)
    finally:
        os.close(fd)
    assert result['atomic_replace'] is False
    assert 'busy' in result['publication_error'] and files(tmp_path) == []


def test_crash_after_exchange_retains_recovery_and_blocks_blind_retry(tmp_path):
    (tmp_path / 'file').write_bytes(b'old')
    program = '''import os,sys
from harness.tools.publication import publish_bytes
from harness.tools.safe_path_io import _renameat2
parent=os.open(sys.argv[1],os.O_RDONLY|os.O_DIRECTORY)
def crash(parent,source,destination,flags):
    _renameat2(parent,source,destination,flags)
    os._exit(0)
publish_bytes(parent,'file',b'candidate',crash)
'''
    subprocess.run([sys.executable, '-c', program, str(tmp_path)], check=True, timeout=5)
    assert (tmp_path / 'file').read_bytes() == b'candidate'
    assert [p.read_bytes() for p in tmp_path.glob('*.data')] == [b'old']
    journal = next(tmp_path.glob('*.intent'))
    assert json.loads(journal.read_text())['target'] == 'file'
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        result = publish_bytes(fd, 'file', b'blind retry', _renameat2)
    finally:
        os.close(fd)
    assert result['atomic_replace'] is False and result['no_auto_retry']
    assert (tmp_path / 'file').read_bytes() == b'candidate'
