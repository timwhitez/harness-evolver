from __future__ import annotations

import os
from pathlib import Path
import signal
import socket
import stat
import sys

import pytest

from harness.tools import descriptor_open, safe_path_io, bounded_path_io, stable_tree

pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux O_PATH/procfs security boundary')


@pytest.fixture(autouse=True)
def outer_deadline():
    def expired(*_): raise AssertionError('descriptor acquisition blocked')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 3)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def fds(): return len(list(Path('/proc/self/fd').iterdir()))


@pytest.mark.parametrize('operation', ['binary', 'text', 'expected', 'identity', 'edit', 'traverse'])
def test_fifo_never_waits_for_a_writer(tmp_path, operation):
    path = tmp_path / 'pipe'
    os.mkfifo(path)
    baseline = fds()
    before = path.lstat()
    with pytest.raises((OSError, ValueError)):
        if operation == 'binary':
            with bounded_path_io.open_binary_nofollow(path): pytest.fail('FIFO accepted')
        elif operation == 'text': safe_path_io.read_text_nofollow(path)
        elif operation == 'edit': safe_path_io.edit_text_nofollow(path, lambda s: s+'x')
        elif operation == 'traverse': list(stable_tree.iter_stable_regular_files(path))
        else:
            with safe_path_io._open_parent_nofollow(path, create_parents=False) as (parent, name, target):
                if operation == 'expected':
                    safe_path_io._open_and_validate_expected(parent,name,target,expected_identity=(before.st_dev,before.st_ino),expected_sha256=None)
                else: safe_path_io._assert_name_identity(parent,name,(before.st_dev,before.st_ino),target)
    assert path.lstat().st_ino == before.st_ino and stat.S_ISFIFO(path.lstat().st_mode)
    assert fds() == baseline


@pytest.mark.parametrize('kind', ['fifo', 'socket', 'directory', 'symlink'])
def test_unvalidated_special_inode_is_never_read_opened(tmp_path, monkeypatch, kind):
    target = tmp_path / 'entry'
    server = None
    if kind == 'fifo': os.mkfifo(target)
    elif kind == 'socket':
        server = socket.socket(socket.AF_UNIX); server.bind(str(target))
    elif kind == 'directory': target.mkdir()
    else:
        ordinary = tmp_path/'other';ordinary.write_text('text');target.symlink_to(ordinary)
    original = os.open
    readable_opens = []
    def track(path, flags, *a, **kw):
        if not flags & os.O_PATH and not flags & os.O_DIRECTORY: readable_opens.append(str(path))
        return original(path,flags,*a,**kw)
    parent = original(tmp_path,os.O_RDONLY|os.O_DIRECTORY)
    try:
        monkeypatch.setattr(os, 'open',track)
        with pytest.raises(ValueError,match='regular'):
            descriptor_open.open_readonly_checked(parent,target.name)
        assert readable_opens == []
    finally:
        os.close(parent)
        if server is not None: server.close()


def test_character_device_is_only_opened_as_opath(monkeypatch):
    # No read or write open is performed against /dev/null.
    parent=os.open('/dev',os.O_RDONLY|os.O_DIRECTORY)
    original=os.open
    observed=[]
    def track(path,flags,*a,**kw):
        observed.append((path,flags)); return original(path,flags,*a,**kw)
    try:
        monkeypatch.setattr(os,'open',track)
        with pytest.raises(ValueError,match='regular'):
            descriptor_open.open_readonly_checked(parent,'null')
        assert len(observed)==1 and observed[0][1]&os.O_PATH
    finally: os.close(parent)


def test_regular_to_fifo_swap_after_pin_does_not_redirect_read(tmp_path, monkeypatch):
    target=tmp_path/'file'; target.write_text('original')
    saved=tmp_path/'retained-original'
    original=os.open
    swapped=False
    def swap(path, flags, *a, **kw):
        nonlocal swapped
        fd=original(path,flags,*a,**kw)
        if path == 'file' and flags & os.O_PATH and not swapped:
            target.rename(saved);os.mkfifo(target);swapped=True
        return fd
    monkeypatch.setattr(os,'open',swap)
    with bounded_path_io.open_binary_nofollow(target) as (stream, metadata):
        assert stream.read()==b'original'
        assert metadata.st_ino==saved.stat().st_ino
    assert swapped and stat.S_ISFIFO(target.lstat().st_mode)


def test_regular_to_fifo_swap_before_pin_is_rejected(tmp_path,monkeypatch):
    target=tmp_path/'file';target.write_text('original')
    original=os.open
    def swap(path,flags,*a,**kw):
        if path=='file' and flags & os.O_PATH:
            target.unlink();os.mkfifo(target)
        return original(path,flags,*a,**kw)
    monkeypatch.setattr(os,'open',swap)
    with pytest.raises((OSError,ValueError)):
        with bounded_path_io.open_binary_nofollow(target): pytest.fail('FIFO accepted')


def test_proc_reopen_failure_does_not_leak_pin(tmp_path,monkeypatch):
    path=tmp_path/'file';path.write_text('hello')
    baseline=fds(); original=os.open
    def fail(path,flags,*a,**kw):
        if str(path).startswith('/proc/self/fd/'):raise OSError('procfs unavailable')
        return original(path,flags,*a,**kw)
    monkeypatch.setattr(os,'open',fail)
    with pytest.raises(OSError,match='procfs'):
        with bounded_path_io.open_binary_nofollow(path): pass
    assert fds()==baseline


def test_fdopen_failure_closes_returned_read_descriptor(tmp_path,monkeypatch):
    path=tmp_path/'file';path.write_text('hello');baseline=fds()
    def fail(*a,**kw): raise OSError('stream creation fault')
    monkeypatch.setattr(os,'fdopen',fail)
    with pytest.raises(OSError,match='stream creation'):
        with bounded_path_io.open_binary_nofollow(path): pass
    assert fds()==baseline


def test_unique_link_and_regular_bytes_contract(tmp_path):
    path=tmp_path/'text';path.write_bytes('a\r\né'.encode())
    assert safe_path_io.read_text_nofollow(path,errors='strict')[0]=='a\r\né'
    assert bounded_path_io.read_bounded_bytes_nofollow(path,max_bytes=2)[0]==b'a\r'
    other=tmp_path/'link';os.link(path,other)
    with pytest.raises((ValueError,OSError),match='linked'):
        with bounded_path_io.open_binary_nofollow(path): pass


def test_pin_close_failure_does_not_orphan_readable_descriptor(tmp_path,monkeypatch):
    target=tmp_path/'file';target.write_text('hello')
    parent=os.open(tmp_path,os.O_RDONLY|os.O_DIRECTORY);baseline=fds()
    original=os.close
    calls=0
    def close(fd):
        nonlocal calls
        calls+=1
        original(fd)
        if calls==1:raise OSError('pin close fault')
    try:
        with monkeypatch.context() as patch:
            patch.setattr(os,'close',close)
            with pytest.raises(OSError,match='pin close fault'):
                descriptor_open.open_readonly_checked(parent,target.name)
        assert calls==2 and fds()==baseline
    finally:original(parent)
