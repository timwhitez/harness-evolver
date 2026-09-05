from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys

import pytest

import harness.tools.file_edit as file_edit_module
import harness.tools.safe_path_io as safe_path_io
from harness.tools.file_edit import FileEditTool
from harness.tools.file_write import FileWriteTool
from harness.tools.safe_path_io import SafePathError, atomic_write_text_nofollow, file_identity, read_text_nofollow

pytestmark=pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux serialized publication')


def _temps(target): return list(target.parent.glob('.hl-publish-*'))


def test_transform_publication_rejects_replaced_inode(tmp_path):
    target=tmp_path/'target.txt';target.write_text('old\n')
    original,metadata=read_text_nofollow(target,errors='strict')
    moved=tmp_path/'original.txt';target.replace(moved);target.write_text('concurrent\n')
    with pytest.raises(SafePathError,match='changed identity'):
        atomic_write_text_nofollow(target,'updated\n',mode=metadata.st_mode,
            expected_identity=file_identity(metadata),expected_sha256=hashlib.sha256(original.encode()).hexdigest())
    assert moved.read_text()=='old\n' and target.read_text()=='concurrent\n'
    assert _temps(target)==[]


def test_transform_publication_rejects_same_inode_content_change(tmp_path):
    target=tmp_path/'target.txt';target.write_text('old\n')
    original,metadata=read_text_nofollow(target,errors='strict');before=file_identity(metadata)
    target.write_text('concurrent\n');assert file_identity(target.stat())==before
    with pytest.raises(SafePathError,match='content changed'):
        atomic_write_text_nofollow(target,'updated\n',mode=metadata.st_mode,expected_identity=before,
            expected_sha256=hashlib.sha256(original.encode()).hexdigest())
    assert target.read_text()=='concurrent\n' and _temps(target)==[]


def test_exchange_race_retains_displaced_inode_instead_of_destructive_rollback(tmp_path,monkeypatch):
    target=tmp_path/'target.txt';target.write_text('old\n')
    original,metadata=read_text_nofollow(target,errors='strict')
    moved=tmp_path/'original.txt';real=safe_path_io._renameat2;calls=[]
    def racing(parent,source,destination,flags):
        calls.append(flags)
        target.replace(moved);target.write_text('concurrent\n')
        real(parent,source,destination,flags)
    monkeypatch.setattr(safe_path_io,'_renameat2',racing)
    with pytest.raises(SafePathError,match='changed identity') as caught:
        atomic_write_text_nofollow(target,'updated\n',mode=metadata.st_mode,
            expected_identity=file_identity(metadata),expected_sha256=hashlib.sha256(original.encode()).hexdigest())
    outcome=caught.value.publication_outcome
    assert calls==[2] and outcome['atomic_replace'] is None and outcome['no_auto_retry']
    assert moved.read_text()=='old\n' and target.read_text()=='updated\n'
    retained=[tmp_path/e for e in outcome['recovery_entries']]
    assert next(p for p in retained if p.suffix=='.data').read_text()=='concurrent\n'
    assert any(p.suffix=='.intent' and p.exists() for p in retained)


def test_file_edit_passes_identity_and_digest_to_publication(tmp_path,monkeypatch):
    target=tmp_path/'target.txt';target.write_text('old value\n')
    moved=tmp_path/'original.txt';real=safe_path_io.publish_text_nofollow;captured={}
    def racing(path,content,**kwargs):
        captured.update(kwargs);target.replace(moved);target.write_text('concurrent value\n')
        return real(path,content,**kwargs)
    monkeypatch.setattr(file_edit_module,'publish_text_nofollow',racing)
    result=FileEditTool().execute(str(target),'old','new')
    assert not result.success and result.metadata['atomic_replace'] is False
    assert result.metadata['target_identity_verified'] is False
    assert captured['expected_identity']==file_identity(moved.stat())
    assert captured['expected_sha256']==hashlib.sha256(b'old value\n').hexdigest()
    assert target.read_text()=='concurrent value\n' and moved.read_text()=='old value\n'


def test_crlf_edit_uses_the_exact_raw_byte_digest(tmp_path):
    target=tmp_path/'target.txt';target.write_bytes(b'old value\r\nsecond line\r\n')
    result=FileEditTool().execute(str(target),'old','new')
    assert result.success and target.read_bytes()==b'new value\r\nsecond line\r\n'
    assert result.metadata['target_identity_verified'] is True


def test_directory_fsync_failure_after_publication_is_warning(tmp_path,monkeypatch):
    target=tmp_path/'target.txt';target.write_text('old\n');real=os.fsync;calls=0
    def fail(fd):
        nonlocal calls
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            calls+=1
            if calls==2: raise OSError('injected directory fsync failure')
        return real(fd)
    monkeypatch.setattr(os,'fsync',fail)
    result=FileWriteTool().execute(str(target),'new\n')
    assert result.success and target.read_text()=='new\n'
    assert result.metadata['atomic_replace'] is True
    assert result.metadata['directory_fsync'] is False and result.metadata['durability_warning']
    assert 'published' in result.output and result.metadata['no_auto_retry']


def test_directory_fsync_failure_before_publication_preserves_old_bytes(tmp_path,monkeypatch):
    target=tmp_path/'target.txt';target.write_text('old\n');real=os.fsync
    def fail(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):raise OSError('intent directory fsync failure')
        return real(fd)
    monkeypatch.setattr(os,'fsync',fail)
    result=FileWriteTool().execute(str(target),'new\n')
    assert not result.success and result.metadata['atomic_replace'] is False
    assert target.read_text()=='old\n'
