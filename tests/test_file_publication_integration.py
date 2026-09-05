from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from harness.tools.file_write import FileWriteTool
from harness.tools.file_edit import FileEditTool
from harness.tools import safe_path_io
from harness.tools.base import ToolResult
from harness.tools.canonical_path_guard import resolve_guarded_path, guarded_path_failure
from bench import _canonical_harbor_identity_guard as harbor
from bench import _canonical_harbor_special_write as special

pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux publication and trusted procfs')


@pytest.fixture(autouse=True)
def deadline():
    def expire(*_): raise AssertionError('public operation exceeded outer deadline')
    before = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, before)


def tool(cls, transform=lambda script: script):
    result = object.__new__(cls)
    def guard(path, operation, must_exist):
        decision=resolve_guarded_path(path,operation=operation,must_exist=must_exist)
        return (decision.resolved,None) if decision.allowed else ('',guarded_path_failure(operation,decision))
    result._guard_environment_path=guard
    def execute(command, *, env=None, **_):
        # Run the real generated script locally instead of a Docker SDK call.
        # Optionally inject a precise fault before executing its entry point.
        import shlex
        arguments=shlex.split(command)
        assert arguments[:2]==['python3','-c']
        completed=subprocess.run([sys.executable,'-c',transform(arguments[2])],
            env={**os.environ,**(env or {})},capture_output=True,text=True,timeout=3)
        return SimpleNamespace(return_code=completed.returncode,stdout=completed.stdout,stderr=completed.stderr)
    result._exec=execute
    return result


def test_harbor_compatibility_writer_is_single_implementation():
    assert special.HarborFileWriteTool is harbor.HarborFileWriteTool
    assert special.HarborFileEditTool is harbor.HarborFileEditTool
    assert special._SECURE_ATOMIC_WRITE is harbor._v3._SECURE_ATOMIC_WRITE


@pytest.mark.parametrize('kind',['local-write','local-edit','harbor-write','harbor-append','harbor-edit'])
def test_public_success_state_matches_bytes(tmp_path,kind):
    path=tmp_path/'file';path.write_text('old')
    if kind=='local-write': result=FileWriteTool().execute(str(path),'new')
    elif kind=='local-edit': result=FileEditTool().execute(str(path),'old','new')
    elif kind=='harbor-write': result=tool(special.HarborFileWriteTool).execute(str(path),'new')
    elif kind=='harbor-append': result=tool(special.HarborFileWriteTool).execute(str(path),'+new',append=True)
    else: result=tool(harbor.HarborFileEditTool).execute(str(path),'old','new')
    assert result.success, result.error
    assert path.read_text()==('old+new' if kind=='harbor-append' else 'new')
    assert result.metadata['atomic_replace'] is True
    assert result.metadata['publication_state']=='published'
    assert result.metadata['directory_fsync'] is True
    assert result.metadata['recovery_entries']==[]
    assert sorted(p.name for p in tmp_path.iterdir())==['file']


@pytest.mark.parametrize('kind',['local-edit','harbor-append','harbor-edit','harbor-read'])
def test_public_read_paths_reject_fifo_promptly(tmp_path,kind):
    path=tmp_path/'fifo';os.mkfifo(path);before=path.stat()
    if kind=='local-edit': result=FileEditTool().execute(str(path),'a','b')
    elif kind=='harbor-append': result=tool(special.HarborFileWriteTool).execute(str(path),'x',append=True)
    elif kind=='harbor-edit': result=tool(harbor.HarborFileEditTool).execute(str(path),'a','b')
    else: result=tool(harbor._v4.HarborFileReadTool).execute(str(path))
    assert result.success is False
    assert path.stat().st_ino==before.st_ino and stat.S_ISFIFO(path.stat().st_mode)
    assert len(list(tmp_path.iterdir()))==1


_UNLINK_FAULT = '''import os
_original_unlink = os.unlink
def fail_displaced(path,*a,**kw):
    if str(path).startswith('.hl-publish-') and str(path).endswith('.data'):
        raise OSError('injected old inode cleanup failure')
    return _original_unlink(path,*a,**kw)
os.unlink = fail_displaced
'''


@pytest.mark.parametrize('kind',['local-write','local-edit','harbor-write','harbor-append','harbor-edit'])
def test_public_cleanup_failure_is_a_published_write_not_a_retryable_failure(tmp_path,monkeypatch,kind):
    path=tmp_path/'file';path.write_text('old')
    original=os.unlink
    def fail(entry,*a,**kw):
        if str(entry).startswith('.hl-publish-') and str(entry).endswith('.data'):
            raise OSError('injected old inode cleanup failure')
        return original(entry,*a,**kw)
    if kind.startswith('local'):
        with monkeypatch.context() as patch:
            patch.setattr(os,'unlink',fail)
            result=(FileWriteTool().execute(str(path),'new') if kind=='local-write'
                    else FileEditTool().execute(str(path),'old','new'))
    else:
        adapter=tool(special.HarborFileWriteTool if kind!='harbor-edit' else harbor.HarborFileEditTool,
                     lambda script: _UNLINK_FAULT+script)
        result=(adapter.execute(str(path),'old','new') if kind=='harbor-edit'
                else adapter.execute(str(path),'+new' if kind=='harbor-append' else 'new',append=kind=='harbor-append'))
    assert result.success, result.error
    assert result.metadata['atomic_replace'] is True
    assert result.metadata['cleanup_warning'] is True
    assert result.metadata['no_auto_retry'] is True
    assert path.read_text()==('old+new' if kind=='harbor-append' else 'new')
    retained=[tmp_path/e for e in result.metadata['recovery_entries']]
    assert len(retained)==2 and all(p.exists() for p in retained)
    assert next(p for p in retained if p.suffix=='.data').read_text()=='old'
    retry=FileWriteTool().execute(str(path),'retry')
    assert not retry.success and retry.metadata['no_auto_retry']
    assert 'intent' in retry.error


@pytest.mark.parametrize('kind',['local','harbor'])
def test_public_two_writer_race_returns_indeterminate_and_preserves_both_updates(tmp_path,monkeypatch,kind):
    path=tmp_path/'file';path.write_text('original')
    def inject_script(script):
        # Inject immediately around the single publication exchange, not a
        # fake parser or outcome. No rollback exchange is permitted.
        needle='''            rename_at2(parent, temporary, name, 2)  # RENAME_EXCHANGE'''
        replacement='''            concurrent = '.concurrent-fixture'
            with open(concurrent, 'w', opener=lambda p,f: os.open(p,f,dir_fd=parent)) as f:
                f.write('concurrent-A')
            os.replace(concurrent, name, src_dir_fd=parent, dst_dir_fd=parent)
            rename_at2(parent, temporary, name, 2)  # RENAME_EXCHANGE
            with open(concurrent, 'w', opener=lambda p,f: os.open(p,f,dir_fd=parent)) as f:
                f.write('concurrent-B')
            os.replace(concurrent, name, src_dir_fd=parent, dst_dir_fd=parent)'''
        assert needle in script
        return script.replace(needle,replacement,1)
    if kind=='local':
        original=safe_path_io._renameat2;calls=[]
        def exchange(parent,source,dest,flags):
            calls.append(flags)
            replacement=tmp_path/'concurrent';replacement.write_text('concurrent-A');os.replace(replacement,path)
            original(parent,source,dest,flags)
            replacement.write_text('concurrent-B');os.replace(replacement,path)
        with monkeypatch.context() as patch:
            patch.setattr(safe_path_io,'_renameat2',exchange)
            result=FileWriteTool().execute(str(path),'candidate')
        assert calls==[2]
    else:
        result=tool(special.HarborFileWriteTool,inject_script).execute(str(path),'candidate')
    assert not result.success
    assert result.metadata['publication_state']=='indeterminate'
    assert result.metadata['atomic_replace'] is None
    assert result.metadata['no_auto_retry']
    assert path.read_text()=='concurrent-B'
    recovery=[tmp_path/e for e in result.metadata['recovery_entries']]
    assert next(p for p in recovery if p.suffix=='.data').read_text()=='concurrent-A'
    assert any(p.suffix=='.intent' and p.exists() for p in recovery)


@pytest.mark.parametrize('payload', ['', '__HL_PUBLICATION__[]', '__HL_PUBLICATION__{}', '__HL_PUBLICATION__null'])
def test_missing_or_malformed_attempted_outcome_is_not_claimed_uncommitted(payload):
    result=ToolResult(False,payload,error='transport interrupted',metadata={'publication_attempted':True})
    parsed=harbor._publication_metadata(result,identity_verified=False)
    assert not parsed.success
    assert parsed.metadata['atomic_replace'] is None and parsed.metadata['no_auto_retry']
    assert parsed.metadata['publication_state']=='indeterminate'


def test_prepublication_authorization_failure_is_not_mislabelled_unknown():
    before=ToolResult(False,'',error='policy block',metadata={'blocked_by':'canonical_path_guard'})
    assert harbor._publication_metadata(before,identity_verified=False) is before
    assert 'publication_state' not in before.metadata


def test_harbor_invalid_utf8_append_keeps_original_bytes(tmp_path):
    path=tmp_path/'file';path.write_bytes(b'\xffold')
    result=tool(special.HarborFileWriteTool).execute(str(path),'new',append=True)
    assert not result.success and result.metadata['text_decode_error']
    assert path.read_bytes()==b'\xffold'
    assert not list(tmp_path.glob('.hl-publish-*'))


def test_parent_close_error_cannot_turn_committed_write_into_uncommitted_failure(tmp_path,monkeypatch):
    target=tmp_path/'file';target.write_text('old')
    parent_identity=(tmp_path.stat().st_dev,tmp_path.stat().st_ino)
    real=os.close
    faulted=False
    def close(fd):
        nonlocal faulted
        metadata=os.fstat(fd)
        is_parent=(metadata.st_dev,metadata.st_ino)==parent_identity
        real(fd)
        if is_parent and not faulted:
            faulted=True
            raise OSError('injected parent close error')
    with monkeypatch.context() as patch:
        patch.setattr(os,'close',close)
        result=FileWriteTool().execute(str(target),'new')
    assert faulted and result.success and target.read_text()=='new'
    assert result.metadata['atomic_replace'] is True
    assert result.metadata['cleanup_warning'] and result.metadata['no_auto_retry']
