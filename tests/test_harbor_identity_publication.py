from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys

import pytest

from bench import _canonical_harbor_identity_guard as identity_guard
from bench.harbor_adapter import HarborFileEditTool, HarborFileWriteTool
from harness.tools.base import ToolResult

pytestmark=pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux Harbor publication')


def _outcome(directory_synced=True):
    return dict(publication_protocol='hl-publication-v1',publication_state='published',atomic_replace=True,
        directory_fsync=directory_synced,durability_warning=not directory_synced,cleanup_warning=False,
        no_auto_retry=True,recovery_entries=[],publication_error='')


def _response(directory_synced=True):
    return 'write complete\n__HL_PUBLICATION__'+json.dumps(_outcome(directory_synced))+'\n'


def _run_script(script,env,wrapper=''):
    return subprocess.run([sys.executable,'-c',wrapper+script],env={**os.environ,**env},capture_output=True,text=True,timeout=3)


def _metadata(completed):
    result=ToolResult(completed.returncode==0,completed.stdout,error=completed.stderr,metadata={'publication_attempted':True})
    return identity_guard._publication_metadata(result,identity_verified=True)


def _capturing_append_tool(snapshot,captured):
    tool=object.__new__(HarborFileWriteTool)
    tool._guard_environment_path=lambda path,**_: (path,None)
    tool._secure_snapshot=lambda path:(snapshot,None)
    def capture(script,*,env):
        captured.update(script=script,env=env)
        return ToolResult(True,_response(),metadata={'exit_code':0})
    tool._run_secure_python=capture
    return tool


def test_public_append_forwards_existing_snapshot_identity_and_digest():
    captured={};snapshot=identity_guard._Snapshot(True,'existing\n',12,34,hashlib.sha256(b'existing\n').hexdigest())
    result=_capturing_append_tool(snapshot,captured).execute('/workspace/target.txt','tail\n',append=True)
    assert result.success
    env=captured['env']
    assert env['HL_EXPECTED_PRESENT']=='1' and env['HL_EXPECTED_DEV']=='12' and env['HL_EXPECTED_INO']=='34'
    assert env['HL_EXPECTED_SHA256']==snapshot.sha256
    assert base64.b64decode(env['HL_FILE_CONTENT'])==b'existing\ntail\n'
    assert result.metadata['target_identity_verified'] and result.metadata['directory_fsync'] and result.metadata['atomic_append']


def test_public_append_keeps_missing_snapshot_expectation_disjoint():
    captured={};snapshot=identity_guard._Snapshot(False,'',None,None,hashlib.sha256(b'').hexdigest())
    result=_capturing_append_tool(snapshot,captured).execute('/workspace/target.txt','tail\n',append=True)
    assert result.success
    env=captured['env'];assert env['HL_EXPECTED_PRESENT']=='0'
    assert not {'HL_EXPECTED_DEV','HL_EXPECTED_INO','HL_EXPECTED_SHA256'} & env.keys()
    assert base64.b64decode(env['HL_FILE_CONTENT'])==b'tail\n'


def test_append_script_rejects_replaced_inode(tmp_path):
    target=tmp_path/'target.txt';target.write_text('old\n');old=target.stat();digest=hashlib.sha256(target.read_bytes()).hexdigest()
    moved=tmp_path/'old.txt';target.replace(moved);target.write_text('concurrent\n')
    completed=_run_script(identity_guard._v3._SECURE_ATOMIC_WRITE,{
        'HL_FILE_PATH':str(target),'HL_FILE_CONTENT':base64.b64encode(b'old\ntail\n').decode(),
        'HL_EXPECTED_PRESENT':'1','HL_EXPECTED_DEV':str(old.st_dev),'HL_EXPECTED_INO':str(old.st_ino),
        'HL_EXPECTED_SHA256':digest})
    assert completed.returncode!=0 and _metadata(completed).metadata['atomic_replace'] is False
    assert target.read_text()=='concurrent\n' and moved.read_text()=='old\n'
    assert not list(tmp_path.glob('.hl-publish-*'))


def test_append_script_rejects_target_appearing_after_missing_snapshot(tmp_path):
    target=tmp_path/'target.txt';target.write_text('concurrent\n')
    completed=_run_script(identity_guard._v3._SECURE_ATOMIC_WRITE,{'HL_FILE_PATH':str(target),
        'HL_FILE_CONTENT':base64.b64encode(b'tail\n').decode(),'HL_EXPECTED_PRESENT':'0'})
    assert completed.returncode!=0 and target.read_text()=='concurrent\n'
    assert not list(tmp_path.glob('.hl-publish-*'))


def test_append_script_creates_target_when_missing_snapshot_remains_missing(tmp_path):
    target=tmp_path/'target.txt'
    completed=_run_script(identity_guard._v3._SECURE_ATOMIC_WRITE,{'HL_FILE_PATH':str(target),
        'HL_FILE_CONTENT':base64.b64encode(b'tail\n').decode(),'HL_EXPECTED_PRESENT':'0'})
    assert completed.returncode==0 and target.read_bytes()==b'tail\n'
    assert _metadata(completed).metadata['directory_fsync']


def test_harbor_directory_fsync_failure_is_success_with_warning(tmp_path):
    target=tmp_path/'target.txt';target.write_text('old\n')
    wrapper='''import os,stat
_real_fsync=os.fsync
_directory_syncs=0
def fail(fd):
    global _directory_syncs
    if stat.S_ISDIR(os.fstat(fd).st_mode):
        _directory_syncs+=1
        if _directory_syncs==2:raise OSError('injected directory fsync failure')
    return _real_fsync(fd)
os.fsync=fail
'''
    completed=_run_script(identity_guard._v3._SECURE_ATOMIC_WRITE,{'HL_FILE_PATH':str(target),
        'HL_FILE_CONTENT':base64.b64encode(b'new\n').decode()},wrapper)
    parsed=_metadata(completed)
    assert completed.returncode==0 and target.read_text()=='new\n'
    assert parsed.success and not parsed.metadata['directory_fsync'] and parsed.metadata['durability_warning']


def test_public_edit_reports_post_publication_durability_warning(monkeypatch):
    parent=identity_guard._base.HarborFileEditTool
    monkeypatch.setattr(parent,'execute',lambda *a,**kw:ToolResult(True,_response(False)))
    result=object.__new__(HarborFileEditTool).execute('/workspace/target.txt','old','new')
    assert result.success and not result.metadata['directory_fsync'] and result.metadata['durability_warning']
    assert '__HL_PUBLICATION__' not in result.output and 'published' in result.output


def test_public_registry_uses_identity_bound_harbor_tools():
    from bench.harbor_adapter import HLWorkerHarborAgent
    agent=object.__new__(HLWorkerHarborAgent);agent.tool_timeout_seconds=1.0;agent._goal_path=lambda:None
    registry=agent._build_environment_registry(object(),object())
    assert isinstance(registry.get('write'),HarborFileWriteTool)
    assert isinstance(registry.get('edit'),HarborFileEditTool)
