from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

import pytest

from bench._canonical_harbor_special_write import HarborFileWriteTool as GuardedHarborFileWriteTool, _SECURE_ATOMIC_WRITE
from bench.harbor_adapter import HarborFileWriteTool
from harness.tools import safe_path_io
from harness.tools.file_write import FileWriteTool
from harness.tools.safe_path_io import SafePathError, atomic_write_text_nofollow

pytestmark=pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux publication')


def _identity(path):
    m=path.lstat();return m.st_dev,m.st_ino,stat.S_IFMT(m.st_mode)


def _temporary_paths(path):return list(path.parent.glob('.hl-publish-*'))


def _run_harbor_write(path,content,script=_SECURE_ATOMIC_WRITE):
    return subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=3,
        env={**os.environ,'HL_FILE_PATH':str(path),'HL_FILE_CONTENT':base64.b64encode(content.encode()).decode()})


def test_public_harbor_registry_exports_the_guarded_writer():
    assert HarborFileWriteTool is GuardedHarborFileWriteTool


@pytest.mark.parametrize('engine',['local','harbor'])
@pytest.mark.parametrize('kind',['fifo','socket'])
def test_static_special_entries_are_unchanged_before_publication(tmp_path,engine,kind):
    path=tmp_path/'endpoint';server=None
    if kind=='fifo':os.mkfifo(path)
    else:server=socket.socket(socket.AF_UNIX);server.bind(str(path))
    try:
        before=_identity(path)
        if engine=='local':assert not FileWriteTool().execute(str(path),'replacement').success
        else:assert _run_harbor_write(path,'replacement').returncode!=0
        assert _identity(path)==before and not _temporary_paths(path)
    finally:
        if server is not None:server.close()


def test_character_device_type_is_rejected_without_touching_a_real_device(tmp_path,monkeypatch):
    path=tmp_path/'fixture';path.write_text('unchanged');real=os.stat
    def special(name,*a,**kw):
        observed=real(name,*a,**kw)
        if name=='fixture' and kw.get('dir_fd') is not None:
            values=list(observed);values[0]=stat.S_IFCHR|0o600;return os.stat_result(values)
        return observed
    monkeypatch.setattr(os,'stat',special)
    with pytest.raises(SafePathError,match='non-regular overwrite target'):
        atomic_write_text_nofollow(path,'replacement')
    assert path.read_text()=='unchanged'


@pytest.mark.parametrize('engine',['local','harbor'])
def test_regular_hardlink_overwrite_dealiases_only_selected_entry(tmp_path,engine):
    original=tmp_path/'original';target=tmp_path/'target';original.write_text('old');os.link(original,target)
    if engine=='local':assert FileWriteTool().execute(str(target),'new').success
    else:assert _run_harbor_write(target,'new').returncode==0
    assert original.read_text()=='old' and target.read_text()=='new'
    assert original.stat().st_ino!=target.stat().st_ino and not _temporary_paths(target)


def test_missing_target_creation_remains_supported_locally_and_in_harbor(tmp_path):
    local=tmp_path/'local';harbor=tmp_path/'harbor'
    assert FileWriteTool().execute(str(local),'local').success
    assert _run_harbor_write(harbor,'harbor').returncode==0
    assert local.read_text()=='local' and harbor.read_text()=='harbor'
    assert not _temporary_paths(local)


def test_missing_local_target_that_appears_as_fifo_is_not_replaced(tmp_path,monkeypatch):
    target=tmp_path/'appeared';real=safe_path_io._renameat2
    def race(parent,source,dest,flags):
        assert flags==1;os.mkfifo(target);real(parent,source,dest,flags)
    monkeypatch.setattr(safe_path_io,'_renameat2',race)
    with pytest.raises(OSError):atomic_write_text_nofollow(target,'new')
    assert stat.S_ISFIFO(target.lstat().st_mode) and not _temporary_paths(target)


def test_existing_local_target_swapped_to_fifo_is_retained_for_recovery(tmp_path,monkeypatch):
    target=tmp_path/'swapped';target.write_text('old');real=safe_path_io._renameat2;calls=[]
    def race(parent,source,dest,flags):
        calls.append(flags);target.unlink();os.mkfifo(target);real(parent,source,dest,flags)
    monkeypatch.setattr(safe_path_io,'_renameat2',race)
    with pytest.raises(SafePathError,match='non-regular overwrite target') as caught:
        atomic_write_text_nofollow(target,'new')
    outcome=caught.value.publication_outcome
    assert calls==[2] and outcome['publication_state']=='indeterminate' and outcome['no_auto_retry']
    assert target.read_text()=='new'
    retained=[tmp_path/e for e in outcome['recovery_entries']]
    assert any(stat.S_ISFIFO(p.lstat().st_mode) for p in retained)
    assert any(p.suffix=='.intent' for p in retained)


def test_missing_harbor_target_that_appears_as_fifo_is_not_replaced(tmp_path):
    target=tmp_path/'appeared'
    needle='''            rename_at2(parent, temporary, name, 1)  # RENAME_NOREPLACE'''
    assert needle in _SECURE_ATOMIC_WRITE
    injected=_SECURE_ATOMIC_WRITE.replace(needle,'''            os.mkfifo(name,dir_fd=parent)
'''+needle,1)
    result=_run_harbor_write(target,'new',injected)
    assert result.returncode!=0 and stat.S_ISFIFO(target.lstat().st_mode) and not _temporary_paths(target)


def test_existing_harbor_target_swapped_to_fifo_is_retained_for_recovery(tmp_path):
    target=tmp_path/'swapped';target.write_text('old')
    needle='''            rename_at2(parent, temporary, name, 2)  # RENAME_EXCHANGE'''
    assert needle in _SECURE_ATOMIC_WRITE
    injected=_SECURE_ATOMIC_WRITE.replace(needle,'''            os.unlink(name,dir_fd=parent)
            os.mkfifo(name,dir_fd=parent)
'''+needle,1)
    result=_run_harbor_write(target,'new',injected)
    assert result.returncode!=0 and target.read_text()=='new'
    prefix='__HL_PUBLICATION__'
    outcome=json.loads(next(line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix)))
    assert outcome['atomic_replace'] is None and outcome['no_auto_retry']
    assert any(stat.S_ISFIFO((tmp_path/e).lstat().st_mode) for e in outcome['recovery_entries'])
