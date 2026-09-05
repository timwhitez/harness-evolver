from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import pytest
from bench._harbor_adapter_issue13_impl import _HARBOR_GREP_COUNTING_PYTHON

pytestmark=pytest.mark.skipif(not sys.platform.startswith('linux'),reason='Linux pinned acquisition')


def run(path: Path, script=_HARBOR_GREP_COUNTING_PYTHON):
    return subprocess.run([sys.executable,'-c',script], env={**os.environ,'HL_ROOT':str(path),'HL_PATTERN':'needle'},
                          capture_output=True,text=True,timeout=2)


@pytest.mark.parametrize('root_is_fifo',[True,False])
def test_grep_fifo_is_rejected_without_a_peer(tmp_path,root_is_fifo):
    fifo=tmp_path/'pipe';os.mkfifo(fifo);before=fifo.lstat()
    result=run(fifo if root_is_fifo else tmp_path)
    assert result.returncode!=0 and '__HL_GREP_COUNT__' not in result.stdout
    assert fifo.lstat().st_ino==before.st_ino


def test_normal_root_file_and_directory_keep_counts(tmp_path):
    (tmp_path/'one').write_text('needle\n')
    (tmp_path/'two').write_text('needle\nneedle\n')
    assert run(tmp_path).stdout.splitlines()[0]=='__HL_GREP_COUNT__3'
    assert run(tmp_path/'one').stdout.splitlines()[0]=='__HL_GREP_COUNT__1'


def test_hardlink_status_remains_distinct(tmp_path):
    (tmp_path/'one').write_text('needle\n');os.link(tmp_path/'one',tmp_path/'two')
    result=run(tmp_path/'one')
    assert result.returncode==74 and result.stdout==''


def test_grep_file_swap_to_fifo_before_pin_does_not_stall(tmp_path):
    (tmp_path/'one').write_text('needle\n')
    needle='''                descriptor, _ = open_readonly_checked(
                    directory_fd, name, unique=False,
                )'''
    replace='''                os.unlink(name,dir_fd=directory_fd)
                os.mkfifo(name,dir_fd=directory_fd)
'''+needle
    assert needle in _HARBOR_GREP_COUNTING_PYTHON
    result=run(tmp_path,_HARBOR_GREP_COUNTING_PYTHON.replace(needle,replace,1))
    assert result.returncode!=0 and '__HL_GREP_COUNT__' not in result.stdout
