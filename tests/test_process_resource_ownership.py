from __future__ import annotations

import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest

from harness.tools import process_runner as runner
from harness.tools.pipe_io import PipeReader


pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc resource-lifecycle fixtures")


def fd_count():
    return len(list(Path('/proc/self/fd').iterdir()))


@pytest.fixture(autouse=True)
def deadline():
    def expired(*_):
        raise AssertionError('test exceeded outer deadline')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture(params=[False, True], ids=['process-group', 'subreaper'])
def mode(request, monkeypatch):
    if request.param and not sys.platform.startswith('linux'):
        pytest.skip('subreaper requires Linux')
    monkeypatch.setattr(runner._runtime, '_is_linux_subreaper_available', lambda: request.param)
    monkeypatch.setattr(runner._base, '_pid_namespace_prefix', lambda: None)
    return request.param


@pytest.fixture
def children(monkeypatch):
    original = subprocess.Popen
    launched = []
    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        launched.append(child)
        return child
    monkeypatch.setattr(runner.subprocess, 'Popen', launch)
    try:
        yield launched
    finally:
        for child in launched:
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            try: child.wait(timeout=2)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=2)
            for stream in (child.stdout, child.stderr, child.stdin):
                if stream is not None: stream.close()


@pytest.mark.parametrize('stage', ['pipe1', 'pipe2', 'token', 'command'])
def test_partial_supervisor_setup_closes_every_allocated_fd(monkeypatch, stage):
    baseline = fd_count()
    original_pipe = os.pipe
    count = 0
    def pipe():
        nonlocal count
        count += 1
        if stage == f'pipe{count}': raise OSError('pipe fixture failure')
        return original_pipe()
    monkeypatch.setattr(os, 'pipe', pipe)
    if stage == 'token':
        monkeypatch.setattr(runner.secrets, 'token_bytes', lambda *_: (_ for _ in ()).throw(OSError('token fixture failure')))
    if stage == 'command':
        monkeypatch.setattr(runner._base, 'supervised_command_for_argv', lambda *_: (_ for _ in ()).throw(OSError('command fixture failure')))
    with pytest.raises(OSError, match='fixture failure'):
        runner._prepare_supervised_launch([sys.executable, '-c', 'pass'])
    assert fd_count() == baseline


@pytest.mark.parametrize('position', [1, 2])
def test_capture_allocation_failure_never_launches(mode, children, monkeypatch, position):
    original = runner._runtime._BoundedCapture
    calls = 0
    def capture(*args):
        nonlocal calls
        calls += 1
        if calls == position: raise MemoryError('capture fixture failure')
        return original(*args)
    monkeypatch.setattr(runner._runtime, '_BoundedCapture', capture)
    baseline = fd_count()
    with pytest.raises(MemoryError):
        runner.run_bounded_argv([sys.executable, '-c', 'pass'], timeout_seconds=1)
    assert children == []
    assert fd_count() == baseline


def test_popen_failure_releases_supervisor_pipes(mode, monkeypatch):
    baseline = fd_count()
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(OSError('popen fixture failure')))
    with pytest.raises(OSError, match='popen fixture'):
        runner.run_bounded_argv([sys.executable, '-c', 'pass'], timeout_seconds=1)
    assert fd_count() == baseline


@pytest.mark.parametrize('position', [1, 2])
def test_reader_setup_failure_reaps_real_child(mode, children, monkeypatch, position):
    original = runner.PipeReader
    count = 0
    baseline = fd_count()
    threads = {t.ident for t in threading.enumerate()}
    def reader(*args):
        nonlocal count
        count += 1
        if count == position: raise RuntimeError('reader fixture failure')
        return original(*args)
    monkeypatch.setattr(runner, 'PipeReader', reader)
    with pytest.raises(RuntimeError, match='reader fixture'):
        runner.run_bounded_argv([sys.executable, '-c', 'import time; time.sleep(10)'], timeout_seconds=1)
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdout.closed and children[0].stderr.closed
    assert {t.ident for t in threading.enumerate()} == threads
    assert fd_count() == baseline


def test_capture_feed_error_also_reaps_real_child(mode, children, monkeypatch):
    baseline = fd_count()
    monkeypatch.setattr(runner._runtime._BoundedCapture, 'feed', lambda *a: (_ for _ in ()).throw(RuntimeError('feed fixture failure')))
    with pytest.raises(RuntimeError, match='feed fixture'):
        runner.run_bounded_argv([sys.executable, '-c', 'import time; print("ready", flush=True); time.sleep(10)'], timeout_seconds=2)
    assert children[0].poll() is not None
    assert fd_count() == baseline


def test_no_thread_start_is_needed_to_run_or_drain(mode, monkeypatch):
    # The old first/second thread-start failure modes no longer exist.
    monkeypatch.setattr(threading.Thread, 'start', lambda *_: pytest.fail('reader must not create a thread'))
    result = runner.run_bounded_argv([sys.executable, '-c', 'print("ok")'], timeout_seconds=2)
    assert result.returncode == 0 and result.stdout == 'ok\n' and result.output_eof


def test_large_stdout_and_stderr_are_drained_and_bounded(mode):
    program = 'import os\nfor i in range(512):\n os.write(1,b"a"*4096)\n os.write(2,b"b"*4096)\n'
    result = runner.run_bounded_argv([sys.executable, '-c', program], timeout_seconds=6, output_limit_bytes=4096)
    assert result.returncode == 0 and not result.timed_out
    assert len(result.stdout.encode()) <= 4096 and len(result.stderr.encode()) <= 4096
    assert 'omitted' in result.stdout and 'omitted' in result.stderr
    assert result.output_eof


def test_timeout_and_cancellation_keep_cleanup_contract(mode, children):
    result = runner.run_bounded_argv([sys.executable, '-c', 'import time; time.sleep(10)'], timeout_seconds=.35)
    assert result.timed_out and not result.cancelled
    assert children[-1].poll() is not None
    assert result.managed_process_group_terminated
    cancel = threading.Event()
    timer = threading.Timer(.35, cancel.set)
    timer.start()
    try:
        result = runner.run_bounded_argv([sys.executable, '-c', 'import time; time.sleep(10)'], timeout_seconds=5, cancel_event=cancel)
    finally:
        timer.cancel(); timer.join(1)
    assert result.cancelled and not result.timed_out
    assert children[-1].poll() is not None
    assert result.managed_process_group_terminated


def test_environment_replacement_and_exact_argv(mode, monkeypatch):
    monkeypatch.setenv('HL_PARENT_ONLY', 'must-not-leak')
    code = 'import os,sys; print(os.environ.get("HL_PARENT_ONLY","absent")); print(sys.argv[1])'
    result = runner.run_bounded_argv([sys.executable, '-c', code, 'space ; literal'], timeout_seconds=2, env={})
    assert result.returncode == 0
    assert result.stdout == 'absent\nspace ; literal\n'


def test_rc124_without_cleanup_proof_is_not_attestation(mode):
    result = runner.run_bounded_argv([sys.executable, '-c', 'raise SystemExit(124)'], timeout_seconds=2)
    assert result.returncode == 124 and not result.timed_out
    assert not result.managed_process_group_terminated


def test_retained_pipe_writer_cannot_stall_teardown_or_close_reused_fd(tmp_path, monkeypatch):
    monkeypatch.setattr(runner._runtime, '_is_linux_subreaper_available', lambda: False)
    monkeypatch.setattr(runner._runtime, '_STREAM_JOIN_SECONDS', .05)
    raw_read, retained_write = os.pipe()
    original_popen = subprocess.Popen
    child = None
    replacement = None
    stream = None
    original_close = runner._runtime._close_stream
    def popen(*args, **kwargs):
        nonlocal child, stream
        kwargs['stderr'] = retained_write
        child = original_popen(*args, **kwargs)
        stream = os.fdopen(raw_read, 'rb', buffering=0)
        child.stderr = stream
        return child
    def close_once(value):
        nonlocal replacement
        original_close(value)
        if value is stream and replacement is None:
            replacement = os.open(tmp_path / 'unrelated', os.O_CREAT | os.O_RDWR, 0o600)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(runner._runtime, '_close_stream', close_once)
    try:
        result = runner.run_bounded_argv([sys.executable, '-c', 'print("done")'], timeout_seconds=2)
        assert result.stdout == 'done\n' and not result.output_eof
        assert child.poll() is not None and stream.closed
        assert replacement is not None
        # Repeated owner close/finalization is idempotent; the number is not
        # closed again behind the back of an unrelated new descriptor owner.
        stream.close(); gc.collect()
        os.fstat(replacement)
        assert os.write(replacement, b'usable') == 6
    finally:
        if child is not None and child.poll() is None:
            child.kill(); child.wait(timeout=2)
        if stream is not None: stream.close()
        else: os.close(raw_read)
        os.close(retained_write)
        if replacement is not None:
            try: os.close(replacement)
            except OSError: pass


def test_nonblocking_pipe_without_writer_data_returns_immediately():
    read, write = os.pipe()
    with os.fdopen(read, 'rb', buffering=0) as stream:
        try:
            reader = PipeReader(stream)
            assert reader.read() is None
            os.write(write, b'abc')
            assert reader.read() == b'abc'
            os.close(write); write = None
            assert reader.read() == b''
        finally:
            if write is not None: os.close(write)
