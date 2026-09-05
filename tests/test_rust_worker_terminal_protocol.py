from __future__ import annotations

import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from bench import agent as module
from bench.agent import HLAgent
from bench.worker_protocol import WorkerStdout, validate_shutdown_bounds

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX executable/watchdog fixtures')


@pytest.fixture(autouse=True)
def outer_deadline():
    def expired(*_):
        raise AssertionError('Rust bridge exceeded the outer test deadline')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 6)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    agent = object.__new__(HLAgent)
    agent.rust_stderr_tail_bytes = 4096
    agent.rust_stdout_frame_bytes = 1024 * 1024
    agent.rust_shutdown_timeout_seconds = .35
    agent._process_lock = threading.Lock()
    agent._active_process = None
    agent._rust_worker_request = lambda instruction, context: {'instruction': instruction}
    agent._trial_result_from_rust = lambda result, context: result
    events = []
    agent._append_trajectory = events.append
    children = []
    popen = subprocess.Popen
    def track(*args, **kwargs):
        process = popen(*args, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(module.subprocess, 'Popen', track)
    def run(program):
        path = tmp_path / 'worker.py'
        path.write_text('import sys, os, json, time\nsys.stdin.readline()\n' + program, encoding='utf-8')
        agent._rust_worker_command = lambda: [sys.executable, str(path)]
        return agent._run_rust_core('test', {'task_id': 'fixture'})
    try:
        yield agent, run, children, events
    finally:
        for child in children:
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            child.wait(timeout=2)
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()


FINAL = 'print(json.dumps({"type":"final", "result":{"message":"成功"}}, ensure_ascii=False), flush=True)\n'


def test_clean_final_is_returned_and_owning_streams_close(bridge):
    agent, run, children, _ = bridge
    assert run(FINAL) == {'message': '成功'}
    assert children[0].poll() == 0
    assert children[0].stdin.closed and children[0].stdout.closed
    assert agent._active_process is None


@pytest.mark.parametrize('tail', ['{"type":"final","result":{}}', 'late log', 'not-json'])
def test_trailing_data_in_same_write_is_not_lost_in_read_ahead(bridge, tail):
    _, run, children, _ = bridge
    data = json.dumps({'type': 'final', 'result': {}}) + '\n' + tail + '\n'
    with pytest.raises(RuntimeError, match='after its final event'):
        run(f'os.write(1, {data.encode()!r})\n')
    assert children[0].poll() is not None


@pytest.mark.parametrize('delayed', [False, True])
def test_final_then_stdout_flood_never_deadlocks(bridge, delayed):
    _, run, children, _ = bridge
    with pytest.raises(RuntimeError, match='after its final event'):
        run(FINAL + ('time.sleep(.05)\n' if delayed else '') + 'os.write(1, b"x" * (2 * 1024 * 1024))\n')
    assert children[0].poll() is not None


def test_whitespace_trailer_drains_without_unbounded_buffering(bridge):
    agent, run, _, _ = bridge
    agent.rust_shutdown_timeout_seconds = 2
    assert run(FINAL + 'os.write(1, b" \\n" * (1024 * 1024))\n') == {'message': '成功'}


@pytest.mark.parametrize('close_stdout', [False, True])
def test_worker_that_emits_final_but_never_exits_is_stopped(bridge, close_stdout):
    _, run, children, _ = bridge
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='final shutdown timed out'):
        run(FINAL + ('os.close(1)\n' if close_stdout else '') + 'time.sleep(60)\n')
    assert time.monotonic() - started < 3
    assert children[0].poll() is not None


def test_no_final_is_an_error_even_if_stdout_closed_and_worker_alive(bridge):
    _, run, children, _ = bridge
    with pytest.raises(RuntimeError, match='without a final result'):
        run('os.close(1)\ntime.sleep(60)\n')
    assert children[0].poll() is not None


def test_nonzero_exit_after_final_is_not_accepted(bridge):
    _, run, _, _ = bridge
    with pytest.raises(RuntimeError, match='exited with code 7'):
        run(FINAL + 'raise SystemExit(7)\n')


def test_final_without_terminal_newline_is_supported(bridge):
    agent, run, _, _ = bridge
    data = json.dumps({'type': 'final', 'result': {'ok': True}}).encode()
    agent.rust_stdout_frame_bytes = len(data)
    assert run(f'os.write(1, {data!r})\n') == {'ok': True}


@pytest.mark.parametrize('payload', ['[]', '{"type":"final","result":[]}', 'garbage', '\\xff'])
def test_invalid_protocol_is_rejected_and_process_reaped(bridge, payload):
    _, run, children, _ = bridge
    data = b'\xff\n' if payload == '\\xff' else (payload + '\n').encode()
    with pytest.raises((RuntimeError, ValueError, UnicodeError)):
        run(f'os.write(1, {data!r})\ntime.sleep(60)\n')
    assert children[0].poll() is not None


def test_unterminated_oversized_record_has_a_memory_bound(bridge):
    agent, run, children, _ = bridge
    agent.rust_stdout_frame_bytes = 1024
    with pytest.raises(RuntimeError, match='frame exceeds'):
        run('os.write(1, b"x" * 65536)\ntime.sleep(60)\n')
    assert children[0].poll() is not None


def test_late_stderr_stays_bounded_and_is_in_error(bridge):
    _, run, _, _ = bridge
    with pytest.raises(RuntimeError) as error:
        run('os.write(2,b"e"*(2*1024*1024)+b"STDERR-END")\n' + FINAL + 'raise SystemExit(9)\n')
    assert 'STDERR-END' in str(error.value)
    assert len(str(error.value)) < 5000


def test_real_tool_and_llm_request_response_dispatch_remains_ordered(bridge, monkeypatch):
    agent, run, _, events = bridge
    agent._execute_bridge_tool = lambda event: {'tool_ok': event['name']}
    agent._completion_kwargs_for_bridge = lambda messages, schemas: {}
    agent._llm_response_payload = lambda response: {'model_ok': True}
    agent._llm_error_response_payload = lambda error: {'error': str(error)}
    monkeypatch.setattr(module._base.litellm, 'completion', lambda **kw: object())
    script = '''print(json.dumps({"type":"tool_request","name":"fixture"}),flush=True)
assert json.loads(sys.stdin.readline()) == {"type":"tool_response","payload":{"tool_ok":"fixture"}}
print(json.dumps({"type":"trajectory_event","event":{"checked":True}}),flush=True)
print(json.dumps({"type":"llm_request","messages":[],"tool_schemas":[]}),flush=True)
assert json.loads(sys.stdin.readline()) == {"type":"llm_response","payload":{"model_ok":True}}
'''
    assert run(script + FINAL) == {'message': '成功'}
    assert events == [{'checked': True}]


def test_parent_retained_stdout_writer_has_bounded_postfinal_handling(bridge, monkeypatch):
    agent, run, children, _ = bridge
    real_popen = module.subprocess.Popen
    read_fd, write_fd = os.pipe()
    read_stream = None
    def held_writer(*args, **kwargs):
        nonlocal read_stream
        kwargs['stdout'] = write_fd
        process = real_popen(*args, **kwargs)
        read_stream = os.fdopen(read_fd, 'rb', buffering=0)
        process.stdout = read_stream
        return process
    monkeypatch.setattr(module.subprocess, 'Popen', held_writer)
    try:
        with pytest.raises(RuntimeError, match='final shutdown timed out'):
            run(FINAL)
        assert children[0].poll() == 0
        assert read_stream.closed
    finally:
        os.close(write_fd)
        if read_stream is None: os.close(read_fd)
        else: read_stream.close()


@pytest.mark.parametrize('seconds', [False, 0, -1, float('nan'), float('inf'), None])
def test_invalid_shutdown_timeout_fails_before_launch(bridge, monkeypatch, seconds):
    agent, _, children, _ = bridge
    agent.rust_shutdown_timeout_seconds = seconds
    agent._rust_worker_command = lambda: pytest.fail('must validate before discovery')
    with pytest.raises(ValueError, match='shutdown_timeout'):
        agent._run_rust_core('test', {})
    assert children == []


@pytest.mark.parametrize('size', [True, 0, -1, 1.5, '1024', None])
def test_invalid_frame_limit_fails_before_launch(bridge, size):
    agent, _, children, _ = bridge
    agent.rust_stdout_frame_bytes = size
    agent._rust_worker_command = lambda: pytest.fail('must validate before discovery')
    with pytest.raises(ValueError, match='frame_bytes'):
        agent._run_rust_core('test', {})
    assert children == []


def test_binary_writer_handles_short_writes():
    class ShortWriter:
        def __init__(self): self.data = bytearray()
        def write(self, value): self.data.extend(value[:3]); return min(3, len(value))
        def flush(self): pass
    stream = ShortWriter()
    HLAgent._write_bridge_event(SimpleNamespace(stdin=stream), {'message': '成功'})
    assert json.loads(stream.data) == {'message': '成功'}
    text = io.StringIO()
    HLAgent._write_bridge_event(SimpleNamespace(stdin=text), {'ok': True})
    assert json.loads(text.getvalue()) == {'ok': True}


def test_binary_writer_rejects_no_progress():
    stream = SimpleNamespace(write=lambda _: 0, flush=lambda: None)
    with pytest.raises(BrokenPipeError):
        HLAgent._write_bridge_event(SimpleNamespace(stdin=stream), {'ok': True})


def test_cancellation_thread_does_not_close_stdout_owned_by_bridge(bridge, monkeypatch):
    agent, run, children, _ = bridge
    real_popen = module.subprocess.Popen
    closing_threads = []
    owner_thread = threading.get_ident()
    class StreamProxy:
        def __init__(self, stream): self.stream = stream
        def fileno(self): return self.stream.fileno()
        def close(self):
            closing_threads.append(threading.get_ident())
            self.stream.close()
        @property
        def closed(self): return self.stream.closed
    def wrap(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        child.stdout = StreamProxy(child.stdout)
        return child
    monkeypatch.setattr(module.subprocess, 'Popen', wrap)
    cancelled = threading.Event()
    def cancel():
        deadline = time.monotonic() + 2
        while agent._active_process is None and time.monotonic() < deadline:
            time.sleep(.005)
        if agent._active_process is not None:
            agent._terminate_process(agent._active_process)
            cancelled.set()
    thread = threading.Thread(target=cancel)
    thread.start()
    try:
        with pytest.raises((RuntimeError, BrokenPipeError)):
            run('time.sleep(60)\n')
    finally:
        thread.join(3)
    assert cancelled.is_set() and not thread.is_alive()
    assert children[0].poll() is not None
    assert closing_threads and set(closing_threads) == {owner_thread}
