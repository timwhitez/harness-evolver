"""Rust Worker bridge with bounded stderr capture independent of pipe EOF."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import math
import os
import re
import secrets
import signal
import subprocess
import threading
from typing import Any

from bench import _agent_issue8_base as _base
from bench.worker_protocol import WorkerStdout, WorkerProtocolError, validate_shutdown_bounds

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals().setdefault(_name, _value)


@dataclass
class _BoundedStderrTail:
    """Continuously retain only the final configured stderr bytes."""

    limit_bytes: int
    total_bytes: int = 0
    tail: bytearray = field(default_factory=bytearray)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def feed(self, payload: bytes) -> None:
        if not payload:
            return
        with self._lock:
            self.total_bytes += len(payload)
            self.tail.extend(payload)
            if len(self.tail) > self.limit_bytes:
                del self.tail[: len(self.tail) - self.limit_bytes]

    def text(self) -> str:
        with self._lock:
            total = self.total_bytes
            payload = bytes(self.tail)
        text = payload.decode("utf-8", errors="replace")
        omitted = max(0, total - len(payload))
        if omitted:
            return f"... [{omitted} stderr bytes omitted] ...\n{text}"
        return text


def _validated_stderr_tail_bytes(value: object) -> int:
    """Validate an exact positive integer before allocating process resources."""

    if isinstance(value, bool):
        raise ValueError("rust_stderr_tail_bytes must be an integer >= 1")

    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\+?\d+", text) is None:
            raise ValueError("rust_stderr_tail_bytes must be an integer >= 1")
        limit = int(text)
    else:
        if isinstance(value, float) and (
            not math.isfinite(value) or not value.is_integer()
        ):
            raise ValueError("rust_stderr_tail_bytes must be an integer >= 1")
        try:
            limit = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("rust_stderr_tail_bytes must be an integer >= 1") from exc
        try:
            if value != limit:
                raise ValueError("rust_stderr_tail_bytes must be an integer >= 1")
        except TypeError as exc:
            raise ValueError("rust_stderr_tail_bytes must be an integer >= 1") from exc

    if limit < 1:
        raise ValueError("rust_stderr_tail_bytes must be an integer >= 1")
    return limit


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _drain_stderr_fd(
    read_fd: int,
    capture: _BoundedStderrTail,
    stop_marker: bytes,
) -> None:
    """Drain a raw pipe until EOF or an explicit parent-owned stop marker.

    The marker is parsed across read boundaries and is never included in user
    diagnostics. This avoids relying on every descendant to close its inherited
    stderr descriptor and avoids closing a buffered stream from another thread.
    """

    pending = bytearray()
    try:
        while True:
            chunk = os.read(read_fd, 65_536)
            if not chunk:
                capture.feed(bytes(pending))
                return
            pending.extend(chunk)

            marker_at = pending.find(stop_marker)
            if marker_at >= 0:
                capture.feed(bytes(pending[:marker_at]))
                return

            # Retain only the suffix that could still be the beginning of a
            # marker split across the next read. Everything before it is proven
            # diagnostic data and can enter the bounded tail immediately.
            safe_length = max(0, len(pending) - len(stop_marker) + 1)
            if safe_length:
                capture.feed(bytes(pending[:safe_length]))
                del pending[:safe_length]
    except (OSError, ValueError):
        capture.feed(bytes(pending))


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while stopping stderr drain")
        view = view[written:]


@dataclass
class HLAgent(_base.HLAgent):
    """Packaged-runtime HLAgent whose stderr storage stays bounded in flight."""

    rust_stderr_tail_bytes: int = 65_536
    rust_shutdown_timeout_seconds: float = 5.0
    rust_stdout_frame_bytes: int = 64 * 1024 * 1024

    @staticmethod
    def _write_bridge_event(process: subprocess.Popen, event: dict[str, Any]) -> None:
        """Send one complete JSONL message to binary stdin, including short writes."""
        if process.stdin is None:
            raise WorkerProtocolError("Rust Worker stdin is unavailable")
        text = json.dumps(event, ensure_ascii=False) + "\n"
        if isinstance(process.stdin, io.TextIOBase):
            # Retain compatibility with direct text-stream adapter fixtures.
            process.stdin.write(text)
        else:
            remaining = memoryview(text.encode("utf-8"))
            while remaining:
                written = process.stdin.write(remaining)
                if written is None or written <= 0:
                    raise BrokenPipeError("Rust Worker stdin did not accept a complete event")
                remaining = remaining[written:]
        process.stdin.flush()

    def _terminate_process(self, process: subprocess.Popen) -> None:
        """Stop the Worker without closing stdout from a cancellation thread.

        The _run_rust_core owner closes streams in its finalizer. Termination
        itself has a bounded wait, including after a terminal protocol timeout.
        """
        if process.poll() is not None:
            return
        self._send_process_signal(process, signal.SIGTERM)
        try:
            process.wait(timeout=0.5)
            return
        except subprocess.TimeoutExpired:
            pass
        self._send_process_signal(process, signal.SIGKILL)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            raise WorkerProtocolError("Rust Worker did not exit after termination") from exc

    def _run_rust_core(
        self,
        task_instruction: str,
        task_context: dict[str, Any],
    ) -> _base.TrialResult:
        # Validate before allocating descriptors or starting a child. A malformed
        # externally supplied constructor value must never leave a live Worker or
        # leaked pipe outside the normal teardown region.
        stderr_tail_limit = _validated_stderr_tail_bytes(self.rust_stderr_tail_bytes)
        shutdown_seconds, frame_bytes = validate_shutdown_bounds(
            self.rust_shutdown_timeout_seconds, self.rust_stdout_frame_bytes,
        )
        capture = _BoundedStderrTail(stderr_tail_limit)
        stop_marker = b"\x00HL-STDERR-STOP:" + secrets.token_bytes(32) + b"\x00"

        read_fd: int | None = None
        worker_stderr_fd: int | None = None
        wake_stderr_fd: int | None = None
        process: subprocess.Popen[bytes] | None = None
        stderr_thread: threading.Thread | None = None

        try:
            read_fd, worker_stderr_fd = os.pipe()
            try:
                wake_stderr_fd = os.dup(worker_stderr_fd)
            except Exception:
                _close_fd(read_fd)
                _close_fd(worker_stderr_fd)
                read_fd = None
                worker_stderr_fd = None
                raise

            stderr_thread = threading.Thread(
                target=_drain_stderr_fd,
                args=(read_fd, capture, stop_marker),
                name="harness-evolver-rust-stderr-drain",
                daemon=True,
            )

            try:
                process = subprocess.Popen(
                    self._rust_worker_command(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=worker_stderr_fd,
                    text=False,
                    bufsize=0,
                    start_new_session=(os.name != "nt"),
                )
            except Exception:
                _close_fd(read_fd)
                _close_fd(wake_stderr_fd)
                read_fd = None
                wake_stderr_fd = None
                raise
            finally:
                _close_fd(worker_stderr_fd)
                worker_stderr_fd = None

            assert process.stdin is not None
            assert process.stdout is not None

            try:
                stderr_thread.start()
            except Exception:
                if process.poll() is None:
                    self._terminate_process(process)
                _close_fd(wake_stderr_fd)
                _close_fd(read_fd)
                wake_stderr_fd = None
                read_fd = None
                raise
        except Exception:
            _close_fd(worker_stderr_fd)
            _close_fd(wake_stderr_fd)
            _close_fd(read_fd)
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
            raise

        assert process is not None
        assert stderr_thread is not None
        assert read_fd is not None
        assert wake_stderr_fd is not None

        stderr_finished = False

        def finish_stderr() -> str:
            nonlocal stderr_finished, read_fd, wake_stderr_fd
            if stderr_finished:
                return capture.text()
            stderr_finished = True
            try:
                _write_all(wake_stderr_fd, stop_marker)
            except OSError:
                # EOF or an already-failed reader is also a completed drain.
                pass
            finally:
                _close_fd(wake_stderr_fd)
                wake_stderr_fd = None

            # The marker wakes a reader even while a descendant still owns fd 2.
            # A timeout is defensive only; the thread is daemonized and the raw
            # descriptor close below cannot deadlock on a buffered-stream lock.
            stderr_thread.join(timeout=2.0)
            _close_fd(read_fd)
            read_fd = None
            if stderr_thread.is_alive():
                stderr_thread.join(timeout=0.1)
            return capture.text()

        with self._process_lock:
            self._active_process = process

        try:
            self._write_bridge_event(
                process,
                {
                    "type": "start",
                    "request": self._rust_worker_request(task_instruction, task_context),
                },
            )

            stdout = WorkerStdout(process.stdout, process,
                shutdown_seconds=shutdown_seconds, frame_bytes=frame_bytes)
            final_payload: dict[str, Any] | None = None
            for line in stdout:
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise WorkerProtocolError("Rust Worker event must be a JSON object")
                event_type = event.get("type")
                if event_type == "llm_request":
                    self.messages = list(event.get("messages") or [])
                    try:
                        response = _base.litellm.completion(
                            **self._completion_kwargs_for_bridge(
                                self.messages,
                                list(event.get("tool_schemas") or []),
                            )
                        )
                    except Exception as exc:
                        self._write_bridge_event(
                            process,
                            {
                                "type": "llm_response",
                                "payload": self._llm_error_response_payload(exc),
                            },
                        )
                        continue
                    try:
                        payload = self._llm_response_payload(response)
                    except Exception as exc:
                        self._write_bridge_event(
                            process,
                            {
                                "type": "llm_response",
                                "payload": self._llm_error_response_payload(exc),
                            },
                        )
                        continue
                    self._write_bridge_event(
                        process,
                        {
                            "type": "llm_response",
                            "payload": payload,
                        },
                    )
                elif event_type == "tool_request":
                    self._write_bridge_event(
                        process,
                        {
                            "type": "tool_response",
                            "payload": self._execute_bridge_tool(event),
                        },
                    )
                elif event_type == "trajectory_event":
                    trajectory_event = event.get("event")
                    if isinstance(trajectory_event, dict):
                        self._append_trajectory(trajectory_event)
                elif event_type == "final":
                    final_payload = event.get("result")
                    if not isinstance(final_payload, dict):
                        raise WorkerProtocolError("Rust Worker final result must be a JSON object")
                    break
                elif event_type == "fatal":
                    raise RuntimeError(
                        str(event.get("error") or "Rust Worker core fatal error")
                    )
                else:
                    raise RuntimeError(
                        f"Unknown Rust Worker core event: {event_type!r}"
                    )

            if final_payload is None:
                raise WorkerProtocolError("Rust Worker core exited without a final result")
            return_code = stdout.finish()
            stderr = finish_stderr()
            if return_code != 0:
                raise RuntimeError(
                    f"Rust Worker core exited with code {return_code}"
                    + (f": {stderr.strip()}" if stderr.strip() else "")
                )
            return self._trial_result_from_rust(final_payload, task_context)
        except Exception as exc:
            if process.poll() is None:
                self._terminate_process(process)
            stderr = finish_stderr()
            if stderr.strip() and stderr.strip() not in str(exc):
                raise RuntimeError(f"{exc}: {stderr.strip()}") from exc
            raise
        finally:
            try:
                if process.poll() is None:
                    self._terminate_process(process)
            finally:
                try:
                    finish_stderr()
                finally:
                    for stream in (process.stdin, process.stdout):
                        if stream is not None:
                            try:
                                stream.close()
                            except (OSError, ValueError):
                                pass
                    with self._process_lock:
                        if self._active_process is process:
                            self._active_process = None
