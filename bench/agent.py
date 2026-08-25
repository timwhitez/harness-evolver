"""Rust Worker bridge with bounded stderr capture independent of pipe EOF."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import secrets
import subprocess
import threading
from typing import Any

from bench import _agent_issue8_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


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
    """Validate capture bounds before creating pipes or launching the Worker."""

    if isinstance(value, bool):
        raise ValueError("rust_stderr_tail_bytes must be an integer >= 1")
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rust_stderr_tail_bytes must be an integer >= 1") from exc
    if limit < 1:
        raise ValueError("rust_stderr_tail_bytes must be an integer >= 1")
    return limit


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

    def _run_rust_core(
        self,
        task_instruction: str,
        task_context: dict[str, Any],
    ) -> _base.TrialResult:
        # Validate before allocating descriptors or starting a child. A malformed
        # externally supplied constructor value must never leave a live Worker or
        # leaked pipe outside the normal teardown region.
        stderr_tail_limit = _validated_stderr_tail_bytes(self.rust_stderr_tail_bytes)

        read_fd, worker_stderr_fd = os.pipe()
        wake_stderr_fd = os.dup(worker_stderr_fd)
        process: subprocess.Popen[str] | None = None

        try:
            process = subprocess.Popen(
                self._rust_worker_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=worker_stderr_fd,
                text=True,
                bufsize=1,
                start_new_session=(os.name != "nt"),
            )
        except Exception:
            for descriptor in (read_fd, wake_stderr_fd):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise
        finally:
            try:
                os.close(worker_stderr_fd)
            except OSError:
                pass

        assert process.stdin is not None
        assert process.stdout is not None

        capture = _BoundedStderrTail(stderr_tail_limit)
        stop_marker = b"\x00HL-STDERR-STOP:" + secrets.token_bytes(32) + b"\x00"
        stderr_thread = threading.Thread(
            target=_drain_stderr_fd,
            args=(read_fd, capture, stop_marker),
            name="harness-evolver-rust-stderr-drain",
            daemon=True,
        )
        stderr_thread.start()
        stderr_finished = False

        def finish_stderr() -> str:
            nonlocal stderr_finished
            if stderr_finished:
                return capture.text()
            stderr_finished = True
            try:
                _write_all(wake_stderr_fd, stop_marker)
            except OSError:
                # EOF or an already-failed reader is also a completed drain.
                pass
            finally:
                try:
                    os.close(wake_stderr_fd)
                except OSError:
                    pass

            # The marker wakes a reader even while a descendant still owns fd 2.
            # A timeout is defensive only; the thread is daemonized and the raw
            # descriptor close below cannot deadlock on a buffered-stream lock.
            stderr_thread.join(timeout=2.0)
            try:
                os.close(read_fd)
            except OSError:
                pass
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

            final_payload: dict[str, Any] | None = None
            for line in process.stdout:
                if not line.strip():
                    continue
                event = json.loads(line)
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
                    final_payload = dict(event.get("result") or {})
                    break
                elif event_type == "fatal":
                    raise RuntimeError(
                        str(event.get("error") or "Rust Worker core fatal error")
                    )
                else:
                    raise RuntimeError(
                        f"Unknown Rust Worker core event: {event_type!r}"
                    )

            return_code = process.wait()
            stderr = finish_stderr()
            if final_payload is None:
                raise RuntimeError(
                    "Rust Worker core exited without a final result"
                    + (f": {stderr.strip()}" if stderr.strip() else "")
                )
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
            if process.poll() is None:
                self._terminate_process(process)
            finish_stderr()
            with self._process_lock:
                if self._active_process is process:
                    self._active_process = None
