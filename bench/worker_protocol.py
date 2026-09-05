"""Bounded JSONL framing and terminal-event shutdown for the Rust bridge.

One consumer owns stdout. No text-buffered reader may consume this stream in
parallel, and the caller closes its owning stream after protocol completion.
A final event ends task execution, not stdout processing: trailing data and
missing EOF/exit are handled before accepting the result.
"""
from __future__ import annotations

import math
import subprocess
import time
from typing import BinaryIO, Iterator

from harness.tools.pipe_io import PipeReader


class WorkerProtocolError(RuntimeError):
    """Worker output violated its framing or terminal-event contract."""


def validate_shutdown_bounds(seconds: object, frame_bytes: object) -> tuple[float, int]:
    if isinstance(seconds, bool):
        raise ValueError("rust_shutdown_timeout_seconds must be finite and > 0")
    try:
        duration = float(seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("rust_shutdown_timeout_seconds must be finite and > 0") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("rust_shutdown_timeout_seconds must be finite and > 0")
    if isinstance(frame_bytes, bool) or not isinstance(frame_bytes, int) or frame_bytes < 1:
        raise ValueError("rust_stdout_frame_bytes must be an integer >= 1")
    return duration, frame_bytes


class WorkerStdout(Iterator[str]):
    """Read UTF-8 JSONL records without text read-ahead or unbounded frames."""

    def __init__(self, stream: BinaryIO, process: subprocess.Popen,
                 *, shutdown_seconds: float, frame_bytes: int) -> None:
        self.reader = PipeReader(stream)
        self.process = process
        self.shutdown_seconds = shutdown_seconds
        self.frame_bytes = frame_bytes
        self.pending = bytearray()
        self.eof = False
        self.exit_deadline: float | None = None

    def __iter__(self) -> WorkerStdout:
        return self

    def __next__(self) -> str:
        while True:
            newline = self.pending.find(b"\n")
            if newline >= 0:
                if newline > self.frame_bytes:
                    raise WorkerProtocolError("Rust Worker stdout frame exceeds configured limit")
                line = bytes(self.pending[:newline])
                del self.pending[:newline + 1]
                return line.decode("utf-8", errors="strict")
            if len(self.pending) > self.frame_bytes:
                raise WorkerProtocolError("Rust Worker stdout frame exceeds configured limit")
            if self.eof:
                if not self.pending:
                    raise StopIteration
                line = bytes(self.pending)
                self.pending.clear()
                return line.decode("utf-8", errors="strict")
            if self.process.poll() is not None:
                if self.exit_deadline is None:
                    self.exit_deadline = time.monotonic() + self.shutdown_seconds
                if time.monotonic() >= self.exit_deadline:
                    raise WorkerProtocolError("Rust Worker exited but stdout remained open")
            # One proof byte detects a too-large frame without materializing it.
            chunk = self.reader.read(min(65_536, self.frame_bytes + 1 - len(self.pending)))
            if chunk == b"":
                self.eof = True
            elif chunk is not None:
                self.pending.extend(chunk)
            else:
                time.sleep(0.005)

    def finish(self) -> int:
        """Require whitespace-only trailing stdout, EOF, and bounded process exit.

        Pending bytes from the read that contained final are checked as well as
        later pipe bytes. We never append to the framing buffer after final.
        """
        deadline = time.monotonic() + self.shutdown_seconds
        if self.pending.strip(b" \t\r\n"):
            raise WorkerProtocolError("Rust Worker emitted stdout after its final event")
        self.pending.clear()
        while True:
            code = self.process.poll()
            if self.eof and code is not None:
                return code
            if time.monotonic() >= deadline:
                raise WorkerProtocolError("Rust Worker final shutdown timed out waiting for EOF/exit")
            chunk = None if self.eof else self.reader.read()
            if chunk == b"":
                self.eof = True
            elif chunk is not None:
                if chunk.strip(b" \t\r\n"):
                    raise WorkerProtocolError("Rust Worker emitted stdout after its final event")
            else:
                time.sleep(0.005)
