"""Single-consumer, nonblocking pipe reads; the caller owns and closes the stream.

Never close a raw descriptor behind an owning Python stream. No read thread or
buffered read may share this stream. POSIX uses O_NONBLOCK; Windows uses a
PeekNamedPipe availability check (also works on Python 3.11).
"""
from __future__ import annotations

import os
from typing import BinaryIO


class PipeReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.fd = stream.fileno()
        self._peek = None
        if os.name == "posix":
            os.set_blocking(self.fd, False)
        elif os.name == "nt":
            import ctypes
            from ctypes import wintypes
            import msvcrt

            self._ctypes = ctypes
            self._available = wintypes.DWORD
            self._handle = wintypes.HANDLE(msvcrt.get_osfhandle(self.fd))
            self._peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
            self._peek.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                   wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD),
                                   wintypes.LPVOID]
            self._peek.restype = wintypes.BOOL
        else:
            raise OSError("nonblocking process pipe reading is unsupported")

    def read(self, size: int = 65_536) -> bytes | None:
        """Return bytes, b'' at EOF, or None when there is currently no data."""
        if self._peek is not None:
            available = self._available()
            if not self._peek(self._handle, None, 0, None,
                              self._ctypes.byref(available), None):
                error = self._ctypes.get_last_error()
                if error in (109, 232):  # broken/disconnected pipe
                    return b""
                raise self._ctypes.WinError(error)
            if not available.value:
                return None
            size = min(size, available.value)
        try:
            return os.read(self.fd, size)
        except (BlockingIOError, InterruptedError):
            return None
