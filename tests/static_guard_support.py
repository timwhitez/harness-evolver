"""No-dispatch spies for policy decisions that must precede environment execution."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from bench.harbor_adapter import HarborFileWriteTool


def no_dispatch_write_tool():
    # Synchronous spy raises when the API is invoked, before any coroutine body.
    environment = SimpleNamespace(
        exec=Mock(side_effect=AssertionError("static guard fixture attempted environment.exec"))
    )
    loop = Mock(spec=asyncio.AbstractEventLoop)
    loop.call_soon_threadsafe.side_effect = AssertionError(
        "static guard fixture attempted event-loop scheduling"
    )
    tool = HarborFileWriteTool(environment=environment, loop=loop, timeout_seconds=5)
    return tool, environment, loop
