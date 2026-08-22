"""TerminalBench integration exports.

Keep these imports lazy. Importing any ``bench.*`` submodule used to execute
this package initializer, which eagerly imported :mod:`bench.agent` and its
heavy LiteLLM dependency. Lightweight commands such as campaign ``--dry-run``
and network-preflight tests then paid provider-SDK startup cost before doing
any work.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "HarborRunner",
    "HLAgent",
    "TaskCatalog",
    "Scoring",
    "TrajectoryReader",
]


_LAZY_EXPORTS = {
    "HarborRunner": ("bench.harbor", "HarborRunner"),
    "HLAgent": ("bench.agent", "HLAgent"),
    "TaskCatalog": ("bench.tasks", "TaskCatalog"),
    "Scoring": ("bench.scoring", "Scoring"),
    "TrajectoryReader": ("bench.trajectory", "TrajectoryReader"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
