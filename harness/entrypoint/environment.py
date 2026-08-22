"""Environment awareness — auto-bootstrap system info at startup.

Factory Droid pattern: auto-detect OS, installed packages, git state,
project structure at session start.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.entrypoint.base import EntryPointDiscovery


@dataclass
class EnvironmentAwareness(EntryPointDiscovery):
    name: str = "environment_awareness"
    version: str = "0.1.0"

    detect_os: bool = True
    detect_packages: bool = True
    detect_git: bool = True
    detect_languages: bool = True
