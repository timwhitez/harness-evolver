"""Semantic entry-point scanner — pre-scan before agent exploration.

ForgeCode pattern: lightweight semantic pass identifies most likely
starting files before exploration begins, saving context budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.entrypoint.base import EntryPointDiscovery


@dataclass
class SemanticScanner(EntryPointDiscovery):
    name: str = "semantic_scanner"
    version: str = "0.1.0"

    scan_depth: int = 2
    max_scan_files: int = 50
