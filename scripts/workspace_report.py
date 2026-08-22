#!/usr/bin/env python3
"""Classify the current worktree into controllable workspace categories."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


LOCAL_PRIVATE = {
    ".env",
    ".env.local",
    ".claude",
    ".claude/settings.local.json",
    "config/local.yaml",
    "config/local.yml",
}

RUNTIME_PREFIXES = (
    "jobs/",
    "trials/",
    "terminal-bench/",
    "terminal-bench-tasks/",
)

ITERATIVE_POLICY_PREFIXES = (
    "bench/agent.py",
    "bench/harbor.py",
    "crates/hl-worker-core/",
    "config/models.yaml",
    "config/trials.yaml",
    "harness/context/",
    "harness/prompts/",
    "harness/recovery/",
    "harness/tools/",
    "hl/compression.py",
    "hl/goals.py",
    "hl/loop.py",
    "hl/submit.py",
    "meta/codex_update.py",
    "meta/missions.py",
    "meta/packager.py",
    "meta/reviewer.py",
)

FIXED_BASELINE_PREFIXES = (
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "bench/",
    "config/",
    "crates/",
    "docs/",
    "harness/",
    "hl/",
    "harness_evolver/",
    "meta/",
    "pyproject.toml",
    "scripts/",
    "tests/",
)

TEMP_MARKERS = (
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "htmlcov/",
    ".egg-info/",
)

TEMP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
}

TEMP_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".coverage",
    ".tmp",
    ".bak",
    ".orig",
)

CATEGORY_DESCRIPTIONS = {
    "fixed_baseline": "Versioned project truth; commit when accepted.",
    "iterative_policy": "Source-controlled policy expected to evolve with evidence.",
    "local_private_config": "Machine-local config or secrets; keep ignored.",
    "runtime_evidence": "Harbor/trial/task evidence; keep ignored unless pruning explicitly.",
    "temporary_generated": "Disposable generated cache/build files.",
    "unclassified": "Needs manual review before commit or deletion.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    entries = read_git_status()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        category = classify_path(entry["path"])
        grouped[category].append(entry)

    payload = {
        "counts": {category: len(grouped.get(category, [])) for category in CATEGORY_DESCRIPTIONS},
        "categories": {
            category: {
                "description": CATEGORY_DESCRIPTIONS[category],
                "entries": sorted(grouped.get(category, []), key=lambda item: item["path"]),
            }
            for category in CATEGORY_DESCRIPTIONS
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_report(payload)
    return 0


def read_git_status() -> list[dict[str, str]]:
    completed = subprocess.run(
        ["git", "status", "--short", "--ignored"],
        capture_output=True,
        text=True,
        check=True,
    )
    entries: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        entries.append({"status": status, "path": path})
    return entries


def classify_path(path: str) -> str:
    normalized = path.replace("\\", "/").rstrip("/")
    if normalized in LOCAL_PRIVATE:
        return "local_private_config"
    if is_temporary(normalized):
        return "temporary_generated"
    if any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in RUNTIME_PREFIXES):
        return "runtime_evidence"
    if any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in ITERATIVE_POLICY_PREFIXES):
        return "iterative_policy"
    if any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in FIXED_BASELINE_PREFIXES):
        return "fixed_baseline"
    return "unclassified"


def is_temporary(path: str) -> bool:
    name = Path(path).name
    parts = Path(path).parts
    return (
        any(part in TEMP_DIR_NAMES for part in parts)
        or any(part.endswith(".egg-info") for part in parts)
        or any(marker in path for marker in TEMP_MARKERS)
        or name.endswith(TEMP_SUFFIXES)
        or name.endswith("~")
        or name in {".DS_Store", "Thumbs.db"}
    )


def print_report(payload: dict[str, Any]) -> None:
    for category, content in payload["categories"].items():
        entries = content["entries"]
        print(f"{category}: {len(entries)}")
        print(f"  {content['description']}")
        for entry in entries:
            print(f"  {entry['status']} {entry['path']}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
