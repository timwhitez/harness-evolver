"""Shared policy helpers for Codex update evaluation.

These helpers keep Codex-update bookkeeping deterministic: file deltas are
classified into harness layers, validation commands are derived from the actual
changed files, and update records can carry a comparable component delta.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any


LAYER_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("bench/agent.py", "worker_loop"),
    ("crates/hl-worker-core/", "worker_loop"),
    ("bench/harbor_adapter.py", "harbor_adapter"),
    ("bench/harbor.py", "harbor_adapter"),
    ("bench/scoring.py", "scoring"),
    ("bench/trajectory.py", "trajectory"),
    ("bench/tasks.py", "task_catalog"),
    ("harness/prompts/", "prompt"),
    ("harness/tools/registry.py", "tool_schema"),
    ("harness/tools/base.py", "tool_schema"),
    ("harness/tools/", "tool_impl"),
    ("harness/planning/", "planning"),
    ("harness/context/", "context_compaction"),
    ("harness/recovery/", "recovery"),
    ("harness/entrypoint/", "entrypoint"),
    ("harness/verification/", "verification"),
    ("harness/skill_loading/", "skill_loading"),
    ("hl/loop.py", "campaign_loop"),
    ("hl/memory.py", "memory"),
    ("hl/coupling.py", "coupling"),
    ("hl/compression.py", "context_compaction"),
    ("hl/submit.py", "submit_gate"),
    ("meta/codex_update.py", "codex_update"),
    ("meta/packager.py", "codex_packet"),
    ("meta/reviewer.py", "diff_review"),
    ("meta/missions.py", "mission_debug"),
    ("meta/", "meta_updater"),
    ("scripts/run_campaign.py", "campaign_runner"),
    ("scripts/regression_check.py", "regression_gate"),
    ("scripts/run_trial.py", "harbor_adapter"),
    ("scripts/", "script_orchestration"),
    ("config/", "config"),
    ("tests/", "tests"),
    ("docs/", "docs"),
    ("README.md", "docs"),
    ("AGENTS.md", "repo_guidance"),
)


def classify_component_delta(changed_files: list[str]) -> dict[str, Any]:
    """Map changed files to stable Worker/harness component layers."""

    normalized = [_normalize_path(path) for path in changed_files if str(path).strip()]
    file_layers: dict[str, list[str]] = {}
    layer_counts: dict[str, int] = {}
    for path in normalized:
        layers = _layers_for_path(path)
        file_layers[path] = layers
        for layer in layers:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

    layers = sorted(layer_counts)
    primary_layer = ""
    if layer_counts:
        primary_layer = sorted(
            layer_counts,
            key=lambda layer: (-layer_counts[layer], layer),
        )[0]
    return {
        "changed_files": normalized,
        "file_layers": file_layers,
        "layers": layers,
        "layer_counts": dict(sorted(layer_counts.items())),
        "primary_layer": primary_layer,
    }


def validation_ladder_for_changed_files(
    changed_files: list[str],
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Build a cheap-to-expensive validation ladder for a Codex diff."""

    root = Path(repo_root)
    component_delta = classify_component_delta(changed_files)
    layers = set(component_delta["layers"])
    normalized = component_delta["changed_files"]
    phases: list[dict[str, Any]] = []

    changed_py = [
        path for path in normalized if path.endswith(".py") and (root / path).is_file()
    ]
    if changed_py:
        phases.append(
            {
                "name": "import",
                "commands": [
                    "python -m py_compile "
                    + " ".join(shlex.quote(path) for path in changed_py)
                ],
                "rationale": "Changed Python files must at least parse before broader gates run.",
            }
        )

    changed_rust_crates = sorted(
        {
            str(Path(path).parents[1])
            for path in normalized
            if path.startswith("crates/")
            and path.endswith(".rs")
            and len(Path(path).parts) >= 4
            and (root / path).is_file()
        }
    )
    for crate_path in changed_rust_crates:
        manifest = Path(crate_path) / "Cargo.toml"
        if (root / manifest).is_file():
            phases.append(
                {
                    "name": "rust-check",
                    "commands": [
                        "cargo +stable check --manifest-path "
                        + shlex.quote(str(manifest))
                    ],
                    "rationale": "Changed Rust Worker core code must compile before broader gates run.",
                }
            )

    changed_tests = [
        path
        for path in normalized
        if path.startswith("tests/") and path.endswith(".py") and (root / path).is_file()
    ]
    if changed_tests:
        phases.append(
            {
                "name": "targeted-tests",
                "commands": [
                    "pytest " + " ".join(shlex.quote(path) for path in changed_tests) + " -v"
                ],
                "rationale": "Changed tests should execute directly before broad validation.",
            }
        )
    elif _path_exists(root, "tests"):
        phases.append(
            {
                "name": "project-tests",
                "commands": ["pytest tests/ -v"],
                "rationale": "Shared harness behavior changed; run the project suite.",
            }
        )

    if layers & {"meta_updater", "codex_update", "codex_packet", "campaign_runner", "script_orchestration"}:
        if _path_exists(root, "scripts/run_campaign.py"):
            phases.append(
                {
                    "name": "campaign-dry-run",
                    "commands": [
                        (
                            "python scripts/run_campaign.py --dry-run "
                            "--tasks fix-git,vulnerable-secret "
                            "--worker-role worker_deepseek"
                        )
                    ],
                    "rationale": "Meta/script edits must preserve campaign command construction.",
                }
            )

    runtime_layers = {
        "worker_loop",
        "harbor_adapter",
        "tool_schema",
        "tool_impl",
        "planning",
        "context_compaction",
        "recovery",
        "entrypoint",
        "verification",
        "skill_loading",
    }
    if layers & runtime_layers and _path_exists(root, "scripts/run_trial.py"):
        phases.append(
            {
                "name": "harbor-smoke-dry-run",
                "commands": [
                    (
                        "python scripts/run_trial.py "
                        "--path terminal-bench-tasks/terminal-bench "
                        "--task fix-git --dry-run --worker-role worker_deepseek"
                    )
                ],
                "rationale": "Worker/runtime edits must still produce a valid Harbor job shape.",
            }
        )

    if layers & {"regression_gate"}:
        if _path_exists(root, "scripts/regression_check.py"):
            phases.append(
                {
                    "name": "regression-gate-dry-run",
                    "commands": [
                        "python scripts/regression_check.py --dry-run --lane smoke",
                        (
                            "python scripts/regression_check.py --dry-run "
                            "--lane smoke --selection-policy adaptive"
                        ),
                    ],
                    "rationale": (
                        "Regression-gate edits must preserve command planning. "
                        "Campaign-scoped Codex updates inject the real same-model "
                        "solved-task regression gate separately."
                    ),
                }
            )

    commands = _unique(
        command
        for phase in phases
        for command in phase.get("commands", [])
        if str(command).strip()
    )
    return {
        "component_delta": component_delta,
        "phases": phases,
        "commands": commands,
        "policy": (
            "Import/targeted tests first, campaign dry-run for orchestration, "
            "Harbor dry-run for Worker runtime, then solved-task regression slices."
        ),
    }


def validation_ladder_contract() -> dict[str, Any]:
    """Static contract included in Codex packets before changed files are known."""

    return {
        "objective": (
            "Codex must choose validation commands from the changed-file ladder and "
            "the host updater will re-run the applicable commands before accepting a patch."
        ),
        "rules": [
            "Docs/tests-only edits need targeted pytest or a direct static check.",
            "meta/ or scripts/ edits need campaign dry-run plus focused tests.",
            "bench/ or harness runtime edits need project tests, campaign dry-run, and Harbor dry-run smoke.",
            "Campaign-scoped updates inject a real same-model solved-task regression gate; regression-gate edits also need command-planning dry-runs.",
        ],
        "host_gate": (
            "The updater derives the final host validation ladder from review.changed_files; "
            "missing local scripts or test directories are skipped only when absent in that checkout."
        ),
    }


def merge_validation_commands(
    base_commands: list[str],
    ladder: dict[str, Any],
) -> list[str]:
    return _unique([*base_commands, *list(ladder.get("commands") or [])])


def _layers_for_path(path: str) -> list[str]:
    layers: list[str] = []
    for prefix, layer in LAYER_BY_PREFIX:
        if path == prefix or path.startswith(prefix):
            layers.append(layer)
    if not layers:
        layers.append("other")
    return _unique(layers)


def _normalize_path(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


def _path_exists(root: Path, relative: str) -> bool:
    return (root / relative).exists()


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value)
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
