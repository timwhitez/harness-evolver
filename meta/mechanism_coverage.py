"""Deterministic evidence that a mission mechanism is already implemented."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkerPolicyCoverageSpec:
    signature: str
    policy_path: str
    policy_markers: tuple[str, ...]
    test_path: str
    test_markers: tuple[str, ...]


_WORKER_POLICY_COVERAGE: tuple[WorkerPolicyCoverageSpec, ...] = (
    WorkerPolicyCoverageSpec(
        "structured_csv_table_contract",
        "crates/hl-worker-core/src/main.rs",
        ("structured_csv_table_contracts_from_text", "structured_csv_table_preflight"),
        "tests/test_models_and_worker_policy.py",
        ("repair_structured_csv_table_contract", "structured_csv_table_contract:"),
    ),
    WorkerPolicyCoverageSpec(
        "missing_output_artifact_contract",
        "crates/hl-worker-core/src/main.rs",
        ("missing_output_artifact_contract", "repair_missing_output_artifact_contract"),
        "tests/test_models_and_worker_policy.py",
        ("repair_missing_output_artifact_contract", "missing_output_artifact_contract:"),
    ),
    WorkerPolicyCoverageSpec(
        "stan_dependency_stack_pivot_mechanism",
        "crates/hl-worker-core/src/main.rs",
        ("stan_dependency_stack_pivot_mechanism", "pystan_analysis.py"),
        "tests/test_models_and_worker_policy.py",
        ("stan_dependency_stack_pivot_mechanism",),
    ),
    WorkerPolicyCoverageSpec(
        "dependency_loop_without_deliverable_progress_mechanism",
        "crates/hl-worker-core/src/main.rs",
        ("dependency_loop_without_deliverable_progress_mechanism",),
        "tests/test_campaign_runner.py",
        ("dependency_loop_without_deliverable_progress_mechanism",),
    ),
    WorkerPolicyCoverageSpec(
        "dna_assembly_primer_contract",
        "crates/hl-worker-core/src/main.rs",
        ("dna_assembly_primer_contract", "repair_dna_assembly_primer_contract"),
        "tests/test_models_and_worker_policy.py",
        ("repair_dna_assembly_primer_contract", "dna_assembly_primer_contract:"),
    ),
    WorkerPolicyCoverageSpec(
        "dna_insert_primer_pair_contract",
        "crates/hl-worker-core/src/main.rs",
        ("dna_insert_primer_pair_contract", "repair_dna_insert_primer_pair_contract"),
        "tests/test_models_and_worker_policy.py",
        ("repair_dna_insert_primer_pair_contract", "dna_insert_primer_pair_contract:"),
    ),
    WorkerPolicyCoverageSpec(
        "image_similarity_contract",
        "crates/hl-worker-core/src/main.rs",
        ("image_similarity_contract", "image_similarity_preflight"),
        "tests/test_models_and_worker_policy.py",
        ("image_similarity_contract:",),
    ),
    WorkerPolicyCoverageSpec(
        "ml_cv_heavy_import_pivot_mechanism",
        "crates/hl-worker-core/src/main.rs",
        ("Treat ML/CV imports and package probes as optional", "mobile_sam"),
        "tests/test_models_and_worker_policy.py",
        ("ml_cv_heavy_import_pivot_mechanism",),
    ),
    WorkerPolicyCoverageSpec(
        "tokenized_output_file_contract",
        "crates/hl-worker-core/src/main.rs",
        ("fn tokenized_output_file_contract", "repair_tokenized_output_file_contract"),
        "crates/hl-worker-core/src/main.rs",
        (
            "fn ctrf_summary_extracts_tokenized_output_file_contract",
            "fn live_verifier_semantic_contracts_extract_tokenized_output_file_contracts",
        ),
    ),
    WorkerPolicyCoverageSpec(
        "dataset_shard_generalization_contract",
        "crates/hl-worker-core/src/main.rs",
        (
            "fn dataset_shard_generalization_contract",
            "dataset_shard_generalization_preflight_contracts",
        ),
        "crates/hl-worker-core/src/main.rs",
        ("fn live_verifier_semantic_contracts_extract_dataset_shard_generalization_contracts",),
    ),
    WorkerPolicyCoverageSpec(
        "deliverable_size_cap_contract",
        "crates/hl-worker-core/src/main.rs",
        (
            "fn deliverable_size_cap_contract",
            "extract_declared_deliverable_size_cap_contract",
        ),
        "crates/hl-worker-core/src/main.rs",
        (
            "fn declared_deliverable_size_cap_contract_blocks_done_until_size_preflight",
            "fn ctrf_summary_extracts_deliverable_size_cap_contract",
        ),
    ),
    WorkerPolicyCoverageSpec(
        "cross_arch_toolchain_pivot_mechanism",
        "crates/hl-worker-core/src/main.rs",
        (
            "fn cross_arch_timeout_reason",
            "fn dependency_checkpoint_should_use_cross_arch_pivot",
        ),
        "crates/hl-worker-core/src/main.rs",
        (
            "fn dependency_checkpoint_adds_mips_binutils_cross_arch_pivot",
            "fn blocks_repeated_cross_arch_timeout_path_but_allows_small_smoke",
        ),
    ),
    WorkerPolicyCoverageSpec(
        "terminal_environment_unavailable_after_dependency_loop_mechanism",
        "crates/hl-worker-core/src/main.rs",
        ("fn terminal_environment_failure_text", "terminal_environment_unavailable_result"),
        "crates/hl-worker-core/src/main.rs",
        ("fn terminal_environment_unavailable_observation_preserves_outer_loop_ownership",),
    ),
    WorkerPolicyCoverageSpec(
        "verifier_runtime_prepare_timeout",
        "bench/_harbor_issue9_base.py",
        (
            "def _has_verifier_runtime_prepare_timeout_marker",
            "def _with_verifier_runtime_prepare_timeout_metadata",
        ),
        "tests/test_campaign_runner.py",
        (
            "def test_iteration_analysis_labels_infrastructure_timeout_weakness_contribution",
            "verifier_runtime_prepare_timeout=True",
        ),
    ),
)

_WORKER_COVERAGE_CACHE: dict[
    str, tuple[tuple[tuple[str, int], ...], list[dict[str, Any]]]
] = {}
_ACCEPTED_COVERAGE_CACHE: dict[str, list[dict[str, Any]]] = {}


def deterministic_worker_policy_coverage(repo_root: str | Path) -> list[dict[str, Any]]:
    """Return mechanisms with both executable policy and deterministic test evidence."""

    root = Path(repo_root)
    cache_key = str(root.resolve())
    coverage_paths = {
        root / relative_path
        for spec in _WORKER_POLICY_COVERAGE
        for relative_path in (spec.policy_path, spec.test_path)
    }
    stamp = tuple(
        sorted((str(path), _mtime_ns(path)) for path in coverage_paths)
    )
    cached = _WORKER_COVERAGE_CACHE.get(cache_key)
    if cached is not None and cached[0] == stamp:
        return [dict(entry) for entry in cached[1]]
    evidence: list[dict[str, Any]] = []
    for spec in _WORKER_POLICY_COVERAGE:
        policy_path = root / spec.policy_path
        test_path = root / spec.test_path
        if not policy_path.is_file() or not test_path.is_file():
            continue
        policy_text = policy_path.read_text(errors="replace")
        test_text = test_path.read_text(errors="replace")
        if not all(marker in policy_text for marker in spec.policy_markers):
            continue
        if not all(marker in test_text for marker in spec.test_markers):
            continue
        evidence.append(
            {
                "signature": spec.signature,
                "source": "worker_policy_and_test",
                "policy_path": spec.policy_path,
                "test_path": spec.test_path,
            }
        )
    _WORKER_COVERAGE_CACHE[cache_key] = (stamp, evidence)
    return [dict(entry) for entry in evidence]


def accepted_update_coverage(memory_path: str | Path) -> list[dict[str, Any]]:
    """Return exact mission ids/categories previously accepted by host review."""

    diffs = Path(memory_path) / "diffs"
    if not diffs.is_dir():
        return []
    cache_key = str(Path(memory_path).resolve())
    cached = _ACCEPTED_COVERAGE_CACHE.get(cache_key)
    if cached is not None:
        return [dict(entry) for entry in cached]
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for review_path in sorted(diffs.glob("codex_packet_*/review.json")):
        review = _read_json(review_path)
        if not review.get("accepted"):
            continue
        run_dir = review_path.parent
        packet_id = run_dir.name
        candidates = _accepted_mission_markers(run_dir)
        for signature in candidates:
            normalized = signature.strip().lower()
            if len(normalized) < 3 or normalized in seen:
                continue
            seen.add(normalized)
            evidence.append(
                {
                    "signature": normalized,
                    "source": "accepted_update_memory",
                    "packet_id": packet_id,
                    "review_path": str(review_path),
                }
            )
    _ACCEPTED_COVERAGE_CACHE[cache_key] = evidence
    return [dict(entry) for entry in evidence]


def covered_mechanism_evidence(
    repo_root: str | Path,
    memory_path: str | Path,
) -> list[dict[str, Any]]:
    combined = [
        *deterministic_worker_policy_coverage(repo_root),
        *accepted_update_coverage(memory_path),
    ]
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in combined:
        signature = str(entry.get("signature") or "").strip().lower()
        if len(signature) < 3 or signature in seen:
            continue
        seen.add(signature)
        evidence.append({**entry, "signature": signature})
    return evidence


def clear_coverage_cache(memory_path: str | Path | None = None) -> None:
    """Invalidate accepted-update coverage after a review record is written."""

    if memory_path is None:
        _ACCEPTED_COVERAGE_CACHE.clear()
        return
    _ACCEPTED_COVERAGE_CACHE.pop(str(Path(memory_path).resolve()), None)


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _accepted_mission_markers(run_dir: Path) -> list[str]:
    markers: list[str] = []
    record = _read_json(run_dir / "update_record.json")
    selection = (
        (record.get("update_decision_inputs") or {}).get("mission_selection") or {}
    )
    if isinstance(selection, dict):
        markers.extend(
            str(selection.get(key) or "")
            for key in ("selected_candidate_id", "selected_failure_category")
        )
    manifest = _read_json(run_dir / "change_manifest.json")
    manifest_selection = (manifest.get("root_cause") or {}).get("mission_selection") or {}
    if isinstance(manifest_selection, dict):
        markers.extend(
            str(manifest_selection.get(key) or "")
            for key in ("selected_candidate_id", "selected_failure_category")
        )
    return [marker for marker in markers if marker.strip()]
