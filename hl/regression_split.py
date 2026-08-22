"""Deterministic held-in / held-out partition for regression snapshots.

The Self-Harness acceptance contract (Weng, "Harness Engineering for
Self-Improvement") accepts a harness edit only if it introduces no regression on
both a held-in split (D_in, shown to the proposer) and a held-out split (D_out,
hidden from the proposer). D_out catches collateral damage / overfitting: if an
edit only patches what it was shown, the hidden solved tasks reveal the breakage.

The partition must be deterministic (stable across runs and processes) and
target-agnostic (no per-task literals), so a task's split membership does not
drift between the proposer-context step and the gate step.
"""

from __future__ import annotations

import hashlib

_HASH_DENOMINATOR = 10_000


def _task_holdout_score(task_id: str, seed: int) -> int:
    """Stable 0.._HASH_DENOMINATOR-1 score for a task under a seed."""
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()
    return int(digest[:8], 16) % _HASH_DENOMINATOR


def is_holdout_task(task_id: str, *, fraction: float, seed: int = 0) -> bool:
    """Return True when ``task_id`` belongs to the held-out (D_out) split.

    ``fraction`` is the target held-out share in [0, 1]. ``fraction <= 0`` means
    no held-out split (every task is held-in); ``fraction >= 1`` holds out all.
    """
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    threshold = int(fraction * _HASH_DENOMINATOR)
    return _task_holdout_score(str(task_id), seed) < threshold


def partition_holdout_tasks(
    task_ids: list[str],
    *,
    fraction: float,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Split ``task_ids`` into (held_in, held_out), preserving input order."""
    held_in: list[str] = []
    held_out: list[str] = []
    for task_id in task_ids:
        if is_holdout_task(task_id, fraction=fraction, seed=seed):
            held_out.append(task_id)
        else:
            held_in.append(task_id)
    return held_in, held_out
