"""Tests for deterministic held-in / held-out regression partition."""

from __future__ import annotations

from hl.regression_split import (
    is_holdout_task,
    partition_holdout_tasks,
)


def test_zero_fraction_holds_out_nothing():
    tasks = [f"task-{i}" for i in range(50)]
    held_in, held_out = partition_holdout_tasks(tasks, fraction=0.0)
    assert held_in == tasks
    assert held_out == []


def test_full_fraction_holds_out_everything():
    tasks = [f"task-{i}" for i in range(20)]
    held_in, held_out = partition_holdout_tasks(tasks, fraction=1.0)
    assert held_in == []
    assert held_out == tasks


def test_partition_is_deterministic():
    tasks = [f"task-{i}" for i in range(200)]
    a_in, a_out = partition_holdout_tasks(tasks, fraction=0.3, seed=7)
    b_in, b_out = partition_holdout_tasks(tasks, fraction=0.3, seed=7)
    assert a_in == b_in
    assert a_out == b_out


def test_partition_is_a_true_split_preserving_order():
    tasks = [f"task-{i}" for i in range(200)]
    held_in, held_out = partition_holdout_tasks(tasks, fraction=0.25, seed=1)
    # Disjoint, complete, order-preserving.
    assert set(held_in).isdisjoint(held_out)
    assert sorted(held_in + held_out) == sorted(tasks)
    assert held_in == [t for t in tasks if t in set(held_in)]


def test_fraction_roughly_matches_holdout_share():
    tasks = [f"task-{i}" for i in range(1000)]
    _, held_out = partition_holdout_tasks(tasks, fraction=0.3, seed=3)
    share = len(held_out) / len(tasks)
    # Deterministic hashing should land near the target fraction.
    assert 0.24 <= share <= 0.36


def test_seed_changes_membership():
    tasks = [f"task-{i}" for i in range(400)]
    _, out_a = partition_holdout_tasks(tasks, fraction=0.3, seed=1)
    _, out_b = partition_holdout_tasks(tasks, fraction=0.3, seed=2)
    assert set(out_a) != set(out_b)


def test_is_holdout_task_matches_partition():
    tasks = [f"task-{i}" for i in range(100)]
    _, held_out = partition_holdout_tasks(tasks, fraction=0.4, seed=5)
    held_out_set = set(held_out)
    for task in tasks:
        assert is_holdout_task(task, fraction=0.4, seed=5) == (task in held_out_set)


def _seed_snapshots(tmp_path, tasks):
    import json as _json

    from hl.memory import FileSystemMemory

    memory = FileSystemMemory(base_path=str(tmp_path / "trials"))
    memory.regressions_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        (memory.regressions_dir / f"{task}.json").write_text(
            _json.dumps({"task_id": task, "validation_status": "stable"})
        )
    return memory


def test_loop_regression_contracts_exclude_holdout(tmp_path):
    import json as _json

    from hl.loop import HLLoop

    tasks = [f"task-{i}" for i in range(40)]
    memory = _seed_snapshots(tmp_path, tasks)
    loop = HLLoop(memory=memory)
    loop.regression_holdout_fraction = 0.25
    loop.regression_holdout_seed = 0

    contract_tasks = {
        _json.loads(open(path).read())["task_id"]
        for path in loop._regression_contracts()
    }
    _, held_out = partition_holdout_tasks(tasks, fraction=0.25, seed=0)

    assert set(held_out)
    # Held-out tasks must be hidden from the proposer's regression contracts.
    assert contract_tasks.isdisjoint(set(held_out))
    assert contract_tasks == set(tasks) - set(held_out)


def test_loop_regression_contracts_include_all_when_disabled(tmp_path):
    from hl.loop import HLLoop

    tasks = [f"task-{i}" for i in range(20)]
    memory = _seed_snapshots(tmp_path, tasks)
    loop = HLLoop(memory=memory)
    loop.regression_holdout_fraction = 0.0

    assert len(loop._regression_contracts()) == len(tasks)

