"""TaskCatalog — registry of all 89 TerminalBench 2.0 tasks.

Tasks are organized by domain and difficulty.
The catalog tracks which tasks are solved (regression targets)
and which remain as improvement targets.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from random import Random


DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "unknown": 3}


class TaskStatus(str, Enum):
    UNSOLVED = "unsolved"
    SOLVED = "solved"  # Becomes a regression target
    FAILED = "failed"  # Tried but couldn't solve
    SKIPPED = "skipped"


@dataclass
class TaskEntry:
    """A single TerminalBench task entry."""

    task_id: str
    domain: str
    difficulty: str  # easy / medium / hard
    status: TaskStatus = TaskStatus.UNSOLVED
    score: float = 0.0
    attempts: int = 0
    last_trial_id: str = ""
    notes: str = ""


@dataclass
class TaskCatalog:
    """Registry of all TerminalBench 2.0 tasks with their status.

    This is a living registry — the meta-agent updates it as
    tasks are solved, failed, or regress.
    """

    tasks: dict[str, TaskEntry] = field(default_factory=dict)

    @classmethod
    def from_terminal_bench_path(cls, dataset_path: str | Path) -> "TaskCatalog":
        """Load the local TerminalBench task catalog from task directories."""
        root = Path(dataset_path)
        catalog = cls()
        if not root.exists():
            raise FileNotFoundError(f"TerminalBench task path not found: {root}")

        for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            task_toml = task_dir / "task.toml"
            metadata: dict[str, object] = {}
            if task_toml.exists():
                try:
                    data = tomllib.loads(task_toml.read_text())
                    raw_metadata = data.get("metadata", {})
                    if isinstance(raw_metadata, dict):
                        metadata = raw_metadata
                except tomllib.TOMLDecodeError:
                    metadata = {}

            domain = str(metadata.get("category") or "unknown")
            difficulty = str(metadata.get("difficulty") or "unknown")
            catalog.register(
                task_id=task_dir.name,
                domain=domain,
                difficulty=difficulty,
            )

        return catalog

    @classmethod
    def load_or_empty(cls, dataset_path: str | Path) -> "TaskCatalog":
        try:
            return cls.from_terminal_bench_path(dataset_path)
        except FileNotFoundError:
            return cls()

    def register(self, task_id: str, domain: str, difficulty: str) -> TaskEntry:
        entry = TaskEntry(task_id=task_id, domain=domain, difficulty=difficulty)
        self.tasks[task_id] = entry
        return entry

    def list_task_ids(self) -> list[str]:
        return sorted(self.tasks)

    def select_curriculum(
        self,
        task_set: str,
        *,
        domains: list[str] | None = None,
        difficulties: list[str] | None = None,
        max_tasks: int | None = None,
    ) -> list[str]:
        """Select task ids for a named campaign curriculum.

        ``max_tasks`` is retained as a compatibility/audit input only. It must
        not shrink the task pool for master, sub-agent, validation/regression,
        mission-debug, context, or Worker loops.
        """
        _ = max_tasks
        normalized = task_set.replace("_", "-").lower()
        entries = self._filtered_entries(domains=domains, difficulties=difficulties)
        if normalized == "full":
            selected = self._sort_entries(entries)
            return [entry.task_id for entry in selected]
        if normalized == "smoke":
            selected = self._sort_entries(entries)
            return [entry.task_id for entry in selected]
        if normalized == "hard-focus":
            hard_entries = [
                entry for entry in entries if entry.difficulty.lower() == "hard"
            ]
            selected = self._sort_entries(hard_entries or entries)
            return [entry.task_id for entry in selected]
        if normalized == "domain-balanced":
            return self._select_domain_balanced(entries)
        raise ValueError(
            f"Unknown task set {task_set!r}; expected smoke, domain-balanced, "
            "hard-focus, or full"
        )

    def select_random(
        self,
        *,
        count: int,
        seed: str | int | None = None,
        domains: list[str] | None = None,
        difficulties: list[str] | None = None,
    ) -> list[str]:
        """Return the full filtered catalog in a stable random order.

        ``count`` is retained as an audit/scheduling reference for callers that
        need a per-round batch size. It must not shrink the campaign task pool.
        """
        _ = count
        entries = self._sort_entries(
            self._filtered_entries(domains=domains, difficulties=difficulties)
        )
        if not entries:
            return []
        task_ids = [entry.task_id for entry in entries]
        rng = Random(seed)
        rng.shuffle(task_ids)
        return task_ids

    def select_by_indices(
        self,
        indices: list[int],
        *,
        domains: list[str] | None = None,
        difficulties: list[str] | None = None,
    ) -> list[str]:
        """Select tasks by 1-based index in the stable full-catalog order."""
        entries = self._sort_entries(
            self._filtered_entries(domains=domains, difficulties=difficulties)
        )
        if not entries:
            return []
        task_ids = [entry.task_id for entry in entries]
        selected: list[str] = []
        for index in indices:
            if index <= 0 or index > len(task_ids):
                raise IndexError(
                    f"Task index {index} is out of range; valid range is 1-{len(task_ids)}"
                )
            selected.append(task_ids[index - 1])
        return list(dict.fromkeys(selected))

    def get(self, task_id: str) -> TaskEntry | None:
        return self.tasks.get(task_id)

    def list_by_domain(self, domain: str) -> list[TaskEntry]:
        return [t for t in self.tasks.values() if t.domain == domain]

    def list_by_difficulty(self, difficulty: str) -> list[TaskEntry]:
        return [t for t in self.tasks.values() if t.difficulty == difficulty]

    def list_unsolved(self) -> list[TaskEntry]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.UNSOLVED]

    def list_solved(self) -> list[TaskEntry]:
        """Solved tasks are regression targets — must not regress."""
        return [t for t in self.tasks.values() if t.status == TaskStatus.SOLVED]

    def mark_solved(self, task_id: str, score: float, trial_id: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.SOLVED
            self.tasks[task_id].score = score
            self.tasks[task_id].last_trial_id = trial_id

    def mark_failed(self, task_id: str, trial_id: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.FAILED
            self.tasks[task_id].attempts += 1
            self.tasks[task_id].last_trial_id = trial_id

    def record_attempt(self, task_id: str, score: float, trial_id: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].attempts += 1
            self.tasks[task_id].score = max(self.tasks[task_id].score, score)
            self.tasks[task_id].last_trial_id = trial_id

    @property
    def solved_count(self) -> int:
        return len(self.list_solved())

    @property
    def total_count(self) -> int:
        return len(self.tasks)

    @property
    def overall_score(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(t.score for t in self.tasks.values()) / len(self.tasks)

    def _filtered_entries(
        self,
        *,
        domains: list[str] | None = None,
        difficulties: list[str] | None = None,
    ) -> list[TaskEntry]:
        domain_set = {item.lower() for item in domains or [] if item}
        difficulty_set = {item.lower() for item in difficulties or [] if item}
        entries = list(self.tasks.values())
        if domain_set:
            entries = [entry for entry in entries if entry.domain.lower() in domain_set]
        if difficulty_set:
            entries = [
                entry for entry in entries if entry.difficulty.lower() in difficulty_set
            ]
        return entries

    def _sort_entries(self, entries: list[TaskEntry]) -> list[TaskEntry]:
        return sorted(
            entries,
            key=lambda entry: (
                DIFFICULTY_ORDER.get(entry.difficulty.lower(), 99),
                entry.domain,
                entry.task_id,
            ),
        )

    def _select_domain_balanced(
        self,
        entries: list[TaskEntry],
    ) -> list[str]:
        by_domain: dict[str, list[TaskEntry]] = {}
        for entry in self._sort_entries(entries):
            by_domain.setdefault(entry.domain, []).append(entry)

        selected: list[str] = []
        while by_domain:
            made_progress = False
            for domain in sorted(list(by_domain)):
                bucket = by_domain[domain]
                if not bucket:
                    by_domain.pop(domain, None)
                    continue
                selected.append(bucket.pop(0).task_id)
                made_progress = True
            if not made_progress:
                break
        return selected
