"""Unified task-identity resolution across snapshots, Harbor receipts, and CLI gates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import posixpath
from typing import Any


def _normalize_task_text(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return ""
    if "/" not in text:
        return "" if text in {".", ".."} else text
    normalized = posixpath.normpath(text)
    if normalized in {".", "..", "/"}:
        return ""
    return normalized


@dataclass(frozen=True)
class ResolvedTaskIdentity:
    """Result of resolving one raw task identifier within a dataset boundary."""

    raw: str
    canonical_id: str | None
    bare_name: str | None
    dataset_name: str | None
    status: str  # "resolved" | "missing" | "ambiguous" | "contradictory"
    in_dataset: bool = False
    candidates: tuple[str, ...] = ()
    reason: str = ""


class DatasetTaskResolver:
    """Resolve bare, namespaced, and path task IDs using one dataset-scoped rule.

    Rules:
    - Empty/missing identifiers never resolve and never compare equal.
    - A bare task name (no ``/``) maps to a namespaced ``<dataset>/<task>`` ONLY
      when it is uniquely resolvable within the active dataset directory.
    - If a bare task name exists in multiple datasets (either under a multi-dataset
      directory or across known dataset-qualified task IDs when no single dataset
      owns it), it resolves to ``ambiguous`` and must be rejected.
    - An explicitly qualified identifier (``<dataset>/<task>`` or path) keeps its
      dataset/path identity and never collapses into a different dataset.
    """

    def __init__(
        self,
        *,
        dataset_root: Path | None = None,
        canonical_ids: set[str] | None = None,
        by_bare_dataset: dict[str, set[str]] | None = None,
        by_bare_known: dict[str, set[str]] | None = None,
        by_abs_path: dict[str, str] | None = None,
        by_rel_path: dict[str, str] | None = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.canonical_ids: set[str] = set(canonical_ids or ())
        self.by_bare_dataset: dict[str, set[str]] = {
            k: set(v) for k, v in (by_bare_dataset or {}).items()
        }
        self.by_bare_known: dict[str, set[str]] = {
            k: set(v) for k, v in (by_bare_known or {}).items()
        }
        self.by_abs_path: dict[str, str] = dict(by_abs_path or {})
        self.by_rel_path: dict[str, str] = dict(by_rel_path or {})

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str | Path | None,
        *,
        known_task_ids: Iterable[str] | None = None,
    ) -> "DatasetTaskResolver":
        canonical_ids: set[str] = set()
        by_bare_dataset: dict[str, set[str]] = {}
        by_bare_known: dict[str, set[str]] = {}
        by_abs_path: dict[str, str] = {}
        by_rel_path: dict[str, str] = {}
        resolved_root: Path | None = None

        if dataset_path:
            root = Path(dataset_path)
            if root.exists() and root.is_dir():
                resolved_root = root.resolve()
                root_name = resolved_root.name

                def _register_task(dataset_name: str, task_dir: Path) -> None:
                    bare = task_dir.name
                    canonical = f"{dataset_name}/{bare}"
                    canonical_ids.add(canonical)
                    by_bare_dataset.setdefault(bare, set()).add(canonical)

                    abs_norm = _normalize_task_text(task_dir.resolve())
                    if abs_norm:
                        by_abs_path[abs_norm] = canonical
                    raw_norm = _normalize_task_text(task_dir)
                    if raw_norm:
                        by_rel_path[raw_norm] = canonical
                    try:
                        rel_from_root = _normalize_task_text(
                            task_dir.resolve().relative_to(resolved_root)
                        )
                        if rel_from_root and "/" in rel_from_root:
                            by_rel_path[rel_from_root] = canonical
                    except ValueError:
                        pass
                    try:
                        rel_from_parent = _normalize_task_text(
                            task_dir.resolve().relative_to(resolved_root.parent)
                        )
                        if rel_from_parent:
                            by_rel_path[rel_from_parent] = canonical
                    except ValueError:
                        pass
                    by_rel_path[canonical] = canonical

                # Pass 1: discover tasks with explicit task.toml at depth 1 or depth 2.
                found_with_toml = False
                for child in sorted(
                    p for p in resolved_root.iterdir() if p.is_dir() and not p.name.startswith(".")
                ):
                    if (child / "task.toml").is_file():
                        _register_task(root_name, child)
                        found_with_toml = True
                    else:
                        for grandchild in sorted(
                            p
                            for p in child.iterdir()
                            if p.is_dir() and not p.name.startswith(".")
                        ):
                            if (grandchild / "task.toml").is_file():
                                _register_task(child.name, grandchild)
                                found_with_toml = True

                # Pass 2: fallback for lightweight test/fixture directories without task.toml.
                if not found_with_toml:
                    children = sorted(
                        p
                        for p in resolved_root.iterdir()
                        if p.is_dir() and not p.name.startswith(".")
                    )
                    for child in children:
                        sub_dirs = sorted(
                            p
                            for p in child.iterdir()
                            if p.is_dir() and not p.name.startswith(".")
                        )
                        has_direct_files = any(p.is_file() for p in child.iterdir())
                        if sub_dirs and not has_direct_files:
                            for grandchild in sub_dirs:
                                _register_task(child.name, grandchild)
                        else:
                            _register_task(root_name, child)

        for raw_known in known_task_ids or ():
            norm_known = _normalize_task_text(raw_known)
            if "/" in norm_known:
                bare = norm_known.rsplit("/", 1)[-1]
                if bare:
                    by_bare_known.setdefault(bare, set()).add(norm_known)

        return cls(
            dataset_root=resolved_root,
            canonical_ids=canonical_ids,
            by_bare_dataset=by_bare_dataset,
            by_bare_known=by_bare_known,
            by_abs_path=by_abs_path,
            by_rel_path=by_rel_path,
        )

    def resolve(self, raw_task_id: object) -> ResolvedTaskIdentity:
        raw_text = str(raw_task_id or "").strip()
        normalized = _normalize_task_text(raw_text)
        if not normalized:
            return ResolvedTaskIdentity(
                raw=raw_text,
                canonical_id=None,
                bare_name=None,
                dataset_name=None,
                status="missing",
                reason="missing task identity",
            )

        if "/" in normalized:
            # 1. Match absolute path (resolved or normalized).
            if normalized in self.by_abs_path:
                canonical = self.by_abs_path[normalized]
                ds_name, bare = canonical.rsplit("/", 1)
                return ResolvedTaskIdentity(
                    raw=raw_text,
                    canonical_id=canonical,
                    bare_name=bare,
                    dataset_name=ds_name,
                    status="resolved",
                    in_dataset=True,
                )
            try:
                resolved_candidate = _normalize_task_text(Path(raw_text).resolve())
            except OSError:
                resolved_candidate = ""
            if resolved_candidate and resolved_candidate in self.by_abs_path:
                canonical = self.by_abs_path[resolved_candidate]
                ds_name, bare = canonical.rsplit("/", 1)
                return ResolvedTaskIdentity(
                    raw=raw_text,
                    canonical_id=canonical,
                    bare_name=bare,
                    dataset_name=ds_name,
                    status="resolved",
                    in_dataset=True,
                )

            # 2. Match exact dataset-relative or canonical `<dataset>/<task>`.
            if normalized in self.by_rel_path:
                canonical = self.by_rel_path[normalized]
                ds_name, bare = canonical.rsplit("/", 1)
                return ResolvedTaskIdentity(
                    raw=raw_text,
                    canonical_id=canonical,
                    bare_name=bare,
                    dataset_name=ds_name,
                    status="resolved",
                    in_dataset=True,
                )

            # 3. Explicitly qualified path/namespace outside active dataset: keep verbatim.
            parts = normalized.rsplit("/", 1)
            bare = parts[-1]
            parent = parts[0].rsplit("/", 1)[-1] if parts[0] else None
            return ResolvedTaskIdentity(
                raw=raw_text,
                canonical_id=normalized,
                bare_name=bare or None,
                dataset_name=parent or None,
                status="resolved",
                in_dataset=False,
            )

        # Bare task name: only resolve to `<dataset>/<task>` when uniquely owned by
        # the active dataset; reject immediately if ambiguous across datasets.
        dataset_matches = sorted(self.by_bare_dataset.get(normalized, ()))
        if len(dataset_matches) > 1:
            return ResolvedTaskIdentity(
                raw=raw_text,
                canonical_id=None,
                bare_name=normalized,
                dataset_name=None,
                status="ambiguous",
                candidates=tuple(dataset_matches),
                reason=(
                    f"bare task name {normalized!r} is ambiguous across datasets: "
                    + ", ".join(dataset_matches)
                ),
            )
        if len(dataset_matches) == 1:
            canonical = dataset_matches[0]
            ds_name, bare = canonical.rsplit("/", 1)
            return ResolvedTaskIdentity(
                raw=raw_text,
                canonical_id=canonical,
                bare_name=bare,
                dataset_name=ds_name,
                status="resolved",
                in_dataset=True,
            )

        known_matches = sorted(self.by_bare_known.get(normalized, ()))
        if len(known_matches) > 1:
            return ResolvedTaskIdentity(
                raw=raw_text,
                canonical_id=None,
                bare_name=normalized,
                dataset_name=None,
                status="ambiguous",
                candidates=tuple(known_matches),
                reason=(
                    f"bare task name {normalized!r} is ambiguous across datasets: "
                    + ", ".join(known_matches)
                ),
            )

        return ResolvedTaskIdentity(
            raw=raw_text,
            canonical_id=normalized,
            bare_name=normalized,
            dataset_name=None,
            status="resolved",
            in_dataset=False,
        )

    def harbor_include_task_name(self, task_id: str) -> str:
        """Return the task filter name to pass to Harbor ``--include-task-name``."""
        if self.dataset_root is None:
            return task_id
        resolved = self.resolve(task_id)
        if (
            resolved.status == "resolved"
            and resolved.in_dataset
            and resolved.dataset_name == self.dataset_root.name
            and resolved.bare_name
        ):
            return resolved.bare_name
        return task_id


def resolve_trial_task_identity(
    trial: Any,
    resolver: DatasetTaskResolver,
) -> ResolvedTaskIdentity:
    """Resolve a TrialResult's authoritative task identity using the unified resolver."""
    raw_task_id = str(getattr(trial, "task_id", "") or "").strip()
    metadata = getattr(trial, "metadata", {}) or {}
    harbor_task_id = metadata.get("harbor_task_id") if isinstance(metadata, dict) else None

    structured_path = ""
    structured_name = ""
    if isinstance(harbor_task_id, dict):
        structured_path = str(harbor_task_id.get("path") or "").strip()
        structured_name = str(harbor_task_id.get("name") or "").strip()
    elif isinstance(harbor_task_id, str) and harbor_task_id.strip():
        if "/" in harbor_task_id or "\\" in harbor_task_id:
            structured_path = harbor_task_id.strip()
        else:
            structured_name = harbor_task_id.strip()

    if structured_path:
        path_resolved = resolver.resolve(structured_path)
        if path_resolved.status != "resolved":
            return path_resolved
        if raw_task_id:
            raw_resolved = resolver.resolve(raw_task_id)
            if raw_resolved.status == "ambiguous":
                return raw_resolved
            if (
                raw_resolved.status == "resolved"
                and raw_resolved.canonical_id != path_resolved.canonical_id
            ):
                # Allow bare display name matching the structured path's tail only when
                # raw_task_id wasn't qualified to a different dataset/path.
                if "/" in _normalize_task_text(raw_task_id) or (
                    raw_resolved.bare_name != path_resolved.bare_name
                ):
                    return ResolvedTaskIdentity(
                        raw=raw_task_id,
                        canonical_id=None,
                        bare_name=path_resolved.bare_name,
                        dataset_name=path_resolved.dataset_name,
                        status="contradictory",
                        reason=(
                            f"trial task_id {raw_task_id!r} contradicts structured "
                            f"harbor_task_id path {structured_path!r}"
                        ),
                    )
        return path_resolved

    if structured_name and not raw_task_id:
        return resolver.resolve(structured_name)
    if structured_name and raw_task_id:
        raw_resolved = resolver.resolve(raw_task_id)
        name_resolved = resolver.resolve(structured_name)
        if raw_resolved.status != "resolved":
            return raw_resolved
        if name_resolved.status != "resolved":
            return name_resolved
        if raw_resolved.canonical_id != name_resolved.canonical_id:
            return ResolvedTaskIdentity(
                raw=raw_task_id,
                canonical_id=None,
                bare_name=raw_resolved.bare_name,
                dataset_name=raw_resolved.dataset_name,
                status="contradictory",
                reason=(
                    f"trial task_id {raw_task_id!r} contradicts structured "
                    f"harbor_task_id name {structured_name!r}"
                ),
            )
        return raw_resolved

    return resolver.resolve(raw_task_id)
