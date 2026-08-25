"""Patch reviewer with complete and isolated Git delta rollback support.

The NUL-delimited status parser and deterministic policy gates are retained in
:mod:`meta._reviewer_issue15_base`. This facade makes staged changes, renames,
copies, and untracked binary files part of one reversible delta while preserving
unrelated index state.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

from meta import _reviewer_issue15_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


class PatchReviewer(_base.PatchReviewer):
    """Review and roll back the complete index+worktree delta."""

    def diff_text(self, paths: list[str] | None = None) -> str:
        if paths is not None and not paths:
            return ""
        requested_paths = list(paths) if paths is not None else None
        command = [
            "git",
            "diff",
            "--binary",
            "--full-index",
            "--find-renames",
            "HEAD",
            "--",
        ]
        if requested_paths is not None:
            command.extend(requested_paths)
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        diff = completed.stdout if completed.returncode == 0 else ""
        diff += "".join(self._untracked_file_diffs(requested_paths))
        return diff

    def _untracked_file_diffs(self, paths: list[str] | None) -> list[str]:
        candidates = self._untracked_files()
        if paths is not None:
            requested = set(paths)
            candidates = [path for path in candidates if path in requested]

        diffs: list[str] = []
        for path in candidates:
            full_path = self.repo_root / path
            if not full_path.is_file():
                continue
            completed = subprocess.run(
                [
                    "git",
                    "diff",
                    "--no-index",
                    "--binary",
                    "--full-index",
                    "--",
                    "/dev/null",
                    path,
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )
            # `git diff --no-index` returns 1 after successfully producing a diff.
            if completed.returncode in {0, 1} and completed.stdout:
                diffs.append(completed.stdout)
        return diffs

    def save_reverse_patch(self, path: str | Path) -> Path:
        """Persist the canonical forward delta used by :meth:`rollback`.

        The historical method name is retained. `rollback()` auto-detects old
        reverse patches as well as the current forward representation.
        """

        patch_path = Path(path)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(self.diff_text(self.changed_files()), encoding="utf-8")
        return patch_path

    def rollback(self, diff_path: str | Path) -> bool:
        patch_text = Path(diff_path).read_text(encoding="utf-8")
        if not patch_text.strip():
            return False

        # Current patches describe baseline -> dirty state, so rollback applies
        # them in reverse. Historical callers may supply an already-reversed
        # patch; detect that direction without mutating the repository.
        if self._patch_applies(patch_text, reverse=True):
            reverse_worktree = True
        elif self._patch_applies(patch_text, reverse=False):
            reverse_worktree = False
        else:
            return False

        # Prove whether the forward form is exactly HEAD-relative in a private
        # temporary index. If so, derive both rename endpoints with --no-renames
        # and later reset only those paths in the real index. A dirty-baseline
        # patch that is not HEAD-relative leaves the existing index untouched.
        head_paths = self._head_relative_patch_paths(
            patch_text,
            patch_is_forward=reverse_worktree,
        )

        command = ["git", "apply", "--binary", "--whitespace=nowarn"]
        if reverse_worktree:
            command.append("-R")
        applied = subprocess.run(
            command,
            input=patch_text,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        if applied.returncode != 0:
            return False

        if not head_paths:
            return True

        # Reset only patch-derived paths. Unrelated staged baseline work remains
        # byte-for-byte in the real index, unlike a repository-wide mixed reset.
        reset = subprocess.run(
            ["git", "reset", "--mixed", "HEAD", "--", *head_paths],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        return reset.returncode == 0

    def _patch_applies(self, patch_text: str, *, reverse: bool) -> bool:
        command = ["git", "apply", "--check", "--binary"]
        if reverse:
            command.append("-R")
        completed = subprocess.run(
            command,
            input=patch_text,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0

    def _head_relative_patch_paths(
        self,
        patch_text: str,
        *,
        patch_is_forward: bool,
    ) -> list[str]:
        """Return exact HEAD-relative endpoints without touching the real index."""

        with tempfile.TemporaryDirectory(prefix="hl-review-index-") as directory:
            index_path = str(Path(directory) / "index")
            env = {**os.environ, "GIT_INDEX_FILE": index_path}
            read_tree = subprocess.run(
                ["git", "read-tree", "HEAD"],
                cwd=self.repo_root,
                env=env,
                capture_output=True,
                text=True,
            )
            if read_tree.returncode != 0:
                return []

            command = ["git", "apply", "--cached", "--binary"]
            # If the supplied patch is historical reverse form, reverse it to
            # reconstruct the forward HEAD -> dirty state in the temp index.
            if not patch_is_forward:
                command.append("-R")
            applied = subprocess.run(
                command,
                input=patch_text,
                cwd=self.repo_root,
                env=env,
                capture_output=True,
                text=True,
            )
            if applied.returncode != 0:
                return []

            names = subprocess.run(
                [
                    "git",
                    "diff",
                    "--cached",
                    "--name-only",
                    "-z",
                    "--no-renames",
                    "HEAD",
                    "--",
                ],
                cwd=self.repo_root,
                env=env,
                capture_output=True,
                text=False,
            )
            if names.returncode != 0:
                return []
            return [
                os.fsdecode(record)
                for record in names.stdout.split(b"\0")
                if record
            ]
