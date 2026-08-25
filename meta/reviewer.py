"""Patch reviewer with complete HEAD-to-worktree delta and rollback support.

The NUL-delimited status parser and deterministic policy gates are retained in
:mod:`meta._reviewer_issue15_base`. This facade makes staged changes, renames,
copies, and untracked binary files part of the same reversible delta.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

from meta import _reviewer_issue15_base as _base

for _name, _value in vars(_base).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value


class PatchReviewer(_base.PatchReviewer):
    """Review and roll back the complete index+worktree delta from HEAD."""

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
            # `git diff --no-index` returns 1 when it successfully found a diff.
            if completed.returncode in {0, 1} and completed.stdout:
                diffs.append(completed.stdout)
        return diffs

    def save_reverse_patch(self, path: str | Path) -> Path:
        """Persist the forward HEAD delta consumed by :meth:`rollback`.

        The historical method name is retained for API compatibility. Keeping
        the canonical forward patch lets `git apply -R` restore both staged and
        unstaged rename endpoints as well as untracked binary files.
        """

        patch_path = Path(path)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(self.diff_text(self.changed_files()), encoding="utf-8")
        return patch_path

    def rollback(self, diff_path: str | Path) -> bool:
        patch_text = Path(diff_path).read_text(encoding="utf-8")
        if not patch_text.strip():
            return False

        checked = subprocess.run(
            ["git", "apply", "--check", "--binary", "-R"],
            input=patch_text,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        if checked.returncode != 0:
            return False
        applied = subprocess.run(
            ["git", "apply", "--binary", "--whitespace=nowarn", "-R"],
            input=patch_text,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        if applied.returncode != 0:
            return False

        # `git apply` restores bytes in the worktree. Reset the index separately
        # so a staged `git mv` or copy cannot remain hidden after rollback.
        reset = subprocess.run(
            ["git", "reset", "--mixed", "HEAD"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        return reset.returncode == 0
