"""Deterministic review gates for Codex-generated patches."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from meta import report_contract


@dataclass
class PatchReviewResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    violations: list[report_contract.ReportViolation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.violations:
            self.reasons = [finding.reason for finding in self.violations]
        elif self.reasons:
            self.violations = [
                report_contract.untyped_violation(reason) for reason in self.reasons
            ]
            self.reasons = [finding.reason for finding in self.violations]
        if any(report_contract.is_blocking_violation(item) for item in self.violations):
            self.accepted = False

    @property
    def reason_details(self) -> list[dict[str, str]]:
        return [finding.as_dict() for finding in self.violations]


class PatchReviewer:
    """Reject patches that violate the roadmap's update contract."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        allowed_roots: list[str] | None = None,
        forbidden_roots: list[str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.allowed_roots = allowed_roots or [
            "bench",
            "harness",
            "crates",
            "hl",
            "meta",
            "scripts",
            "config",
            "tests",
            "README.md",
            "docs",
        ]
        self.forbidden_roots = forbidden_roots or [
            "terminal-bench-tasks",
            "terminal-bench",
            "jobs",
            "trials/runs",
            "trials/regressions",
            "trials/submissions",
        ]
        self._known_task_ids: set[str] | None = None

    def review_worktree(self, *, ignore_files: list[str] | None = None) -> PatchReviewResult:
        ignored = set(ignore_files or [])
        changed_files = [path for path in self.changed_files() if path not in ignored]
        return self.review_delta(changed_files, self.diff_text(changed_files))

    def review_delta(
        self,
        changed_files: list[str],
        diff_text: str | None = None,
    ) -> PatchReviewResult:
        findings = report_contract.ViolationCollector()
        if not changed_files:
            findings.add("patch.no_files_changed", "no files changed")
        for path in changed_files:
            if self._is_forbidden(path):
                findings.add("patch.forbidden_path", f"forbidden path changed: {path}")
            if not self._is_allowed(path):
                findings.add(
                    "patch.outside_allowed_roots",
                    f"path is outside allowed edit roots: {path}",
                )
        if changed_files and all(path.startswith("trials/") for path in changed_files):
            findings.add("patch.memory_only", "patch only changes memory/log artifacts")
        if self._weakens_gates(changed_files, diff_text=diff_text):
            findings.add(
                "patch.gate_weakening",
                "patch appears to weaken verifier/regression/submit gates",
            )
        nested_agent_hits = self._nested_sub_agent_creation_hits(diff_text or "")
        if nested_agent_hits:
            findings.add(
                "patch.nested_agent",
                "production diff creates nested sub-agent or external agent launch path: "
                + "; ".join(nested_agent_hits[:8])
            )
        hardcoded_task_hits = self._hardcoded_task_id_hits(diff_text or "")
        if hardcoded_task_hits:
            findings.add(
                "patch.task_id_hardcoding",
                "production diff hardcodes TerminalBench task ids: "
                + "; ".join(hardcoded_task_hits[:8])
            )
        return PatchReviewResult(
            accepted=not findings.blocking,
            changed_files=changed_files,
            violations=findings.violations,
        )

    def changed_files(self) -> list[str]:
        """Return exact changed paths, expanding rename/copy endpoints.

        Human-readable ``git status --short`` is ambiguous for arrows, quoting,
        whitespace, and embedded newlines. Porcelain v1 with ``-z`` is stable and
        NUL-delimited; for rename/copy records Git emits the destination followed
        by a second NUL-delimited source path. Both endpoints are returned so
        isolation, root-policy, diff, validation, and rollback code sees the
        rename as a deletion plus an addition.
        """

        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=self.repo_root,
            capture_output=True,
            text=False,
        )
        if completed.returncode != 0:
            return []
        return self._parse_porcelain_v1_z(completed.stdout)

    def _parse_porcelain_v1_z(self, payload: bytes) -> list[str]:
        records = payload.split(b"\0")
        files: list[str] = []
        seen: set[str] = set()
        index = 0
        while index < len(records):
            record = records[index]
            index += 1
            if not record:
                continue
            if len(record) < 4 or record[2:3] != b" ":
                continue

            status = record[:2].decode("ascii", errors="replace")
            destination = os.fsdecode(record[3:])
            for path in [destination]:
                if path and path not in seen:
                    seen.add(path)
                    files.append(path)

            if "R" in status or "C" in status:
                if index >= len(records):
                    break
                source = os.fsdecode(records[index])
                index += 1
                if source and source not in seen:
                    seen.add(source)
                    files.append(source)
        return files

    def diff_text(self, paths: list[str] | None = None) -> str:
        if paths is not None and not paths:
            return ""
        requested_paths = list(paths) if paths is not None else None
        command = ["git", "diff", "--"]
        if requested_paths is not None:
            command.extend(requested_paths)
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        diff = completed.stdout if completed.returncode == 0 else ""
        untracked = self._untracked_text_file_diffs(requested_paths)
        if untracked:
            diff += "".join(untracked)
        return diff

    def _untracked_text_file_diffs(self, paths: list[str] | None) -> list[str]:
        candidates = self._untracked_files()
        if paths is not None:
            requested = set(paths)
            candidates = [path for path in candidates if path in requested]
        diffs: list[str] = []
        for path in candidates:
            full_path = self.repo_root / path
            if not full_path.is_file():
                continue
            try:
                content = full_path.read_bytes()
            except OSError:
                continue
            if self._looks_binary(content):
                continue
            text = content.decode("utf-8")
            diffs.append(self._new_file_diff(path, text))
        return diffs

    def _untracked_files(self) -> list[str]:
        completed = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=self.repo_root,
            capture_output=True,
            text=False,
        )
        if completed.returncode != 0:
            return []
        return [
            os.fsdecode(record)
            for record in completed.stdout.split(b"\0")
            if record
        ]

    def _new_file_diff(self, path: str, text: str) -> str:
        lines = [
            f"diff --git a/{path} b/{path}\n",
            "--- /dev/null\n",
            f"+++ b/{path}\n",
            "@@ -0,0 +1,{} @@\n".format(len(text.splitlines())),
        ]
        for line in text.splitlines(keepends=True):
            if line.endswith("\n"):
                lines.append("+" + line)
            else:
                lines.append("+" + line + "\n")
                lines.append("\\ No newline at end of file\n")
        return "".join(lines)

    def _looks_binary(self, content: bytes) -> bool:
        if b"\0" in content:
            return True
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            return True
        return False

    def save_reverse_patch(self, path: str | Path) -> Path:
        patch_path = Path(path)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["git", "diff", "-R", "--"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        patch_path.write_text(completed.stdout)
        return patch_path

    def rollback(self, diff_path: str | Path) -> bool:
        patch_text = Path(diff_path).read_text()
        if not patch_text.strip():
            return False
        completed = subprocess.run(
            ["git", "apply", "-R"],
            input=patch_text,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0

    def _is_forbidden(self, path: str) -> bool:
        return any(path == root or path.startswith(root + "/") for root in self.forbidden_roots)

    def _is_allowed(self, path: str) -> bool:
        return any(path == root or path.startswith(root + "/") for root in self.allowed_roots)

    def _hardcoded_task_id_hits(self, diff_text: str) -> list[str]:
        if not diff_text:
            return []
        known_task_ids = self._terminalbench_task_ids()
        if not known_task_ids:
            return []

        hits: list[str] = []
        current_path = ""
        for raw_line in diff_text.splitlines():
            if raw_line.startswith("+++ "):
                current_path = self._diff_path_from_to_file(raw_line[4:].strip())
                continue
            if not raw_line.startswith("+") or raw_line.startswith("+++"):
                continue
            if not current_path or self._allows_task_id_literals(current_path):
                continue
            added = raw_line[1:]
            found = sorted(
                task_id
                for task_id in known_task_ids
                if self._line_mentions_task_id(added, task_id)
            )
            for task_id in found:
                hits.append(f"{current_path}: {task_id}")
        return hits

    def _diff_path_from_to_file(self, to_file: str) -> str:
        if to_file == "/dev/null":
            return ""
        if to_file.startswith("b/"):
            return to_file[2:]
        return to_file

    def _allows_task_id_literals(self, path: str) -> bool:
        normalized = path.replace("\\", "/")
        if normalized.startswith(("tests/", "test/", "docs/")):
            return True
        if normalized.endswith((".md", ".rst")):
            return True
        return any(part in {"fixtures", "fixture"} for part in normalized.split("/"))

    def _line_mentions_task_id(self, line: str, task_id: str) -> bool:
        quoted = re.compile(rf"(['\"]){re.escape(task_id)}\1")
        return bool(quoted.search(line))

    def _nested_sub_agent_creation_hits(self, diff_text: str) -> list[str]:
        if not diff_text:
            return []

        hits: list[str] = []
        added_by_path: dict[str, list[str]] = {}
        current_path = ""
        for raw_line in diff_text.splitlines():
            if raw_line.startswith("+++ "):
                current_path = self._diff_path_from_to_file(raw_line[4:].strip())
                continue
            if not raw_line.startswith("+") or raw_line.startswith("+++"):
                continue
            if not current_path:
                continue
            added = raw_line[1:]
            added_by_path.setdefault(current_path, []).append(added)
            if self._allows_nested_agent_test_or_doc_literals(current_path):
                continue
            if self._line_launches_external_agent(added):
                hits.append(f"{current_path}: {added.strip()[:120]}")
                continue
            if self._allows_nested_agent_policy_literals(current_path):
                continue
            if self._line_starts_external_agent(added):
                hits.append(f"{current_path}: {added.strip()[:120]}")
        hit_paths = {hit.split(":", 1)[0] for hit in hits}
        for path, added_lines in added_by_path.items():
            if path in hit_paths:
                continue
            if self._allows_nested_agent_test_or_doc_literals(path):
                continue
            added_block = "\n".join(added_lines)
            if self._line_launches_external_agent(
                added_block
            ) or self._shell_fragment_launches_external_agent(added_block):
                hits.append(f"{path}: multi-line external agent launch")
        return hits

    def _allows_nested_agent_test_or_doc_literals(self, path: str) -> bool:
        normalized = path.replace("\\", "/")
        if normalized.startswith(("tests/", "test/", "docs/")):
            return True
        if normalized.endswith((".md", ".rst")):
            return True
        return any(part in {"fixtures", "fixture"} for part in normalized.split("/"))

    def _allows_nested_agent_policy_literals(self, path: str) -> bool:
        normalized = path.replace("\\", "/")
        if normalized in {
            "meta/packager.py",
            "meta/codex_update.py",
            "meta/reviewer.py",
            "hl/loop_limits.py",
        }:
            return True
        return False

    def _line_starts_external_agent(self, line: str) -> bool:
        stripped = line.strip().lstrip("'\"")
        if self._shell_fragment_launches_external_agent(stripped):
            return True
        for literal in re.finditer(r"(['\"])(?P<content>.*?)(?<!\\)\1", line):
            content = literal.group("content")
            if self._quoted_literal_launches_external_agent(content):
                return True
        return False

    def _quoted_literal_launches_external_agent(self, content: str) -> bool:
        stripped = content.strip()
        if re.fullmatch(
            r"(?:openai-codex|codex|claude|claude-code|forgecode|factory-droid|factory|droid|gemini|gemini-cli|opencode|aider|amp|cursor-agent)",
            stripped,
            re.IGNORECASE,
        ):
            return False
        return self._shell_fragment_launches_external_agent(stripped)

    def _shell_fragment_launches_external_agent(self, fragment: str) -> bool:
        try:
            from harness.tools.shell import external_agent_command_reason
        except Exception:
            return bool(
                re.search(
                    r"\b(?:openai-codex\s+(?:exec|run)|codex\s+(?:exec|run)|claude(?:-code)?\b|forgecode\b|factory-droid\b|factory\s+(?:droid|mission|missions)|droid\s+(?:mission|missions|run)|gemini(?:-cli)?\b|opencode\b|aider\b|amp\b|cursor-agent\b)",
                    fragment,
                    re.IGNORECASE,
                )
            )
        return external_agent_command_reason(fragment) is not None

    def _line_launches_external_agent(self, line: str) -> bool:
        launch_pattern = re.compile(
            r"\b(?:subprocess\.(?:run|popen|call|check_call|check_output)"
            r"|asyncio\.create_subprocess_(?:exec|shell)"
            r"|os\.(?:system|popen|exec(?:l|le|lp|lpe|v|ve|vp|vpe)|spawn(?:l|le|lp|lpe|v|ve|vp|vpe))"
            r"|pexpect\.spawn|pty\.spawn"
            r"|child_process\.(?:exec|execFile|spawn|fork|execSync|execFileSync|spawnSync)"
            r"|require\s*\(\s*['\"](?:node:)?child_process['\"]\s*\)\s*\.\s*(?:exec|execFile|spawn|fork|execSync|execFileSync|spawnSync)"
            r"|Bun\.spawn|Deno\.Command)\s*\(",
            re.IGNORECASE,
        )
        if not launch_pattern.search(line):
            return False
        return bool(
            re.search(
                r"\b(?:openai-codex|codex|claude|claude-code|forgecode|factory-droid|factory|droid|gemini|gemini-cli|opencode|aider|amp|cursor-agent)\b",
                line,
                re.IGNORECASE,
            )
        )

    def _terminalbench_task_ids(self) -> set[str]:
        if self._known_task_ids is not None:
            return self._known_task_ids
        task_root = self.repo_root / "terminal-bench-tasks" / "terminal-bench"
        task_ids: set[str] = set()
        if task_root.exists():
            for path in task_root.iterdir():
                if path.is_dir() and (path / "task.toml").exists():
                    task_ids.add(path.name)
        self._known_task_ids = task_ids
        return task_ids

    def _weakens_gates(
        self,
        paths: list[str] | None = None,
        *,
        diff_text: str | None = None,
    ) -> bool:
        diff = (diff_text if diff_text is not None else self.diff_text(paths)).lower()
        risky_pairs = [
            ("disable_verification", "true"),
            ("require_clean_git", "false"),
            ("require_full_regression", "false"),
            ("submit:", "enabled: true"),
        ]
        return any(left in diff and right in diff for left, right in risky_pairs)
