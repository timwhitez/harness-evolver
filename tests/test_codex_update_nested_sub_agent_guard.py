import json
import subprocess

from meta.codex_update import CodexUpdateEngine
from meta.reviewer import PatchReviewer, PatchReviewResult


def _minimal_noop_report() -> dict[str, object]:
    return {
        "status": "noop",
        "summary": "no repository changes",
        "changed_files": [],
        "validation_commands": [],
        "skipped_validation_reason": "no changes",
    }


def test_codex_update_rejects_nested_agent_command_in_events(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    events_text = json.dumps(
        {
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "command": "/bin/bash -lc 'codex exec fix the harness'",
            },
        }
    )

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=[]),
        exit_code=0,
        final_report=_minimal_noop_report(),
        required_validation_commands=[],
        exec_output_text=events_text,
    )

    assert review.accepted is False
    assert any(
        "Codex update sub-agent attempted to create a nested sub-agent" in reason
        for reason in review.reasons
    )


def test_codex_update_rejects_bare_codex_cli_in_events(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    events_text = json.dumps(
        {
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "command": "command codex --help",
            },
        }
    )

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=[]),
        exit_code=0,
        final_report=_minimal_noop_report(),
        required_validation_commands=[],
        exec_output_text=events_text,
    )

    assert review.accepted is False
    assert any("command codex --help" in reason for reason in review.reasons)
    assert any(
        "Codex update sub-agent attempted to create a nested sub-agent" in reason
        for reason in review.reasons
    )


def test_codex_update_rejects_unquoted_shell_c_agent_command_in_events(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    events_text = json.dumps(
        {
            "type": "item.started",
            "item": {
                "type": "command_execution",
                "command": "bash -c codex --help",
            },
        }
    )

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=[]),
        exit_code=0,
        final_report=_minimal_noop_report(),
        required_validation_commands=[],
        exec_output_text=events_text,
    )

    assert review.accepted is False
    assert any("bash -c codex --help" in reason for reason in review.reasons)
    assert any(
        "Codex update sub-agent attempted to create a nested sub-agent" in reason
        for reason in review.reasons
    )


def test_codex_update_allows_non_agent_codex_mentions_in_events(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    events_text = json.dumps(
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "printf 'codex exec is blocked by policy'",
            },
        }
    )

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=[]),
        exit_code=0,
        final_report=_minimal_noop_report(),
        required_validation_commands=[],
        exec_output_text=events_text,
    )

    assert review.accepted is True
    assert not any("nested sub-agent" in reason for reason in review.reasons)


def test_codex_update_rejects_nested_agent_command_in_args_payload(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    events_text = json.dumps(
        {
            "type": "tool_call",
            "tool": "bash",
            "args": {"cmd": "python -c \"import subprocess; subprocess.run(['opencode', 'run', 'fix'])\""},
        }
    )

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=[]),
        exit_code=0,
        final_report=_minimal_noop_report(),
        required_validation_commands=[],
        exec_output_text=events_text,
    )

    assert review.accepted is False
    assert any("opencode" in reason.lower() for reason in review.reasons)


def test_codex_update_rejects_nested_agent_argv_payloads(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    events_text = "\n".join(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "bash",
                    "args": {"cmd": ["codex", "exec", "fix"]},
                }
            ),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "type": "command_execution",
                        "command": ["bash", "-lc", "opencode run fix"],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "exec",
                    "args": {"argv": ["uvx", "openai-codex", "exec", "fix"]},
                }
            ),
        ]
    )

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=[]),
        exit_code=0,
        final_report=_minimal_noop_report(),
        required_validation_commands=[],
        exec_output_text=events_text,
    )

    assert review.accepted is False
    assert any("codex exec fix" in reason for reason in review.reasons)
    assert any("opencode run fix" in reason for reason in review.reasons)
    assert any("openai-codex exec fix" in reason for reason in review.reasons)


def test_codex_update_rejects_indirect_nested_agent_commands_in_events(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    events_text = "\n".join(
        [
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "bash",
                    "args": {"cmd": "env c=codex bash -lc '$c exec fix'"},
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "bash",
                    "args": {
                        "cmd": "python -c \"import subprocess; cmd = 'cod' + 'ex'; subprocess.run([cmd, 'exec', 'fix'])\""
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "bash",
                    "args": {
                        "cmd": "python -c \"import subprocess; cmd = ''.join(['co','dex']); subprocess.run([cmd, 'exec', 'fix'])\""
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "bash",
                    "args": {"cmd": "ruby -e \"c=%q{codex}; spawn c, 'exec', 'fix'\""},
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "bash",
                    "args": {
                        "cmd": "python -c \"import subprocess; cmd = chr(99)+chr(111)+chr(100)+chr(101)+chr(120); subprocess.run([cmd, 'exec', 'fix'])\""
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "bash",
                    "args": {
                        "cmd": "node -e \"const c = ['co','dex'].join(''); require('child_process').spawn(c, ['exec','fix'])\""
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "bash",
                    "args": {"cmd": "lua -e \"os.execute('codex exec fix')\""},
                }
            ),
            json.dumps(
                {
                    "type": "tool_call",
                    "tool": "bash",
                    "args": {"cmd": "php -r \"exec('codex exec fix');\""},
                }
            ),
        ]
    )

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=[]),
        exit_code=0,
        final_report=_minimal_noop_report(),
        required_validation_commands=[],
        exec_output_text=events_text,
    )

    assert review.accepted is False
    assert any("env c=codex" in reason for reason in review.reasons)
    assert any("'cod' + 'ex'" in reason for reason in review.reasons)
    assert any("''.join" in reason for reason in review.reasons)
    assert any("%q{codex}" in reason for reason in review.reasons)
    assert any("chr(99)" in reason for reason in review.reasons)
    assert any("['co','dex'].join" in reason for reason in review.reasons)
    assert any("os.execute" in reason for reason in review.reasons)
    assert any("php -r" in reason for reason in review.reasons)


def test_codex_update_rejects_staged_nested_agent_script_write_event(tmp_path):
    engine = CodexUpdateEngine(repo_root=tmp_path, events_dir=tmp_path / "diffs")
    events_text = json.dumps(
        {
            "type": "tool_call",
            "tool": "bash",
            "args": {
                "cmd": "python -c \"from pathlib import Path; Path('/tmp/run_agent.sh').write_text('codex exec fix')\""
            },
        }
    )

    review = engine._apply_report_gates(
        PatchReviewResult(accepted=True, changed_files=[]),
        exit_code=0,
        final_report=_minimal_noop_report(),
        required_validation_commands=[],
        exec_output_text=events_text,
    )

    assert review.accepted is False
    assert any("write_text" in reason for reason in review.reasons)


def test_reviewer_rejects_untracked_nested_agent_script(tmp_path):
    _init_repo(tmp_path)
    delegate = tmp_path / "harness" / "tools" / "delegate.py"
    delegate.parent.mkdir(parents=True, exist_ok=True)
    delegate.write_text(
        "import subprocess\n"
        "def run_delegate():\n"
        "    return subprocess.run(['codex', 'exec', 'fix'], check=False)\n"
    )

    reviewer = PatchReviewer(repo_root=tmp_path)
    review = reviewer.review_worktree()
    diff = reviewer.diff_text(review.changed_files)

    assert review.accepted is False
    assert review.changed_files == ["harness/tools/delegate.py"]
    assert "subprocess.run(['codex', 'exec', 'fix']" in diff
    assert any("creates nested sub-agent" in reason for reason in review.reasons)


def _init_repo(path):
    (path / "bench").mkdir()
    (path / "bench" / "agent.py").write_text("original\n")
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=path,
        check=True,
        capture_output=True,
    )
