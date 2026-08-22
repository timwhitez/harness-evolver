"""Tests for prompt template rendering and validation."""

from harness.prompts.system import SystemPrompt, ThreeTierPromptSystem
from harness.prompts.task import TaskPrompt
from harness.prompts.recovery import RecoveryPrompt
from harness.recovery.patterns import ErrorPatterns


class TestSystemPrompt:
    def test_render(self):
        sp = SystemPrompt()
        result = sp.render({})
        assert "autonomous coding agent" in result
        assert "Docker container" in result
        assert "Create and check artifacts early" in result
        assert "before long builds, installs, or final claims" in result
        assert "Keep operations bounded, keep solving" in result
        assert "continue the Worker loop" in result
        assert "Only the master HL orchestrator may create sub-agents" in result
        assert "must not start Codex" in result
        assert "create nested sub-agents" in result
        assert "task time budget" not in result
        assert "agent timeout" not in result

    def test_validate(self):
        sp = SystemPrompt()
        errors = sp.validate()
        assert errors == []  # Default template is valid


class TestTaskPrompt:
    def test_render(self):
        tp = TaskPrompt()
        result = tp.render({
            "instruction": "Write a hello world program",
            "task_id": "test::hello::1.0",
            "domain": "software_engineering",
            "difficulty": "easy",
            "environment_context": "",
            "verification_guidance": "",
            "previous_errors": [],
        })
        assert "Write a hello world program" in result
        assert "software_engineering" in result
        assert "create the expected artifact shape early" in result
        assert "by mid-run and again near completion" in result
        assert "Keep each operation bounded while continuing the task" in result
        assert "Only the master HL orchestrator may create sub-agents" in result
        assert "another external coding-agent CLI" in result
        assert "create nested sub-agents" in result
        assert "agent timeout" not in result

    def test_default_worker_policy_does_not_describe_time_or_round_limits(self):
        system = SystemPrompt().render({})
        task = TaskPrompt().render({
            "instruction": "Test",
            "task_id": "test",
            "domain": "test",
            "difficulty": "easy",
            "environment_context": "",
            "verification_guidance": "",
            "previous_errors": [],
        })
        combined = f"{system}\n{task}"

        forbidden = [
            "task time budget",
            "agent timeout",
            "time-based task stop condition",
            "turn-count stop condition",
            "round limit",
        ]
        for phrase in forbidden:
            assert phrase not in combined
        assert "If a single command or check times out" in combined
        assert "switch to a smaller or faster strategy" in combined

    def test_previous_errors_injected(self):
        tp = TaskPrompt()
        result = tp.render({
            "instruction": "Test",
            "task_id": "test",
            "domain": "test",
            "difficulty": "easy",
            "environment_context": "",
            "verification_guidance": "",
            "previous_errors": ["Error: command not found", "Error: timeout"],
        })
        assert "command not found" in result
        assert "timeout" in result

    def test_verification_guidance_injected(self):
        tp = TaskPrompt()
        result = tp.render({
            "instruction": "Test",
            "task_id": "test",
            "domain": "software_engineering",
            "difficulty": "hard",
            "environment_context": "",
            "verification_guidance": "## Verification Policy\nUse subprocess checks.",
            "previous_errors": [],
        })

        assert "## Verification Policy" in result
        assert "Use subprocess checks." in result

    def test_validate_missing_placeholder(self):
        tp = TaskPrompt(template="No placeholder here")
        errors = tp.validate()
        assert len(errors) == 1
        assert "placeholder" in errors[0]


class TestRecoveryPrompt:
    def test_render(self):
        rp = RecoveryPrompt()
        result = rp.render({
            "error_type": "command_not_found",
            "known_pattern": "missing_dependency",
            "recovery_strategy": "Install the package via apt-get",
        })
        assert "command_not_found" in result
        assert "missing_dependency" in result
        assert "apt-get" in result

    def test_timeout_pattern_is_not_any_loop_stop(self):
        guidance = ErrorPatterns().recovery_for("command timed out")

        assert guidance is not None
        assert "master, sub-agent, or Worker loop stop condition" in guidance
        assert "Worker-loop stop condition" not in guidance


class TestThreeTierPromptSystem:
    def test_separate_tiers(self):
        system = ThreeTierPromptSystem()
        system.set_tool_descriptions("- bash: execute commands")
        system.set_notifications("Warning: low disk space")

        full = system.render_full()
        assert "autonomous coding agent" in full
        assert "- bash" in full
        assert "low disk space" in full

    def test_empty_tiers(self):
        system = ThreeTierPromptSystem()
        full = system.render_full()
        assert "autonomous coding agent" in full
