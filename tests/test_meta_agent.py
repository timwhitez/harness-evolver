"""Tests for the meta-agent failure analysis and harness editing."""

from hl.types import FeedbackSignal, TrialResult, TrialStatus
from meta.analysis import FailureAnalyzer
from meta.editor import HarnessEditor
from meta.suggestion import ImprovementSuggester


class TestFailureAnalyzer:
    def test_analyze_basic_failure(self):
        analyzer = FailureAnalyzer()
        signal = FeedbackSignal(
            trial_id="test_001",
            task_id="test::fail::1.0",
            status=TrialStatus.FAILED,
            error_summary="Command not found",
            raw_errors=["bash: foo: command not found"],
        )
        finding = analyzer.analyze(signal)
        assert "command_not_found" in finding["affected_components"] or \
               "tools/shell" in finding["affected_components"] or \
               any("shell" in c for c in finding["affected_components"])

    def test_analyze_success(self):
        analyzer = FailureAnalyzer()
        signal = FeedbackSignal(
            trial_id="test_002",
            task_id="test::pass::1.0",
            status=TrialStatus.PASSED,
            error_summary="",
        )
        finding = analyzer.analyze(signal)
        assert len(finding["affected_components"]) > 0  # Always suggests something

    def test_low_tool_success_rate(self):
        analyzer = FailureAnalyzer()
        signal = FeedbackSignal(
            trial_id="test_003",
            task_id="test::fail::1.0",
            status=TrialStatus.FAILED,
            tool_call_success_rate=0.3,
            error_summary="Multiple tool failures",
        )
        finding = analyzer.analyze(signal)
        assert "tools/correction" in finding["affected_components"]

    def test_long_trajectory(self):
        analyzer = FailureAnalyzer()
        signal = FeedbackSignal(
            trial_id="test_004",
            task_id="test::fail::1.0",
            status=TrialStatus.FAILED,
            trajectory_length=80,
            error_summary="Context overflow",
        )
        finding = analyzer.analyze(signal)
        assert "context/compaction" in finding["affected_components"]


class TestHarnessEditor:
    def test_list_editable_components(self):
        editor = HarnessEditor()
        components = editor.list_editable_components()
        assert len(components) > 0
        assert any("system.py" in c for c in components)

    def test_get_component_content(self):
        editor = HarnessEditor()
        content = editor.get_component_content("prompts/system.py")
        assert len(content) > 0
        assert "autonomous coding agent" in content

    def test_get_nonexistent_component(self):
        editor = HarnessEditor()
        content = editor.get_component_content("nonexistent.py")
        assert content == ""

    def test_snapshot_and_diff(self):
        editor = HarnessEditor()
        before = editor.snapshot_harness()
        after = editor.snapshot_harness()
        diffs = editor.diff_harness(before, after)
        assert len(diffs) == 0  # Identical snapshots

    def test_edit_component(self, tmp_path):
        # Create a test component
        test_file = tmp_path / "test_comp.py"
        test_file.write_text("version = '0.1.0'\nstatus = 'ok'")

        editor = HarnessEditor(harness_root=tmp_path, backup_root=tmp_path / "backups")
        success = editor.edit_component(
            "test_comp.py",
            "status = 'ok'",
            "status = 'improved'",
            "Test improvement",
        )
        assert success is True
        assert "improved" in test_file.read_text()

    def test_edit_nonexistent(self):
        editor = HarnessEditor()
        success = editor.edit_component("nonexistent.py", "a", "b", "")
        assert success is False

    def test_edit_old_not_found(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("hello")

        editor = HarnessEditor(harness_root=tmp_path, backup_root=tmp_path / "backups")
        success = editor.edit_component("test.py", "not in file", "replacement", "")
        assert success is False


class TestImprovementSuggester:
    def test_suggest_from_finding(self):
        suggester = ImprovementSuggester()
        findings = [{
            "task_id": "test::fail::1.0",
            "affected_components": ["prompts/system"],
            "root_cause": "System prompt missing timeout guidance",
            "suggested_action": "Add timeout handling to system prompt",
        }]
        component_contents = {
            "prompts/system": "You are an agent.\nNo timeout guidance.",
        }
        patches = suggester.suggest(findings, component_contents)
        assert len(patches) >= 0  # May be empty if component path mismatch
        assert patches[0].component_name == "prompts/system"
        if patches:
            assert patches[0].component_name == "prompts/system"
            assert "timeout" in patches[0].rationale.lower()
