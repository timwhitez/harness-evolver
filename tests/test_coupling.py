"""Tests for coupling complexity tracker."""

from datetime import datetime

from hl.coupling import CouplingTracker
from hl.types import HarnessPatch


class TestCouplingTracker:
    def test_initial_state(self):
        ct = CouplingTracker()
        assert ct.component_count == 0
        assert ct.average_coupling == 0.0
        assert ct.compression_count == 0

    def test_register_component(self):
        ct = CouplingTracker()
        ct.register_component("prompts/system", [])
        assert ct.component_count == 1

    def test_touch_count_direct(self):
        ct = CouplingTracker()
        ct.register_component("prompts/system", [])
        ct.register_component("planning/todo", [])

        patch = HarnessPatch(
            component_name="prompts/system",
            before_version="0.1.0",
            after_version="0.1.1",
            file_path="x",
            diff="",
            rationale="test",
        )
        score = ct.touch_count(patch)
        assert score == 1  # Only touches prompts/system directly

    def test_touch_count_with_dependents(self):
        ct = CouplingTracker()
        ct.register_component("prompts/system", [])
        ct.register_component("planning/todo", ["prompts/system"])  # depends on system

        patch = HarnessPatch(
            component_name="prompts/system",
            before_version="0.1.0",
            after_version="0.1.1",
            file_path="x",
            diff="",
            rationale="test",
        )
        score = ct.touch_count(patch)
        assert score == 2  # Touches system + todo (depends on system)

    def test_record_patch(self):
        ct = CouplingTracker()
        ct.register_component("prompts/system", [])

        patch = HarnessPatch(
            component_name="prompts/system",
            before_version="0.1.0",
            after_version="0.1.1",
            file_path="x",
            diff="test",
            rationale="test",
        )
        score = ct.record_patch(patch)
        assert score == 1
        assert ct.average_coupling == 1.0

    def test_needs_compression_empty(self):
        ct = CouplingTracker()
        assert ct.needs_compression() is False

    def test_needs_compression_by_coupling(self):
        ct = CouplingTracker(max_coupling_per_patch=0)
        ct.register_component("a", ["b", "c"])
        ct.register_component("b", [])
        ct.register_component("c", [])

        patch = HarnessPatch(
            component_name="a",
            before_version="0.1.0",
            after_version="0.1.1",
            file_path="x",
            diff="",
            rationale="test",
        )
        ct.record_patch(patch)
        # patch touches a, b, c = score 3 > threshold 0
        assert ct.needs_compression() is True

    def test_needs_compression_by_ratio(self):
        ct = CouplingTracker(patch_to_compression_ratio=0.5)
        ct.register_component("only_component", [])

        for i in range(5):
            patch = HarnessPatch(
                component_name="only_component",
                before_version="0.1.0",
                after_version=f"0.1.{i+1}",
                file_path="x",
                diff="",
                rationale="test",
            )
            ct.record_patch(patch)

        # 5 patches / 1 component = 5 > 0.5 threshold
        assert ct.needs_compression() is True

    def test_record_compression(self):
        ct = CouplingTracker()
        ct.register_component("old_a", [])
        ct.register_component("old_b", [])

        ct.record_compression(["old_a", "old_b"], "merged")

        assert ct.component_count == 1  # two merged into one
        assert ct.compression_count == 1
        assert "merged" in ct.summary()["per_component_deps"]

    def test_remove_component(self):
        ct = CouplingTracker()
        ct.register_component("a", ["b"])
        ct.register_component("b", [])

        ct.remove_component("a")
        assert ct.component_count == 1
        assert "a" not in ct.summary()["per_component_deps"]
