"""Tests for HarnessConfig versioning and serialization."""

import pytest
import yaml

from harness.config import HarnessConfig, ComponentRef


class TestHarnessConfig:
    def test_create_default(self):
        config = HarnessConfig.create_default()
        assert config.version == "0.1.0"
        assert config.model == "claude-sonnet-4-6"
        assert config.max_turns_audit == 0

    def test_legacy_max_turns_loads_as_audit_only(self, tmp_path):
        config_path = tmp_path / "harness.yaml"
        config_path.write_text('version: "0.1.0"\nmax_turns: 7\n')

        config = HarnessConfig.from_yaml(config_path)

        assert config.max_turns_audit == 7
        assert "max_turns_audit: 7" in config.to_yaml()
        assert "max_turns:" not in config.to_yaml()

    def test_get_all_components_empty(self):
        config = HarnessConfig.create_default()
        components = config.get_all_components()
        assert isinstance(components, dict)

    def test_bump_version(self):
        config = HarnessConfig.create_default()
        config.prompts["system"] = ComponentRef(
            name="system",
            path="harness/prompts/system.py",
            version="0.1.0",
            content_hash="abc123",
        )
        new_version = config.bump_version("system")
        assert new_version == "0.1.1"

    def test_bump_version_missing(self):
        config = HarnessConfig.create_default()
        with pytest.raises(KeyError):
            config.bump_version("nonexistent")

    def test_update_hash(self):
        config = HarnessConfig.create_default()
        config.prompts["system"] = ComponentRef(
            name="system",
            path="harness/prompts/system.py",
            version="0.1.0",
            content_hash="old",
        )
        new_hash = config.update_hash("system", "new content")
        assert new_hash != "old"
        assert len(new_hash) == 12

    def test_get_enabled_components(self):
        config = HarnessConfig.create_default()
        config.prompts["system"] = ComponentRef(
            name="system",
            path="x",
            version="0.1.0",
            content_hash="abc",
            enabled=True,
        )
        config.prompts["disabled"] = ComponentRef(
            name="disabled",
            path="x",
            version="0.1.0",
            content_hash="abc",
            enabled=False,
        )
        enabled = config.get_enabled_components()
        assert "prompts/system" in enabled
        assert "prompts/disabled" not in enabled

    def test_default_config_tracks_rust_worker_policy_as_runtime_prompt_surface(self):
        data = yaml.safe_load(open("config/default.yaml"))

        assert data["prompts"]["worker_policy"]["path"] == "crates/hl-worker-core/src/main.rs"
        assert data["entrypoint"]["bounded_scan"]["path"] == "crates/hl-worker-core/src/main.rs"
        assert data["verification"]["checks"]["path"] == "crates/hl-worker-core/src/main.rs"
        assert data["verification"]["self_test"]["path"] == "crates/hl-worker-core/src/main.rs"
        prompt_paths = {item["path"] for item in data["prompts"].values()}
        assert "harness/prompts/system.py" not in prompt_paths
        assert "harness/prompts/task.py" not in prompt_paths

    def test_checked_in_timeout_fields_are_not_loop_deadlines(self):
        default_text = open("config/default.yaml").read()
        models_text = open("config/models.yaml").read()
        benchmark_text = open("config/benchmark.yaml").read()

        assert "max_turns_audit" in default_text
        assert "max_turns:" not in default_text
        assert "not a" in default_text
        assert "master, sub-agent" in default_text
        assert "single model-provider" in models_text
        assert "request timeout" in models_text
        assert "not a loop stop condition" in models_text
        assert "must not become master, sub-agent" in benchmark_text


class TestComponentRef:
    def test_defaults(self):
        ref = ComponentRef(name="test", path="x", version="0.1.0", content_hash="abc")
        assert ref.enabled is True
        assert ref.dependencies == []
